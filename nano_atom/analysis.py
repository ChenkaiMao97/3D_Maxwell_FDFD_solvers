from __future__ import annotations
import math
from typing import Dict, Optional
import torch
from src.utils.physics import r2c, E_to_H_batch
from src.solvers.NN_solver import NN_solve
import yaml
from pathlib import Path
general_config_path = Path(__file__).parent / "configs" / "general.yaml"

C0 = 299_792_458.0
ETA0 = 376.730313668  # Free-space impedance
_FLOAT_EPS = 1e-30
P_DENSITY = 4.3067553e-08
# P_DENSITY = 5.4123211e-15

def _field_to_complex(field):
    """
    Convert field to complex (..., 3) tensor, accepting real (..., 6) or complex (..., 3).

    Args:
        field: Field tensor, shape (sx, sy, sz, 6|3) or (bs, sx, sy, sz, 6|3)
    
    Returns:
        Complex field tensor of shape (bs, sx, sy, sz, 3)
    """
    if field.ndim == 4:
        field = field.unsqueeze(0)
    if field.ndim != 5 or field.shape[-1] not in (3, 6):
        raise ValueError("Field must have shape (sx, sy, sz, 6|3) or (bs, sx, sy, sz, 6|3).")

    if field.shape[-1] == 6:
        e_complex = r2c(field)
    else:
        if not torch.is_complex(field):
            raise ValueError("Field with last dim 3 must be complex dtype (complex64/128).")
        e_complex = field

    return e_complex

def _polarization_vector(pol, amp=1.0, norm=True):
    """
    Return the complex polarization vector (Ex, Ey, Ez) for a given polarization.

    Args:
        pol: Polarization string: 'x', 'y', 'z', 'LCP', 'RCP'
        amp: Amplitude (default 1.0)
        norm: If True, normalize LCP/RCP to sqrt(2)

    Returns:
        Complex tensor of shape (3,), [Ex, Ey, Ez]
    """
    pol = pol.upper()
    if pol == "X":
        return torch.tensor([amp, 0.0, 0.0], dtype=torch.complex64)
    elif pol == "Y":
        return torch.tensor([0.0, amp, 0.0], dtype=torch.complex64)
    elif pol == "Z":
        return torch.tensor([0.0, 0.0, amp], dtype=torch.complex64)
    elif pol == "LCP":
        scale = amp / math.sqrt(2) if norm else amp
        return torch.tensor([scale, 1j * scale, 0.0], dtype=torch.complex64)
    elif pol == "RCP":
        scale = amp / math.sqrt(2) if norm else amp
        return torch.tensor([scale, -1j * scale, 0.0], dtype=torch.complex64)
    else:
        raise ValueError("Unknown polarization; use 'x', 'y', 'z', 'LCP', or 'RCP'.")

def _proj(F, p):
    """
    Project a 3-vector field F(...,3) onto polarization vector p(3).
    
    Args:
        F: 3-vector field tensor with shape (..., 3)
        p: Polarization vector tensor with shape (3,)
    
    Returns:
        Projected field tensor with same shape as F
    """
    p_xy = p[:2]
    denom = (p_xy.conj() * p_xy).sum().real
    a = (F[..., :2] * p_xy.conj()).sum(dim=-1) / denom
    E_proj_xy = a.unsqueeze(-1) * p_xy
    E_proj = torch.zeros_like(F)
    E_proj[..., :2] = E_proj_xy
    return E_proj


def e2h(E, dL, wvln, n):
    """
    Convert E field to H field.

    Args:
        E: (bs, sx, sy, sz, 6|3) tensor of E field
        dL: float, grid spacing in length units (nm)
        wvln: float, wavelength in length units (nm)
        n: float, refractive index
    Returns:
        (bs, sx, sy, sz, 3) tensor of H field
    """
    E = _field_to_complex(E)
    device = E.device
    Ex, Ey, Ez = E[..., 0], E[..., 1], E[..., 2]
    omega = 2 * math.pi * C0 / (wvln * 1e-9)
    sx, sy, sz = E.shape[1], E.shape[2], E.shape[3]
    dx = torch.full((sx,), dL * 1e-9, dtype=torch.float32, device=device)
    dy = torch.full((sy,), dL * 1e-9, dtype=torch.float32, device=device)
    dz = torch.full((sz,), dL * 1e-9, dtype=torch.float32, device=device)
    dxes = (
        [dx, dy, dz], 
        [dx, dy, dz], 
    )
    Hx, Hy, Hz = E_to_H_batch(Ex, Ey, Ez, dxes, omega, bloch_vector=None)
    return torch.stack((Hx, Hy, Hz), dim=-1)


def poynting_flux(E, dL, wvln, n, z_index):
    '''
    Calculate Poynting flux for a specific z-plane.

    Args:
        E: (bs, sx, sy, sz, 6|3) tensor of E field
        dL: float, grid spacing in length units (nm)
        wvln: float, wavelength in length units (nm)
        n: float, refractive index
        z_index: int, index of the z-plane
    Returns:
        tensor of shape (bs,), Poynting flux in W/m^2
    '''
    E = _field_to_complex(E)
    H = e2h(E, dL, wvln, n)
    E_plane = E[..., z_index, :]
    H_plane = H[..., z_index, :]

    # Calculate area of the plane
    dx, dy = dL * 1e-9, dL * 1e-9
    area = dx * dy  # Units m^2

    # Calculate Poynting flux
    Ex, Ey = E_plane[..., 0], E_plane[..., 1]
    Hx, Hy = H_plane[..., 0], H_plane[..., 1]
    Sz = 0.5 * (Ex * Hy.conj() - Ey * Hx.conj()).real
    Sz = Sz.sum(dim=(-1, -2)) * area
    
    return Sz


def _measure_incident_flux(src, dL, wvln, n_source, config):
    """
    Measure incident flux by solving a uniform medium with n_source.
    
    Args:
        src: Source tensor (bs, sx, sy, sz, 6) - used to extract sim_shape
        dL: Grid spacing in nm
        wvln: Wavelength in nm
        n_source: Refractive index of source medium
        config: Configuration dict for NN_solve (must contain model_path, pmls, max_iter, tol, verbose, restart)
    
    Returns:
        float: Incident flux in W
    """
    # Extract sim_shape from src tensor
    if src.ndim != 5 or src.shape[-1] != 6:
        raise ValueError("src must have shape (bs, sx, sy, sz, 6)")
    
    bs, sx, sy, sz = src.shape[:4]
    sim_shape = (sx, sy, sz)
    
    pmls = config.get("pmls", [0, 0, 0, 0, 10, 10])
    top_pml_z = pmls[5]  # z_end (top PML)
    
    # Create uniform medium with n_source^2 permittivity
    eps_source = n_source ** 2
    eps = torch.full(sim_shape, eps_source, dtype=torch.float32)
        
    solve_config = {
        "sim_shape": list(sim_shape),
        "wavelength": wvln,
        "dL": dL,
        "pmls": pmls,
        "model_path": config["model_path"],
        "max_iter": config.get("max_iter", 1000),
        "tol": config.get("tol", 1e-5),
        "verbose": config.get("verbose", False),
        "restart": config.get("restart", 0),
    }
    
    # Solve for E field in uniform medium using the same source
    solution, residual_history, final_residual = NN_solve(
        solve_config,
        eps[None].clone(),
        src.clone()
    )
    
    # Calculate flux at monitor plane (4 pixels below top PML)
    monitor_z = sz - top_pml_z - 4
    if monitor_z < 0:
        monitor_z = sz - 1  # Fallback to last pixel if calculation goes negative
    inc_flux = poynting_flux(
        solution,
        dL,
        wvln,
        n_source,
        monitor_z
    ).squeeze().item()
    
    return float(inc_flux)


def compute_transmittance(
    solution_field,
    monitor_z_idx,
    n_monitor,
    n_source,
    wvln=None,
    dL=None,
    inc_flux=None,
    src=None,
    config=None,
):
    """
    Compute transmittance from solution field.
    
    Args:
        solution_field: Solution field tensor (bs, sx, sy, sz, 6|3)
        monitor_z_idx: Z-index of monitor plane
        n_monitor: Refractive index at monitor plane
        n_source: Refractive index of source medium
        wvln: Wavelength in nm (optional)
        dL: Grid spacing in nm (optional)
        inc_flux: Incident flux in W (optional, will be computed if None)
        src: Source tensor (bs, sx, sy, sz, 6) (required if inc_flux is None)
        config: Configuration dict for NN_solve (optional)
    
    Returns:
        Dictionary with keys: transmittance, transmitted_flux, incident_flux
    """
    E = _field_to_complex(solution_field)
    trans_flux = poynting_flux(E, dL, wvln, n_monitor, monitor_z_idx).squeeze(0)

    if inc_flux is None:
        if src is None:
            raise ValueError("src is required when inc_flux is None")
        if config is None:
            if general_config_path.exists():
                with open(general_config_path, "r") as f:
                    general_config = yaml.safe_load(f)
                config = {
                    "model_path": general_config.get("model_path"),
                    "pmls": general_config.get("pmls", [0, 0, 0, 0, 10, 10]),
                    "max_iter": general_config.get("max_iter", 1000),
                    "tol": general_config.get("tol", 1e-5),
                    "verbose": general_config.get("verbose", False),
                    "restart": general_config.get("restart", 0),
                }
            else:
                default_model_path = str(Path(__file__).parent.parent / "models")
                config = {
                    "model_path": default_model_path,
                    "pmls": [0, 0, 0, 0, 10, 10],
                    "max_iter": 1000,
                    "tol": 1e-5,
                    "verbose": False,
                    "restart": 0,
                }
        inc_flux = _measure_incident_flux(src, dL, wvln, n_source, config)

    return {
        "transmittance": float(trans_flux / inc_flux),
        "transmitted_flux": float(trans_flux),
        "incident_flux": float(inc_flux),
    }


def compute_polarization_conversion(
    solution_field,
    monitor_z_idx,
    n_monitor,
    pol="rcp",
    wvln=None,
    dL=None,
):
    """
    Compute polarization conversion efficiency.
    
    Args:
        solution_field: Solution field tensor (bs, sx, sy, sz, 6|3)
        monitor_z_idx: Z-index of monitor plane
        n_monitor: Refractive index at monitor plane
        pol: Target polarization string: 'x', 'y', 'z', 'LCP', 'RCP' (default 'rcp')
        wvln: Wavelength in nm (optional)
        dL: Grid spacing in nm (optional)
    
    Returns:
        Dictionary with keys: conversion_efficiency, converted_flux, transmitted_flux
    """

    E = _field_to_complex(solution_field)
    trans_flux = poynting_flux(E, dL, wvln, n_monitor, monitor_z_idx).squeeze(0)

    pol_vec = _polarization_vector(pol)
    E_pol = _proj(E, pol_vec)
    pol_flux = poynting_flux(E_pol, dL, wvln, n_monitor, monitor_z_idx).squeeze(0)

    return {
        "conversion_efficiency": float(pol_flux / trans_flux),
        "converted_flux": float(pol_flux),
        "transmitted_flux": float(trans_flux),
    }


def compute_far_field_phase(
    solution_field,
    monitor_z_idx,
    pol="rcp",
):
    """
    Compute far field phase and amplitude.
    
    Args:
        solution_field: Solution field tensor (bs, sx, sy, sz, 6|3)
        monitor_z_idx: Z-index of monitor plane
        pol: Polarization string: 'x', 'y', 'z', 'LCP', 'RCP' (default 'rcp')
    
    Returns:
        Dictionary with keys: far_field_amplitude, far_field_phase_deg
    """

    e_complex = _field_to_complex(solution_field)
    e_plane = e_complex[:, :, :, monitor_z_idx, :].squeeze(0)
    pol_vec = _polarization_vector(pol)
    projection = (
        torch.conj(pol_vec[0]) * e_plane[..., 0]
        + torch.conj(pol_vec[1]) * e_plane[..., 1]
        + torch.conj(pol_vec[2]) * e_plane[..., 2]
    )

    spectrum = torch.fft.fft2(projection)
    far_field = spectrum[0, 0]
    amplitude = float(far_field.abs().item())
    phase_deg = float(torch.angle(far_field).item())
    return {
        "far_field_amplitude": amplitude,
        "far_field_phase_deg": phase_deg,
    }
