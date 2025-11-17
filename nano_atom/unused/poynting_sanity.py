import math
import numpy as np
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from nano_atom.analysis import P_DENSITY, poynting_flux
from nano_atom.source import set_source_plane, set_source_unidir
from src.solvers.NN_solver import NN_solve


@dataclass
class Experiment:
    """
    Experiment configuration dataclass.
    
    Attributes:
        sim_shape: Simulation shape tuple (sx, sy, sz)
        medium_eps: Medium permittivity
        source_amplitude: Source amplitude (default 1.0)
        source_polarization: Source polarization string (default "RCP")
        source_z_index: Source z-index (optional)
        monitor_z_index: Monitor z-index (optional)
        label: Experiment label string (optional)
    """
    sim_shape: Tuple[int, int, int]
    medium_eps: float
    source_amplitude: float = 1.0
    source_polarization: str = "RCP"
    source_z_index: int | None = None
    monitor_z_index: int | None = None
    label: str | None = None


def build_uniform_medium(
    sim_shape,
    _wl,
    _dL,
    pmls,
    kwargs,
):
    """
    Construct a uniform medium with a planar source.
    
    Args:
        sim_shape: Simulation shape tuple (sx, sy, sz)
        _wl: Wavelength in nm
        _dL: Grid spacing in nm
        pmls: PML configuration list
        kwargs: Dictionary with medium_eps, source_amplitude, source_polarization, source_z_index
    
    Returns:
        Tuple of (eps, src, z_idx) where eps and src are tensors with batch dimension
    """
    medium_eps = float(kwargs["medium_eps"])
    amp = float(kwargs.get("source_amplitude", 1.0))
    pol = kwargs.get("source_polarization", "RCP")
    z_idx = kwargs.get("source_z_index")
    
    if z_idx is None:
        # place the source 5 voxels above the lower z PML
        z_idx = int(pmls[4]) + 5

    sx, sy, sz = sim_shape
    if not (0 <= z_idx < sz):
        raise ValueError(f"Source plane z-index {z_idx} outside simulation bounds (0, {sz-1}).")

    eps = torch.full(sim_shape, medium_eps, dtype=torch.float32)
    src = torch.zeros((*sim_shape, 6), dtype=torch.float32)
    src = set_source_plane(src, z_idx, pol=pol, amp=amp, direction='forward', dL=_dL, wavelength=_wl, n=np.sqrt(medium_eps))

    return eps[None], src[None], z_idx


def run_experiment(base_cfg, experiment):
    """
    Run a single experiment: solve Maxwell equations and compute Poynting flux.
    
    Args:
        base_cfg: Base configuration dictionary
        experiment: Experiment dataclass instance
    
    Returns:
        Dictionary with experiment results including flux, expected flux, and metrics
    """
    cfg = deepcopy(base_cfg)
    cfg["sim_shape"] = list(experiment.sim_shape)

    eps, src, source_z = build_uniform_medium(
        experiment.sim_shape,
        float(cfg["wavelength"]),
        float(cfg["dL"]),
        cfg["pmls"],
        {
            "medium_eps": experiment.medium_eps,
            "source_amplitude": experiment.source_amplitude,
            "source_polarization": experiment.source_polarization,
            "source_z_index": experiment.source_z_index,
        },
    )

    solution, residual_history, final_residual = NN_solve(cfg, eps.clone(), src.clone())
    solution = solution.cpu()
    final_residual = final_residual.cpu()

    monitor_z = experiment.monitor_z_index
    if monitor_z is None:
        monitor_z = min(source_z + 1, experiment.sim_shape[2] - 1)

    refractive_index = math.sqrt(experiment.medium_eps)
    flux = poynting_flux(
        solution,
        float(cfg["dL"]),
        float(cfg["wavelength"]),
        refractive_index,
        monitor_z,
    ).squeeze().item()

    sx, sy, _ = experiment.sim_shape
    area = (sx * sy) * (float(cfg["dL"]) * 1e-9) ** 2  # m^2
    # Analytic calculation: flux = area * power_density * amplitude^2 * refractive_index
    expected_flux = area * P_DENSITY * (experiment.source_amplitude ** 2) * refractive_index

    return {
        "label": experiment.label
        or f"{experiment.sim_shape}, eps={experiment.medium_eps}, amp={experiment.source_amplitude}",
        "sim_shape": experiment.sim_shape,
        "medium_eps": experiment.medium_eps,
        "source_amplitude": experiment.source_amplitude,
        "source_z": source_z,
        "monitor_z": monitor_z,
        "flux_W": flux,
        "expected_W": expected_flux,
        "flux_to_expected": flux / expected_flux if expected_flux != 0 else float("nan"),
        "nn_residual": float(torch.mean(torch.abs(final_residual)).item()),
        "nn_iterations": len(residual_history),
        "refractive_index": refractive_index,
    }


def main():
    """
    Run experiments with different media and compare with analytic calculations.
    """
    base_cfg_path = "nano_atom/configs/nanofin_g.yaml"
    with open(base_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Define experiments with different media (permittivity values)
    experiments = [
        Experiment(
            sim_shape=(32, 32, base_cfg["sim_shape"][2]),
            medium_eps=1.0,
            source_amplitude=1.0,
            label="Air (eps=1.0)"
        ),
        Experiment(
            sim_shape=(32, 32, base_cfg["sim_shape"][2]),
            medium_eps=2.25,  # e.g., SiO2
            source_amplitude=1.0,
            label="SiO2-like (eps=2.25)"
        ),
        Experiment(
            sim_shape=(32, 32, base_cfg["sim_shape"][2]),
            medium_eps=1.0,
            source_amplitude=2.0,
            label="Air (eps=1.0, amp=2.0)"
        ),
        Experiment(
            sim_shape=(64, 64, base_cfg["sim_shape"][2]),
            medium_eps=1.0,
            source_amplitude=1.0,
            label="Air 64x64 (eps=1.0)"
        ),
        Experiment(
            sim_shape=(64, 64, base_cfg["sim_shape"][2]),
            medium_eps=2.25,
            source_amplitude=1.0,
            label="SiO2-like 64x64 (eps=2.25)"
        ),
    ]

    print("=" * 80)
    print("Poynting Flux Comparison: Simulation vs Analytic Calculation")
    print("=" * 80)
    print(f"P_DENSITY (reference power density): {P_DENSITY:.6e} W/m^2")
    print()

    results = []
    for i, exp in enumerate(experiments):
        print(f"Experiment {i+1}/{len(experiments)}: {exp.label}")
        print(f"  Medium: eps = {exp.medium_eps}, n = {math.sqrt(exp.medium_eps):.3f}")
        print(f"  Source amplitude: {exp.source_amplitude}")
        print(f"  Simulating...")
        
        res = run_experiment(base_cfg, exp)
        results.append(res)
        
        print(f"  Source z-index: {res['source_z']}, Monitor z-index: {res['monitor_z']}")
        print(f"  Computed flux:     {res['flux_W']:.6e} W")
        print(f"  Expected flux:     {res['expected_W']:.6e} W")
        print(f"  Ratio (comp/exp):  {res['flux_to_expected']:.4f}")
        print(f"  NN residual:       {res['nn_residual']:.3e}")
        print(f"  NN iterations:     {res['nn_iterations']}")
        print()

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Label':<30} {'Flux (W)':<15} {'Expected (W)':<15} {'Ratio':<10} {'Residual':<12}")
    print("-" * 80)
    for res in results:
        print(
            f"{res['label']:<30} "
            f"{res['flux_W']:>14.6e} "
            f"{res['expected_W']:>14.6e} "
            f"{res['flux_to_expected']:>9.4f} "
            f"{res['nn_residual']:>11.3e}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
