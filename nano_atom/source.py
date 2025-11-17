import torch
import numpy as np

# -----------------------
# helpers
# -----------------------
def _write_complex_plane(src, z_idx, Ex=None, Ey=None, Ez=None):
    """
    Write complex fields into src[..., 6] at plane z_idx.
    
    Args:
        src: Source tensor with shape (sx, sy, sz, 6)
        z_idx: Z-index of the plane to write
        Ex: Complex Ex field (optional)
        Ey: Complex Ey field (optional)
        Ez: Complex Ez field (optional)
    
    Returns:
        Modified source tensor
    """
    if Ex is not None:
        if torch.is_complex(Ex):
            src[:, :, z_idx, 0] = Ex.real
            src[:, :, z_idx, 1] = Ex.imag
        else:
            src[:, :, z_idx, 0] = Ex
            src[:, :, z_idx, 1] = 0.0
    if Ey is not None:
        if torch.is_complex(Ey):
            src[:, :, z_idx, 2] = Ey.real
            src[:, :, z_idx, 3] = Ey.imag
        else:
            src[:, :, z_idx, 2] = Ey
            src[:, :, z_idx, 3] = 0.0
    if Ez is not None:
        if torch.is_complex(Ez):
            src[:, :, z_idx, 4] = Ez.real
            src[:, :, z_idx, 5] = Ez.imag
        else:
            src[:, :, z_idx, 4] = Ez
            src[:, :, z_idx, 5] = 0.0
    return src


def _xy_coords(sx, sy, dL, device, dtype, center=True):
    """
    Return 2D coordinate grids (x,y) in meters for a pixel grid.
    dL is in nm, converted to meters by multiplying by 1e-9.
    
    Args:
        sx: Size in x direction
        sy: Size in y direction
        dL: Grid spacing in nm
        device: Torch device
        dtype: Torch dtype
        center: If True, center coordinates around origin
    
    Returns:
        Tuple of (X, Y) coordinate grids in meters
    """
    x_range = torch.arange(sx, device=device, dtype=dtype)
    y_range = torch.arange(sy, device=device, dtype=dtype)
    if center:
        x_range = x_range - (sx // 2)
        y_range = y_range - (sy // 2)
    X, Y = torch.meshgrid(x_range, y_range, indexing='xy')
    X = X * dL * 1e-9   
    Y = Y * dL * 1e-9
    X = X.to(device)
    Y = Y.to(device)
    return X, Y

def _pol_vector(pol, amp=1.0, norm=True, device=None, dtype=None):
    """
    Return polarization vector components (Ex, Ey, Ez).
    
    Args:
        pol: Polarization string: 'x', 'y', 'z', 'LCP', 'RCP'
        amp: Amplitude (default 1.0)
        norm: If True, normalize LCP/RCP to sqrt(2) (default True)
        device: Torch device (optional)
        dtype: Torch dtype (optional)
    
    Returns:
        Tuple of (Ex, Ey, Ez) complex scalars
    """
    pol = pol.upper()
    a = torch.tensor(amp, device=device, dtype=dtype)
    j = 1j if dtype is None or torch.complex64 == dtype or torch.complex128 == dtype else 1j  # fine
    if pol == 'X':
        return a+0j, torch.zeros((), device=device, dtype=a.dtype)+0j, torch.zeros((), device=device, dtype=a.dtype)+0j
    if pol == 'Y':
        return torch.zeros((), device=device, dtype=a.dtype)+0j, a+0j, torch.zeros((), device=device, dtype=a.dtype)+0j
    if pol == 'Z':
        return torch.zeros((), device=device, dtype=a.dtype)+0j, torch.zeros((), device=device, dtype=a.dtype)+0j, a+0j
    if pol == 'LCP':
        s = a/np.sqrt(2) if norm else a
        return s+0j, (1j*s), torch.zeros((), device=device, dtype=a.dtype)+0j
    if pol == 'RCP':
        s = a/np.sqrt(2) if norm else a
        return s+0j, (-1j*s), torch.zeros((), device=device, dtype=a.dtype)+0j
    raise ValueError("Unknown polarization; use 'x', 'y', 'z', 'LCP', or 'RCP'.")


# -----------------------
# sources
# -----------------------
def set_source_bidir(src, z_idx, pol, amp=1.0, norm=True):
    """
    Set a source plane with either linear or circular polarization.

    Args:
        src: Source array, shape (sx, sy, sz, 6), channels [Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i]
        z_idx: Z-index to set the source at
        pol: Polarization string: 'x', 'y', 'z', 'LCP', 'RCP'
        amp: Amplitude (default 1.0)
        norm: Normalize for circular (default True)
    
    Returns:
        Modified source tensor
    """
    pol = pol.upper()
    if pol in ['X', 'Y', 'Z']:
        if pol == 'X':
            src[:, :, z_idx, 0] = amp
        elif pol == 'Y':
            src[:, :, z_idx, 2] = amp
        elif pol == 'Z':
            src[:, :, z_idx, 4] = amp
    elif pol in ['LCP', 'RCP']:
        scale = (amp / np.sqrt(2)) if norm else amp
        src[:, :, z_idx, 0] = scale  # Re(Ex)
        src[:, :, z_idx, 3] = scale if pol == 'LCP' else -scale  # Im(Ey)
    else:
        raise ValueError("Unknown polarization; use 'x', 'y', 'z', 'LCP', or 'RCP'.")
    return src


def set_source_unidir(
    src,
    z_idx,
    pol='x',
    amp=1.0,
    dL=20.0,
    kx=0.0,
    ky=0.0,
    kz=None,
    n=None,
    wavelength=None,
    x=None,
    y=None,
    norm=True,
    direction='forward'
):
    """
    Build a one-way plane wave using two phased sheets:
      forward (+z): at z_idx and z_idx-1 with [1, -exp(+i kz dL)]
      backward (-z): at z_idx and z_idx+1 with [1, -exp(-i kz dL)]
    
    Args:
        src: Source tensor, shape (sx, sy, sz, 6)
        z_idx: Z-index to set the source at
        pol: Polarization string: 'x', 'y', 'z', 'LCP', 'RCP' (default 'x')
        amp: Amplitude (default 1.0)
        dL: Grid spacing in nm (default 20.0)
        kx: Wave vector x component in rad/m (default 0.0)
        ky: Wave vector y component in rad/m (default 0.0)
        kz: Wave vector z component in rad/m (optional)
        n: Refractive index (required if kz is None)
        wavelength: Wavelength in nm (required if kz is None)
        x: X coordinate grid (optional)
        y: Y coordinate grid (optional)
        norm: Normalize for circular polarization (default True)
        direction: Direction string: 'forward' (+z) or 'backward' (-z) (default 'forward')
    
    Returns:
        Modified source tensor
    """
    device = src.device
    cdtype = torch.complex64 if src.dtype in (torch.float32, torch.complex64) else torch.complex128

    sx, sy, sz, ch = src.shape
    assert ch == 6, "src[...,6] expected"

    dz_m = dL * 1e-9

    if kz is None:
        if n is None or wavelength is None:
            raise ValueError("Provide either kz, or (n and wavelength) to compute kz.")
        k0 = 2*np.pi / float(wavelength ** 1e-9)
        kk = (float(n) * k0)**2 - kx**2 - ky**2
        kz = np.sqrt(kk + 0j)  # principal branch

    if x is None or y is None:
        x, y = _xy_coords(sx, sy, dL, device=device, dtype=src.dtype, center=True)
    map_xy = torch.exp(-1j * (kx*x + ky*y)).to(cdtype)

    Ex0_s, Ey0_s, Ez0_s = _pol_vector(pol, amp=amp, norm=norm, device=device, dtype=src.dtype)
    Ex0 = torch.ones((sx, sy), device=device, dtype=cdtype) * (Ex0_s.to(cdtype))
    Ey0 = torch.ones((sx, sy), device=device, dtype=cdtype) * (Ey0_s.to(cdtype))
    Ez0 = torch.ones((sx, sy), device=device, dtype=cdtype) * (Ez0_s.to(cdtype))

    if direction.lower() in ('forward', '+z', 'unidirectional', 'one-way'):
        if z_idx <= 0:
            raise IndexError("Need z_idx >= 1 for forward (+z) source (uses z_idx-1).")
        phase_step = torch.tensor(np.exp(1j * kz * dz_m), device=device, dtype=cdtype)
        s0, s1 = 1.0 + 0j, -(phase_step)            # [z_idx, z_idx-1]
        z0, z1 = z_idx, z_idx - 1
    elif direction.lower() in ('backward', '-z'):
        if z_idx >= sz - 1:
            raise IndexError("Need z_idx <= sz-2 for backward (-z) source (uses z_idx+1).")
        phase_step = torch.tensor(np.exp(-1j * kz * dz_m), device=device, dtype=cdtype)
        s0, s1 = 1.0 + 0j, -(phase_step)            # [z_idx, z_idx+1]
        z0, z1 = z_idx, z_idx + 1
    else:
        raise ValueError("direction must be 'forward' or 'backward'.")

    Ex1 = Ex0 * map_xy * s0
    Ey1 = Ey0 * map_xy * s0
    Ez1 = Ez0 * map_xy * s0

    Ex2 = Ex0 * map_xy * s1
    Ey2 = Ey0 * map_xy * s1
    Ez2 = Ez0 * map_xy * s1

    _write_complex_plane(src, z0, Ex1, Ey1, Ez1)
    _write_complex_plane(src, z1, Ex2, Ey2, Ez2)

    return src


def set_source_plane(
    src,
    z_idx,
    amp=1.0,
    pol='x',
    direction='bidirectional',
    dL=20.0,
    kx=0.0,
    ky=0.0,
    kz=None,
    n=None,
    wavelength=None,
    x=None,
    y=None,
    norm=True
):
    """
    Set a source plane with specified polarization and direction.
    
    Args:
        src: Source tensor, shape (sx, sy, sz, 6)
        z_idx: Z-index to set the source at
        amp: Amplitude (default 1.0)
        pol: Polarization string: 'x', 'y', 'z', 'LCP', 'RCP' (default 'x')
        direction: Direction string: 'bidirectional', 'forward', 'backward' (default 'bidirectional')
        dL: Grid spacing in nm (default 20.0)
        kx: Wave vector x component in rad/m (default 0.0)
        ky: Wave vector y component in rad/m (default 0.0)
        kz: Wave vector z component in rad/m (optional)
        n: Refractive index (optional)
        wavelength: Wavelength in nm (optional)
        x: X coordinate grid (optional)
        y: Y coordinate grid (optional)
        norm: Normalize for circular polarization (default True)
    
    Returns:
        Modified source tensor
    """
    if direction.lower() in ('bidirectional','bidir','two-way'):
        return set_source_bidir(src, z_idx, pol, amp, norm=norm)
    elif direction.lower() in ('forward','unidirectional','one-way','+z'):
        return set_source_unidir(src, z_idx, pol, amp, dL, kx, ky, kz, n, wavelength, x, y, norm, 'forward')
    elif direction.lower() in ('backward','-z'):
        return set_source_unidir(src, z_idx, pol, amp, dL, kx, ky, kz, n, wavelength, x, y, norm, 'backward')
    else:
        raise ValueError("direction must be 'bidirectional', 'forward', or 'backward'.")