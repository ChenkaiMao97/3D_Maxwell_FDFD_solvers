import torch
import numpy as np
from src.utils.utils import get_pixels

def axisymetric_source(sx, sy, sz, src_z, mode="radial"):
    """
    Axisymmetric source sheet at z = src_z.

    mode:
      "ez"        -> Ez only (simplest m=0)
      "radial"    -> Er mapped to Ex,Ey (m=0)
      "azimuthal" -> Ephi mapped to Ex,Ey (m=0)

    Returns:
      (sx, sy, sz, 6):
      [Re(Ex), Im(Ex), Re(Ey), Im(Ey), Re(Ez), Im(Ez)]
    """
    assert 0 <= src_z < sz

    cx = (sx - 1) / 2
    cy = (sy - 1) / 2
    x = np.arange(sx) - cx
    y = np.arange(sy) - cy
    X, Y = np.meshgrid(x, y, indexing="ij")

    R = np.sqrt(X**2 + Y**2)
    R_safe1 = np.maximum(R, 1e-12)
    R_safe2 = np.maximum(2*R, 1e-12)

    # Axisymmetric amplitude profile A(r)
    sigma = 0.2 * min(sx, sy)
    A = np.exp(-(R / sigma) ** 2).astype(np.complex64)

    src = np.zeros((sx, sy, sz, 6), dtype=np.float32)

    # if mode == "ez":
    #     src[:, :, src_z, 4] = A.real
    #     src[:, :, src_z, 5] = A.imag

    # elif mode == "radial":
    #     Ex = A * (X / R_safe)
    #     Ey = A * (Y / R_safe)

    #     src[:, :, src_z, 0] = Ex.real
    #     src[:, :, src_z, 1] = Ex.imag
    #     src[:, :, src_z, 2] = Ey.real
    #     src[:, :, src_z, 3] = Ey.imag

    Ex = A * (X / R_safe1)
    Ey = A * (Y / R_safe1)

    src[:, :, src_z, 0] += Ex.real
    src[:, :, src_z, 1] += Ex.imag
    src[:, :, src_z, 2] += Ey.real
    src[:, :, src_z, 3] += Ey.imag

    # elif mode == "azimuthal":
    #     Ex = -A * (Y / R_safe)
    #     Ey =  A * (X / R_safe)

    #     src[:, :, src_z, 0] = Ex.real
    #     src[:, :, src_z, 1] = Ex.imag
    #     src[:, :, src_z, 2] = Ey.real
    #     src[:, :, src_z, 3] = Ey.imag

    Ex = -A * (Y / R_safe2)
    Ey =  A * (X / R_safe2)

    src[:, :, src_z, 0] += Ex.real
    src[:, :, src_z, 1] += Ex.imag
    src[:, :, src_z, 2] += Ey.real
    src[:, :, src_z, 3] += Ey.imag

    # else:
    #     raise ValueError("mode must be 'ez', 'radial', or 'azimuthal'")

    return torch.from_numpy(src)

def circle_supersample(h, w, cx, cy, r, ss=8, dtype=np.float32):
    """
    Per-pixel coverage for a disk centered at (cx, cy) with radius r.
    Pixel centers are at integer coords (i, j). Returns HxW in [0,1].
    ss: supersamples per axis (e.g., 4, 8). Higher = smoother/more accurate.
    """
    # pixel grid
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    # stratified subpixel offsets in [0,1)
    ofs = (np.arange(ss) + 0.5) / ss
    oy, ox = np.meshgrid(ofs, ofs, indexing="ij")  # (ss, ss)

    # sample coordinates (broadcast to HxW)
    X = x[..., None, None] + ox  # (H, W, ss, ss)
    Y = y[..., None, None] + oy

    # distance to circle center
    d2 = (X - cx)**2 + (Y - cy)**2
    inside = d2 <= r*r

    # average over subpixels → coverage in [0,1]
    cov = inside.mean(axis=(-1, -2)).astype(dtype)
    return torch.from_numpy(cov)

def make_rings(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a simple cylinder-shaped meta atom
    """
    assert len(sim_shape) == 3
    substrate_eps = kwargs['substrate_eps']
    medium_eps = kwargs['medium_eps']
    medium_height_pixel = get_pixels(kwargs, 'medium_height_nm', dL)
    ring_radius1_pixel = get_pixels(kwargs, 'ring_radius1_nm', dL)
    ring_width1_pixel = get_pixels(kwargs, 'ring_width1_nm', dL)
    ring_radius2_pixel = get_pixels(kwargs, 'ring_radius2_nm', dL)
    ring_width2_pixel = get_pixels(kwargs, 'ring_width2_nm', dL)
    ring_position_z_pixel = get_pixels(kwargs, 'ring_position_z_nm', dL)
    source_polarization = kwargs['source_polarization']
    source_position_z_pixel = get_pixels(kwargs, 'source_position_z_nm', dL)

    ### make the circular cross-section in x,y:
    outer_ring = circle_supersample(sim_shape[0], sim_shape[1], sim_shape[0] // 2, sim_shape[1] // 2, ring_radius2_pixel + ring_width2_pixel//2) - \
                 circle_supersample(sim_shape[0], sim_shape[1], sim_shape[0] // 2, sim_shape[1] // 2, ring_radius2_pixel - ring_width2_pixel//2)
    inner_ring = circle_supersample(sim_shape[0], sim_shape[1], sim_shape[0] // 2, sim_shape[1] // 2, ring_radius1_pixel + ring_width1_pixel//2) - \
                 circle_supersample(sim_shape[0], sim_shape[1], sim_shape[0] // 2, sim_shape[1] // 2, ring_radius1_pixel - ring_width1_pixel//2)
    cross = 1.0 + (medium_eps - 1.0) * (outer_ring + inner_ring)
    
    sx, sy, sz = sim_shape
    eps = torch.ones(sim_shape)
    eps[...,sz//2 - medium_height_pixel//2:sz//2 + medium_height_pixel//2] = cross[...,None]
    eps[...,:sz//2 - medium_height_pixel//2] = substrate_eps

    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[:,:,source_position_z_pixel, 0] = 1
    elif source_polarization == 'y':
        src[:,:,source_position_z_pixel, 2] = 1
    elif source_polarization == 'z':
        src[:,:,source_position_z_pixel, 4] = 1
    elif source_polarization == 'lcp':
        src[:,:,source_position_z_pixel, 0] = 1
        src[:,:,source_position_z_pixel, 3] = 1
    elif source_polarization == 'rcp':
        src[:,:,source_position_z_pixel, 0] = 1
        src[:,:,source_position_z_pixel, 3] = -1

    # src = axisymetric_source(sim_shape[0], sim_shape[1], sim_shape[2], source_position_z_pixel, mode="azimuthal")

    return eps[None], src[None] # add a batch dimension