import torch
import numpy as np
from src.utils.utils import get_pixels

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


def make_meta_atom_cylinder(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a simple cylinder-shaped meta atom
    """
    assert len(sim_shape) == 3
    substrate_eps = kwargs['substrate_eps']
    meta_atom_eps = kwargs['meta_atom_eps']
    top_medium_eps = kwargs['top_medium_eps']
    source_polarization = kwargs['source_polarization']
    
    substrate_thickness_pixel = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    meta_atom_height_pixel = get_pixels(kwargs, 'meta_atom_height_nm', dL)
    meta_atom_size_pixel = get_pixels(kwargs, 'meta_atom_size', dL)
    source_below_meta_pixel = get_pixels(kwargs, 'source_below_meta_nm', dL)
    assert substrate_thickness_pixel > source_below_meta_pixel

    eps = top_medium_eps * torch.ones(sim_shape)
    eps[:,:,:substrate_thickness_pixel] = substrate_eps

    ### make the circular cross-section in x,y:
    cross = top_medium_eps + (meta_atom_eps - top_medium_eps) * circle_supersample(sim_shape[0], sim_shape[1], sim_shape[0] // 2, sim_shape[1] // 2, meta_atom_size_pixel // 2)
    eps[:,:,substrate_thickness_pixel:substrate_thickness_pixel + meta_atom_height_pixel] = cross[...,None]

    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[:,:,substrate_thickness_pixel - source_below_meta_pixel, 0] = 1
    elif source_polarization == 'y':
        src[:,:,substrate_thickness_pixel - source_below_meta_pixel, 2] = 1
    elif source_polarization == 'z':
        src[:,:,substrate_thickness_pixel - source_below_meta_pixel, 4] = 1

    return eps[None], src[None] # add a batch dimension