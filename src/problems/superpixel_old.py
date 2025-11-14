import torch
import numpy as np
from src.utils.utils import get_pixels

# import src.simulator
from src.utils.PML_utils import make_dxes_numpy
from src.utils.physics import eps_to_yee
import scipy.sparse as sparse

import matplotlib.pyplot as plt

def ellipse_supersample(h, w, cx, cy, rx, ry, theta=0.0, ss=8, dtype=np.float32):
    """
    Per-pixel coverage for an ellipse centered at (cx, cy) with radii (rx, ry),
    rotated by angle 'theta' (radians, counterclockwise). Pixel centers are at
    integer coords (i, j). Returns a HxW Torch tensor in [0,1].

    ss: supersamples per axis (e.g., 4, 8). Higher = smoother/more accurate.
    """

    # pixel grid
    y = np.arange(h)[:, None]           # (H, 1)
    x = np.arange(w)[None, :]           # (1, W)

    # stratified subpixel offsets in [0,1)
    ofs = (np.arange(ss) + 0.5) / ss
    oy, ox = np.meshgrid(ofs, ofs, indexing="ij")  # (ss, ss)

    # sample coordinates (broadcast to HxW)
    X = x[..., None, None] + ox         # (H, W, ss, ss)
    Y = y[..., None, None] + oy         # (H, W, ss, ss)

    # move to ellipse-centered coordinates
    Xc = X - cx
    Yc = Y - cy

    # un-rotate points by -theta so ellipse is axis-aligned
    c, s = np.cos(theta), np.sin(theta)
    Xr =  c * Xc + s * Yc
    Yr = -s * Xc + c * Yc

    # inside test for axis-aligned ellipse: (x/rx)^2 + (y/ry)^2 <= 1
    # (guard against zero radii)
    rx = max(rx, 1e-12)
    ry = max(ry, 1e-12)
    inside = (Xr / rx) ** 2 + (Yr / ry) ** 2 <= 1.0

    # average over subpixels → coverage in [0,1]
    cov = inside.mean(axis=(-1, -2)).astype(dtype)  # (H, W)
    return torch.from_numpy(cov)

def make_meta_atom(xy_map, pmls):
    """
    makes a single meta atom
    """
    count = 0
    while True:
        rx = np.random.randint(8, 9)
        ry = np.random.randint(16, 17)
        radius = max(rx, ry)
        cx = np.random.randint(pmls[0] + 1 + radius, xy_map.shape[0] - pmls[1]-1 - radius)
        cy = np.random.randint(pmls[2] + 1 + radius, xy_map.shape[1] - pmls[3]-1 - radius)
        theta = np.random.rand() * np.pi
        meta_atom = ellipse_supersample(xy_map.shape[0], xy_map.shape[1], cx, cy, rx, ry, theta)
        if torch.max(xy_map + meta_atom) > 1.0 and count < 20:
            count += 1
            continue
        break
    xy_map += meta_atom
    xy_map = xy_map.clamp(0, 1)
    return xy_map

def make_superpixel(sim_shape, wl, dL, pmls, kwargs):
    """
    assumes that the waveguide bend is centered in x,y,z, source is parallel to xz plane, propagating in y direction
    """
    # shape, pmls, eps_sub=2.25, eps_meta_max=8.0, meta_height_pixels = 10, width = 10, radius=10):
    eps_sub = kwargs['substrate_eps']
    eps_meta = kwargs['meta_eps']
    eps_top = kwargs['top_medium_eps']
    ln_R = kwargs['ln_R']
    source_polarization = kwargs['source_polarization']
    num_meta_atoms = kwargs['num_meta_atoms']

    meta_height = get_pixels(kwargs, 'meta_height_nm', dL)
    source_below_meta = get_pixels(kwargs, 'source_below_meta_nm', dL)
    st = round(sim_shape[2]/2-meta_height/2) # substrate thickness in pixels
    source_z = st - source_below_meta
    
    eps = eps_top * torch.ones(sim_shape, dtype=torch.float32)
    eps[:,:,:st] = eps_sub

    # make the curved portion
    xy_map = torch.zeros(sim_shape[0], sim_shape[1], dtype=torch.float32) 
    for i in range(num_meta_atoms):
        xy_map = make_meta_atom(xy_map, pmls)
    xy_map = xy_map * (eps_meta - eps_top) + eps_top
    
    eps[:,:,st:st + meta_height] = xy_map[:,:,None]
    
    ############### (2) plane wave source #############
    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[pmls[0] + 1:sim_shape[0] - pmls[1]-1, pmls[2] + 1:sim_shape[1] - pmls[3]-1, source_z, 0] = 1
    elif source_polarization == 'y':
        src[pmls[0] + 1:sim_shape[0] - pmls[1]-1, pmls[2] + 1:sim_shape[1] - pmls[3]-1, source_z, 2] = 1
    elif source_polarization == 'z':
        src[pmls[0] + 1:sim_shape[0] - pmls[1]-1, pmls[2] + 1:sim_shape[1] - pmls[3]-1, source_z, 4] = 1

    return eps[None], src[None]