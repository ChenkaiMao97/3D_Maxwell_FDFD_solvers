import numpy as np
import torch
import torch.nn.functional as F
from src.utils.utils import get_pixels


# Binarization and filtering functions for pixel-level super pixel inverse design
def identity_op(x, beta=0.0):
    return x

def make_conic_filter(radius: int):
    """
    Make a conic (cone-shaped) 2D kernel of size (2r+1) × (2r+1).
    Values increase linearly from center to radius, then normalized to sum=1.
    Returned shape: (1, 1, H, W) for conv2d.
    """
    filt = torch.zeros(2 * radius + 1, 2 * radius + 1, dtype=torch.float32)
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            d = np.sqrt((i - radius) ** 2 + (j - radius) ** 2)
            if d <= radius:
                filt[i, j] = d / float(radius)  # 0 at center, 1 at rim
    filt = filt / torch.sum(filt)
    return filt[None, None, :, :]  # (out_ch=1, in_ch=1, H, W)

def conic_filter(x: torch.Tensor, radius: int, periodic: bool = False):
    """
    Apply the cone kernel via conv2d.
    x: (N, C, H, W). The kernel is applied depthwise per-channel by looping.
    If periodic=True, circular padding; else zero padding.
    """
    N, C, H, W = x.shape
    kernel = make_conic_filter(radius).to(x.device, x.dtype)  # (1,1,kh,kw)
    pad_mode = 'circular' if periodic else 'constant'
    xpad = F.pad(x, (radius, radius, radius, radius), mode=pad_mode)
    # Apply to each channel separately to avoid cross-channel mixing
    out = []
    for c in range(C):
        xc = xpad[:, c:c+1, :, :]
        yc = F.conv2d(xc, kernel, padding=0)
        out.append(yc)
    return torch.cat(out, dim=1)

def tanh_proj(x: torch.Tensor, beta: float, eta: float = 0.5):
    """
    Smooth, monotone projection to [0,1] with slope controlled by beta
    and inflection at eta in (0,1). Works elementwise and broadcasts.
    """
    # ensure dtype/device consistency
    eta_t = torch.as_tensor(eta, device=x.device, dtype=x.dtype)
    num = torch.tanh(beta * eta_t) + torch.tanh(beta * (x - eta_t))
    den = torch.tanh(beta * eta_t) + torch.tanh(beta * (1 - eta_t))
    return num / den

def make_super_pixel_binary(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a freeform-shaped super pixel 
    """
    assert len(sim_shape) == 3
    substrate_eps = kwargs['substrate_eps']
    meta_atom_eps = kwargs['meta_atom_eps']
    top_medium_eps = kwargs['top_medium_eps']
    source_polarization = kwargs['source_polarization']

    # Binarization parameters:
    beta = kwargs['beta']
    eta = kwargs['eta']
    
    substrate_thickness_pixel = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    meta_atom_height_pixel = get_pixels(kwargs, 'meta_atom_height_nm', dL)
    source_below_meta_pixel = get_pixels(kwargs, 'source_below_meta_nm', dL)
    assert substrate_thickness_pixel > source_below_meta_pixel

    # Create a toy single-channel image: radial ramp in [0,1]
    H, W = sim_shape[0], sim_shape[1]
    n = torch.randn(1, 1, H, W)
    ms = sum(F.interpolate(F.avg_pool2d(n, s, s), size=(H, W), mode='bilinear', align_corners=False) for s in
             [2, 4, 8, 16]) / 4
    tex = ms + 0.15 * torch.randn_like(n)
    tex = (tex - tex.amin((-2, -1), True)) / (tex.amax((-2, -1), True) - tex.amin((-2, -1), True) + 1e-8)
    img = torch.round(tex * 6) / 6
    img = (img + 0.07 * torch.randn_like(img)).clamp(0, 1)

    # 2) Conic filtering
    radius = 5
    x_conic_zero = conic_filter(img, radius=radius, periodic=False)

    # 3) Smooth "thresholding" via tanh_proj
    #    Make a copy that is slightly blurred, then project to [0,1]
    x_proj = tanh_proj(x_conic_zero, beta=beta, eta=eta)

    eps = top_medium_eps * torch.ones(sim_shape)
    eps[:,:,:substrate_thickness_pixel] = substrate_eps

    ### make the freeform super pixel cross-section in x,y:
    cross = top_medium_eps + (meta_atom_eps - top_medium_eps) * x_proj[0,0,:,:]
    eps[:,:,substrate_thickness_pixel:substrate_thickness_pixel + meta_atom_height_pixel] = cross[...,None]

    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[:,:,substrate_thickness_pixel - source_below_meta_pixel, 0] = 1
    elif source_polarization == 'y':
        src[:,:,substrate_thickness_pixel - source_below_meta_pixel, 2] = 1
    elif source_polarization == 'z':
        src[:,:,substrate_thickness_pixel - source_below_meta_pixel, 4] = 1

    return eps[None], src[None] # add a batch dimension



