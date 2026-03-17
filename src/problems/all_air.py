import torch
import numpy as np
from src.utils.utils import get_pixels

def make_all_air(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a all-air simulation
    """
    assert len(sim_shape) == 3
    eps = 1.0 * torch.ones(sim_shape)
    source_polarization = kwargs['source_polarization']
    source_position_z = get_pixels(kwargs, 'source_position_z_nm', dL)

    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[:,:,source_position_z, 0] = 1
    elif source_polarization == 'y':
        src[:,:,source_position_z, 2] = 1
    elif source_polarization == 'z':
        src[:,:,source_position_z, 4] = 1

    return eps[None], src[None] # add a batch dimension