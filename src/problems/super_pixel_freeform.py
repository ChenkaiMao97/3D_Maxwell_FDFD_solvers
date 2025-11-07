import numpy as np
import torch
import torch.nn.functional as F
from src.utils.utils import get_pixels

# Get super pixel sample from this file --> shape: [50, 1, 2000, 2000]
loaded_data = np.load("/home/chenkaim/scripts/models/MAML_EM_simulation/data_gen/boundary_CG_ieterative_gen/shape_gen/freeform_gen_2000_2000_binary_tilted_50devices.npy")
print(loaded_data.shape)  # (N, H, W)

def make_super_pixel_freeform(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a freeform-shaped super pixel 
    """
    assert len(sim_shape) == 3
    substrate_eps = kwargs['substrate_eps']
    meta_atom_eps = kwargs['meta_atom_eps']
    top_medium_eps = kwargs['top_medium_eps']
    source_polarization = kwargs['source_polarization']
    pattern_size = kwargs['pattern_size']

    # Define the super pixel to be sampled in the dataset:
    mat_idx = kwargs['material_idx']
    start_x = kwargs['start_x']
    start_y = kwargs['start_y']
    
    substrate_thickness_pixel = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    meta_atom_height_pixel = get_pixels(kwargs, 'meta_atom_height_nm', dL)
    source_below_meta_pixel = get_pixels(kwargs, 'source_below_meta_nm', dL)
    assert substrate_thickness_pixel > source_below_meta_pixel

    eps = top_medium_eps * torch.ones(sim_shape)
    eps[:,:,:substrate_thickness_pixel] = substrate_eps

    ### make the freeform super pixel cross-section in x,y:
    cross = top_medium_eps + (meta_atom_eps - top_medium_eps) * loaded_data[mat_idx, 0, start_x:start_x + pattern_size[0], start_y:start_y + pattern_size[1]]
    cross = top_medium_eps * np.ones((pattern_size[0], pattern_size[1]))
    cross = torch.from_numpy(cross.astype(np.float32))
    pad_x = (sim_shape[0] - cross.shape[0]) // 2
    pad_y = (sim_shape[1] - cross.shape[1]) // 2
    cross = F.pad(cross, (pad_x, pad_x, pad_y, pad_y), mode="constant", value=1)  # pad to the sim_shape

    eps[:,:,substrate_thickness_pixel:substrate_thickness_pixel + meta_atom_height_pixel] = cross[...,None]

    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[pad_x:-pad_x, pad_y:-pad_y, substrate_thickness_pixel - source_below_meta_pixel, 0] = 1
    elif source_polarization == 'y':
        src[pad_x:-pad_x, pad_y:-pad_y, substrate_thickness_pixel - source_below_meta_pixel, 2] = 1
    elif source_polarization == 'z':
        src[pad_x:-pad_x, pad_y:-pad_y, substrate_thickness_pixel - source_below_meta_pixel, 4] = 1

    return eps[None], src[None] # add a batch dimension



