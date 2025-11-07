import numpy as np
import torch
import torch.nn.functional as F
from src.utils.utils import get_pixels

# Reparametrization functions for super pixel patterns
def sigmoid(x):
    return torch.sigmoid(x)


def eps_calcu(xm, ym, a, b, theta, delta, T1, grid_x, grid_y):
    exponent = -torch.pow((((grid_x - xm) * torch.cos(theta) + (grid_y - ym - delta) * torch.sin(theta)) ** 2 / a ** 2
                           + ((grid_y - ym - delta) * torch.cos(theta) - (grid_x - xm) * torch.sin(
                theta)) ** 2 / b ** 2), 1 / T1)
    return torch.exp(exponent)


def eps_calcu_bi(xm, ym, a, b, theta, delta, T1, grid_x, grid_y, threshold=0.5):
    exponent = -torch.pow((((grid_x - xm) * torch.cos(theta) + (grid_y - ym - delta) * torch.sin(theta)) ** 2 / a ** 2
                           + ((grid_y - ym - delta) * torch.cos(theta) - (grid_x - xm) * torch.sin(
                theta)) ** 2 / b ** 2), 1 / T1)
    region = torch.exp(exponent)
    return (region >= threshold).float()


def orthogonalize(s):
    n = s.size(1)
    t = s[:, 0:1]
    for i in range(1, n):
        sigma = torch.sum(t ** 2, dim=1, keepdim=True)
        ti = s[:, i].unsqueeze(1) * torch.sqrt(1 - sigma)
        t = torch.cat((t, ti), dim=1)
    return t


def reparam_pattern(
        arr,
        size,
        mfs,
        m,
        n,
        iteration
):
    iteration = torch.tensor(iteration)
    T1 = torch.exp(-0.025 * iteration)

    device = arr.device
    batch_size = arr.size(0)
    M = m
    N = n

    # Create grid only once for all batches
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, size[0] - 1, steps=size[0], device=device),
        torch.linspace(0, size[1] - 1, steps=size[1], device=device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 54, 54]
    grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 54, 54]

    # Sigmoid applied once for all the values at once
    s_col = sigmoid(arr[:, :, 0])  # Shape: [batch_size, M]
    t_col = orthogonalize(s_col)  # Shape: [batch_size, M]

    # Compute w_col for all batches at once
    w_col = 2 * mfs + (size[0] - t_col.size(1) * 2 * mfs) * t_col / torch.sum(t_col, dim=1, keepdim=True)
    y_cumsum = torch.cumsum(w_col, dim=1)

    # Prepare the pattern container for the whole batch
    mat_pattern_batch = torch.zeros((batch_size, size[0], size[1]), dtype=torch.float32, device=device)

    # We process each region in a fully vectorized manner
    for k in range(M):
        # Get the corresponding row values for the batch
        s_row = sigmoid(arr[:, k, 1:1 + N])  # Shape: [batch_size, N]
        t_row = orthogonalize(s_row)  # Shape: [batch_size, N]
        w_row = 2 * mfs + (size[1] - N * 2 * mfs) * t_row / torch.sum(t_row, dim=1, keepdim=True)
        x_cumsum = torch.cumsum(w_row, dim=1)

        y_bottom = torch.cat([torch.zeros(batch_size, 1, device=device), y_cumsum[:, :-1]], dim=1)
        y_bottom = y_bottom[:, k].unsqueeze(-1).unsqueeze(-1)
        y_top = y_cumsum[:, k].unsqueeze(-1).unsqueeze(-1)

        for i in range(N):
            # Calculate the x_left, x_right, y_bottom, and y_top in one shot for all batches
            x_left = torch.cat([torch.zeros(batch_size, 1, device=device), x_cumsum[:, :-1]], dim=1)
            x_left = x_left[:, i].unsqueeze(-1).unsqueeze(-1)
            x_right = x_cumsum[:, i].unsqueeze(-1).unsqueeze(-1)

            # Now we compute all the regions in parallel for all batches and all regions
            xm = (x_left + x_right) / 2  # Center x positions
            ym = (y_bottom + y_top) / 2  # Center y positions

            va = arr[:, k, 1 * N + 1 + i]
            vb = arr[:, k, 2 * N + 1 + i]
            v_theta = arr[:, k, 3 * N + 1 + i]
            v_delta = arr[:, k, 4 * N + 1 + i]

            # Vectorized computation of parameters (no need for loops)
            theta = torch.pi * sigmoid(v_theta) * 0  # Assuming no rotation, can be modified
            a_max = (w_row[:, i] - mfs) / 2
            b_max = (w_col[:, k] - mfs) / 2
            a = (a_max - mfs / 2) * sigmoid(va) + mfs / 2
            b = (b_max - mfs / 2) * sigmoid(vb) + mfs / 2
            delta = (b_max - b) * (sigmoid(v_delta) - 0.5)
            a = a.unsqueeze(-1).unsqueeze(-1)
            b = b.unsqueeze(-1).unsqueeze(-1)
            theta = theta.unsqueeze(-1).unsqueeze(-1)
            delta = delta.unsqueeze(-1).unsqueeze(-1)

            # Apply the region calculation (compute eps for all batches)
            region = torch.where(
                iteration < 80,
                eps_calcu(xm, ym, a, b, theta, delta, T1, grid_x, grid_y),
                eps_calcu_bi(xm, ym, a, b, theta, delta, T1, grid_x, grid_y, threshold=0.5)
            )

            # Create the mask and accumulate the result for each batch
            mask = ((grid_x >= x_left) & (grid_x < x_right) &
                    (grid_y >= y_bottom) & (grid_y < y_top)).float()

            # Vectorized addition of all regions
            mat_pattern_batch += region * mask

    return mat_pattern_batch

def make_super_pixel_reparam(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a simple cylinder-shaped meta atom
    """
    assert len(sim_shape) == 3
    substrate_eps = kwargs['substrate_eps']
    meta_atom_eps = kwargs['meta_atom_eps']
    top_medium_eps = kwargs['top_medium_eps']
    source_polarization = kwargs['source_polarization']

    # Define the super pixel sample:
    pattern_size = kwargs['pattern_size']
    M = kwargs['M']
    N = kwargs['N']
    
    substrate_thickness_pixel = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    meta_atom_height_pixel = get_pixels(kwargs, 'meta_atom_height_nm', dL)
    source_below_meta_pixel = get_pixels(kwargs, 'source_below_meta_nm', dL)
    minimal_feature_size_pixel = get_pixels(kwargs, 'minimal_feature_size_nm', dL)
    assert substrate_thickness_pixel > source_below_meta_pixel

    batch_size = 1
    # Example input with shape [batch_size, M, 1 + 5 * N]
    param = torch.rand((batch_size, M, 1 + 5 * N), dtype=torch.float32, requires_grad=True)
    # Generate patterns for the batch
    pattern_batch = reparam_pattern(param, pattern_size, minimal_feature_size_pixel, M, N, iteration=99)
    pattern_reparam = pattern_batch[0]  # Select the first pattern for use
    pad_x = (sim_shape[0] - pattern_reparam.shape[0]) // 2
    pad_y = (sim_shape[1] - pattern_reparam.shape[1]) // 2
    pattern_reparam = F.pad(pattern_reparam, (pad_x, pad_x, pad_y, pad_y), mode="constant", value=0)  # pad to the sim_shape

    eps = top_medium_eps * torch.ones(sim_shape)
    eps[:,:,:substrate_thickness_pixel] = substrate_eps

    ### make the freeform super pixel cross-section in x,y:
    cross = top_medium_eps + (meta_atom_eps - top_medium_eps) * pattern_reparam
    eps[:,:,substrate_thickness_pixel:substrate_thickness_pixel + meta_atom_height_pixel] = cross[...,None]

    src = torch.zeros((*sim_shape,6)) # source shape: (sx, sy, sz, 6), last dimension is src_x_real, src_x_imag, src_y_real, src_y_imag, src_z_real, src_z_imag
    if source_polarization == 'x':
        src[pad_x:-pad_x, pad_y:-pad_y, substrate_thickness_pixel - source_below_meta_pixel, 0] = 1
    elif source_polarization == 'y':
        src[pad_x:-pad_x, pad_y:-pad_y, substrate_thickness_pixel - source_below_meta_pixel, 2] = 1
    elif source_polarization == 'z':
        src[pad_x:-pad_x, pad_y:-pad_y, substrate_thickness_pixel - source_below_meta_pixel, 4] = 1

    return eps[None], src[None] # add a batch dimension



