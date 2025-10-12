import torch
import numpy as np
from src.utils import get_pixels

import src.simulator
from src.PML_utils import make_dxes_numpy
from src.physics import eps_to_yee
from spins.fdfd_tools.waveguide_mode import solve_waveguide_mode, compute_source
from spins.gridlock import direction
from spins.fdfd_tools.vectorization import vec
from spins.fdfd_tools import operators
import scipy.sparse as sparse

def get_waveguide_source(wavelength, dL, pml_layers, eps, src_direction, src_slice, mode_num=0, power=1.0, ln_R=-16):
    # direction: e.g. [0,0,1], [0,-1,0]
    # src_slice: e.g. [(start_x, start_y, start_z), (end_x, end_y, end_z)] (inclusive)
    omega=2 * np.pi / wavelength
    dxes = make_dxes_numpy(wavelength, dL, eps.shape, pml_layers, ln_R)
    axis = direction.Direction(direction.axisvec2axis(src_direction))
    slices = tuple([slice(i, f+1) for i, f in zip(*src_slice)])

    eps_yee = eps_to_yee(torch.from_numpy(eps[None]))[0].permute(3,0,1,2).numpy()
    mu = np.ones_like(eps_yee)
    
    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis.value,
        'slices': slices,
        'polarity': direction.axisvec2polarity(src_direction),
        'mu': mu
    }

    wgmode_result = solve_waveguide_mode(
        mode_number = mode_num,
        epsilon = eps_yee,
        **sim_params)
    J = compute_source(**wgmode_result, **sim_params)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)
    return J

def make_waveguide_bend(sim_shape, wl, dL, pmls, kwargs):
    """
    assumes that the waveguide bend is centered in x,y,z, source is parallel to xz plane, propagating in y direction
    """
    # shape, pmls, eps_sub=2.25, eps_meta_max=8.0, meta_height_pixels = 10, width = 10, radius=10):
    eps_sub = kwargs['substrate_eps']
    eps_wg = kwargs['waveguide_eps']
    eps_top = kwargs['top_medium_eps']
    ln_R = kwargs['ln_R']

    wgh = get_pixels(kwargs, 'waveguide_height_nm', dL)
    wgw = get_pixels(kwargs, 'waveguide_width_nm', dL)
    radius = get_pixels(kwargs, 'radius_nm', dL)
    sdff = get_pixels(kwargs, 'source_distance_from_face_nm', dL)
    st = round(sim_shape[2]/2-wgh/2) # substrate thickness in pixels
    mp = max(wgh, wgw) * 3 # mode padding pixels to compute the waveguide mode profile
    
    eps = eps_top * torch.ones(sim_shape, dtype=torch.float32)
    eps[:,:,:st] = eps_sub

    # make the straint portions
    eps[round(sim_shape[0]/2-wgw/2):round(sim_shape[0]/2+wgw/2),:round(sim_shape[1]/2+wgw/2) - radius, st:st + wgh] = eps_wg
    eps[round(sim_shape[0]/2-wgw/2) + radius:,round(sim_shape[1]/2-wgw/2):round(sim_shape[1]/2+wgw/2),st:st + wgh] = eps_wg

    # make the curved portion
    xy_map = eps_top * torch.ones(radius,radius, dtype=torch.float32) 
    for x in range(0, radius):
        for y in range(0, radius):
            r_square = (x-radius)**2 + y**2 
            if r_square <= radius**2 and r_square>=(radius - wgw)**2:
                xy_map[x,y] = eps_wg
    
    for z in range(st, st + wgh):
        eps[round(sim_shape[0]/2-wgw/2) : round(sim_shape[0]/2-wgw/2)+radius, round(sim_shape[1]/2+wgw/2) - radius:round(sim_shape[1]/2+wgw/2), z] = xy_map

    ############### (2) waveguide mode source #############
    source_dir = np.array((0,1,0))
    x_start = max(round(sim_shape[0]/2-wgw/2) - mp, 0)
    x_end = min(round(sim_shape[0]/2+wgw/2) + mp, sim_shape[0])
    ypos = sdff
    z_start = max(round(st+wgh/2) - mp, 0)
    z_end = min(round(st+wgh/2) + mp, sim_shape[2])

    print("computing waveguide mode source...")
    source_slice = tuple([(x_start, ypos, z_start), (x_end, ypos, z_end)])
    source = get_waveguide_source(wl, dL, pmls, eps.numpy(), source_dir, source_slice, ln_R=ln_R)
    source = torch.view_as_real(torch.from_numpy(source).permute(1,2,3,0)).reshape(*sim_shape,6).to(torch.float32)
    print("waveguide mode source computed")

    return eps[None], source[None]