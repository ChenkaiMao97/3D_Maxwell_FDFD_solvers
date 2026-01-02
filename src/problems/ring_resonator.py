import torch
import numpy as np
from src.utils.utils import get_pixels

from src.utils.PML_utils import make_dxes_numpy
from src.utils.physics import eps_to_yee
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
    # mu = np.ones_like(eps_yee)
    
    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis.value,
        'slices': slices,
        'polarity': direction.axisvec2polarity(src_direction)
    }
    # print("eps_yee.shape: ", eps_yee.shape)
    # print("slices: ", slices)
    # print("mu.shape: ", mu.shape)
    # print("dxes[0][0].shape: ", dxes[0][0].shape)
    # print("dxes[0][1].shape: ", dxes[0][1].shape)
    # print("dxes[0][2].shape: ", dxes[0][2].shape)
    # print("omega: ", omega)
    # print("axis: ", axis.value)
    # print("polarity: ", direction.axisvec2polarity(src_direction))

    wgmode_result = solve_waveguide_mode(
        mode_number = mode_num,
        epsilon = eps_yee,
        **sim_params)
    J = compute_source(**wgmode_result, **sim_params)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)
    return J

def circle_supersample(h, w, cx, cy, r, ss=8, dtype=np.float32):
    """
    Per-pixel coverage for a disk centered at (cx, cy) with radius r.
    Pixel centers are at integer coords (i, j). Returns HxW in [0,1].
    ss: supersamples per axis (e.g., 4, 8). Higher = smoother/more accurate.
    """
    # pixel grid
    x = np.arange(h)[:, None]
    y = np.arange(w)[None, :]

    # stratified subpixel offsets in [0,1)
    ofs = (np.arange(ss) + 0.5) / ss
    ox, oy = np.meshgrid(ofs, ofs, indexing="ij")  # (ss, ss)

    # sample coordinates (broadcast to HxW)
    X = x[..., None, None] + ox  # (H, W, ss, ss)
    Y = y[..., None, None] + oy

    # distance to circle center
    d2 = (X - cx)**2 + (Y - cy)**2
    inside = d2 <= r*r

    # average over subpixels → coverage in [0,1]
    cov = inside.mean(axis=(-1, -2)).astype(dtype)
    return torch.from_numpy(cov)


def make_ring_resonator(sim_shape, wl, dL, pmls, kwargs):
    """
    makes a simple cylinder-shaped meta atom
    """
    assert len(sim_shape) == 3
    substrate_eps = kwargs['substrate_eps']
    medium_eps = kwargs['medium_eps']
    top_medium_eps = kwargs['top_medium_eps']
    ln_R = kwargs['ln_R']
    
    substrate_thickness_pixel = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    wgw = get_pixels(kwargs, 'waveguide_width_nm', dL)
    gap_pixel = get_pixels(kwargs, 'gap_nm', dL)
    print("gap_pixel: ", gap_pixel)
    wgh = get_pixels(kwargs, 'waveguide_height_nm', dL)
    ring_radius_pixel = get_pixels(kwargs, 'ring_radius_nm', dL)
    mp = max(wgw, wgh)

    total_width = 2*wgw + 2*ring_radius_pixel + gap_pixel
    ring_center_x = (sim_shape[0]+total_width-wgw)//2-ring_radius_pixel
    wg_center_x = (sim_shape[0]-total_width+wgw)//2

    eps = top_medium_eps * torch.ones(sim_shape)
    eps[:,:,:substrate_thickness_pixel] = substrate_eps

    ### make the ring and waveguide in x,y:
    circle1 = circle_supersample(sim_shape[0], sim_shape[1], ring_center_x, sim_shape[1] // 2, ring_radius_pixel + wgw//2)
    circle2 = circle_supersample(sim_shape[0], sim_shape[1], ring_center_x, sim_shape[1] // 2, ring_radius_pixel + wgw//2 - wgw)
    ring = circle1 - circle2
    
    ring[wg_center_x-wgw//2:wg_center_x+wgw//2,:] = 1
    
    cross = top_medium_eps + (medium_eps - top_medium_eps) * ring
    
    eps[:,:,substrate_thickness_pixel:substrate_thickness_pixel + wgh] = cross[...,None]

    ############### (2) waveguide mode source #############
    source_dir = np.array((0,1,0))
    x_start = max(round(wg_center_x-wgw/2) - mp, 0)
    x_end = min(round(wg_center_x+wgw/2) + mp, sim_shape[0])
    ypos = round(sim_shape[1]/6)
    z_start = max(substrate_thickness_pixel - mp, 0)
    z_end = min(substrate_thickness_pixel+wgh + mp, sim_shape[2])

    print("computing waveguide mode source...")
    source_slice = tuple([(x_start, ypos, z_start), (x_end, ypos, z_end)])

    print("wl, dL, pmls: ", wl, dL, pmls)
    print("shapes: ", eps.shape, source_slice)
    source = get_waveguide_source(wl, dL, pmls, eps.numpy(), source_dir, source_slice, ln_R=ln_R)
    source = torch.view_as_real(torch.from_numpy(source).permute(1,2,3,0)).reshape(*sim_shape,6).to(torch.float32)
    print("waveguide mode source computed")

    return eps[None], source[None] # add a batch dimension