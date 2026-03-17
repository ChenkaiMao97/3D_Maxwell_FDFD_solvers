import sys
from src.solvers.NN_solver import NN_solve
from src.solvers.spins_solver import spins_solve
from src.utils.utils import *
from src.utils.plot_field3D import plot_3slices, plot_3slices_together, plot_3slices_plotly
import matplotlib.pyplot as plt
import yaml
import time
import numpy as np
import os

from functools import partial

from src.problems.meta_atom import make_meta_atom_cylinder
from src.problems.waveguide_bend import make_waveguide_bend
from src.problems.all_air import make_all_air
from src.problems.rings import make_rings
# from src.problems.superpixel import make_superpixel

def get_problem_constructor(sim_type):
    problems = {
        'meta_atom': make_meta_atom_cylinder,
        'waveguide_bend': make_waveguide_bend,
        'all_air': make_all_air,
        'rings': make_rings,
        # 'superpixel': make_superpixel
    }
    return problems[sim_type]

def theta_map_xy(sx, sy):
    """
    Generate azimuthal angle map (phi) for an (sx, sy) x-y grid.

    Returns:
      theta : (sx, sy) float32
              phi angle in radians, range (-pi, pi]
    """
    cx = (sx - 1) / 2.0
    cy = (sy - 1) / 2.0

    x = np.arange(sx, dtype=np.float32) - cx
    y = np.arange(sy, dtype=np.float32) - cy
    X, Y = np.meshgrid(x, y, indexing="ij")

    theta = np.arctan2(Y, X)  # phi

    return theta

def main(config):
    problem_constructor = get_problem_constructor(config['sim_type'])
    eps, src = problem_constructor(config["sim_shape"], config['wavelength'], config['dL'], config['pmls'], config['kwargs'])

    solution, residual_history, final_residual = NN_solve(config, eps, src)
    print(f"final residual absolute mean: {torch.mean(torch.abs(final_residual))}")

    if config['save_for_plotting']:
        np.save(os.path.join(config['output_path'], 'solution.npy'), solution.detach().cpu().numpy())
        np.save(os.path.join(config['output_path'], 'src.npy'), src.detach().cpu().numpy())
        np.save(os.path.join(config['output_path'], 'eps.npy'), eps.detach().cpu().numpy())

    # plot the results:
    # plot_fn_eps = partial(plot_3slices_plotly, colorscale="Greys")
    # plot_fn_fields = partial(plot_3slices_plotly, colorscale="RdBu")
    plot_fn_eps = partial(plot_3slices, ticks=False, colorbar=True, my_cmap=plt.cm.binary, cm_zero_center=False)
    plot_fn_fields = partial(plot_3slices,ticks=False, colorbar=True, my_cmap=plt.cm.seismic, cm_zero_center=True)

    intensity = torch.sum(torch.abs(solution[0])**2, dim=-1).detach().cpu().numpy()
    plot_fn_eps(eps[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'eps'))
    src_intensity = torch.sum(torch.abs(src[0]), dim=-1)

    theta_map = theta_map_xy(src_intensity.shape[0], src_intensity.shape[1])[:,:,None]
    Ex = solution[0,:,:,:,0].detach().cpu().numpy()
    Ey = solution[0,:,:,:,1].detach().cpu().numpy()
    Ez = solution[0,:,:,:,2].detach().cpu().numpy()
    Er = Ex * np.cos(theta_map) + Ey * np.sin(theta_map)
    Ephi = -Ex * np.sin(theta_map) + Ey * np.cos(theta_map)

    plot_fn_fields(Er.real, fname=os.path.join(config['output_path'], 'Er_r'))
    plot_fn_fields(Er.imag, fname=os.path.join(config['output_path'], 'Er_i'))
    plot_fn_fields(Ephi.real, fname=os.path.join(config['output_path'], 'Ephi_r'))
    plot_fn_fields(Ephi.imag, fname=os.path.join(config['output_path'], 'Ephi_i'))
    plot_fn_fields(Ez.real, fname=os.path.join(config['output_path'], 'Ez_r'))
    plot_fn_fields(Ez.imag, fname=os.path.join(config['output_path'], 'Ez_i'))
    plot_fn_fields(src_intensity.detach().cpu().numpy(), fname=os.path.join(config['output_path'], 'src'))
    plot_fn_fields(final_residual[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'residual_xr'))

    if config['spins_verification']:
        ln_R = -10 if 'ln_R' not in config['kwargs'] else config['kwargs']['ln_R'] # PML parameter, don't change
        spins_solution, spins_residual = spins_solve(config, eps, src, ln_R=ln_R)
        rel_diff, E_diff = scaled_MAE(c2r(solution).cpu(), spins_solution)
        print("relative error between E_spin and E_model", rel_diff)

        intensity = torch.sum(torch.abs(spins_solution[0])**2, dim=-1).detach().cpu().numpy()
        plot_3slices(intensity, fname=os.path.join(config['output_path'], 'spins_intensity.png'), my_cmap=plt.cm.seismic)
        plot_3slices(spins_solution[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'spins_solution_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices(spins_residual[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'spins_residual_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices((spins_solution.detach().cpu().numpy() - c2r(solution).detach().cpu().numpy())[0,:,:,:,0], fname=os.path.join(config['output_path'], 'difference_xr.png'), my_cmap=plt.cm.seismic)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"using GPU {gpu_id}")
    main(config)