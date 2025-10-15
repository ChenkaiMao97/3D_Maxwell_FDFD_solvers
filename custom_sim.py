import sys
from src.NN_solver import NN_solve
from src.spins_solver import spins_solve
from src.utils import *
from src.plot_field3D import plot_3slices
import matplotlib.pyplot as plt
import yaml
import time

from src.problems.meta_atom import make_meta_atom_cylinder
from src.problems.waveguide_bend import make_waveguide_bend

def get_problem_constructor(sim_type):
    problems = {
        'meta_atom': make_meta_atom_cylinder,
        'waveguide_bend': make_waveguide_bend
    }
    return problems[sim_type]

def main(config):
    problem_constructor = get_problem_constructor(config['sim_type'])
    eps, src = problem_constructor(config["sim_shape"], config['wavelength'], config['dL'], config['pmls'], config['kwargs'])

    solution, residual_history, final_residual = NN_solve(config, eps, src)
    print(f"final residual absolute mean: {torch.mean(torch.abs(final_residual))}")

    if config['spins_verification']:
        spins_solution, spins_residual = spins_solve(config, eps, src)
        rel_diff, E_diff = scaled_MAE(c2r(solution).cpu(), spins_solution)
        print("relative error between E_spin and E_model", rel_diff)
    
    # plot the results:
    intensity = torch.sum(torch.abs(solution[0])**2, dim=-1).detach().cpu().numpy()
    plot_3slices(eps[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'eps.png'), my_cmap=plt.cm.binary, cm_zero_center=False)
    print("solution.shape", solution.shape)
    plot_3slices(solution[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_xr.png'), my_cmap=plt.cm.seismic)
    plot_3slices(solution[0,:,:,:,0].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_xi.png'), my_cmap=plt.cm.seismic)
    plot_3slices(solution[0,:,:,:,1].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_yr.png'), my_cmap=plt.cm.seismic)
    plot_3slices(solution[0,:,:,:,1].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_yi.png'), my_cmap=plt.cm.seismic)
    plot_3slices(solution[0,:,:,:,2].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_zr.png'), my_cmap=plt.cm.seismic)
    plot_3slices(solution[0,:,:,:,2].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_zi.png'), my_cmap=plt.cm.seismic)
    plot_3slices(intensity, fname=os.path.join(config['output_path'], 'intensity.png'), my_cmap=plt.cm.seismic)
    plot_3slices(final_residual[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'residual_xr.png'), my_cmap=plt.cm.seismic)

    if config['spins_verification']:
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