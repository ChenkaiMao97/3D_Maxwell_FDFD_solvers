import sys
from src.solvers.NN_random_solver import NN_random_solve
from src.solvers.spins_solver import spins_solve
from src.utils.utils import *
from src.utils.plot_field3D import plot_3slices
from functools import partial
import matplotlib.pyplot as plt
import yaml
import time

def main(config):
    solution, residual_history, final_residual, eps, src, dL, wl, pmls = NN_random_solve(config)
    print(f"final residual absolute mean: {torch.mean(torch.abs(final_residual))}")

    if config['spins_verification']:
        spins_solution, spins_residual = spins_solve(config, eps[...,0].detach().cpu(), src.detach().cpu(), dL=float(dL[0].numpy()), wl=float(wl[0].numpy()), pmls=pmls)
        rel_diff, E_diff = scaled_MAE(c2r(solution).cpu(), spins_solution)
        print("relative error between E_spin and E_model", rel_diff)
    
    # plot the results:
    plot_fn_eps = partial(plot_3slices, ticks=False, colorbar=False, my_cmap=plt.cm.binary, cm_zero_center=False)
    plot_fn_fields = partial(plot_3slices,ticks=False, colorbar=False, my_cmap=plt.cm.seismic, cm_zero_center=True)

    intensity = torch.sum(torch.abs(solution[0])**2, dim=-1).detach().cpu().numpy()
    plot_fn_eps(eps[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'eps'))
    plot_fn_fields(src[0,...,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'src'))
    plot_fn_fields(solution[0,:,:,:,2].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_zr'))
    plot_fn_fields(solution[0,:,:,:,2].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_zi'))
    plot_fn_fields(final_residual[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'residual_xr'))

    if config['spins_verification']:
        plot_3slices(spins_solution[0,:,:,:,0].detach().cpu().numpy(), fname=os.path.join(config['output_path'], 'spins_solution_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices(spins_residual[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'spins_residual_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices((spins_solution.detach().cpu().numpy() - c2r(solution).detach().cpu().numpy())[0,:,:,:,0], fname=os.path.join(config['output_path'], 'difference_xr.png'), my_cmap=plt.cm.seismic)


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"using GPU {gpu_id}")
    main(config)