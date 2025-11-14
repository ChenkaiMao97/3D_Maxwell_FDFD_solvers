import sys
from src.solvers.NN_solver import NN_solve
from src.solvers.spins_solver import spins_solve
from src.utils.utils import *
from src.utils.plot_field3D import plot_3slices, plot_poynting_radial_scatter

from src.utils.physics import E_to_H_batch as E_to_H
from src.utils.stratton_chu import strattonChu3D_full_sphere_GPU

# from src.utils.stratton_chu_jax import strattonChu3D_GPU, strattonChu3D_full_sphere_GPU, E_to_H

import matplotlib.pyplot as plt
import yaml
import time

import jax.numpy as jnp
import numpy as np
import torch

from src.problems.meta_atom import make_meta_atom_cylinder
from src.problems.waveguide_bend import make_waveguide_bend
from src.problems.super_pixel_freeform import make_super_pixel_freeform
from src.problems.super_pixel_reparam import make_super_pixel_reparam
from src.problems.super_pixel_binary import make_super_pixel_binary
from src.utils.PML_utils import apply_scpml

C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6

def get_problem_constructor(sim_type):
    problems = {
        'meta_atom': make_meta_atom_cylinder,
        'waveguide_bend': make_waveguide_bend,
        'super_pixel_freeform': make_super_pixel_freeform,
        'super_pixel_reparam': make_super_pixel_reparam,
        'super_pixel_binary': make_super_pixel_binary
    }
    return problems[sim_type]

def main(config):
    problem_constructor = get_problem_constructor(config['sim_type'])
    eps, src = problem_constructor(config["sim_shape"], config['wavelength'], config['dL'], config['pmls'], config['kwargs'])
    print("eps shape:", eps.shape)

    solution, residual_history, final_residual = NN_solve(config, eps, src)
    print(f"final residual absolute mean: {torch.mean(torch.abs(final_residual))}")

    ln_R = -10 if 'ln_R' not in config['kwargs'] else config['kwargs']['ln_R']

    if config['spins_verification']: # PML parameter, don't change
        spins_solution, spins_residual = spins_solve(config, eps, src, ln_R=ln_R)
        rel_diff, E_diff = scaled_MAE(c2r(solution).cpu(), spins_solution)
        print("relative error between E_spin and E_model", rel_diff)
    
    # plot the results:
    intensity = torch.sum(torch.abs(solution[0])**2, dim=-1).detach().cpu().numpy()
    # plot_3slices(eps[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'eps.png'), my_cmap=plt.cm.binary, cm_zero_center=False)
    # plot_3slices(solution[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_xr.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(solution[0,:,:,:,0].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_xi.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(solution[0,:,:,:,1].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_yr.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(solution[0,:,:,:,1].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_yi.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(solution[0,:,:,:,2].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'solution_zr.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(solution[0,:,:,:,2].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'solution_zi.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(intensity, fname=os.path.join(config['output_path'], 'intensity.png'), my_cmap=plt.cm.seismic)
    
    # plot_3slices(np.abs(final_residual[0,:,:,:,0].detach().cpu().numpy()), fname=os.path.join(config['output_path'], 'residual_x.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(np.abs(final_residual[0,:,:,:,1].detach().cpu().numpy()), fname=os.path.join(config['output_path'], 'residual_y.png'), my_cmap=plt.cm.seismic)
    # plot_3slices(np.abs(final_residual[0,:,:,:,2].detach().cpu().numpy()), fname=os.path.join(config['output_path'], 'residual_z.png'), my_cmap=plt.cm.seismic)
    # save the first batch E-field solution
    if config['save_fields']:
        solution_cpu = solution[0].detach().cpu()
        torch.save(solution_cpu, os.path.join(config['output_path'], "E_sol.pt"))
        torch.save(eps.detach().cpu(), os.path.join(config['output_path'], "eps_sol.pt"))
        print("E-field solution saved.")

    if config['spins_verification']:
        intensity = torch.sum(torch.abs(spins_solution[0])**2, dim=-1).detach().cpu().numpy()
        plot_3slices(intensity, fname=os.path.join(config['output_path'], 'spins_intensity.png'), my_cmap=plt.cm.seismic)
        plot_3slices(spins_solution[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'spins_solution_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices(spins_residual[0,:,:,:,0].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'spins_residual_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices((spins_solution.detach().cpu().numpy() - c2r(solution).detach().cpu().numpy())[0,:,:,:,0], fname=os.path.join(config['output_path'], 'difference_xr.png'), my_cmap=plt.cm.seismic)
    
    solution = solution.to(torch.complex64)
    eps = eps.to(solution.device)
    pml_thickness = config['pmls'][0]
    grid_size = config['dL']
    lambda_val = config['wavelength']
    omega = 2 * torch.pi / lambda_val

    dxes = ([np.array([config['dL']]*eps.shape[1]), np.array([config['dL']]*eps.shape[2]), np.array([config['dL']]*eps.shape[3])], [np.array([config['dL']]*eps.shape[1]), np.array([config['dL']]*eps.shape[2]), np.array([config['dL']]*eps.shape[3])])    
    dxes = apply_scpml(dxes, config['pmls'], omega, ln_R=ln_R)
    dxes = [[torch.tensor(i).to(solution.device).to(torch.complex64) for i in dxes[0]], [torch.tensor(i).to(solution.device).to(torch.complex64) for i in dxes[1]]]
    # dxes = [[jnp.array(i) for i in dxes[0]], [jnp.array(i) for i in dxes[1]]]

    if config['plot_farfield']:
        Ex = solution[:,:,:,:,0]
        Ey = solution[:,:,:,:,1]
        Ez = solution[:,:,:,:,2]
        # Ex = jnp.array(solution[:,:,:,:,0].detach().cpu().numpy())
        # Ey = jnp.array(solution[:,:,:,:,1].detach().cpu().numpy())
        # Ez = jnp.array(solution[:,:,:,:,2].detach().cpu().numpy())
        Hx, Hy, Hz = E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None)

        plot_3slices(Hx[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'H_xr.png'), my_cmap=plt.cm.seismic)
        plot_3slices(Hx[0,:,:,:].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'H_xi.png'), my_cmap=plt.cm.seismic)
        plot_3slices(Hy[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'H_yr.png'), my_cmap=plt.cm.seismic)
        plot_3slices(Hy[0,:,:,:].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'H_yi.png'), my_cmap=plt.cm.seismic)
        plot_3slices(Hz[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'H_zr.png'), my_cmap=plt.cm.seismic)
        plot_3slices(Hz[0,:,:,:].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'H_zi.png'), my_cmap=plt.cm.seismic)
    

        print("Plotting far-field patterns using Stratton-Chu...")
        # u0, far_E, far_H = strattonChu3D_GPU(
        #     dl=grid_size,
        #     xc=Ex.shape[1]//2,
        #     yc=Ex.shape[2]//2,
        #     zc=Ex.shape[3]//2,
        #     Rx=Ex.shape[1]//2 - 1 - pml_thickness,
        #     Ry=Ex.shape[2]//2 - 1 - pml_thickness,
        #     Rz=Ex.shape[3]//2 - 1 - pml_thickness,
        #     lambda_val=lambda_val,
        #     Ex_OBJ=Ex,
        #     Ey_OBJ=Ey,
        #     Ez_OBJ=Ez,
        #     Hx_OBJ=Hx,
        #     Hy_OBJ=Hy,
        #     Hz_OBJ=Hz,
        #     device=solution.device,
        #     t_theta=45, t_phi=45,
        #     bs=1
        #     )

        thetas, phis, u0, far_E, far_H = strattonChu3D_full_sphere_GPU(
            dl=grid_size,
            xc=Ex.shape[1]//2,
            yc=Ex.shape[2]//2,
            zc=Ex.shape[3]//2,
            # Rx=Ex.shape[1]//2 - 1 - pml_thickness,
            # Ry=Ex.shape[2]//2 - 1 - pml_thickness,
            # Rz=Ex.shape[3]//2 - 1 - pml_thickness,
            Rx=30,
            Ry=30,
            Rz=30,
            lambda_val=lambda_val,
            eps_background = config['kwargs']['top_medium_eps'],
            Ex_OBJ=Ex,
            Ey_OBJ=Ey,
            Ez_OBJ=Ez,
            Hx_OBJ=Hx,
            Hy_OBJ=Hy,
            Hz_OBJ=Hz,
            device=solution.device,
            bs=1
        )
        print("Full sphere case shapes", thetas.shape, phis.shape, u0.shape, far_E.shape, far_H.shape)
        # plot_poynting_radial_scatter(torch.from_numpy(np.array(u0)), torch.from_numpy(np.array(far_E)), torch.from_numpy(np.array(far_H)), fname=os.path.join(config['output_path'], 'poynting_radial_scatter.png'),
        #                      plot_batch_idx=0, normalize=True, point_size=8)
        plot_poynting_radial_scatter(u0, far_E, far_H, fname=os.path.join(config['output_path'], 'poynting_radial_scatter.png'),
                             plot_batch_idx=0, normalize=False, point_size=8)

    # if config['physics_verification']:
    #     Ex = solution[:,:,:,:,0]
    #     Ey = solution[:,:,:,:,1]
    #     Ez = solution[:,:,:,:,2]

    #     Hx, Hy, Hz = E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None)
    #     Ex_1, Ey_1, Ez_1 = H_to_E(Hx, Hy, Hz, dxes, omega, eps, bloch_vector=None)
    #     delta = pml_thickness + 1  # to avoid boundary effects from PML
    #     Ex = Ex[:, delta:-delta,delta:-delta,delta:-delta]
    #     Ey = Ey[:, delta:-delta,delta:-delta,delta:-delta]
    #     Ez = Ez[:, delta:-delta,delta:-delta,delta:-delta]
    #     Ex_1 = Ex_1[:, delta:-delta,delta:-delta,delta:-delta]
    #     Ey_1 = Ey_1[:, delta:-delta,delta:-delta,delta:-delta]
    #     Ez_1 = Ez_1[:, delta:-delta,delta:-delta,delta:-delta]
    #     print("Ex residue", torch.mean(torch.abs(Ex - Ex_1))/torch.mean(torch.abs(Ex)))
    #     print("Ey residue", torch.mean(torch.abs(Ey - Ey_1))/torch.mean(torch.abs(Ey)))
    #     print("Ez residue", torch.mean(torch.abs(Ez - Ez_1))/torch.mean(torch.abs(Ez)))
        
if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"using GPU {gpu_id}")
    main(config)