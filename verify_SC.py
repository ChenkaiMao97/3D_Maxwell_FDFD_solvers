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

# C_0 = 299792458.13099605
# EPSILON_0 = 8.85418782e-12
# MU_0 = 1.25663706e-6

def main(solution, eps, config):
    solution = solution.to(torch.complex64)
    eps = eps.to(solution.device)
    pml_thickness = config['pmls'][0]
    grid_size = config['dL'] # convert nm to meters
    lambda_val = config['wavelength'] # convert nm to meters
    omega = 2 * torch.pi / lambda_val
    ln_R = -10 if 'ln_R' not in config['kwargs'] else config['kwargs']['ln_R']

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

        # plot_3slices(Hx[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'H_xr.png'), my_cmap=plt.cm.seismic)
        # plot_3slices(Hx[0,:,:,:].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'H_xi.png'), my_cmap=plt.cm.seismic)
        # plot_3slices(Hy[0,:,:,:].detach().cpu().numpy().real, fname=os.path.join(config['output_path'], 'H_yr.png'), my_cmap=plt.cm.seismic)
        # plot_3slices(Hy[0,:,:,:].detach().cpu().numpy().imag, fname=os.path.join(config['output_path'], 'H_yi.png'), my_cmap=plt.cm.seismic)
    

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
            Rx=20,
            Ry=20,
            Rz=20,
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

        # sanity plot
        plt.figure(figsize=(18, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(far_E[0,0,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 2)
        plt.imshow(far_E[0,1,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 3)
        plt.imshow(far_E[0,2,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 4)
        plt.imshow(far_H[0,0,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 5)
        plt.imshow(far_H[0,1,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 6)
        plt.imshow(far_H[0,2,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.savefig(os.path.join(config['output_path'], 'far_field_patterns.png'))
        plt.close()

        # compute analytical farfield expression:
        def far_E_fn(theta, phi, r, k):
            r = torch.tensor(r)
            E_theta = torch.sin(theta) / r * torch.exp(1j * k * r)
            Ex = E_theta * torch.cos(theta) * torch.cos(phi)
            Ey = E_theta * torch.cos(theta) * torch.sin(phi)
            Ez = -E_theta * torch.sin(theta)
            return torch.stack([Ex, Ey, Ez], dim=0) # shape: (3,N_theta,N_phi)
        def far_H_fn(theta, phi, r, k):
            r = torch.tensor(r)
            H_theta = torch.sin(theta) / r * torch.exp(1j * k * r)
            Hx = H_theta * -torch.sin(phi)
            Hy = H_theta * torch.cos(phi)
            Hz = torch.zeros_like(Hx)
            return torch.stack([Hx, Hy, Hz], dim=0) # shape: (3,N_theta,N_phi)
        
        far_E_analytical = far_E_fn(thetas, phis, 1e2*lambda_val, 2 * torch.pi / (lambda_val/np.sqrt(config['kwargs']['top_medium_eps'])))
        far_H_analytical = far_H_fn(thetas, phis, 1e2*lambda_val, 2 * torch.pi / (lambda_val/np.sqrt(config['kwargs']['top_medium_eps'])))
        print("far_E_analytical shape:", far_E_analytical.shape, "far_H_analytical shape:", far_H_analytical.shape)
        plt.figure(figsize=(18, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(far_E_analytical[0,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 2)
        plt.imshow(far_E_analytical[1,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 3)
        plt.imshow(far_E_analytical[2,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 4)
        plt.imshow(far_H_analytical[0,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 5)
        plt.imshow(far_H_analytical[1,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.subplot(2, 3, 6)
        plt.imshow(far_H_analytical[2,:,:].abs(), cmap='seismic')
        plt.colorbar()
        plt.savefig(os.path.join(config['output_path'], 'far_field_patterns_analytical.png'))
        plt.close()
        

        plot_poynting_radial_scatter(u0, far_E, far_H, fname=os.path.join(config['output_path'], 'poynting_radial_scatter.png'),
                             plot_batch_idx=0, normalize=False, point_size=8)

    

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"using GPU {gpu_id}")

    solution = torch.load(os.path.join(config['output_path'], "E_sol.pt"))[None]
    eps = torch.load(os.path.join(config['output_path'], "eps_sol.pt"))
    print("solution shape:", solution.shape, "eps shape:", eps.shape)
    main(solution, eps, config)