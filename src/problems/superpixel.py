import functools
import jax.tree_util
import jax.numpy as jnp
import numpy as np
import torch
from typing import Sequence, Union, Callable, Optional, Tuple, Any, Dict
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

import concurrent.futures

from src.problems.base_problem import BaseProblem
from src.problems.base_challenge import BaseChallenge

from src.utils.physics import residue_E
from src.utils.utils import resolve, printc, c2r, r2c
from src.utils.stratton_chu_jax import strattonChu3D_full_sphere_GPU, E_to_H
from src.utils.PML_utils import apply_scpml

import gin

_DENSITY_LABEL = 'density'

@gin.configurable
class SuperpixelSpec:
    """
    Attributes:
    source_substrate_spacing: the spacing between the source and the substrate
    source_pml_spacing: the spacing between the source and the PML
    source_pol: the polarization of the source
    """
    def __init__(
        self,
        source_substrate_spacing,
        source_pml_spacing,
        source_angles,
        source_pol, # ('x', 'y', 'lcp', 'rcp')
        global_frame_coordinate,
    ):
        self.source_substrate_spacing = source_substrate_spacing
        self.source_pml_spacing = source_pml_spacing
        self.source_angles = source_angles
        self.source_pol = source_pol
        self.global_frame_coordinate = global_frame_coordinate
        self.ln_R = -10

    def __post_init__(self):
        assert self.source_substrate_spacing > 0, "source_substrate_spacing must be greater than 0"
        assert self.source_pml_spacing > 0, "source_pml_spacing must be greater than 0"

@gin.configurable
class SuperpixelProblem(BaseProblem):
    def __init__(self, *args, spec: SuperpixelSpec, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec

    def init(self, shape_only=False):
        self._waveguides = []
        self.make_sources()
    
    def make_sources(self):
        source_substrate_spacing = resolve(self.spec.source_substrate_spacing, self.dL)
        source_pml_spacing = resolve(self.spec.source_pml_spacing, self.dL)
        source_angles = self.spec.source_angles
        source_pol = self.spec.source_pol
        global_frame_coordinate = self.spec.global_frame_coordinate

        self.source_xs = (self.pmls[0] + source_pml_spacing, self.grid_shape[0]-self.pmls[1]-source_pml_spacing + 1) 
        self.source_ys = (self.pmls[2] + source_pml_spacing, self.grid_shape[1]-self.pmls[3]-source_pml_spacing + 1) 
        self.source_zs = (self.pmls[4] + self.surrounding_spaces[4] - source_substrate_spacing - 1, self.pmls[4] + self.surrounding_spaces[4] - source_substrate_spacing+1) 
        assert self.source_zs[1] - self.source_zs[0] == 2, "source in z direction should be 2 pixel thick"

        self.sources = {}
        for wl in self.wavelengths:
            # kz = 2 * jnp.pi * self.eps_background**.5 / wl
            sx = self.source_xs[1]-self.source_xs[0]
            sy = self.source_ys[1]-self.source_ys[0]
            sz = self.source_zs[1]-self.source_zs[0]
            source = torch.zeros((sx, sy, sz, 3), dtype=torch.complex128)

            theta, phi = source_angles
            kx = 2*np.pi * self.eps_substrate**.5 / wl * np.sin(theta) * np.cos(phi)
            ky = 2*np.pi * self.eps_substrate**.5 / wl * np.sin(theta) * np.sin(phi)
            kz = 2*np.pi * self.eps_substrate**.5 / wl * np.cos(theta)

            x = torch.linspace(self.source_xs[0]*self.dL, (self.source_xs[1]-1)*self.dL, sx) + global_frame_coordinate[0]
            y = torch.linspace(self.source_ys[0]*self.dL, (self.source_ys[1]-1)*self.dL, sy) + global_frame_coordinate[1]
            z = torch.linspace(self.source_zs[0]*self.dL, (self.source_zs[1]-1)*self.dL, sz) + global_frame_coordinate[2]
            
            x, y = torch.meshgrid(x,y,indexing='ij')
            map1 = torch.exp(-1j*(kx*x+ky*y+kz*z[1]))
            map2 = -torch.exp(-1j*(kx*x+ky*y+kz*z[0]))
                
            if source_pol == 'x':
                source[:,:,1,0] = map1
                source[:,:,0,0] = map2
            elif source_pol == 'y':
                source[:,:,1,1] = map1
                source[:,:,0,1] = map2
            elif source_pol == 'z':
                source[:,:,1,2] = map1
                source[:,:,0,2] = map2
            elif source_pol == 'lcp':
                source[:,:,1,0] = map1
                source[:,:,1,1] = torch.exp(1j*np.pi/2) * map1
                source[:,:,0,0] = map2
                source[:,:,0,1] = torch.exp(1j*np.pi/2) * map2
            elif source_pol == 'rcp':
                source[:,:,1,0] = map1
                source[:,:,1,1] = torch.exp(-1j*np.pi/2) * map1
                source[:,:,0,0] = map2
                source[:,:,0,1] = torch.exp(-1j*np.pi/2) * map2
            self.sources[wl] = source

    def simulate(self, design_variable):
        fields = [None] * len(self.wavelengths)
        epsilon_r_bg = self.epsilon_r_bg()

        def _simulate(wavelength):
            epsilon_r = self.epsilon_r(design_variable)
            source = np.zeros(epsilon_r.shape + (3,), dtype=np.complex128)
            source[self.source_xs[0]:self.source_xs[1], self.source_ys[0]:self.source_ys[1], self.source_zs[0]:self.source_zs[1]] = self.sources[wavelength].numpy()
            E = self.compute_FDFD(wavelength, epsilon_r, source, 'forward')
            return wavelength, E

        tasks = []
        for wavelength in self.wavelengths:
            tasks.append(wavelength)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            simulate_results = list(executor.map(_simulate, tasks))

        for wavelength, E in simulate_results:
            wl_idx = self.wavelengths.index(wavelength)
            fields[wl_idx] = E

        return fields

    def simulate_adjoint(self, design_variable, forward_output, grad_outputs):
        assert self._backend == 'NN'
        epsilon_r = self.epsilon_r(design_variable)

        def _adjoint_simulate(wavelength, forward_E, grad_E):
            source_torch = torch.conj(grad_E).to(torch.complex64).resolve_conj()  # adjoint source

            adjoint_E = self.compute_FDFD(wavelength, epsilon_r, source_torch, 'adjoint')

            design_variable_torch = design_variable.clone().detach().requires_grad_(True)
            epsilon_for_residual = self.make_torch_epsilon_r(design_variable_torch)[None]
            forward_E = c2r(forward_E[None].detach())

            # forward_source = self._ports[self.excite_port_idx].source(wavelength)
            forward_source = torch.zeros(epsilon_r.shape + (3,), dtype=torch.complex128)
            forward_source[self.source_xs[0]:self.source_xs[1], self.source_ys[0]:self.source_ys[1], self.source_zs[0]:self.source_zs[1]] = self.sources[wavelength]
            forward_source = c2r(forward_source[None].detach())

            residual = self.residual_fn(
                forward_E,
                epsilon_for_residual,
                forward_source,
                self.pmls,
                self.dL,
                wavelength,
                # bloch_vector=None,
                # batched_compute=False,
                # input_yee=False,
                # Aop=False,
                # ln_R=-10,
                # scale_PML=False,
            )

            input_grad = torch.autograd.grad(r2c(residual)[0], design_variable_torch, grad_outputs=torch.conj(adjoint_E))[0]

            return input_grad

        input_grads = []
        forward_Es = forward_output # shape (num_wavelengths, height, width, depth)
        grad_output_E = grad_outputs # shape (num_wavelengths, )
        # map jobs to all available GPUs
        def worker(wavelength):
            wavelength_idx = self.wavelengths.index(wavelength)

            return _adjoint_simulate(
                wavelength,
                forward_Es[wavelength_idx],
                grad_output_E[wavelength_idx],
            )

        tasks = []
        for wavelength in self.wavelengths:
            tasks.append(wavelength)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            input_grads = list(executor.map(worker, tasks))

        return (sum(input_grads)/len(self.wavelengths), ) # tuple of gradients for each input variable
    

@gin.configurable
class SuperpixelChallenge(BaseChallenge):
    def __init__(self, *args, target_angles, target_angle_ranges, SC_pml_space, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_angles = target_angles
        self.target_angle_ranges = target_angle_ranges  
        self.SC_pml_space = resolve(SC_pml_space, self.problem.dL)

        # compute the box size for the SC transformation
        p = self.problem
        self.Rx = (p.grid_shape[0]-p.pmls[0] - p.pmls[1] - 2*self.SC_pml_space)//2
        self.Ry = (p.grid_shape[1]-p.pmls[2] - p.pmls[3] - 2*self.SC_pml_space)//2
        self.Rz = (p.grid_shape[2]-p.pmls[4] - p.pmls[5] - 2*self.SC_pml_space)//2
        self.dxes = ([np.array([p.dL]*p.grid_shape[0]), np.array([p.dL]*p.grid_shape[1]), np.array([p.dL]*p.grid_shape[2])], [np.array([p.dL]*p.grid_shape[0]), np.array([p.dL]*p.grid_shape[1]), np.array([p.dL]*p.grid_shape[2])])
            

    def response(self, params):
        density = params[_DENSITY_LABEL].density
        return self._jax_sim_fn(density)

    def loss(self, response, plot_farfield=False):
        loss = 0.
        aux = {}
        for wavelength in self._wavelengths:
            wl_idx = self._wavelengths.index(wavelength)
            omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
            E = response[wl_idx]
            Ex = E[:,:,:,0]
            Ey = E[:,:,:,1]
            Ez = E[:,:,:,2]

            dxes = apply_scpml(self.dxes, self.problem.pmls, omega, ln_R=self.problem.spec.ln_R)
            dxes = [[jnp.array(i) for i in dxes[0]], [jnp.array(i) for i in dxes[1]]]

            # near to farfield transformation based on stratton-chu:
            Hx, Hy, Hz = E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None)
            thetas, phis, u0, far_E, far_H = strattonChu3D_full_sphere_GPU(
                dl=self.problem.dL,
                xc=Ex.shape[0]//2,
                yc=Ex.shape[1]//2,
                zc=Ex.shape[2]//2,
                Rx=self.Rx,
                Ry=self.Ry,
                Rz=self.Rz,
                lambda_val=wavelength,
                eps_background=self.problem.eps_background,
                Ex_OBJ=Ex[None],
                Ey_OBJ=Ey[None],
                Ez_OBJ=Ez[None],
                Hx_OBJ=Hx[None],
                Hy_OBJ=Hy[None],
                Hz_OBJ=Hz[None]
            )
            if plot_farfield:
                plot_poynting_radial_scatter(u0, far_E, far_H, fname=os.path.join(config['output_path'], 'poynting_radial_scatter.png'),
                                plot_batch_idx=0, normalize=True, point_size=8)
            
            t_theta, t_phi = self.target_angles
            d_theta, d_phi = self.target_angle_ranges

            assert far_E.shape == 4 and far_E.shape[0] == 1, "for now, far_E should be a 3D tensor with batch dimension 1"
            E3 = far_E[0].permute(1, 2, 0) # (Nt,Np,3)
            H3 = far_H[0].permute(1, 2, 0) # (Nt,Np,3)
            U3 = u0[0].permute(1, 2, 0)    # (Nt,Np,3)
            # S = 0.5 * Re(E × H*)
            S3 = 0.5 * jnp.real(jnp.cross(E3, jnp.conj(H3), axis=-1))  # (Nt,Np,3), real

            # radial component S_r = S · r̂
            Sr = jnp.sum(S3 * U3, axis=-1)      # (Nt,Np), real
            eff = jnp.abs(Sr)          

            target_region = eff[t_theta - int(d_theta/2):t_theta + int(d_theta/2), t_phi - int(d_phi/2):t_phi + int(d_phi/2)]
            
            loss += - jnp.sum(jnp.abs(target_region) ** 2) / jnp.sum(jnp.abs(eff) ** 2)
            aux[wavelength] = (u0, Sr)

        return loss, aux