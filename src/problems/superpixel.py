import os
import functools
import jax.tree_util
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
import math
import torch
from typing import Sequence, Union, Callable, Optional, Tuple, Any, Dict
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

import concurrent.futures

from src.problems.base_problem import BaseProblem
from src.problems.base_challenge import BaseChallenge

from src.utils.physics import residue_E
from src.utils.utils import resolve, printc, c2r, r2c
from src.utils.stratton_chu_jax import strattonChu3D_full_sphere_GPU, E_to_H, fibonacci_sphere
from src.utils.PML_utils import apply_scpml
from src.utils.plot_field3D import plot_3slices, plot_poynting_radial_scatter

import gin

_DENSITY_LABEL = 'density'

def debug_plot(field, name):
    assert field.ndim == 2, "field should be a 2D tensor"
    plt.figure(figsize=(5, 5))
    plt.imshow(field.detach().cpu().numpy().real, cmap='seismic')
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def find_indices_in_range(sorted_data, low, high):
    begin, end = None, None
    for idx, i in enumerate(sorted_data):
        if begin is None and i >= low:
            begin = idx
        if begin is not None and i > high:
            end = idx
            break
    assert begin is not None and end is not None, f"begin and end indices not found, check again for range: data: {sorted_data}, low: {low}, high: {high}"
    if begin == end:
        print("warning: target angle range is too small, incrementing end")
        end += 1
    return begin, end

def sph_to_cart(theta, phi):
    sin_th = np.sin(theta)
    return np.stack([
        sin_th * np.cos(phi),
        sin_th * np.sin(phi),
        np.cos(theta)
    ], axis=-1)

class SphereGridKD:
    def __init__(self, u0):
        assert u0.shape[1] == 3, "u0 should be a 3D tensor"
        self.u0 = u0
        self.tree = cKDTree(u0)

    def query_cone(self, theta_q, phi_q, angle_radius):
        x, y, z = sph_to_cart(
            np.array(theta_q),
            np.array(phi_q)
        )
        q = np.array([x, y, z])
        r = 2 * np.sin(angle_radius)
        idx = self.tree.query_ball_point(q, r)
        return np.array(idx)

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
        self.residual_fn = residue_E

    def init(self, shape_only=False):
        self._waveguides = []
        self.make_sources()
    
    def make_sources(self):
        source_substrate_spacing = resolve(self.spec.source_substrate_spacing, self.dL)
        source_pml_spacing = resolve(self.spec.source_pml_spacing, self.dL)
        source_angles = self.spec.source_angles
        source_pol = self.spec.source_pol
        global_frame_coordinate = self.spec.global_frame_coordinate

        self.source_xs = (self.pmls[0] + source_pml_spacing, self.grid_shape[0]-self.pmls[1]-source_pml_spacing-1) if self.pml_pad_mode \
                    else (source_pml_spacing, self.grid_shape[0]-source_pml_spacing-1)
        self.source_ys = (self.pmls[2] + source_pml_spacing, self.grid_shape[1]-self.pmls[3]-source_pml_spacing-1) if self.pml_pad_mode \
                    else (source_pml_spacing, self.grid_shape[1]-source_pml_spacing-1)
        self.source_zs = (self.pmls[4] + self.surrounding_spaces[4] - source_substrate_spacing - 1, self.pmls[4] + self.surrounding_spaces[4] - source_substrate_spacing + 1) if self.pml_pad_mode \
                    else (self.surrounding_spaces[4] - source_substrate_spacing - 1, self.surrounding_spaces[4] - source_substrate_spacing + 1)

        assert self.source_zs[1] - self.source_zs[0] == 2, "source in z direction should be 2 pixel thick"
        print("source extent: ", self.source_xs[0], self.source_xs[1], self.source_ys[0], self.source_ys[1], self.source_zs[0], self.source_zs[1])

        self.sources = {}
        for wl in self.wavelengths:
            # kz = 2 * jnp.pi * self.eps_background**.5 / wl
            sx = self.source_xs[1]-self.source_xs[0]
            sy = self.source_ys[1]-self.source_ys[0]
            sz = self.source_zs[1]-self.source_zs[0]
            source = torch.zeros((sx, sy, sz, 3), dtype=torch.complex64)

            theta, phi = source_angles
            theta = theta * np.pi / 180
            phi = phi * np.pi / 180
            kx = 2*np.pi * self.eps_substrate**.5 / wl * np.sin(theta) * np.cos(phi)
            ky = 2*np.pi * self.eps_substrate**.5 / wl * np.sin(theta) * np.sin(phi)
            kz = 2*np.pi * self.eps_substrate**.5 / wl * np.cos(theta)

            x = torch.linspace(self.source_xs[0]*self.dL, (self.source_xs[1]-1)*self.dL, sx) + global_frame_coordinate[0]
            y = torch.linspace(self.source_ys[0]*self.dL, (self.source_ys[1]-1)*self.dL, sy) + global_frame_coordinate[1]
            z = torch.linspace(self.source_zs[0]*self.dL, (self.source_zs[1]-1)*self.dL, sz) + global_frame_coordinate[2]
            
            x, y = torch.meshgrid(x,y,indexing='ij')
            map1 = torch.exp(-1j*(kx*x+ky*y+kz*z[1]))
            map2 = -torch.exp(-1j*(kx*x+ky*y+kz*z[1])-1j*kz*self.dL)
                
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
                source[:,:,1,1] = np.exp(1j*np.pi/2) * map1
                source[:,:,0,0] = map2
                source[:,:,0,1] = np.exp(1j*np.pi/2) * map2
            elif source_pol == 'rcp':
                source[:,:,1,0] = map1
                source[:,:,1,1] = np.exp(-1j*np.pi/2) * map1
                source[:,:,0,0] = map2
                source[:,:,0,1] = np.exp(-1j*np.pi/2) * map2
            self.sources[wl] = source

    def simulate(self, design_variable):
        fields = [None] * len(self.wavelengths)
        epsilon_r_bg = self.epsilon_r_bg()

        def _simulate(wavelength):
            epsilon_r = self.epsilon_r(design_variable)
            source = np.zeros(epsilon_r.shape + (3,), dtype=np.complex64)
            source[self.source_xs[0]:self.source_xs[1], self.source_ys[0]:self.source_ys[1], self.source_zs[0]:self.source_zs[1]] = self.sources[wavelength].numpy()
            # source[source.shape[0]//2, source.shape[1]//2, source.shape[2]//2, 2] = 1e6
            E = self.compute_FDFD(wavelength, epsilon_r, source, 'forward')

            # debug plot:
            # plot_3slices(epsilon_r, cm_zero_center=False, fname=os.path.join('epsilon_r.png'))
            # debug_plot(E[:,:,E.shape[2]//2,0], 'forward_Ex.png')
            # debug_plot(E[:,:,E.shape[2]//2,1], 'forward_Ey.png')
            # debug_plot(E[:,:,E.shape[2]//2,2], 'forward_Ez.png')
            # plot_3slices(E[...,0].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('forward_Ex.png'))
            # plot_3slices(E[...,1].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('forward_Ey.png'))
            # plot_3slices(E[...,2].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('forward_Ez.png'))
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
            source_torch = torch.conj(grad_E.to(torch.complex64)).resolve_conj()  # adjoint source
            # source_torch = grad_E.to(torch.complex64)

            # debug plot:
            # print("source_torch shape: ", source_torch.shape)
            # plot_3slices(source_torch[...,0].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('adj_sourcex.png'))
            # plot_3slices(source_torch[...,1].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('adj_sourcey.png'))
            # plot_3slices(source_torch[...,2].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('adj_sourcez.png'))

            adjoint_E = self.compute_FDFD(wavelength, epsilon_r, source_torch, 'adjoint')

            # plot_3slices(adjoint_E[...,0].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('adjoint_Ex.png'))
            # plot_3slices(adjoint_E[...,1].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('adjoint_Ey.png'))
            # plot_3slices(adjoint_E[...,2].detach().cpu().numpy().real, my_cmap=plt.cm.seismic, fname=os.path.join('adjoint_Ez.png'))
            # debug_plot(adjoint_E[:,:,adjoint_E.shape[2]//2,0], 'adjoint_Ex.png')
            # debug_plot(adjoint_E[:,:,adjoint_E.shape[2]//2,1], 'adjoint_Ey.png')
            # debug_plot(adjoint_E[:,:,adjoint_E.shape[2]//2,2], 'adjoint_Ez.png')

            design_variable_torch = design_variable.clone().detach().requires_grad_(True)
            epsilon_for_residual = self.make_torch_epsilon_r(design_variable_torch)[None]
            forward_E = c2r(forward_E[None].detach())

            forward_source = torch.zeros(epsilon_r.shape + (3,), dtype=torch.complex64)
            forward_source[self.source_xs[0]:self.source_xs[1], self.source_ys[0]:self.source_ys[1], self.source_zs[0]:self.source_zs[1]] = self.sources[wavelength]
            forward_source = c2r(forward_source[None].detach())

            residual = self.residual_fn(
                forward_E,
                epsilon_for_residual,
                forward_source,
                self.pmls,
                self.dL,
                wavelength,
            )
            
            input_grad = torch.autograd.grad(r2c(residual)[0], design_variable_torch, grad_outputs=torch.conj(adjoint_E))[0]
            # debug_plot(input_grad.squeeze(), 'input_grad.png')
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
    def __init__(self, *args, target_angles, target_angle_radius, SC_pml_space, farfield_points, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_angles = target_angles
        self.target_angle_radius = target_angle_radius
        self.SC_pml_space = resolve(SC_pml_space, self.problem.dL)
        self.farfield_points = farfield_points
        self.u0, self.thetas, self.phis = fibonacci_sphere(self.farfield_points)
        self.sphere_grid_kd = SphereGridKD(np.array(self.u0).transpose(1, 0))

        sss = resolve(self.problem.spec.source_substrate_spacing, self.problem.dL)
        assert self.SC_pml_space + sss < self.problem.surrounding_spaces[4], "SC monitor needs to be outside of the source"

        # compute the box size for the SC transformation
        p = self.problem
        self.Rx = (p.grid_shape[0]-p.pmls[0] - p.pmls[1] - 2*self.SC_pml_space)//2 if p.pml_pad_mode \
                    else (p.grid_shape[0] - 2*self.SC_pml_space)//2
        self.Ry = (p.grid_shape[1]-p.pmls[2] - p.pmls[3] - 2*self.SC_pml_space)//2 if p.pml_pad_mode \
                    else (p.grid_shape[1] - 2*self.SC_pml_space)//2
        self.Rz = (p.grid_shape[2]-p.pmls[4] - p.pmls[5] - 2*self.SC_pml_space)//2 if p.pml_pad_mode \
                    else (p.grid_shape[2] - 2*self.SC_pml_space)//2
        print("Rx, Ry, Rz: ", self.Rx, self.Ry, self.Rz)
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

            # (1) dipole near field within medium:
            # Ex = 1/2* (Ex + jnp.roll(Ex, 1, axis=0))
            # Ey = 1/2* (Ey + jnp.roll(Ey, 1, axis=1))
            # Ez = 1/2* (Ez + jnp.roll(Ez, 1, axis=2))

            sx, sy, sz = Ex.shape
            # print("sz-self.problem.pmls[5] - 5: ", sz-self.problem.pmls[5] - 5)
            offset_z = 10
            target = jnp.abs(Ex[sx//2, sy//2, sz//2+offset_z])**2 + jnp.abs(Ey[sx//2, sy//2, sz//2+offset_z])**2 + jnp.abs(Ez[sx//2, sy//2, sz//2+offset_z])**2
            loss += -target

            Ex_plot = jnp.abs(Ex[:,sy//2,:])
            Ey_plot = jnp.abs(Ey[:,sy//2,:])
            Ez_plot = jnp.abs(Ez[:,sy//2,:])
            aux[wavelength] = (None, None, (Ex_plot, Ey_plot, Ez_plot), None, None)

            # (2) near field above device:
            ## interpolate fields to be in the same physical space
            # Ex = 1/2* (Ex + jnp.roll(Ex, 1, axis=0))
            # Ey = 1/2* (Ey + jnp.roll(Ey, 1, axis=1))
            # Ez = 1/2* (Ez + jnp.roll(Ez, 1, axis=2))

            # sx, sy, sz = Ex.shape
            # target_z = sz-self.problem.pmls[5] - 5
            # # target = -jnp.abs(Ex[sx//2, sy//2, target_z])**2
            # target = (jnp.abs(Ex[sx//2, sy//2, target_z])**2 + jnp.abs(Ey[sx//2, sy//2, target_z])**2 + jnp.abs(Ez[sx//2, sy//2, target_z])**2)
            # loss += target

            # Ex_plot = Ex[:,sy//2, :].real
            # aux[wavelength] = (None, None, Ex_plot, None, None)

            # (3) farfield with straton-chu:
            # Hx, Hy, Hz = E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None)
            # thetas, phis, u0, far_E, far_H = strattonChu3D_full_sphere_GPU(
            #     dl=self.problem.dL,
            #     xc=Ex.shape[0]//2,
            #     yc=Ex.shape[1]//2,
            #     zc=Ex.shape[2]//2,
            #     Rx=self.Rx,
            #     Ry=self.Ry,
            #     Rz=self.Rz,
            #     lambda_val=wavelength,
            #     eps_background=self.problem.eps_background,
            #     Ex_OBJ=Ex[None],
            #     Ey_OBJ=Ey[None],
            #     Ez_OBJ=Ez[None],
            #     Hx_OBJ=Hx[None],
            #     Hy_OBJ=Hy[None],
            #     Hz_OBJ=Hz[None],
            #     N_points_on_sphere=self.farfield_points,
            # )

            # # plot_poynting_radial_scatter(u0, far_E, far_H, fname='poynting_radial_scatter.png',
            # #                 plot_batch_idx=0, normalize=True, point_size=8)

            # t_theta, t_phi = self.target_angles
            # idx = self.sphere_grid_kd.query_cone(t_theta * np.pi / 180, t_phi * np.pi / 180, self.target_angle_radius * np.pi / 180)

            # print("target thetas: ", thetas[idx]*180/np.pi, "target phis: ", phis[idx]*180/np.pi)

            # S = 0.5 * jnp.real(jnp.cross(far_E[0], jnp.conj(far_H[0]), axis=0))  # (3,N_points_on_sphere), real
            # Sr = jnp.sum(S * u0[0], axis=0)      # (N_points_on_sphere), real

            # # eff = jnp.abs(Sr)
            # eff = Sr
            # assert (eff>0).all(), "eff should be positive"

            # target_region = eff[idx]
            # # loss += -1e2*jnp.sum(target_region)
            # loss += 1e2*(1 - jnp.sum(target_region) / jnp.sum(eff))

            # sx, sy, sz = Ez.shape
            # Ex_plot = Ex[:,sy//2, :].real
            # aux[wavelength] = (u0[0], Sr, Ex_plot, thetas, phis)

        return loss, aux