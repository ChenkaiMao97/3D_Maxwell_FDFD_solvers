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
import src.utils.waveguide_mode as modes

import gin

_DENSITY_LABEL = "density"

@dataclass
class ResponseArray:
    array: jnp.ndarray

jax.tree_util.register_pytree_node(
    nodetype=ResponseArray,
    flatten_func=lambda s: ( (s.array,), (None,) ),
    unflatten_func=lambda aux, array: ResponseArray(*array),
)

@gin.configurable
class IntegratedPhotonicsSpec:
    """Parameters specifying the setup of the integrated photonics problem.
    Specifically, this spec specifies how are the waveguides connected to the 4 sides of the design region

    Attributes:
    port_pml_offset: the offset between the ports and the PML (same unit as dL)
    input_monitor_offset: The offset of the input monitor from the input source. (same unit as dL)
    wg_min_separation: the minimum spacing between adjascent waveguides
    wg_mode_padding: padding space added to the waveguide width for computing the modes
    <side>_wg_specs: list of tuples of two lengths and one integer: (center_offset, width, order), first is offset from the block side-center, second is width, third is order of the waveguide mode
    """
    def __init__(self,
        port_pml_offset: float,
        input_monitor_offset: float,
        wg_eps: float,
        wg_min_separation: float,
        wg_mode_padding: float,
        excite_port_idx: int,
        left_wg_specs: Tuple[Tuple[float, float, int]], # (center_offset, width, order)
        right_wg_specs: Tuple[Tuple[float, float, int]], # (center_offset, width, order)
        top_wg_specs: Tuple[Tuple[float, float, int]], # (center_offset, width, order)
        bottom_wg_specs: Tuple[Tuple[float, float, int]], # (center_offset, width, order)
    ):
        self.port_pml_offset = port_pml_offset
        self.input_monitor_offset = input_monitor_offset
        self.wg_eps = wg_eps
        self.wg_min_separation = wg_min_separation
        self.wg_mode_padding = wg_mode_padding

        self.excite_port_idx = excite_port_idx
        self.left_wg_specs = left_wg_specs
        self.right_wg_specs = right_wg_specs
        self.top_wg_specs = top_wg_specs
        self.bottom_wg_specs = bottom_wg_specs

        self.ln_R = -10
        printc(f"using ln_R: {self.ln_R}", 'o')

    def __post_init__(self):
        assert self.wg_mode_padding <= self.wg_min_separation

@gin.configurable
class IntegratedPhotonicsProblem(BaseProblem):
    def __init__(
        self,
        *args,
        spec: IntegratedPhotonicsSpec,
        **kwargs):
        super().__init__(*args, **kwargs)

        self.spec = spec
        self.excite_port_idx = spec.excite_port_idx
        self.residual_fn = residue_E
    
    def init(self, shape_only=False):
        self._make_ports(precompute_mode=not shape_only)

    def _make_ports(self, precompute_mode=True):
        self._ports = []
        self._waveguides = []

        s = self.spec

        def make_wgs_and_ports_for_one_side(specs, loc='l'):
            if len(specs) == 0:
                return

            # The first port on the left side will be the excitation port (with largest y coord), so all ports have coordinates sorted from max to min
            previous_front = self.design_region_y_start if loc in ['l', 'r'] else self.design_region_x_start
            for idx, wg_spec in enumerate(specs):
                # create the waveguide
                center_offset, width, order = wg_spec

                z_start = self.design_region_z_start
                z_end = self.design_region_z_end
                eps = s.wg_eps
                if loc == 'l':
                    x_start = 0
                    x_end = self.design_region_x_start
                    y_start = round(1/2*(self.design_region_y_start + self.design_region_y_end)) + resolve(center_offset, self.dL) - resolve(width/2, self.dL)
                    y_end = y_start + resolve(width, self.dL)
                    axis_vector = [-1, 0, 0]
                    wg_min, wg_max = y_start, y_end
                    
                    port_x = self.pmls[0] + resolve(s.port_pml_offset, self.dL)
                    assert port_x < x_end, "waveguide not long enough, port is inside design region"
                    port_y = round(1/2*(y_start + y_end))
                elif loc == 'r':
                    x_start = self.design_region_x_end
                    x_end = self.grid_shape[0]
                    y_start = round(1/2*(self.design_region_y_start + self.design_region_y_end)) + resolve(center_offset, self.dL) - resolve(width/2, self.dL)
                    y_end = y_start + resolve(width, self.dL)
                    axis_vector = [1, 0, 0]
                    wg_min, wg_max = y_start, y_end

                    port_x = self.grid_shape[0] - self.pmls[1] - resolve(s.port_pml_offset, self.dL)
                    assert port_x > x_start, "waveguide not long enough, port is inside design region"
                    port_y = round(1/2*(y_start + y_end))
                elif loc == 't':
                    x_start = round(1/2*(self.design_region_x_start + self.design_region_x_end)) + resolve(center_offset, self.dL) - resolve(width/2, self.dL)
                    x_end = x_start + resolve(width, self.dL)
                    y_start = self.design_region_y_end
                    y_end = self.grid_shape[1]
                    axis_vector = [0, 1, 0]
                    wg_min, wg_max = x_start, x_end

                    port_x = round(1/2*(x_start + x_end))
                    port_y = self.grid_shape[1] - self.pmls[3] - resolve(s.port_pml_offset, self.dL)
                    assert port_y > y_start, "waveguide not long enough, port is inside design region"
                elif loc == 'b':
                    x_start = round(1/2*(self.design_region_x_start + self.design_region_x_end)) + resolve(center_offset, self.dL) - resolve(width/2, self.dL)
                    x_end = x_start + resolve(width, self.dL)
                    y_start = 0
                    y_end = self.design_region_y_start
                    axis_vector = [0, -1, 0]
                    wg_min, wg_max = x_start, x_end

                    port_x = round(1/2*(x_start + x_end))
                    port_y = self.pmls[2] + resolve(s.port_pml_offset, self.dL)
                    assert port_y < y_end, "waveguide not long enough, port is inside design region"
                else:
                    raise ValueError("loc needs to be in 'tblr'")
                assert wg_min>=previous_front # check waveguide spacing, and also within design region extent
                previous_front = wg_max + resolve(s.wg_min_separation, self.dL)

                self._waveguides.append(
                    modes.Waveguide(
                        x_start=x_start,
                        x_end=x_end,
                        y_start=y_start,
                        y_end=y_end,
                        z_start=z_start,
                        z_end=z_end,
                        eps=eps
                    )    
                )
                if len(self._ports) == s.excite_port_idx:
                    # if this is the excitation port, add a port at the same location but in the opposite direction
                    # so at this location there will be two ports, one for excitation and one for reflection
                    # for all other ports, there is only one port for transmission
                    self._ports.append(
                        modes.WaveguidePort(
                            x=port_x,
                            y=port_y,
                            z=round(1/2*(z_start + z_end)),
                            width=resolve(width + 2 * s.wg_mode_padding, self.dL),
                            height=z_end-z_start+2*resolve(s.wg_mode_padding, self.dL),
                            order=order,
                            axis_vector=[-i for i in axis_vector],
                            offset=resolve(s.input_monitor_offset, self.dL) # constant doesn't depend no location
                        )
                    )

                self._ports.append(
                    modes.WaveguidePort(
                        x=port_x,
                        y=port_y,
                        z=round(1/2*(z_start + z_end)),
                        width=resolve(width + 2 * s.wg_mode_padding, self.dL),
                        height=z_end-z_start+2*resolve(s.wg_mode_padding, self.dL),
                        order=order,
                        axis_vector=axis_vector,
                        offset=resolve(s.input_monitor_offset, self.dL) # constant doesn't depend no location
                    )
                )

            assert wg_max <= (self.design_region_y_end if loc in ['l', 'r'] else self.design_region_x_end)

        make_wgs_and_ports_for_one_side(s.left_wg_specs, loc='l')
        make_wgs_and_ports_for_one_side(s.top_wg_specs, loc='t')
        make_wgs_and_ports_for_one_side(s.right_wg_specs, loc='r')
        make_wgs_and_ports_for_one_side(s.bottom_wg_specs, loc='b')
        print(f"simulation configured with {len(self._ports)} ports and {len(self._waveguides)} waveguides")

        # precompute waveguide modes sources and overlap_es
        if precompute_mode:
            epsilon_r_bg = self.epsilon_r_bg()
            pbar = tqdm(self._ports, desc="Precomputing waveguide modes sources and overlap_es", total=len(self._ports), leave=False)
            idx = 0
            for port in pbar:
                port.precompute_mode(
                    wavelengths=self.wavelengths,
                    dL=self.dL,
                    epsilon_r=epsilon_r_bg,
                    pml_layers=self.pmls,
                    power=1,
                    ln_R=s.ln_R,
                    precompute_source=True if port == self._ports[s.excite_port_idx] else False
                )
                # debugging plot for mode:
                # plt.figure(figsize=(10, 12))
                # overlap_e = port.overlap_e(self.wavelengths[0])
                # overlap_e_plot = overlap_e[:,:,round(1/2*(overlap_e.shape[2]-1))]
                # plt.subplot(3,2,1)
                # plt.imshow(overlap_e_plot[...,0].real, cmap="seismic")
                # plt.colorbar()
                # plt.subplot(3,2,2)
                # plt.imshow(overlap_e_plot[...,0].imag, cmap="seismic")
                # plt.colorbar()
                # plt.subplot(3,2,3)
                # plt.imshow(overlap_e_plot[...,1].real, cmap="seismic")
                # plt.colorbar()
                # plt.subplot(3,2,4)
                # plt.imshow(overlap_e_plot[...,1].imag, cmap="seismic")
                # plt.colorbar()
                # plt.subplot(3,2,5)
                # plt.imshow(overlap_e_plot[...,2].real, cmap="seismic")
                # plt.colorbar()
                # plt.subplot(3,2,6)
                # plt.imshow(overlap_e_plot[...,2].imag, cmap="seismic")
                # plt.colorbar()
                # plt.savefig(f"debug_overlap_e_port_{idx}_order_{port.order}.png", dpi=300)
                # plt.close()
                # idx += 1

    def simulate(self, design_variable):
        fields = [None] * len(self.wavelengths)

        def _simulate(wavelength):
            epsilon_r = self.epsilon_r(design_variable)

            source = self._ports[self.excite_port_idx].source(
                wavelength
            )
            source = np.transpose(source, (1,2,3,0))

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

            forward_source = self._ports[self.excite_port_idx].source(wavelength)
            forward_source = torch.from_numpy(np.transpose(forward_source, (1,2,3,0)))
            forward_source = c2r(forward_source[None].detach())

            residual = self.residual_fn(
                forward_E,
                epsilon_for_residual,
                forward_source,
                self.pmls,
                self.dL,
                wavelength,
            )

            input_grad = 2*torch.autograd.grad(r2c(residual)[0], design_variable_torch, grad_outputs=torch.conj(adjoint_E))[0]

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
class IntegratedPhotonicsChallenge(BaseChallenge):
    def __init__(
        self, 
        *args, 
        target_s_params: Tuple[Tuple[float]], # target transmission and reflection ratios for each wavelength
        **kwargs):
        super().__init__(*args, **kwargs)

        self.target_s_params = jnp.array(target_s_params)

    def response(self, params):
        density = params[_DENSITY_LABEL].density
        return self._jax_sim_fn(density)

    def loss(self, response):
        sp = None
        sm = []
        s_params = []
        for wavelength in self._wavelengths:
            wl_idx = self._wavelengths.index(wavelength)
            E = response[wl_idx]
            for j, port in enumerate(self.problem._ports):
                overlap_e = port.overlap_e(wavelength)

                mode_overlap = jnp.sum(jnp.conj(E) * overlap_e)
                if j == self.problem.excite_port_idx:
                    sp = mode_overlap
                else:
                    sm.append(mode_overlap)
            s_params.append(jnp.stack([smi / sp for smi in sm]))

        aux = jnp.stack(s_params)
        loss = jnp.sum((jnp.abs(aux) ** 2 - self.target_s_params)**2)

        return loss, aux
