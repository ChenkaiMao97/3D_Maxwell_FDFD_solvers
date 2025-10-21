import os
from src.utils.utils import printc, is_multiple, resolve
from src.invde.utils.utils import *

import numpy as np
import torch
import multiprocessing as mp
import threading
import concurrent.futures

from functools import cached_property

from src.utils.GPU_worker_utils import solver_worker
from src.utils.physics import E_to_H

import gin

@gin.configurable
class BaseProblem:
    def __init__(
        self,
        design_region_size,
        surrounding_sizes,
        pmls,
        dL,
        wavelengths,
        eps_design_max,
        eps_design_min,
        eps_background,
        _backend='NN', # 'NN' or 'spins'
        density_dim = 2,
        eps_substrate = None
    ):
        """
        The Problem classes takes care of constructing the geometry of the problem, and the simulation of the problem.
        The core functions are the forward simulation and the adjoint simulation, which can be done by either spins or NN solver.

        Args:
        - design_region_size: size of the design region (same unit as dL): (x_size, y_size, z_size)
        - surrounding_sizes: empty space around the design region (same unit as dL): (x_start_space, x_end_space, y_start_space, y_end_space, z_start_space, z_end_space)
        - pmls: number of pixels of PML for x_start, x_end, y_start, y_end, z_start, z_end: (x_pml_start, x_pml_end, y_pml_start, y_pml_end, z_pml_start, z_pml_end)
        - dL: float, resolution of the grid
        - wavelengths: list of wavelengths to simulate
        - eps_design_max: maximum permittivity of the design region
        - eps_design_min: minimum permittivity of the design region
        - eps_background: permittivity of the background
        - _backend: backend to use for simulation ('NN' or 'spins')
        - density_dim: dimension of the density (2 for 2d pattern extrudded into 3d)
        - eps_substrate: permittivity of the substrate
        """
        
        assert len(design_region_size) == 3, "design_region_size must be a tuple of 3 elements"
        assert len(surrounding_sizes) == 6, "surrounding_sizes must be a tuple of 6 elements"
        assert len(pmls) == 6, "pmls must be a tuple of 6 elements"
        assert is_multiple(design_region_size, dL), f"design_region_size: {design_region_size} must be a multiple of dL: {dL}"
        assert is_multiple(surrounding_sizes, dL), f"surrounding_sizes: {surrounding_sizes} must be a multiple of dL: {dL}"

        assert density_dim == 2, "currently we assume design region is 2d pattern extrudded into 3d, so density_dim should be 2"
        self.density_dim = density_dim

        self.design_variable_shape = [resolve(i, dL) for i in design_region_size]
        self.surrounding_spaces = [resolve(i, dL) for i in surrounding_sizes]
        self.pmls = pmls
        self.dL = dL
        self.wavelengths = wavelengths
        
        self.grid_shape = (
            self.design_variable_shape[0] + self.surrounding_spaces[0] + self.surrounding_spaces[1] + self.pmls[0] + self.pmls[1],
            self.design_variable_shape[1] + self.surrounding_spaces[2] + self.surrounding_spaces[3] + self.pmls[2] + self.pmls[3],
            self.design_variable_shape[2] + self.surrounding_spaces[4] + self.surrounding_spaces[5] + self.pmls[4] + self.pmls[5]
        )
        printc(f"simulation with grid shape: {self.grid_shape}", "g")

        self.eps_design_max = eps_design_max
        self.eps_design_min = eps_design_min
        self.eps_background = eps_background
        self.eps_substrate = eps_substrate

        self._backend = _backend

        self.ports = [] # waveguides
        self.plane_waves = []
    
    def init_GPU_workers(self):
        # init solvers on each GPU
        gpu_ids = list(range(len(self.wavelengths)))
        self.num_gpus = len(gpu_ids)
        assert self.num_gpus <= torch.cuda.device_count(), f"number of wavelengths, hence GPUs, {len(gpu_ids)} is greater than the number of available GPUs {torch.cuda.device_count()}"

        self.task_queues = [mp.Queue() for _ in range(self.num_gpus)]
        self.result_queue = mp.Queue()
        # self.init_queues = [mp.Queue() for _ in range(self.num_gpus)] # for passing back values after init
        
        self.processes = []

        for device_id in gpu_ids:
            init_kwargs = {
                'sim_shape': self.grid_shape,
                'wl': self.wavelengths[device_id],
                'dL': self.dL,
                'pmls': self.pmls,
                'save_intermediate': False,
                'output_dir': None,
            }
            # p = mp.Process(target=solver_worker, args=(device_id, init_kwargs, self.task_queues[device_id], self.result_queue, self.init_queues[device_id]))
            p = mp.Process(target=solver_worker, args=(device_id, init_kwargs, self.task_queues[device_id], self.result_queue))
            p.start()
            self.processes.append(p)

        self.task_id_counter = 0
        self.results = {}

        self.results_lock = threading.Lock()
        self.results_cond = threading.Condition(self.results_lock)
        self.listener_thread = threading.Thread(target=self._result_listener, daemon=True)
        self.listener_thread.start()
    
    def _result_listener(self):
        while True:
            item = self.result_queue.get()
            if item is None:
                break
            task_id, result = item
            with self.results_cond:
                self.results[task_id] = result
                self.results_cond.notify_all()
    
    def stop_workers(self):
        if self._backend == 'NN':
            for i in range(self.num_gpus):
                self.task_queues[i].put(None)
            for p in self.processes:
                p.join()

            self.result_queue.put(None)
            self.listener_thread.join()
        print("all process and threads stopped")
    
    def compute_FDFD(self, wl, epsilon_r, source):
        assert wl in self.wavelengths, f"wavelength {wl} is not in the list of wavelengths: {self.wavelengths}"

        # wl_torch = torch.tensor([wl], dtype=torch.float32)
        # dl_torch = torch.tensor([self.dL], dtype=torch.float32)
        # epsilon_r_torch = torch.tensor(epsilon_r, dtype=torch.float32)
        # source_torch = torch.tensor(source, dtype=torch.complex64)

        # send task to GPU worker queue
        task_id = self.task_id_counter
        self.task_id_counter += 1
        device_id = self.wavelengths.index(wl) # each device handles one omega, to reuse the precomputed PML 
        
        # last_E = self.last_forward_E.get(omega, None)
        last_E = None

        self.task_queues[device_id].put((task_id, (epsilon_r, source, wl, self.dL, self.pmls, None, last_E)))

        # wait and fetch the result
        with self.results_cond:
            while task_id not in self.results:
                self.results_cond.wait()
        e = self.results.pop(task_id)[None]
        
        return e
    
    def simulate(self):
        raise NotImplementedError("This method must be implemented in the subclass")
        pass
    
    def simulate_adjoint(self):
        raise NotImplementedError("This method must be implemented in the subclass")
        pass
    
    def add_waveguide(self, waveguide):
        pass
    
    @cached_property
    def density_bg(self):
        """
        background density has 0 in the design region, for background medium, substrate and waveguides, it is linearly interpolated (or extrapolated)
        """
        background_density = (self.eps_background - self.eps_design_min) / (self.eps_design_max - self.eps_design_min)
        d = background_density * np.ones(self.grid_shape, dtype=np.float32)
        d[self.design_region_x_start:self.design_region_x_end, self.design_region_y_start:self.design_region_y_end, self.design_region_z_start:self.design_region_z_end] = 0
        
        if self.eps_substrate is not None:
            d[:, :, :self.design_region_z_start] = (self.eps_substrate - self.eps_design_min) / (self.eps_design_max - self.eps_design_min)

        for wg in self._waveguides:
            wg_eps = wg.eps
            wg_density = (wg_eps - self.eps_design_min) / (self.eps_design_max - self.eps_design_min)
            d[wg.x_start:wg.x_end, wg.y_start:wg.y_end, wg.z_start:wg.z_end] = wg_density

        return d
    
    def density2d(self, design_variable):
        # for extracting solid and void pixels
        assert len(design_variable.shape) == 2
        density2d = self.density_bg.copy()[:,:,round(1/2*(self.design_region_z_start + self.design_region_z_end))]
        density2d[self.design_region_x_start:self.design_region_x_end, self.design_region_y_start:self.design_region_y_end] = design_variable
        return density2d

    def density(self, design_variable):
        assert len(design_variable.shape) == self.density_dim, f"design_variable dimension {len(design_variable.shape)} and density_dim {self.density_dim} mismatch"
        density = self.density_bg.copy()
        if len(design_variable.shape) == 2:
            design_variable = design_variable[..., None]
        density[self.design_region_x_start:self.design_region_x_end, self.design_region_y_start:self.design_region_y_end, self.design_region_z_start:self.design_region_z_end] = design_variable
        return density

    def epsilon_r_bg(self):
        density_bg = self.density_bg
        eps_bg = density_bg * (self.eps_design_max - self.eps_design_min) + self.eps_design_min
        return eps_bg
    
    def epsilon_r(self, design_variable):
        # design_variable is from 0 to 1
        assert design_variable.shape == tuple(self.design_variable_shape[:2]), f"design_variable shape should be {tuple(self.design_variable_shape[:2])}"
        full_density = self.density(design_variable)
        full_eps = full_density * (self.eps_design_max - self.eps_design_min) + self.eps_design_min
        return full_eps
    
    @cached_property
    def design_region_x_start(self):
        return self.pmls[0] + self.surrounding_spaces[0]
    @cached_property
    def design_region_x_end(self):
        return self.grid_shape[0] - self.pmls[1] - self.surrounding_spaces[1]
    @cached_property
    def design_region_y_start(self):
        return self.pmls[2] + self.surrounding_spaces[2]
    @cached_property
    def design_region_y_end(self):
        return self.grid_shape[1] - self.pmls[3] - self.surrounding_spaces[3]
    @cached_property
    def design_region_z_start(self):
        return self.pmls[4] + self.surrounding_spaces[4]
    @cached_property
    def design_region_z_end(self):
        return self.grid_shape[2] - self.pmls[5] - self.surrounding_spaces[5]