import sys, os
import numpy as np
import torch
import importlib
import ast
from src.solvers.gmres import mygmrestorch
from src.solvers.bicgstab import mybicgstab
from src.utils.physics import residue_E, src2rhs
from src.utils.utils import *
import time
import gc
import gin

@gin.configurable
class NN_solver:
    def __init__(
        self, 
        model_path = None,
        sim_shape = None,
        wl = None,
        dL = None,
        pmls = None,
        max_iter = None,
        tol = None,
        verbose = None,
        restart = None,
        save_intermediate = False,
        output_dir = None,
        gpu_id = None,
        solver_type = 'gmres',
    ):
        self.model_path = model_path
        self.sim_shape = sim_shape
        self.wl = wl
        self.dL = dL
        self.pmls = pmls
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.restart = restart
        self.gpu_id = gpu_id
        self.save_intermediate = save_intermediate
        self.output_dir = output_dir
        self.solver_type = solver_type
        self.residual_fn = residue_E
        self.residual_type = 'SC-PML'

    def init(self):
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)
        for file in os.listdir(self.model_path):
            if file.endswith(".gin"):
                gin.parse_config_file(os.path.join(self.model_path, file), skip_unknown=True)
        
        # load the model
        from waveynet3d.models import model_factory as model_fn
        self.model = prepare_model(self.sim_shape, self.model_path, model_fn, device_id=self.gpu_id)

        def _checkpoint_binding(name, default=None):
            for gin_file in os.listdir(self.model_path):
                if not gin_file.endswith('.gin'):
                    continue
                with open(os.path.join(self.model_path, gin_file), 'r') as fh:
                    for raw_line in fh:
                        line = raw_line.split('#', 1)[0].strip()
                        if not line or '=' not in line:
                            continue
                        lhs, rhs = line.split('=', 1)
                        if lhs.strip().endswith('.' + name):
                            try:
                                return ast.literal_eval(rhs.strip())
                            except Exception:
                                return rhs.strip().strip("'\"")
            return default

        self.residual_type = _checkpoint_binding('residual_type', 'SC-PML')
        self.dummy_ds = None

        # Non-damping legacy checkpoints need the old dummy dataset path to
        # reconstruct SC-PML feature channels. Damping checkpoints do not: the
        # absorber feature is deterministic from (shape, pmls, wl, dL).
        if self.residual_type != 'damping':
            from waveynet3d.data.simulation_dataset import SyntheticDataset_same_wl_dL_shape as dataset_fn
            trainer_errors = []
            self.dummy_trainer = None
            for module_name, class_name in (
                ('waveynet3d.trainers.iterative_trainer_restarted', 'IterativeTrainerRestarted'),
                ('waveynet3d.trainers.iterative_trainer', 'IterativeTrainer'),
            ):
                try:
                    trainer_cls = getattr(importlib.import_module(module_name), class_name)
                    self.dummy_trainer = trainer_cls(model_config=None, model_saving_path=None)
                    print(f"Recovered checkpoint trainer class: {class_name}")
                    break
                except Exception as exc:
                    trainer_errors.append(f"{class_name}: {exc}")
            if self.dummy_trainer is None:
                raise TypeError(
                    "Could not reconstruct a checkpoint trainer to recover domain_sizes, "
                    "pml_ranges, and residual_type. Tried: " + " | ".join(trainer_errors)
                )
            self.dummy_ds = dataset_fn(self.dummy_trainer.domain_sizes, self.dummy_trainer.pml_ranges, residual_type=self.dummy_trainer.residual_type)
            self.dummy_ds.set_ln_R(self.dummy_trainer.ln_R)

        # Precompute boundary/absorber feature channels for the model and use
        # the same operator family that the checkpoint was trained against.
        # For damping models the absorber is an imaginary eps channel; do not
        # call build_complex_eps here because that samples a random pml thickness
        # from the training range when residual_type == 'damping'.
        if self.residual_type == 'damping':
            from waveynet3d.gym.full_residual_util_for_training import residue_E_damping
            from waveynet3d.gym.PML_utils import adiabatic_damping_imag_eps

            self.residual_fn = residue_E_damping
            damping_imag = adiabatic_damping_imag_eps(
                tuple(self.sim_shape),
                [int(p) for p in self.pmls],
                dL_nm=float(self.dL),
                wl_nm=float(self.wl),
                order=getattr(self.dummy_ds, 'adiabatic_order', 3.0),
                R_target=getattr(self.dummy_ds, 'adiabatic_R', 1e-10),
                dtype=np.float32,
            )
            self.PML_channels = torch.from_numpy(damping_imag)[None, ..., None].cuda()
        else:
            self.residual_fn = residue_E
            dummy_eps = torch.zeros(self.sim_shape)
            dummy_eps, _ = self.dummy_ds.build_complex_eps(dummy_eps, self.wl, self.dL, self.sim_shape, pml=self.pmls)
            self.PML_channels = dummy_eps[None, ..., 1:].cuda()
        print(f"NN solver residual_type: {self.residual_type}, pmls: {self.pmls}")

    def solve(self, eps, src, gt=None, init_x=None, transpose=False):
        # Build the eps representation expected by the checkpoint. For damping
        # this is a two-channel complex eps (real eps, absorber imag eps). For
        # SC-PML the operator still consumes only real eps and the model sees
        # separate stretch-coordinate feature channels.
        eps = torch.cat([eps[..., None], self.PML_channels], dim=-1)
        operator_eps = eps if self.residual_type == 'damping' else eps[..., 0]

        # prepare the GMRES solver:
        def Aop_forward(x):
            return r2c(self.residual_fn(c2r(x), operator_eps, src, self.pmls, self.dL, self.wl, batched_compute=True, Aop=True))

        def Aop_transpose(x):
            # JAX complex cotangents use the unconjugated transpose. PyTorch
            # autograd returns the Hermitian VJP, so conj(grad(..., conj(x)))
            # gives an A.T matvec without hand-writing transposed PML stencils.
            with torch.enable_grad():
                probe = torch.zeros_like(x, requires_grad=True)
                A_probe = Aop_forward(probe)
                grad = torch.autograd.grad(
                    A_probe,
                    probe,
                    grad_outputs=torch.conj(x),
                    retain_graph=False,
                    create_graph=False,
                )[0]
            return torch.conj(grad).detach()

        Aop = Aop_transpose if transpose else Aop_forward
        residual_fn = lambda x: r2c(self.residual_fn(c2r(x), operator_eps, src, self.pmls, self.dL, self.wl, batched_compute=True, Aop=False))
        if self.solver_type == 'gmres':
            solver = mygmrestorch(self.model, Aop, tol=self.tol, max_iter=self.max_iter)
        elif self.solver_type == 'bicgstab':
            solver = mybicgstab(self.model, Aop, tol=self.tol, max_iter=self.max_iter)

        complex_rhs = r2c(src) if transpose else r2c(src2rhs(src, self.dL, self.wl))
        freq = torch.tensor(self.dL/self.wl)[None].cuda()
        solver.setup_eps(eps, freq)
        if self.restart == 0:
            x, history, _, _ = solver.solve(complex_rhs, self.verbose, init_x=init_x)
        else:
            x, history = solver.solve_with_restart(complex_rhs, self.tol, self.max_iter, self.restart, self.verbose, init_x=init_x)
        # final_residual = self.residual_fn(x)
        # release memory:
        del complex_rhs, solver, history
        gc.collect()
        torch.cuda.empty_cache()

        return x

@torch.no_grad()
def NN_solve(config, eps, src, return_xr_history=False, plot_iters=None):
    print("in NN_solve: ", return_xr_history, plot_iters)
    model_path = config["model_path"]

    sim_shape = config["sim_shape"]
    wl = float(config["wavelength"])
    dL = float(config["dL"])
    pmls = config["pmls"]

    max_iter = int(config["max_iter"])
    tol = float(config["tol"])
    verbose = config["verbose"]
    restart = int(config["restart"])

    solver_type = config["solver_type"] if "solver_type" in config else "gmres"
    epoch = config["epoch"] if "epoch" in config else None

    ########## first parse the gin files, which contains the model configurations ##########
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    for file in os.listdir(model_path):
        if file.endswith(".gin"):
            gin.parse_config_file(os.path.join(model_path, file), skip_unknown=True)
    
    # load the model
    from waveynet3d.models import model_factory as model_fn
    model = prepare_model(sim_shape, model_path, model_fn, epoch=epoch)

    # use the dummy trainer and ds to reproduce the feature engineering for eps (this part should be rewritten to be cleaner)
    from waveynet3d.data.simulation_dataset import SyntheticDataset_same_wl_dL_shape as dataset_fn
    # from waveynet3d.data.simulation_dataset import SyntheticDataset_same_wl_dL as dataset_fn
    # from waveynet3d.trainers.iterative_trainer_bicgstab import IterativeTrainerBiCGStab as trainer_fn

    trainer_errors = []
    dummy_trainer = None
    for module_name, class_name in (
        ('waveynet3d.trainers.iterative_trainer_restarted', 'IterativeTrainerRestarted'),
        ('waveynet3d.trainers.iterative_trainer', 'IterativeTrainer'),
    ):
        try:
            trainer_cls = getattr(importlib.import_module(module_name), class_name)
            dummy_trainer = trainer_cls(model_config=None, model_saving_path=None)
            print(f"Recovered checkpoint trainer class: {class_name}")
            break
        except Exception as exc:
            trainer_errors.append(f"{class_name}: {exc}")
    if dummy_trainer is None:
        raise TypeError(
            "Could not reconstruct a checkpoint trainer to recover domain_sizes, "
            "pml_ranges, and residual_type. Tried: " + " | ".join(trainer_errors)
        )
    dummy_ds = dataset_fn(dummy_trainer.domain_sizes, dummy_trainer.pml_ranges, residual_type=dummy_trainer.residual_type)
    # check_data_distribution(eps, pmls, wl, dL, dummy_trainer, dummy_ds)

    dummy_ds.set_ln_R(dummy_trainer.ln_R)
    print(f"NN solver uses ln_R (parameter for PML): {dummy_trainer.ln_R}")

    eps, _ = dummy_ds.build_complex_eps(eps[0], wl, dL, sim_shape, pml=pmls) # add more channels to eps, which contains pml features
    eps = eps[None]

    eps = eps.cuda()
    src = src.cuda()

    # prepare the GMRES solver:
    Aop = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL, wl, batched_compute=True, Aop=True))
    residual_fn = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL, wl, batched_compute=True, Aop=False))

    # solve the problem:
    time_start = time.time()
    complex_rhs = r2c(src2rhs(src, dL, wl))
    freq = torch.tensor(dL/wl)[None].cuda()


    if solver_type == 'gmres':
        solver = mygmrestorch(model, Aop, tol=tol, max_iter=max_iter)
    elif solver_type == 'bicgstab':
        solver = mybicgstab(model, Aop, tol=tol, max_iter=max_iter)
    solver.setup_eps(eps, freq)
    if restart == 0:
        x, history, x_history, r_history = solver.solve(complex_rhs, tol, max_iter, return_xr_history=return_xr_history, plot_iters=plot_iters, verbose=verbose)
    else:
        x, history, x_history, r_history = solver.solve_with_restart(complex_rhs, tol, max_iter, restart, return_xr_history=return_xr_history, plot_iters=plot_iters, verbose=verbose)
    time_end = time.time()
    print(f"time taken for NN {solver_type} solver: {time_end - time_start} seconds")
    final_residual = residual_fn(x)

    if return_xr_history:
        return x, history, final_residual, x_history, r_history
    else:
        return x, history, final_residual