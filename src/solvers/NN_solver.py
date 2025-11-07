import sys, os
from src.solvers.gmres import mygmrestorch
from src.utils.physics import residue_E, src2rhs
from src.utils.utils import *
import time
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

        self.residual_fn = residue_E

    def init(self):
        sys.path.append(self.model_path)
        for file in os.listdir(self.model_path):
            if file.endswith(".gin"):
                gin.parse_config_file(os.path.join(self.model_path, file))
        
        # load the model
        from waveynet3d.models import model_factory as model_fn
        self.model = prepare_model(self.sim_shape, self.model_path, model_fn, device_id=self.gpu_id)

        # use the dummy trainer and ds to reproduce the feature engineering for eps (this part should be rewritten to be cleaner)
        from waveynet3d.data.simulation_dataset import SyntheticDataset_same_wl_dL_shape as dataset_fn
        from waveynet3d.trainers.iterative_trainer import IterativeTrainer as trainer_fn
        self.dummy_trainer = trainer_fn(model_config=None, model_saving_path=None)
        self.dummy_ds = dataset_fn(self.dummy_trainer.domain_sizes, self.dummy_trainer.pml_ranges, residual_type=self.dummy_trainer.residual_type)
        self.dummy_ds.set_ln_R(self.dummy_trainer.ln_R)

        # precompute the PML features:
        dummy_eps = torch.zeros(self.sim_shape)
        dummy_eps, _ = self.dummy_ds.build_complex_eps(dummy_eps, self.wl, self.dL, self.sim_shape, pml=self.pmls)
        self.PML_channels = dummy_eps[...,1:]

    def solve(self, eps, src, gt=None, init_x=None):
        # build the complex eps:
        eps = torch.cat([eps[..., None], self.PML_channels], dim=-1)

        # prepare the GMRES solver:
        Aop = lambda x: r2c(self.residual_fn(c2r(x), eps[...,0], src, self.pmls, self.dL, self.wl, batched_compute=True, Aop=True))
        residual_fn = lambda x: r2c(self.residual_fn(c2r(x), eps[...,0], src, self.pmls, self.dL, self.wl, batched_compute=True, Aop=False))
        gmres = mygmrestorch(self.model, Aop, tol=self.tol, max_iter=self.max_iter)

        complex_rhs = r2c(src2rhs(src, dL, wl))
        freq = torch.tensor(self.dL/self.wl)[None].cuda()
        gmres.setup_eps(eps, freq)
        if self.restart == 0:
            x, history, _, _ = gmres.solve(complex_rhs, self.verbose)
        else:
            x, history = gmres.solve_with_restart(complex_rhs, self.tol, self.max_iter, self.restart, self.verbose)
        # final_residual = self.residual_fn(x)

        return x

@torch.no_grad()
def NN_solve(config, eps, src):
    model_path = config["model_path"]

    sim_shape = config["sim_shape"]
    wl = float(config["wavelength"])
    dL = float(config["dL"])
    pmls = config["pmls"]

    max_iter = int(config["max_iter"])
    tol = float(config["tol"])
    verbose = config["verbose"]
    restart = int(config["restart"])

    ########## first parse the gin files, which contains the model configurations ##########
    sys.path.append(model_path)
    for file in os.listdir(model_path):
        if file.endswith(".gin"):
            gin.parse_config_file(os.path.join(model_path, file))
    
    # load the model
    from waveynet3d.models import model_factory as model_fn
    model = prepare_model(sim_shape, model_path, model_fn)

    # use the dummy trainer and ds to reproduce the feature engineering for eps (this part should be rewritten to be cleaner)
    from waveynet3d.data.simulation_dataset import SyntheticDataset_same_wl_dL_shape as dataset_fn
    from waveynet3d.trainers.iterative_trainer import IterativeTrainer as trainer_fn
    dummy_trainer = trainer_fn(model_config=None, model_saving_path=None)
    dummy_ds = dataset_fn(dummy_trainer.domain_sizes, dummy_trainer.pml_ranges, residual_type=dummy_trainer.residual_type)
    check_data_distribution(eps, pmls, wl, dL, dummy_trainer, dummy_ds)

    dummy_ds.set_ln_R(dummy_trainer.ln_R)
    print(f"NN solver uses ln_R (parameter for PML): {dummy_trainer.ln_R}")

    eps, _ = dummy_ds.build_complex_eps(eps[0], wl, dL, sim_shape, pml=pmls) # add more channels to eps, which contains pml features
    eps = eps[None]

    eps = eps.cuda()
    src = src.cuda()

    # prepare the GMRES solver:
    Aop = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL, wl, batched_compute=True, Aop=True))
    residual_fn = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL, wl, batched_compute=True, Aop=False))
    gmres = mygmrestorch(model, Aop, tol=tol, max_iter=max_iter)

    # solve the problem:
    time_start = time.time()
    complex_rhs = r2c(src2rhs(src, dL, wl))
    freq = torch.tensor(dL/wl)[None].cuda()
    gmres.setup_eps(eps, freq)
    if restart == 0:
        x, history, _, _ = gmres.solve(complex_rhs, verbose)
    else:
        x, history = gmres.solve_with_restart(complex_rhs, tol, max_iter, restart, verbose)
    time_end = time.time()
    print(f"time taken for NN GMRES solver: {time_end - time_start} seconds")
    final_residual = residual_fn(x)

    return x, history, final_residual