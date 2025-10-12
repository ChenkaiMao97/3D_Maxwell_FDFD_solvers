import sys, os
from src.gmres import mygmrestorch
from src.physics import residue_E, src2rhs
from src.utils import *
import gin

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
    print("The training data shape is between ", f"({dummy_trainer.domain_sizes[0]}, {dummy_trainer.domain_sizes[2]}, {dummy_trainer.domain_sizes[4]})", "and ", \
                                                 f"({dummy_trainer.domain_sizes[1]}, {dummy_trainer.domain_sizes[3]}, {dummy_trainer.domain_sizes[5]})")
    print("Current problem shape: ", eps.shape[1:4])

    dummy_ds = dataset_fn(dummy_trainer.domain_sizes, dummy_trainer.pml_ranges, residual_type=dummy_trainer.residual_type)
    dummy_ds.set_ln_R(dummy_trainer.ln_R)
    eps, _ = dummy_ds.build_complex_eps(eps[0], wl, dL, sim_shape, pml=pmls) # add more channels to eps, which contains pml features
    eps = eps[None]

    eps = eps.cuda()
    src = src.cuda()

    # prepare the GMRES solver:
    Aop = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL, wl, batched_compute=True, Aop=True))
    residual_fn = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL, wl, batched_compute=True, Aop=False))
    gmres = mygmrestorch(model, Aop, tol=tol, max_iter=max_iter)

    # solve the problem:
    complex_rhs = r2c(src2rhs(src, dL, wl))
    freq = torch.tensor(dL/wl)[None].cuda()
    gmres.setup_eps(eps, freq)
    if restart == 0:
        x, history, _, _ = gmres.solve(complex_rhs, verbose)
    else:
        x, history = gmres.solve_with_restart(complex_rhs, tol, max_iter, restart, verbose)
    final_residual = residual_fn(x)

    return x, history, final_residual