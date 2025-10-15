import sys, os
import numpy as np
import torch
from src.gmres import mygmrestorch
from src.physics import residue_E, src2rhs
from src.utils import *
import time
import gin

@torch.no_grad()
def NN_random_solve(config):
    model_path = config["model_path"]

    max_iter = int(config["max_iter"])
    tol = float(config["tol"])
    verbose = config["verbose"]
    restart = int(config["restart"])

    ########## first parse the gin files, which contains the model configurations ##########
    sys.path.append(model_path)
    for file in os.listdir(model_path):
        if file.endswith(".gin"):
            gin.parse_config_file(os.path.join(model_path, file))

    # use the dummy trainer and ds to reproduce the feature engineering for eps (this part should be rewritten to be cleaner)
    from waveynet3d.data.simulation_dataset import SyntheticDataset_same_wl_dL_shape as dataset_fn
    from waveynet3d.trainers.iterative_trainer import IterativeTrainer as trainer_fn
    # from waveynet3d.trainers.distillation_trainer import DistillationTrainer as trainer_fn
    dummy_trainer = trainer_fn(model_config=None, model_saving_path=None)

    dummy_ds = dataset_fn(dummy_trainer.domain_sizes, dummy_trainer.pml_ranges, residual_type=dummy_trainer.residual_type)
    dummy_ds.set_ln_R(dummy_trainer.ln_R)

    # load the model
    from waveynet3d.models import model_factory as model_fn
    model = prepare_model(dummy_trainer.domain_sizes, model_path, model_fn)
    
    ds_loader = torch.utils.data.DataLoader(
        dataset=dummy_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dummy_ds.collate_fn_same_wl_dL
    )
    ith_data = np.random.randint(0, 10)
    ds_iter = iter(ds_loader)
    for i in range(ith_data):
        sample = next(ds_iter)
    eps, src, dL, wl, pmls = sample['eps'].cuda(), sample['source'].cuda(), sample['dL'], sample['wl'], sample['pmls']

    print("The training data shape is between ", f"({dummy_trainer.domain_sizes[0]}, {dummy_trainer.domain_sizes[2]}, {dummy_trainer.domain_sizes[4]})", "and ", \
                                                 f"({dummy_trainer.domain_sizes[1]}, {dummy_trainer.domain_sizes[3]}, {dummy_trainer.domain_sizes[5]})")
    print("Current problem shape: ", eps.shape[1:4])

    # prepare the GMRES solver:
    Aop = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL[0].numpy(), wl[0].numpy(), batched_compute=True, Aop=True))
    residual_fn = lambda x: r2c(residue_E(c2r(x), eps[...,0], src, pmls, dL[0].numpy(), wl[0].numpy(), batched_compute=True, Aop=False))
    gmres = mygmrestorch(model, Aop, tol=tol, max_iter=max_iter)

    # solve the problem:
    time_start = time.time()
    complex_rhs = r2c(src2rhs(src, dL, wl))
    gmres.setup_eps(eps, dL/wl)
    if restart == 0:
        x, history, _, _ = gmres.solve(complex_rhs, verbose)
    else:
        x, history = gmres.solve_with_restart(complex_rhs, tol, max_iter, restart, verbose)
    time_end = time.time()
    print(f"time taken for NN GMRES solver: {time_end - time_start} seconds")
    final_residual = residual_fn(x)

    return x, history, final_residual, eps, src, dL, wl, pmls