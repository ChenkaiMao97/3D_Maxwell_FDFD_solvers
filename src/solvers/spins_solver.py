import sys, os
from src.utils.physics import residue_E, src2rhs
from src.solvers.simulator import simulate, get_results
from src.utils.utils import *
import gin

def spins_solve(config, eps, src, dL=None, wl=None, pmls=None, ln_R=-10):
    wl = float(config["wavelength"]) if wl is None else wl
    dL = float(config["dL"]) if dL is None else dL
    pmls = config["pmls"] if pmls is None else pmls

    residual_fn = lambda x: r2c(residue_E(x, eps, src, pmls, dL, wl, batched_compute=True, Aop=False))
    
    simulate(
        wl,
        dL,
        eps[0].detach().cpu(),
        src[0].detach().cpu(),
        pmls,
        proj_folder = "spins_files/",
        output_data_folder = "spins_files/",
        ln_R = ln_R
    )
    solution = get_results(store_dir="spins_files/")[None]
    final_residual = residual_fn(solution)

    return solution, final_residual