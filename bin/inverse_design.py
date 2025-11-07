import logging
import os
import shutil

import gin
import jax
import numpy as onp

from bin.run_job_utils import ExperimentConfig, seeding

from src.invde.opt import Designer
from src.invde.utils.utils import get_integrated_photonics_challenge, get_metasurface_challenge, get_coupling_challenge

design_schemes = ["integrated_photonics", 'metasurface', 'coupling']

@gin.configurable
def run_inverse_design(
        config: ExperimentConfig,
        design_challenge,
        seed_ids=0
    ):
    shutil.copy(config.design_config, config.base_dir)
    shutil.copy(config.solver_config, config.base_dir)
    if config.solver_config is not None:
        shutil.copy(config.solver_config, config.base_dir)

    if design_challenge not in design_schemes:
        raise ValueError(f"Design challenge {design_challenge} not found")

    key = jax.random.PRNGKey(seeding([seed_ids]))

    design_scheme_kwargs = {}
    if design_challenge == 'integrated_photonics':
        design_scheme_kwargs = {
                "challenge": get_integrated_photonics_challenge(key=key)
            }
    elif design_challenge == 'metasurface':
        design_scheme_kwargs = {
                "challenge": get_metasurface_challenge(key=key)
            }
    elif design_challenge == 'coupling':
        design_scheme_kwargs = {
                "challenge": get_coupling_challenge(key=key)
            }
    designer = Designer(log_dir=config.log_dir, **design_scheme_kwargs)
    # build the design
    designer.init(key=key)
    results = designer.optimize()
    designer.stop_workers()

    input_eps, E, source, wls, dLs, state = results

    os.makedirs(config.output_dir, exist_ok=True)
    onp.save(os.path.join(config.output_dir, "input_eps.npy"), input_eps)
    onp.save(os.path.join(config.output_dir, "Ez.npy"), E[..., 2])
    onp.save(os.path.join(config.output_dir, "source.npy"), source)
    onp.save(os.path.join(config.output_dir, "wls.npy"), wls)
    onp.save(os.path.join(config.output_dir, "dLs.npy"), dLs)
    onp.save(os.path.join(config.output_dir, "loss_curve.npy"), state.loss)

if __name__ == "__main__":
    run_inverse_design()
