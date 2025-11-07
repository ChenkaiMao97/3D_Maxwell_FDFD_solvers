import gin
from clize import run
import os

from bin.inverse_design import run_inverse_design
from bin.run_job_utils import ExperimentConfig, set_up_experiment

@gin.configurable
def run_job(
    pipeline: str,
    experiment_name: str,
    experiment_dir: str=None,
    design_config: str=None,
    solver_config: str=None,
    gpu_ids: str="0,1"
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    unique_exp_name, base_dir, log_dir, code_dir, output_dir = set_up_experiment(
        name=experiment_name,
        base_dir=experiment_dir,
    )

    config = ExperimentConfig(
        design_config=design_config,
        solver_config=solver_config,
        experiment_dir=base_dir,
        experiment_name=unique_exp_name,
        base_dir=base_dir,
        log_dir=log_dir,
        code_dir=code_dir,
        output_dir=output_dir,
    )
    if  pipeline == "inverse_design":
        job_fn = run_inverse_design
    else:
        raise ValueError(f"Pipeline {pipeline} not found")

    job_fn(config)

def main(*,
    experiment_name: str,
    design_config: str=None,
    solver_config: str=None,
    pipeline: str):

    if design_config is not None:
        gin.parse_config_file(design_config)
    if solver_config is not None:
        gin.parse_config_file(solver_config)

    run_job(
        pipeline=pipeline,
        experiment_name=experiment_name,
        design_config=design_config,
        solver_config=solver_config,
    )


if __name__ == "__main__":
    run(main)