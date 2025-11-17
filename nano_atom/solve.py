import argparse
import copy
import time
from pathlib import Path
from typing import Callable

import torch
import yaml
from matplotlib import pyplot as plt

from src.solvers.NN_solver import NN_solve
from src.utils.plot_field3D import plot_3slices
from src.solvers.spins_solver import spins_solve
from src.utils.utils import c2r, scaled_MAE

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs/nanofin.yaml"

GeometryBuilder = Callable[[tuple, float, float, list, dict], tuple[torch.Tensor, torch.Tensor]]


def _plot_solution_fields(solution, residual, eps, out_dir):
    """
    Plot solution fields and save to output directory.
    
    Args:
        solution: Solution field tensor
        residual: Residual tensor
        eps: Permittivity tensor
        out_dir: Output directory path
    """
    intensity = torch.sum(torch.abs(solution[0]) ** 2, dim=-1).detach().cpu().numpy()
    eps_np = eps[0].detach().cpu().numpy().real

    plot_3slices(eps_np, fname=str(out_dir / "eps.png"), my_cmap=plt.cm.binary, cm_zero_center=False)

    for idx, name in enumerate(("x", "y", "z")):
        comp = solution[0, :, :, :, idx].detach().cpu()
        plot_3slices(comp.real.numpy(), fname=str(out_dir / f"solution_{name}r.png"), my_cmap=plt.cm.seismic)
        plot_3slices(comp.imag.numpy(), fname=str(out_dir / f"solution_{name}i.png"), my_cmap=plt.cm.seismic)

    plot_3slices(intensity, fname=str(out_dir / "intensity.png"), my_cmap=plt.cm.seismic)
    plot_3slices(residual[0, :, :, :, 0].detach().cpu().numpy().real, fname=str(out_dir / "residual_xr.png"), my_cmap=plt.cm.seismic)


def solve(builder, config_path, result_path=None, plot=False):
    """
    Build a nano-atom problem, run the neural GMRES solver, and optionally persist artifacts.

    Args:
        builder: Geometry builder function that takes (sim_shape, wl, dL, pmls, kwargs) and returns (eps, src)
        config_path: Path to the configuration YAML file
        result_path: Optional path to save results. If None, no results are saved
        plot: Whether to generate plots

    Returns:
        Dictionary containing the solution tensors, residuals, metrics, and optional paths
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    
    sim_shape = tuple(cfg["sim_shape"])
    wl = float(cfg["wavelength"])
    dL = float(cfg["dL"])
    pmls = cfg["pmls"]
    kwargs = copy.deepcopy(cfg.get("kwargs", {}))

    eps, src = builder(sim_shape, wl, dL, pmls, kwargs)

    out_dir: Path | None = None
    if result_path is not None:
        out_dir = Path(result_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(eps.detach().cpu(), out_dir / "eps.pt")
        torch.save(src.detach().cpu(), out_dir / "src.pt")

    start_time = time.time()
    solution, residual_history, final_residual = NN_solve(cfg, eps.clone(), src.clone())
    solve_time = time.time() - start_time

    if out_dir is not None:
        torch.save(solution.detach().cpu(), out_dir / "solution.pt")
        torch.save(final_residual.detach().cpu(), out_dir / "final_residual.pt")

    spins_data = {}
    if cfg.get("spins_verification"):
        spins_solution, spins_residual = spins_solve(cfg, eps, src)
        rel_diff, _ = scaled_MAE(c2r(solution).cpu(), spins_solution)
        spins_data = {
            "relative_error": float(rel_diff),
            "spins_solution": spins_solution,
            "spins_residual": spins_residual,
        }
        if out_dir is not None:
            torch.save(spins_solution, out_dir / "spins_solution.pt")
            torch.save(spins_residual, out_dir / "spins_residual.pt")

    metrics = {
        "nn_residual_mean_abs": float(torch.mean(torch.abs(final_residual)).item()),
        "nn_iters": len(residual_history),
        "nn_solve_time_sec": solve_time,
    }
    metrics.update({k: v for k, v in spins_data.items() if not isinstance(v, torch.Tensor)})

    if out_dir is not None:
        if plot:
            _plot_solution_fields(solution, final_residual, eps, out_dir)
        with (out_dir / "metrics.yaml").open("w") as f:
            yaml.safe_dump(metrics, f)
        metrics["output_dir"] = str(out_dir)

    return {
        "eps": eps,
        "src": src,
        "solution": solution,
        "final_residual": final_residual,
        "residual_history": residual_history,
        "metrics": metrics,
        "spins": spins_data,
    }


def _parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Run nano-atom simulations.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to a YAML config file (default: configs/nanofin.yaml).",
    )
    parser.add_argument(
        "--result-path",
        help="Path to save results. If not provided, no results are saved.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of the solution fields.",
    )
    return parser.parse_args()