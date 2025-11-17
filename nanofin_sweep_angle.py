#!/usr/bin/env python3
"""
Nanofin theta angle sweep script.
Sweeps the fin_angle_deg parameter from 0 to 360 degrees in 10-degree steps.
Uses the same settings as nanofin.yaml configuration file.
"""

import copy
import time
from pathlib import Path

import torch
import yaml
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
from matplotlib import pyplot as plt

from nano_atom.solve import solve
from nano_atom.builders.nanofin import set_nanofin
from src.solvers.NN_solver import NN_solve
from src.utils.plot_field3D import plot_3slices
from src.solvers.spins_solver import spins_solve
from src.utils.utils import c2r, scaled_MAE


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _plot_solution_fields(solution, residual, eps, out_dir):
    intensity = torch.sum(torch.abs(solution[0]) ** 2, dim=-1).detach().cpu().numpy()
    eps_np = eps[0].detach().cpu().numpy().real

    plot_3slices(eps_np, fname=str(out_dir / "eps.png"), my_cmap=plt.cm.binary, cm_zero_center=False)

    for idx, name in enumerate(("x", "y", "z")):
        comp = solution[0, :, :, :, idx].detach().cpu()
        plot_3slices(comp.real.numpy(), fname=str(out_dir / f"solution_{name}r.png"), my_cmap=plt.cm.seismic)
        plot_3slices(comp.imag.numpy(), fname=str(out_dir / f"solution_{name}i.png"), my_cmap=plt.cm.seismic)

    plot_3slices(intensity, fname=str(out_dir / "intensity.png"), my_cmap=plt.cm.seismic)
    plot_3slices(residual[0, :, :, :, 0].detach().cpu().numpy().real, fname=str(out_dir / "residual_xr.png"), my_cmap=plt.cm.seismic)


def run_simulation_with_params(builder, sim_shape, wl, dL, pmls, kwargs, config, result_path, plot=False):
    eps, src = builder(sim_shape, wl, dL, pmls, kwargs)

    out_dir = None
    if result_path is not None:
        out_dir = Path(result_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(eps.detach().cpu(), out_dir / "eps.pt")
        torch.save(src.detach().cpu(), out_dir / "src.pt")

    start_time = time.time()
    solution, residual_history, final_residual = NN_solve(config, eps.clone(), src.clone())
    solve_time = time.time() - start_time

    if out_dir is not None:
        torch.save(solution.detach().cpu(), out_dir / "solution.pt")
        torch.save(final_residual.detach().cpu(), out_dir / "final_residual.pt")

    spins_data = {}
    if config.get("spins_verification"):
        spins_solution, spins_residual = spins_solve(config, eps, src)
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


def run_angle_sweep(config_path, output_base_dir="nanofin_sweep_results"):
    print("=" * 60)
    print("NANOFIN THETA ANGLE SWEEP")
    print("=" * 60)
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    sim_shape = tuple(config["sim_shape"])
    wl = float(config["wavelength"])
    dL = float(config["dL"])
    pmls = config["pmls"]
    kwargs = copy.deepcopy(config.get("kwargs", {}))
    
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    angles = np.arange(0, 180, 5)
    total_angles = len(angles)
    
    print(f"Simulation shape: {sim_shape}")
    print(f"Wavelength: {wl} nm")
    print(f"Grid size: {dL} nm/pixel")
    print(f"Number of angles to sweep: {total_angles}")
    print(f"Output directory: {output_dir}")
    print()
    
    results_summary = []
    start_time_total = time.time()
    
    for i, angle in enumerate(angles):
        print(f"Angle {i+1}/{total_angles}: {angle}°")
        
        kwargs_angle = copy.deepcopy(kwargs)
        kwargs_angle['fin_angle_deg'] = float(angle) if angle is not None else 0.0
        
        angle_dir = output_dir / f"angle_{angle:03d}deg"
        
        try:
            start_time = time.time()
            results = run_simulation_with_params(
                builder=set_nanofin,
                sim_shape=sim_shape,
                wl=wl,
                dL=dL,
                pmls=pmls,
                kwargs=kwargs_angle,
                config=config,
                result_path=str(angle_dir),
                plot=False
            )
            solve_time = time.time() - start_time
            
            residual = results['metrics']['nn_residual_mean_abs']
            iterations = results['metrics']['nn_iters']
            
            print(f"Completed in {solve_time:.2f}s")
            print(f"Residual: {residual:.4e}")
            print(f"Iterations: {iterations}")
            
            results_summary.append({
                'angle_deg': angle,
                'residual': residual,
                'iterations': iterations,
                'solve_time_sec': solve_time,
                'output_dir': str(angle_dir)
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results_summary.append({
                'angle_deg': angle,
                'error': str(e),
                'output_dir': str(angle_dir)
            })
        
        print()
    
    total_time = time.time() - start_time_total
    
    summary_file = output_dir / "sweep_summary.yaml"
    with open(summary_file, 'w') as f:
        converted_results = convert_numpy_types(results_summary)
        yaml.safe_dump(converted_results, f, default_flow_style=False)
    
    print("=" * 60)
    print("SWEEP COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per angle: {total_time/total_angles:.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    successful = [r for r in results_summary if 'error' not in r]
    failed = [r for r in results_summary if 'error' in r]
    
    print(f"Successful runs: {len(successful)}/{total_angles}")
    if failed:
        print(f"Failed runs: {len(failed)}")
        for fail in failed:
            print(f"  - Angle {fail['angle_deg']}°: {fail['error']}")
    
    if successful:
        residuals = [r['residual'] for r in successful]
        print(f"Residual range: {min(residuals):.4e} - {max(residuals):.4e}")
        print(f"Mean residual: {np.mean(residuals):.4e}")
    
    return results_summary


def main():
    config_path = "nano_atom/configs/nanofin_g.yaml"
    output_path = "nano_atom/results/angle_sweep_forward"
    
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        return
    
    results = run_angle_sweep(config_path, output_path)
    
    print("\nSweep completed successfully!")


if __name__ == "__main__":
    main()
