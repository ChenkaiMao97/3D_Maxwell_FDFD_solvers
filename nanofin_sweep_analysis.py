#!/usr/bin/env python3
"""
Analyze nanofin sweep results by plotting transmittance, polarization conversion efficiency,
and far-field phase as functions of fin angle.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from nano_atom.analysis import (
    compute_far_field_phase,
    compute_polarization_conversion,
    compute_transmittance,
)

from nano_atom.material import get_n


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize nanofin sweep metrics vs. rotation angle.")
    parser.add_argument(
        "--results-dir",
        type=Path, 
        default=Path("nano_atom/results/angle_sweep_forward"),
        help="Directory containing per-angle simulation folders.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("nano_atom/configs/nanofin_g.yaml"),
        help="Configuration file that specifies wavelength, voxel size, PML thickness, etc.",
    )
    parser.add_argument(
        "--plane-index",
        type=int,
        default=None,
        help="Optional z-plane on the transmission side; inferred from config when omitted.",
    )
    parser.add_argument(
        "--plane-offset",
        type=int,
        default=5,
        help="How many voxels to back off from the start of the output-side PML when inferring plane index.",
    )
    parser.add_argument(
        "--device-polarization",
        type=str,
        default="lcp",
        help="Polarization used for conversion efficiency and phase projections.",
    )
    parser.add_argument(
        "--reference-polarization",
        type=str,
        default='rcp',
        help="Optional polarization for co-polarized efficiency reporting.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="none",
        choices=("none", "hann"),
        help="Spatial windowing to apply before the far-field FFT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the plot (defaults to <results-dir>/nanofin_sweep_analysis.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args()


def load_config(path):
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def infer_plane_index(sim_shape, pmls, offset):
    nz = int(sim_shape[2])
    plane = nz - int(pmls[5]) - offset
    return max(0, min(nz - 1, plane))


def discover_angle_dirs(results_dir):
    pattern = re.compile(r"angle_(?P<angle>[-+]?\d+(?:\.\d*)?)deg")
    angle_dirs = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if match:
            angle_dirs.append((float(match.group("angle")), entry))
    if not angle_dirs:
        raise RuntimeError(f"No angle_*deg subdirectories found in {results_dir}")
    angle_dirs.sort(key=lambda pair: pair[0])
    return angle_dirs


def load_tensor(path):
    if not path.exists():
        raise FileNotFoundError(f"Expected tensor at {path}")
    return torch.load(path, map_location="cpu")


def get_source_plane_index(config):
    from src.utils.utils import get_pixels

    dL_nm = config["dL"]
    kwargs = config["kwargs"]
    z_sub = get_pixels(kwargs, "substrate_thickness_nm", dL_nm)
    src_below = get_pixels(kwargs, "source_below_meta_nm", dL_nm)
    return int(z_sub - src_below)


def main():
    args = parse_args()
    config = load_config(args.config)
    wvln = float(config["wavelength"])
    dL = float(config["dL"])

    substrate_material = config.get("kwargs", {}).get("substrate_material")
    top_medium_material = config.get("kwargs", {}).get("top_medium_material")
    meta_atom_material = config.get("kwargs", {}).get("meta_atom_material")
    n_substrate = get_n(substrate_material, wvln * 1e-3)
    n_top_medium = get_n(top_medium_material, wvln * 1e-3)
    n_meta_atom = get_n(meta_atom_material, wvln * 1e-3)

    sim_shape = tuple(config["sim_shape"])
    pmls = list(config["pmls"])
    plane_index = (
        args.plane_index
        if args.plane_index is not None
        else infer_plane_index(sim_shape, pmls, args.plane_offset)
    )
    print(f"Plane index: {plane_index}")

    angle_dirs = discover_angle_dirs(args.results_dir)
    src_field = load_tensor(angle_dirs[0][1] / "src.pt")
    source_plane_index = get_source_plane_index(config)

    kwargs = config.get("kwargs", {})
    angles = []
    transmittances = []
    conversions = []
    phases = []
    efficiencies = []

    inc_flux = None

    for angle_deg, folder in angle_dirs:
        solution_field = load_tensor(folder / "solution.pt")
        src_field = load_tensor(folder / "src.pt")

        t_metrics = compute_transmittance(
            solution_field,
            monitor_z_idx=plane_index,
            n_monitor=n_top_medium,
            n_source=n_substrate,
            wvln=wvln,
            dL=dL,
            inc_flux=inc_flux,
            src=src_field,
            config=config,
        )
        c_metrics = compute_polarization_conversion(
            solution_field,
            monitor_z_idx=plane_index,
            n_monitor=n_top_medium,
            pol=args.device_polarization,
            wvln=wvln,
            dL=dL,
        )
        p_metrics = compute_far_field_phase(
            solution_field,
            monitor_z_idx=plane_index,
            pol=args.device_polarization,
            window=args.window,
        )

        if inc_flux is None:
            inc_flux = t_metrics["incident_flux"]
        angles.append(angle_deg)
        transmittances.append(t_metrics["transmittance"])
        print(f"Transmittance: {t_metrics['transmittance']}")
        conversions.append(c_metrics["conversion_efficiency"])
        phases.append(p_metrics["far_field_phase_deg"])
        efficiencies.append(c_metrics["conversion_efficiency"] * t_metrics["transmittance"])

    output_path = args.output or (args.results_dir / "nanofin_sweep_analysis.png")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    axes[0].plot(angles, [v * 100 for v in transmittances], marker="o", label="Transmittance")
    axes[0].plot(angles, [v * 100 for v in conversions], marker="s", label="Conversion Efficiency")
    axes[0].plot(angles, [v * 100 for v in efficiencies], marker="^", label="Efficiency")
    axes[0].set_ylabel("Value (%)")
    axes[0].set_xlabel("Fin angle (deg)")
    axes[0].set_title("Transmittance / Polarization Conversion / Efficiency (%)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(angles, phases, marker="^", color="tab:red")
    axes[1].set_xlabel("Fin angle (deg)")
    axes[1].set_ylabel("Far-field Phase (deg)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Nanofin Frequency Domain Analysis")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved analysis plot to: {output_path}")


if __name__ == "__main__":
    main()