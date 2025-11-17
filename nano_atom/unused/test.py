import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to find src module
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.utils.utils import get_pixels
import yaml

# Load config file relative to script location
config_path = current_dir / "configs" / "nanofin_g.yaml"
config = yaml.safe_load(open(config_path, "r"))
from nano_atom.solve import solve
from nano_atom.builders.reference import set_reference_cell
from nano_atom.analysis import poynting_flux, _field_to_complex

WVLN = config["wavelength"]
DL = config["dL"]

sx, sy, sz = config["sim_shape"]


# Solve the problem with uniform substrate using reference builder
print("Solving problem with uniform substrate...")
results = solve(
    builder=set_reference_cell,
    config_path=str(config_path),
    result_path=None,  # Don't save results
    plot=False
)
solution_field = results["solution"]  # Complex tensor (bs, sx, sy, sz, 3)
print(f"Solved problem. Residual: {results['metrics']['nn_residual_mean_abs']:.4e}")

z_sub      = get_pixels(config["kwargs"], 'substrate_thickness_nm', config["dL"])
src_below  = get_pixels(config["kwargs"], 'source_below_meta_nm', config["dL"])
z_src = z_sub - src_below
print(f"Source plane index: {z_src}")

slice_above_source = z_src + 10
slice_below_source = z_src - 10

# Extract full 3D field components first
def extract_full_3d_field(field):
    """
    Extract full 3D Ex, Ey, Ez from field tensor.
    
    Args:
        field: Field tensor with shape (sx, sy, sz, 6|3) or (bs, sx, sy, sz, 6|3)
    
    Returns:
        Tuple of (Ex_3d, Ey_3d, Ez_3d) complex tensors
    """
    # Remove batch dim if present
    if field.ndim == 5:
        field = field.squeeze(0)
    
    # Real-valued [sx, sy, sz, 6]
    if field.shape[-1] == 6:
        Ex_3d = field[..., 0] + 1j * field[..., 1]
        Ey_3d = field[..., 2] + 1j * field[..., 3]
        Ez_3d = field[..., 4] + 1j * field[..., 5]
    elif field.shape[-1] == 3:
        # Complex [sx, sy, sz, 3]
        Ex_3d = field[..., 0]
        Ey_3d = field[..., 1]
        Ez_3d = field[..., 2]
    else:
        raise ValueError("Field must have last dimension 6 (real) or 3 (complex)")
    return Ex_3d, Ey_3d, Ez_3d

# Extract full 3D field
try:
    Ex_3d, Ey_3d, Ez_3d = extract_full_3d_field(solution_field)
    print(f"Extracted 3D field shapes: Ex={Ex_3d.shape}, Ey={Ey_3d.shape}, Ez={Ez_3d.shape}")
except Exception as e:
    print(f"Error extracting 3D field: {e}")
    exit(1)

# Determine number of z slices
nz = Ex_3d.shape[2]
dx = config["dL"] * 1e-9
dy = config["dL"] * 1e-9

flux_sums = []
for z_index in range(nz):
    E_3d = _field_to_complex(solution_field)
    try:
        # Compute Poynting flux for this z-plane using full 3D field
        flux_sum = float(poynting_flux(solution_field, DL, WVLN, 1.0, z_index))
        flux_sums.append(flux_sum)
        if z_index % 20 == 0:  # Print progress every 20 slices
            print(f"Processed z={z_index}, flux_sum={flux_sum:.6f}")
    except Exception as e:
        print(f"Skipping z={z_index} due to error: {e}")
        flux_sums.append(0.0)

plt.figure()
plt.plot(flux_sums)
plt.xlabel("z")
plt.ylabel("Power")
plt.title("Power vs z")
plt.savefig(f"power_vs_z.png")
plt.close()

z_target = 34
if 0 <= z_target < len(flux_sums):
    print(f"Poynting flux at z={z_target}: {flux_sums[z_target]:.3e}")
else:
    print(f"z={z_target} out of range (0 to {len(flux_sums)-1})")

P_DENSITY = flux_sums[z_target] / (sx * sy * dx * dy)
print(f"Poynting flux density at z={z_target}: {P_DENSITY:.7e}")