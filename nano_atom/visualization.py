import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.plot_field3D import plot_3slices

def visualize_eps(eps, out_dir=None, prefix="nano_atom"):
    """
    Visualize permittivity tensor and save plots.
    
    Args:
        eps: Permittivity tensor
        out_dir: Output directory path (optional)
        prefix: Filename prefix (default "nano_atom")
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Epsilon visualization
    eps_vol = eps.squeeze(0)
    if torch.is_complex(eps_vol):
        eps_real = eps_vol.real.cpu().numpy()
        eps_imag = eps_vol.imag.cpu().numpy()
    else:
        eps_real = eps_vol.cpu().numpy()
        eps_imag = None

    plot_3slices(
        eps_real,
        fname=str(out_dir / f"{prefix}_eps_real.png"),
        my_cmap=plt.cm.viridis,
        cm_zero_center=False,
        title="Re{ε}",
    )
    if eps_imag is not None and np.max(np.abs(eps_imag)) > 1e-6:
        plot_3slices(
            eps_imag,
            fname=str(out_dir / f"{prefix}_eps_imag.png"),
            my_cmap=plt.cm.seismic,
            cm_zero_center=True,
            title="Im{ε}",
        )

def visualize_src(src, out_dir=None, prefix="nano_atom"):
    """
    Visualize source tensor and save plots.
    
    Args:
        src: Source tensor
        out_dir: Output directory path (optional)
        prefix: Filename prefix (default "nano_atom")
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

     # Source visualization
    src_vol = src.squeeze(0).cpu().numpy()
    channel_names = ["Ex_r", "Ex_i", "Ey_r", "Ey_i", "Ez_r", "Ez_i"]
    for idx, name in enumerate(channel_names):
        chan = src_vol[..., idx]
        if np.max(np.abs(chan)) <= 1e-6:
            continue
        plot_3slices(
            chan,
            fname=str(out_dir / f"{prefix}_src_{name}.png"),
            my_cmap=plt.cm.seismic,
            cm_zero_center=True,
            title=f"{name} source",
        )

    src_amp = np.linalg.norm(src_vol, axis=-1)
    if np.max(src_amp) > 0:
        plot_3slices(
            src_amp,
            fname=str(out_dir / f"{prefix}_src_amp.png"),
            my_cmap=plt.cm.magma,
            cm_zero_center=False,
            title="Source amplitude",
        )

def visualize_eps_src(eps, src, out_dir=None, prefix="nano_atom"):
    """
    Visualize both permittivity and source tensors.
    
    Args:
        eps: Permittivity tensor
        src: Source tensor
        out_dir: Output directory path (optional)
        prefix: Filename prefix (default "nano_atom")
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    visualize_eps(eps, out_dir, prefix)
    visualize_src(src, out_dir, prefix)