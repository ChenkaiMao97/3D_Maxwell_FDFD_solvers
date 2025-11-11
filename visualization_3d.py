import os
from mayavi import mlab
from tvtk.api import tvtk

import numpy as np
import torch
from src.utils.utils import c2r, r2c

def mayavi_em_volumes(
    eps,                        # (nx, ny, nz) float
    E,                          # (nx, ny, nz, 3) real or complex
    source=None,                # (nx,ny,nz) or (nx,ny,nz,3), real or complex
    spacing=(1.0, 1.0, 1.0),    # (dx, dy, dz)
    # Appearance
    cmap_eps="Blues",
    cmap_E="YlOrRd",
    cmap_src="Purples",
    # Percentile clipping for robust range
    pct_E=99.5,
    pct_src=99.5,
    pct_eps=99.5,
    # Opacity curve control (value → opacity)
    # points are relative to [0, vmax] after clipping
    opacity_points=((0.00, 0.00), (0.10, 0.02), (0.40, 0.08), (0.70, 0.20), (1.00, 0.45)),
    # Output
    output=None,                # e.g. "em_viz.png" to save, or None to show
    figure_size=(1000, 800),
    bgcolor=(1,1,1),
):
    """
    Render eps, |E|, and source as volumes with value-dependent transparency.
    If `output` is not None, saves an image off-screen; otherwise opens an interactive window.
    """
    # ---- Prep scalars ----
    print("1")
    Emag = np.linalg.norm(E if np.isrealobj(E) else np.abs(E), axis=-1)

    src_mag = None
    if source is not None:
        if source.ndim == 4:  # vector
            src_mag = np.linalg.norm(source if np.isrealobj(source) else np.abs(source), axis=-1)
        else:                 # scalar
            src_mag = source if np.isrealobj(source) else np.abs(source)

    def _add_volume(data, cmap, spacing, pct=99.5,
                opacity_points=((0.00,0.00),(0.10,0.02),(0.40,0.08),(0.70,0.20),(1.00,0.45)),
                low_texture=True):
        vmax = float(np.percentile(data, pct))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = float(np.max(data)) if np.max(data) > 0 else 1.0
        vmin = 0.0

        src = mlab.pipeline.scalar_field(data)
        try:
            src.spacing = spacing
        except Exception:
            src.image_data.spacing = spacing

        vol = mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)

        # Colormap / data range
        vol.module_manager.scalar_lut_manager.lut_mode = cmap
        vol.module_manager.scalar_lut_manager.data_range = (vmin, vmax)

        # Opacity TF
        pwf = tvtk.PiecewiseFunction()
        for x_rel, a in opacity_points:
            pwf.add_point(vmin + x_rel*(vmax - vmin), a)
        vp = vol._volume_property
        vp.set_scalar_opacity(pwf)

        if low_texture:
            # Disable shading & gradient opacity (saves textures)
            vp.shade = False
            empty = tvtk.PiecewiseFunction()
            empty.add_point(0.0, 0.0)
            vp.set_gradient_opacity(empty)

            # Try simpler mapper mode (names are tvtk enums; ignore if not present)
            mapper = getattr(vol, "_volume_mapper", None)
            if mapper is not None:
                try:
                    # Prefer CPU ray cast when GPU runs out of texture units
                    mapper.requested_render_mode = 'ray_cast'            # or 'texture'
                except Exception:
                    pass
                try:
                    mapper.blend_mode = 'composite'                     # safest blending
                except Exception:
                    pass
        else:
            # Mild shading if you keep it on
            vp.shade = True
            gpf = tvtk.PiecewiseFunction()
            gpf.add_point(0.0, 0.0); gpf.add_point((vmax-vmin)*0.1, 0.05); gpf.add_point((vmax-vmin)*0.3, 0.15)
            vp.set_gradient_opacity(gpf)

        return vol

    # ---- Figure ----
    # if output:
        # mlab.options.offscreen = True
    
    print("7")
    fig = mlab.figure(size=figure_size, bgcolor=bgcolor)

    # Order: field (glow) → dielectric (outline) → source (accent)
    print("8")
    vE   = _add_volume(Emag,  spacing=spacing, cmap=cmap_E,   pct=pct_E)
    vEPS = _add_volume(eps,    spacing=spacing, cmap=cmap_eps, pct=pct_eps)
    if src_mag is not None:
        vSRC = _add_volume(src_mag, spacing=spacing, cmap=cmap_src, pct=pct_src)
    
    print("6")

    # Axes & view
    mlab.orientation_axes()
    mlab.view(azimuth=40, elevation=70, distance='auto', focalpoint='auto')

    if output:
        mlab.savefig(output)   # e.g. "em_viz.png"
        mlab.close(fig)
    else:
        mlab.show()

# Example usage:
# visualize_em(eps, E, spacing=(0.05,0.05,0.05), eps_isovalue=2.0, source_axis='z', source_index=20)

if __name__ == "__main__":
    output_path = './'
    eps = np.load(os.path.join(output_path, 'eps.npy'))[0]
    src = r2c(torch.from_numpy(np.load(os.path.join(output_path, 'src.npy'))))[0].numpy()
    solution = np.load(os.path.join(output_path, 'solution.npy'))[0]
    print("shapes: ", eps.shape, src.shape, solution.shape)
    mayavi_em_volumes(eps, solution, source=src, spacing=(0.05,0.05,0.05), output=os.path.join(output_path, 'em_viz.png'))