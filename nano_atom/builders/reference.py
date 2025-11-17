import torch
import numpy as np
from src.utils.utils import get_pixels
from nano_atom.source import set_source_plane

def set_reference_cell(sim_shape, wl, dL, pmls, kwargs):
    """
    Reference meta-cell: substrate + top medium.
    
    Args:
        sim_shape: Simulation shape tuple (sx, sy, sz)
        wl: Wavelength in nm
        dL: Grid size in nm/pixel
        pmls: PML configuration list
        kwargs: Dictionary with keys:
            - substrate_material, top_medium_material (material names)
            - source_polarization: polarization string (default 'x')
            - substrate_thickness_nm, source_below_meta_nm (in nm)
    
    Returns:
        Tuple of (eps, src) tensors with batch dimension
    """
    assert len(sim_shape) == 3
    sx, sy, sz = sim_shape

    from nano_atom.material import get_n
    substrate_material = kwargs.get('substrate_material')
    top_medium_material = kwargs.get('top_medium_material')
    if substrate_material is None:
        raise ValueError("substrate_material is required but not provided")
    if top_medium_material is None:
        raise ValueError("top_medium_material is required but not provided")
    substrate_eps = get_n(substrate_material, wl * 1e-3) ** 2
    top_medium_eps = get_n(top_medium_material, wl * 1e-3) ** 2
    source_polarization = kwargs.get('source_polarization', 'x')

    z_sub      = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    src_below  = get_pixels(kwargs, 'source_below_meta_nm', dL)
    assert z_sub > src_below, "Source must lie inside the substrate."

    eps = top_medium_eps * torch.ones(sim_shape, dtype=torch.float32)
    eps[:, :, :z_sub] = substrate_eps

    src = torch.zeros((*sim_shape, 6), dtype=torch.float32)
    z_src = z_sub - src_below
    src = set_source_plane(src, z_src, pol=source_polarization, amp=1.0, direction='forward', dL=dL, wavelength=wl, n=np.sqrt(substrate_eps))

    return eps[None], src[None]
