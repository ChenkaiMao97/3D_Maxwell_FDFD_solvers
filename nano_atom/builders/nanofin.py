import torch
import numpy as np
from pathlib import Path
import yaml

from nano_atom.material import get_n
from nano_atom.utils import get_pixels
from nano_atom.source import set_source_plane

def rotated_rect(h, w, cx, cy, length, width, angle_rad, ss=8, dtype=np.float32):
    """
    Per-pixel coverage of a rotated rectangle centered at (cx, cy).
    length, width are full extents (in pixels) along the local x' and y' axes.
    angle_rad rotates the long axis CCW from +x (radians).
    
    Args:
        h: Height in pixels
        w: Width in pixels
        cx: Center x coordinate
        cy: Center y coordinate
        length: Full extent along local x' axis (pixels)
        width: Full extent along local y' axis (pixels)
        angle_rad: Rotation angle in radians (CCW from +x)
        ss: Supersampling per axis (default 8)
        dtype: Output dtype (default np.float32)
    
    Returns:
        Coverage tensor of shape (h, w) with values in [0,1]
    """
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    ofs = (np.arange(ss) + 0.5) / ss
    oy, ox = np.meshgrid(ofs, ofs, indexing="ij")  # (ss, ss)

    X = x[..., None, None] + ox
    Y = y[..., None, None] + oy

    X0 = X - cx
    Y0 = Y - cy

    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Xp =  c * X0 + s * Y0
    Yp = -s * X0 + c * Y0

    inside = (np.abs(Xp) <= (length * 0.5)) & (np.abs(Yp) <= (width * 0.5))
    cov = inside.mean(axis=(-1, -2)).astype(dtype)
    return torch.from_numpy(cov)


def rounded_rect(h, w, cx, cy, length, width, corner_radius, angle_rad, ss=8, dtype=np.float32):
    """
    Rotated rounded-rectangle via intersection/union test:
    - core rectangle of (length-2R) x width
    - two semicircles of radius R at the ends
    If corner_radius<=0, this reduces to rotated_rect_supersample.
    
    Args:
        h: Height in pixels
        w: Width in pixels
        cx: Center x coordinate
        cy: Center y coordinate
        length: Full extent along local x' axis (pixels)
        width: Full extent along local y' axis (pixels)
        corner_radius: Corner radius in pixels
        angle_rad: Rotation angle in radians (CCW from +x)
        ss: Supersampling per axis (default 8)
        dtype: Output dtype (default np.float32)
    
    Returns:
        Coverage tensor of shape (h, w) with values in [0,1]
    """
    if corner_radius <= 0:
        return rotated_rect(h, w, cx, cy, length, width, angle_rad, ss, dtype)
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    ofs = (np.arange(ss) + 0.5) / ss
    oy, ox = np.meshgrid(ofs, ofs, indexing="ij")

    X = x[..., None, None] + ox
    Y = y[..., None, None] + oy

    X0 = X - cx
    Y0 = Y - cy

    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Xp =  c * X0 + s * Y0
    Yp = -s * X0 + c * Y0

    half_core = max(length * 0.5 - corner_radius, 0.0)
    rect_core = (np.abs(Xp) <= half_core) & (np.abs(Yp) <= (width * 0.5))

    Xc1, Yc1 = -half_core, 0.0
    Xc2, Yc2 =  half_core, 0.0
    cap1 = ((Xp - Xc1)**2 + (Yp - Yc1)**2) <= corner_radius**2
    cap2 = ((Xp - Xc2)**2 + (Yp - Yc2)**2) <= corner_radius**2

    inside = rect_core | (cap1 & (np.abs(Yp) <= (width * 0.5))) | (cap2 & (np.abs(Yp) <= (width * 0.5)))
    cov = inside.mean(axis=(-1, -2)).astype(dtype)
    return torch.from_numpy(cov)


def set_nanofin(sim_shape, wl, dL, pmls, kwargs):
    """
    Builds a single rotated nano-fin meta-atom.
    
    Args:
        sim_shape: Simulation shape tuple (sx, sy, sz)
        wl: Wavelength in nm
        dL: Grid size in nm/pixel
        pmls: PML configuration list
        kwargs: Dictionary with keys:
            - substrate_material, meta_atom_material, top_medium_material (material names)
            - source_polarization: {"x","y","z","LCP","RCP"}
            - substrate_thickness_nm, source_below_meta_nm (in nm)
            - fin_height_nm, fin_length_nm, fin_width_nm (in nm)
            - fin_angle_deg: rotation CCW from +x (degrees)
            - use_rounded: bool, use rounded corners (optional)
            - corner_radius_nm: corner radius in nm (optional)
            - ss: supersampling per axis, int
    
    Returns:
        Tuple of (eps, src) tensors with batch dimension
    """
    assert len(sim_shape) == 3
    sx, sy, sz = sim_shape

    substrate_material = kwargs.get('substrate_material')
    meta_atom_material = kwargs.get('meta_atom_material')
    top_medium_material = kwargs.get('top_medium_material')
    src_pol            = kwargs.get('source_polarization')
    ss                 = int(kwargs.get('ss'))
    use_rounded        = bool(kwargs.get('use_rounded', False))
    theta_deg          = float(kwargs.get('fin_angle_deg', 0.0) or 0.0)

    if substrate_material is None:
        raise ValueError("substrate_material is required but not provided")
    if meta_atom_material is None:
        raise ValueError("meta_atom_material is required but not provided")
    if top_medium_material is None:
        raise ValueError("top_medium_material is required but not provided")

    substrate_eps = get_n(substrate_material, wl * 1e-3) ** 2
    meta_atom_eps = get_n(meta_atom_material, wl * 1e-3) ** 2
    top_medium_eps = get_n(top_medium_material, wl * 1e-3) ** 2

    z_sub              = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    fin_h              = get_pixels(kwargs, 'fin_height_nm', dL)
    fin_L              = get_pixels(kwargs, 'fin_length_nm', dL)
    fin_W              = get_pixels(kwargs, 'fin_width_nm', dL)
    corner_R           = get_pixels(kwargs, 'corner_radius_nm', dL) if use_rounded else 0
    src_below          = get_pixels(kwargs, 'source_below_meta_nm', dL)

    assert z_sub > src_below, "Source must lie inside the substrate."

    eps = top_medium_eps * torch.ones(sim_shape, dtype=torch.float32)
    eps[:, :, :z_sub] = substrate_eps

    cx = sx // 2
    cy = sy // 2
    angle_rad = float(np.deg2rad(theta_deg))
    if use_rounded and corner_R > 0:
        cross = top_medium_eps + (meta_atom_eps - top_medium_eps) * rounded_rect(
            sx, sy, cx, cy, fin_L, fin_W, corner_R, angle_rad, ss=ss
        )
    else:
        cross = top_medium_eps + (meta_atom_eps - top_medium_eps) * rotated_rect(
            sx, sy, cx, cy, fin_L, fin_W, angle_rad, ss=ss
        )
    z0 = z_sub
    z1 = min(z_sub + fin_h, sz)
    eps[:, :, z0:z1] = cross[..., None].to(eps.dtype)

    src = torch.zeros((*sim_shape, 6), dtype=torch.float32)
    z_src = z_sub - src_below
    src = set_source_plane(src, z_src, pol=src_pol, amp=1.0, direction='forward', dL=dL, wavelength=wl, n=np.sqrt(substrate_eps))

    return eps[None], src[None]


def set_nanofin_pair(sim_shape, wl, dL, pmls, kwargs):
    """
    Two parallel fins separated by 'gap_nm' along the local y' (width) axis.
    
    Args:
        sim_shape: Simulation shape tuple (sx, sy, sz)
        wl: Wavelength in nm
        dL: Grid size in nm/pixel
        pmls: PML configuration list
        kwargs: Dictionary with keys (additions over single fin):
            - gap_nm: gap between fins in nm
            - pair_angle_deg: angle for pair (if different from fin_angle_deg)
            - center_offset_nm: tuple (dx, dy) in nm to nudge the pair center
            - source_polarization: {"x","y","z","LCP","RCP"}
            - All keys from set_nanofin
    
    Returns:
        Tuple of (eps, src) tensors with batch dimension
    """
    pass 

    # assert len(sim_shape) == 3
    # sx, sy, sz = sim_shape

    # substrate_material = kwargs.get('substrate_material')
    # meta_atom_material = kwargs.get('meta_atom_material')
    # top_medium_material = kwargs.get('top_medium_material')
    # src_pol            = kwargs.get('source_polarization')
    # ss                 = int(kwargs.get('ss'))
    # use_rounded        = bool(kwargs.get('use_rounded', False))
    # theta_deg          = float(kwargs.get('fin_angle_deg', 0.0) or 0.0)

    # if substrate_material is None:
    #     raise ValueError("substrate_material is required but not provided")
    # if meta_atom_material is None:
    #     raise ValueError("meta_atom_material is required but not provided")
    # if top_medium_material is None:
    #     raise ValueError("top_medium_material is required but not provided")

    # substrate_eps = get_n(substrate_material, wl * 1e-3) ** 2   
    # meta_atom_eps = get_n(meta_atom_material, wl * 1e-3) ** 2
    # top_medium_eps = get_n(top_medium_material, wl * 1e-3) ** 2

    # z_sub              = get_pixels(kwargs, 'substrate_thickness_nm', dL)
    # fin_h              = get_pixels(kwargs, 'fin_height_nm', dL)
    # fin_L              = get_pixels(kwargs, 'fin_length_nm', dL)
    # fin_W              = get_pixels(kwargs, 'fin_width_nm', dL)
    # gap                = get_pixels(kwargs, 'gap_nm', dL)
    # corner_R           = get_pixels(kwargs, 'corner_radius_nm', dL) if use_rounded else 0
    # src_below          = get_pixels(kwargs, 'source_below_meta_nm', dL)
    # theta_deg = float(kwargs.get('pair_angle_deg', kwargs.get('fin_angle_deg', 0.0)) or 0.0)
    # angle_rad = float(np.deg2rad(theta_deg))
    # assert z_sub > src_below, "Source must lie inside the substrate."

    # dx_nm, dy_nm = kwargs.get('center_offset_nm', (0.0, 0.0))
    # dx = int(round(dx_nm / float(dL)))
    # dy = int(round(dy_nm / float(dL)))

    # eps = top_medium_eps * torch.ones(sim_shape, dtype=torch.float32)
    # eps[:, :, :z_sub] = substrate_eps

    # cx = sx // 2 + dx
    # cy = sy // 2 + dy

    # def draw_bar(cxi, cyi, angle_rad):
    #     if corner_R > 0:
    #         cov = rounded_rect(sx, sy, cxi, cyi, fin_L, fin_W, corner_R, angle_rad, ss=ss)
    #     else:
    #         cov = rotated_rect(sx, sy, cxi, cyi, fin_L, fin_W, angle_rad, ss=ss)
    #     return cov

    # dy_local = 0.5 * (gap + fin_W)
    # c, s = np.cos(angle_rad), np.sin(angle_rad)
    # off1_x =  int(round( 0 * c + (+dy_local) * (-s)))
    # off1_y =  int(round( 0 * s + (+dy_local) * ( c)))
    # off2_x =  int(round( 0 * c + (-dy_local) * (-s)))
    # off2_y =  int(round( 0 * s + (-dy_local) * ( c)))

    # cov1 = draw_bar(cx + off1_x, cy + off1_y, angle_rad)
    # cov2 = draw_bar(cx + off2_x, cy + off2_y, angle_rad)

    # cross = cov1 + cov2
    # cross = torch.clamp(cross, 0.0, 1.0)

    # cross_eps = top_medium_eps + (meta_atom_eps - top_medium_eps) * cross
    # z0 = z_sub
    # z1 = min(z_sub + fin_h, sz)
    # eps[:, :, z0:z1] = cross_eps[..., None].to(eps.dtype)

    # src = torch.zeros((*sim_shape, 6), dtype=torch.float32)
    # z_src = z_sub - src_below
    # src = set_source_plane(src, z_src, pol=src_pol, amp=1.0, direction='forward', dL=dL, wavelength=wl, n=np.sqrt(substrate_eps))

    # return eps[None], src[None]
