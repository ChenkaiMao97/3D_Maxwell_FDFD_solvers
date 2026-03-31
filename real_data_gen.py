"""
Device dataset generator using gdsfactory -> polygons -> raster -> 3D index volume -> NPZ.

Devices:
(0) Geometric metasurface: rotated rectangles on substrate
(1) Freeform metasurface: low-pass filtered random field -> threshold -> blob mask
(2) Adiabatic coupler: two waveguides approach (small gap) then separate
(3) Grating coupler: concentric arcs (rings segments) converging to a waveguide
(4) Ring resonator: ring + bus waveguide with gap

Exports:
- <name>.npz containing:
    n: float32 index volume [Nz, Ny, Nx]
    params: JSON string
    meta: JSON string (grid/units)
- <name>.gds (layout editable in KLayout / gdsfactory)
"""

from __future__ import annotations

import os
import json
import math
from math import floor, ceil
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
import skimage.draw as skdraw
from tqdm import tqdm
import h5py

# Required: pip install gdsfactory
import gdsfactory as gf
# from gdsfactory.export import to_np

from matplotlib.path import Path
import matplotlib.pyplot as plt

from src.utils.plot_field3D import plot_3slices
from src.utils.PML_utils import make_dxes_numpy
from src.utils.physics import eps_to_yee
from spins.fdfd_tools.waveguide_mode import solve_waveguide_mode, compute_source
from spins.gridlock import direction
import torch

# Optional but recommended:
#   pip install shapely scikit-image
try:
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

try:
    from skimage.draw import polygon as sk_polygon
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def get_waveguide_source(wavelength, dL, pml_layers, eps, src_direction, src_slice, mode_num=0, power=1.0, ln_R=-16):
    # direction: e.g. [0,0,1], [0,-1,0]
    # src_slice: e.g. [(start_x, start_y, start_z), (end_x, end_y, end_z)] (inclusive)
    omega=2 * np.pi / wavelength
    dxes = make_dxes_numpy(wavelength, dL, eps.shape, pml_layers, ln_R)
    axis = direction.Direction(direction.axisvec2axis(src_direction))
    slices = tuple([slice(i, f+1) for i, f in zip(*src_slice)])

    eps_yee = eps_to_yee(torch.from_numpy(eps[None]))[0].permute(3,0,1,2).numpy()
    # mu = np.ones_like(eps_yee)
    
    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis.value,
        'slices': slices,
        'polarity': direction.axisvec2polarity(src_direction)
    }

    wgmode_result = solve_waveguide_mode(
        mode_number = mode_num,
        epsilon = eps_yee,
        **sim_params)
    J = compute_source(**wgmode_result, **sim_params)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)
    return J

def find_wg_center(eps_s):
    # eps_s: slice of eps 
    eps_min = np.min(eps_s)
    assert eps_s[0] == eps_min and len(eps_s.shape) == 1
    for i, eps in enumerate(eps_s):
        if eps > eps_min and eps_s[i-1] == eps_min:
            start = i
        if eps == eps_min and eps_s[i-1] > eps_min:
            end = i
            return (start + end)//2
    raise ValueError("didn't find waveguide center")
    
def _rng_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _rot2d(points: np.ndarray, theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ R.T


def _rectangle_poly(cx: float, cy: float, lx: float, ly: float, theta_rad: float) -> np.ndarray:
    """Returns 4x2 polygon points of a rotated rectangle centered at (cx,cy)."""
    pts = np.array(
        [[-lx/2, -ly/2],
         [ lx/2, -ly/2],
         [ lx/2,  ly/2],
         [-lx/2,  ly/2]],
        dtype=np.float64
    )
    pts = _rot2d(pts, theta_rad)
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts

def rasterize_polygon_aa(poly_rc, shape_rc, ss=8):
    """
    poly_rc: (N,2) vertices in (r, c) pixel coordinates at base resolution
    shape_rc: (H,W) at base resolution
    ss: supersampling factor
    returns coverage in [0,1] with shape (H,W)
    """
    H, W = shape_rc
    HH, WW = H * ss, W * ss

    # Supersampled mask
    mask_hi = np.zeros((HH, WW), dtype=np.uint8)

    rr, cc = skdraw.polygon(poly_rc[:, 0] * ss, poly_rc[:, 1] * ss, shape=(HH, WW))
    mask_hi[rr, cc] = 1

    # Block-average downsample: (HH,WW) -> (H,W)
    cov = mask_hi.reshape(H, ss, W, ss).mean(axis=(1, 3))
    return cov

def get_local_poly(poly_rc, H, W,padding = 2):
    # note that coordinates in poly_rc are float numbers
    x_min = max(floor(np.min(poly_rc[:, 0])-padding), 0)
    x_max = min(ceil(np.max(poly_rc[:, 0])+padding), H)
    y_min = max(floor(np.min(poly_rc[:, 1])-padding), 0)
    y_max = min(ceil(np.max(poly_rc[:, 1])+padding), W)
    local_poly = poly_rc.copy()
    local_poly[:, 0] -= x_min
    local_poly[:, 1] -= y_min
    return local_poly, x_min, y_min, x_max - x_min, y_max - y_min

def to_np_aa(component, nm_per_pixel=20, layers=((1,0),), values=None, pad_width=((0,0), (0,0)), ss=8):
    pixels_per_um = (1 / nm_per_pixel) * 1e3
    xmin, ymin = component.bbox[0]
    xmax, ymax = component.bbox[1]

    H = int((xmax - xmin) * pixels_per_um)
    W = int((ymax - ymin) * pixels_per_um)
    img = np.zeros((H, W), dtype=float)

    layer_to_polygons = component.get_polygons(by_spec=True, depth=None)
    values = values or [1] * len(layers)

    for layer, value in zip(layers, values):
        if layer not in layer_to_polygons:
            continue
        for polygon in tqdm(layer_to_polygons[layer]):
            # convert polygon points to base pixel coordinates (r,c)
            r = (polygon[:, 0] - xmin) * pixels_per_um
            c = (polygon[:, 1] - ymin) * pixels_per_um
            poly_rc = np.stack([r, c], axis=1)

            local_poly, x_min, y_min, x_size, y_size = get_local_poly(poly_rc, H, W)

            cov = rasterize_polygon_aa(local_poly, (x_size, y_size), ss=ss)
            # "latter overwrite former" behavior:
            img[x_min:x_min+x_size, y_min:y_min+y_size] = np.where(cov > 0, cov * value, img[x_min:x_min+x_size, y_min:y_min+y_size])

    result = np.pad(img, pad_width=pad_width)
    return result


def to_np(
    component: Component,
    nm_per_pixel: int = 20,
    layers: Layers = ((1, 0),),
    values: Optional[Floats] = None,
    pad_width: Tuple[int, int] = (0,0),
) -> np.ndarray:
    """Returns a pixelated numpy array from Component polygons.

    Args:
        component: Component.
        nm_per_pixel: you can go from 20 (coarse) to 4 (fine).
        layers: to convert. Order matters (latter overwrite former).
        values: associated to each layer (defaults to 1).
        pad_width: padding pixels around the image.

    """
    import skimage.draw as skdraw

    pixels_per_um = (1 / nm_per_pixel) * 1e3
    xmin, ymin = component.bbox[0]
    xmax, ymax = component.bbox[1]
    shape = (
        int(np.ceil(xmax - xmin) * pixels_per_um),
        int(np.ceil(ymax - ymin) * pixels_per_um),
    )
    img = np.zeros(shape, dtype=float)
    layer_to_polygons = component.get_polygons(by_spec=True, depth=None)
    values = values or [1] * len(layers)

    for layer, value in zip(layers, values):
        if layer in layer_to_polygons:
            polygons = layer_to_polygons[layer]
            for polygon in polygons:
                r = polygon[:, 0] - xmin
                c = polygon[:, 1] - ymin
                rr, cc = skdraw.polygon(
                    r * pixels_per_um, c * pixels_per_um, shape=shape
                )
                img[rr, cc] = value

    return np.pad(img, pad_width=((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1])))


def _arc_polyline(
    r: float,
    theta0: float,
    theta1: float,
    n: int,
    cx: float = 0.0,
    cy: float = 0.0
) -> np.ndarray:
    """Polyline points on a circular arc."""
    t = np.linspace(theta0, theta1, n)
    x = cx + r * np.cos(t)
    y = cy + r * np.sin(t)
    return np.stack([x, y], axis=1)


def _thick_arc_polygon(
    r: float,
    width: float,
    theta0: float,
    theta1: float,
    n: int,
    cx: float = 0.0,
    cy: float = 0.0
) -> np.ndarray:
    """
    Approximate an arc "ring segment" polygon by stitching outer + inner arcs.
    """
    r_out = r + width / 2
    r_in = max(1e-6, r - width / 2)
    outer = _arc_polyline(r_out, theta0, theta1, n, cx, cy)
    inner = _arc_polyline(r_in, theta1, theta0, n, cx, cy)  # reverse
    pts = np.vstack([outer, inner])
    return pts

def poly_contains_points(poly_xy: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    # poly_xy: (M,2), pts_xy: (N,2)
    return Path(poly_xy).contains_points(pts_xy)

def _rasterize_polygons_to_mask(
    polygons_um: List[np.ndarray],
    bbox_um: Tuple[float, float, float, float],
    dx_um: float,
) -> np.ndarray:
    """
    Rasterize polygons into a boolean mask [H,W] over bbox.

    bbox_um = (xmin, ymin, xmax, ymax) in um.
    dx_um is pixel size (um/pixel). Uses skimage if available; otherwise shapely.
    """
    xmin, ymin, xmax, ymax = bbox_um
    W = int(math.ceil((xmax - xmin) / dx_um))
    H = int(math.ceil((ymax - ymin) / dx_um))
    mask = np.zeros((H, W), dtype=np.bool_)

    if _HAS_SKIMAGE:
        for pts in polygons_um:
            # map um -> pixel coordinates
            x = (pts[:, 0] - xmin) / dx_um
            y = (pts[:, 1] - ymin) / dx_um
            rr, cc = sk_polygon(y, x, shape=mask.shape)
            mask[rr, cc] = True
        return mask

    if not _HAS_SHAPELY:
        raise RuntimeError(
            "Need either scikit-image or shapely for rasterization. "
            "Install one of: `pip install scikit-image` or `pip install shapely`."
        )

    # shapely fallback: test pixel centers
    from shapely.prepared import prep
    union = unary_union([Polygon(p) for p in polygons_um if len(p) >= 3])
    P = prep(union)
    ys = ymin + (np.arange(H) + 0.5) * dx_um
    xs = xmin + (np.arange(W) + 0.5) * dx_um
    # brute force (ok for moderate sizes)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if poly_contains_points(P, np.array([[x, y]])):  # not ideal
                mask[iy, ix] = True
    return mask


def _component_polygons_um(component: gf.Component, layer: Tuple[int, int]) -> List[np.ndarray]:
    """
    Returns list of polygons (Nx2 float arrays) in um for a given layer.
    Uses Component.get_polygons_points(by='tuple').
    """

    d = get_polygons_points_compat(component)
    # d = component.get_polygons_points(by="tuple")  # keys like (layer, datatype) :contentReference[oaicite:3]{index=3}
    polys = d.get(layer, [])
    out = []
    for p in polys:
        # p is an array of shape (N,2)
        p = np.asarray(p, dtype=np.float64)
        if p.shape[0] >= 3:
            out.append(p)
    return out

def _polygon_to_points(poly):
    """
    Convert a KLayout / gdsfactory polygon to (N,2) numpy array.
    """
    # Most common: kdb.Polygon / kdb.DPolygon
    for attr in ("each_point_hull", "each_point"):
        if hasattr(poly, attr):
            return np.array(
                [(p.x, p.y) for p in getattr(poly, attr)()],
                dtype=float
            )

    # Fallbacks
    if hasattr(poly, "points"):
        return np.array([(p.x, p.y) for p in poly.points], dtype=float)

    raise TypeError(f"Cannot extract points from polygon type {type(poly)}")

def get_polygons_points_compat(component, layers=None):
    """
    Returns:
        dict[layer_key -> list of (Ni,2) float arrays]
    Works across gdsfactory versions where:
      - get_polygons_points() exists
      - get_polygons(by=...) returns dict
      - get_polygons() returns list
    """
    # ----------------------------
    # Case 1: modern API
    # ----------------------------
    if hasattr(component, "get_polygons_points"):
        poly_dict = component.get_polygons_points(by="tuple", layers=layers)
        return {
            k: [np.asarray(p, dtype=float) for p in v if len(p) >= 3]
            for k, v in poly_dict.items()
        }

    # ----------------------------
    # Case 2 / 3: fallback to get_polygons
    # ----------------------------
    if not hasattr(component, "get_polygons"):
        raise TypeError(
            f"{type(component)} has no get_polygons or get_polygons_points"
        )

    try:
        poly_out = component.get_polygons(by="tuple", layers=layers)
    except TypeError:
        poly_out = component.get_polygons()

    # ----------------------------
    # Case 2: dict[layer -> list]
    # ----------------------------
    if isinstance(poly_out, dict):
        result = {}
        for layer_key, poly_list in poly_out.items():
            pts_list = []
            for poly in poly_list:
                # pts = _polygon_to_points(poly)
                pts = poly
                if pts.shape[0] >= 3:
                    pts_list.append(pts)
            if pts_list:
                result[layer_key] = pts_list
        return result

    # ----------------------------
    # Case 3: list of polygons (no layer info)
    # ----------------------------
    if isinstance(poly_out, list):
        result = {}
        for poly in poly_out:
            # pts = _polygon_to_points(poly)
            pts = poly
            # pts = _polygon_to_points(poly)
            if pts.shape[0] < 3:
                continue

            # try to recover layer info if available
            layer_key = None
            if hasattr(poly, "layer"):
                layer_key = (poly.layer, poly.datatype)
            elif hasattr(poly, "layer_index"):
                layer_key = poly.layer_index
            else:
                layer_key = "unknown"

            result.setdefault(layer_key, []).append(pts)
        return result

    raise TypeError(f"Unexpected get_polygons() return type: {type(poly_out)}")

@dataclass
class Range:
    lo: float
    hi: float

class PhotonicsDeviceDataset:
    """
    Generates devices and exports to .npz (3D index volume) + .gds (editable layout).
    """

    def __init__(
        self,
        out_dir: str = "out_devices",
        N_per_device: int = 5,
        seed: int = 7,
        # Raster / volume grid
        wl: float = 800,
        dx_um: float = 0.05,     # XY resolution (um/pixel)
        dz_um: float = 0.05,     # Z resolution (um/voxel)
        # Layers (GDS)
        layer_device: Tuple[int, int] = (2, 0),
        layer_substrate: Tuple[int, int] = (1, 0),
    ):
        self.out_dir = out_dir
        self.N_per_device = int(N_per_device)
        self.rng = np.random.default_rng(seed)

        self.wl = float(wl)
        self.dx_um = float(dx_um)
        self.dz_um = float(dz_um)

        self.pmls = [30,30,30,30,30,30]
        self.ln_R = -10

        self.layer_device = layer_device
        self.layer_substrate = layer_substrate

        os.makedirs(self.out_dir, exist_ok=True)

        # -------------------------
        # Parameter ranges (min/max)
        # -------------------------

        # Global stack
        self.stack = {
            "t_sub_n_clad_um": Range(1.2, 1.4),
            "t_dev_um": Range(0.6, 1.0),
            "n_sub": Range(1.44, 1.50),     # e.g. silica-ish
            "n_dev": Range(2.0, 2.8),      # SiN to TiO2-ish
            "n_clad": Range(1.0, 1.0),      # air
        }

        # (0) geometric metasurface
        self.geo_ms = {
            "aperture_um": Range(4.0, 8.0),       # square aperture side
            "pitch_um": Range(0.25, 0.4),
            "rect_l_um": Range(0.15, 0.22),
            "rect_w_um": Range(0.07, 0.12),
            "pad_um": Range(0.3, 0.6)
        }

        # (1) freeform metasurface
        self.free_ms = {
            "aperture_um": Range(4.0, 8.0),
            "fft_keep_frac": Range(0.05, 0.15),   # low-pass radius as fraction of Nyquist
            "threshold": Range(0.45, 0.65),       # binarization threshold
            "pad_um": Range(0.3, 0.6)
        }

        # (2) adiabatic coupler
        self.cplr = {
            "L_um": Range(8.0, 12.0),
            "center_len_percent": Range(0.2, 0.3),
            "taper_len_percent": Range(0.2, 0.3),
            "wg_w_um": Range(0.4, 0.6),
            "gap_min_um": Range(0.15, 0.3),
            "gap_max_um": Range(1.0, 3.0),
            "padx_um": Range(0.0, 0.0),
            "pady_um": Range(1.5, 2.0),
        }

        # (3) grating coupler
        self.grating = {
            "n_rings": Range(4, 8),               # integer
            "r0_um": Range(0.5, 1.0),
            "pitch_um": Range(0.6, 0.8),
            "arc_width_um": Range(0.25, 0.5),
            "arc_span_deg": Range(80.0, 120.0),    # total span
            "wg_w_um": Range(0.3, 1.0),
            "wg_L_um": Range(3.0, 5.0),
            "padx_um": Range(0.0, 0.0),
            "pady_um": Range(1.0, 2.0),
        }

        # (4) ring resonator
        self.ring = {
            "radius_um": Range(2.0, 4.0),
            "gap_um": Range(0.15, 0.4),
            "wg_w_um": Range(0.3, 0.6),
            "bus_L_um": Range(10.0, 12.0),
            "padx_um": Range(0.0, 0.0),
            "pady_um": Range(1.0, 2.0),
        }

    # -------------------------
    # Device generators (2D)
    # -------------------------

    def _device_geometric_metasurface(self, p: Dict[str, Any]) -> gf.Component:
        c = gf.Component("geo_ms")
        A = p["aperture_um"]

        pitch = p["pitch_um"]
        rect_l = p["rect_l_um"]
        rect_w = p["rect_w_um"]

        # grid of rectangles
        xs = np.arange(-A/2 + pitch/2, A/2, pitch)
        ys = np.arange(-A/2 + pitch/2, A/2, pitch)
        for x in xs:
            for y in ys:
                # simple phase-map: angle varies with x,y
                u = (x / (A/2 + 1e-9)) * 0.5 + (y / (A/2 + 1e-9)) * 0.5
                theta = np.random.uniform(-np.pi/2, np.pi/2)
                pts = _rectangle_poly(x, y, rect_l, rect_w, theta)
                c.add_polygon(pts, layer=self.layer_device)

        return c

    def _device_freeform_metasurface(self, p: Dict[str, Any]) -> gf.Component:
        c = gf.Component("free_ms")
        
        A = p["aperture_um"]

        # Create random field, low-pass in Fourier, then threshold to binary mask
        N = int(math.ceil(A / self.dx_um))
        # N = max(64, int(2 ** math.ceil(math.log2(N))))  # power of 2 convenience
        noise = self.rng.standard_normal((N, N)).astype(np.float64)

        F = np.fft.fftshift(np.fft.fft2(noise))
        ky = np.linspace(-0.5, 0.5, N, endpoint=False)
        kx = np.linspace(-0.5, 0.5, N, endpoint=False)
        KX, KY = np.meshgrid(kx, ky)
        KR = np.sqrt(KX**2 + KY**2)

        keep = float(p["fft_keep_frac"])
        filt = (KR <= keep).astype(np.float64)
        field = np.real(np.fft.ifft2(np.fft.ifftshift(F * filt)))

        # normalize to [0,1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-12)
        
        mask = field > float(p["threshold"])

        yy, xx = np.where(mask)
        # downsample points to avoid millions of rectangles
        stride = 1
        sel = (yy % stride == 0) & (xx % stride == 0)
        yy, xx = yy[sel], xx[sel]

        xmin = -A/2
        ymin = -A/2
        for (j, i) in zip(yy, xx):
            x0 = xmin + i * self.dx_um
            y0 = ymin + j * self.dx_um
            pts = np.array([[x0, y0],
                            [x0 + self.dx_um, y0],
                            [x0 + self.dx_um, y0 + self.dx_um],
                            [x0, y0 + self.dx_um]], dtype=float)
            c.add_polygon(pts, layer=self.layer_device)

        return c

    def _device_adiabatic_coupler(self, p: Dict[str, Any]) -> gf.Component:
        if not _HAS_SHAPELY:
            raise RuntimeError("Adiabatic coupler generator uses shapely. `pip install shapely`")

        c = gf.Component("adiabatic_coupler")

        L = p["L_um"]
        wg_w = p["wg_w_um"]
        gap_min = p["gap_min_um"]
        gap_max = p["gap_max_um"]
        center_len = p["center_len_percent"] * L
        taper_len = p["taper_len_percent"] * L
        assert center_len + taper_len < L, "Center length and taper length must be less than the total length"
        transition_width = (L - center_len - taper_len)/2

        # substrate

        # Build two centerlines with separation varying along x:
        # start sep=gap_max+wg_w, approach to gap_min+wg_w, then separate.
        x = np.linspace(-L/2, L/2, 250)
        def sep_profile(xv):
            ax = abs(xv)
            if ax < center_len/2:
                return (gap_min + wg_w)
            elif ax < center_len/2 + transition_width:
                # pick a sigmoid profile:
                var = (ax - center_len/2 - transition_width/2) / transition_width
                t = 1 / (1 + math.exp(-var * 10))
                return (1-t)* (gap_min + wg_w) + t*(gap_max + wg_w)
            else:
                return (gap_max + wg_w)

        sep = np.array([sep_profile(xv) for xv in x])
        y_top = +0.5 * sep
        y_bot = -0.5 * sep

        line_top = LineString(np.stack([x, y_top], axis=1))
        line_bot = LineString(np.stack([x, y_bot], axis=1))

        poly_top = line_top.buffer(wg_w/2, cap_style=2, join_style=2)
        poly_bot = line_bot.buffer(wg_w/2, cap_style=2, join_style=2)

        for poly in [poly_top, poly_bot]:
            pts = np.asarray(poly.exterior.coords, dtype=float)
            c.add_polygon(pts, layer=self.layer_device)

        return c

    def _device_grating_coupler(self, p: Dict[str, Any]) -> gf.Component:
        c = gf.Component("grating_coupler")

        n_rings = int(round(p["n_rings"]))
        r0 = p["r0_um"]
        pitch = p["pitch_um"]
        w = p["arc_width_um"]
        span = math.radians(p["arc_span_deg"])
        wg_w = p["wg_w_um"]
        wg_L = p["wg_L_um"]

        # arcs centered at origin, opening to the left towards a waveguide
        theta0 = - span/2
        theta1 = span/2
        for k in range(n_rings):
            r = r0 + k * pitch
            pts = _thick_arc_polygon(r=r, width=w, theta0=theta0, theta1=theta1, n=220, cx=0.0, cy=0.0)
            c.add_polygon(pts, layer=self.layer_device)

        # feed waveguide (a simple rectangle) from left towards origin
        # place it slightly offset in y
        y0 = 0.0
        x1 = 0.0
        x2 = x1 - wg_L
        pts_wg = np.array([[x2, y0 - wg_w/2],
                           [x1, y0 - wg_w/2],
                           [x1, y0 + wg_w/2],
                           [x2, y0 + wg_w/2]], dtype=float)
        c.add_polygon(pts_wg, layer=self.layer_device)

        return c

    def _device_ring_resonator(self, p: Dict[str, Any]) -> gf.Component:
        c = gf.Component("ring")

        radius = p["radius_um"]
        gap = p["gap_um"]
        wg_w = p["wg_w_um"]
        bus_L = p["bus_L_um"]

        size_um = 2.5 * (radius + bus_L/4)

        # Use built-in ring_single if available (common in gdsfactory components list) :contentReference[oaicite:5]{index=5}
        try:
            ring = gf.components.ring_single(gap=gap, radius=radius, cross_section=gf.cross_section.strip(width=wg_w))
            ref = c << ring
            ref.move((0, 0))
            # NOTE: ring_single already includes bus waveguide; we’re keeping it “real-looking”
        except Exception:
            # fallback: draw a ring (annulus) + a bus rectangle
            pts_out = _thick_arc_polygon(r=radius, width=wg_w, theta0=0.0, theta1=2*math.pi, n=400)
            c.add_polygon(pts_out, layer=self.layer_device)
            y_bus = -(radius + gap + wg_w)
            pts_bus = np.array([[-bus_L/2, y_bus - wg_w/2],
                                [ bus_L/2, y_bus - wg_w/2],
                                [ bus_L/2, y_bus + wg_w/2],
                                [-bus_L/2, y_bus + wg_w/2]], dtype=float)
            c.add_polygon(pts_bus, layer=self.layer_device)

        return c

    def generate_one_device(self, device_type: int) -> List[Dict[str, Any]]:
        """
        Randomize parameters and generate ONE device instance of `device_type`.
        Returns a list of exported sample dicts (usually length 1).
        """
        rng = self.rng

        # sample stack params
        p_stack = {k: _rng_uniform(rng, v.lo, v.hi) for k, v in self.stack.items()}

        # (1) prepare eps
        if device_type == 0:
            p = {k: _rng_uniform(rng, v.lo, v.hi) for k, v in self.geo_ms.items()}
            c = self._device_geometric_metasurface(p)
            padx, pady = p["pad_um"], p["pad_um"]
            pad_width = ((int(padx/self.dx_um), int(padx/self.dx_um)), (int(pady/self.dx_um), int(pady/self.dx_um)))

        elif device_type == 1:
            p = {k: _rng_uniform(rng, v.lo, v.hi) for k, v in self.free_ms.items()}
            c = self._device_freeform_metasurface(p)
            padx, pady = p["pad_um"], p["pad_um"]
            pad_width = ((int(padx/self.dx_um), int(padx/self.dx_um)), (int(pady/self.dx_um), int(pady/self.dx_um)))

        elif device_type == 2:
            p = {k: _rng_uniform(rng, v.lo, v.hi) for k, v in self.cplr.items()}
            c = self._device_adiabatic_coupler(p)
            padx, pady = p["padx_um"], p["pady_um"]
            pad_width = ((int(padx/self.dx_um), int(padx/self.dx_um)), (int(pady/self.dx_um), int(pady/self.dx_um)))

        elif device_type == 3:
            p = {}
            for k, v in self.grating.items():
                if k == "n_rings":
                    p[k] = int(rng.integers(int(v.lo), int(v.hi) + 1))
                else:
                    p[k] = _rng_uniform(rng, float(v.lo), float(v.hi))
            c = self._device_grating_coupler(p)
            padx, pady = p["padx_um"], p["pady_um"]
            pad_width = ((int(padx/self.dx_um), int(pady/self.dx_um)), (int(pady/self.dx_um), int(pady/self.dx_um))) # notice the difference here

        elif device_type == 4:
            p = {k: _rng_uniform(rng, v.lo, v.hi) for k, v in self.ring.items()}
            c = self._device_ring_resonator(p)
            padx, pady = p["padx_um"], p["pady_um"]
            pad_width = ((int(padx/self.dx_um), int(padx/self.dx_um)), (int(pady/self.dx_um), int(pady/self.dx_um)))

        else:
            raise ValueError(f"Unknown device_type={device_type}")

        # cross_section:
        device_array = to_np_aa(c, layers=((2,0),), values=(1,0), pad_width=pad_width, nm_per_pixel=round(self.dx_um*1000), ss=8)

        D = device_array.shape[0]
        W = device_array.shape[1]
        H = int(p_stack["t_sub_n_clad_um"] / self.dz_um) + int(p_stack["t_dev_um"] / self.dz_um) + int(p_stack["t_sub_n_clad_um"] / self.dz_um)

        print("final data shape: D, W, H", D, W, H)

        np_array = np.ones((D, W, H), dtype=float)
        np_array[:, :, 0:int(p_stack["t_sub_n_clad_um"] / self.dz_um)] = p_stack["n_sub"]
        np_array[:, :, int(p_stack["t_sub_n_clad_um"] / self.dz_um):int(p_stack["t_sub_n_clad_um"] / self.dz_um) + int(p_stack["t_dev_um"] / self.dz_um)] = device_array[:,:,None] * (p_stack["n_dev"] - p_stack["n_clad"]) + p_stack["n_clad"]
        np_array[:, :, int(p_stack["t_sub_n_clad_um"] / self.dz_um) + int(p_stack["t_dev_um"] / self.dz_um):] = p_stack["n_clad"]

        eps_array = np_array**2

        # (2) prepare src
        src_array = np.zeros((D, W, H, 3), dtype=np.complex64)
        kz = 2*np.pi * p_stack["n_sub"] / self.wl
        pixels_per_um = 1 / self.dx_um
        if device_type in [0,1]:
            source_z = int(0.75*int(p_stack["t_sub_n_clad_um"] / self.dz_um))
            src_array[:,:,source_z,0] = 1.0
            src_array[:,:,source_z-1,0] = -np.exp(-1j*kz*self.dz_um*1000)
        elif device_type == 2:
            mp = round(max(p["wg_w_um"], p_stack["t_dev_um"])*pixels_per_um) # mode padding
            source_dir = np.array((1,0,0))
            x_pos = self.pmls[0] + 5
            z_center = H//2
            y_center = find_wg_center(eps_array[x_pos,:, z_center])
            y_start = max(round(y_center - mp*1.5), 0)
            y_end = min(round(y_center + mp*1.5), D)
            z_start = max(round(z_center - mp*1.5), 0)
            z_end = min(round(z_center + mp*1.5), H)
        
            source_slice = tuple([(x_pos, y_start, z_start), (x_pos, y_end, z_end)])
            print(f"computing waveguide mode source with slice: {source_slice}")
            source = get_waveguide_source(self.wl, self.dx_um*1000, self.pmls, eps_array, source_dir, source_slice, ln_R=self.ln_R)
            src_array = source.transpose(1,2,3,0) # D, W, H, 3
        elif device_type == 3:
            x_start, x_end = D//2, D-50
            y_start, y_end = (1*W)//4, (3*W)//4
            z_pos = H-self.pmls[5]-5

            src_array[x_start:x_end, y_start:y_end, z_pos] = 1
            src_array[x_start:x_end, y_start:y_end, z_pos+1] = -np.exp(-1j*kz*self.dz_um*1000)
        elif device_type == 4:
            mp = round(max(p["wg_w_um"], p_stack["t_dev_um"])*pixels_per_um) # mode padding
            source_dir = np.array((1,0,0))
            x_pos = self.pmls[0] + 5
            z_center = H//2
            y_center = find_wg_center(eps_array[x_pos, :, z_center])
            y_start = max(round(y_center - mp*1.5), 0)
            y_end = min(round(y_center + mp*1.5), D)
            z_start = max(round(z_center - mp*1.5), 0)
            z_end = min(round(z_center + mp*1.5), H)
        
            source_slice = tuple([(x_pos, y_start, z_start), (x_pos, y_end, z_end)])
            print(f"computing waveguide mode source with slice: {source_slice}")
            source = get_waveguide_source(self.wl, self.dx_um*1000, self.pmls, eps_array, source_dir, source_slice, ln_R=self.ln_R)
            src_array = source.transpose(1,2,3,0) # D, W, H, 3  
        
        return eps_array, src_array

    def generate_all(self):
        eps_all = []
        src_all = []
        for device_type in [0, 1, 2, 3]:
            for _ in range(self.N_per_device):
                eps, src = self.generate_one_device(device_type)
                eps_all.append(eps)
                src_all.append(src)
        return eps_all, src_all

def save_paired_h5(path, eps_list, src_list):
    assert len(eps_list) == len(src_list)
    with h5py.File(path, "w") as f:
        for i, (eps, src) in enumerate(zip(eps_list, src_list)):
            g = f.create_group(f"{i:06d}")
            g.create_dataset("eps", data=eps, compression="gzip")
            g.create_dataset("src", data=src, compression="gzip")


if __name__ == "__main__":
    gen = PhotonicsDeviceDataset(
        out_dir="out_devices",
        N_per_device=5,
        seed=123,
        dx_um=0.03,
        dz_um=0.03,
    )
    eps_all, src_all = gen.generate_all()

    # for i in range(len(eps_all)):
    #     source_combined = np.sum(np.abs(src_all[i]), axis=-1)
    #     plot_3slices(eps_all[i], fname=f"eps_{i}.png", cm_zero_center=False)
    #     plot_3slices(source_combined, fname=f"src_{i}.png", cm_zero_center=False)

    save_paired_h5('data/real_dataset.h5', eps_all, src_all)

    

