# material.py
from __future__ import annotations
import json, math, os
from functools import lru_cache
from typing import Iterable, Tuple, Union, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None

Number = Union[float, int]
ArrayLike = Union[Number, Iterable[Number]]

@lru_cache(maxsize=1)
def _load_db(path=None):
    """
    Load material database from JSON file.
    
    Args:
        path: Path to material database JSON file (optional, defaults to material_data.json)
    
    Returns:
        Dictionary of materials
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "material_data.json")
    with open(path, "r") as f:
        data = json.load(f)
    mats = data.get("materials")
    if not isinstance(mats, dict):
        raise ValueError("materials.json must contain a top-level 'materials' object.")
    return mats


def available_materials(db_path=None):
    """
    Get list of available material names.
    
    Args:
        db_path: Path to material database JSON file (optional)
    
    Returns:
        Sorted list of material names
    """
    return sorted(_load_db(db_path).keys())


def _to_array(x):
    """
    Convert input to array and return scalar flag.
    
    Args:
        x: Scalar or array-like input
    
    Returns:
        Tuple of (array, is_scalar_flag)
    """
    if isinstance(x, (int, float)):
        return (np.array([x], dtype=float) if np else [float(x)]), True
    if np:
        return np.asarray(list(x), dtype=float), False
    return [float(v) for v in x], False


def _constant_like(x, value):
    """
    Create constant array matching input shape.
    
    Args:
        x: Scalar or array-like input (used to determine shape)
        value: Constant value to fill
    
    Returns:
        Constant value or array matching x shape
    """
    arr, is_scalar = _to_array(x)
    if np and isinstance(arr, np.ndarray):
        out = np.full_like(arr, float(value), dtype=float)
        return float(out[0]) if is_scalar else out
    out = [float(value) for _ in (arr if isinstance(arr, list) else [arr])]
    return out[0] if is_scalar else out


def _check_range(material, w_um, name):
    """
    Check if wavelength(s) are within material's valid range.
    
    Args:
        material: Material dictionary
        w_um: Wavelength(s) in micrometers
        name: Material name string
    """
    rng = material.get("range_um")
    if not rng:
        return
    lo, hi = float(rng[0]), float(rng[1])
    arr, _ = _to_array(w_um)
    vals = arr if (np and isinstance(arr, np.ndarray)) else (arr if isinstance(arr, list) else [arr])
    bad = [lam for lam in vals if lam < lo or lam > hi]
    if bad:
        raise ValueError(f"Wavelength(s) {bad} μm out of declared range {lo}–{hi} μm for '{name}'.")

# -------- n-models --------

def _n_constant(w_um, params):
    n0 = float(params["n"])
    return _constant_like(w_um, n0)

def _n_ab_over_l2_minus_c(w_um, params):
    """
    Compute n using n^2 = A + B / (λ^2 - C); λ in μm, C in μm^2 (TiO2 form).
    
    Args:
        w_um: Wavelength(s) in micrometers
        params: Dictionary with keys A, B, C
    
    Returns:
        Refractive index (scalar or array matching w_um shape)
    """
    A = float(params["A"]); B = float(params["B"]); C = float(params["C"])
    arr, is_scalar = _to_array(w_um)
    if np and isinstance(arr, np.ndarray):
        l2 = arr * arr
        n2 = A + B / (l2 - C)
        vals = np.sqrt(n2)
        return float(vals[0]) if is_scalar else vals
    vals = []
    for lam in arr:
        l2 = lam * lam
        vals.append(math.sqrt(A + B / (l2 - C)))
    return vals[0] if is_scalar else vals


def _n_sellmeier_inv(w_um, params):
    """
    Compute n using n^2 = 1 + Σ_i B_i / (1 - (C_i/λ)^2); λ, C_i in μm (fused-silica form).
    
    Args:
        w_um: Wavelength(s) in micrometers
        params: Dictionary with keys B1..Bn and C1..Cn
    
    Returns:
        Refractive index (scalar or array matching w_um shape)
    """
    # expect keys B1..B3 and C1..C3 (C in μm, not squared)
    B = [float(params[f"B{i}"]) for i in (1,2,3) if f"B{i}" in params]
    C = [float(params[f"C{i}"]) for i in (1,2,3) if f"C{i}" in params]
    if len(B) != len(C):
        raise ValueError("sellmeier_inv params must have matching B1..Bn and C1..Cn.")
    arr, is_scalar = _to_array(w_um)
    if np and isinstance(arr, np.ndarray):
        vals = np.ones_like(arr, dtype=float)
        for Bi, Ci in zip(B, C):
            vals += Bi / (1.0 - (Ci / arr) ** 2)
        vals = np.sqrt(vals)
        return float(vals[0]) if is_scalar else vals
    out = []
    for lam in arr:
        n2 = 1.0
        for Bi, Ci in zip(B, C):
            n2 += Bi / (1.0 - (Ci / lam) ** 2)
        out.append(math.sqrt(n2))
    return out[0] if is_scalar else out

# k-models

def _k_constant(w_um, k_value):
    """
    Return constant extinction coefficient.
    
    Args:
        w_um: Wavelength(s) in micrometers (unused, for API consistency)
        k_value: Constant extinction coefficient value
    
    Returns:
        Constant extinction coefficient (scalar or array matching w_um shape)
    """
    return _constant_like(w_um, float(k_value))

# -------- public API --------

def get_n(name, w_um, db_path=None):
    """
    Get refractive index n for a material at given wavelength(s).
    
    Args:
        name: Material name string
        w_um: Wavelength(s) in micrometers (scalar or array-like)
        db_path: Path to material database JSON file (optional)
    
    Returns:
        Refractive index (scalar or array matching w_um shape)
    """
    mats = _load_db(db_path)
    if name not in mats:
        raise KeyError(f"Material '{name}' not found. Available: {available_materials(db_path)}")
    m = mats[name]
    _check_range(m, w_um, name)
    model = m.get("model", "").lower()
    params = m.get("params", {})

    if model == "ab_over_l2_minus_c":
        return _n_ab_over_l2_minus_c(w_um, params)
    elif model == "sellmeier_inv":
        return _n_sellmeier_inv(w_um, params)
    elif model == "constant":
        return _n_constant(w_um, params)
    else:
        raise ValueError(f"Unsupported n-model '{model}' for material '{name}'")


def get_k(name, w_um, db_path=None):
    """
    Get extinction coefficient k for a material at given wavelength(s).
    
    Args:
        name: Material name string
        w_um: Wavelength(s) in micrometers (scalar or array-like)
        db_path: Path to material database JSON file (optional)
    
    Returns:
        Extinction coefficient (scalar or array matching w_um shape)
    """
    mats = _load_db(db_path)
    if name not in mats:
        raise KeyError(f"Material '{name}' not found. Available: {available_materials(db_path)}")
    m = mats[name]
    km = (m.get("k_model") or "constant").lower()
    if km == "constant":
        return _k_constant(w_um, float(m.get("k_params", {}).get("k", 0.0)))
    raise ValueError(f"Unsupported k-model '{km}' for material '{name}'")


def get_nk(name, w_um, db_path=None):
    """
    Get both refractive index n and extinction coefficient k.
    
    Args:
        name: Material name string
        w_um: Wavelength(s) in micrometers (scalar or array-like)
        db_path: Path to material database JSON file (optional)
    
    Returns:
        Tuple of (n, k) where both match w_um shape
    """
    return get_n(name, w_um, db_path), get_k(name, w_um, db_path)


def get_eps(name, w_um, db_path=None):
    """
    Get complex permittivity (n + ik)^2 for a material.
    
    Args:
        name: Material name string
        w_um: Wavelength(s) in micrometers (scalar or array-like)
        db_path: Path to material database JSON file (optional)
    
    Returns:
        Complex permittivity (scalar or array matching w_um shape)
    """
    nn, kk = get_nk(name, w_um, db_path)
    if np and isinstance(nn, np.ndarray):
        return (nn + 1j * kk) ** 2
    if isinstance(nn, list):
        return [(n_i + 1j * k_i) ** 2 for n_i, k_i in zip(nn, kk)]
    return (complex(nn, kk)) ** 2
