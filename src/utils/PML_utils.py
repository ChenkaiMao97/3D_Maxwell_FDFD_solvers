# code snippets taken from SPINS-B: https://github.com/stanfordnqp/spins-b

import numpy as np
import torch
from typing import Callable, List, Optional, Tuple, Union
import itertools

import gin

s_function_type = Callable[[float], float]

GridSpacing = Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[
    np.ndarray, np.ndarray, np.ndarray]]

PmlLayers = Tuple[int, int, int, int, int, int]

@gin.configurable
def ln_R_exp_schedule(epoch: int, max_epoch: int, ln_R_start: float = -1, ln_R_end: float = -16, mult=10):
    # apply a exponential schedule to ln_R
    x = mult * epoch/(max_epoch-1)

    ln_R = np.exp(-(x-np.log(ln_R_start-ln_R_end))) + ln_R_end
    # when x = 0, ln_R = ln_R_start
    # when x = max_epoch-1, ln_R ~ ln_R_end
    return ln_R


def prepare_s_function(ln_R: float = -16, m: float = 4) -> s_function_type:
    """Create an s_function to pass to the SCPML functions.

    This is used when you would like to customize the PML parameters.

    Args:
        ln_R: Natural logarithm of the desired reflectance.
        m: Polynomial order for the PML (imaginary part increases as
            `distance**m`).

    Returns:
        An s_function, which takes an ndarray (distances) and returns an ndarray
        (complex part of the cell width; needs to be divided by
         `sqrt(epilon_effective) * real(omega)` before use).
    """

    def s_factor(distance: np.ndarray) -> np.ndarray:
        s_max = (m + 1) * ln_R / 2  # / 2 because we assume periodic boundaries
        return s_max * (distance**m)

    return s_factor

def stretch_with_scpml(
        dxes: GridSpacing,
        axis: int,
        polarity: int,
        omega: float,
        epsilon_effective: float = 1.0,
        thickness: int = 10,
        s_function: s_function_type = None,
        ln_R: float = -16,
) -> GridSpacing:
    """
        Stretch dxes to contain a stretched-coordinate PML (SCPML) in one direction along one axis.

        :param dxes: dx_tuple with coordinates to stretch
        :param axis: axis to stretch (0=x, 1=y, 2=z)
        :param polarity: direction to stretch (-1 for -ve, +1 for +ve)
        :param omega: Angular frequency for the simulation
        :param epsilon_effective: Effective epsilon of the PML. Match this to the material at the
            edge of your grid. Default 1.
        :param thickness: number of cells to use for pml (default 10)
        :param s_function: s_function created by prepare_s_function(...), allowing customization
            of pml parameters. Default uses prepare_s_function() with no parameters.
        :return: Complex cell widths
    """
    if s_function is None:
        s_function = prepare_s_function(ln_R=ln_R)

    dx_ai = dxes[0][axis].astype(complex)
    dx_bi = dxes[1][axis].astype(complex)

    pos = np.hstack((0, dx_ai.cumsum()))
    pos_a = (pos[:-1] + pos[1:]) / 2
    pos_b = pos[:-1]

    # divide by this to adjust for epsilon_effective and omega
    s_correction = np.sqrt(epsilon_effective) * np.real(omega)

    if polarity > 0:
        # front pml
        bound = pos[thickness]
        d = bound - pos[0]

        def l_d(x):
            return (bound - x) / (bound - pos[0])

        slc = slice(thickness)

    else:
        # back pml
        bound = pos[-thickness - 1]
        d = pos[-1] - bound

        def l_d(x):
            return (x - bound) / (pos[-1] - bound)

        if thickness == 0:
            slc = slice(None)
        else:
            slc = slice(-thickness, None)

    dx_ai[slc] *= 1 + 1j * s_function(l_d(pos_a[slc])) / d / s_correction
    dx_bi[slc] *= 1 + 1j * s_function(l_d(pos_b[slc])) / d / s_correction

    dxes[0][axis] = dx_ai
    dxes[1][axis] = dx_bi

    return dxes

def apply_scpml(dxes: GridSpacing,
                pml_layers: Optional[Union[int, PmlLayers]],
                omega: float,
                ln_R: float = -16) -> GridSpacing:
    """Applies PMLs to the grid spacing.

    This function implements SC-PMLs by modifying the grid spacing based on
    the PML layers.

    Args:
        dxes: Grid spacing to modify.
        pml_layers: Indicates number of PML layers to apply on each side. If
            `None`, no PMLs are applied. If this is a scalar, the same number of
            PML layers are applied to each side.
        omega: Frequency of PML operation.

    Returns:
        A new grid spacing with SC-PML applied.
    """
    # Make a copy of `dxes`. We write this out so that we have an array of
    # array of numpy arrays.
    dxes = [
        [np.array(dxes[grid_num][i]) for i in range(3)] for grid_num in range(2)
    ]

    if isinstance(pml_layers, torch.Tensor):
        assert pml_layers.shape == (6,) or pml_layers.shape[0] == 1, f"unexpected pml_layers shape: {pml_layers.shape}"
        pml_layers = pml_layers.cpu().squeeze().numpy()

    if np.sum(np.abs(np.array(pml_layers))) == 0:
        return dxes

    if isinstance(pml_layers, int):
        pml_layers = [pml_layers] * 6

    for pml, (axis, polarity) in zip(pml_layers,
                                     itertools.product(range(3), [1, -1])):
        if pml > 0:
            dxes = stretch_with_scpml(
                dxes, omega=omega, axis=axis, polarity=polarity, thickness=pml, ln_R=ln_R)

    return dxes

def generate_periodic_dx(pos: List[np.ndarray]) -> GridSpacing:
    """
    Given a list of 3 ndarrays cell centers, creates the cell width parameters for a periodic grid.

    :param pos: List of 3 ndarrays of cell centers
    :return: (dx_a, dx_b) cell widths (no pml)
    """
    if len(pos) != 3:
        raise Exception('Must have len(pos) == 3')

    dx_a = [np.array(np.inf)] * 3
    dx_b = [np.array(np.inf)] * 3

    for i, p_orig in enumerate(pos):
        p = np.array(p_orig, dtype=float)
        if p.size != 1:
            p_shifted = np.hstack((p[1:], p[-1] + (p[1] - p[0])))
            dx_a[i] = np.diff(p)
            dx_b[i] = np.diff((p + p_shifted) / 2)
    return dx_a, dx_b

def make_dxes(wavelength, dL, sizes, pml_layers, ln_R, device):
    x_size, y_size, z_size = sizes
    omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    dxes = ([np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)], [np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)])
    dxes = apply_scpml(dxes, pml_layers, omega, ln_R=ln_R)
    dxes = [[torch.tensor(i).to(device).to(torch.complex64) for i in dxes[0]], [torch.tensor(i).to(device).to(torch.complex64) for i in dxes[1]]]
    return dxes

def make_dxes_numpy(wavelength, dL, sizes, pml_layers, ln_R):
    x_size, y_size, z_size = sizes
    omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    dxes = ([np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)], [np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)])
    dxes = apply_scpml(dxes, pml_layers, omega, ln_R=ln_R)
    return dxes


def conditioners(dxes: GridSpacing,
                 dL: float):
    """ Form the functions for both the preconditioner and postconditioner. """
    #
    #     # Code for the post step function.
    #     code = """
    #         if (_in_global) {
    #             Ex(0,0,0) *= tx1(_X) * ty0(_Y) * tz0(_Z);
    #             Ey(0,0,0) *= tx0(_X) * ty1(_Y) * tz0(_Z);
    #             Ez(0,0,0) *= tx0(_X) * ty0(_Y) * tz1(_Z);
    #         } """
    def reshaper(f):
        for k in range(3):
            new_shape = [1, 1, 1]
            new_shape[k] = np.prod(f[k].shape)
            f[k] = f[k].reshape(new_shape)
        return f

    # Consts that are used.
    sqrt_sc_pml_0 = reshaper([torch.sqrt(s)**1 for s in dxes[1]])
    sqrt_sc_pml_1 = reshaper([torch.sqrt(t)**1 for t in dxes[0]])
    inv_sqrt_sc_pml_0 = reshaper([torch.sqrt(s)**-1 for s in dxes[1]])
    inv_sqrt_sc_pml_1 = reshaper([torch.sqrt(t)**-1 for t in dxes[0]])

    # Define the actual functions.

    def apply_cond(x, t0, t1):
        y = x.clone()
        y[...,0] *= (t1[0] * t0[1] * t0[2]).unsqueeze(0)
        y[...,1] *= (t0[0] * t1[1] * t0[2]).unsqueeze(0)
        y[...,2] *= (t0[0] * t0[1] * t1[2]).unsqueeze(0)
        return y

    def pre_step(x):
        return apply_cond(x, sqrt_sc_pml_0, sqrt_sc_pml_1)

    def post_step(x):
        return apply_cond(x, inv_sqrt_sc_pml_0, inv_sqrt_sc_pml_1)

    return pre_step, post_step