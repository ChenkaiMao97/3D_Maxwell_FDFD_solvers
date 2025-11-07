from typing import List, Tuple, Optional, Union

import numpy as np
import os, sys
import torch
from matplotlib import pyplot as plt

# a temporary local path for the modified spins library
sys.path.append("/home/maxcmk/Desktop/spins/spins_env/lib/python3.8/site-packages")
from spins.fdfd_tools.waveguide_mode import solve_waveguide_mode, compute_source, compute_overlap_e

from src.utils.plot_field3D import plot_3slices
from src.utils.physics import residue_E, eps_to_yee
from src.utils.PML_utils import make_dxes_numpy
import time

from dataclasses import dataclass, field
import enum

Geometry = np.ndarray
Slice = Tuple[np.ndarray, np.ndarray]
Field = np.ndarray
VectorField = Tuple[Union[Field, float], Union[Field, float], Union[Field, float]]

from typing import List

from enum import Enum
import numpy as np

class Direction(Enum):
    """
    Enum for axis->integer mapping
    """
    x = 0
    y = 1
    z = 2

def axisvec2polarity(vector: np.ndarray) -> int:
    """Return the polarity along the vector's primary coordinate axis.

     Args:
         vector: The direction vector.

     Returns:
         The polarity of the vector, which is either 1 (for positive direction)
         and -1 (for negative direction).
    """
    if isinstance(vector, List):
        vec = np.array(vector)
    else:
        vec = vector

    axis = axisvec2axis(vec)

    return np.sign(vec[axis])


def axisvec2axis(vector: np.ndarray) -> int:
    """Return the vector's primary coordinate axis.

     Args:
         vector: The direction vector.

     Returns:
         axis: Direction axis.

     Raises:
         ValueError: If the vector is not axis-aligned.
    """
    if isinstance(vector, List):
        vec = np.array(vector)
    else:
        vec = vector

    norm = np.linalg.norm(vec)
    delta = 1e-6 * norm

    # Check that only one element of vector is larger than delta.
    if sum(abs(vec) > delta) != 1:
        raise ValueError(
            "Vector has no valid primary coordinate axis, got: {}".format(vec))

    axis = np.argwhere(abs(vec) > delta).flatten()[0]

    return axis


@dataclass
class Waveguide:
    """A generic waveguide, represented as a slice (a line of pixels) in a 2D domain.
    """
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    z_start: int
    z_end: int
    eps: float

@dataclass
class WaveguidePort:
    """A generic WaveguidePort, represented as a slice (a line of pixels) in a 2D domain.

    Attributes:
        x: `int` specifying the x-coordinate of the center of the mode slice, in
            pixel units.
        y: `int` specifying the y-coordinate of the center of the mode slice, in
            pixel units.
        width: `int` specifying the transverse width of the mode slice, in pixel
            units.
        dir: `Direction` specifying the direction perpendicular to the mode slice.
        offset: `int` specfying distance from the source slice to perform the
            decomposition of the forward and backward mode amplitudes.
    """
    x: int
    y: int
    z: int
    width: int
    height: int # for extending into 3d
    axis_vector: np.array
    offset: int
    order: int
    Js: dict = field(default_factory=dict)
    overlap_es: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.width % 2 == 1 or self.height % 2 == 1:
            raise ValueError('Odd values for the port width and height are currently not supported')

    def coords(self):
        """Generate coordinate vectors for slicing in/out of 2D arrays."""
        # pylint:disable=g-bad-todo
        # TODO: correctly handle an odd width value, rather than round off
        direction = axisvec2axis(self.axis_vector)
        if direction == 0:
            x_start, x_end = self.x, self.x
            y_start, y_end = self.y - self.width // 2, self.y + self.width // 2
        elif direction == 1:
            x_start, x_end = self.x - self.width // 2, self.x + self.width // 2
            y_start, y_end = self.y, self.y
        else:
            raise ValueError(f"Invalid direction: {direction}")
        z_start, z_end = self.z - self.height // 2, self.z + self.height // 2

        return [(x_start, y_start, z_start), (x_end, y_end, z_end)]
    
    def precompute_mode(
        self,
        wavelengths,
        dL,
        epsilon_r,
        pml_layers,
        power=1,
        ln_R=-10,
        precompute_source=False # only the input port needs to compute source, others only need to compute overlap_e
    ):
        # direction: e.g. [0,0,1], [0,-1,0]
        # src_slice: e.g. [(start_x, start_y, start_z), (end_x, end_y, end_z)] (inclusive)
        for wavelength in wavelengths:
            omega=2 * np.pi / wavelength
            dxes = make_dxes_numpy(wavelength, dL, epsilon_r.shape, pml_layers, ln_R)
            axis = Direction(axisvec2axis(self.axis_vector))

            eps_yee = eps_to_yee(torch.from_numpy(epsilon_r[None]))[0].permute(3,0,1,2).numpy()
            # mu = np.ones_like(eps_yee)

            sim_params = {
                'omega': omega,
                'dxes': dxes,
                'axis': axis.value,
                'slices': tuple([slice(i, f+1) for i, f in zip(*self.coords())]),
                'polarity': axisvec2polarity(self.axis_vector),
                # 'mu': mu
            }

            wgmode_result = solve_waveguide_mode(
                mode_number = self.order,
                epsilon = eps_yee,
                **sim_params
            )

            overlap_e = compute_overlap_e(**wgmode_result, **sim_params)
            self.overlap_es[wavelength] = overlap_e

            if precompute_source:
                J = compute_source(**wgmode_result, **sim_params)
                for k in range(len(J)):
                    J[k] *= np.sqrt(power)
                self.Js[wavelength] = J
    
    def source(
        self,
        wavelength
    ):
        assert wavelength in self.Js, f"source is not precomputed for wavelength: {wavelength}"
        return self.Js[wavelength]
    
    def overlap_e(
        self,
        wavelength
    ):
        assert wavelength in self.overlap_es, f"overlap_e is not precomputed for wavelength: {wavelength}"
        return self.overlap_es[wavelength]