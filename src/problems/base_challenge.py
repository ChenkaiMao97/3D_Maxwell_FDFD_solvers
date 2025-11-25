import functools
import jax
import jax.numpy as jnp
import numpy as onp
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable

from src.invde.utils.jax_utils import box_downsample, kronecker_upsample, extract_fixed_solid_void_pixels, ResponseArray, Density, _get_default_initializer
import src.invde.utils.jax_torch_wrapper as torch_wrapper

import gin

_DENSITY_LABEL = "density"

@gin.configurable
class BaseChallenge:
    """
    The Challenge class defines the problem to be optimized, it takes care of:
    (1) Getting the response from the underlying Problem class (i.e. E and H fields) and define the loss function
    (2) Handling solid and void boundaires for the design variable
    (3) Constructing the jax-compatible simulation function, which involves a custom VJP for building the computational graph.

    Args:
        dL: The size of a design pixel, in nanometers.
        wavelengths: The default wavelengths for simulation.
        pmls: The number of pixels of PML for x_start, x_end, y_start, y_end, z_start, z_end.
        problem_constructor: The constructor for the model.
        density_initializer: The function used to initialize density design variables.
        _backend: The backend to use for simulation.
    """

    def __init__(
        self,
        dL: float,
        wavelengths: Union[onp.ndarray, Sequence[float]],
        pmls: Tuple[int, int, int, int, int, int],
        problem_constructor: Any,  # pyre-ignore[2]
        density_initializer = _get_default_initializer(),
        _backend: str = 'NN',
        solver_config: str = None,
    ):
        """Initializes the challenge.

        Args:
            dL: The size of a design pixel, in nanometers.
            wavelengths: The default wavelengths for simulation.
            problem_constructor: The constructor for the underlying problem.
            density_initializer: The function used to initialize density design variables.
            _backend: The backend to use for simulation.
        """

        # Extract the fixed pixels (i.e. pixels which should be kept solid or void) for the
        # sepecific resolution configuration. Also validates that resolutions are compatible.
        self._fixed_solid: jnp.ndarray
        self._fixed_void: jnp.ndarray
        self._fixed_solid, self._fixed_void = _extract_fixed_solid_void(
            dL=dL,
            wavelengths=wavelengths,
            pmls=pmls,
            problem_constructor=problem_constructor,
        )
        self.problem = problem_constructor(
            dL=dL,
            wavelengths=wavelengths,
            pmls=pmls,
            _backend=_backend,
        )
        self.problem.init()

        assert self.problem.density_dim == 2, "currently we assume design region is 2d pattern extrudded into 3d, so density_dim should be 2"
        self._design_shape = tuple(self.problem.design_variable_shape[0:2])
        self._density_initializer = density_initializer
        self._wavelengths = wavelengths

        # Construct the jax.grad compatible simulation function for the model.
        self._backend = _backend
        if _backend == 'NN':
            self.problem.init_GPU_workers(solver_config=solver_config)
        self._jax_sim_fn = self.construct_jax_sim_fn(self.problem)

    def construct_jax_sim_fn(self,
            problem: Callable,
        ):
        """Constructs the jax-compatible simulation function for the model."""
        if self._backend == 'spins':
            raise NotImplementedError("spins backend autograd is not supported yet")
            # _jax_wrapped_sim_fn = autograd_wrapper.jax_wrap_autograd(
            #     problem.simulate, argnums=0, outputnums=0
            # )
        elif self._backend == 'NN':
            _jax_wrapped_sim_fn = torch_wrapper.jax_wrap_torch(
                problem.simulate, problem.simulate_adjoint, argnums=0
            )
        return _jax_wrapped_sim_fn
    
    def init(self, key: jax.Array):
        """Returns the initial design, and masks indicating fixed pixels.

        The initial design is random, and value depends on the numpy random
        number generator state.

        args:
            key: The key used in random initialization.

        Returns:
            The initial parameters.
        """
        return {
            _DENSITY_LABEL: Density(
                density=self._density_initializer(  # pyre-ignore[28]
                    key=key,
                    shape=self._design_shape,
                    fixed_solid=self._fixed_solid,
                    fixed_void=self._fixed_void,
                ),
                fixed_solid=jnp.asarray(self._fixed_solid) if self._fixed_solid is not None else None,
                fixed_void=jnp.asarray(self._fixed_void) if self._fixed_void is not None else None,
            )
        }

    def response(
        self,
        params
    ):
        """
        Computes the response of the component to excitation.
        params: the density-based design variable
        """
        raise NotImplementedError("Should be implemented in subclasses")
    
    def loss(self, response: ResponseArray):
        raise NotImplementedError("Should be implemented in subclasses")

def _extract_fixed_solid_void(
    dL: float,
    wavelengths: Union[onp.ndarray, Sequence[float]],
    pmls: Tuple[int, int, int, int, int, int],
    problem_constructor: Callable,
):
    """Extracts the fixed solid and void pixels, as appropriate for the resolutions.

    This function also validates that the selected resolution is compatible with the model.

    Args:
        design_resolution_nm: The size of a design pixel, in nanometers.
        sim_resolution_nm: The size of a simulation pixel, in nanometers. The sim resolution must
            be evenly divisible by `design_resolution_nm`, or be an integer multiple thereof.
        problem_constructor: The constructor for the ceviche model.

    Returns:
        The fixed solid and void pixels.
    """
    if pmls[0] == 0 and pmls[1] == 0 and pmls[2] == 0 and pmls[3] == 0:
        return None, None # periodic in x and y, there is no boundary solid or void pixels

    problem = problem_constructor(dL=dL, wavelengths=wavelengths, pmls=pmls, _backend='NN')
    problem.init(shape_only=True)

    density_with_gray_design = problem.density2d(
        onp.ones(problem.design_variable_shape[0:2]) * 0.5
    )
    _fixed_solid, _fixed_void = extract_fixed_solid_void_pixels(
        density_with_gray_design
    )
    fixed_solid = jnp.asarray(_fixed_solid)
    fixed_void = jnp.asarray(_fixed_void)

    return fixed_solid, fixed_void