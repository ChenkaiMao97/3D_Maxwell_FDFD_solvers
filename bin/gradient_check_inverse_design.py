import argparse
import concurrent.futures
import copy
import multiprocessing as mp

import gin
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from bin.run_job_utils import seeding
from src.invde.opt import Designer
from src.invde.step_fn import mse_loss_fn
from src.invde.utils.utils import get_integrated_photonics_challenge, get_superpixel_challenge, get_coupling_challenge


def _patch_superpixel_adjoint(variant):
    if variant == "current":
        return

    from src.problems.superpixel import SuperpixelProblem
    from src.utils.utils import c2r, r2c
    import torch

    def simulate_adjoint(self, design_variable, forward_output, grad_outputs):
        epsilon_r = self.epsilon_r(design_variable)

        def _adjoint_simulate(wavelength, forward_E, grad_E):
            if variant.startswith("conj"):
                source_torch = torch.conj(grad_E).to(torch.complex64).resolve_conj()
            else:
                source_torch = grad_E.to(torch.complex64)

            adjoint_E = self.compute_FDFD(wavelength, epsilon_r, source_torch, "adjoint")

            design_variable_torch = design_variable.clone().detach().requires_grad_(True)
            epsilon_for_residual = self.make_torch_epsilon_r(design_variable_torch)[None]
            forward_E_real = c2r(forward_E[None].detach())

            forward_source = torch.zeros(epsilon_r.shape + (3,), dtype=torch.complex64)
            forward_source[
                self.source_xs[0]:self.source_xs[1],
                self.source_ys[0]:self.source_ys[1],
                self.source_zs[0]:self.source_zs[1],
            ] = self.sources[wavelength]
            forward_source = c2r(forward_source[None].detach())

            residual = self.residual_fn(
                forward_E_real,
                epsilon_for_residual,
                forward_source,
                self.pmls,
                self.dL,
                wavelength,
            )

            if variant.endswith("_i"):
                grad_output = 1j * torch.conj(adjoint_E)
                scale = 1.0
            elif variant.endswith("_2"):
                grad_output = torch.conj(adjoint_E)
                scale = 2.0
            else:
                grad_output = torch.conj(adjoint_E)
                scale = 1.0
            return scale * torch.autograd.grad(
                r2c(residual)[0],
                design_variable_torch,
                grad_outputs=grad_output,
            )[0]

        def worker(wavelength):
            wavelength_idx = self.wavelengths.index(wavelength)
            return _adjoint_simulate(
                wavelength,
                forward_output[wavelength_idx],
                grad_outputs[wavelength_idx],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            input_grads = list(executor.map(worker, self.wavelengths))

        return (sum(input_grads) / len(self.wavelengths),)

    SuperpixelProblem.simulate_adjoint = simulate_adjoint


def _make_challenge(design_challenge, key, solver_config):
    if design_challenge == "integrated_photonics":
        return get_integrated_photonics_challenge(key=key, solver_config=solver_config)
    if design_challenge == "superpixel":
        return get_superpixel_challenge(key=key, solver_config=solver_config)
    if design_challenge == "coupling":
        return get_coupling_challenge(key=key)
    raise ValueError(f"Unknown design_challenge: {design_challenge}")


def _set_density(latents, density):
    latents = copy.deepcopy(latents)
    latents["density"].density = density
    return latents


def _clear_solver_warm_starts(designer):
    problem = designer.challenge.problem
    if hasattr(problem, "last_forward_E"):
        problem.last_forward_E.clear()
    if hasattr(problem, "last_adjoint_E"):
        problem.last_adjoint_E.clear()


def _loss(latents, designer):
    _clear_solver_warm_starts(designer)
    value, _ = mse_loss_fn(latents, designer.challenge, designer.latent_to_params, designer.state)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-config", required=True)
    parser.add_argument("--solver-config", required=True)
    parser.add_argument("--design-challenge", default="superpixel")
    parser.add_argument("--load-latent-h5", default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed-id", type=int, default=0)
    parser.add_argument("--direction-seed", type=int, default=123)
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--repeatability-checks", type=int, default=2)
    parser.add_argument(
        "--superpixel-adjoint-variant",
        choices=["current", "conj_2", "conj_1", "conj_i", "raw_2", "raw_1"],
        default="current",
    )
    args = parser.parse_args()

    if args.design_challenge == "superpixel":
        _patch_superpixel_adjoint(args.superpixel_adjoint_variant)

    gin.parse_config_file(args.design_config)
    gin.parse_config_file(args.solver_config)

    key = jax.random.PRNGKey(seeding([args.seed_id]))
    challenge = _make_challenge(args.design_challenge, key, args.solver_config)
    designer = Designer(log_dir=None, challenge=challenge)
    designer.init(key=key)
    if args.beta is not None:
        designer.state.beta = args.beta

    if args.load_latent_h5 is not None:
        with h5py.File(args.load_latent_h5, "r") as f:
            designer.state.latents["density"].density = jnp.asarray(f["latent"][:])

    density = designer.state.latents["density"].density
    direction = jax.random.normal(jax.random.PRNGKey(args.direction_seed), density.shape, dtype=density.dtype)

    # Avoid clipping/boundary effects in the finite-difference perturbation.
    max_eps = max(args.eps)
    mask = (density > max_eps * 2) & (density < 1 - max_eps * 2)
    direction = jnp.where(mask, direction, 0)
    direction = direction / (jnp.sqrt(jnp.mean(direction**2)) + 1e-12)

    _clear_solver_warm_starts(designer)
    (loss_value, _), grad = jax.value_and_grad(
        mse_loss_fn,
        has_aux=True,
    )(designer.state.latents, designer.challenge, designer.latent_to_params, designer.state)
    grad_dir = jnp.sum(grad["density"].density * direction)
    loss_value.block_until_ready()
    grad_dir.block_until_ready()

    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print(f"base_loss: {float(loss_value):.8e}")
    print(f"adjoint_directional_derivative: {float(grad_dir):.8e}")
    print(f"active_direction_fraction: {float(jnp.mean(mask)):.6f}")
    print(f"superpixel_adjoint_variant: {args.superpixel_adjoint_variant}")

    for repeat_idx in range(args.repeatability_checks):
        repeated_loss = _loss(designer.state.latents, designer)
        repeated_loss.block_until_ready()
        print(f"repeat_loss_{repeat_idx}: {float(repeated_loss):.8e}")

    for eps in args.eps:
        plus = _set_density(designer.state.latents, density + eps * direction)
        minus = _set_density(designer.state.latents, density - eps * direction)
        loss_plus = _loss(plus, designer)
        loss_minus = _loss(minus, designer)
        loss_plus.block_until_ready()
        loss_minus.block_until_ready()
        fd = (loss_plus - loss_minus) / (2 * eps)
        rel_err = jnp.abs(fd - grad_dir) / (jnp.maximum(jnp.abs(fd), jnp.abs(grad_dir)) + 1e-12)
        print(
            f"eps={eps:.1e} finite_diff={float(fd):.8e} "
            f"rel_error={float(rel_err):.6e} "
            f"loss_plus={float(loss_plus):.8e} loss_minus={float(loss_minus):.8e}"
        )

    designer.stop_workers()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
