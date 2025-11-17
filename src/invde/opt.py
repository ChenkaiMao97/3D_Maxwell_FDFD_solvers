import os
import gin
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple
import h5py

import copy
import jax
import jax.numpy as jnp
import numpy as onp
import nlopt
import optax
from tqdm import tqdm

from src.invde.utils.utils import DesignState
from src.problems.base_challenge import BaseChallenge
from src.utils.plot_field3D import plot_3slices, plot_Sr_subplot

import matplotlib.pyplot as plt

@gin.configurable
class Designer:
    def __init__(
        self,
        step_fn: Callable,
        projection_fn: Callable,
        decode_fn: Callable,
        constraints_function: Callable,
        filter_fn: Callable,
        length_scale_pixel: float,
        opt_type: str,
        steps_per_beta: int,
        latent_bounds: Tuple[float, float],
        beta_schedule: List[float],
        constraint_tolerance_schedule: List[float] = None,
        constraint_weight_schedule: List[float] = None,
        steps_per_beta_during_constraint: Optional[int] = None,
        lr: float = None,
        end_lr: float = None,
        log_fn_type: str = "integrated_photonics",
        challenge: BaseChallenge = None,
        latent_init_fn: Optional[Callable] = None,
        latent_mean: Optional[float] = None,
        latent_std: Optional[float] = None,
        final_step_discretization: str = 'binary',
        log_dir: Optional[str] = None,
        save_bin: bool=False,
        load_latent_path: Optional[str] = None
    ) -> None:
        self.step_fn = step_fn
        self.projection_fn = projection_fn
        self.decode_fn = decode_fn
        self.constraints_function = constraints_function
        self.filter_fn = filter_fn
        self.length_scale_pixel = length_scale_pixel
        self.opt_type = opt_type
        self.steps_per_beta = steps_per_beta
        self.latent_bounds = latent_bounds
        self.lower_bound = latent_bounds[0]
        self.upper_bound = latent_bounds[1]
        self.final_step_discretization = final_step_discretization

        self.beta_schedule = beta_schedule
        self.constraint_tolerance_schedule = constraint_tolerance_schedule
        if constraint_tolerance_schedule is not None:
            assert len(constraint_tolerance_schedule) == len(beta_schedule), "constraint_tolerance_schedule must be the same length as beta_schedule"
        self.constraint_weight_schedule = constraint_weight_schedule
        if constraint_weight_schedule is not None:
            assert len(constraint_weight_schedule) == len(beta_schedule), "constraint_weight_schedule must be the same length as beta_schedule"
        
        self.steps_per_beta_during_constraint = steps_per_beta_during_constraint
        self.lr = lr
        self.end_lr = end_lr
        self.log_fn_type = log_fn_type
        self.challenge = challenge
        self.latent_init_fn = latent_init_fn
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.log_dir = log_dir
        self.save_bin = save_bin
        self.load_latent_path = load_latent_path

    def init(self, key) -> None:
        # This should return a list of densities, one "image" for each layer.
        latents = self.challenge.init(key)
        density = latents['density']
        density_shape = density.shape

        self.state = DesignState(
            step=0,
            beta_schedule_step=0,
            latents=latents,
            params=self.latent_to_params(
                latents=latents,
                beta=self.beta_schedule[0],
            ),
            beta=self.beta_schedule[0],
            latent_shape=density_shape,
            density_shape=density_shape,
        )

        if self.load_latent_path is not None:
            print("continuing optimization from", self.load_latent_path)
            with h5py.File(self.load_latent_path, "r") as f:
                self.state.latents["density"].density = f["latent"][:]
                self.state.params["density"].density = f["params"][:]

    def latent_to_params(
        self,
        latents: jnp.ndarray,
        beta: float
    ) -> Dict[str, Any]:
        params = copy.deepcopy(latents)
        layer = self.decode_fn(params['density'].density)
        layer = self.filter_fn(layer, radius=self.length_scale_pixel)
        layer = self.projection_fn(layer, beta=beta)
        params['density'].density = layer
        return params

    def latent_to_binarized_params(
        self,
        latents: jnp.ndarray,
        beta: float
    ) -> Dict[str, Any]:
        params = copy.deepcopy(latents)
        layer = self.decode_fn(params['density'].density)
        layer = self.filter_fn(layer, radius=self.length_scale_pixel)
        layer = self.projection_fn(layer, beta=beta)
        layer = jnp.where(layer > 0.5, 1, 0)
        params['density'].density = layer
        return params
    
    @gin.configurable
    def latent_to_multilevel_params(
        self,
        latents: jnp.ndarray,
        beta: float,
        num_levels=20
    ) -> Dict[str, Any]:
        params = copy.deepcopy(latents)
        layer = self.decode_fn(params['density'].density)
        layer = self.filter_fn(layer, radius=self.length_scale_pixel)
        layer = self.projection_fn(layer, beta=beta)

        # discretize layer to num_layer of discrete values
        vmin, vmax = layer.min(), layer.max()
        bins = onp.linspace(vmin, vmax, num_levels + 1)
        discretized = onp.digitize(layer, bins) - 1
        discretized = onp.clip(discretized, 0, num_levels - 1)
        layer = vmin + discretized * (vmax-vmin)/(num_levels - 1)

        params['density'].density = layer
        return params
    
    def optimize(self):
        if self.opt_type == "nlopt":
            return self.optimize_nlopt()
        elif self.opt_type == "optax":
            return self.optimize_optax()
        else:
            raise ValueError(f"Invalid optimizer type: {self.opt_type}")

    def optimize_optax(self):
        assert self.state is not None, "Please call init to initialize state"
        n_params = reduce(lambda x, y: x * y, self.state.latents["density"].density.shape)

        total_steps = len(self.beta_schedule) * self.steps_per_beta + (self.steps_per_beta_during_constraint - self.steps_per_beta) * sum([i>0 for i in self.constraint_weight_schedule])
        print("total_steps", total_steps, 'init_lr', self.lr, 'end_lr', self.end_lr)
        schedule = optax.exponential_decay(
            init_value=self.lr,
            transition_steps=total_steps,
            decay_rate=self.end_lr / self.lr,
            staircase=False  # for smooth decay
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(self.state.latents)

        print("self.beta_schedule", self.beta_schedule)
        print("self.constraint_weight_schedule", self.constraint_weight_schedule)
        
        with tqdm(
            total=len(self.beta_schedule), initial=self.state.beta_schedule_step
        ) as pbar:
            while self.state.beta_schedule_step < len(self.beta_schedule):
                beta = self.beta_schedule[self.state.beta_schedule_step]
                constraint_weight = self.constraint_weight_schedule[self.state.beta_schedule_step]

                print("Beta:", beta, "Constraint weight:", constraint_weight)
                nlopt_loss, loss_fn = self.step_fn(
                    state=self.state,
                    decode_fn=self.decode_fn,
                    latent_to_param_fn=self.latent_to_params,
                    latent_shape=self.state.latent_shape,
                    challenge=self.challenge,
                    log_fn=self.log_step
                )
                self.state.beta = beta

                if constraint_weight > 0:
                    constraint_beta = beta
                    _, constraint_value_and_grad = self.constraints_function(
                        beta=constraint_beta,
                        length_scale_pixel=self.length_scale_pixel,
                        projection_fn=self.projection_fn,
                        filter_fn=self.filter_fn,
                        decode_fn=self.decode_fn,
                        latent_shape=self.state.latent_shape,
                        constraint_tolerance=0,
                        num_layers=1
                    )

                opt_steps = self.steps_per_beta_during_constraint if constraint_weight > 0 else self.steps_per_beta
                for step in range(opt_steps):
                    print("step", step)
                    (loss_value, (response, aux, params)), my_grad = jax.value_and_grad(loss_fn, has_aux=True)(self.state.latents, self.challenge, self.latent_to_params, self.state)

                    solid_v, void_v = 0, 0
                    if constraint_weight > 0:
                        solid_v, void_v, solid_g, void_g = constraint_value_and_grad(self.state.latents["density"].density)
                        grad_loss_mean = jnp.mean(jnp.abs(my_grad['density'].density))
                        grad_const_s_mean = jnp.mean(jnp.abs(solid_g))
                        grad_const_v_mean = jnp.mean(jnp.abs(void_g))

                        my_grad["density"].density += constraint_weight * grad_loss_mean / grad_const_s_mean * solid_g
                        my_grad["density"].density += constraint_weight * grad_loss_mean / grad_const_v_mean * void_g
                    
                    # sanity check for gradient using finite difference:
                    # print("sanity check for gradient using finite difference:")
                    # epsilon = 1e-3
                    # new_latent = copy.deepcopy(self.state.latents)

                    # key = jax.random.PRNGKey(0)
                    # new_key, subkey = jax.random.split(key)
                    # v = jax.random.normal(subkey, shape=new_latent["density"].density.shape, dtype=new_latent["density"].density.dtype)
                    # dx = epsilon * v

                    # mask1 = jnp.where(new_latent["density"].density + dx < 1, 1, 0)
                    # mask2 = jnp.where(new_latent["density"].density - dx > 0, 1, 0)
                    # dx = dx * mask1 * mask2
                    # v = v * mask1 * mask2

                    # new_latent["density"].density = new_latent["density"].density + dx
                    # new_loss1, (response, aux, params) = loss_fn(
                    #     latents=new_latent,
                    #     challenge=self.challenge,
                    #     latent_to_param_fn=self.latent_to_params,
                    #     state=self.state
                    # )
                    # new_latent["density"].density = new_latent["density"].density - 2*dx
                    # new_loss2, (response, aux, params) = loss_fn(
                    #     latents=new_latent,
                    #     challenge=self.challenge,
                    #     latent_to_param_fn=self.latent_to_params,
                    #     state=self.state
                    # )
                    # finite_difference_grad = (new_loss1 - new_loss2) / (2 * epsilon)
                    # computed_grad = jnp.sum(v * my_grad["density"].density)
                    # print(f'finite difference grad: {finite_difference_grad:.3f}, computed grad: {computed_grad:.3f}')

                    updates, opt_state = opt.update(my_grad, opt_state)
                    self.state.latents = optax.apply_updates(self.state.latents, updates)
                    self.state.latents["density"].density = jnp.clip(self.state.latents["density"].density, self.lower_bound, self.upper_bound)
                    print(f'step {step} loss {loss_value:.3f} constraint values: {solid_v:.3f}, {void_v:.3f}, lr: {schedule(self.state.step):.2e}')

                    self.state.params = params
                    self.state.loss.append(float(loss_value))
                    self.state.step += 1

                    self.log_step(my_grad, params, response, aux)
                
                pbar.update(1)
                self.state.beta_schedule_step += 1

        if self.final_step_discretization == 'binary':
            final_latent_to_param_fn = self.latent_to_binarized_params
        elif self.final_step_discretization == 'multilevel':
            final_latent_to_param_fn = self.latent_to_multilevel_params
        elif self.final_step_discretization == 'grayscale':
            final_latent_to_param_fn = self.latent_to_params
        else:
            raise ValueError(f"final_step_discretization {self.final_step_discretization} not recognized")

        loss_value, (response, aux, params) = loss_fn(
            latents=self.state.latents,
            challenge=self.challenge,
            latent_to_param_fn=final_latent_to_param_fn,
            state=self.state
        )
        self.state.params = params

        if self.log_dir is not None:
            self.log_final_loss_and_response(response, aux, params)

        return self.state
    
    def optimize_nlopt(self):
        assert self.state is not None, "Please call init to initialize state"
        n_params = reduce(lambda x, y: x * y, self.state.latents["density"].density.shape)
        x = self.state.latents["density"].density.flatten()

        print("self.beta_schedule", self.beta_schedule)
        print("self.constraint_tolerance_schedule", self.constraint_tolerance_schedule)

        with tqdm(
            total=len(self.beta_schedule), initial=self.state.beta_schedule_step
        ) as pbar:
            while self.state.beta_schedule_step < len(self.beta_schedule):
                beta = self.beta_schedule[self.state.beta_schedule_step]
                constraint_tolerance = self.constraint_tolerance_schedule[
                    self.state.beta_schedule_step
                ]

                print("Beta:", beta)
                print("Constraint Tolerance:", constraint_tolerance)
                opt = nlopt.opt(nlopt.LD_CCSAQ, int(n_params))
                opt.set_lower_bounds(self.lower_bound)
                opt.set_upper_bounds(self.upper_bound)
                nlopt_loss, loss_fn = self.step_fn(
                    state=self.state,
                    decode_fn=self.decode_fn,
                    latent_to_param_fn=self.latent_to_params,
                    latent_shape=self.state.latent_shape,
                    challenge=self.challenge,
                    log_fn = self.log_step
                )
                self.state.beta = beta
                opt.set_min_objective(nlopt_loss)
                if constraint_tolerance > 0:
                    constraint_beta = beta
                    constr_fn, _ = self.constraints_function(
                        beta=constraint_beta,
                        length_scale_pixel=self.length_scale_pixel,
                        projection_fn=self.projection_fn,
                        filter_fn=self.filter_fn,
                        decode_fn=self.decode_fn,
                        latent_shape=self.state.latent_shape,
                        constraint_tolerance=constraint_tolerance,
                        num_layers=1
                    )
                    opt.add_inequality_mconstraint(constr_fn, 2*[1e-8])
                    opt.set_maxeval(
                        self.steps_per_beta_during_constraint or self.steps_per_beta
                    )
                else:
                    opt.set_maxeval(self.steps_per_beta)

                x = opt.optimize(x)
                self.state.latents["density"].density = jnp.reshape(jnp.asarray(x), self.state.latent_shape)
                pbar.update(1)
                self.state.beta_schedule_step += 1

        if self.final_step_discretization == 'binary':
            final_latent_to_param_fn = self.latent_to_binarized_params
        elif self.final_step_discretization == 'multilevel':
            final_latent_to_param_fn = self.latent_to_multilevel_params
        elif self.final_step_discretization == 'grayscale':
            final_latent_to_param_fn = self.latent_to_params
        else:
            raise ValueError(f"final_step_discretization {self.final_step_discretization} not recognized")

        loss_value, (response, aux, params) = loss_fn(
            latents=self.state.latents,
            challenge=self.challenge,
            latent_to_param_fn=final_latent_to_param_fn,
            state=self.state
        )
        self.state.params = params

        if self.log_dir is not None:
            self.log_final_loss_and_response(response, aux, params)

        # prepare the data for DDM:
       
        return input_eps, Ez_out_forward_RI, source_RI, wls, dLs, self.state
    
    def stop_workers(self):
        self.challenge.problem.stop_workers()
    
    def log_step(self, my_grad, params, response, aux):
        if self.log_fn_type == "integrated_photonics":
            self.log_step_integrated_photonics(my_grad, params, response, aux)
        elif self.log_fn_type == "superpixel":
            self.log_step_superpixel(my_grad, params, response, aux)
        else:
            raise ValueError(f"Invalid log_fn_type: {self.log_fn_type}")

    def log_step_integrated_photonics(self, my_grad, params, response, aux):
        # log step
        print("sparam: ", aux)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

            num_wls = len(self.challenge._wavelengths)
            plt.figure(figsize=((num_wls+3)*4, 4))
            plt.subplot(1,num_wls+3,1)
            plt.imshow(onp.rot90(self.state.latents["density"].density), cmap="binary")
            plt.title("latent")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(1,num_wls+3,2)
            plt.imshow(onp.rot90(params["density"].density), cmap="binary")
            plt.title(f"params\nstep: {self.state.step}")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(1,num_wls+3,3)
            vm = jnp.max(jnp.abs(my_grad["density"].density))
            plt.imshow(onp.rot90(my_grad["density"].density), cmap="seismic", vmin=-vm, vmax=vm)
            plt.title("grad")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            couple_eff = jnp.abs(aux) ** 2
            for i in range(num_wls):
                max_port = jnp.argmax(self.challenge.target_s_params[i])
                plt.subplot(1,num_wls+3,i+4)
                E = response[i]
                sx, sy, sz, _ = E.shape
                center_z_E = E[:,:,round(1/2*(sz-1)),:]
                plt.imshow(onp.rot90(center_z_E[...,-1].real), cmap="seismic")
                plt.title(f"Ez real at z=0, wl {self.challenge._wavelengths[i]}\nport {max_port} couple eff: {couple_eff[i,max_port]:.3f}")
                eps = self.challenge.problem.epsilon_r(params["density"].density)
                eps_plot = eps[:,:,round(1/2*(sz-1))]
                plt.imshow(onp.rot90(eps_plot), cmap="binary", alpha=0.3)
                # h, w = eps_plot.shape
                for port in self.challenge.problem._ports:
                    coords = port.coords()
                    plt.plot((coords[0][0], coords[1][0]), (coords[0][1], coords[1][1]), color='red', linewidth=2)
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"step_{self.state.step}.png"))
            plt.close()

            if self.save_bin and self.state.step % 10 == 0:
                with h5py.File(os.path.join(self.log_dir, f"step_{self.state.step}.h5"), "w") as f:
                    f["latent"] = self.state.latents["density"].density
                    f["params"] = params["density"].density
                    f["grad"] = my_grad["density"].density
                    f["ez"] = onp.stack(response)[...,-1]
                    f["eps"] = self.challenge.problem.epsilon_r(params["density"].density)
    
    def log_step_superpixel(self, my_grad, params, response, aux):
        # log step
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

            num_wls = len(self.challenge._wavelengths)
            fig = plt.figure(figsize=((2*num_wls+3)*4, 4))
            plt.subplot(1,2*num_wls+3,1)
            plt.imshow(onp.rot90(self.state.latents["density"].density), cmap="binary")
            plt.title("latent")
            plt.colorbar()
            plt.subplot(1,2*num_wls+3,2)
            plt.imshow(onp.rot90(params["density"].density), cmap="binary")
            plt.title(f"params\nstep: {self.state.step}")
            plt.colorbar()
            plt.subplot(1,2*num_wls+3,3)
            vm = jnp.max(jnp.abs(my_grad["density"].density))
            plt.imshow(onp.rot90(my_grad["density"].density), cmap="seismic", vmin=-vm, vmax=vm)
            plt.title("grad")
            plt.colorbar()

            for idx, (wl, data) in enumerate(aux.items()):
                u0, Sr, Ex, thetas, phis = data
                # print("total Sr: ", onp.sum(onp.array(Sr)))
                # assert (Sr>0).all(), "Sr should be positive"
                # max_idx = jnp.argmax(Sr)
                # max_th = thetas[max_idx] * 180 / jnp.pi
                # max_phi = phis[max_idx] * 180 / jnp.pi

                ax = plt.subplot(1,2*num_wls+3,4+2*idx, projection='3d')
                # sc = plot_Sr_subplot(onp.array(u0), onp.array(Sr), ax=ax, point_size=8)
                # fig.colorbar(sc, ax=ax)
                plt.title(f"Sr for wl {wl}")
                ax = plt.subplot(1,2*num_wls+3,4+2*idx+1)
                plt.imshow(onp.rot90(Ex), cmap="seismic")
                plt.colorbar()
                # plt.title(f"Ex for wl {wl}\nmax_th: {max_th:.2f}, max_phi: {max_phi:.2f}")

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"step_{self.state.step}.png"))
            plt.close()

            if self.save_bin and self.state.step % 10 == 0:
                with h5py.File(os.path.join(self.log_dir, f"step_{self.state.step}.h5"), "w") as f:
                    f["latent"] = self.state.latents["density"].density
                    f["params"] = params["density"].density
                    f["grad"] = my_grad["density"].density
                    f["ez"] = onp.stack(response)[...,-1]
                    f["eps"] = self.challenge.problem.epsilon_r(params["density"].density)
    


    # def log_step_gc(self, my_grad, params, response, aux):
    #     # log step
    #     if self.log_dir is not None:
    #         os.makedirs(self.log_dir, exist_ok=True)

    #         num_wls = len(self.challenge._wavelengths)
    #         plt.figure(figsize=((num_wls+3)*4, 4))
    #         plt.subplot(1,num_wls+3,1)
    #         plt.imshow(self.state.latents["density"].density, cmap="binary")
    #         plt.title("latent")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.colorbar()
    #         plt.subplot(1,num_wls+3,2)
    #         plt.imshow(params["density"].density, cmap="binary")
    #         plt.title(f"params\nstep: {self.state.step}")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.colorbar()
    #         plt.subplot(1,num_wls+3,3)
    #         vm = jnp.max(jnp.abs(my_grad["density"].density))
    #         plt.imshow(my_grad["density"].density, cmap="seismic", vmin=-vm, vmax=vm)
    #         plt.title("grad")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.colorbar()

    #         couple_eff = jnp.abs(response.array) ** 2
    #         for i in range(num_wls):
    #             plt.subplot(1,num_wls+3,i+4)
    #             plt.imshow(aux["fields_ez"][i,:,:].real, cmap="seismic")
    #             plt.title(f"Ez real {self.challenge._wavelengths[i]} nm\n couple eff: {couple_eff[i]:.3f}")
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.colorbar()
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(self.log_dir, f"step_{self.state.step}.png"))
    #         plt.close()

    #         if self.save_bin:
    #             with h5py.File(os.path.join(self.log_dir, f"step_{self.state.step}.h5"), "w") as f:
    #                 f["latent"] = self.state.latents["density"].density
    #                 f["params"] = params["density"].density
    #                 f["grad"] = my_grad["density"].density
    #                 f['ez'] = aux["fields_ez"]
    
    # def log_step_meta(self, my_grad, params, response, aux):
    #     # log step
    #     if self.log_dir is not None:
    #         os.makedirs(self.log_dir, exist_ok=True)

    #         num_wls = len(self.challenge._wavelengths)
    #         plt.figure(figsize=((num_wls+3)*4, 4))
    #         plt.subplot(1,num_wls+3,1)
    #         plt.imshow(self.state.latents["density"].density, cmap="binary")
    #         plt.title("latent")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.colorbar()
    #         plt.subplot(1,num_wls+3,2)
    #         plt.imshow(params["density"].density, cmap="binary")
    #         plt.title(f"params\nstep: {self.state.step}")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.colorbar()
    #         plt.subplot(1,num_wls+3,3)
    #         vm = jnp.max(jnp.abs(my_grad["density"].density))
    #         plt.imshow(my_grad["density"].density, cmap="seismic", vmin=-vm, vmax=vm)
    #         plt.title("grad")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.colorbar()

    #         transmission_eff = aux['FOM']
    #         for i in range(num_wls):
    #             plt.subplot(1,num_wls+3,i+4)
    #             plt.imshow(aux["fields_ez"][i,:,:].real, cmap="seismic")
    #             plt.title(f"Ez real {self.challenge._wavelengths[i]} nm\n couple eff: {transmission_eff[i]:.3f}")
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.colorbar()
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(self.log_dir, f"step_{self.state.step}.png"))
    #         plt.close()

    #         if self.save_bin:
    #             with h5py.File(os.path.join(self.log_dir, f"step_{self.state.step}.h5"), "w") as f:
    #                 f["latent"] = self.state.latents["density"].density
    #                 f["params"] = params["density"].density
    #                 f["grad"] = my_grad["density"].density
    #                 f['ez'] = aux["fields_ez"]

    def log_final_loss_and_response(self, response, aux, params):
        if self.log_fn_type == "integrated_photonics":
            self.log_final_integrated_photonics(response, aux, params)
        elif self.log_fn_type == "superpixel":
            self.log_final_superpixel(response, aux, params)
        else:
            raise ValueError(f"Invalid log_fn_type: {self.log_fn_type}")

    def log_final_integrated_photonics(self, response, aux, params):
        # log the final loss and response:
        os.makedirs(self.log_dir, exist_ok=True)

        num_wls = len(self.challenge._wavelengths)
        plt.figure(figsize=((num_wls+2)*4, 4))
        plt.subplot(1,num_wls+2,1)
        plt.imshow(onp.rot90(self.state.latents["density"].density), cmap="binary")
        plt.title("latent")
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()
        plt.subplot(1,num_wls+2,2)
        plt.imshow(onp.rot90(params["density"].density), cmap="binary")
        plt.title("params")
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()

        couple_eff = jnp.abs(aux) ** 2
        for i in range(num_wls):
            max_port = jnp.argmax(self.challenge.target_s_params[i])
            plt.subplot(1,num_wls+2,i+3)
            E = response[i]
            sx, sy, sz, _ = E.shape
            center_z_E = E[:,:,round(1/2*(sz-1)),:]
            plt.imshow(onp.rot90(center_z_E[...,-1].real), cmap="seismic")
            eps = self.challenge.problem.epsilon_r(params["density"].density)
            eps_plot = eps[:,:,round(1/2*(sz-1))]
            plt.imshow(onp.rot90(eps_plot), cmap="binary", alpha=0.3)
            plt.title(f"Ez real at z=0, wl {self.challenge._wavelengths[i]}\nport {max_port} couple eff: {couple_eff[i,max_port]:.3f}")
            plt.xticks([])
            plt.yticks([])
            # plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"final_step.png"))
        plt.close()

        eps = self.challenge.problem.epsilon_r(params["density"].density)
        onp.save(os.path.join(self.log_dir, f"final_eps.npy"), eps)
        onp.save(os.path.join(self.log_dir, f"final_latent.npy"), self.state.latents["density"].density)
        onp.save(os.path.join(self.log_dir, f"final_params.npy"), params["density"].density)
        onp.save(os.path.join(self.log_dir, f"final_fields.npy"), jnp.stack(response))

    def log_final_superpixel(self, response, aux, params):
        # log step
        os.makedirs(self.log_dir, exist_ok=True)

        num_wls = len(self.challenge._wavelengths)
        plt.figure(figsize=((2*num_wls+3)*4, 4))
        plt.subplot(1,2*num_wls+3,1)
        plt.imshow(onp.rot90(self.state.latents["density"].density), cmap="binary")
        plt.title("latent")
        plt.colorbar()
        plt.subplot(1,2*num_wls+3,2)
        plt.imshow(onp.rot90(params["density"].density), cmap="binary")
        plt.title(f"params\nstep: {self.state.step}")
        plt.colorbar()
        plt.subplot(1,2*num_wls+3,3)
        vm = jnp.max(jnp.abs(my_grad["density"].density))
        plt.imshow(onp.rot90(my_grad["density"].density), cmap="seismic", vmin=-vm, vmax=vm)
        plt.title("grad")
        plt.colorbar()

        for idx, (wl, data) in enumerate(aux.items()):
            u0, Sr, Ex, theta_range, phi_range = data
            max_th_idx, max_phi_idx = jnp.unravel_index(jnp.argmax(jnp.abs(Sr)), Sr.shape)

            max_th = theta_range[max_th_idx] * 180 / jnp.pi
            max_phi = phi_range[max_phi_idx] * 180 / jnp.pi

            ax = plt.subplot(1,2*num_wls+3,4+2*idx, projection='3d')
            plot_Sr_subplot(onp.array(u0), onp.array(Sr), ax=ax, point_size=8)
            plt.title(f"Sr for wl {wl}")
            ax = plt.subplot(1,2*num_wls+3,4+2*idx+1)
            plt.imshow(onp.rot90(Ex), cmap="seismic")
            plt.colorbar()
            plt.title(f"Ex for wl {wl}\nmax_th: {max_th:.2f}, max_phi: {max_phi:.2f}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"final_step.png"))
        plt.close()

        eps = self.challenge.problem.epsilon_r(params["density"].density)
        onp.save(os.path.join(self.log_dir, f"final_eps.npy"), eps)
        onp.save(os.path.join(self.log_dir, f"final_latent.npy"), self.state.latents["density"].density)
        onp.save(os.path.join(self.log_dir, f"final_params.npy"), params["density"].density)
        onp.save(os.path.join(self.log_dir, f"final_fields.npy"), onp.stack(response))
    
    # def log_final_gc(self, response, aux, params):
    #     # log the final loss and response:
    #     os.makedirs(self.log_dir, exist_ok=True)

    #     num_wls = len(self.challenge._wavelengths)
    #     # plt.figure(figsize=(4, 4))
    #     # plt.subplot(1,num_wls+2,1)
    #     plt.imsave(os.path.join(self.log_dir, f"final_step_param.png"), params, cmap="binary", dpi=300)
    #     couple_eff = jnp.abs(response.array) ** 2
    #     for i in range(num_wls):
    #         vm = onp.max(onp.abs(aux["fields_ez"][i,:,:].real))
    #         plt.imsave(os.path.join(self.log_dir, f"final_step_{self.challenge._wavelengths[i]}_mm_eff_{couple_eff[i]:.3f}.png"), aux["fields_ez"][i,:,:].real, cmap="seismic", dpi=300, vmax=vm, vmin=-vm)

    #     onp.save(os.path.join(self.log_dir, f"final_latent.npy"), self.state.latents["density"].density)
    #     onp.save(os.path.join(self.log_dir, f"final_params.npy"), params)
    #     onp.save(os.path.join(self.log_dir, f"final_fields.npy"), aux["fields_ez"])
    
    # def log_final_meta(self, response, aux, params):
    #     # log the final loss and response:
    #     os.makedirs(self.log_dir, exist_ok=True)

    #     num_wls = len(self.challenge._wavelengths)
    #     # plt.figure(figsize=(4, 4))
    #     # plt.subplot(1,num_wls+2,1)
    #     plt.imsave(os.path.join(self.log_dir, f"final_step_param.png"), params, cmap="binary", dpi=300)
    #     transmission_eff = aux['FOM']
    #     for i in range(num_wls):
    #         vm = onp.max(onp.abs(aux["fields_ez"][i,:,:].real))
    #         plt.imsave(os.path.join(self.log_dir, f"final_step_{self.challenge._wavelengths[i]}_mm_eff_{transmission_eff[i]:.3f}.png"), aux["fields_ez"][i,:,:].real, cmap="seismic", dpi=300, vmax=vm, vmin=-vm)

    #     onp.save(os.path.join(self.log_dir, f"final_latent.npy"), self.state.latents["density"].density)
    #     onp.save(os.path.join(self.log_dir, f"final_params.npy"), params)
    #     onp.save(os.path.join(self.log_dir, f"final_fields.npy"), aux["fields_ez"])