# this is a temporary custom implementation of the spins-b simulator
# which servers as a ground truth verification for the model outputs
# there are many hacks, e.g. saving and loading eps and source files,
# which will be updated in future versions

from typing import List, Tuple, Optional

import numpy as np
import os, sys
import torch
from matplotlib import pyplot as plt

# a temporary local path for the modified spins library
sys.path.append("/home/maxcmk/Desktop/spins/spins_env/lib/python3.8/site-packages")

from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph.optplan.optplan import EmSource
from spins.invdes.problem_graph import workspace
from spins.fdfd_tools.waveguide_mode import solve_waveguide_mode, compute_source
from spins.gridlock import direction
import gdspy

# sys.path.append("/media/ps3/chenkaim/Waveynet3d")
from src.plot_field3D import plot_3slices
from src.physics import residue_E, eps_to_yee
from src.PML_utils import make_dxes_numpy
import time

HACK_DATA_PATH="/media/ps3/chenkaim/Waveynet3d/waveynet3d/spins/temp_files/"

@optplan.register_node(optplan.CustomSource)
class custom_source(EmSource):
    """
    wrapper function that takes in a source distribution with shape (sx, sy, sz, 6),
    and returns an object that is callable with inputs (simspace: SimulationSpace, wlen: float)
    """
    def __init__(self,
                 params: optplan.CustomSource,
                 work: Optional[workspace.Workspace] = None):
        self._params = params

    def __call__(self, simspace: optplan.SimulationSpace, wlen: float, solver, **kwargs):
        # self._params.source is a torch tensor with shape (sx, sy, sz, 6), dtype float32
        source = self._params.source
        sx, sy, sz, _ = source.shape
        source = torch.view_as_complex(source.reshape(sx, sy, sz, 3, 2))
        source = source.permute(3, 0, 1, 2).numpy()

        # save the source to the output folder
        np.save(self._params.store_dir+'inputs/cached_source', source)
        return source

def simulate(
    wavelength,
    dL,
    eps, 
    source,
    pml_layers,
    proj_folder=HACK_DATA_PATH,
    output_data_folder=HACK_DATA_PATH,
    plot=False
):
    # a temporary solution: save the eps to the output folder, which maxwell-b will read and run the simulation
    np.save(output_data_folder+'epsilon_verify',eps)

    if plot:
        plot_3slices(eps.numpy(), os.path.join(output_data_folder, "epsilon.png"), my_cmap = plt.cm.binary, cm_zero_center=False, title=None)
        plot_3slices(source[...,0].numpy(), os.path.join(output_data_folder, "source.png"), my_cmap = plt.cm.seismic, cm_zero_center=True, title=None)

    """Runs the simulation"""
    # Create the simulation space using the GDS files.
    sim_space = create_sim_space(eps.shape, pml_layers, dL=dL)

    # Setup the objectives and all values that should be recorded (monitors).
    dummy_obj, dummy_monitors = create_objective(sim_space, wavelength, source=source, store_dir = output_data_folder)
    trans_list = create_transformations(dummy_obj, dummy_monitors, sim_space)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    time1 = time.time()
    problem_graph.run_plan(plan, proj_folder, output_data_folder)
    time2 = time.time()
    print("run time for run_plan: ", time2-time1)


def create_sim_space(
    shape,
    pml_layers,
    dL=25,
):  
    thickness = 20 # dummy parameters
    mat_air = optplan.Material(index=optplan.ComplexNumber(real=1.0))
    mat_sub = optplan.Material(index=optplan.ComplexNumber(real=1.5))
    mat_mat = optplan.Material(index=optplan.ComplexNumber(real=6.0))

    LAYER_DUMMY = 101
    LAYER_SUB = 300

    dummy_layout_0 = gdspy.Rectangle((-dL, -dL),
                                    (0, 0),
                                    LAYER_DUMMY)
    dummy_layout_1 = gdspy.Rectangle((0, 0),
                                    (dL, dL),
                                    LAYER_DUMMY)

    # Generate the foreground and background GDS files.
    gds_fg = gdspy.Cell("FOREGROUND", exclude_from_current=True)
    gds_fg.add(dummy_layout_0)
    gds_fg.add(dummy_layout_1)

    gds_bg = gdspy.Cell("BACKGROUND", exclude_from_current=True)
    gds_bg.add(dummy_layout_0)

    gds_fg_name = "dummy_fg.gds"
    # gdspy.write_gds(gds_fg_name, [gds_fg], unit=1e-9, precision=1e-9)
    gds_bg_name = "dummy_bg.gds"
    # gdspy.write_gds(gds_bg_name, [gds_bg], unit=1e-9, precision=1e-9)

    stack = []

    stack.append(
        optplan.GdsMaterialStackLayer(
            foreground=mat_air, # change to mat_air if no substrate
            background=mat_sub, # change to mat_air if no substrate
            gds_layer=[LAYER_SUB, 0],
            extents=[-10000, 0],
        ))

    # Dummy layer
    stack.append(
        optplan.GdsMaterialStackLayer(
            foreground=mat_mat,
            background=mat_air,
            gds_layer=[LAYER_DUMMY, 0],
            extents=[(thickness-4)*dL, (thickness-2)*dL],
        ))

    mat_stack = optplan.GdsMaterialStack(
        background=mat_air,
        stack=stack,
    )

    sim_region = optplan.Box3d(center=[0, 0, 0], extents=[shape[0]*dL, shape[1]*dL, shape[2]*dL])

    return optplan.SimulationSpace(
        name="simspace_cont",
        mesh=optplan.UniformMesh(dx=dL),
        eps_fg=optplan.GdsEps(gds=gds_fg_name, mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds=gds_bg_name, mat_stack=mat_stack),
        sim_region=sim_region,
        selection_matrix_type="direct_lattice",
        boundary_conditions=[optplan.BlochBoundary()] * 6,
        pml_thickness=pml_layers,
    )


def create_objective(sim_space: optplan.SimulationSpace, wavelength, source=None, store_dir = None) -> Tuple[optplan.Function, List[optplan.Monitor]]:
    if source is None:
        # raise NotImplementedError("source is not implemented yet")
        source = optplan.PlaneWaveSource(
            center=[0, 0, 400],
            # extents=[0.7*args.X_dL*args.dL, 0.7*args.Y_dL*args.dL, args.dL],
            extents=[40*25, 40*25, 25],
            normal=[0, 0, 1],
            theta=0,
            psi=0,
            polarization_angle=0,
            power=1.0,
            store_dir = store_dir
        )
    else:
        source = optplan.CustomSource(source=source, store_dir=store_dir)

    epsilon = optplan.Epsilon(
        simulation_space=sim_space,
        wavelength=wavelength,
    )

    sim = optplan.FdfdSimulation(
        source=source,
        solver= "maxwell_cg",
        wavelength=wavelength,
        simulation_space=sim_space,
        epsilon=epsilon
    )

    dummy_overlap = optplan.ImportOverlap(
        file_name='dummy_overlap.mat',
        center=[0, 0, 0],
    )

    dummy_obj = optplan.Overlap(simulation=sim, overlap=dummy_overlap)
    dummy_monitor = [optplan.SimpleMonitor(name="objective", function=dummy_obj)]

    return dummy_obj, dummy_monitor

def create_transformations(
        obj: optplan.Function,
        monitors: List[optplan.Monitor],
        sim_space: optplan.SimulationSpaceBase
) -> List[optplan.Transformation]:
    """Creates a list of transformations for the device optimization.

    The transformations dictate the sequence of steps used to optimize the
    device. The optimization uses `num_stages` of continuous optimization. For
    each stage, the "discreteness" of the structure is increased (through
    controlling a parameter of a sigmoid function).

    Args:
        opt: The objective function to minimize.
        monitors: List of monitors to keep track of.
        sim_space: Simulation space ot use.


    Returns:
        A list of transformations.
    """

    # First do continuous relaxation optimization.
    # This is done through cubic interpolation and then applying a sigmoid
    # function.
    param = optplan.PixelParametrization(
        simulation_space=sim_space,
        init_method=optplan.UniformInitializer(min_val=0, max_val=0))

    trans = optplan.Transformation(
        name="opt_cont",
        parametrization=param,
        transformation=optplan.ScipyOptimizerTransformation(
            optimizer="L-BFGS-B",
            objective=obj,
            monitor_lists=optplan.ScipyOptimizerMonitorList(
                callback_monitors=monitors,
                start_monitors=monitors,
                end_monitors=monitors),
            optimization_options=optplan.ScipyOptimizerOptions(
                maxiter=1),
        ),
    )

    return [trans]

def get_results(store_dir=HACK_DATA_PATH, plot=False):
    Ex = torch.from_numpy(np.load(store_dir+'0.0001Ex.npy'))
    Ey = torch.from_numpy(np.load(store_dir+'0.0001Ey.npy'))
    Ez = torch.from_numpy(np.load(store_dir+'0.0001Ez.npy'))

    if plot:
        plot_3slices(Ex.real.numpy(), os.path.join(store_dir, "Ex.png"), my_cmap = plt.cm.seismic, cm_zero_center=True, title=None)
        
    return torch.cat((torch.view_as_real(Ex), torch.view_as_real(Ey), torch.view_as_real(Ez)), dim=-1)

def get_waveguide_source(wavelength, dL, pml_layers, eps, src_direction, src_slice, mode_num=0, power=1.0, ln_R=-16):
    # direction: e.g. [0,0,1], [0,-1,0]
    # src_slice: e.g. [(start_x, start_y, start_z), (end_x, end_y, end_z)] (inclusive)
    omega=2 * np.pi / wavelength,
    dxes = make_dxes_numpy(wavelength, dL, eps.shape, pml_layers, ln_R)
    axis = direction.Direction(direction.axisvec2axis(src_direction))

    eps_yee = eps_to_yee(torch.from_numpy(eps[None]))[0].permute(3,0,1,2).numpy()
    mu = np.ones_like(eps_yee)

    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis.value,
        'slices': tuple([slice(i, f+1) for i, f in zip(*src_slice)]),
        'polarity': direction.axisvec2polarity(src_direction),
        'mu': mu
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

if __name__ == "__main__":
    pml_layers = [10,10,10,10,10,10]
    dL = 25
    wavelength = 800

    from waveynet3d.data.simulation_dataset import SyntheticDataset
    ds = SyntheticDataset(
        shape=(64,64,64),
        pml_sizes=pml_layers,
        eps_min=1.0,
        eps_max=6.0,
        dataset_size=10, 
        zoom_eps_list=[4.0],
        zoom_src_list=[4.0],
        sigma_eps_list=[5.0],
        sigma_src_list=[5.0],
        residual_type='SC-PML',
        dL=dL,
        wl=wavelength
    )

    data = ds[0]
    eps, source = data['eps'], data['source']
    # eps = torch.ones_like(eps)
    simulate(wavelength, dL, eps[...,0], source, pml_layers)

    # plot the results:
    Ex = torch.from_numpy(np.load(os.path.join(HACK_DATA_PATH, "0.0001Ex.npy"))).to(torch.complex64)
    Ey = torch.from_numpy(np.load(os.path.join(HACK_DATA_PATH, "0.0001Ey.npy"))).to(torch.complex64)
    Ez = torch.from_numpy(np.load(os.path.join(HACK_DATA_PATH, "0.0001Ez.npy"))).to(torch.complex64)

    E = torch.cat((torch.view_as_real(Ex), torch.view_as_real(Ey), torch.view_as_real(Ez)), dim=-1)

    cached_source = np.load(os.path.join(HACK_DATA_PATH, "inputs/cached_source.npy"))
    cached_source = cached_source.transpose(1, 2, 3, 0)
    cached_source = torch.view_as_real(torch.from_numpy(cached_source)).reshape(source.shape)

    residue = residue_E(E[None], eps[None,...,0], cached_source[None], pml_layers, dL, wavelength)

    plot_3slices(eps[...,0].numpy(), os.path.join(HACK_DATA_PATH, "epsilon.png"), my_cmap = plt.cm.binary, cm_zero_center=False, title=None)
    plot_3slices(source[...,0].numpy(), os.path.join(HACK_DATA_PATH, "source.png"), my_cmap = plt.cm.seismic, cm_zero_center=True, title=None)
    plot_3slices(cached_source[...,0].numpy(), os.path.join(HACK_DATA_PATH, "cached_source.png"), my_cmap = plt.cm.seismic, cm_zero_center=True, title=None)
    plot_3slices(Ex.real.numpy(), os.path.join(HACK_DATA_PATH, "Ex.png"), my_cmap = plt.cm.seismic, cm_zero_center=True, title=None)
    plot_3slices(residue[0,...,0].numpy(), os.path.join(HACK_DATA_PATH, "residue.png"), my_cmap = plt.cm.Reds, cm_zero_center=False, title=None)
