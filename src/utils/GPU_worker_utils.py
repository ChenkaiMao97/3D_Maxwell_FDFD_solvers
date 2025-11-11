import torch
from src.solvers.NN_solver import NN_solver

def solver_worker(device_id, init_kwargs, task_queue, result_queue):
    """
    starts a worker process that listens for tasks from a queue and returns results to a result queue
    """
    torch.cuda.set_device(device_id)

    # initialize the solver
    solver = NN_solver(**init_kwargs, gpu_id=device_id)  # your class with model on GPU
    solver.init()

    # init_queue.put(solver.subdomain_solver.source_mult)

    while True:
        task = task_queue.get()
        if task is None:
            break  # shutdown signal

        task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, pml_thickness, gt, init_x) = task

        # move tensors to GPU
        epsilon_r_torch = epsilon_r_torch.to(f'cuda:{device_id}')
        source_torch = source_torch.to(f'cuda:{device_id}')
        if init_x is not None:
            init_x = init_x.to(f'cuda:{device_id}')
        output = solver.solve(epsilon_r_torch, source_torch, gt=gt, init_x=init_x)

        result_queue.put((task_id, output.cpu()))