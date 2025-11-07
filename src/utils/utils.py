import os
import torch
import torch.distributed as dist
import gpustat
from typing import Sequence

def printc(text, color=None):
    if color == 'r':
        print("\033[91m" + text + "\033[0m", flush=True)
    elif color == 'g':
        print("\033[92m" + text + "\033[0m", flush=True)
    elif color == 'y':
        print("\033[93m" + text + "\033[0m", flush=True)
    elif color == 'b':
        print("\033[94m" + text + "\033[0m", flush=True)
    else:
        print(text, flush=True)

def MAE(a, b):
    return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def scaled_MAE(a, b):
    scaled_a = a / torch.norm(a)
    scaled_b = b / torch.norm(b)
    diff = a - b
    return torch.mean(torch.abs(scaled_a-scaled_b))/torch.mean(torch.abs(scaled_b)), diff

def c2r(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_real(x).reshape(bs, sx, sy, sz, 6)

def r2c(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_complex(x.reshape(bs, sx, sy, sz, 3, 2))

def is_array_like(x):
    if isinstance(x, (str, bytes)):
        return False
    return isinstance(x, Sequence) or hasattr(x, "__array__") or torch.is_tensor(x)

def is_multiple(a, b, tol=1e-9):
    if b == 0:
        return False  # avoid division by zero
    if is_array_like(a):
        for i in range(len(a)):
            if not is_multiple(a[i], b, tol):
                return False
        return True
    else:
        quotient = a / b
        return abs(round(quotient) - quotient) < tol

def resolve(a, b, tol=1e-9):
    assert is_multiple(a, b, tol=tol)
    return round(a / b)

class IdentityModel:
    def setup(self, x, freq):
        pass
    def __call__(self, x, freq):
        return x

def init_dist():
    if dist.is_initialized():
        return
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8778'

    dist.init_process_group(                                   
        backend='nccl',
        init_method='env://',
        world_size=1,
        rank=0
    )

def prepare_model(sim_shape, model_path, model_fn, device_id=0):
    # init_dist()
    model = model_fn(domain_sizes=sim_shape, paddings=[0,0,0])
    try:
        checkpoint = torch.load(os.path.join(model_path, "models/best_model.pt"), weights_only=False, map_location=f'cuda:{device_id}')
    except:
        checkpoint = torch.load(os.path.join(model_path, "models/last_model.pt"), weights_only=False, map_location=f'cuda:{device_id}')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda(device_id)
    return model

def get_pixels(kwargs, key, dL):
    value = kwargs[key]
    if value % dL != 0:
        printc(f"Warning: {key} is not divisible by dL, rounding to the nearest integer", 'r')
    return round(value / dL)

def get_least_used_gpu():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

def check_data_distribution(eps, pmls, wl, dL, dummy_trainer, dummy_ds):
    sim_shape = eps.shape[1:4]
    if sim_shape[0] < dummy_trainer.domain_sizes[0] or sim_shape[0] > dummy_trainer.domain_sizes[1]:
        printc(f"Warning: the simulation x size {sim_shape[0]} is not within the training data distribution {dummy_trainer.domain_sizes[0]} and {dummy_trainer.domain_sizes[1]}", 'r')
    if sim_shape[1] < dummy_trainer.domain_sizes[2] or sim_shape[1] > dummy_trainer.domain_sizes[3]:
        printc(f"Warning: the simulation y size {sim_shape[1]} is not within the training data distribution {dummy_trainer.domain_sizes[2]} and {dummy_trainer.domain_sizes[3]}", 'r')
    if sim_shape[2] < dummy_trainer.domain_sizes[4] or sim_shape[2] > dummy_trainer.domain_sizes[5]:
        printc(f"Warning: the simulation z size {sim_shape[2]} is not within the training data distribution {dummy_trainer.domain_sizes[4]} and {dummy_trainer.domain_sizes[5]}", 'r')

    if pmls[0] < dummy_trainer.pml_ranges[0] or pmls[0] > dummy_trainer.pml_ranges[1]:
        printc(f"Warning: the simulation x pml size {pmls[0]} is not within the training data distribution {dummy_trainer.pml_ranges[0]} and {dummy_trainer.pml_ranges[1]}", 'r')
    if pmls[1] < dummy_trainer.pml_ranges[0] or pmls[1] > dummy_trainer.pml_ranges[1]:
        printc(f"Warning: the simulation x pml size {pmls[1]} is not within the training data distribution {dummy_trainer.pml_ranges[0]} and {dummy_trainer.pml_ranges[1]}", 'r')
    if pmls[2] < dummy_trainer.pml_ranges[2] or pmls[2] > dummy_trainer.pml_ranges[3]:
        printc(f"Warning: the simulation y pml size {pmls[2]} is not within the training data distribution {dummy_trainer.pml_ranges[2]} and {dummy_trainer.pml_ranges[3]}", 'r')
    if pmls[3] < dummy_trainer.pml_ranges[2] or pmls[3] > dummy_trainer.pml_ranges[3]:
        printc(f"Warning: the simulation z pml size {pmls[3]} is not within the training data distribution {dummy_trainer.pml_ranges[4]} and {dummy_trainer.pml_ranges[5]}", 'r')
    if pmls[4] < dummy_trainer.pml_ranges[4] or pmls[4] > dummy_trainer.pml_ranges[5]:
        printc(f"Warning: the simulation x pml size {pmls[4]} is not within the training data distribution {dummy_trainer.pml_ranges[4]} and {dummy_trainer.pml_ranges[5]}", 'r')
    if pmls[5] < dummy_trainer.pml_ranges[4] or pmls[5] > dummy_trainer.pml_ranges[5]:
        printc(f"Warning: the simulation z pml size {pmls[5]} is not within the training data distribution {dummy_trainer.pml_ranges[4]} and {dummy_trainer.pml_ranges[5]}", 'r')

    lambda_in_pixels = wl/dL
    if lambda_in_pixels < dummy_ds.lambda_in_pixel_range[0] or lambda_in_pixels > dummy_ds.lambda_in_pixel_range[1]:
        printc(f"Warning: the simulation wavelength/dL {(wl/dL):.2)f} is not within the training data distribution {dummy_ds.lambda_in_pixel_range[0]} and {dummy_ds.lambda_in_pixel_range[1]}", 'r')
    
    if torch.min(eps) < dummy_ds.eps_min or torch.max(eps) > dummy_ds.eps_max:
        printc(f"Warning: the simulation eps min {torch.min(eps):.2f} and max {torch.max(eps):.2f} are not within the training data distribution {dummy_ds.eps_min} and {dummy_ds.eps_max}", 'r')
