import os
import torch
import torch.distributed as dist
import gpustat

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

def prepare_model(sim_shape, model_path, model_fn):
    # init_dist()
    model = model_fn(domain_sizes=sim_shape, paddings=[0,0,0])
    try:
        checkpoint = torch.load(os.path.join(model_path, "models/best_model.pt"), weights_only=False, map_location='cuda:0')
    except:
        checkpoint = torch.load(os.path.join(model_path, "models/last_model.pt"), weights_only=False, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    return model

def get_pixels(kwargs, key, dL):
    value = kwargs[key]
    if value % dL != 0:
        print(f"Warning: {key} is not divisible by dL, rounding to the nearest integer")
    return round(value / dL)


def get_least_used_gpu():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU
