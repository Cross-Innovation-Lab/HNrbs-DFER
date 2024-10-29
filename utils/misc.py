import os
import random
from re import I
import wandb
import shutil
import time
import datetime
import warnings
import numpy as np
import torch
import torch.distributed as dist


def set_seed(SEED):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


def setup(args):
    warnings.filterwarnings("ignore")
    seed = args.seed
    set_seed(seed)

    
def init_wandb_workspace(args):
    if args.wandb.name is None:
        args.wandb.name = args.config.split('/')[-1].replace('.yaml', '')
    

    if int(os.environ["LOCAL_RANK"]) == 0:
        print(args.wandb)
        wandb.init(**args.wandb)
        allow_val_change = False if args.wandb.resume is None else True
        wandb.config.update(args, allow_val_change)
        wandb.save(args.config)
        
        if args.debug or wandb.run.dir == '/tmp':
            args.exam_dir = 'wandb/debug'
        else:
            args.exam_dir = os.path.dirname(wandb.run.dir)

        if os.path.exists(args.exam_dir):
            shutil.rmtree(args.exam_dir)
        os.makedirs(args.exam_dir, exist_ok=True)
        os.makedirs(os.path.join(args.exam_dir, 'ckpts'), exist_ok=True)

    return args


def save_test_results(img_paths, y_preds, y_trues, filename='results.log'):
    assert len(y_trues) == len(y_preds) == len(img_paths)

    with open(filename, 'w') as f:
        for i in range(len(img_paths)):
            print(img_paths[i], end=' ', file = f)
            print(y_preds[i], file = f)
            print(y_trues[i], end=' ', file = f)
 
@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


def update_meter(meter, value, size, is_dist=False):
    if is_dist:
        meter.update(reduce_tensor(value.data).item(), size)
    else:
        meter.update(value.item(), size)
    return meter
