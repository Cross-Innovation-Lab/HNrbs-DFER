from .dataset_DFEW import DFEWDataset
from .dataset_FERV39K import FERV39KDataset
import torch
import torch.utils.data.distributed
import torch.utils.data

def create_dataloader(args, mode):

    if args.dataset.name == 'DFEW':
        dataset = DFEWDataset(args, mode)
    elif args.dataset.name == 'FERV39K':
        dataset = FERV39KDataset(args, mode)

    dataloader = None

    if args.use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    if mode == "train":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.train.batch_size,
                                                num_workers=args.dataset.workers,
                                                pin_memory=True,
                                                sampler=sampler,
                                                drop_last=True)
    elif mode == "test":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.train.batch_size,
                                                shuffle=False,
                                                num_workers=args.dataset.workers,
                                                pin_memory=True, 
                                                drop_last=True)
    return dataloader
