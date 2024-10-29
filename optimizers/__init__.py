import torch

def create_optimizer(params, args):
    
    if args.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.optimizer.SGD.lr, momentum=args.optimizer.SGD.momentum, weight_decay=args.optimizer.SGD.weight_decay)
    elif args.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.optimizer.AdamW.lr, eps=args.optimizer.AdamW.eps, weight_decay=args.optimizer.AdamW.weight_decay)
    
    return optimizer