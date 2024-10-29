from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import torch

def create_scheduler(args, optimizer, n_iter_per_epoch):
    num_steps = int(args.train.epochs * n_iter_per_epoch)
    lr_scheduler = None
    if args.scheduler.name == 'CosineLRScheduler':
        warmup_steps = int(args.scheduler.CosineLRScheduler.warmup_epochs * n_iter_per_epoch)
        if args.optimizer.name == 'AdamW':
            min_lr = args.optimizer.AdamW.min_lr
        elif args.optimizer.name == 'SGD':
            min_lr = args.optimizer.SGD.min_lr
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=min_lr,
            warmup_lr_init=args.scheduler.CosineLRScheduler.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif args.scheduler.name == 'StepLRScheduler':
        warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
        decay_steps = int(args.decay_epochs * n_iter_per_epoch)
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif args.scheduler.name == 'lambda_lr':
        warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
        def lambda_lr(current_step: int):
            flat_steps = warmup_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step < (warmup_steps + flat_steps):
                return 1.0
            return max(
                0.0, float(num_steps - current_step) / float(max(1, num_steps - (warmup_steps + flat_steps)))
            )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)


    return lr_scheduler