#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import wandb
import time
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

from timm.utils import CheckpointSaver
from timm.models import resume_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

from einops import rearrange

from datasets import create_dataloader
from models import create_model
from optimizers import create_optimizer
from schedulers import create_scheduler
from losses import create_loss
from utils import *

import better_exceptions
better_exceptions.hook()

emotions = ["ha", "sa", "ne", "an", "su", "di", "fe"]
args = get_parameters()
if args.use_ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    local_rank = 0
args = init_wandb_workspace(args)
if local_rank == 0:
    logger.add(f'{args.exam_dir}/train.log', level="INFO")
    logger.info(OmegaConf.to_yaml(args))

def main():

    if args.use_ddp:
        dist.init_process_group(backend='nccl', init_method="env://", rank=local_rank)
        torch.cuda.set_device(local_rank)
    
    # Init setup
    setup(args)

    # Create dataloader
    train_dataloader = create_dataloader(args, 'train')
    test_dataloader = create_dataloader(args, 'test')

    if args.use_ddp:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", args.gpu_ids[0])
    model = create_model(args)
    model.to(device)

    # Create model

    optimizer = create_optimizer(model.parameters(), args)
    scheduler = create_scheduler(args, optimizer, len(train_dataloader))

    # Resume from checkpoint
    checkpoint_dir = os.path.join(args.exam_dir, 'ckpts') if args.exam_dir else None

    start_epoch = 1
    if args.model.resume is not None:
        start_epoch = resume_checkpoint(model, args.model.resume, optimizer)
        logger.info(f'resume model from {args.model.resume}')

    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = torch.nn.parallel.DataParallel(model, args.gpu_ids)

    # Traing misc
    saver = None
    if local_rank == 0:
        wandb.watch(model, log='all')
        saver = CheckpointSaver(model,
                                optimizer,
                                args=args,
                                checkpoint_dir=checkpoint_dir,
                                max_history=1)

    # Training loop
    for epoch in range(start_epoch, args.train.epochs + 1):
        
        if local_rank == 0:
            start = time.time()
        train(train_dataloader, model, optimizer, scheduler, device, epoch)
        val(test_dataloader, model, scheduler, saver, device, epoch)
        if local_rank == 0:
            epoch_time = time.time() - start
            logger.info(f'time: {epoch_time:.1f}')

    if local_rank == 0:
        wandb.finish()


def train(train_dataloader, model, optimizer, scheduler, device, epoch):

    if args.use_ddp:
        train_dataloader.sampler.set_epoch(epoch)
    losses = AverageMeter('Loss', ':.4f')
    losses_ce = AverageMeter('Loss_ce', ':.4f')
    losses_snet_ce = AverageMeter('Loss_snet_ce', ':.4f')
    losses_synp_instance_ce = AverageMeter('Loss_synp_instance_ce', ':.4f')
    losses_deta_instance_ce = AverageMeter('Loss_deta_instance_ce', ':.4f')
    losses_key_instance = AverageMeter('Loss_key_instance', ':.4f')

    model.train()

    ce = nn.CrossEntropyLoss()
    KILoss = create_loss(args, 'KILoss')
    KICELoss = create_loss(args, 'KICELoss')

    for i, batch in enumerate(train_dataloader):
        # measure data loading time

        if local_rank == 0:
            print("Training epoch \t{}: {}\\{}".format(epoch, i + 1, len(train_dataloader)), end='\r')

        images = batch['images'].to(device)
        targets = batch['targets'].to(device)

        output, info = model(images, targets)

        ce_loss = ce(output, targets)
        snet_ce_loss = ce(info['snet_pred'], targets)
        kiloss = KILoss(info['pred_instance_id'][0], info['manual_instance_id'])

        synp_instance_preds = info['synp_instance_emo']
        kiceloss = KICELoss(synp_instance_preds, targets)
        synp_instance_targets = targets.view(-1, 1)
        synp_instance_targets = synp_instance_targets.repeat(1, synp_instance_preds.shape[1])
        synp_instance_preds = rearrange(synp_instance_preds, 'b l c -> (b l) c')
        synp_instance_targets = rearrange(synp_instance_targets, 'b l -> (b l)')
        synp_instance_ce_loss = ce(synp_instance_preds, synp_instance_targets) / synp_instance_preds.shape[1]

        deta_instance_preds = info['deta_instance_emo']
        deta_instance_targets = targets.view(-1, 1)
        deta_instance_targets = deta_instance_targets.repeat(1, deta_instance_preds.shape[1])
        deta_instance_preds = rearrange(deta_instance_preds, 'b l c -> (b l) c')
        deta_instance_targets = rearrange(deta_instance_targets, 'b l -> (b l)')
        deta_instance_ce_loss = ce(deta_instance_preds, deta_instance_targets) / deta_instance_preds.shape[1]

        if args.model.two_stage:
            if epoch < 50:
                ce_loss = torch.tensor(0)
                deta_instance_ce_loss = torch.tensor(0)

        if args.model.two_stage_pro:
            if epoch >= 50:
                snet_ce_loss = torch.tensor(0)
                synp_instance_ce_loss = torch.tensor(0)
                kiloss = torch.tensor(0)
        
        if args.loss.ki_ce_loss > 0:
            loss =  ce_loss \
                + args.loss.snet_ce_loss * snet_ce_loss \
                + args.loss.ki_ce_loss * kiceloss \
                + args.loss.deta_instance_ce_loss * deta_instance_ce_loss \
                + args.loss.key_instance_loss * kiloss
        else:
            loss =  ce_loss \
                + args.loss.snet_ce_loss * snet_ce_loss \
                + args.loss.synp_instance_ce_loss * synp_instance_ce_loss \
                + args.loss.deta_instance_ce_loss * deta_instance_ce_loss \
                + args.loss.key_instance_loss * kiloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.scheduler.CosineLRScheduler.two_stage_pm and epoch > 50:
            scheduler.step_update((epoch - 50) * len(train_dataloader) + i)
        else:
            scheduler.step_update(epoch * len(train_dataloader) + i)
        
        losses = update_meter(losses, loss, args.train.batch_size, args.distributed)
        losses_ce = update_meter(losses_ce, ce_loss, args.train.batch_size, args.distributed)
        losses_snet_ce = update_meter(losses_snet_ce, snet_ce_loss, args.train.batch_size, args.distributed)
        losses_synp_instance_ce = update_meter(losses_synp_instance_ce, synp_instance_ce_loss, args.train.batch_size, args.distributed)
        losses_deta_instance_ce = update_meter(losses_deta_instance_ce, deta_instance_ce_loss, args.train.batch_size, args.distributed)
        losses_key_instance = update_meter(losses_key_instance, kiloss, args.train.batch_size, args.distributed)
 
    if args.loss.relabel and epoch > 50:
        sm = torch.softmax(output, dim = 1)
        Pmax, predicted_labels = torch.max(sm, 1) # predictions
        Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
        true_or_false = Pmax - Pgt > 0.5
        update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
        label_idx = batch['idx'][update_idx] # get samples' index in train_loader
        print(label_idx)
        relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
        print(relabels)
        if label_idx.shape != torch.Size([]):
            for i in range(label_idx.shape[0]):
                train_dataloader.dataset.data[label_idx[i]]['emotion'] = relabels.cpu().numpy()[i] # relabel samples in train_loader

    if local_rank == 0:
        print('\n')
        results = {
            'train_loss': losses.avg,
            'train_loss_ce': losses_ce.avg,
            'train_loss_snet_ce': losses_snet_ce.avg,
            'train_loss_synp_instance_ce': losses_synp_instance_ce.avg,
            'train_loss_deta_instance_ce': losses_deta_instance_ce.avg,
            'train_loss_key_instance': losses_key_instance.avg
        }
        wandb.log(results, step=epoch)


def val(val_dataloader, model, scheduler, saver, device, epoch):

    y_preds, y_snet_preds, y_labels, y_conf = [], [], [], []
    synp_preds, synp_labels, deta_preds, deta_labels = [], [], [], []
    synp_mil_preds, deta_mil_preds = [], []
    model.eval()

    detail = None
    synopsis = None

    for i, batch in enumerate(val_dataloader):

        if local_rank == 0:
            print("Testing epoch \t{}: {}\\{}".format(epoch, i + 1, len(val_dataloader)), end='\r')

        with torch.no_grad():
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)

            output, info = model(images, None)

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            
            snet_prob = F.softmax(info['snet_pred'], dim=1)
            _, snet_pred = snet_prob.max(1)

            if local_rank == 0 and detail is None:
                # vis_num = 6
                vis_num = random.randint(0, len(info['detail']) - 1)
                if args.vis_s2d:
                    detail = info['detail'][vis_num].cpu().detach()
                    synopsis = info['synopsis'][vis_num].cpu().detach()
                    instance_emo = info['synp_instance_emo'][vis_num].cpu().detach()
                    sign = torch.max(instance_emo, dim=1).values
                    emotion_pred = prob[vis_num].cpu().detach()
                    label = targets[vis_num].cpu().detach()
                    image_length = images[vis_num].shape[0]
                    pred_id = info['pred_instance_id'][0][vis_num]

            deta_instance_preds = info['deta_instance_emo']
            deta_instance_targets = targets.view(-1, 1)
            deta_instance_targets = deta_instance_targets.repeat(1, deta_instance_preds.shape[1])
            deta_instance_preds = rearrange(deta_instance_preds, 'b l c -> (b l) c')
            deta_instance_targets = rearrange(deta_instance_targets, 'b l -> (b l)')
            deta_prob = F.softmax(deta_instance_preds, dim=1)
            _, deta_pred = deta_prob.max(1)

            synp_instance_preds = info['synp_instance_emo']
            synp_instance_targets = targets.view(-1, 1)
            synp_instance_targets = synp_instance_targets.repeat(1, synp_instance_preds.shape[1])
            synp_instance_preds = rearrange(synp_instance_preds, 'b l c -> (b l) c')
            synp_instance_targets = rearrange(synp_instance_targets, 'b l -> (b l)')
            synp_prob = F.softmax(synp_instance_preds, dim=1)
            _, synp_pred = synp_prob.max(1)

            synp_mil_pred = info['synp_instance_emo']
            _, synp_mil_pred = synp_mil_pred.max(2)
            synp_mil_label = targets.view(-1, 1)
            synp_mil_label = synp_mil_label.repeat(1, synp_mil_pred.shape[1])
            synp_mil_pred = torch.where(synp_mil_pred == synp_mil_label, 1, 0)
            synp_mil_pred, _ = synp_mil_pred.max(1)
            synp_mil_preds.extend(synp_mil_pred.cpu().detach().numpy())

            deta_mil_pred = info['deta_instance_emo']
            _, deta_mil_pred = deta_mil_pred.max(2)
            deta_mil_label = targets.view(-1, 1)
            deta_mil_label = deta_mil_label.repeat(1, deta_mil_pred.shape[1])
            deta_mil_pred = torch.where(deta_mil_pred == deta_mil_label, 1, 0)
            deta_mil_pred, _ = deta_mil_pred.max(1)
            deta_mil_preds.extend(deta_mil_pred.cpu().detach().numpy())

            y_preds.extend(pred.cpu().detach().numpy())
            y_snet_preds.extend(snet_pred.cpu().detach().numpy())
            y_labels.extend(targets.cpu().numpy())
            y_conf.extend(conf.cpu().detach().numpy())

            synp_preds.extend(synp_pred.cpu().detach().numpy())
            synp_labels.extend(synp_instance_targets.cpu().detach().numpy())
            deta_preds.extend(deta_pred.cpu().detach().numpy())
            deta_labels.extend(deta_instance_targets.cpu().detach().numpy())

    
    if local_rank == 0:
        print('\n')
        logger.info(f'##### Eval Results #####')

        wa = 100 * accuracy_score(y_labels, y_preds)
        ua = 100 * balanced_accuracy_score(y_labels, y_preds)
        snet_wa = 100 * accuracy_score(y_labels, y_snet_preds)
        synp_instance_wa = 100 * accuracy_score(synp_labels, synp_preds)
        deta_instance_wa = 100 * accuracy_score(deta_labels, deta_preds)
        synp_mil_wa = 100 * torch.sum(synp_mil_pred) / synp_mil_pred.shape[0]
        deta_mil_wa = 100 * torch.sum(deta_mil_pred) / deta_mil_pred.shape[0]
        
        plt.rcParams['figure.figsize'] = (6, 5)
        c_m = confusion_matrix(y_labels, y_preds)
        cm = []
        for row in c_m:
            row = row / np.sum(row)
            cm.append(row)
        ax = seaborn.heatmap(cm, xticklabels=emotions, yticklabels=emotions, cmap='rocket_r', linewidth=.5, annot=True, fmt='.2f')
        confusion_matrix_figure = ax.get_figure()
        plt.close()

        if args.vis_s2d:
            
            nrow = int(args.model.S_frames ** 0.5)
            plt.rcParams['figure.figsize'] = (21, 5)
            plt.tight_layout()
            
            detail = torchvision.utils.make_grid(detail, nrow=nrow, padding=1)
            detail = detail.permute(1, 2, 0)
            plt.imshow(detail)
            plt.xlabel('Detail', fontsize=16)
            axes.set_xticks([])
            axes.set_yticks([])

            axes=plt.subplot(1, 4, 2)
            synopsis = torchvision.utils.make_grid(synopsis, nrow=nrow, padding=1)
            synopsis = synopsis.permute(1, 2, 0)
            plt.imshow(synopsis)

            pred_id = (pred_id + 1) * (16 - 1) / 2.0
            title = 'Synopsis'
            for x in pred_id:
                title += str('{:.1f}'.format(x.item())) + ', '

            plt.xlabel(title, fontsize=16)
            axes.set_xticks([])
            axes.set_yticks([])

            axes=plt.subplot(1, 4, 3)
            if args.model.name == 'S2DFE_real':
                x = np.arange(0, args.model.S_bagsize)
            emos = torch.max(instance_emo, dim=1).indices
            x = []
            for i in range(emos.shape[0]):
                x.append(emotions[emos[i]] + str(i))
            data = {'x': x, 'sign': sign}
            data = pd.DataFrame(data)
            seaborn.barplot(data=data, x='x', y='sign')
            plt.xticks(rotation=45)

            axes=plt.subplot(1, 4, 4)
            x = emotions
            data = {emotions[label]: x, 'emotion': emotion_pred}
            data = pd.DataFrame(data)
            seaborn.barplot(data=data, x=emotions[label], y='emotion')

            s2d_figure = axes.get_figure()
            plt.close()
            # s2d_figure = plt.figure()

            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=wa)
            wandb.run.summary['best_acc'] = best_metric
            last_lr = [group['lr'] for group in scheduler.optimizer.param_groups][0]

            results = {
                'lr': last_lr,
                'val_wa': wa,
                'val_ua': ua,
                'snet_wa': snet_wa,
                'synp_instance_wa': synp_instance_wa,
                'deta_instance_wa': deta_instance_wa,
                'synp_mil_wa': synp_mil_wa,
                'deta_mil_wa': deta_mil_wa,
                'val_best_wa':best_metric,
                'vis_s2d': wandb.Image(s2d_figure),
                'vis_confus': wandb.Image(confusion_matrix_figure)
            }
        else:
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=wa)
            wandb.run.summary['best_acc'] = best_metric
            last_lr = [group['lr'] for group in scheduler.optimizer.param_groups][0]

            results = {
                'lr': last_lr,
                'val_wa': wa,
                'val_ua': ua,
                'snet_wa': snet_wa,
                'synp_instance_wa': synp_instance_wa,
                'deta_instance_wa': deta_instance_wa,
                'synp_mil_wa': synp_mil_wa,
                'deta_mil_wa': deta_mil_wa,
                'val_best_wa':best_metric
            }

        logger.info(f'lr: {last_lr}')
        logger.info(f'best_val_acc: {best_metric:.4f} (Epoch-{best_epoch})')
        logger.info(f'[WA] Avg: {wa:.4f}')
        logger.info(f'[UA] Avg: {ua:.4f}')
        logger.info(f'[SNET_wa] Avg: {snet_wa:.4f}')
        logger.info(f'[SYNP_MIL_WA] Avg: {synp_mil_wa:.4f}')
        logger.info(f'[DETA_MIL_WA] Avg: {deta_mil_wa:.4f}')
        logger.info(f'{wa:.4f},{ua:.4f}')

        wandb.log(results, step=epoch)

if __name__ == '__main__':
    main()
