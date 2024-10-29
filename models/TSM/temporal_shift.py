# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


def make_temporal_shift(net, n_segment, n_div=8, place='blockres'):
    
    n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0

    import torchvision
    if place == 'block':
        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
            return nn.Sequential(*(blocks))

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    elif 'blockres' in place:
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2

        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
