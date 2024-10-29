"""
The TSM model for S2DNet, which is modified using TSM/ops/models.py. 
Please prepare the necessary auxiliary codes for TSM (e.g., transform.py) according to  the original repo.
"""

import torch
from torch import nn
from torch.nn.init import normal_, constant_
from torch.nn import functional as F
import torchvision
import torchvision.models as models

from einops import rearrange

from .temporal_shift import make_temporal_shift

class TSM(nn.Module):

    def __init__(self,
                 num_segments=4,
                 base_model='resnet50',
                 dropout=0.8,
                 partial_bn=True,
                 shift_div=8,
                 shift_place='blockres',
                 args=None
                 ):

        super(TSM, self).__init__()
        self.args = args
        self.num_segments = num_segments
        self.dropout = dropout

        self.shift_div = shift_div
        self.shift_place = shift_place

        if args.model.tsm.base_model == 'resnet18':
            self.base_model = models.resnet18()
            ckpt = torch.load('/youtu-pangu-public/hanyangwang/pretrained_models/MS-Celeb-1M/ijba_res18_naive.pth.tar')
            ckpt = ckpt['state_dict']
            ckpt = {k.replace('module.', ''):v for k,v in ckpt.items()}
            self.base_model.fc = nn.Sequential()
            del ckpt['feature.weight'], ckpt['feature.bias'], ckpt['fc.weight'], ckpt['fc.bias']
            self.base_model.load_state_dict(ckpt)
            self.base_model.add_module('fc', nn.Linear(512, 256))

        elif args.model.tsm.base_model == 'resnet50':
            self.base_model = getattr(torchvision.models, base_model)(True)

        make_temporal_shift(self.base_model,
                            self.num_segments,
                            n_div=self.shift_div,
                            place=self.shift_place)

        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)  # replace nn.AdaptiveAvgPool2d(1), to retain the spatial dim.

        feature_dim = getattr(self.base_model,'fc').in_features
        setattr(self.base_model, 'fc',nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, 512)                 
        self._enable_pbn = partial_bn
        if hasattr(self.new_fc, 'weight'):
            normal_(self.new_fc.weight, 0, 0.001)
            constant_(self.new_fc.bias, 0)    

    def forward(self, input, is_S=False):
        
        # [128, 4, 3, 512, 512]
        input = rearrange(input, 'b t c h w-> (b t) c h w')
        base_out = self.base_model(input)
        # [512, 2048]

        base_out = self.new_fc(base_out)
        # [512, 512]
        
        if is_S:
            return base_out

        base_out = rearrange(base_out, '(b t) c-> b t c', t=self.num_segments)
        
        # [128, 4, 512]
        base_out = torch.mean(base_out, dim=1, keepdim=False)

        return base_out