import torch
from torch import nn
import torch.nn.functional as F
import random
import math

import glob
import os
import PIL.Image as Image
import torchvision
from einops import rearrange

class UniformSampler(nn.Module):

    def __init__(self, args):
        super(UniformSampler, self).__init__()
        self.args = args
        self.bag_size_s = int(self.args.model.S_frames // self.args.model.clip_length)
        self.bag_size_d = int(self.args.model.D_frames // self.args.model.clip_length)

    def spatial_sampling(self, x, spatial_size=None):
        """For uniform sampler, downsample the input frames to the required scale (used after temporal sampling).
        Parameters
        ----------
        x : torch.Tensor
            the input frames (after temporal sampling)
        """

        if spatial_size is None:
            spatial_size = self.args.transform.image_size
        
        B, D1, D2, _, _ = x.size()
        x = x.view((B, -1) + x.size()[-2:])
        x = F.interpolate(x, size=spatial_size)
        x = x.view((B, D1, D2) + x.size()[-2:])

        return x

    def temporal_sampling(self, x, instance_id=None):
        """For uniform sampler, sample uniformly across all frames.
        Parameters
        ----------
        x : torch.Tensor
            raw input frames
        meta : dict
            the meta information of the video, 
                - in_frame_num: num of valid frames
        dim : int, optional
            the dim to sample (starting from the first non-Batch dim), by default 0
        """
        if instance_id is None:
            fnum = self.args.model.S_frames
            starts = torch.zeros(x.shape[0])
            ends = torch.ones(x.shape[0]) * x.size()[1] - 1
        else:
            fnum = self.args.model.D_frames
            starts = torch.ones(x.shape[0], device=x.device) * x.size()[1] * instance_id / self.bag_size_d
            ends = torch.ones(x.shape[0], device=x.device) * x.size()[1] * (instance_id + 1) / self.bag_size_d - 1

        starts = starts.tolist()
        ends = ends.tolist()
            
        tmp_x = []
        for _x, start, end in zip(x, starts, ends):

            start = int(start)
            end = int(end)
            fid = torch.linspace(start, end, steps=fnum, device=_x.device).long()
            if int((end-start) / fnum) != 0:
                y = torch.randint_like(fid, 0, int((end-start) / fnum))
                y[fnum-1] = 0
                fid += y
            tmp_x += [torch.index_select(_x, 0, fid)]
        x = torch.stack(tmp_x)
        return x

    def forward(self, x, instance_id=None):

        x = self.temporal_sampling(x, instance_id)
        # x = self.spatial_sampling(x, spatial_size=self.args.transform.image_size)
            
        return x

class BinarySampler(nn.Module):

    def __init__(self, args):
        super(BinarySampler, self).__init__()
        self.args = args


    def forward(self, x, number=1):

        # [B, T, C, H, W]
        num_frames = x.shape[1]
        step = num_frames // (number + 1) # 16, 10

        ids = torch.arange(step, step * number + 1, step).to(x.device) # [16], [10, 20]

        x = torch.index_select(x, 1, ids)
            
        return x

class AdaptiveSampler(nn.Module):

    def __init__(self, args=None):
        super(AdaptiveSampler, self).__init__()
        self.args = args

        self.fixed_temporal_sampling = False
        self.t_sigma = math.log(1.)

    def forward(self, x, params):

        # Get (B,T,C,H,W) input
        args = self.args
        device = x[0].device
        dtype = x[0].dtype
        tmp_x = []

        if args is None:
            atten_out_t = 16
        else:
            atten_out_t = 4

        """ parse temporal parameters """
        batch_dt, batch_delta_t = params

        batch_sigma_t = batch_dt.new_zeros(batch_dt.size()) + self.t_sigma

        # if not args.EVALUATE:
        if self.fixed_temporal_sampling:
            batch_dt = [p.new_zeros(p.size()) for p in batch_dt]
            batch_delta_t = [p.new_zeros(p.size()) for p in batch_delta_t]  # before exp, should be zeros.

        """ temporal sampling """
        # (B,T,C,H,W) -> (B,C,H,W,T)
        x = [_x.permute(1, 2, 3, 0).contiguous() for _x in x]
        for i, _x in enumerate(x):
            dt = batch_dt[i]
            
            delta_t = batch_delta_t[i]
            sigma_t = batch_sigma_t[i]

            in_t = _x.shape[-1]

            _x = _x.unsqueeze(0)  # (C,H,W,T) -> (B=1,C,H,W,T)
            _x = _x[:, :, :, :, :in_t]  # take the valid frames as input
            _x = rearrange(_x, 'b c h w (t r) -> b (c h w) t r', r = 1)
            _x = _x.repeat(1,1,1,4)

            """ get temporal steps """
            anchor_t = (in_t - 1) / 2.0  # int(round(in_t / 2.0))

            # time center
            dt = dt * (in_t - 1) / 2.0 + anchor_t

            # time stride
            if self.args.model.ki.better_delta:
                delta_t = (in_t / (atten_out_t - 1) - 1) * delta_t + 1
            else:
                delta_t = (in_t / (atten_out_t - 1)) * torch.exp(delta_t)
            # time sigma
            sigma_t = torch.exp(sigma_t)

            tmp = []
            for j in range(dt.shape[-1]):
                
                if self.args.model.ki.delta_1:
                    delta_t[j] = 1
                grid_t_i = torch.arange(0, atten_out_t).type(dtype).to(device)
                mu_t = dt[j] + (grid_t_i - (atten_out_t - 1) / 2.0) * delta_t[j] # (1,t)

                mu_t = mu_t.view(1, -1) # (1,t)

                """ temporal sampling """
                t = torch.arange(0, in_t).view(in_t, 1).type(dtype).to(device)  # (T,1)

                # eps_tensor_t = 1e-7 * torch.ones(atten_out_t).to(device).view(1, -1)  # (1,t)
                # Ft = torch.exp(-1 * torch.pow(t - mu_t, 2) / (2 * sigma_t[j])).float()  # (T,t)
                # Ft = Ft / torch.max(torch.sum(Ft, 0, keepdim=True), eps_tensor_t)  # (T,t)
                
                Ft = t - mu_t.ceil()
                Ft = torch.where(Ft == 0, 1, 0)
                Ft = Ft.type(dtype)
                
                tmp.append(torch.matmul(_x[..., j], Ft))
            _x = torch.cat(tmp, dim=0)
            _x = rearrange(_x, 'r (c h w) t -> c h w (r t)', c=3, h=112, w=112)
            tmp_x.append(_x)
        x = torch.cat(tmp_x, dim=0)
        x = rearrange(x, '(b c) h w t -> b t c h w', c=3)

        return x
