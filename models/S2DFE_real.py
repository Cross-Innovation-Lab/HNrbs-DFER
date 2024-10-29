from curses.ascii import EM
from re import X
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights, r2plus1d_18, R2Plus1D_18_Weights

from einops import rearrange

from .TSM.TSM import *
from .Sampler import *

class S2DFE_real(nn.Module):

    def __init__(self, args):
        super(S2DFE_real, self).__init__()

        self.args = args
        self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')

        self.sampler_s = UniformSampler(args)
        self.sampler_d = AdaptiveSampler(args)

        self.bag_size = self.args.model.D_frames // self.args.model.clip_length

        self.snet = SNet(args)
        self.dnet = DNet(args)

        if self.args.model.ezdelta:
            self.delta = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # [B, T, C, H, W]
        input = self.sampler_s(x)
        # [B, T / 4, C, H, W]
        synopsis = input

        snet_pred, info = self.snet(input, y)
        # pred_instance_id = info['pred_instance_id']
        if self.args.model.ki.manual:
            pred_instance_id = [info['manual_instance_id'], info['pred_instance_id'][1]]
        elif self.args.model.ki.manual_real and y is not None:
            pred_instance_id = [info['manual_instance_id'], info['pred_instance_id'][1]]
        else:
            pred_instance_id = info['pred_instance_id']
        synp_feature = info['instance_feature']
        instance_emo = info['instance_emo']
        attn = info['attn']
        manual_instance_id = info['manual_instance_id']

        if self.args.model.s_only:
            info['synopsis'] = synopsis
            info['detail'] = synopsis
            info['snet_pred'] = snet_pred
            info['deta_instance_emo'] = instance_emo
            info['synp_instance_emo'] = instance_emo
            return snet_pred, info

        if self.args.model.same_input:
            input = synopsis
        else:
            input = self.sampler_d(x, pred_instance_id)
        # [B, T / 4, C, H, W]
        detail = input

        out, info = self.dnet(input, attn)

        info['synopsis'] = synopsis
        info['detail'] = detail
        info['snet_pred'] = snet_pred
        info['deta_instance_emo'] = info['instance_emo']
        info['synp_instance_emo'] = instance_emo
        info['manual_instance_id'] = manual_instance_id
        info['pred_instance_id'] = pred_instance_id

        return out, info
            

class SNet(nn.Module):

    def __init__(self, args):

        super(SNet, self).__init__()
        self.args = args
        self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = args.model.S_bagsize

        # backbone
        self.model = TSM(args=self.args, base_model='resnet50')
        feature_dim = 512

        if self.args.model.params.LSTM:
            self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, batch_first=True, bidirectional=True)
            feature_dim *= 2

        if self.args.model.params.MHSA:
            # attn
            self.heads = 8
            self.dim_head = feature_dim // self.heads
            self.scale = self.dim_head ** -0.5
            self.attend = nn.Softmax(dim = -1)
            self.to_qkv = nn.Linear(feature_dim, (self.dim_head * self.heads) * 3, bias = False)

        if self.args.model.params.Norm:
            self.layerNorm = nn.LayerNorm(feature_dim)

        if self.args.model.params.convPool:
            self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        if self.args.model.s2dattn:
            self.toAttn = nn.Linear(feature_dim, 1024)
            self.attnSigmoid = nn.Sigmoid()

        if self.args.model.milpool:
            self.mil_pool = MAB(feature_dim, feature_dim)

        # classifier
        self.fc = nn.Linear(feature_dim, self.args.model.num_classes) # new fc layer 512x7
        self.s2d_fc1 = nn.Linear(feature_dim, 4)
        self.s2d_fc2 = nn.Linear(feature_dim, 4)

    def MIL(self, x, y):
        
        # [B, bag_size, 512]
        instance_feature = x

        if self.args.model.params.LSTM:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)

        if self.args.model.params.MHSA:
            ori_x = x
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # [batch, head, n, d]

            attn = self.attend(dots)
            # [B, bag_size, bag_size]
            x = torch.matmul(attn, v)
            x = rearrange(x, 'b h n d -> b n (h d)')
            # x = self.softmax(x)
            if self.args.model.params.Norm:
                x = self.layerNorm(x)
            x = torch.sigmoid(x)
            x = ori_x * x

        # Calculate manual Key Instance
        instance_emo = self.fc(x)
        # [B, bag_size, 7]
        instance_emo = F.softmax(instance_emo, dim=2)

        # #留下一个最好的
        if self.args.model.milpool:
            emo, emoind = instance_emo.max(1)
            topemo = torch.topk(emo, 1, 1).indices
            highest_instance = emo.gather(1, topemo)

        
        if self.args.model.ki.manual_top_4:
            emo, emoind = instance_emo.max(1)
            topemo = torch.topk(emo, 4, 1).indices
            manual_instance_id = emoind.gather(1, topemo)
        elif self.args.model.ki.manual_sign:
            emo_sign = torch.max(instance_emo, dim=2).values
            # [B, bag_size]
            manual_instance_id = torch.topk(emo_sign, 4, dim=1).values
            # [B, 4]

        if self.args.model.ki.manual_real:
            if y is not None:
                y = y.view(-1, 1)
                y = y.repeat(1, instance_emo.shape[1])
                manual_instance_id = rearrange(instance_emo, 'b l c -> (b l) c')
                y = rearrange(y, 'b l -> (b l)')
                manual_instance_id = manual_instance_id.gather(1, y.unsqueeze(1))
                manual_instance_id = rearrange(manual_instance_id, '(b l) c -> b l c', b=x.shape[0])
                manual_instance_id = torch.topk(manual_instance_id, 4, 1).indices.squeeze()
        
        manual_instance_id = (manual_instance_id / 15 - 0.5) * 2
        if self.args.model.ki.sort:
            manual_instance_id, _ = torch.sort(manual_instance_id)

        # Merge feature
        if self.args.model.params.convPool:
            x = self.pwconv(x).squeeze()
        elif self.args.model.milpool:
            # 在这里把highest_instance和x做QKV
            x = self.mil_pool(highest_instance, x)
        else:
            x = torch.mean(x, dim=1)

        # Predict Key Instance
        mu_t = self.s2d_fc1(x)
        mu_t = F.tanh(mu_t)
        if self.args.model.ki.sort:
            mu_t, _ = torch.sort(mu_t)
        delta_t = self.s2d_fc2(x)
        if self.args.model.ki.better_delta:
            delta_t = F.sigmoid(delta_t)
        else:
            delta_t = torch.where(delta_t > 0, -delta_t, delta_t)
        pred_instance_id = [mu_t, delta_t]

        bag_feature = x
        
        info = {'pred_instance_id': pred_instance_id, 
                'instance_feature': instance_feature,
                'instance_emo': instance_emo,
                'bag_feature': bag_feature,
                'manual_instance_id': manual_instance_id
                }
        return x, info

    def forward(self, x, y):

        batch_size = x.shape[0]
        bag_size = self.bag_size

        if self.bag_size == 16:
            x = x
            x = self.model(x, True).squeeze()
        else:
            x = rearrange(x, 'b (t1 t2) c h w -> (b t1) t2 c h w', t1=bag_size, t2=self.args.model.clip_length)
            # [batch*bag_size, cl, 3, 112, 112]
            x = self.model(x, False).squeeze()

        # [batch*bag_size, 512]
        x = rearrange(x, '(b t) c -> b t c', b=batch_size)
        # [batch, bag_size, 512]

        out, info = self.MIL(x, y)
        
        out = self.fc(out)
        if self.args.model.s2dattn:
            feature = info['bag_feature']
            attn = self.attnSigmoid(self.toAttn(feature))
            info['attn'] = attn
        else:
            info['attn'] = None

        return out, info
            
class DNet(nn.Module):

    def __init__(self, args):
        super(DNet, self).__init__()

        self.args = args
        self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = args.model.D_bagsize

        # backbone
        self.model = TSM(args=self.args, base_model='resnet50')
        feature_dim = 512

        if self.args.model.params.LSTM:
            self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, batch_first=True, bidirectional=True)
            feature_dim *= 2

        if self.args.model.params.MHSA:
            # attn
            self.heads = 8
            self.dim_head = feature_dim // self.heads
            self.scale = self.dim_head ** -0.5
            self.attend = nn.Softmax(dim = -1)
            self.to_qkv = nn.Linear(feature_dim, (self.dim_head * self.heads) * 3, bias = False)

        if self.args.model.params.Norm:
            self.layerNorm = nn.LayerNorm(feature_dim)

        if self.args.model.params.convPool:
            self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        if self.args.model.s2dattn:
            self.embed = nn.Parameter(torch.ones(feature_dim))

        if self.args.model.milpool:
            self.mil_pool = MAB(feature_dim, feature_dim)

        # classifier
        self.fc = nn.Linear(feature_dim, self.args.model.num_classes) # new fc layer 512x7

    def MIL(self, x, attn=None):
        
        # [B, bag_size, 512]
        # x = self.transformer_encoder(x)
        instance_feature = x

        if self.args.model.params.LSTM:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)

        if self.args.model.s2dattn:
            if attn is None:
                x = x * self.embed
            else:
                attn = attn.repeat(x.shape[1], 1, 1)
                attn = rearrange(attn, 't b c -> b t c')
                x = x * attn

        if self.args.model.params.MHSA:
            ori_x = x
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # [batch, head, n, d]

            attn = self.attend(dots)
            # [B, bag_size, bag_size]
            x = torch.matmul(attn, v)
            x = rearrange(x, 'b h n d -> b n (h d)')
            # x = self.softmax(x)
            if self.args.model.params.Norm:
                x = self.layerNorm(x)
            x = torch.sigmoid(x)
            x = ori_x * x

        # [B, bag_size, 512]
        instance_emo = self.fc(x)
        # [B, bag_size, 7]
        instance_emo = F.softmax(instance_emo, dim=2)

        #留下一个最好的
        if self.args.model.milpool:
            emo, _ = instance_emo.max(1)
            topemo = torch.topk(emo, 1, 1).indices
            highest_instance = emo.gather(1, topemo)

        # Merge feature
        if self.args.model.params.convPool:
            x = self.pwconv(x).squeeze()
        elif self.args.model.milpool:
            # 在这里把highest_instance和x做QKV
            x = self.mil_pool(highest_instance, x)
        else:
            x = torch.mean(x, dim=1)
            
        bag_feature = x
        
        info = {'instance_feature': instance_feature,
                'instance_emo': instance_emo,
                'bag_feature': bag_feature
                }
        return x, info

    def forward(self, x, attn=None):

        batch_size = x.shape[0]
        bag_size = self.bag_size

        if self.bag_size == 16:
            x = x
            x = self.model(x, True).squeeze()
        else:
            x = rearrange(x, 'b (t1 t2) c h w -> (b t1) t2 c h w', t1=bag_size, t2=self.args.model.clip_length)
            # [batch*bag_size, cl, 3, 112, 112]
            x = self.model(x, False).squeeze()

        # [batch*bag_size, 512]
        x = rearrange(x, '(b t) c -> b t c', b=batch_size)
        # [batch, bag_size, 512]
        out, info = self.MIL(x, attn)
        
        out = self.fc(out)

        return out, info
            
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_V, num_heads=1):
        super(MAB, self).__init__()
        self.dim_V     = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        print(Q.shape, K.shape)
        Q    = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O + F.relu(self.fc_o(O))
        print(O.shape)
        # [N,1,D] --> [N,D]
        return O.squeeze()