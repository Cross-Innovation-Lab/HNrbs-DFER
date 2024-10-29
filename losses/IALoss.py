import torch.nn as nn
import torch

import torch.nn.functional as F

class IALoss(nn.Module):
    def __init__(self):
        super(IALoss, self).__init__()

    def forward(self, input, target, device):
        
        loss = torch.tensor(0.0).to(device)
        input = F.softmax(input)

        e_t = input.gather(1, target.unsqueeze(1))
        topk = torch.topk(input, 2, 1).indices
        max = torch.where(topk[:,0]==target, topk[:,1], topk[:,0])
        e_max = input.gather(1, max.unsqueeze(1))
        loss = -torch.log(torch.mean(e_t/(e_t + e_max + 1e5)))
        return loss