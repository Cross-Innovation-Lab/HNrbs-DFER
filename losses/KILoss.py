import torch.nn as nn
import torch

import torch.nn.functional as F

class KILoss(nn.Module):
    def __init__(self):
        super(KILoss, self).__init__()

    def forward(self, input, target):
        
        target, _ = torch.sort(target)
        input, _ = torch.sort(input)
        loss = torch.sum(abs(target - input))
        loss = abs(loss) / input.shape[0]
        return loss