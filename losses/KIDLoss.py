import torch.nn as nn
import torch

import torch.nn.functional as F

class KIDLoss(nn.Module):
    def __init__(self):
        super(KIDLoss, self).__init__()

    def forward(self, input):
        
        # [B, 4]
        
        
        return loss