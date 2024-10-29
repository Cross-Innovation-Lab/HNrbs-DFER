import torch.nn as nn
import torch

import torch.nn.functional as F

class KICELoss(nn.Module):
    def __init__(self):
        super(KICELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        
        # input [B, L, C]
        # target [B]

        y = target.view(-1, 1, 1)
        y = y.repeat(1, 3, 1)
        e_t = input.gather(2, y)
        _, id = torch.max(e_t, dim=1)
        id = id.view(1, -1, 1)
        id = id.repeat(1, 1, 7)
        input = input.gather(1, id).squeeze()

        return self.ce(input, target)