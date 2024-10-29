from .KILoss import KILoss
from .KICELoss import KICELoss
import torch

def create_loss(args, loss_name):

    if loss_name == 'KILoss':
        return KILoss()
    elif loss_name == 'RLLoss':
        return KICELoss()
    elif loss_name == 'KICELoss':
        return KICELoss()
