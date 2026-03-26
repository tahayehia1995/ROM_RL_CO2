"""
Model weight initialization utilities
"""

import torch
import torch.nn as nn


def weights_init(m):
    """Initialize weights using orthogonal initialization"""
    if type(m) in [nn.Conv2d, nn.Conv3d, nn.Linear, nn.ConvTranspose2d, nn.ConvTranspose3d]:
        torch.nn.init.orthogonal_(m.weight)

