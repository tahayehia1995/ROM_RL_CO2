"""
Spatial enhancement loss functions
Includes gradient loss and discriminator for adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """Spatial gradient loss for preserving sharp edges and fine details"""
    def __init__(self, directions=['x', 'y', 'z'], loss_type='l1'):
        super(GradientLoss, self).__init__()
        self.directions = directions
        self.loss_type = loss_type
        
    def forward(self, pred, target):
        """
        Compute gradient loss between predicted and target 3D fields
        Args:
            pred: (batch, channels, D, H, W) predicted field
            target: (batch, channels, D, H, W) target field
        """
        total_loss = 0.0
        
        for direction in self.directions:
            if direction == 'x' and pred.size(-2) > 1:  # Height dimension
                pred_grad = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
                target_grad = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
            elif direction == 'y' and pred.size(-1) > 1:  # Width dimension  
                pred_grad = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
                target_grad = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
            elif direction == 'z' and pred.size(-3) > 1:  # Depth dimension
                pred_grad = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
                target_grad = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
            else:
                continue
                
            if self.loss_type == 'l1':
                grad_loss = torch.mean(torch.abs(pred_grad - target_grad))
            else:  # l2
                grad_loss = torch.mean((pred_grad - target_grad) ** 2)
                
            total_loss += grad_loss
            
        return total_loss / len(self.directions)


# ===== SPATIAL ENHANCEMENT: Discriminator Network (Option 4) =====  
class Discriminator3D(nn.Module):
    """3D CNN Discriminator for adversarial training"""
    def __init__(self, config):
        super(Discriminator3D, self).__init__()
        self.config = config
        
        if not config['discriminator']['enable']:
            return
            
        # Build discriminator layers from config
        disc_config = config['discriminator']['conv_layers']
        input_channels = disc_config['conv1'][0] if disc_config['conv1'][0] is not None else config['data']['input_shape'][0]
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv3d(
                input_channels, disc_config['conv1'][1],
                kernel_size=tuple(disc_config['conv1'][2]),
                stride=tuple(disc_config['conv1'][3]),
                padding=tuple(disc_config['conv1'][4])
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv3d(
                disc_config['conv2'][0], disc_config['conv2'][1],
                kernel_size=tuple(disc_config['conv2'][2]),
                stride=tuple(disc_config['conv2'][3]),
                padding=tuple(disc_config['conv2'][4])
            ),
            nn.BatchNorm3d(disc_config['conv2'][1]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv3d(
                disc_config['conv3'][0], disc_config['conv3'][1],
                kernel_size=tuple(disc_config['conv3'][2]),
                stride=tuple(disc_config['conv3'][3]),
                padding=tuple(disc_config['conv3'][4])
            ),
            nn.BatchNorm3d(disc_config['conv3'][1]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv3d(
                disc_config['conv4'][0], disc_config['conv4'][1],
                kernel_size=tuple(disc_config['conv4'][2]),
                stride=tuple(disc_config['conv4'][3]),
                padding=tuple(disc_config['conv4'][4])
            ),
            nn.BatchNorm3d(disc_config['conv4'][1]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Global average pooling + final classification
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(disc_config['conv4'][1], config['discriminator']['final_layers']['hidden_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config['discriminator']['final_layers']['dropout']),
            nn.Linear(config['discriminator']['final_layers']['hidden_dim'], 1)
        )
        
        # Initialize weights
        try:
            from ..utils.initialization import weights_init
            self.apply(weights_init)
        except ImportError:
            # Fallback if utils not available
            pass
        
    def forward(self, x):
        if not self.config['discriminator']['enable']:
            return torch.zeros(x.size(0), 1, device=x.device)
            
        x = self.conv_layers(x)
        x = self.final_layers(x)
        return x

