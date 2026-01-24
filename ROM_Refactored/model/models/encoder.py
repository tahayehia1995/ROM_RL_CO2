"""
Encoder module for E2C architecture
Encodes spatial states to latent representations
"""

import torch
import torch.nn as nn
from model.layers.standard_layers import conv_bn_relu_3d, ResidualConv3D
from model.utils.initialization import weights_init


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config
        self.sigma = config.model['sigma']
        self.input_shape = config.data['input_shape']
        
        # Build CNN layers from config with skip connection support
        conv_config = config['encoder']['conv_layers']
        
        # Layer 1: Input channels from config (replace null with actual channels)
        conv1_in_channels = conv_config['conv1'][0] if conv_config['conv1'][0] is not None else self.input_shape[0]
        self.conv1 = conv_bn_relu_3d(
            conv1_in_channels, conv_config['conv1'][1], 
            kernel_size=tuple(conv_config['conv1'][2]), 
            stride=tuple(conv_config['conv1'][3]), 
            padding=tuple(conv_config['conv1'][4])
        )
        # Layer 2
        self.conv2 = conv_bn_relu_3d(
            conv_config['conv2'][0], conv_config['conv2'][1], 
            kernel_size=tuple(conv_config['conv2'][2]), 
            stride=tuple(conv_config['conv2'][3]), 
            padding=tuple(conv_config['conv2'][4])
        )
        # Layer 3
        self.conv3 = conv_bn_relu_3d(
            conv_config['conv3'][0], conv_config['conv3'][1], 
            kernel_size=tuple(conv_config['conv3'][2]), 
            stride=tuple(conv_config['conv3'][3]), 
            padding=tuple(conv_config['conv3'][4])
        )
        # Layer 4
        self.conv4 = conv_bn_relu_3d(
            conv_config['conv4'][0], conv_config['conv4'][1], 
            kernel_size=tuple(conv_config['conv4'][2]), 
            stride=tuple(conv_config['conv4'][3]), 
            padding=tuple(conv_config['conv4'][4])
        )
        
        # Apply initialization
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        
        # 3D ResNet blocks from config
        res_layers = []
        residual_blocks = config['encoder'].get('residual_blocks', 0) or 0
        for i in range(residual_blocks):
            res_layers.append(
                ResidualConv3D(
                    config['encoder']['residual_channels'], 
                    config['encoder']['residual_channels'], 
                    kernel_size=(3, 3, 3), 
                    stride=(1, 1, 1)
                )
            )
        self.res_layers = nn.Sequential(*res_layers)
        self.res_layers.apply(weights_init)

        self.flatten = nn.Flatten()
        
        # Use flattened size from config
        flattened_size = config['encoder']['flattened_size']
        latent_dim = config['model']['latent_dim']
        
        if config['runtime']['verbose']:
            print(f"🔧 ENCODER: Input shape {self.input_shape} → CNN output {config['encoder']['output_dims']}")
            print(f"🔧 ENCODER: Flattened size {flattened_size} → Latent dim {latent_dim}")
            print(f"📊 ENCODER: Using {flattened_size * latent_dim:,} parameters in linear layer")
        
        self.fc_mean = nn.Linear(flattened_size, latent_dim)
        self.fc_mean.apply(weights_init)
        
        # Store dimensions for decoder
        self.flattened_size = flattened_size

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        
        # Layer 2
        x = self.conv2(x)
            
        # Layer 3
        x = self.conv3(x)
            
        # Layer 4
        x = self.conv4(x)
        
        # ResNet blocks
        x = self.res_layers(x)
        
        # Flatten and encode
        x_flattened = self.flatten(x)
        xi_mean = self.fc_mean(x_flattened)
        
        return xi_mean

