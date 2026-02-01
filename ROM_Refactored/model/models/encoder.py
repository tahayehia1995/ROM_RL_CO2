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
        
        # VAE mode flag
        self.enable_vae = config.model.get('enable_vae', False)
        
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
        
        # VAE: Add log-variance layer if VAE mode enabled
        if self.enable_vae:
            self.fc_logvar = nn.Linear(flattened_size, latent_dim)
            self.fc_logvar.apply(weights_init)
            if config['runtime']['verbose']:
                print(f"🔧 ENCODER: VAE mode enabled - added fc_logvar layer")
        
        # Store dimensions for decoder
        self.flattened_size = flattened_size
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        Samples z = mu + eps * std where eps ~ N(0, I)
        
        Args:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        
        # VAE mode: compute log-variance and sample
        if self.enable_vae:
            xi_logvar = self.fc_logvar(x_flattened)
            z = self.reparameterize(xi_mean, xi_logvar)
            return z, xi_mean, xi_logvar
        
        # Deterministic AE mode: return mean only (with None placeholders for compatibility)
        return xi_mean, None, None

