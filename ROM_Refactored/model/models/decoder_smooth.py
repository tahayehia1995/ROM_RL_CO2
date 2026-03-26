"""
Smooth Decoder module for E2C architecture
Uses Upsample + Conv instead of ConvTranspose3d to eliminate checkerboard artifacts.

This decoder is recommended for:
- Physics simulations requiring smooth reconstructions
- Cases where checkerboard/grid artifacts are visible
- Reservoir dimensions with odd sizes (like 34x16x25)

Supports two normalization/activation schemes:
- BatchNorm + ReLU (default, norm_type='batchnorm')
- IGDN + GELU (norm_type='gdn', better for perceptual quality)
"""

import torch
import torch.nn as nn
from model.layers.standard_layers import (
    SmoothUpsample3D, 
    smooth_conv_bn_relu_3d,
    smooth_conv_norm_act_3d,
    ResidualConv3D,
    ResidualConv3D_Configurable,
    DimensionAdjuster3D
)
from model.utils.initialization import weights_init


class DecoderSmooth(nn.Module):
    """
    Smooth decoder using trilinear upsampling + convolution.
    
    Architecture for (2, 34, 16, 25) reservoir:
        Latent (128) → FC → (128, 9, 4, 7)
        → ResBlocks → (128, 9, 4, 7)
        → Conv → (64, 9, 4, 7)
        → Upsample×2 + Conv → (32, 18, 8, 14)
        → Conv → (16, 18, 8, 14)
        → Upsample×2 + Conv → (16, 36, 16, 28)
        → Conv → (16, 36, 16, 28)
        → DimensionAdjust → (2, 34, 16, 25)
    
    Benefits:
        - No checkerboard artifacts
        - Smooth gradient flow
        - Better for physics reconstruction
    
    Supports:
        - norm_type='batchnorm': BatchNorm + ReLU (default)
        - norm_type='gdn': IGDN + GELU (better perceptual quality)
    """
    
    def __init__(self, config):
        super(DecoderSmooth, self).__init__()
        
        self.config = config
        self.input_shape = config['data']['input_shape']  # [2, 34, 16, 25]
        
        # Normalization type: 'batchnorm' (default) or 'gdn'
        self.norm_type = config['decoder'].get('norm_type', 
                         config['encoder'].get('norm_type', 'batchnorm'))
        
        # Use flattened size from config
        flattened_size = config['encoder']['flattened_size']
        latent_dim = config['model']['latent_dim']
        output_dims = config['encoder']['output_dims']  # [128, 9, 4, 7]
        
        if config['runtime']['verbose']:
            print(f"🔧 SMOOTH DECODER: Latent dim {latent_dim} → Flattened size {flattened_size}")
            print(f"🔧 SMOOTH DECODER: Reshape to {output_dims} → Output {self.input_shape}")
            print(f"✨ SMOOTH DECODER: Using Upsample + Conv (no checkerboard artifacts)")
            if self.norm_type == 'gdn':
                print(f"✨ SMOOTH DECODER: Using IGDN + GELU normalization/activation")
            else:
                print(f"🔧 SMOOTH DECODER: Using BatchNorm + ReLU normalization/activation")
        
        # Store dimensions for reshaping
        self.output_dims = output_dims
        self.flattened_size = flattened_size
        
        # ============ FC LAYERS ============
        if self.norm_type == 'gdn':
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, flattened_size),
                nn.GELU()
            )
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, flattened_size),
                nn.ReLU()
            )
        self.fc_layers.apply(weights_init)
        
        # ============ RESIDUAL BLOCKS ============
        res_layers = []
        residual_blocks = config['encoder'].get('residual_blocks', 0) or 0
        residual_channels = config['encoder'].get('residual_channels', 128)
        
        for i in range(residual_blocks):
            if self.norm_type == 'gdn':
                res_layers.append(
                    ResidualConv3D_Configurable(
                        residual_channels, 
                        residual_channels, 
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        norm_type='gdn'
                    )
                )
            else:
                res_layers.append(
                    ResidualConv3D(
                        residual_channels, 
                        residual_channels, 
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1)
                    )
                )
        self.res_layers = nn.Sequential(*res_layers)
        self.res_layers.apply(weights_init)
        
        # ============ SMOOTH UPSAMPLING LAYERS ============
        # Calculate intermediate sizes
        # Start: (128, 9, 4, 7)
        # After UP1: (64, 18, 8, 14)
        # After UP2: (16, 36, 16, 28)
        # Final: (2, 34, 16, 25)
        
        # Choose conv function based on norm_type
        if self.norm_type == 'gdn':
            conv_fn = lambda in_ch, out_ch: smooth_conv_norm_act_3d(
                in_ch, out_ch, kernel_size=3, padding=1, norm_type='gdn', inverse=True
            )
        else:
            conv_fn = lambda in_ch, out_ch: smooth_conv_bn_relu_3d(in_ch, out_ch, kernel_size=3, padding=1)
        
        # Layer 1: Channel reduction (128 → 64)
        self.conv1 = conv_fn(128, 64)
        self.conv1.apply(weights_init)
        
        # Layer 2: First upsample (9,4,7) → (18,8,14) with channel reduction (64 → 32)
        self.up1 = SmoothUpsample3D(64, 32, scale_factor=2, kernel_size=3, mode='trilinear', norm_type=self.norm_type)
        self.up1.apply(weights_init)
        
        # Layer 3: Feature refinement (32 → 16)
        self.conv2 = conv_fn(32, 16)
        self.conv2.apply(weights_init)
        
        # Layer 4: Second upsample (18,8,14) → (36,16,28) - maintain 16 channels
        self.up2 = SmoothUpsample3D(16, 16, scale_factor=2, kernel_size=3, mode='trilinear', norm_type=self.norm_type)
        self.up2.apply(weights_init)
        
        # Layer 5: Feature refinement before final adjustment
        self.conv3 = conv_fn(16, 16)
        self.conv3.apply(weights_init)
        
        # ============ DIMENSION ADJUSTMENT ============
        # (36, 16, 28) → (34, 16, 25)
        upsampled_size = (36, 16, 28)  # After 2x upsampling from (18, 8, 14)
        target_size = (self.input_shape[1], self.input_shape[2], self.input_shape[3])
        
        self.dim_adjust = DimensionAdjuster3D(
            in_channels=16,
            out_channels=self.input_shape[0],  # Output channels (2)
            input_size=upsampled_size,
            target_size=target_size,
            method='crop'  # Use symmetric cropping for smooth edges
        )
        self.dim_adjust.apply(weights_init)
        
        if config['runtime']['verbose']:
            print(f"📊 SMOOTH DECODER architecture:")
            print(f"   FC: {latent_dim} → {flattened_size}")
            print(f"   Reshape: → (128, 9, 4, 7)")
            print(f"   ResBlocks: {residual_blocks} blocks")
            print(f"   Conv1: 128 → 64, (9,4,7)")
            print(f"   UP1: 64 → 32, (9,4,7) → (18,8,14)")
            print(f"   Conv2: 32 → 16, (18,8,14)")
            print(f"   UP2: 16 → 16, (18,8,14) → (36,16,28)")
            print(f"   Conv3: 16 → 16, (36,16,28)")
            print(f"   Adjust: 16 → {self.input_shape[0]}, (36,16,28) → {target_size}")
    
    def forward(self, z):
        # FC expansion
        x = self.fc_layers(z)
        
        # Reshape to spatial dimensions
        x = x.view(-1, self.output_dims[0], self.output_dims[1], 
                   self.output_dims[2], self.output_dims[3])
        
        # Residual blocks
        x = self.res_layers(x)
        
        # Smooth upsampling path
        x = self.conv1(x)      # (128, 9, 4, 7) → (64, 9, 4, 7)
        x = self.up1(x)        # (64, 9, 4, 7) → (32, 18, 8, 14)
        x = self.conv2(x)      # (32, 18, 8, 14) → (16, 18, 8, 14)
        x = self.up2(x)        # (16, 18, 8, 14) → (16, 36, 16, 28)
        x = self.conv3(x)      # (16, 36, 16, 28) → (16, 36, 16, 28)
        
        # Final dimension adjustment
        y = self.dim_adjust(x)  # (16, 36, 16, 28) → (2, 34, 16, 25)
        
        return y


class DecoderSmoothGeneric(nn.Module):
    """
    Generic smooth decoder that adapts to any reservoir size.
    
    Automatically calculates the upsampling path based on encoder output
    dimensions and target input shape.
    """
    
    def __init__(self, config):
        super(DecoderSmoothGeneric, self).__init__()
        
        self.config = config
        self.input_shape = config['data']['input_shape']
        
        flattened_size = config['encoder']['flattened_size']
        latent_dim = config['model']['latent_dim']
        output_dims = config['encoder']['output_dims']
        
        self.output_dims = output_dims
        self.flattened_size = flattened_size
        
        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, flattened_size),
            nn.ReLU()
        )
        
        # Residual blocks
        res_layers = []
        residual_blocks = config['encoder'].get('residual_blocks', 0) or 0
        residual_channels = config['encoder'].get('residual_channels', output_dims[0])
        
        for _ in range(residual_blocks):
            res_layers.append(
                ResidualConv3D(residual_channels, residual_channels, 
                              kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
        self.res_layers = nn.Sequential(*res_layers)
        
        # Calculate upsampling path
        encoder_spatial = (output_dims[1], output_dims[2], output_dims[3])
        target_spatial = (self.input_shape[1], self.input_shape[2], self.input_shape[3])
        
        # Determine number of 2x upsamples needed
        scale_x = target_spatial[0] / encoder_spatial[0]
        scale_y = target_spatial[1] / encoder_spatial[1]
        scale_z = target_spatial[2] / encoder_spatial[2]
        
        # Typically 2 upsamples of 2x each = 4x total
        num_upsamples = 2
        
        # Build decoder layers dynamically
        channels = [output_dims[0], 64, 32, 16, 16]  # Channel progression
        
        layers = []
        in_ch = channels[0]
        
        for i in range(num_upsamples):
            out_ch = channels[i + 1]
            
            # Conv before upsample
            if i == 0:
                layers.append(('conv_pre', smooth_conv_bn_relu_3d(in_ch, out_ch, 3, 1)))
                in_ch = out_ch
                out_ch = channels[i + 2]
            
            # Upsample + Conv
            layers.append((f'up{i+1}', SmoothUpsample3D(in_ch, out_ch, scale_factor=2, kernel_size=3)))
            in_ch = out_ch
            
            # Refinement conv
            out_ch = channels[min(i + 3, len(channels) - 1)]
            layers.append((f'conv{i+1}', smooth_conv_bn_relu_3d(in_ch, out_ch, 3, 1)))
            in_ch = out_ch
        
        self.decoder_layers = nn.ModuleDict(dict(layers))
        
        # Calculate upsampled size
        upsampled_size = (
            encoder_spatial[0] * (2 ** num_upsamples),
            encoder_spatial[1] * (2 ** num_upsamples),
            encoder_spatial[2] * (2 ** num_upsamples)
        )
        
        # Dimension adjustment
        self.dim_adjust = DimensionAdjuster3D(
            in_channels=in_ch,
            out_channels=self.input_shape[0],
            input_size=upsampled_size,
            target_size=target_spatial,
            method='crop'
        )
        
        # Initialize weights
        self.apply(weights_init)
        
        if config['runtime']['verbose']:
            print(f"🔧 GENERIC SMOOTH DECODER initialized")
            print(f"   Encoder output: {output_dims}")
            print(f"   Target shape: {self.input_shape}")
            print(f"   Upsampled size: {upsampled_size}")
    
    def forward(self, z):
        x = self.fc_layers(z)
        x = x.view(-1, self.output_dims[0], self.output_dims[1], 
                   self.output_dims[2], self.output_dims[3])
        x = self.res_layers(x)
        
        for name, layer in self.decoder_layers.items():
            x = layer(x)
        
        y = self.dim_adjust(x)
        return y
