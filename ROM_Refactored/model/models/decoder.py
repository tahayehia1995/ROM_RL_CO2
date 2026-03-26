"""
Decoder module for E2C architecture
Decodes latent representations back to spatial states

Supports two normalization/activation schemes:
- BatchNorm + ReLU (default, norm_type='batchnorm')
- IGDN + GELU (norm_type='gdn', better for perceptual quality)
"""

import torch
import torch.nn as nn
from model.layers.standard_layers import (
    dconv_bn_nolinear_3d, 
    dconv_norm_act_3d,
    ResidualConv3D,
    ResidualConv3D_Configurable,
    IGDN3D
)
from model.utils.initialization import weights_init


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.config = config
        self.input_shape = config['data']['input_shape']
        
        # Normalization type: 'batchnorm' (default) or 'gdn'
        self.norm_type = config['decoder'].get('norm_type', 
                         config['encoder'].get('norm_type', 'batchnorm'))
        
        # Use flattened size from config
        flattened_size = config['encoder']['flattened_size']
        latent_dim = config['model']['latent_dim']
        
        if config['runtime']['verbose']:
            print(f"🔧 DECODER: Latent dim {latent_dim} → Flattened size {flattened_size}")
            print(f"🔧 DECODER: Reshape to {config['encoder']['output_dims']} → Output {self.input_shape}")
            print(f"📊 DECODER: Using {latent_dim * flattened_size:,} parameters in linear layer")
            if self.norm_type == 'gdn':
                print(f"✨ DECODER: Using IGDN + GELU normalization/activation")
            else:
                print(f"🔧 DECODER: Using BatchNorm + ReLU normalization/activation")
        
        # FC expansion from latent to encoded space
        # Use GELU for GDN mode, ReLU otherwise
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
        
        # 3D ResNet blocks for feature refinement
        res_layers = []
        residual_blocks = config['encoder'].get('residual_blocks', 0) or 0
        for i in range(residual_blocks):
            if self.norm_type == 'gdn':
                res_layers.append(
                    ResidualConv3D_Configurable(
                        config['encoder']['residual_channels'], 
                        config['encoder']['residual_channels'], 
                        kernel_size=(3, 3, 3),
                        norm_type='gdn'
                    )
                )
            else:
                res_layers.append(
                    ResidualConv3D(
                        config['encoder']['residual_channels'], 
                        config['encoder']['residual_channels'], 
                        kernel_size=(3, 3, 3)
                    )
                )
        self.upsample_layers = nn.Sequential(*res_layers)
        self.upsample_layers.apply(weights_init)

        # Build deconv layers from config with EXACT dimensions
        deconv_config = config['decoder']['deconv_layers']
        
        # Choose deconv function based on norm_type
        if self.norm_type == 'gdn':
            dconv_fn = lambda in_ch, out_ch, ks, st, pd: dconv_norm_act_3d(
                in_ch, out_ch, kernel_size=ks, stride=st, padding=pd, norm_type='gdn'
            )
        else:
            dconv_fn = dconv_bn_nolinear_3d

        # Layer 1: (128,9,4,7) → (64,9,4,7)
        self.deconv1 = dconv_fn(
            deconv_config['deconv1'][0], deconv_config['deconv1'][1], 
            tuple(deconv_config['deconv1'][2]), 
            tuple(deconv_config['deconv1'][3]), 
            tuple(deconv_config['deconv1'][4])
        )
        
        # Layer 2: (64,9,4,7) → (32,18,8,14)
        self.deconv2 = dconv_fn(
            deconv_config['deconv2'][0], deconv_config['deconv2'][1], 
            tuple(deconv_config['deconv2'][2]), 
            tuple(deconv_config['deconv2'][3]), 
            tuple(deconv_config['deconv2'][4])
        )
        
        # Layer 3: (32,18,8,14) → (16,18,8,14)  
        self.deconv3 = dconv_fn(
            deconv_config['deconv3'][0], deconv_config['deconv3'][1], 
            tuple(deconv_config['deconv3'][2]), 
            tuple(deconv_config['deconv3'][3]), 
            tuple(deconv_config['deconv3'][4])
        )
        
        # Layer 4: (16,18,8,14) → (16,34,16,25) - EXACT dimensions with output_padding
        deconv4_config = deconv_config['deconv4']
        if len(deconv4_config) > 5:  # Has output_padding
            if self.norm_type == 'gdn':
                self.deconv4 = nn.Sequential(
                    nn.ConvTranspose3d(
                        deconv4_config[0], deconv4_config[1],
                        kernel_size=tuple(deconv4_config[2]),
                        stride=tuple(deconv4_config[3]),
                        padding=tuple(deconv4_config[4]),
                        output_padding=tuple(deconv4_config[5])
                    ),
                    IGDN3D(deconv4_config[1]),
                    nn.GELU()
                )
            else:
                self.deconv4 = nn.Sequential(
                    nn.ConvTranspose3d(
                        deconv4_config[0], deconv4_config[1],
                        kernel_size=tuple(deconv4_config[2]),
                        stride=tuple(deconv4_config[3]),
                        padding=tuple(deconv4_config[4]),
                        output_padding=tuple(deconv4_config[5])
                    ),
                    nn.BatchNorm3d(deconv4_config[1]),
                    nn.ReLU()
                )
        else:
            self.deconv4 = dconv_fn(
                deconv4_config[0], deconv4_config[1], 
                tuple(deconv4_config[2]), 
                tuple(deconv4_config[3]), 
                tuple(deconv4_config[4])
            )
        
        # Final layer: (16,36,16,25) → (3,34,16,25) - EXACT dimensions
        final_conv_out_channels = deconv_config['final_conv'][1] if deconv_config['final_conv'][1] is not None else config['data']['input_shape'][0]
        self.final_conv = nn.Conv3d(
            deconv_config['final_conv'][0], 
            final_conv_out_channels,
            kernel_size=tuple(deconv_config['final_conv'][2]), 
            stride=tuple(deconv_config['final_conv'][3]), 
            padding=tuple(deconv_config['final_conv'][4])
        )
        
        # Apply initialization
        self.deconv1.apply(weights_init)
        self.deconv2.apply(weights_init)
        self.deconv3.apply(weights_init)
        self.deconv4.apply(weights_init)
        self.final_conv.apply(weights_init)
        
        # Store dimensions
        self.flattened_size = flattened_size
        self.use_exact_dimensions = config['decoder'].get('use_exact_dimensions', False)

    def forward(self, z):
        x = self.fc_layers(z)
        
        # Reshape using config dimensions
        output_dims = self.config['encoder']['output_dims']
        x = x.view(-1, output_dims[0], output_dims[1], output_dims[2], output_dims[3])
        x = self.upsample_layers(x)
        
        # Decoder layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        # Final convolution for exact output dimensions
        y = self.final_conv(x)
        
        # No cropping needed with exact dimensions
        if not self.use_exact_dimensions and self.config['decoder'].get('crop_z_to', None):
            y = y[:, :, :, :, :self.config['decoder']['crop_z_to']]
        
        return y


