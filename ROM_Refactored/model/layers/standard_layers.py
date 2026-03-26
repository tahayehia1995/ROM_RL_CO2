"""
Standard foundational layers for E2C model
Includes convolutional, residual, deconvolutional, and pooling layers

Supports two normalization/activation schemes:
- BatchNorm + ReLU (default, good for classification/physics accuracy)
- GDN + GELU (better for perceptual quality/smooth reconstructions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# GDN (Generalized Divisive Normalization) LAYERS
# ============================================================================

class LowerBound(nn.Module):
    """Lower bound operator with custom gradient.
    
    Computes torch.max(x, bound) with a modified gradient that passes through
    when x is moved towards the bound.
    """
    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    def forward(self, x: Tensor) -> Tensor:
        # Use straight-through estimator for gradient
        return torch.max(x, self.bound)


class NonNegativeParametrizer(nn.Module):
    """Non-negative reparametrization for stable training.
    
    Ensures parameters stay non-negative using a soft-plus-like transformation.
    """
    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        
        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


class GDN3D(nn.Module):
    """3D Generalized Divisive Normalization layer.
    
    Introduced in "Density Modeling of Images Using a Generalized Normalization
    Transformation" by Ballé et al., 2016.
    
    GDN performs local divisive normalization which:
    - Models lateral inhibition in visual cortex
    - Gaussianizes responses to natural images
    - Provides better perceptual quality than BatchNorm for reconstruction tasks
    
    Math:
        y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j,i] * x[j]^2))
    
    Args:
        in_channels: Number of input channels
        inverse: If True, computes IGDN (Inverse GDN) for decoder
        beta_min: Minimum value for beta (numerical stability)
        gamma_init: Initial value for gamma diagonal
    """
    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.inverse = bool(inverse)
        
        # Beta parameter (per-channel)
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)
        
        # Gamma parameter (channel interaction matrix)
        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, channels, X, Y, Z)
        C = x.size(1)
        
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        
        # Reshape gamma for 3D convolution: (out_channels, in_channels, 1, 1, 1)
        gamma = gamma.reshape(C, C, 1, 1, 1)
        
        # Compute normalization factor using 3D convolution
        # This computes: sum_j(gamma[j,i] * x[j]^2) for each spatial position
        norm = F.conv3d(x**2, gamma, beta)
        
        if self.inverse:
            # IGDN: multiply by sqrt(norm) - used in decoder
            norm = torch.sqrt(norm)
        else:
            # GDN: divide by sqrt(norm) - used in encoder
            norm = torch.rsqrt(norm)
        
        return x * norm


class IGDN3D(GDN3D):
    """Inverse GDN for decoder/upsampling paths."""
    def __init__(self, in_channels: int, beta_min: float = 1e-6, gamma_init: float = 0.1):
        super().__init__(in_channels, inverse=True, beta_min=beta_min, gamma_init=gamma_init)


# ============================================================================
# CONFIGURABLE CONV BLOCKS (BatchNorm+ReLU or GDN+GELU)
# ============================================================================

def get_norm_activation_3d(out_channels: int, norm_type: str = 'batchnorm', inverse: bool = False):
    """
    Get normalization and activation layers based on configuration.
    
    Args:
        out_channels: Number of output channels
        norm_type: 'batchnorm' (default) or 'gdn'
        inverse: If True and norm_type='gdn', use IGDN (for decoder)
    
    Returns:
        Tuple of (normalization_layer, activation_layer)
    """
    if norm_type == 'gdn':
        if inverse:
            norm = IGDN3D(out_channels)
        else:
            norm = GDN3D(out_channels)
        activation = nn.GELU()
    else:
        # Default: BatchNorm + ReLU
        norm = nn.BatchNorm3d(out_channels)
        activation = nn.ReLU()
    
    return norm, activation


def conv_norm_act_3d(in_filter, out_filter, kernel_size=(3, 3, 3), stride=(1, 1, 1), 
                     padding=(1, 1, 1), norm_type='batchnorm'):
    """
    3D Convolution with configurable normalization and activation.
    
    Args:
        in_filter: Input channels
        out_filter: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        norm_type: 'batchnorm' (BatchNorm3d + ReLU) or 'gdn' (GDN + GELU)
    """
    norm, activation = get_norm_activation_3d(out_filter, norm_type, inverse=False)
    return nn.Sequential(
        nn.Conv3d(in_filter, out_filter, kernel_size=kernel_size, stride=stride, padding=padding),
        norm,
        activation
    )


def dconv_norm_act_3d(in_filter, out_filter, kernel_size=(3, 3, 3), stride=(2, 2, 2), 
                      padding=(1, 1, 1), norm_type='batchnorm'):
    """
    3D Transpose Convolution with configurable normalization and activation.
    
    Args:
        in_filter: Input channels
        out_filter: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        norm_type: 'batchnorm' (BatchNorm3d + ReLU) or 'gdn' (IGDN + GELU for decoder)
    """
    # Use inverse GDN (IGDN) for decoder/upsampling
    norm, activation = get_norm_activation_3d(out_filter, norm_type, inverse=True)
    return nn.Sequential(
        nn.ConvTranspose3d(in_filter, out_filter, kernel_size=kernel_size, stride=stride, padding=padding),
        norm,
        activation
    )


class ResidualConv3D_Configurable(nn.Module):
    """
    3D Residual Convolutional block with configurable normalization.
    
    Args:
        in_filter: Input channels
        out_filter: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        norm_type: 'batchnorm' or 'gdn'
    """
    def __init__(self, in_filter, out_filter, kernel_size=(3, 3, 3), stride=(1, 1, 1), norm_type='batchnorm'):
        super().__init__()
        
        self.norm_type = norm_type
        
        self.conv1 = nn.Conv3d(in_filter, out_filter, kernel_size=kernel_size, stride=stride, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(out_filter, out_filter, kernel_size=kernel_size, stride=stride, padding=(1, 1, 1))
        
        if norm_type == 'gdn':
            self.norm1 = GDN3D(out_filter)
            self.norm2 = GDN3D(out_filter)
            self.activation = nn.GELU()
        else:
            self.norm1 = nn.BatchNorm3d(out_filter)
            self.norm2 = nn.BatchNorm3d(out_filter)
            self.activation = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        
        a = self.conv1(x)
        a = self.norm1(a)
        a = self.activation(a)
        
        a = self.conv2(a)
        a = self.norm2(a)
        
        y = identity + a
        return y


# ============================================================================
# ORIGINAL LAYERS (BatchNorm + ReLU) - Kept for backward compatibility
# ============================================================================

def fc_bn_relu(input_dim, output_dim=None):
    """
    Fully connected layer with batch normalization and ReLU
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (if None, uses input_dim for residual connections)
    """
    if output_dim is None:
        output_dim = input_dim
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )


def conv_bn_relu(in_filter, out_filter, nb_row, nb_col, stride=1):
    """2D Convolutional layer with batch normalization and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_filter, out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1)),
        nn.BatchNorm2d(out_filter),
        nn.ReLU()
    )


# 3D CNN version for reservoir data processing (batch, channels, X, Y, Z)
def conv_bn_relu_3d(in_filter, out_filter, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
    """3D Convolutional layer with batch normalization and ReLU"""
    return nn.Sequential(
        nn.Conv3d(in_filter, out_filter, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm3d(out_filter),
        nn.ReLU()
    )


class ResidualConv(nn.Module):
    """2D Residual Convolutional block with skip connections"""
    def __init__(self, in_filter, out_filter, nb_row, nb_col, stride=(1, 1)):
        super(ResidualConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filter)
        self.conv2 = nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_filter)

    def forward(self, x):
        identity = x.clone()

        a = self.conv1(x)
        a = self.bn1(a)
        a = F.relu(a)

        a = self.conv2(a)
        a = self.bn2(a)

        y = identity + a

        return y


# 3D Residual Convolution for reservoir data with skip connections
class ResidualConv3D(nn.Module):
    """3D Residual Convolutional block with skip connections"""
    def __init__(self, in_filter, out_filter, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super(ResidualConv3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_filter, out_channels=out_filter, kernel_size=kernel_size, stride=stride, padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_filter)
        self.conv2 = nn.Conv3d(in_channels=in_filter, out_channels=out_filter, kernel_size=kernel_size, stride=stride, padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_filter)

    def forward(self, x):
        identity = x.clone()

        a = self.conv1(x)
        a = self.bn1(a)
        a = F.relu(a)

        a = self.conv2(a)
        a = self.bn2(a)

        y = identity + a

        return y


def dconv_bn_nolinear(in_filter, out_filter, nb_row, nb_col, stride=(2, 2), activation="relu", padding=0):
    """2D Transpose Convolutional layer with batch normalization and ReLU"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_filter, out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=padding),
        nn.BatchNorm2d(out_filter),
        nn.ReLU()
    )


# 3D Transpose Convolution for decoder upsampling in reservoir reconstruction
def dconv_bn_nolinear_3d(in_filter, out_filter, kernel_size=(3, 3, 3), stride=(2, 2, 2), activation="relu", padding=(1, 1, 1)):
    """3D Transpose Convolutional layer with batch normalization and ReLU"""
    return nn.Sequential(
        nn.ConvTranspose3d(in_filter, out_filter, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm3d(out_filter),
        nn.ReLU()
    )


class ReflectionPadding2D(nn.Module):
    """2D Reflection Padding layer"""
    def __init__(self, padding=(1, 1)):
        super(ReflectionPadding2D, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), 'reflect')


class UnPooling2D(nn.Module):
    """2D UnPooling layer using interpolation"""
    def __init__(self, size=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.size, mode='nearest')


# 3D UnPooling for reservoir layer upsampling
class UnPooling3D(nn.Module):
    """3D UnPooling layer using interpolation"""
    def __init__(self, size=(2, 2, 2)):
        super(UnPooling3D, self).__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.size, mode='nearest')


# Smooth upsampling block (Upsample + Conv) - NO CHECKERBOARD ARTIFACTS
class SmoothUpsample3D(nn.Module):
    """
    Smooth 3D upsampling block using trilinear interpolation followed by convolution.
    This approach eliminates checkerboard artifacts that occur with ConvTranspose3d.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale_factor: Upsampling factor (default: 2)
        kernel_size: Convolution kernel size (default: 3)
        mode: Interpolation mode - 'trilinear' or 'nearest' (default: 'trilinear')
        norm_type: 'batchnorm' (default) or 'gdn' for GELU+GDN
    """
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, mode='trilinear', norm_type='batchnorm'):
        super(SmoothUpsample3D, self).__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode
        self.norm_type = norm_type
        
        # Padding to maintain spatial dimensions after conv
        padding = kernel_size // 2
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False if mode == 'trilinear' else None)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Use IGDN for decoder (inverse=True) with GELU
        if norm_type == 'gdn':
            self.norm = IGDN3D(out_channels)
            self.activation = nn.GELU()
        else:
            self.norm = nn.BatchNorm3d(out_channels)
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# Smooth convolution block with configurable normalization
def smooth_conv_norm_act_3d(in_filter, out_filter, kernel_size=3, padding=1, norm_type='batchnorm', inverse=False):
    """
    3D Convolution block with configurable normalization.
    Uses same padding to maintain spatial dimensions.
    
    Args:
        in_filter: Input channels
        out_filter: Output channels
        kernel_size: Convolution kernel size
        padding: Convolution padding
        norm_type: 'batchnorm' or 'gdn'
        inverse: If True and norm_type='gdn', use IGDN (for decoder)
    """
    norm, activation = get_norm_activation_3d(out_filter, norm_type, inverse=inverse)
    return nn.Sequential(
        nn.Conv3d(in_filter, out_filter, kernel_size=kernel_size, stride=1, padding=padding),
        norm,
        activation
    )


# Legacy function - kept for backward compatibility
def smooth_conv_bn_relu_3d(in_filter, out_filter, kernel_size=3, padding=1):
    """
    3D Convolution block with smooth gradient flow.
    Uses same padding to maintain spatial dimensions.
    
    Note: For configurable normalization, use smooth_conv_norm_act_3d instead.
    """
    return nn.Sequential(
        nn.Conv3d(in_filter, out_filter, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm3d(out_filter),
        nn.ReLU()
    )


# Final adjustment layer for exact output dimensions
class DimensionAdjuster3D(nn.Module):
    """
    Adjusts spatial dimensions to exact target size.
    Uses asymmetric convolution or cropping to achieve exact dimensions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        input_size: Tuple (X, Y, Z) of input spatial dimensions
        target_size: Tuple (X, Y, Z) of target spatial dimensions
        method: 'crop' or 'conv' (default: 'crop')
    """
    def __init__(self, in_channels, out_channels, input_size, target_size, method='crop'):
        super(DimensionAdjuster3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.target_size = target_size
        self.method = method
        
        # Calculate difference
        diff_x = input_size[0] - target_size[0]
        diff_y = input_size[1] - target_size[1]
        diff_z = input_size[2] - target_size[2]
        
        if method == 'crop':
            # Calculate crop amounts (symmetric cropping where possible)
            self.crop_x = (diff_x // 2, diff_x - diff_x // 2)
            self.crop_y = (diff_y // 2, diff_y - diff_y // 2)
            self.crop_z = (diff_z // 2, diff_z - diff_z // 2)
            
            # Final 1x1x1 conv to adjust channels
            self.final_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        elif method == 'conv':
            # Use asymmetric convolution to adjust dimensions
            # Kernel size = diff + 1 to reduce dimension by diff
            kernel_x = max(1, diff_x + 1) if diff_x > 0 else 1
            kernel_y = max(1, diff_y + 1) if diff_y > 0 else 1
            kernel_z = max(1, diff_z + 1) if diff_z > 0 else 1
            
            self.final_conv = nn.Conv3d(
                in_channels, out_channels, 
                kernel_size=(kernel_x, kernel_y, kernel_z),
                stride=1, padding=0
            )
    
    def forward(self, x):
        if self.method == 'crop':
            # Apply symmetric cropping
            x = x[:, :, 
                  self.crop_x[0]:x.shape[2] - self.crop_x[1] if self.crop_x[1] > 0 else x.shape[2],
                  self.crop_y[0]:x.shape[3] - self.crop_y[1] if self.crop_y[1] > 0 else x.shape[3],
                  self.crop_z[0]:x.shape[4] - self.crop_z[1] if self.crop_z[1] > 0 else x.shape[4]]
            x = self.final_conv(x)
        else:
            x = self.final_conv(x)
        
        return x

