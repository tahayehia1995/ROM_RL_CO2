"""
Standard foundational layers for E2C model
Includes convolutional, residual, deconvolutional, and pooling layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

