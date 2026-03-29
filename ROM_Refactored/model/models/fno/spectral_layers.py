"""
Spectral Convolution Layers for FNO Encoder-Decoder
=====================================================
3D spectral convolution and invertible spectral blocks using
torch.fft.rfftn / irfftn for global receptive field.

References:
    - Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
    - FINE: arXiv:2505.15329 (invertible residual design)
"""

import torch
import torch.nn as nn
import math


class SpectralConv3d(nn.Module):
    """3D spectral convolution via truncated Fourier modes.

    Performs a linear transform in frequency domain on the retained modes,
    then returns to physical space.  The local bypass (1x1 Conv3d) is handled
    externally by the block wrapper.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes: tuple[int, int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # (m1, m2, m3)

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                modes[0], modes[1], modes[2], 2)
        )

    def _complex_mul(self, x_ft, weight):
        """Batched complex multiply: (B,Ci,m1,m2,m3) x (Ci,Co,m1,m2,m3) -> (B,Co,m1,m2,m3)."""
        w = torch.view_as_complex(weight)
        return torch.einsum("bixyz,ioxyz->boxyz", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Nx, Ny, Nz = x.shape
        m1, m2, m3 = self.modes

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            B, self.out_channels, Nx, Ny, Nz // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        out_ft[:, :, :m1, :m2, :m3] = self._complex_mul(
            x_ft[:, :, :m1, :m2, :m3], self.weight
        )

        return torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), dim=(-3, -2, -1))


class InvertibleSpectralBlock(nn.Module):
    """Approximately invertible FNO block: y = x + alpha * F(x).

    When alpha is small, the block is a near-identity perturbation
    and thus approximately invertible by fixed-point iteration:
        x_approx = y - alpha * F(y)  (one Newton step)

    This design follows FINE (arXiv:2505.15329) for invertibility.
    """

    def __init__(self, channels: int, modes: tuple[int, int, int],
                 alpha: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.alpha = alpha
        self.spectral_conv = SpectralConv3d(channels, channels, modes)
        self.bypass = nn.Conv3d(channels, channels, kernel_size=1)

        act_map = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU}
        self.act = act_map.get(activation, nn.GELU)()

        nn.init.zeros_(self.bypass.bias)
        nn.init.orthogonal_(self.bypass.weight, gain=0.1)

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral_conv(x) + self.bypass(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self._residual(x)

    def inverse(self, y: torch.Tensor, n_steps: int = 3) -> torch.Tensor:
        """Approximate inverse via fixed-point iteration."""
        x = y.clone()
        for _ in range(n_steps):
            x = y - self.alpha * self._residual(x)
        return x
