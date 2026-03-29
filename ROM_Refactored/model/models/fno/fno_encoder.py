"""
FNO Encoder
============
Maps 3D spatial fields to a latent vector via spectral convolution layers
with Fourier-domain truncation for dimensionality reduction.

Architecture:
    Input (B, C_in, Nx, Ny, Nz)
    -> Lifting: Conv3d to spectral width
    -> N x InvertibleSpectralBlock (width -> width)
    -> Projection: Conv3d to proj_channels
    -> Fourier truncation: keep spatial_modes in each dim
    -> Flatten + Linear -> z (latent_dim)
"""

import torch
import torch.nn as nn
import math

from .spectral_layers import InvertibleSpectralBlock


class FNOEncoder(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int, config: dict):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        enc_cfg = config.get('fno', {}).get('encoder', {})
        trunc_cfg = config.get('fno', {}).get('latent_truncation', {})

        self.width = enc_cfg.get('width', 32)
        n_layers = enc_cfg.get('n_layers', 4)
        modes = tuple(enc_cfg.get('n_modes', [12, 8, 10]))
        activation = enc_cfg.get('activation', 'gelu')
        alpha = enc_cfg.get('residual_alpha', 0.1)

        self.trunc_modes = tuple(trunc_cfg.get('spatial_modes', [4, 4, 4]))
        self.proj_channels = trunc_cfg.get('channels', 16)

        self.lifting = nn.Sequential(
            nn.Conv3d(in_channels, self.width, kernel_size=1),
            nn.GELU(),
        )

        self.spectral_blocks = nn.ModuleList([
            InvertibleSpectralBlock(self.width, modes, alpha=alpha,
                                   activation=activation)
            for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(
            nn.Conv3d(self.width, self.proj_channels, kernel_size=1),
            nn.GELU(),
        )

        self._flat_size = self.proj_channels * self.trunc_modes[0] * self.trunc_modes[1] * self.trunc_modes[2]
        self.fc = nn.Linear(self._flat_size, latent_dim)

        self.enable_vae = config.get('model', {}).get('enable_vae', False)
        if self.enable_vae:
            self.fc_logvar = nn.Linear(self._flat_size, latent_dim)

    def _fourier_truncate(self, x: torch.Tensor) -> torch.Tensor:
        """Truncate spatial dims in Fourier domain to retain only low-freq modes."""
        m1, m2, m3 = self.trunc_modes
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        x_trunc_ft = x_ft[:, :, :m1, :m2, :m3]
        return torch.fft.irfftn(x_trunc_ft, s=(m1, m2, m3), dim=(-3, -2, -1))

    def forward(self, x: torch.Tensor):
        """Encode spatial field to latent vector.

        Returns:
            (z_mean, mu, logvar) — mu/logvar are None when VAE is disabled.
        """
        h = self.lifting(x)

        for block in self.spectral_blocks:
            h = block(h)

        h = self.projection(h)
        h = self._fourier_truncate(h)
        h = h.reshape(h.shape[0], -1)

        z_mean = self.fc(h)

        if self.enable_vae:
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = z_mean + eps * std
            return z, z_mean, logvar

        return z_mean, None, None
