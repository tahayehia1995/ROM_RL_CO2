"""
FNO Decoder
============
Maps a latent vector back to 3D spatial fields via inverse spectral
convolution blocks, designed as an approximate inverse of FNOEncoder.

Architecture:
    Input z (B, latent_dim)
    -> Linear + unflatten to (B, proj_channels, m1, m2, m3)
    -> Fourier padding: zero-pad spectrum back to full spatial dims
    -> N x InvertibleSpectralBlock (inverse-mode, width -> width)
    -> Final Conv3d to output channels
"""

import torch
import torch.nn as nn

from .spectral_layers import InvertibleSpectralBlock


class FNODecoder(nn.Module):

    def __init__(self, out_channels: int, latent_dim: int,
                 spatial_shape: tuple[int, int, int], config: dict):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.spatial_shape = spatial_shape  # (Nx, Ny, Nz)

        dec_cfg = config.get('fno', {}).get('decoder', {})
        trunc_cfg = config.get('fno', {}).get('latent_truncation', {})
        enc_cfg = config.get('fno', {}).get('encoder', {})

        self.width = dec_cfg.get('width', enc_cfg.get('width', 32))
        n_layers = dec_cfg.get('n_layers', enc_cfg.get('n_layers', 4))
        modes = tuple(dec_cfg.get('n_modes', enc_cfg.get('n_modes', [12, 8, 10])))
        activation = enc_cfg.get('activation', 'gelu')
        alpha = enc_cfg.get('residual_alpha', 0.1)
        norm_type = dec_cfg.get('norm_type', enc_cfg.get('norm_type', 'batchnorm'))

        self.trunc_modes = tuple(trunc_cfg.get('spatial_modes', [4, 4, 4]))
        self.proj_channels = trunc_cfg.get('channels', 16)

        self._flat_size = self.proj_channels * self.trunc_modes[0] * self.trunc_modes[1] * self.trunc_modes[2]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self._flat_size),
            nn.GELU(),
        )

        unproject_layers = [nn.Conv3d(self.proj_channels, self.width, kernel_size=1)]
        if norm_type == 'batchnorm':
            unproject_layers.append(nn.BatchNorm3d(self.width))
        elif norm_type == 'instancenorm':
            unproject_layers.append(nn.InstanceNorm3d(self.width, affine=True))
        unproject_layers.append(nn.GELU())
        self.unproject = nn.Sequential(*unproject_layers)

        self.spectral_blocks = nn.ModuleList([
            InvertibleSpectralBlock(self.width, modes, alpha=alpha,
                                   activation=activation, norm=norm_type)
            for _ in range(n_layers)
        ])

        self.final_conv = nn.Conv3d(self.width, out_channels, kernel_size=1)

    def _fourier_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad from truncated spatial shape back to full dims via zero-padded spectrum."""
        Nx, Ny, Nz = self.spatial_shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            x.shape[0], x.shape[1], Nx, Ny, Nz // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        m1, m2, m3 = x_ft.shape[-3], x_ft.shape[-2], x_ft.shape[-1]
        out_ft[:, :, :m1, :m2, :m3] = x_ft

        return torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), dim=(-3, -2, -1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m1, m2, m3 = self.trunc_modes
        h = self.fc(z)
        h = h.view(h.shape[0], self.proj_channels, m1, m2, m3)

        h = self._fourier_pad(h)
        h = self.unproject(h)

        for block in self.spectral_blocks:
            h = block(h)

        return self.final_conv(h)
