"""
GNN Encoders for E2C
=====================
Two-branch encoder matching the multimodal pattern:
- GNNStaticEncoder:  PERMI, POROS, well indicators -> z_static  (32-dim)
- GNNDynamicEncoder: SG, PRES, well_indicators, control -> z_dynamic (96-dim)

The dynamic encoder uses GridPooling (scatter GNN embeddings back to the
3D grid, then Conv3D + flatten + Linear) to preserve spatial structure
for the CNN decoder and transition model.

Supports both single-graph and batched (PyG mini-batch) modes.
"""

import torch
import torch.nn as nn

from .gnn_layers import GNNStack, AttentionPooling


class GridPooling(nn.Module):
    """
    Scatter GNN node embeddings back to the 3D grid, then apply a small
    Conv3D pipeline to produce a spatially-structured latent vector.

    This preserves spatial correspondence for the CNN decoder and gives
    the transition model a structured latent space to learn dynamics in.
    """

    def __init__(self, in_channels: int, latent_dim: int, grid_shape: tuple):
        super().__init__()
        Nx, Ny, Nz = grid_shape
        self.grid_shape = grid_shape

        mid_ch = max(in_channels // 2, 16)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, Nx, Ny, Nz)
            out = self.conv(dummy)
            flat_size = out.numel()

        self.fc = nn.Linear(flat_size, latent_dim)

    def forward(self, grid_3d):
        """
        Args:
            grid_3d: (B, in_channels, Nx, Ny, Nz) — scattered GNN features
        Returns:
            (B, latent_dim)
        """
        h = self.conv(grid_3d)
        return self.fc(h.flatten(1))


class WellReadout(nn.Module):
    """
    Predict well observations directly from GNN node embeddings at
    well perforation nodes.  Each well's 25-layer node embeddings are
    average-pooled, then mapped to observation values via a small MLP.
    """

    def __init__(self, hidden_dim: int, num_inj: int, num_prod: int):
        super().__init__()
        self.num_inj = num_inj
        self.num_prod = num_prod
        self.inj_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.prod_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, h_per_well):
        """
        Args:
            h_per_well: list of (hidden_dim,) tensors, one per well.
                        Order: injectors first (sorted), then producers (sorted).
        Returns:
            y_aux: (n_obs,) — predicted observations matching y ordering:
                   [BHP_I1, BHP_I2, BHP_I4, GasRate_P1, GasRate_P2, GasRate_P3,
                    WaterRate_P1, WaterRate_P2, WaterRate_P3]
        """
        inj_preds = []
        for i in range(self.num_inj):
            inj_preds.append(self.inj_mlp(h_per_well[i]))

        prod_preds = []
        for i in range(self.num_prod):
            prod_preds.append(self.prod_mlp(h_per_well[self.num_inj + i]))

        inj_cat = torch.cat(inj_preds, dim=-1)
        gas_rates = torch.cat([p[:1] for p in prod_preds], dim=-1)
        wat_rates = torch.cat([p[1:] for p in prod_preds], dim=-1)
        return torch.cat([inj_cat, gas_rates, wat_rates], dim=-1)


class GNNStaticEncoder(nn.Module):
    """Encodes static per-node features into a graph-level latent vector."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 32,
        hidden_dim: int = 32,
        num_layers: int = 3,
        heads: int = 2,
        edge_dim: int = 7,
        dropout: float = 0.1,
        conv_type: str = 'gine',
    ):
        super().__init__()
        self.gnn = GNNStack(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            conv_type=conv_type,
        )
        self.pool = AttentionPooling(hidden_dim, latent_dim)


class GNNDynamicEncoder(nn.Module):
    """
    Encodes dynamic per-node features into a graph-level latent vector.

    Uses GridPooling: after GNN message passing, node embeddings are
    scattered back to the 3D grid and pooled via Conv3D + Linear.
    This preserves spatial structure for the CNN decoder.
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 96,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 2,
        edge_dim: int = 7,
        dropout: float = 0.1,
        conv_type: str = 'gine',
        grid_shape: tuple = (34, 16, 25),
        num_inj: int = 3,
        num_prod: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn = GNNStack(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            conv_type=conv_type,
        )
        self.grid_pool = GridPooling(hidden_dim, latent_dim, grid_shape)
        self.well_readout = WellReadout(hidden_dim, num_inj, num_prod)
