"""
GNN Encoders for the Hybrid GNN E2C
====================================
Single GNN branch (the dynamic encoder) plus its support modules.  The
static branch is now a 3D CNN built directly inside ``GNNE2C`` (see
``gnn_e2c.py``) and no longer lives here.

- ``GNNDynamicEncoder``: per-active-node features
  ``[SG, PRES, PERMI, POROS, is_inj, is_prod, well_one_hot, control]``
  -> GNN message passing -> GridPooling -> ``z_dynamic``.
- ``GridPooling``: scatter GNN node embeddings back to the 3D grid,
  apply Conv3D + flatten + Linear, preserving spatial structure for the
  shared CNN decoder.
- ``WellReadout``: auxiliary observation predictor pooling per-well
  perforation embeddings.  Vectorised batched implementation.
"""

import torch
import torch.nn as nn

from .gnn_layers import GNNStack


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
    """Predict well observations from per-well pooled GNN embeddings.

    Vectorised: takes a single tensor with all wells stacked along an
    inner axis instead of a Python list.  The two underlying MLPs (one
    per well type) are still applied per group, but each group is
    processed in a single batched matmul.

    Output ordering matches the y-observation tensor:
        [BHP_inj0, BHP_inj1, ...,
         GasRate_prod0, GasRate_prod1, ...,
         WaterRate_prod0, WaterRate_prod1, ...]
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

    def forward(self, well_pooled):
        """
        Args:
            well_pooled: (B, num_wells, hidden_dim) where the wells are
                         ordered injectors-first (sorted) then producers
                         (sorted), matching the order the GNN encoder
                         uses for its precomputed well-readout buffers.

        Returns:
            y_aux: (B, n_obs) where ``n_obs == num_inj + 2 * num_prod``.
        """
        if well_pooled.dim() != 3:
            raise ValueError(
                f"WellReadout expected (B, num_wells, hidden_dim); "
                f"got shape {tuple(well_pooled.shape)}"
            )
        inj = well_pooled[:, :self.num_inj, :]            # (B, num_inj, H)
        prod = well_pooled[:, self.num_inj:, :]           # (B, num_prod, H)

        inj_pred = self.inj_mlp(inj).squeeze(-1)          # (B, num_inj)
        prod_pred = self.prod_mlp(prod)                   # (B, num_prod, 2)

        gas_rates = prod_pred[..., 0]                     # (B, num_prod)
        wat_rates = prod_pred[..., 1]                     # (B, num_prod)
        return torch.cat([inj_pred, gas_rates, wat_rates], dim=-1)


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
