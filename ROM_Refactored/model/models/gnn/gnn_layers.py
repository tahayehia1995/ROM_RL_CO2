"""
GNN Building Blocks
====================
Reusable layers for the GNN encoder and decoder:
- GNNBlock: single message-passing layer + norm + activation + residual
- GNNStack: sequential stack of GNNBlocks
- AttentionPooling: learnable attention-weighted global pooling
- NodeBroadcast: broadcasts a global latent vector to per-node features

Supports two convolution backends (``conv_type``):
- ``'gine'``  (default): GINEConv — fast, edge-feature-aware aggregation
  without per-edge attention overhead.  Transmissibility in edge_attr
  acts as a physics-informed communication weight.
- ``'gatv2'``: GATv2Conv — learned per-edge attention, more expressive
  but significantly slower on large graphs (~10K nodes / ~55K edges).

All layers operate on flat PyG-style tensors (total_nodes, features)
and support batched (mini-batch) mode via a ``batch`` vector.
"""

import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv, GINEConv


class GNNBlock(nn.Module):
    """
    Single message-passing layer with layer normalization,
    ELU activation, dropout, and optional residual connection.

    ``conv_type='gine'`` uses GINEConv (fast, edge-feature-aware).
    ``conv_type='gatv2'`` uses GATv2Conv (learned per-edge attention).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 2,
        edge_dim: int = 7,
        dropout: float = 0.1,
        residual: bool = True,
        concat_heads: bool = True,
        conv_type: str = 'gine',
    ):
        super().__init__()
        self.residual = residual

        if conv_type == 'gatv2':
            if concat_heads:
                assert out_channels % heads == 0, \
                    f"out_channels ({out_channels}) must be divisible by heads ({heads})"
                head_dim = out_channels // heads
            else:
                head_dim = out_channels

            self.conv = GATv2Conv(
                in_channels,
                head_dim,
                heads=heads,
                concat=concat_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=True,
                share_weights=True,
            )
            actual_out = head_dim * heads if concat_heads else head_dim
        else:
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(out_channels, out_channels),
            )
            self.conv = GINEConv(nn=mlp, edge_dim=edge_dim)
            actual_out = out_channels

        self.norm = nn.LayerNorm(actual_out)
        self.act = nn.ELU(inplace=True)

        if residual and in_channels != actual_out:
            self.skip = nn.Linear(in_channels, actual_out, bias=False)
        else:
            self.skip = None

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.norm(h)

        if self.residual:
            skip = self.skip(x) if self.skip is not None else x
            h = h + skip

        return self.act(h)


class GNNStack(nn.Module):
    """Sequential stack of GNNBlock layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 2,
        edge_dim: int = 7,
        dropout: float = 0.1,
        conv_type: str = 'gine',
    ):
        super().__init__()
        layers = []

        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            outc = out_channels if i == num_layers - 1 else hidden_channels

            layers.append(GNNBlock(
                inc, outc,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                residual=(i > 0),
                conv_type=conv_type,
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
        return x


class AttentionPooling(nn.Module):
    """
    Learnable attention-weighted global pooling.

    Can operate in two modes:
    - Single graph: x is (N, C), returns (C_out,) — no batch vector needed.
    - Batched: x is (total_nodes, C), batch is (total_nodes,), returns (B, C_out).

    GNNE2C calls gate_nn and proj directly for batched pooling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // 4, 8)),
            nn.ELU(inplace=True),
            nn.Linear(max(in_channels // 4, 8), 1),
        )
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """Single-graph pooling: (N, C) -> (C_out,)"""
        gate = torch.sigmoid(self.gate_nn(x))
        weighted = (gate * x).sum(dim=0)
        gate_sum = gate.sum().clamp(min=1e-8)
        pooled = weighted / gate_sum
        return self.proj(pooled)


class NodeBroadcast(nn.Module):
    """
    Broadcasts global latent vectors to per-node features.

    Uses a deep MLP with LayerNorm to map (latent, position) -> per-node
    hidden features.  A wider / deeper MLP here directly improves the
    decoder's ability to reconstruct fine spatial detail because every
    node's initial feature vector is produced by this network.

    Batched mode: z is (B, latent_dim), batch_vec maps nodes to samples.
    """

    def __init__(
        self,
        latent_dim: int,
        pos_dim: int,
        out_channels: int,
        broadcast_hidden_dim: int = 256,
        broadcast_num_layers: int = 4,
    ):
        super().__init__()
        in_dim = latent_dim + pos_dim
        layers = []

        for i in range(broadcast_num_layers):
            inp = in_dim if i == 0 else broadcast_hidden_dim
            out = out_channels if i == broadcast_num_layers - 1 else broadcast_hidden_dim
            layers.append(nn.Linear(inp, out))
            layers.append(nn.LayerNorm(out))
            layers.append(nn.ELU(inplace=True))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z, positions, batch_vec):
        """
        Args:
            z: (B, latent_dim)
            positions: (total_nodes, pos_dim)
            batch_vec: (total_nodes,) graph assignment
        Returns:
            (total_nodes, out_channels)
        """
        z_expanded = z[batch_vec]
        combined = torch.cat([z_expanded, positions], dim=-1)
        return self.mlp(combined)
