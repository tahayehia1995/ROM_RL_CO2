"""
GNN Decoder for E2C
====================
Reconstructs per-node spatial features from a graph-level latent vector.

The decoder components (broadcast, gnn, head) are called directly by
GNNE2C in batched mode for efficient grouped-by-realization processing.
"""

import torch
import torch.nn as nn

from .gnn_layers import GNNStack, NodeBroadcast


class GNNDecoder(nn.Module):
    """Decodes latent vectors into per-node feature predictions."""

    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 2,
        edge_dim: int = 7,
        dropout: float = 0.1,
        pos_dim: int = 3,
        broadcast_hidden_dim: int = 256,
        broadcast_num_layers: int = 4,
        conv_type: str = 'gine',
    ):
        super().__init__()
        self.broadcast = NodeBroadcast(
            latent_dim, pos_dim, hidden_dim,
            broadcast_hidden_dim=broadcast_hidden_dim,
            broadcast_num_layers=broadcast_num_layers,
        )
        self.gnn = GNNStack(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            conv_type=conv_type,
        )
        self.head = nn.Linear(hidden_dim, out_channels)
