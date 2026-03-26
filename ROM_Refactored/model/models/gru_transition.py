"""
GRU transition model for E2C architecture
==========================================
Uses PyTorch GRUCell as the transition function.  The gating mechanism
(reset + update gates) provides built-in selective memory, making it
naturally resistant to vanishing gradients.

State equation:
    z_{t+1} = GRUCell(u_t * dt, z_t)

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

Reference: Cho et al. (2014), PyTorch GRUCell.
"""

import math
import torch
import torch.nn as nn
from model.utils.initialization import weights_init


# ===================================================================
# Standard GRU
# ===================================================================

class GRUTransitionModel(nn.Module):
    """GRU-based transition with linear observation head."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        gru_cfg = config['transition'].get('gru', {})
        self.num_layers = gru_cfg.get('num_layers', 1)

        self.gru_cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.u_dim if i == 0 else self.latent_dim
            self.gru_cells.append(nn.GRUCell(input_size, self.latent_dim))

        d = self.latent_dim
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        h = zt
        x = ut_dt
        for cell in self.gru_cells:
            h = cell(x, h)
            x = h
        zt1 = h
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned GRU
# ===================================================================

class ConditionedGRUTransition(nn.Module):
    """GRU variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        gru_cfg = config['transition'].get('gru', {})
        self.num_layers = gru_cfg.get('num_layers', 1)

        gru_input = self.u_dim + static_dim
        self.gru_cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = gru_input if i == 0 else dynamic_dim
            self.gru_cells.append(nn.GRUCell(input_size, dynamic_dim))

        d = dynamic_dim
        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        x = torch.cat([ut_dt, z_static], dim=-1)
        h = z_dyn
        for cell in self.gru_cells:
            h = cell(x, h)
            x = h
        z_dyn_next = h
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
