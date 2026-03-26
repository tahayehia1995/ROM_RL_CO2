"""
LSTM transition model for E2C architecture
============================================
Uses PyTorch LSTMCell as the transition function.  The cell state provides
long-term memory while the hidden state acts as the latent state z.

State equation:
    h_{t+1}, c_{t+1} = LSTMCell(u_t * dt, (h_t, c_t))
    z_{t+1} = h_{t+1}

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

Note: The cell state c is carried internally across steps in
forward_nsteps but is NOT part of the external latent state z.

Reference: Hochreiter & Schmidhuber (1997), PyTorch LSTMCell.
"""

import math
import torch
import torch.nn as nn


# ===================================================================
# Standard LSTM
# ===================================================================

class LSTMTransitionModel(nn.Module):
    """LSTM-based transition with linear observation head."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        lstm_cfg = config['transition'].get('lstm', {})
        self.num_layers = lstm_cfg.get('num_layers', 1)

        self.lstm_cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.u_dim if i == 0 else self.latent_dim
            self.lstm_cells.append(nn.LSTMCell(input_size, self.latent_dim))

        d = self.latent_dim
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._cell_states = None

    def _init_cell_states(self, batch_size, device):
        """Initialize cell states to zeros for each layer."""
        return [torch.zeros(batch_size, self.latent_dim, device=device)
                for _ in range(self.num_layers)]

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        B = zt.size(0)

        if self._cell_states is None or self._cell_states[0].size(0) != B:
            self._cell_states = self._init_cell_states(B, zt.device)

        h = zt
        x = ut_dt
        new_cells = []
        for i, cell in enumerate(self.lstm_cells):
            h, c = cell(x, (h, self._cell_states[i]))
            new_cells.append(c)
            x = h
        self._cell_states = new_cells

        zt1 = h
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        B = zt.size(0)
        self._cell_states = self._init_cell_states(B, zt.device)
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
        self._cell_states = None
        return Zt_k, Yt_k


# ===================================================================
# Conditioned LSTM
# ===================================================================

class ConditionedLSTMTransition(nn.Module):
    """LSTM variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        lstm_cfg = config['transition'].get('lstm', {})
        self.num_layers = lstm_cfg.get('num_layers', 1)

        lstm_input = self.u_dim + static_dim
        self.lstm_cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = lstm_input if i == 0 else dynamic_dim
            self.lstm_cells.append(nn.LSTMCell(input_size, dynamic_dim))

        d = dynamic_dim
        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._cell_states = None

    def _init_cell_states(self, batch_size, device):
        return [torch.zeros(batch_size, self.dynamic_dim, device=device)
                for _ in range(self.num_layers)]

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        B = z_dyn.size(0)

        if self._cell_states is None or self._cell_states[0].size(0) != B:
            self._cell_states = self._init_cell_states(B, z_dyn.device)

        x = torch.cat([ut_dt, z_static], dim=-1)
        h = z_dyn
        new_cells = []
        for i, cell in enumerate(self.lstm_cells):
            h, c = cell(x, (h, self._cell_states[i]))
            new_cells.append(c)
            x = h
        self._cell_states = new_cells

        z_dyn_next = h
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        B = z_dyn.size(0)
        self._cell_states = self._init_cell_states(B, z_dyn.device)
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        self._cell_states = None
        return Zt_k, Yt_k
