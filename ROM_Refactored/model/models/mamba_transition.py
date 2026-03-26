"""
Mamba (Selective SSM) transition model for E2C architecture
============================================================
Input-dependent B, C, and discretization step Delta.  Diagonal A with
simplified ZOH discretization.  The model selectively gates which
information to propagate vs forget based on current state and control.

State equation:
    Delta_t, B_t, C_t = selector(z_t, u_t, dt)
    A_bar = exp(Delta_t * A_diag)
    B_bar = Delta_t * B_t
    z_{t+1} = A_bar * z_t + B_bar * (u_t * dt)

Observation equation:
    y_{t+1} = C_t * z_{t+1} + D * (u_t * dt)

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
           State Spaces", 2023.  johnma2006/mamba-minimal (pure PyTorch).
"""

import math
import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


def _build_selector(input_dim, output_dim, hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [128, 128]
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(fc_bn_relu(prev, h))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ===================================================================
# Standard Mamba
# ===================================================================

class MambaTransitionModel(nn.Module):
    """Selective SSM transition (Mamba-style) for E2C."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        mamba_cfg = config['transition'].get('mamba', {})
        self.dt_rank = mamba_cfg.get('dt_rank', max(1, self.latent_dim // 16))

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])
        selector_input = self.latent_dim + self.u_dim + 1
        hz_dim = self.latent_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.A_log = nn.Parameter(torch.log(torch.rand(self.latent_dim) * 0.5 + 0.5))

        self.delta_proj = nn.Linear(hz_dim, self.latent_dim)
        self.delta_proj.apply(weights_init)
        self.Bt_layer = nn.Linear(hz_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(hz_dim, self.n_y * self.latent_dim)
        self.Ct_layer.apply(weights_init)
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, zt, dt, ut):
        hz = self.selector(torch.cat([zt, ut, dt], dim=-1))

        delta = nn.functional.softplus(self.delta_proj(hz))
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta * A.unsqueeze(0))
        B_bar = delta

        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.n_y, self.latent_dim)

        ut_dt = ut * dt
        zt1 = A_bar * zt + B_bar * torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt1 = (torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1)
               + torch.mm(ut_dt, self.D.T))

        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned Mamba
# ===================================================================

class ConditionedMambaTransition(nn.Module):
    """Mamba variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])
        selector_input = dynamic_dim + static_dim + self.u_dim + 1
        hz_dim = dynamic_dim + static_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.A_log = nn.Parameter(torch.log(torch.rand(dynamic_dim) * 0.5 + 0.5))

        self.delta_proj = nn.Linear(hz_dim, dynamic_dim)
        self.delta_proj.apply(weights_init)
        self.Bt_layer = nn.Linear(hz_dim, dynamic_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(hz_dim, self.n_obs * dynamic_dim)
        self.Ct_layer.apply(weights_init)
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, z_dyn, z_static, dt, ut):
        hz = self.selector(torch.cat([z_dyn, z_static, ut, dt], dim=-1))

        delta = nn.functional.softplus(self.delta_proj(hz))
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta * A.unsqueeze(0))
        B_bar = delta

        Bt = self.Bt_layer(hz).view(-1, self.dynamic_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.n_obs, self.dynamic_dim)

        ut_dt = ut * dt
        z_dyn_next = A_bar * z_dyn + B_bar * torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt = (torch.bmm(Ct, z_dyn_next.unsqueeze(-1)).squeeze(-1)
              + torch.mm(ut_dt, self.D.T))

        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
