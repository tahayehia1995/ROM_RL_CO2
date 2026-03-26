"""
Mamba-2 (SSD) transition model for E2C architecture
=====================================================
Multi-head selective SSM using the State Space Duality framework.
Each head has its own diagonal A, while B/C are shared across groups.

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and
           Efficient Algorithms Through Structured State Space Duality", 2024.
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
# Standard Mamba-2
# ===================================================================

class Mamba2TransitionModel(nn.Module):
    """Multi-head Selective SSM (Mamba-2 / SSD) for E2C."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        m2_cfg = config['transition'].get('mamba2', {})
        self.n_heads = m2_cfg.get('n_heads', 4)
        self.head_dim = self.latent_dim // self.n_heads
        assert self.latent_dim % self.n_heads == 0, \
            f"latent_dim ({self.latent_dim}) must be divisible by n_heads ({self.n_heads})"

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])
        selector_input = self.latent_dim + self.u_dim + 1
        hz_dim = self.latent_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.A_log = nn.Parameter(torch.log(torch.rand(self.n_heads, self.head_dim) * 0.5 + 0.5))

        self.delta_proj = nn.Linear(hz_dim, self.latent_dim)
        self.delta_proj.apply(weights_init)
        self.Bt_layer = nn.Linear(hz_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(hz_dim, self.n_y * self.latent_dim)
        self.Ct_layer.apply(weights_init)

        self.out_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.out_proj.apply(weights_init)

        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, zt, dt, ut):
        B = zt.size(0)
        hz = self.selector(torch.cat([zt, ut, dt], dim=-1))

        delta = nn.functional.softplus(self.delta_proj(hz))
        delta = delta.view(B, self.n_heads, self.head_dim)

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta * A.unsqueeze(0))
        A_bar = A_bar.view(B, self.latent_dim)

        B_bar = delta.view(B, self.latent_dim)

        Bt = self.Bt_layer(hz).view(B, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(B, self.n_y, self.latent_dim)

        ut_dt = ut * dt
        zt_raw = A_bar * zt + B_bar * torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        zt1 = self.out_proj(zt_raw)

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
# Conditioned Mamba-2
# ===================================================================

class ConditionedMamba2Transition(nn.Module):
    """Mamba-2 variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        m2_cfg = config['transition'].get('mamba2', {})
        self.n_heads = m2_cfg.get('n_heads', 4)
        n_heads_actual = self.n_heads
        while dynamic_dim % n_heads_actual != 0 and n_heads_actual > 1:
            n_heads_actual -= 1
        self.n_heads = n_heads_actual
        self.head_dim = dynamic_dim // self.n_heads

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])
        selector_input = dynamic_dim + static_dim + self.u_dim + 1
        hz_dim = dynamic_dim + static_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.A_log = nn.Parameter(torch.log(torch.rand(self.n_heads, self.head_dim) * 0.5 + 0.5))

        self.delta_proj = nn.Linear(hz_dim, dynamic_dim)
        self.delta_proj.apply(weights_init)
        self.Bt_layer = nn.Linear(hz_dim, dynamic_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(hz_dim, self.n_obs * dynamic_dim)
        self.Ct_layer.apply(weights_init)

        self.out_proj = nn.Linear(dynamic_dim, dynamic_dim)
        self.out_proj.apply(weights_init)

        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, z_dyn, z_static, dt, ut):
        B = z_dyn.size(0)
        hz = self.selector(torch.cat([z_dyn, z_static, ut, dt], dim=-1))

        delta = nn.functional.softplus(self.delta_proj(hz))
        delta = delta.view(B, self.n_heads, self.head_dim)

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta * A.unsqueeze(0)).view(B, self.dynamic_dim)
        B_bar = delta.view(B, self.dynamic_dim)

        Bt = self.Bt_layer(hz).view(B, self.dynamic_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(B, self.n_obs, self.dynamic_dim)

        ut_dt = ut * dt
        z_raw = A_bar * z_dyn + B_bar * torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        z_dyn_next = self.out_proj(z_raw)

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
