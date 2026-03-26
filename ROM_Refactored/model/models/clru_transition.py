"""
Control-LRU (CLRU) transition model for E2C architecture
=========================================================
Diagonal recurrence with input-dependent dynamics, stable by construction.

Key differences from LinearTransitionModel:
  - A matrix is diagonal (Lambda), not full d x d.
  - Lambda is parameterised via exp(-exp(nu)) so every eigenvalue lies in (0, 1).
  - The selector (conditioning network) receives [z_t, u_t, dt], not just [z_t, dt],
    making the dynamics implicitly nonlinear in control.
  - Multi-step rollout recomputes dynamics at every step instead of fixing A/B/C/D
    once from the initial state.

Reference: Control-LRU-Approach/1.CCS_E2CO-RL_Base_Fixed/MSE2C.py
"""

import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


# ---------------------------------------------------------------------------
# Selector builder (analogous to create_trans_encoder but for CLRU)
# ---------------------------------------------------------------------------

def _build_selector(input_dim: int, output_dim: int, hidden_dims=None):
    """
    Build the CLRU selector network: MLP that maps
    [z_t, u_t, dt] -> conditioning vector h_z of size output_dim.

    Args:
        input_dim:   latent_dim + u_dim + 1
        output_dim:  latent_dim (size of h_z)
        hidden_dims: list of hidden-layer widths; defaults to [128, 128]
    """
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
# Standard CLRU (drop-in replacement for LinearTransitionModel)
# ===================================================================

class CLRUTransitionModel(nn.Module):
    """Control-LRU transition with diagonal, guaranteed-stable recurrence."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        clru_cfg = config['transition'].get('clru', {})
        self.nu_clamp_max = clru_cfg.get('nu_clamp_max', 2.0)

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])

        selector_input_dim = self.latent_dim + self.u_dim + 1
        self.selector = _build_selector(selector_input_dim, self.latent_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.nu_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.nu_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(self.latent_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(self.latent_dim, self.n_y * self.latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(self.latent_dim, self.n_y * self.u_dim)
        self.Dt_layer.apply(weights_init)

    # ---- single step (same signature as LinearTransitionModel) ----

    def forward(self, zt, dt, ut):
        hz = self.selector(torch.cat([zt, ut, dt], dim=-1))

        nu = torch.clamp(self.nu_layer(hz), max=self.nu_clamp_max)
        Lambda = torch.exp(-torch.exp(nu))

        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.n_y, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.n_y, self.u_dim)

        ut_dt = ut * dt
        zt1 = Lambda * zt + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt1 = (torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))

        return zt1, yt1

    # ---- multi-step (recomputes dynamics each step) ----

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned CLRU (for GNN / multimodal dual-branch models)
# ===================================================================

class ConditionedCLRUTransition(nn.Module):
    """
    CLRU variant that operates on split latent spaces (z_dynamic, z_static),
    matching the interface of _ConditionedLinearTransition used by GNNE2C.
    Lambda is applied only to z_dynamic; z_static conditions the selector.
    """

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        clru_cfg = config['transition'].get('clru', {})
        self.nu_clamp_max = clru_cfg.get('nu_clamp_max', 2.0)

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])

        selector_input_dim = dynamic_dim + static_dim + self.u_dim + 1
        hz_dim = dynamic_dim + static_dim
        self.selector = _build_selector(selector_input_dim, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.nu_layer = nn.Linear(hz_dim, dynamic_dim)
        self.nu_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(hz_dim, dynamic_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(hz_dim, self.n_obs * dynamic_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(hz_dim, self.n_obs * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def forward(self, z_dyn, z_static, dt, ut):
        hz = self.selector(torch.cat([z_dyn, z_static, ut, dt], dim=-1))

        nu = torch.clamp(self.nu_layer(hz), max=self.nu_clamp_max)
        Lambda = torch.exp(-torch.exp(nu))

        Bt = self.Bt_layer(hz).view(-1, self.dynamic_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.n_obs, self.dynamic_dim)
        Dt = self.Dt_layer(hz).view(-1, self.n_obs, self.u_dim)

        ut_dt = ut * dt
        z_dyn_next = Lambda * z_dyn + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt = (torch.bmm(Ct, z_dyn_next.unsqueeze(-1)).squeeze(-1)
              + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))

        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
