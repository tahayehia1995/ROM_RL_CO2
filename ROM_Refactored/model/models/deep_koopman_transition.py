"""
Deep Koopman (State-Dependent Operator) transition model for E2C
================================================================
Instead of a fixed global K, a neural network generates K from the
current state — bridging Koopman and SSM architectures.  Observation
matrices C, D remain global (like standard Koopman).

State equation:
    K_t, L_t = f_theta(z_t, dt)
    z_{t+1} = K_t z_t + L_t (u_t * dt)

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

Reference: "Temporally consistent Koopman autoencoders" (2025),
           dlkoopman library.
"""

import math
import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


def _build_selector(input_dim, output_dim, hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [256, 256]
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(fc_bn_relu(prev, h))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ===================================================================
# Standard Deep Koopman
# ===================================================================

class DeepKoopmanTransitionModel(nn.Module):
    """State-dependent Koopman operator with global observation head."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        dk_cfg = config['transition'].get('deep_koopman', {})
        encoder_hidden = dk_cfg.get('encoder_hidden_dims',
                                    config['transition'].get('encoder_hidden_dims', [256, 256]))

        d = self.latent_dim
        selector_input = d + 1
        hz_dim = d

        self.selector = _build_selector(selector_input, hz_dim, encoder_hidden)
        self.selector.apply(weights_init)

        self.Kt_layer = nn.Linear(hz_dim, d * d)
        self.Kt_layer.apply(weights_init)
        self.Lt_layer = nn.Linear(hz_dim, d * self.u_dim)
        self.Lt_layer.apply(weights_init)

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, zt, dt, ut):
        hz = self.selector(torch.cat([zt, dt], dim=-1))

        Kt = self.Kt_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Lt = self.Lt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        ut_dt = ut * dt
        zt1 = (torch.bmm(Kt, zt.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Lt, ut_dt.unsqueeze(-1)).squeeze(-1))
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
# Conditioned Deep Koopman
# ===================================================================

class ConditionedDeepKoopmanTransition(nn.Module):
    """Deep Koopman variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        dk_cfg = config['transition'].get('deep_koopman', {})
        encoder_hidden = dk_cfg.get('encoder_hidden_dims',
                                    config['transition'].get('encoder_hidden_dims', [256, 256]))

        selector_input = dynamic_dim + static_dim + 1
        hz_dim = dynamic_dim + static_dim
        self.selector = _build_selector(selector_input, hz_dim, encoder_hidden)
        self.selector.apply(weights_init)

        self.Kt_layer = nn.Linear(hz_dim, dynamic_dim * dynamic_dim)
        self.Kt_layer.apply(weights_init)
        self.Lt_layer = nn.Linear(hz_dim, dynamic_dim * self.u_dim)
        self.Lt_layer.apply(weights_init)

        self.C = nn.Parameter(
            torch.randn(self.n_obs, dynamic_dim) * (1.0 / math.sqrt(dynamic_dim))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, z_dyn, z_static, dt, ut):
        hz = self.selector(torch.cat([z_dyn, z_static, dt], dim=-1))

        Kt = self.Kt_layer(hz).view(-1, self.dynamic_dim, self.dynamic_dim)
        Lt = self.Lt_layer(hz).view(-1, self.dynamic_dim, self.u_dim)

        ut_dt = ut * dt
        z_dyn_next = (torch.bmm(Kt, z_dyn.unsqueeze(-1)).squeeze(-1)
                      + torch.bmm(Lt, ut_dt.unsqueeze(-1)).squeeze(-1))
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)

        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
