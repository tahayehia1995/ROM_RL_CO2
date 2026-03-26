"""
S5 (Simplified State Space) MIMO transition model for E2C architecture
========================================================================
Full-matrix transition with eigenvalue-based stability, inspired by
"Simplified State Space Layers for Sequence Modeling" (ICLR 2023).

Core idea:
  A_full = V @ diag(Lambda) @ V_inv
  z_next = A_full @ z  +  B * (u*dt)

where Lambda are complex diagonal eigenvalues with |Lambda| < 1 (stable),
and V is a learnable eigenvector matrix providing full cross-dimensional
coupling like the Linear model, but with guaranteed stability.

This gives the expressiveness of the full (96x96) A matrix from the
Linear model combined with the stability guarantee of CLRU.

Reference: S5 (ICLR 2023), github.com/lindermanlab/S5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _real_to_complex(z_real, dim):
    return torch.complex(z_real[..., :dim], z_real[..., dim:])


def _complex_to_real(z_complex):
    return torch.cat([z_complex.real, z_complex.imag], dim=-1)


# ===================================================================
# Standard S5
# ===================================================================

class S5TransitionModel(nn.Module):
    """Full-matrix transition via eigendecomposition with stability."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.complex_dim = self.latent_dim // 2
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])

        selector_input = self.latent_dim + self.u_dim + 1
        hz_dim = self.latent_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        d = self.complex_dim

        # State-dependent eigenvalues
        self.alpha_layer = nn.Linear(hz_dim, d)
        self.alpha_layer.apply(weights_init)
        self.omega_layer = nn.Linear(hz_dim, d)
        self.omega_layer.apply(weights_init)

        # Learnable eigenvector matrix V (fixed, not state-dependent)
        self.V_real = nn.Parameter(torch.randn(d, d) * 0.01)
        self.V_imag = nn.Parameter(torch.randn(d, d) * 0.01)

        self.Bt_real_layer = nn.Linear(hz_dim, d * self.u_dim)
        self.Bt_real_layer.apply(weights_init)
        self.Bt_imag_layer = nn.Linear(hz_dim, d * self.u_dim)
        self.Bt_imag_layer.apply(weights_init)

        self.Ct_real_layer = nn.Linear(hz_dim, self.n_y * d)
        self.Ct_real_layer.apply(weights_init)
        self.Ct_imag_layer = nn.Linear(hz_dim, self.n_y * d)
        self.Ct_imag_layer.apply(weights_init)

        self.Dt_layer = nn.Linear(hz_dim, self.n_y * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def _get_V(self):
        """Build V and its inverse."""
        V = torch.complex(self.V_real, self.V_imag)
        V_inv = torch.linalg.inv(V)
        return V, V_inv

    def _step(self, z_complex, hz, ut, dt, batch):
        d = self.complex_dim

        alpha = -F.softplus(self.alpha_layer(hz))
        omega = self.omega_layer(hz)
        Lambda = torch.exp(torch.complex(alpha, omega))

        V, V_inv = self._get_V()

        # A_full @ z = V @ diag(Lambda) @ V_inv @ z  (per-sample Lambda)
        z_eig = torch.mm(z_complex, V_inv.T)
        z_eig = Lambda * z_eig
        z_rotated = torch.mm(z_eig, V.T)

        Bt = torch.complex(
            self.Bt_real_layer(hz).view(batch, d, self.u_dim),
            self.Bt_imag_layer(hz).view(batch, d, self.u_dim))
        ut_dt = (ut * dt).to(torch.cfloat)

        z_next = z_rotated + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)

        Ct = torch.complex(
            self.Ct_real_layer(hz).view(batch, self.n_y, d),
            self.Ct_imag_layer(hz).view(batch, self.n_y, d))
        Dt = self.Dt_layer(hz).view(batch, self.n_y, self.u_dim)

        yt = (torch.bmm(Ct, z_next.unsqueeze(-1)).squeeze(-1).real
              + torch.bmm(Dt, (ut * dt).unsqueeze(-1)).squeeze(-1))

        return z_next, yt

    def forward(self, zt, dt, ut):
        batch = zt.shape[0]
        z_complex = _real_to_complex(zt, self.complex_dim)
        hz = self.selector(torch.cat([zt, ut, dt], dim=-1))
        z_next, yt = self._step(z_complex, hz, ut, dt, batch)
        return _complex_to_real(z_next), yt

    def forward_nsteps(self, zt, dt, U):
        z_complex = _real_to_complex(zt, self.complex_dim)
        Zt_k, Yt_k = [], []
        for ut in U:
            batch = z_complex.shape[0]
            zt_real = _complex_to_real(z_complex)
            hz = self.selector(torch.cat([zt_real, ut, dt], dim=-1))
            z_complex, yt = self._step(z_complex, hz, ut, dt, batch)
            Zt_k.append(_complex_to_real(z_complex))
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned S5
# ===================================================================

class ConditionedS5Transition(nn.Module):
    """S5 variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.complex_dim = dynamic_dim // 2
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

        d = self.complex_dim

        self.alpha_layer = nn.Linear(hz_dim, d)
        self.alpha_layer.apply(weights_init)
        self.omega_layer = nn.Linear(hz_dim, d)
        self.omega_layer.apply(weights_init)

        self.V_real = nn.Parameter(torch.randn(d, d) * 0.01)
        self.V_imag = nn.Parameter(torch.randn(d, d) * 0.01)

        self.Bt_real_layer = nn.Linear(hz_dim, d * self.u_dim)
        self.Bt_real_layer.apply(weights_init)
        self.Bt_imag_layer = nn.Linear(hz_dim, d * self.u_dim)
        self.Bt_imag_layer.apply(weights_init)

        self.Ct_real_layer = nn.Linear(hz_dim, self.n_obs * d)
        self.Ct_real_layer.apply(weights_init)
        self.Ct_imag_layer = nn.Linear(hz_dim, self.n_obs * d)
        self.Ct_imag_layer.apply(weights_init)

        self.Dt_layer = nn.Linear(hz_dim, self.n_obs * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def _get_V(self):
        V = torch.complex(self.V_real, self.V_imag)
        V_inv = torch.linalg.inv(V)
        return V, V_inv

    def _step(self, z_complex, hz, ut, dt, batch):
        d = self.complex_dim

        alpha = -F.softplus(self.alpha_layer(hz))
        omega = self.omega_layer(hz)
        Lambda = torch.exp(torch.complex(alpha, omega))

        V, V_inv = self._get_V()

        z_eig = torch.mm(z_complex, V_inv.T)
        z_eig = Lambda * z_eig
        z_rotated = torch.mm(z_eig, V.T)

        Bt = torch.complex(
            self.Bt_real_layer(hz).view(batch, d, self.u_dim),
            self.Bt_imag_layer(hz).view(batch, d, self.u_dim))
        ut_dt = (ut * dt).to(torch.cfloat)

        z_next = z_rotated + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)

        Ct = torch.complex(
            self.Ct_real_layer(hz).view(batch, self.n_obs, d),
            self.Ct_imag_layer(hz).view(batch, self.n_obs, d))
        Dt = self.Dt_layer(hz).view(batch, self.n_obs, self.u_dim)

        yt = (torch.bmm(Ct, z_next.unsqueeze(-1)).squeeze(-1).real
              + torch.bmm(Dt, (ut * dt).unsqueeze(-1)).squeeze(-1))

        return z_next, yt

    def forward(self, z_dyn, z_static, dt, ut):
        batch = z_dyn.shape[0]
        z_complex = _real_to_complex(z_dyn, self.complex_dim)
        hz = self.selector(torch.cat([z_dyn, z_static, ut, dt], dim=-1))
        z_next, yt = self._step(z_complex, hz, ut, dt, batch)
        return _complex_to_real(z_next), yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        z_complex = _real_to_complex(z_dyn, self.complex_dim)
        Zt_k, Yt_k = [], []
        for ut in U:
            batch = z_complex.shape[0]
            z_dyn_real = _complex_to_real(z_complex)
            hz = self.selector(torch.cat([z_dyn_real, z_static, ut, dt], dim=-1))
            z_complex, yt = self._step(z_complex, hz, ut, dt, batch)
            Zt_k.append(_complex_to_real(z_complex))
            Yt_k.append(yt)
        return Zt_k, Yt_k
