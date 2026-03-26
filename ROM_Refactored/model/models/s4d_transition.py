"""
S4D (Diagonal State-Space) transition model for E2C architecture
=================================================================
Complex diagonal recurrence inspired by "On the Parameterization and
Initialization of Diagonal State Space Models" (NeurIPS 2022).

Key improvement over CLRU:
  - Uses COMPLEX eigenvalues Lambda = exp(alpha + i*omega).
  - The imaginary part (omega) introduces OSCILLATORY dynamics that
    preserve latent information through phase encoding.
  - Real-valued CLRU loses 99%+ of initial state after 30 steps;
    complex S4D retains information via oscillation.

Dimensional convention:
  - External interface uses real tensors of size latent_dim (e.g. 128).
  - Internally, the first half encodes Re(z), the second half Im(z),
    giving latent_dim/2 complex dimensions (e.g. 64 complex).
  - This keeps full compatibility with the encoder/decoder.

Stability guarantee:
  |Lambda| = exp(alpha) < 1  since alpha = -softplus(...) < 0.
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
    """Convert a real tensor [..., 2*dim] to complex [..., dim]."""
    return torch.complex(z_real[..., :dim], z_real[..., dim:])


def _complex_to_real(z_complex):
    """Convert a complex tensor [..., dim] to real [..., 2*dim]."""
    return torch.cat([z_complex.real, z_complex.imag], dim=-1)


# ===================================================================
# Standard S4D
# ===================================================================

class S4DTransitionModel(nn.Module):
    """Complex-diagonal transition with oscillatory information preservation."""

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

        self.alpha_layer = nn.Linear(hz_dim, self.complex_dim)
        self.alpha_layer.apply(weights_init)
        self.omega_layer = nn.Linear(hz_dim, self.complex_dim)
        self.omega_layer.apply(weights_init)

        self.Bt_real_layer = nn.Linear(hz_dim, self.complex_dim * self.u_dim)
        self.Bt_real_layer.apply(weights_init)
        self.Bt_imag_layer = nn.Linear(hz_dim, self.complex_dim * self.u_dim)
        self.Bt_imag_layer.apply(weights_init)

        self.Ct_real_layer = nn.Linear(hz_dim, self.n_y * self.complex_dim)
        self.Ct_real_layer.apply(weights_init)
        self.Ct_imag_layer = nn.Linear(hz_dim, self.n_y * self.complex_dim)
        self.Ct_imag_layer.apply(weights_init)

        self.Dt_layer = nn.Linear(hz_dim, self.n_y * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def _step(self, z_complex, hz, ut, dt, batch):
        """Single complex-valued transition step."""
        alpha = -F.softplus(self.alpha_layer(hz))
        omega = self.omega_layer(hz)
        Lambda = torch.exp(torch.complex(alpha, omega))

        Bt = torch.complex(
            self.Bt_real_layer(hz).view(batch, self.complex_dim, self.u_dim),
            self.Bt_imag_layer(hz).view(batch, self.complex_dim, self.u_dim))

        ut_dt = (ut * dt).to(torch.cfloat)
        z_next = Lambda * z_complex + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)

        Ct = torch.complex(
            self.Ct_real_layer(hz).view(batch, self.n_y, self.complex_dim),
            self.Ct_imag_layer(hz).view(batch, self.n_y, self.complex_dim))
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
# Conditioned S4D (for GNN / multimodal dual-branch models)
# ===================================================================

class ConditionedS4DTransition(nn.Module):
    """S4D variant for split (z_dynamic, z_static) latent spaces."""

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

        self.alpha_layer = nn.Linear(hz_dim, self.complex_dim)
        self.alpha_layer.apply(weights_init)
        self.omega_layer = nn.Linear(hz_dim, self.complex_dim)
        self.omega_layer.apply(weights_init)

        self.Bt_real_layer = nn.Linear(hz_dim, self.complex_dim * self.u_dim)
        self.Bt_real_layer.apply(weights_init)
        self.Bt_imag_layer = nn.Linear(hz_dim, self.complex_dim * self.u_dim)
        self.Bt_imag_layer.apply(weights_init)

        self.Ct_real_layer = nn.Linear(hz_dim, self.n_obs * self.complex_dim)
        self.Ct_real_layer.apply(weights_init)
        self.Ct_imag_layer = nn.Linear(hz_dim, self.n_obs * self.complex_dim)
        self.Ct_imag_layer.apply(weights_init)

        self.Dt_layer = nn.Linear(hz_dim, self.n_obs * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def _step(self, z_complex, hz, ut, dt, batch):
        alpha = -F.softplus(self.alpha_layer(hz))
        omega = self.omega_layer(hz)
        Lambda = torch.exp(torch.complex(alpha, omega))

        Bt = torch.complex(
            self.Bt_real_layer(hz).view(batch, self.complex_dim, self.u_dim),
            self.Bt_imag_layer(hz).view(batch, self.complex_dim, self.u_dim))

        ut_dt = (ut * dt).to(torch.cfloat)
        z_next = Lambda * z_complex + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)

        Ct = torch.complex(
            self.Ct_real_layer(hz).view(batch, self.n_obs, self.complex_dim),
            self.Ct_imag_layer(hz).view(batch, self.n_obs, self.complex_dim))
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
