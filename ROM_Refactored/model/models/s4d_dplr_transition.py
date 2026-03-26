"""
S4D-DPLR (Diagonal Plus Low-Rank) transition model for E2C architecture
=========================================================================
Extends S4D with a low-rank correction U*V^H that adds cross-dimensional
coupling while preserving the stability properties of complex diagonal
eigenvalues.

Core equation:
  z_next = (Lambda + U @ V^H) @ z  +  B * (u*dt)
  y = Re(C * z_next) + D * (u*dt)

where Lambda is complex diagonal (stable), and U*V^H (rank r) captures
cross-dimensional interactions (e.g. pressure-saturation coupling) that
pure diagonal models miss.

Reference: S4 (NeurIPS 2021), S4D (NeurIPS 2022).
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
# Standard S4D-DPLR
# ===================================================================

class S4DPLRTransitionModel(nn.Module):
    """Complex diagonal + low-rank transition."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.complex_dim = self.latent_dim // 2
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        dplr_cfg = config['transition'].get('s4d_dplr', {})
        self.rank = dplr_cfg.get('rank', 8)

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])

        selector_input = self.latent_dim + self.u_dim + 1
        hz_dim = self.latent_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.alpha_layer = nn.Linear(hz_dim, self.complex_dim)
        self.alpha_layer.apply(weights_init)
        self.omega_layer = nn.Linear(hz_dim, self.complex_dim)
        self.omega_layer.apply(weights_init)

        # Low-rank factors U, V: each (complex_dim, rank) complex
        self.U_real_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.U_real_layer.apply(weights_init)
        self.U_imag_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.U_imag_layer.apply(weights_init)
        self.V_real_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.V_real_layer.apply(weights_init)
        self.V_imag_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.V_imag_layer.apply(weights_init)

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
        d = self.complex_dim
        r = self.rank

        alpha = -F.softplus(self.alpha_layer(hz))
        omega = self.omega_layer(hz)
        Lambda = torch.exp(torch.complex(alpha, omega))

        U = torch.complex(
            self.U_real_layer(hz).view(batch, d, r),
            self.U_imag_layer(hz).view(batch, d, r))
        V = torch.complex(
            self.V_real_layer(hz).view(batch, d, r),
            self.V_imag_layer(hz).view(batch, d, r))

        # z_next = Lambda * z + (U @ V^H) @ z + B * (u*dt)
        low_rank_term = torch.bmm(U, torch.bmm(V.conj().transpose(1, 2),
                                                 z_complex.unsqueeze(-1))).squeeze(-1)
        Bt = torch.complex(
            self.Bt_real_layer(hz).view(batch, d, self.u_dim),
            self.Bt_imag_layer(hz).view(batch, d, self.u_dim))
        ut_dt = (ut * dt).to(torch.cfloat)

        z_next = Lambda * z_complex + low_rank_term + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)

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
# Conditioned S4D-DPLR
# ===================================================================

class ConditionedS4DPLRTransition(nn.Module):
    """S4D-DPLR variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.complex_dim = dynamic_dim // 2
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        dplr_cfg = config['transition'].get('s4d_dplr', {})
        self.rank = dplr_cfg.get('rank', 8)

        selector_hidden = config['transition'].get('encoder_hidden_dims', [128, 128])

        selector_input = dynamic_dim + static_dim + self.u_dim + 1
        hz_dim = dynamic_dim + static_dim
        self.selector = _build_selector(selector_input, hz_dim, selector_hidden)
        self.selector.apply(weights_init)

        self.alpha_layer = nn.Linear(hz_dim, self.complex_dim)
        self.alpha_layer.apply(weights_init)
        self.omega_layer = nn.Linear(hz_dim, self.complex_dim)
        self.omega_layer.apply(weights_init)

        self.U_real_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.U_real_layer.apply(weights_init)
        self.U_imag_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.U_imag_layer.apply(weights_init)
        self.V_real_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.V_real_layer.apply(weights_init)
        self.V_imag_layer = nn.Linear(hz_dim, self.complex_dim * self.rank)
        self.V_imag_layer.apply(weights_init)

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
        d = self.complex_dim
        r = self.rank

        alpha = -F.softplus(self.alpha_layer(hz))
        omega = self.omega_layer(hz)
        Lambda = torch.exp(torch.complex(alpha, omega))

        U = torch.complex(
            self.U_real_layer(hz).view(batch, d, r),
            self.U_imag_layer(hz).view(batch, d, r))
        V = torch.complex(
            self.V_real_layer(hz).view(batch, d, r),
            self.V_imag_layer(hz).view(batch, d, r))

        low_rank_term = torch.bmm(U, torch.bmm(V.conj().transpose(1, 2),
                                                 z_complex.unsqueeze(-1))).squeeze(-1)
        Bt = torch.complex(
            self.Bt_real_layer(hz).view(batch, d, self.u_dim),
            self.Bt_imag_layer(hz).view(batch, d, self.u_dim))
        ut_dt = (ut * dt).to(torch.cfloat)

        z_next = Lambda * z_complex + low_rank_term + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)

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
