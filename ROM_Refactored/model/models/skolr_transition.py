"""
SKOLR (Structured Koopman Operator Linear RNN) transition for E2C
==================================================================
Bridges Koopman operator theory with linear RNNs.  The latent state
is split into N frequency branches via learnable FFT-domain masks.
Each branch has its own MLP measurement function and a diagonal
linear RNN that implements the structured Koopman operator.

State equation (per branch n):
    g_n = MLP_n(z_t)
    h_{n,t+1} = diag(lambda_n) * h_{n,t} + B_n * [g_n, u_t*dt]
    z_{t+1}   = sum_n  W_out_n @ h_{n,t+1}

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

The diagonal structure makes the RNN highly parallel and the
per-branch spectral gating captures multi-frequency dynamics.

Reference: "SKOLR: Structured Koopman Operator Linear RNN for
            Time-Series Forecasting", ICML 2025.
            github.com/networkslab/SKOLR
"""

import math
import torch
import torch.nn as nn
from model.utils.initialization import weights_init


class _MeasurementMLP(nn.Module):
    """Per-branch MLP measurement function  g: R^d -> R^h."""

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)


# ===================================================================
# Standard SKOLR
# ===================================================================

class SKOLRTransitionModel(nn.Module):
    """Structured Koopman Operator via parallel diagonal linear RNNs."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        sk_cfg = config['transition'].get('skolr', {})
        self.n_branches = sk_cfg.get('n_branches', 4)
        rnn_h = sk_cfg.get('rnn_hidden_dim', 64)
        meas_hidden = sk_cfg.get('measurement_hidden_dims', [128])

        d = self.latent_dim

        self.spectral_gates = nn.ParameterList([
            nn.Parameter(torch.randn(d) * 0.1)
            for _ in range(self.n_branches)
        ])

        self.measurement_mlps = nn.ModuleList([
            _MeasurementMLP(d, meas_hidden, rnn_h)
            for _ in range(self.n_branches)
        ])

        rnn_input = rnn_h + self.u_dim
        self.rnn_lambda_real = nn.ParameterList([
            nn.Parameter(torch.zeros(rnn_h))
            for _ in range(self.n_branches)
        ])
        self.rnn_lambda_imag = nn.ParameterList([
            nn.Parameter(torch.randn(rnn_h) * 0.1)
            for _ in range(self.n_branches)
        ])
        self.rnn_B = nn.ParameterList([
            nn.Parameter(torch.randn(rnn_h, rnn_input) * (1.0 / math.sqrt(rnn_input)))
            for _ in range(self.n_branches)
        ])

        self.W_out = nn.ModuleList([
            nn.Linear(rnn_h, d, bias=False)
            for _ in range(self.n_branches)
        ])
        for w in self.W_out:
            w.apply(weights_init)

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _get_lambda(self, n):
        """Constrained diagonal eigenvalues with magnitude < 1."""
        mag = torch.sigmoid(self.rnn_lambda_real[n]) * 0.99
        phase = self.rnn_lambda_imag[n]
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def _apply_spectral_gate(self, zt, n):
        """Apply learnable frequency mask to latent z."""
        gate = torch.sigmoid(self.spectral_gates[n])
        return zt * gate

    def _rnn_step(self, h_n, zt_gated, ut_dt, n):
        """One diagonal linear RNN step for branch n."""
        g_n = self.measurement_mlps[n](zt_gated)
        inp = torch.cat([g_n, ut_dt], dim=-1)
        b_inp = torch.mm(inp, self.rnn_B[n].T)

        lam = self._get_lambda(n)
        h_real = h_n.real * lam.real - h_n.imag * lam.imag + b_inp
        h_imag = h_n.real * lam.imag + h_n.imag * lam.real
        return torch.complex(h_real, h_imag)

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        B = zt.size(0)
        rnn_h = self.rnn_lambda_real[0].size(0)
        device = zt.device

        zt1 = torch.zeros(B, self.latent_dim, device=device)
        for n in range(self.n_branches):
            h_n = torch.zeros(B, rnn_h, dtype=torch.cfloat, device=device)
            zt_gated = self._apply_spectral_gate(zt, n)
            h_n = self._rnn_step(h_n, zt_gated, ut_dt, n)
            zt1 = zt1 + self.W_out[n](h_n.real)

        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        B = zt.size(0)
        rnn_h = self.rnn_lambda_real[0].size(0)
        device = zt.device

        h_states = [
            torch.zeros(B, rnn_h, dtype=torch.cfloat, device=device)
            for _ in range(self.n_branches)
        ]

        Zt_k, Yt_k = [], []
        for ut in U:
            ut_dt = ut * dt
            zt1 = torch.zeros(B, self.latent_dim, device=device)
            for n in range(self.n_branches):
                zt_gated = self._apply_spectral_gate(zt, n)
                h_states[n] = self._rnn_step(h_states[n], zt_gated, ut_dt, n)
                zt1 = zt1 + self.W_out[n](h_states[n].real)
            yt = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(zt1)
            Yt_k.append(yt)
            zt = zt1
        return Zt_k, Yt_k


# ===================================================================
# Conditioned SKOLR
# ===================================================================

class ConditionedSKOLRTransition(nn.Module):
    """SKOLR variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        sk_cfg = config['transition'].get('skolr', {})
        self.n_branches = sk_cfg.get('n_branches', 4)
        rnn_h = sk_cfg.get('rnn_hidden_dim', 64)
        meas_hidden = sk_cfg.get('measurement_hidden_dims', [128])

        d = dynamic_dim
        meas_in = d + static_dim

        self.spectral_gates = nn.ParameterList([
            nn.Parameter(torch.randn(d) * 0.1)
            for _ in range(self.n_branches)
        ])

        self.measurement_mlps = nn.ModuleList([
            _MeasurementMLP(meas_in, meas_hidden, rnn_h)
            for _ in range(self.n_branches)
        ])

        rnn_input = rnn_h + self.u_dim
        self.rnn_lambda_real = nn.ParameterList([
            nn.Parameter(torch.zeros(rnn_h))
            for _ in range(self.n_branches)
        ])
        self.rnn_lambda_imag = nn.ParameterList([
            nn.Parameter(torch.randn(rnn_h) * 0.1)
            for _ in range(self.n_branches)
        ])
        self.rnn_B = nn.ParameterList([
            nn.Parameter(torch.randn(rnn_h, rnn_input) * (1.0 / math.sqrt(rnn_input)))
            for _ in range(self.n_branches)
        ])

        self.W_out = nn.ModuleList([
            nn.Linear(rnn_h, d, bias=False)
            for _ in range(self.n_branches)
        ])
        for w in self.W_out:
            w.apply(weights_init)

        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _get_lambda(self, n):
        mag = torch.sigmoid(self.rnn_lambda_real[n]) * 0.99
        phase = self.rnn_lambda_imag[n]
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def _apply_spectral_gate(self, z_dyn, n):
        gate = torch.sigmoid(self.spectral_gates[n])
        return z_dyn * gate

    def _rnn_step(self, h_n, z_gated, z_static, ut_dt, n):
        g_n = self.measurement_mlps[n](torch.cat([z_gated, z_static], dim=-1))
        inp = torch.cat([g_n, ut_dt], dim=-1)
        b_inp = torch.mm(inp, self.rnn_B[n].T)
        lam = self._get_lambda(n)
        h_real = h_n.real * lam.real - h_n.imag * lam.imag + b_inp
        h_imag = h_n.real * lam.imag + h_n.imag * lam.real
        return torch.complex(h_real, h_imag)

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        B = z_dyn.size(0)
        rnn_h = self.rnn_lambda_real[0].size(0)
        device = z_dyn.device

        z_dyn_next = torch.zeros(B, self.dynamic_dim, device=device)
        for n in range(self.n_branches):
            h_n = torch.zeros(B, rnn_h, dtype=torch.cfloat, device=device)
            z_gated = self._apply_spectral_gate(z_dyn, n)
            h_n = self._rnn_step(h_n, z_gated, z_static, ut_dt, n)
            z_dyn_next = z_dyn_next + self.W_out[n](h_n.real)
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        B = z_dyn.size(0)
        rnn_h = self.rnn_lambda_real[0].size(0)
        device = z_dyn.device

        h_states = [
            torch.zeros(B, rnn_h, dtype=torch.cfloat, device=device)
            for _ in range(self.n_branches)
        ]

        Zt_k, Yt_k = [], []
        for ut in U:
            ut_dt = ut * dt
            z_dyn_next = torch.zeros(B, self.dynamic_dim, device=device)
            for n in range(self.n_branches):
                z_gated = self._apply_spectral_gate(z_dyn, n)
                h_states[n] = self._rnn_step(
                    h_states[n], z_gated, z_static, ut_dt, n
                )
                z_dyn_next = z_dyn_next + self.W_out[n](h_states[n].real)
            yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(z_dyn_next)
            Yt_k.append(yt)
            z_dyn = z_dyn_next
        return Zt_k, Yt_k
