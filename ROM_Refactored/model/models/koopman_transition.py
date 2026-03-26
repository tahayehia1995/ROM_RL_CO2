"""
Global Koopman transition model for E2C architecture
=====================================================
Uses FIXED global linear matrices K, L, C, D (nn.Parameter) instead of
state-dependent matrices from a selector MLP.

Core equation:
  z_next = K @ z  +  L @ (u*dt)
  y = C @ z_next  +  D @ (u*dt)

The nonlinearity is entirely absorbed into the encoder/decoder; the
transition is globally linear. This is the simplest transition, with
the fewest parameters, and ideal for RL (LQR-compatible dynamics).

No selector MLP is used. Training with multi-step loss (n_steps > 2)
is recommended to improve long-horizon accuracy.

Reference: Deep Koopman Operator with Control (2020),
           github.com/HaojieSHI98/DeepKoopmanWithControl
"""

import torch
import torch.nn as nn
import math


# ===================================================================
# Standard Koopman
# ===================================================================

class KoopmanTransitionModel(nn.Module):
    """Globally linear Koopman transition with fixed K, L, C, D."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        d = self.latent_dim

        # Global linear matrices
        self.K = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)
        self.L = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        zt1 = torch.mm(zt, self.K.T) + torch.mm(ut_dt, self.L.T)
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
# Conditioned Koopman
# ===================================================================

class ConditionedKoopmanTransition(nn.Module):
    """Koopman variant for split (z_dynamic, z_static) latent spaces.

    K operates on z_dynamic only. z_static conditions the observation
    through an extended C matrix.
    """

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        d = dynamic_dim

        self.K = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)
        self.L = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        z_dyn_next = torch.mm(z_dyn, self.K.T) + torch.mm(ut_dt, self.L.T)
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
