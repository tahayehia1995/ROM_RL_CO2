"""
Continuous-Time Koopman transition model for E2C architecture
==============================================================
Models latent dynamics in continuous time and discretizes exactly
via Zero-Order Hold (ZOH) using matrix exponential.

Continuous-time dynamics:
  dz/dt = A_ct * z + B_ct * u

Zero-Order Hold discretization (exact for piecewise-constant u):
  A_bar = expm(A_ct * dt)
  B_bar = (A_bar - I) @ A_ct^{-1} @ B_ct   (or approximated)

  z_next = A_bar @ z + B_bar @ u
  y = C @ z_next + D @ (u*dt)

Stability: guaranteed when all eigenvalues of A_ct have negative
real parts. This is parameterised via A_ct = A_skew - diag(gamma)
where A_skew is skew-symmetric (purely imaginary eigenvalues) and
gamma >= 0 (softplus) provides dissipation.

Reference: Koopman Autoencoders with Continuous-Time Latent Dynamics
           (arXiv:2602.02832, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================================================================
# Standard CT-Koopman
# ===================================================================

class CTKoopmanTransitionModel(nn.Module):
    """Continuous-time Koopman with ZOH discretization."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        d = self.latent_dim

        # A_ct = A_skew - diag(gamma), Re(eig) < 0 guaranteed
        # A_skew is parameterised via its upper-triangular entries
        self.A_skew_params = nn.Parameter(torch.randn(d * (d - 1) // 2) * 0.01)
        self.gamma_raw = nn.Parameter(torch.ones(d) * 0.5)

        self.B_ct = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

    def _build_A_ct(self):
        """Build stable continuous-time A matrix: A = A_skew - diag(gamma)."""
        d = self.latent_dim
        A_skew = torch.zeros(d, d, device=self.A_skew_params.device,
                              dtype=self.A_skew_params.dtype)
        idx = torch.triu_indices(d, d, offset=1)
        A_skew[idx[0], idx[1]] = self.A_skew_params
        A_skew = A_skew - A_skew.T

        gamma = F.softplus(self.gamma_raw)
        return A_skew - torch.diag(gamma)

    def _discretize(self, A_ct, dt_scalar):
        """ZOH discretization: A_bar = expm(A_ct * dt)."""
        A_bar = torch.linalg.matrix_exp(A_ct * dt_scalar)

        # B_bar = (A_bar - I) @ A_ct^{-1} @ B_ct
        # Use solve for numerical stability: A_ct @ X = (A_bar - I) => X = A_ct^{-1} @ (A_bar - I)
        eye = torch.eye(self.latent_dim, device=A_ct.device, dtype=A_ct.dtype)
        rhs = (A_bar - eye) @ self.B_ct
        B_bar = torch.linalg.solve(A_ct, rhs)

        return A_bar, B_bar

    def forward(self, zt, dt, ut):
        A_ct = self._build_A_ct()

        dt_val = dt[0, 0].item() if dt.dim() > 0 else dt.item()
        A_bar, B_bar = self._discretize(A_ct, dt_val)

        zt1 = torch.mm(zt, A_bar.T) + torch.mm(ut, B_bar.T)
        ut_dt = ut * dt
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)

        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        A_ct = self._build_A_ct()
        dt_val = dt[0, 0].item() if dt.dim() > 0 else dt.item()
        A_bar, B_bar = self._discretize(A_ct, dt_val)

        Zt_k, Yt_k = [], []
        for ut in U:
            zt = torch.mm(zt, A_bar.T) + torch.mm(ut, B_bar.T)
            ut_dt = ut * dt
            yt = torch.mm(zt, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned CT-Koopman
# ===================================================================

class ConditionedCTKoopmanTransition(nn.Module):
    """CT-Koopman variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        d = dynamic_dim

        self.A_skew_params = nn.Parameter(torch.randn(d * (d - 1) // 2) * 0.01)
        self.gamma_raw = nn.Parameter(torch.ones(d) * 0.5)

        self.B_ct = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

    def _build_A_ct(self):
        d = self.dynamic_dim
        A_skew = torch.zeros(d, d, device=self.A_skew_params.device,
                              dtype=self.A_skew_params.dtype)
        idx = torch.triu_indices(d, d, offset=1)
        A_skew[idx[0], idx[1]] = self.A_skew_params
        A_skew = A_skew - A_skew.T

        gamma = F.softplus(self.gamma_raw)
        return A_skew - torch.diag(gamma)

    def _discretize(self, A_ct, dt_scalar):
        A_bar = torch.linalg.matrix_exp(A_ct * dt_scalar)
        eye = torch.eye(self.dynamic_dim, device=A_ct.device, dtype=A_ct.dtype)
        rhs = (A_bar - eye) @ self.B_ct
        B_bar = torch.linalg.solve(A_ct, rhs)
        return A_bar, B_bar

    def forward(self, z_dyn, z_static, dt, ut):
        A_ct = self._build_A_ct()
        dt_val = dt[0, 0].item() if dt.dim() > 0 else dt.item()
        A_bar, B_bar = self._discretize(A_ct, dt_val)

        z_dyn_next = torch.mm(z_dyn, A_bar.T) + torch.mm(ut, B_bar.T)
        ut_dt = ut * dt
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)

        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        A_ct = self._build_A_ct()
        dt_val = dt[0, 0].item() if dt.dim() > 0 else dt.item()
        A_bar, B_bar = self._discretize(A_ct, dt_val)

        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn = torch.mm(z_dyn, A_bar.T) + torch.mm(ut, B_bar.T)
            ut_dt = ut * dt
            yt = torch.mm(z_dyn, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
