"""
Dissipative Koopman transition model for E2C architecture
==========================================================
Koopman with dissipativity guarantees via SVD parameterization.
Unlike Stable Koopman (which constrains eigenvalues / spectral radius),
this constrains *singular values* (operator norm), giving a stronger
condition that directly controls energy dissipation:  ||K||_2 < 1.

Parameterization:
    K = U_orth @ diag(sigma) @ V_orth^T
    U_orth = cayley(S_U)    -- orthogonal via Cayley transform
    V_orth = cayley(S_V)    -- of skew-symmetric parameters
    sigma_i = sigmoid(sigma_raw_i) * (1 - eps)

State equation:
    z_{t+1} = K z_t + L (u_t * dt)

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

The model exposes get_singular_values() so an external
dissipativity loss can penalize values approaching 1.

Reference: Xu, Sivaranjani, Gupta, "Learning Neural Koopman Operators
           with Dissipativity Guarantees", CDC 2025.
           arXiv:2509.07294
"""

import math
import torch
import torch.nn as nn


def _cayley(S):
    """Cayley transform: maps skew-symmetric S to orthogonal matrix.
    Q = (I - S)(I + S)^{-1}, with S skew => Q orthogonal.
    """
    d = S.size(0)
    I = torch.eye(d, device=S.device, dtype=S.dtype)
    return torch.linalg.solve(I + S, I - S)


def _skew_from_params(params, d):
    """Build a d x d skew-symmetric matrix from d*(d-1)/2 free parameters."""
    S = torch.zeros(d, d, device=params.device, dtype=params.dtype)
    idx = torch.triu_indices(d, d, offset=1)
    S[idx[0], idx[1]] = params
    return S - S.T


# ===================================================================
# Standard Dissipative Koopman
# ===================================================================

class DissipativeKoopmanTransitionModel(nn.Module):
    """Koopman with SVD-parameterized operator norm constraint."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        dk_cfg = config['transition'].get('dissipative_koopman', {})
        self.svd_constraint = dk_cfg.get('svd_constraint', 'sigmoid')
        self.contraction_eps = dk_cfg.get('contraction_eps', 0.01)

        d = self.latent_dim
        n_skew = d * (d - 1) // 2

        self.U_skew_params = nn.Parameter(torch.randn(n_skew) * 0.01)
        self.V_skew_params = nn.Parameter(torch.randn(n_skew) * 0.01)
        self.sigma_raw = nn.Parameter(torch.zeros(d))

        self.L = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_sigma = None

    def _constrain_sigma(self, raw):
        if self.svd_constraint == 'tanh':
            return torch.tanh(raw).abs() * (1.0 - self.contraction_eps)
        return torch.sigmoid(raw) * (1.0 - self.contraction_eps)

    def _build_K(self):
        d = self.latent_dim
        S_U = _skew_from_params(self.U_skew_params, d)
        S_V = _skew_from_params(self.V_skew_params, d)
        U = _cayley(S_U)
        V = _cayley(S_V)
        sigma = self._constrain_sigma(self.sigma_raw)
        self._last_sigma = sigma
        return U @ torch.diag(sigma) @ V.T

    def get_singular_values(self):
        """Return constrained singular values for dissipativity loss."""
        return self._last_sigma

    def forward(self, zt, dt, ut):
        K = self._build_K()
        ut_dt = ut * dt
        zt1 = torch.mm(zt, K.T) + torch.mm(ut_dt, self.L.T)
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        K = self._build_K()
        Zt_k, Yt_k = [], []
        for ut in U:
            ut_dt = ut * dt
            zt = torch.mm(zt, K.T) + torch.mm(ut_dt, self.L.T)
            yt = torch.mm(zt, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned Dissipative Koopman
# ===================================================================

class ConditionedDissipativeKoopmanTransition(nn.Module):
    """Dissipative Koopman for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        dk_cfg = config['transition'].get('dissipative_koopman', {})
        self.svd_constraint = dk_cfg.get('svd_constraint', 'sigmoid')
        self.contraction_eps = dk_cfg.get('contraction_eps', 0.01)

        d = dynamic_dim
        n_skew = d * (d - 1) // 2

        self.U_skew_params = nn.Parameter(torch.randn(n_skew) * 0.01)
        self.V_skew_params = nn.Parameter(torch.randn(n_skew) * 0.01)
        self.sigma_raw = nn.Parameter(torch.zeros(d))

        self.L = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )
        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_sigma = None

    def _constrain_sigma(self, raw):
        if self.svd_constraint == 'tanh':
            return torch.tanh(raw).abs() * (1.0 - self.contraction_eps)
        return torch.sigmoid(raw) * (1.0 - self.contraction_eps)

    def _build_K(self):
        d = self.dynamic_dim
        S_U = _skew_from_params(self.U_skew_params, d)
        S_V = _skew_from_params(self.V_skew_params, d)
        U = _cayley(S_U)
        V = _cayley(S_V)
        sigma = self._constrain_sigma(self.sigma_raw)
        self._last_sigma = sigma
        return U @ torch.diag(sigma) @ V.T

    def get_singular_values(self):
        return self._last_sigma

    def forward(self, z_dyn, z_static, dt, ut):
        K = self._build_K()
        ut_dt = ut * dt
        z_dyn_next = torch.mm(z_dyn, K.T) + torch.mm(ut_dt, self.L.T)
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        K = self._build_K()
        Zt_k, Yt_k = [], []
        for ut in U:
            ut_dt = ut * dt
            z_dyn = torch.mm(z_dyn, K.T) + torch.mm(ut_dt, self.L.T)
            yt = torch.mm(z_dyn, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
