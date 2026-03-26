"""
Stable Koopman transition model for E2C architecture
=====================================================
Koopman with spectral stability constraints.  K is parameterized via
eigendecomposition K = V diag(lambda) V^{-1} with eigenvalue magnitudes
constrained to stay below 1 (via sigmoid scaling).

State equation:
    K = V diag(sigma(lambda_raw)) V^{-1}
    z_{t+1} = K z_t + L (u_t * dt)

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

The model also exposes its eigenvalues so an external eigenloss can
penalize any that approach the unit-circle boundary.

Reference: "Eigenvalue Initialisation and Regularisation for Koopman
            Autoencoders" (2022).
"""

import math
import torch
import torch.nn as nn


class StableKoopmanTransitionModel(nn.Module):
    """Globally-linear Koopman with spectral stability guarantees."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        sk_cfg = config['transition'].get('stable_koopman', {})
        self.use_complex = sk_cfg.get('use_complex', False)
        self.eigval_constraint = sk_cfg.get('eigval_constraint', 'sigmoid')

        d = self.latent_dim

        if self.use_complex:
            cd = d // 2
            self.eig_mag_raw = nn.Parameter(torch.zeros(cd))
            self.eig_phase = nn.Parameter(torch.randn(cd) * 0.1)
            self.V_real = nn.Parameter(torch.eye(cd) + torch.randn(cd, cd) * 0.01)
            self.V_imag = nn.Parameter(torch.randn(cd, cd) * 0.01)
            self._complex_dim = cd
        else:
            self.eig_raw = nn.Parameter(torch.zeros(d))
            self.V = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)

        self.L = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

        self._last_eigvals = None

    def _constrain(self, raw):
        if self.eigval_constraint == 'tanh':
            return torch.tanh(raw) * 0.99
        return torch.sigmoid(raw) * 0.99

    def _build_K(self):
        if self.use_complex:
            mag = self._constrain(self.eig_mag_raw)
            phase = self.eig_phase
            eigvals = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            self._last_eigvals = eigvals

            V = torch.complex(self.V_real, self.V_imag)
            K_complex = V @ torch.diag(eigvals) @ torch.linalg.inv(V)
            d = self.latent_dim
            cd = self._complex_dim
            K = torch.zeros(d, d, device=K_complex.device)
            K[:cd, :cd] = K_complex.real
            K[:cd, cd:] = -K_complex.imag
            K[cd:, :cd] = K_complex.imag
            K[cd:, cd:] = K_complex.real
            return K
        else:
            eigvals = self._constrain(self.eig_raw)
            self._last_eigvals = eigvals
            V = self.V
            K = V @ torch.diag(eigvals) @ torch.linalg.inv(V)
            return K

    def get_eigenvalues(self):
        """Return the constrained eigenvalues (for eigenloss computation)."""
        return self._last_eigvals

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


class ConditionedStableKoopmanTransition(nn.Module):
    """Stable Koopman variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        sk_cfg = config['transition'].get('stable_koopman', {})
        self.eigval_constraint = sk_cfg.get('eigval_constraint', 'sigmoid')

        d = dynamic_dim
        self.eig_raw = nn.Parameter(torch.zeros(d))
        self.V = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)
        self.L = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))
        self._last_eigvals = None

    def _constrain(self, raw):
        if self.eigval_constraint == 'tanh':
            return torch.tanh(raw) * 0.99
        return torch.sigmoid(raw) * 0.99

    def _build_K(self):
        eigvals = self._constrain(self.eig_raw)
        self._last_eigvals = eigvals
        return self.V @ torch.diag(eigvals) @ torch.linalg.inv(self.V)

    def get_eigenvalues(self):
        return self._last_eigvals

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
