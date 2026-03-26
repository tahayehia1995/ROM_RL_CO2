"""
SINDy (Sparse Identification of Nonlinear Dynamics) transition for E2C
=======================================================================
Instead of a neural network, the transition uses a library of candidate
basis functions (polynomials, optionally trigonometric) with a sparse
coefficient matrix.  Training with L1 regularisation + sequential
thresholding discovers a compact symbolic representation of the latent
dynamics.

State equation:
    dz/dt = Theta(z) @ xi  +  B @ u
    z_{t+1} = z_t + dt * dz/dt                (Euler)

where Theta(z) is a fixed function library:
    [1, z_1, ..., z_d, z_1*z_2, ..., z_d^2, ..., sin(z_1), ...]

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

The model exposes:
  - get_sindy_coefficients() for sparsity loss
  - get_sindy_velocity(zt, dt) for consistency loss
  - apply_threshold(threshold) for sequential thresholding

Reference: Champion et al., "Data-driven discovery of coordinates and
           governing equations", PNAS 2019.
           github.com/kpchamp/SindyAutoencoders
"""

import math
import torch
import torch.nn as nn
from itertools import combinations_with_replacement


def _build_library_indices(d, poly_order, include_sine):
    """Pre-compute the library structure so we can build Theta(z) fast."""
    terms = []
    terms.append(('const',))

    for i in range(d):
        terms.append(('z', i))

    for order in range(2, poly_order + 1):
        for combo in combinations_with_replacement(range(d), order):
            terms.append(('poly', combo))

    if include_sine:
        for i in range(d):
            terms.append(('sin', i))
            terms.append(('cos', i))

    return terms


def _eval_library(z, terms):
    """Evaluate function library Theta(z). Returns [batch, library_dim]."""
    B = z.size(0)
    cols = []
    for t in terms:
        if t[0] == 'const':
            cols.append(torch.ones(B, 1, device=z.device, dtype=z.dtype))
        elif t[0] == 'z':
            cols.append(z[:, t[1]:t[1]+1])
        elif t[0] == 'poly':
            val = torch.ones(B, 1, device=z.device, dtype=z.dtype)
            for idx in t[1]:
                val = val * z[:, idx:idx+1]
            cols.append(val)
        elif t[0] == 'sin':
            cols.append(torch.sin(z[:, t[1]:t[1]+1]))
        elif t[0] == 'cos':
            cols.append(torch.cos(z[:, t[1]:t[1]+1]))
    return torch.cat(cols, dim=1)


# ===================================================================
# Standard SINDy
# ===================================================================

class SINDyTransitionModel(nn.Module):
    """Sparse symbolic transition via SINDy function library."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        sc = config['transition'].get('sindy', {})
        poly_order = sc.get('poly_order', 3)
        include_sine = sc.get('include_sine', False)

        d = self.latent_dim
        self._terms = _build_library_indices(d, poly_order, include_sine)
        lib_dim = len(self._terms)

        self.sindy_coefficients = nn.Parameter(
            torch.randn(lib_dim, d) * 0.01
        )
        self.sindy_B = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_z_dot_sindy = None

    def _sindy_rhs(self, z, u):
        theta = _eval_library(z, self._terms)
        dz_auto = torch.mm(theta, self.sindy_coefficients)
        dz_ctrl = torch.mm(u, self.sindy_B.T)
        return dz_auto + dz_ctrl, dz_auto

    def forward(self, zt, dt, ut):
        dz, dz_auto = self._sindy_rhs(zt, ut)
        zt1 = zt + dt * dz
        self._last_z_dot_sindy = dz_auto
        ut_dt = ut * dt
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def get_sindy_coefficients(self):
        return self.sindy_coefficients

    def get_sindy_velocity(self):
        return self._last_z_dot_sindy

    def apply_threshold(self, threshold):
        """Zero out coefficients below threshold (sequential thresholding)."""
        with torch.no_grad():
            mask = self.sindy_coefficients.abs() >= threshold
            self.sindy_coefficients.mul_(mask.float())


# ===================================================================
# Conditioned SINDy
# ===================================================================

class ConditionedSINDyTransition(nn.Module):
    """SINDy variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        sc = config['transition'].get('sindy', {})
        poly_order = sc.get('poly_order', 3)
        include_sine = sc.get('include_sine', False)

        d = dynamic_dim
        self._terms = _build_library_indices(d, poly_order, include_sine)
        lib_dim = len(self._terms)

        self.sindy_coefficients = nn.Parameter(
            torch.randn(lib_dim, d) * 0.01
        )
        self.sindy_B = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_z_dot_sindy = None

    def _sindy_rhs(self, z, u):
        theta = _eval_library(z, self._terms)
        dz_auto = torch.mm(theta, self.sindy_coefficients)
        dz_ctrl = torch.mm(u, self.sindy_B.T)
        return dz_auto + dz_ctrl, dz_auto

    def forward(self, z_dyn, z_static, dt, ut):
        dz, dz_auto = self._sindy_rhs(z_dyn, ut)
        z_dyn_next = z_dyn + dt * dz
        self._last_z_dot_sindy = dz_auto
        ut_dt = ut * dt
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def get_sindy_coefficients(self):
        return self.sindy_coefficients

    def get_sindy_velocity(self):
        return self._last_z_dot_sindy

    def apply_threshold(self, threshold):
        with torch.no_grad():
            mask = self.sindy_coefficients.abs() >= threshold
            self.sindy_coefficients.mul_(mask.float())
