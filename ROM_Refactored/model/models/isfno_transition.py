"""
IS-FNO-inspired spectral transition model for E2C architecture
================================================================
Adapts Inverse-Scattering FNO principles to latent-space dynamics:
reversible lifting, exponential spectral evolution, and control
injection.

Pipeline:
    1. Reversible lift:   z_lifted = z + g(z)
    2. Spectral evolve:   Z = FFT(z_lifted)
                          Z' = exp(r) * Z        (learnable complex r)
                          z_evolved = IFFT(Z')
    3. Control inject:    z_evolved += B_ctrl @ (u*dt)
    4. Reverse project:   z_{t+1} = z_evolved - g(z_evolved)

The lifting map is approximately invertible by design (residual form);
a reversibility loss encourages exact invertibility.

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

Reference: Yu, "An Inverse Scattering Inspired Fourier Neural Operator
           for Time-Dependent PDE Learning", arXiv:2512.19439, Dec 2025.
"""

import math
import torch
import torch.nn as nn
from model.utils.initialization import weights_init


class _LiftNet(nn.Module):
    """Small MLP residual block: g(z) for  lift(z) = z + g(z)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.net.apply(weights_init)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, z):
        return self.net(z)


# ===================================================================
# Standard IS-FNO
# ===================================================================

class ISFNOTransitionModel(nn.Module):
    """Reversible spectral transition inspired by IS-FNO."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        is_cfg = config['transition'].get('isfno', {})
        lift_hidden = is_cfg.get('lift_hidden_dim', 128)
        n_modes = is_cfg.get('n_spectral_modes', None)
        self.n_inv_steps = is_cfg.get('n_inv_steps', 5)

        d = self.latent_dim
        if n_modes is None:
            n_modes = d // 2 + 1

        self.lift_net = _LiftNet(d, lift_hidden)

        self.exp_r_real = nn.Parameter(torch.zeros(n_modes))
        self.exp_r_imag = nn.Parameter(torch.randn(n_modes) * 0.01)

        self.B_ctrl = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_rev_residual = None

    def _lift(self, z):
        return z + self.lift_net(z)

    def _inv_lift(self, z_lifted):
        """Approximate inverse via fixed-point iteration: z = z_lifted - g(z)."""
        z = z_lifted
        for _ in range(self.n_inv_steps):
            z = z_lifted - self.lift_net(z)
        return z

    def _spectral_evolve(self, z):
        Z = torch.fft.rfft(z, dim=-1)
        n_modes = self.exp_r_real.size(0)
        n_freq = Z.size(-1)
        modes = min(n_modes, n_freq)

        r = torch.complex(self.exp_r_real[:modes], self.exp_r_imag[:modes])
        exp_r = torch.exp(r)

        Z_new = Z.clone()
        Z_new[..., :modes] = Z[..., :modes] * exp_r
        return torch.fft.irfft(Z_new, n=z.size(-1), dim=-1)

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt

        z_lifted = self._lift(zt)
        z_evolved = self._spectral_evolve(z_lifted)
        z_evolved = z_evolved + torch.mm(ut_dt, self.B_ctrl.T)
        zt1 = self._inv_lift(z_evolved)

        rev_check = self._lift(self._inv_lift(z_lifted))
        self._last_rev_residual = torch.mean((z_lifted - rev_check) ** 2)

        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def get_reversibility_residual(self):
        """Return reversibility violation from last forward pass."""
        return self._last_rev_residual

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        rev_total = 0.0
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
            if self._last_rev_residual is not None:
                rev_total = rev_total + self._last_rev_residual
        if len(U) > 0:
            self._last_rev_residual = rev_total / len(U)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned IS-FNO
# ===================================================================

class ConditionedISFNOTransition(nn.Module):
    """IS-FNO variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        is_cfg = config['transition'].get('isfno', {})
        lift_hidden = is_cfg.get('lift_hidden_dim', 128)
        n_modes = is_cfg.get('n_spectral_modes', None)
        self.n_inv_steps = is_cfg.get('n_inv_steps', 5)

        d = dynamic_dim
        if n_modes is None:
            n_modes = d // 2 + 1

        self.lift_net = _LiftNet(d, lift_hidden)

        self.exp_r_real = nn.Parameter(torch.zeros(n_modes))
        self.exp_r_imag = nn.Parameter(torch.randn(n_modes) * 0.01)

        self.B_ctrl = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_rev_residual = None

    def _lift(self, z):
        return z + self.lift_net(z)

    def _inv_lift(self, z_lifted):
        z = z_lifted
        for _ in range(self.n_inv_steps):
            z = z_lifted - self.lift_net(z)
        return z

    def _spectral_evolve(self, z):
        Z = torch.fft.rfft(z, dim=-1)
        n_modes = self.exp_r_real.size(0)
        n_freq = Z.size(-1)
        modes = min(n_modes, n_freq)

        r = torch.complex(self.exp_r_real[:modes], self.exp_r_imag[:modes])
        exp_r = torch.exp(r)

        Z_new = Z.clone()
        Z_new[..., :modes] = Z[..., :modes] * exp_r
        return torch.fft.irfft(Z_new, n=z.size(-1), dim=-1)

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt

        z_lifted = self._lift(z_dyn)
        z_evolved = self._spectral_evolve(z_lifted)
        z_evolved = z_evolved + torch.mm(ut_dt, self.B_ctrl.T)
        z_dyn_next = self._inv_lift(z_evolved)

        rev_check = self._lift(self._inv_lift(z_lifted))
        self._last_rev_residual = torch.mean((z_lifted - rev_check) ** 2)

        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def get_reversibility_residual(self):
        return self._last_rev_residual

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        rev_total = 0.0
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
            if self._last_rev_residual is not None:
                rev_total = rev_total + self._last_rev_residual
        if len(U) > 0:
            self._last_rev_residual = rev_total / len(U)
        return Zt_k, Yt_k
