"""
Latent SDE (Stochastic Differential Equation) transition for E2C
================================================================
Adds a learned diffusion term to the dynamics for calibrated
uncertainty in predictions.  During training, stochastic trajectories
are sampled via Euler-Maruyama integration.  During inference, the
mean (drift-only) trajectory is used.

State equation (Euler-Maruyama):
    z_{t+1} = z_t + f(z_t)*dt + g(z_t)*sqrt(dt)*eps + B*(u_t*dt)
    where eps ~ N(0, I)  (only during training)

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

Exposes get_diffusion_values() for an auxiliary KL/regularisation loss
that prevents diffusion collapse or explosion.

Reference: Li et al., "Scalable Gradients for SDEs", AISTATS 2020.
           google-research/torchsde
"""

import math
import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


class _DriftNet(nn.Module):
    """Drift function f: R^d -> R^d."""

    def __init__(self, dim, hidden_dims):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden_dims:
            layers.append(fc_bn_relu(prev, h))
            prev = h
        layers.append(nn.Linear(prev, dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, z):
        return self.net(z)


class _DiffusionNet(nn.Module):
    """Diffusion function g: R^d -> R^d (diagonal, positive via softplus)."""

    def __init__(self, dim, hidden_dims):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, z):
        return torch.nn.functional.softplus(self.net(z)) + 1e-5


# ===================================================================
# Standard Latent SDE
# ===================================================================

class LatentSDETransitionModel(nn.Module):
    """Stochastic transition with learned drift and diffusion."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        sde_cfg = config['transition'].get('latent_sde', {})
        drift_hidden = sde_cfg.get('drift_hidden_dims', [256, 256])
        diff_hidden = sde_cfg.get('diffusion_hidden_dims', [128])
        self.n_euler = sde_cfg.get('n_euler_steps', 4)

        d = self.latent_dim
        self.drift_net = _DriftNet(d, drift_hidden)
        self.diffusion_net = _DiffusionNet(d, diff_hidden)

        self.B_sde = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_diffusion = None

    def _step(self, z, ut, dt):
        """Single Euler-Maruyama macro-step with n_euler sub-steps."""
        h = dt / self.n_euler
        sqrt_h = torch.sqrt(h.abs() + 1e-8)
        ut_contrib = torch.mm(ut * dt, self.B_sde.T)

        all_g = []
        for _ in range(self.n_euler):
            f = self.drift_net(z)
            g = self.diffusion_net(z)
            all_g.append(g)
            if self.training:
                eps = torch.randn_like(z)
                z = z + h * f + sqrt_h * g * eps
            else:
                z = z + h * f

        z = z + ut_contrib
        self._last_diffusion = torch.stack(all_g).mean(dim=0)
        return z

    def forward(self, zt, dt, ut):
        zt1 = self._step(zt, ut, dt)
        ut_dt = ut * dt
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        diff_sum = 0.0
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
            if self._last_diffusion is not None:
                diff_sum = diff_sum + self._last_diffusion.mean()
        if len(U) > 0:
            self._last_diffusion_scalar = diff_sum / len(U)
        return Zt_k, Yt_k

    def get_diffusion_values(self):
        """Return mean diffusion magnitude for regularisation loss."""
        if hasattr(self, '_last_diffusion_scalar'):
            return self._last_diffusion_scalar
        if self._last_diffusion is not None:
            return self._last_diffusion.mean()
        return None


# ===================================================================
# Conditioned Latent SDE
# ===================================================================

class ConditionedLatentSDETransition(nn.Module):
    """Latent SDE for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        sde_cfg = config['transition'].get('latent_sde', {})
        drift_hidden = sde_cfg.get('drift_hidden_dims', [256, 256])
        diff_hidden = sde_cfg.get('diffusion_hidden_dims', [128])
        self.n_euler = sde_cfg.get('n_euler_steps', 4)

        d = dynamic_dim
        self.drift_net = _DriftNet(d, drift_hidden)
        self.diffusion_net = _DiffusionNet(d, diff_hidden)

        self.B_sde = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )
        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_diffusion = None

    def _step(self, z, ut, dt):
        h = dt / self.n_euler
        sqrt_h = torch.sqrt(h.abs() + 1e-8)
        ut_contrib = torch.mm(ut * dt, self.B_sde.T)

        all_g = []
        for _ in range(self.n_euler):
            f = self.drift_net(z)
            g = self.diffusion_net(z)
            all_g.append(g)
            if self.training:
                eps = torch.randn_like(z)
                z = z + h * f + sqrt_h * g * eps
            else:
                z = z + h * f

        z = z + ut_contrib
        self._last_diffusion = torch.stack(all_g).mean(dim=0)
        return z

    def forward(self, z_dyn, z_static, dt, ut):
        z_dyn_next = self._step(z_dyn, ut, dt)
        ut_dt = ut * dt
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        diff_sum = 0.0
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
            if self._last_diffusion is not None:
                diff_sum = diff_sum + self._last_diffusion.mean()
        if len(U) > 0:
            self._last_diffusion_scalar = diff_sum / len(U)
        return Zt_k, Yt_k

    def get_diffusion_values(self):
        if hasattr(self, '_last_diffusion_scalar'):
            return self._last_diffusion_scalar
        if self._last_diffusion is not None:
            return self._last_diffusion.mean()
        return None
