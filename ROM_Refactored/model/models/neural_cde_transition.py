"""
Neural CDE (Controlled Differential Equation) transition for E2C
=================================================================
The control input drives the dynamics through a continuous path rather
than a simple additive forcing term.  The vector field f_theta(z) is
matrix-valued and multiplies the derivative of the control path.

State equation (CDE + manual Euler integration):
    dz/dt = f_theta(z) @ dX/dt
    where dX/dt = [u_t, 1]  (control + time derivative)
    z_{t+1} = z_t + sum_{i=1}^N (1/N) * f(z_i) @ [u_t, 1]

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

No external dependencies (torchcde not required).  Manual Euler
integration following the same pattern as Neural ODE.

Reference: Kidger et al., "Neural Controlled Differential Equations
           for Irregular Time Series", NeurIPS 2020.
           github.com/patrick-kidger/torchcde
"""

import math
import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


class _CDEFunc(nn.Module):
    """Matrix-valued vector field: R^d -> R^{d x (u_dim+1)}.

    Output is reshaped so that f(z) @ dX gives a d-dimensional vector.
    """

    def __init__(self, latent_dim, control_channels, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.latent_dim = latent_dim
        self.control_channels = control_channels
        output_dim = latent_dim * control_channels

        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(fc_bn_relu(prev, h))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, z):
        """Returns [batch, latent_dim, control_channels]."""
        return self.net(z).view(-1, self.latent_dim, self.control_channels)


# ===================================================================
# Standard Neural CDE
# ===================================================================

class NeuralCDETransitionModel(nn.Module):
    """CDE transition with matrix-valued vector field and Euler integration."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        cde_cfg = config['transition'].get('neural_cde', {})
        self.n_steps = cde_cfg.get('n_integration_steps', 4)
        hidden = cde_cfg.get('cde_hidden_dims', [256, 256])

        control_channels = self.u_dim + 1
        self.cde_func = _CDEFunc(self.latent_dim, control_channels, hidden)

        d = self.latent_dim
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _integrate(self, z, ut, dt):
        """Euler integration of dz/dt = f(z) @ [u, 1]."""
        dXdt = torch.cat([ut, torch.ones_like(dt)], dim=-1)
        h = dt / self.n_steps
        for _ in range(self.n_steps):
            F = self.cde_func(z)
            dz = torch.bmm(F, dXdt.unsqueeze(-1)).squeeze(-1)
            z = z + h * dz
        return z

    def forward(self, zt, dt, ut):
        zt1 = self._integrate(zt, ut, dt)
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


# ===================================================================
# Conditioned Neural CDE
# ===================================================================

class ConditionedNeuralCDETransition(nn.Module):
    """Neural CDE for split (z_dynamic, z_static) latent spaces.

    z_static is concatenated to the control derivative as extra context.
    """

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        cde_cfg = config['transition'].get('neural_cde', {})
        self.n_steps = cde_cfg.get('n_integration_steps', 4)
        hidden = cde_cfg.get('cde_hidden_dims', [256, 256])

        control_channels = self.u_dim + 1 + static_dim
        self.cde_func = _CDEFunc(dynamic_dim, control_channels, hidden)

        d = dynamic_dim
        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _integrate(self, z, ut, z_static, dt):
        dXdt = torch.cat([ut, torch.ones_like(dt), z_static], dim=-1)
        h = dt / self.n_steps
        for _ in range(self.n_steps):
            F = self.cde_func(z)
            dz = torch.bmm(F, dXdt.unsqueeze(-1)).squeeze(-1)
            z = z + h * dz
        return z

    def forward(self, z_dyn, z_static, dt, ut):
        z_dyn_next = self._integrate(z_dyn, ut, z_static, dt)
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
