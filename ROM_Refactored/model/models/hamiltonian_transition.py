"""
Hamiltonian Neural ODE transition model for E2C architecture
=============================================================
Structure-preserving Neural ODE where dynamics follow Hamilton's
equations.  A neural network learns the energy function H(z), and
time derivatives are computed via the symplectic structure.

State equation:
    z = [q, p]   (position + momentum, each d/2 dimensional)
    dq/dt =  dH/dp
    dp/dt = -dH/dq + B_ctrl * u
    Euler integration with n_euler sub-steps.

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

The model also exposes an energy-conservation loss: the variance
of H across trajectory steps, penalizing energy drift.

Reference: Greydanus et al. "Hamiltonian Neural Networks" (2019),
           GeoHNN (2025).
"""

import math
import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


class _HamiltonianNet(nn.Module):
    """MLP that predicts scalar energy H from the full state z = [q, p]."""

    def __init__(self, state_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Softplus())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, z):
        return self.net(z)


# ===================================================================
# Standard Hamiltonian
# ===================================================================

class HamiltonianTransitionModel(nn.Module):
    """Hamiltonian Neural ODE with linear observation head."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        assert self.latent_dim % 2 == 0, \
            f"Hamiltonian requires even latent_dim, got {self.latent_dim}"
        self.half_dim = self.latent_dim // 2

        ham_cfg = config['transition'].get('hamiltonian', {})
        self.n_euler = ham_cfg.get('n_euler_steps', 4)
        h_hidden = ham_cfg.get('hamiltonian_hidden_dims', [128, 128])

        self.H_net = _HamiltonianNet(self.latent_dim, h_hidden)
        self.B_ctrl = nn.Parameter(
            torch.randn(self.half_dim, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        d = self.latent_dim
        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        self._last_energies = None

    def _hamiltonian_rhs(self, z, u):
        """Compute dz/dt = J * dH/dz + control, where J is the symplectic matrix."""
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            H = self.H_net(z_req).sum()
            dH = torch.autograd.grad(H, z_req, create_graph=True)[0]

        dHdq = dH[:, :self.half_dim]
        dHdp = dH[:, self.half_dim:]

        dqdt = dHdp
        dpdt = -dHdq + torch.mm(u, self.B_ctrl.T)

        return torch.cat([dqdt, dpdt], dim=-1)

    def compute_energy(self, z):
        """Compute H(z) for the energy conservation loss."""
        with torch.no_grad():
            return self.H_net(z).squeeze(-1)

    def get_trajectory_energies(self):
        """Return energies from last forward_nsteps call."""
        return self._last_energies

    def forward(self, zt, dt, ut):
        h = dt / self.n_euler
        z = zt
        for _ in range(self.n_euler):
            z = z + h * self._hamiltonian_rhs(z, ut)

        zt1 = z
        ut_dt = ut * dt
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        self._last_energies = [self.compute_energy(zt)]
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
            self._last_energies.append(self.compute_energy(zt))
        return Zt_k, Yt_k


# ===================================================================
# Conditioned Hamiltonian
# ===================================================================

class ConditionedHamiltonianTransition(nn.Module):
    """Hamiltonian variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        assert dynamic_dim % 2 == 0, \
            f"Hamiltonian requires even dynamic_dim, got {dynamic_dim}"
        self.half_dim = dynamic_dim // 2

        ham_cfg = config['transition'].get('hamiltonian', {})
        self.n_euler = ham_cfg.get('n_euler_steps', 4)
        h_hidden = ham_cfg.get('hamiltonian_hidden_dims', [128, 128])

        self.H_net = _HamiltonianNet(dynamic_dim + static_dim, h_hidden)
        self.B_ctrl = nn.Parameter(
            torch.randn(self.half_dim, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        d = dynamic_dim
        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _hamiltonian_rhs(self, z_dyn, z_static, u):
        with torch.enable_grad():
            z_full = torch.cat([z_dyn, z_static], dim=-1)
            z_req = z_full.detach().requires_grad_(True)
            H = self.H_net(z_req).sum()
            dH = torch.autograd.grad(H, z_req, create_graph=True)[0]
        dH_dyn = dH[:, :self.dynamic_dim]

        dHdq = dH_dyn[:, :self.half_dim]
        dHdp = dH_dyn[:, self.half_dim:]

        dqdt = dHdp
        dpdt = -dHdq + torch.mm(u, self.B_ctrl.T)

        return torch.cat([dqdt, dpdt], dim=-1)

    def forward(self, z_dyn, z_static, dt, ut):
        h = dt / self.n_euler
        z = z_dyn
        for _ in range(self.n_euler):
            z = z + h * self._hamiltonian_rhs(z, z_static, ut)

        z_dyn_next = z
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
