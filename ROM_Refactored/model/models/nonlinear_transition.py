"""
Nonlinear (Neural ODE) transition model for E2C architecture
=============================================================
Uses a learned ODE right-hand-side to evolve latent states, integrated with
Euler steps.  A separate observation head maps the predicted latent state
(and scaled control) to well-level observations, consistent with the
interface of every other transition model.

State equation (ODE + Euler):
    dz/dt = f_θ(z, u)
    z_{k+1} = z_k + (Δt / n_euler) * f_θ(z_k, u_k)   (repeated n_euler times)

Observation equation (linear, matching Koopman / SSM convention):
    ŷ_{t+1} = C · z_{t+1} + D · (u_t · Δt)

The observation head uses *learnable global* C, D matrices (same as Koopman)
so that the predicted observations live in the same space as those from
linear models.  A config flag `nonlinear.obs_head` can switch to an MLP
observation head for additional expressiveness.

Reference:
    Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
    torch-neural-ssm  (qu-gg/torch-neural-ssm)
"""

import math
import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu
from model.utils.initialization import weights_init


# ---------------------------------------------------------------------------
# ODE dynamics network  (f_θ)
# ---------------------------------------------------------------------------

class _ODEFunc(nn.Module):
    """MLP that predicts dz/dt given (z, u)."""

    def __init__(self, latent_dim, u_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = latent_dim + u_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(fc_bn_relu(prev, h))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, z, u):
        return self.net(torch.cat([z, u], dim=-1))


def _euler_integrate(z, u, dt, n_euler, ode_func):
    """Fixed-step Euler integration of dz/dt = ode_func(z, u)."""
    h = dt / n_euler
    for _ in range(n_euler):
        z = z + h * ode_func(z, u)
    return z


# ===================================================================
# Standard Nonlinear Transition  (drop-in replacement for other models)
# ===================================================================

class NonlinearTransitionModel(nn.Module):
    """Neural-ODE transition with a linear observation head.

    Constructor accepts ``config`` (dict-like) identical to all other
    transition models so it can be created by ``create_transition_model``.
    """

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        nl_cfg = config['transition'].get('nonlinear', {})
        self.n_euler = nl_cfg.get('n_euler_steps', 4)
        ode_hidden = nl_cfg.get('ode_hidden_dims', [256, 256])
        obs_type = nl_cfg.get('obs_head', 'linear')

        self.ode_func = _ODEFunc(self.latent_dim, self.u_dim, ode_hidden)

        if obs_type == 'mlp':
            obs_input = self.latent_dim + self.u_dim
            self.obs_head = nn.Sequential(
                fc_bn_relu(obs_input, 128),
                fc_bn_relu(128, 128),
                nn.Linear(128, self.n_y),
            )
            self.obs_head.apply(weights_init)
            self._obs_forward = self._obs_mlp
        else:
            d = self.latent_dim
            self.C = nn.Parameter(
                torch.randn(self.n_y, d) * (1.0 / math.sqrt(d))
            )
            self.D = nn.Parameter(
                torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
            )
            self._obs_forward = self._obs_linear

    # ---------- observation helpers ----------

    def _obs_linear(self, z_next, ut_dt):
        yt = torch.mm(z_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return yt

    def _obs_mlp(self, z_next, ut_dt):
        return self.obs_head(torch.cat([z_next, ut_dt], dim=-1))

    # ---------- single step (same signature as all other models) ----------

    def forward(self, zt, dt, ut):
        zt1 = _euler_integrate(zt, ut, dt, self.n_euler, self.ode_func)
        ut_dt = ut * dt
        yt1 = self._obs_forward(zt1, ut_dt)
        return zt1, yt1

    # ---------- multi-step rollout ----------

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            zt, yt = self.forward(zt, dt, ut)
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ===================================================================
# Conditioned Nonlinear (for GNN / multimodal dual-branch models)
# ===================================================================

class ConditionedNonlinearTransition(nn.Module):
    """Neural-ODE variant for split (z_dynamic, z_static) latent spaces.

    The ODE evolves only z_dynamic; z_static is concatenated as extra
    conditioning context to the ODE function.
    """

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        nl_cfg = config['transition'].get('nonlinear', {})
        self.n_euler = nl_cfg.get('n_euler_steps', 4)
        ode_hidden = nl_cfg.get('ode_hidden_dims', [256, 256])

        self.ode_func = _ODEFunc(
            dynamic_dim, self.u_dim + static_dim, ode_hidden
        )

        d = dynamic_dim
        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, z_dyn, z_static, dt, ut):
        u_cond = torch.cat([ut, z_static], dim=-1)
        z_dyn_next = _euler_integrate(
            z_dyn, u_cond, dt, self.n_euler, self.ode_func
        )
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
