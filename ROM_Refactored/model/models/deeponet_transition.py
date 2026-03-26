"""
DeepONet (Deep Operator Network) one-shot transition for E2C
=============================================================
Learns the solution operator mapping (z_0, control_sequence) -> z(t)
at any query time.  A branch network encodes the full control context,
a trunk network encodes the query time, and their dot product produces
the prediction.

Architecture:
    Branch: MLP([z_0, u_0, u_1, ..., u_{T-1}]) -> R^{basis_dim}
    Trunk:  MLP(t_query / T_max)                -> R^{basis_dim}
    z(t) = z_0 + sum_k branch_k * trunk_k       (residual form)

Key property: forward_nsteps is NON-AUTOREGRESSIVE.  All timesteps
are predicted simultaneously from the full control sequence without
sequential rollout, eliminating error accumulation.

Observation equation:
    y(t) = C z(t) + D (u(t) * dt)

Reference: Lu et al., "DeepONet: Learning nonlinear operators",
           Nature Machine Intelligence 2021.
           github.com/katiana22/latent-deeponet
"""

import math
import torch
import torch.nn as nn
from model.utils.initialization import weights_init


class _BranchNet(nn.Module):
    """Branch network: encodes (z_0, full control sequence) -> R^{basis_dim * d}."""

    def __init__(self, input_dim, basis_dim, latent_dim, hidden_dims):
        super().__init__()
        self.basis_dim = basis_dim
        self.latent_dim = latent_dim
        output_dim = basis_dim * latent_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, x):
        """Returns [batch, basis_dim, latent_dim]."""
        return self.net(x).view(-1, self.basis_dim, self.latent_dim)


class _TrunkNet(nn.Module):
    """Trunk network: encodes normalised query time -> R^{basis_dim}."""

    def __init__(self, basis_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = 1
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, basis_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, t_norm):
        """t_norm: [batch, 1] normalised time. Returns [batch, basis_dim]."""
        return self.net(t_norm)


# ===================================================================
# Standard DeepONet
# ===================================================================

class DeepONetTransitionModel(nn.Module):
    """One-shot operator transition: non-autoregressive forward_nsteps."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        don_cfg = config['transition'].get('deeponet', {})
        branch_hidden = don_cfg.get('branch_hidden_dims', [256, 256])
        trunk_hidden = don_cfg.get('trunk_hidden_dims', [128, 128])
        self.basis_dim = don_cfg.get('basis_dim', 64)

        n_steps_train = config['training'].get('nsteps', 2) - 1
        if n_steps_train < 1:
            n_steps_train = 1
        self._max_seq_len = n_steps_train

        d = self.latent_dim
        branch_input = d + self._max_seq_len * self.u_dim
        self.branch_net = _BranchNet(branch_input, self.basis_dim, d, branch_hidden)
        self.trunk_net = _TrunkNet(self.basis_dim, trunk_hidden)

        self.branch_bias = nn.Parameter(torch.zeros(d))

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _build_branch_input(self, z0, U_list):
        """Flatten z0 + control sequence into branch input.

        Pads or truncates to self._max_seq_len.
        """
        B = z0.size(0)
        n_given = len(U_list)
        device = z0.device

        if n_given >= self._max_seq_len:
            u_flat = torch.cat(
                [u for u in U_list[:self._max_seq_len]], dim=-1
            )
        else:
            parts = [u for u in U_list]
            pad = torch.zeros(
                B, (self._max_seq_len - n_given) * self.u_dim, device=device
            )
            u_flat = torch.cat(parts + [pad], dim=-1)

        return torch.cat([z0, u_flat], dim=-1)

    def forward(self, zt, dt, ut):
        """Single-step compatibility: predict one step ahead."""
        branch_in = self._build_branch_input(zt, [ut])
        branch_out = self.branch_net(branch_in)
        t_norm = torch.ones(zt.size(0), 1, device=zt.device)
        trunk_out = self.trunk_net(t_norm)
        delta = torch.einsum('bp,bpd->bd', trunk_out, branch_out) + self.branch_bias
        zt1 = zt + delta
        ut_dt = ut * dt
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        """Non-autoregressive: predict all steps simultaneously."""
        n_steps = len(U)
        B = zt.size(0)
        device = zt.device

        branch_in = self._build_branch_input(zt, U)
        branch_out = self.branch_net(branch_in)

        Zt_k, Yt_k = [], []
        for i, ut in enumerate(U):
            t_norm = torch.full(
                (B, 1), (i + 1) / max(n_steps, 1), device=device
            )
            trunk_out = self.trunk_net(t_norm)
            delta = torch.einsum('bp,bpd->bd', trunk_out, branch_out) + self.branch_bias
            zt_i = zt + delta
            ut_dt = ut * dt
            yt_i = torch.mm(zt_i, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(zt_i)
            Yt_k.append(yt_i)

        return Zt_k, Yt_k


# ===================================================================
# Conditioned DeepONet
# ===================================================================

class ConditionedDeepONetTransition(nn.Module):
    """DeepONet for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        don_cfg = config['transition'].get('deeponet', {})
        branch_hidden = don_cfg.get('branch_hidden_dims', [256, 256])
        trunk_hidden = don_cfg.get('trunk_hidden_dims', [128, 128])
        self.basis_dim = don_cfg.get('basis_dim', 64)

        n_steps_train = config['training'].get('nsteps', 2) - 1
        if n_steps_train < 1:
            n_steps_train = 1
        self._max_seq_len = n_steps_train

        d = dynamic_dim
        branch_input = d + static_dim + self._max_seq_len * self.u_dim
        self.branch_net = _BranchNet(branch_input, self.basis_dim, d, branch_hidden)
        self.trunk_net = _TrunkNet(self.basis_dim, trunk_hidden)

        self.branch_bias = nn.Parameter(torch.zeros(d))

        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _build_branch_input(self, z_dyn, z_static, U_list):
        B = z_dyn.size(0)
        n_given = len(U_list)
        device = z_dyn.device

        if n_given >= self._max_seq_len:
            u_flat = torch.cat(
                [u for u in U_list[:self._max_seq_len]], dim=-1
            )
        else:
            parts = [u for u in U_list]
            pad = torch.zeros(
                B, (self._max_seq_len - n_given) * self.u_dim, device=device
            )
            u_flat = torch.cat(parts + [pad], dim=-1)

        return torch.cat([z_dyn, z_static, u_flat], dim=-1)

    def forward(self, z_dyn, z_static, dt, ut):
        branch_in = self._build_branch_input(z_dyn, z_static, [ut])
        branch_out = self.branch_net(branch_in)
        t_norm = torch.ones(z_dyn.size(0), 1, device=z_dyn.device)
        trunk_out = self.trunk_net(t_norm)
        delta = torch.einsum('bp,bpd->bd', trunk_out, branch_out) + self.branch_bias
        z_dyn_next = z_dyn + delta
        ut_dt = ut * dt
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        n_steps = len(U)
        B = z_dyn.size(0)
        device = z_dyn.device

        branch_in = self._build_branch_input(z_dyn, z_static, U)
        branch_out = self.branch_net(branch_in)

        Zt_k, Yt_k = [], []
        for i, ut in enumerate(U):
            t_norm = torch.full(
                (B, 1), (i + 1) / max(n_steps, 1), device=device
            )
            trunk_out = self.trunk_net(t_norm)
            delta = torch.einsum('bp,bpd->bd', trunk_out, branch_out) + self.branch_bias
            z_i = z_dyn + delta
            ut_dt = ut * dt
            yt_i = torch.mm(z_i, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(z_i)
            Yt_k.append(yt_i)

        return Zt_k, Yt_k
