"""
Bilinear Koopman transition model for E2C architecture
=======================================================
Extends standard Koopman with bilinear state-control interaction
terms.  The bilinear term N_i z u_i captures how the effect of
each control input depends on the current state — critical for
reservoir simulation where injection effects vary with saturation.

State equation:
    z_{t+1} = A z_t + sum_i(N_i z_t * u_i * dt) + B (u_t * dt)

Full-rank:  N is [u_dim, d, d]  (u_dim bilinear matrices)
Low-rank:   N_i = P_i Q_i^T    (each rank-r, stored as [u_dim, d, r])

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

Reference: "Deep bilinear Koopman realization for dynamics modeling
            and predictive control", Springer 2024.
            github.com/roahmlab/koopman-realizations
"""

import math
import torch
import torch.nn as nn


# ===================================================================
# Standard Bilinear Koopman
# ===================================================================

class BilinearKoopmanTransitionModel(nn.Module):
    """Koopman with bilinear state-control coupling."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        bk_cfg = config['transition'].get('bilinear_koopman', {})
        self.use_low_rank = bk_cfg.get('use_low_rank_N', False)
        self.N_rank = bk_cfg.get('N_rank', 16)

        d = self.latent_dim

        self.A_bilinear = nn.Parameter(
            torch.eye(d) + torch.randn(d, d) * 0.01
        )
        self.B_bilinear = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        if self.use_low_rank:
            r = min(self.N_rank, d)
            self.N_P = nn.Parameter(
                torch.randn(self.u_dim, d, r) * (1.0 / math.sqrt(d))
            )
            self.N_Q = nn.Parameter(
                torch.randn(self.u_dim, d, r) * (1.0 / math.sqrt(d))
            )
        else:
            self.N_bilinear = nn.Parameter(
                torch.randn(self.u_dim, d, d) * (0.01 / math.sqrt(d))
            )

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _bilinear_term(self, zt, ut_dt):
        """Compute sum_i N_i @ z_t * u_i_dt."""
        if self.use_low_rank:
            N = torch.bmm(self.N_P, self.N_Q.transpose(1, 2))
        else:
            N = self.N_bilinear

        # N: [u_dim, d, d], zt: [B, d], ut_dt: [B, u_dim]
        # For each control dim i: N[i] @ zt^T * ut_dt[:, i]
        # Efficient: einsum('mij,bj,bm->bi', N, zt, ut_dt)
        return torch.einsum('mij,bj,bm->bi', N, zt, ut_dt)

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        z_linear = torch.mm(zt, self.A_bilinear.T) + torch.mm(ut_dt, self.B_bilinear.T)
        z_bilinear = self._bilinear_term(zt, ut_dt)
        zt1 = z_linear + z_bilinear
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
# Conditioned Bilinear Koopman
# ===================================================================

class ConditionedBilinearKoopmanTransition(nn.Module):
    """Bilinear Koopman for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        bk_cfg = config['transition'].get('bilinear_koopman', {})
        self.use_low_rank = bk_cfg.get('use_low_rank_N', False)
        self.N_rank = bk_cfg.get('N_rank', 16)

        d = dynamic_dim

        self.A_bilinear = nn.Parameter(
            torch.eye(d) + torch.randn(d, d) * 0.01
        )
        self.B_bilinear = nn.Parameter(
            torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

        if self.use_low_rank:
            r = min(self.N_rank, d)
            self.N_P = nn.Parameter(
                torch.randn(self.u_dim, d, r) * (1.0 / math.sqrt(d))
            )
            self.N_Q = nn.Parameter(
                torch.randn(self.u_dim, d, r) * (1.0 / math.sqrt(d))
            )
        else:
            self.N_bilinear = nn.Parameter(
                torch.randn(self.u_dim, d, d) * (0.01 / math.sqrt(d))
            )

        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _bilinear_term(self, z_dyn, ut_dt):
        if self.use_low_rank:
            N = torch.bmm(self.N_P, self.N_Q.transpose(1, 2))
        else:
            N = self.N_bilinear
        return torch.einsum('mij,bj,bm->bi', N, z_dyn, ut_dt)

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        z_linear = torch.mm(z_dyn, self.A_bilinear.T) + \
                   torch.mm(ut_dt, self.B_bilinear.T)
        z_bilinear = self._bilinear_term(z_dyn, ut_dt)
        z_dyn_next = z_linear + z_bilinear
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k
