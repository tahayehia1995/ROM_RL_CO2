"""
Koopman + AFT (Attention-Free Transformer) residual correction for E2C
======================================================================
Standard global Koopman transition augmented with a lightweight
attention-free latent memory block that produces corrective residuals
from a sliding window of recent latent states.

State equation:
    z_linear = K z_t + L (u_t * dt)
    residual = AFT([z_{t-w}, ..., z_t])
    z_{t+1}  = z_linear + residual

AFT mechanism (linear complexity in window size):
    Q = W_q z_t                    [batch, aft_dim]
    K_i = W_k z_i   for i in window
    V_i = W_v z_i   for i in window
    residual = W_o * sigmoid(Q) * (sum exp(K_i)*V_i) / (sum exp(K_i))

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

In single-step forward() where no history exists, falls back to
pure Koopman (no residual).  The memory is populated during
forward_nsteps() rollouts.

Reference: "Learning the Koopman Operator using Attention Free
            Transformers", OpenReview 2025.
            github.com/free-laboratory/koopman-residual
"""

import math
import torch
import torch.nn as nn


class _AFTBlock(nn.Module):
    """Attention-Free Transformer memory block operating on a latent window."""

    def __init__(self, latent_dim, aft_dim):
        super().__init__()
        self.aft_dim = aft_dim
        self.aft_W_q = nn.Linear(latent_dim, aft_dim, bias=False)
        self.aft_W_k = nn.Linear(latent_dim, aft_dim, bias=False)
        self.aft_W_v = nn.Linear(latent_dim, aft_dim, bias=False)
        self.aft_W_o = nn.Linear(aft_dim, latent_dim, bias=False)

    def forward(self, z_current, z_window):
        """
        Args:
            z_current: [batch, latent_dim]  -- current latent
            z_window:  list of [batch, latent_dim] -- past latents (len >= 1)
        Returns:
            residual:  [batch, latent_dim]
        """
        Q = self.aft_W_q(z_current)
        sig_Q = torch.sigmoid(Q)

        num = torch.zeros_like(Q)
        den = torch.zeros(Q.size(0), self.aft_dim, device=Q.device) + 1e-8
        for z_i in z_window:
            K_i = self.aft_W_k(z_i)
            V_i = self.aft_W_v(z_i)
            exp_K = torch.exp(K_i - K_i.detach().max())
            num = num + exp_K * V_i
            den = den + exp_K

        context = num / den
        return self.aft_W_o(sig_Q * context)


# ===================================================================
# Standard Koopman + AFT
# ===================================================================

class KoopmanAFTTransitionModel(nn.Module):
    """Koopman with attention-free residual correction from latent memory."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        aft_cfg = config['transition'].get('koopman_aft', {})
        self.window_size = aft_cfg.get('window_size', 3)
        aft_dim = aft_cfg.get('aft_dim', 64)

        d = self.latent_dim

        self.K = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)
        self.L = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

        self.aft = _AFTBlock(d, aft_dim)

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        zt1 = torch.mm(zt, self.K.T) + torch.mm(ut_dt, self.L.T)
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        buffer = [zt]
        Zt_k, Yt_k = [], []
        for ut in U:
            ut_dt = ut * dt
            z_linear = torch.mm(zt, self.K.T) + torch.mm(ut_dt, self.L.T)

            window = buffer[-self.window_size:]
            residual = self.aft(zt, window)
            zt1 = z_linear + residual

            yt = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(zt1)
            Yt_k.append(yt)

            buffer.append(zt1)
            zt = zt1
        return Zt_k, Yt_k


# ===================================================================
# Conditioned Koopman + AFT
# ===================================================================

class ConditionedKoopmanAFTTransition(nn.Module):
    """Koopman+AFT variant for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        aft_cfg = config['transition'].get('koopman_aft', {})
        self.window_size = aft_cfg.get('window_size', 3)
        aft_dim = aft_cfg.get('aft_dim', 64)

        d = dynamic_dim

        self.K = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)
        self.L = nn.Parameter(torch.randn(d, self.u_dim) * (1.0 / math.sqrt(self.u_dim)))

        self.aft = _AFTBlock(d, aft_dim)

        self.C = nn.Parameter(torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        z_dyn_next = torch.mm(z_dyn, self.K.T) + torch.mm(ut_dt, self.L.T)
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        buffer = [z_dyn]
        Zt_k, Yt_k = [], []
        for ut in U:
            ut_dt = ut * dt
            z_linear = torch.mm(z_dyn, self.K.T) + torch.mm(ut_dt, self.L.T)

            window = buffer[-self.window_size:]
            residual = self.aft(z_dyn, window)
            z_dyn_next = z_linear + residual

            yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(z_dyn_next)
            Yt_k.append(yt)

            buffer.append(z_dyn_next)
            z_dyn = z_dyn_next
        return Zt_k, Yt_k
