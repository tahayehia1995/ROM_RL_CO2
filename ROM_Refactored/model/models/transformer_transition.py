"""
Transformer dynamics transition for E2C architecture
=====================================================
Uses multi-head self-attention over a temporal window of latent states
to predict the next state.  Unlike Markov models (where z_{t+1} depends
only on z_t), the transformer can attend to any past state in the
window, capturing long-range temporal dependencies.

State equation:
    tokens = [embed(z_{t-w}), ..., embed(z_t)]   with positional encoding
    h = TransformerEncoder(tokens, causal_mask)
    z_{t+1} = projection(h[-1])                   (last position output)

Observation equation:
    y_{t+1} = C z_{t+1} + D (u_t * dt)

In single-step forward() without history, falls back to a linear
projection.  The window buffer is populated during forward_nsteps().

Reference: Geneva & Zabaras, "Transformers for Modeling Physical
           Systems" (2022); LatentTSF (2025).
"""

import math
import torch
import torch.nn as nn
from model.utils.initialization import weights_init


class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pos_encoding', pe.unsqueeze(0))

    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        return x + self.pos_encoding[:, :x.size(1), :]


# ===================================================================
# Standard Transformer Dynamics
# ===================================================================

class TransformerTransitionModel(nn.Module):
    """Temporal transformer transition with self-attention over latent window."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_y = self.num_prob * 2 + self.num_inj

        tc = config['transition'].get('transformer', {})
        self.n_heads = tc.get('n_heads', 4)
        n_layers = tc.get('n_layers', 2)
        self.d_model = tc.get('d_model', 128)
        self.window_size = tc.get('window_size', 5)
        dropout = tc.get('dropout', 0.1)

        d = self.latent_dim
        token_dim = d + self.u_dim

        self.input_proj = nn.Linear(token_dim, self.d_model)
        self.input_proj.apply(weights_init)

        self.pos_encoding = _PositionalEncoding(self.d_model, max_len=self.window_size + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.output_proj = nn.Linear(self.d_model, d)
        self.output_proj.apply(weights_init)

        self.fallback_linear = nn.Linear(token_dim, d)
        self.fallback_linear.apply(weights_init)

        self.C = nn.Parameter(torch.randn(self.n_y, d) * (1.0 / math.sqrt(d)))
        self.D = nn.Parameter(
            torch.randn(self.n_y, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _predict_from_window(self, window_tokens):
        """window_tokens: [batch, seq_len, token_dim] -> z_next: [batch, d]."""
        x = self.input_proj(window_tokens)
        x = self.pos_encoding(x)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        h = self.temporal_transformer(x, mask=mask)
        return self.output_proj(h[:, -1, :])

    def forward(self, zt, dt, ut):
        ut_dt = ut * dt
        token = torch.cat([zt, ut_dt], dim=-1)
        zt1 = zt + self.fallback_linear(token)
        yt1 = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
        return zt1, yt1

    def forward_nsteps(self, zt, dt, U):
        Zt_k, Yt_k = [], []
        buffer = []

        for ut in U:
            ut_dt = ut * dt
            token = torch.cat([zt, ut_dt], dim=-1)
            buffer.append(token)

            if len(buffer) >= 2:
                window = torch.stack(buffer[-self.window_size:], dim=1)
                zt1 = self._predict_from_window(window)
            else:
                zt1 = zt + self.fallback_linear(token)

            yt = torch.mm(zt1, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(zt1)
            Yt_k.append(yt)
            zt = zt1

        return Zt_k, Yt_k


# ===================================================================
# Conditioned Transformer Dynamics
# ===================================================================

class ConditionedTransformerTransition(nn.Module):
    """Transformer dynamics for split (z_dynamic, z_static) latent spaces."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_obs = self.num_prob * 2 + self.num_inj

        tc = config['transition'].get('transformer', {})
        self.n_heads = tc.get('n_heads', 4)
        n_layers = tc.get('n_layers', 2)
        self.d_model = tc.get('d_model', 128)
        self.window_size = tc.get('window_size', 5)
        dropout = tc.get('dropout', 0.1)

        d = dynamic_dim
        token_dim = d + self.u_dim + static_dim

        self.input_proj = nn.Linear(token_dim, self.d_model)
        self.input_proj.apply(weights_init)

        self.pos_encoding = _PositionalEncoding(self.d_model, max_len=self.window_size + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.output_proj = nn.Linear(self.d_model, d)
        self.output_proj.apply(weights_init)

        self.fallback_linear = nn.Linear(token_dim, d)
        self.fallback_linear.apply(weights_init)

        self.C = nn.Parameter(
            torch.randn(self.n_obs, d) * (1.0 / math.sqrt(d))
        )
        self.D = nn.Parameter(
            torch.randn(self.n_obs, self.u_dim) * (1.0 / math.sqrt(self.u_dim))
        )

    def _predict_from_window(self, window_tokens):
        x = self.input_proj(window_tokens)
        x = self.pos_encoding(x)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        h = self.temporal_transformer(x, mask=mask)
        return self.output_proj(h[:, -1, :])

    def forward(self, z_dyn, z_static, dt, ut):
        ut_dt = ut * dt
        token = torch.cat([z_dyn, ut_dt, z_static], dim=-1)
        z_dyn_next = z_dyn + self.fallback_linear(token)
        yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        buffer = []

        for ut in U:
            ut_dt = ut * dt
            token = torch.cat([z_dyn, ut_dt, z_static], dim=-1)
            buffer.append(token)

            if len(buffer) >= 2:
                window = torch.stack(buffer[-self.window_size:], dim=1)
                z_dyn_next = self._predict_from_window(window)
            else:
                z_dyn_next = z_dyn + self.fallback_linear(token)

            yt = torch.mm(z_dyn_next, self.C.T) + torch.mm(ut_dt, self.D.T)
            Zt_k.append(z_dyn_next)
            Yt_k.append(yt)
            z_dyn = z_dyn_next

        return Zt_k, Yt_k
