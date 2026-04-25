"""
Multi-Embedding Multimodal E2C Architecture
============================================
A strict generalization of MultimodalMSE2C and FNOE2C that lets each input
field have its own encoder backbone and (optionally) its own decoder backbone,
while still being trained jointly through a single shared conditioned
transition.

Architecture:
    For each branch b:
        Branch b: x[:, b.channels, ...]  ->  enc_b  ->  z_b  (b.latent_dim)

    Static branches: copy-through at reassembly (no decoder).
    Dynamic branches: decoded back to spatial via their own decoder, which
        consumes concat(z_static_all, z_b_next).

    Transition operates on z_dyn = concat(z_b for b in dynamic_branches),
    conditioned on z_static = concat(z_b for b in static_branches).

    Reassembly: final full-channel output is built by placing each dynamic
    branch's predicted channels at their declared indices and copying static
    channels verbatim from the input.

Encoder/Decoder backbones supported per branch: 'cnn' (Encoder/Decoder) and
'fno' (FNOEncoder/FNODecoder). Easily extensible (e.g. 'gnn') by adding a
case to _build_branch_encoder / _build_branch_decoder.

Drop-in for MultimodalMSE2C / FNOE2C: same public interface, same forward
tuple shape, same I/O sentinel discipline. Selected by the new config flag
``multi_embedding.enable``.
"""

import os
import glob
import importlib

import torch
import torch.nn as nn

from model.models.encoder import Encoder
from model.models.decoder import Decoder
from model.models.fno.fno_encoder import FNOEncoder
from model.models.fno.fno_decoder import FNODecoder
from model.models.transition_utils import create_trans_encoder
from model.losses.spatial_enhancements import Discriminator3D
from model.utils.initialization import weights_init
from model.utils import win_long_path
from model.utils.realization_cache import RealizationCache

from model.multimodal_approach import (
    _build_branch_config,
    _ConfigProxy,
)


# ---------------------------------------------------------------------------
# Conditioned Transition fallback (identical to MultimodalMSE2C's default)
# ---------------------------------------------------------------------------
class _ConditionedLinearTransition(nn.Module):
    """Linear state-space transition operating on z_dyn, conditioned on z_static."""

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        n_obs = self.num_prob * 2 + self.num_inj

        encoder_hidden_dims = config['transition'].get('encoder_hidden_dims', [200, 200])
        total_input = dynamic_dim + static_dim + 1
        self.trans_encoder = create_trans_encoder(total_input, encoder_hidden_dims)
        self.trans_encoder.apply(weights_init)

        hz_dim = total_input - 1
        self.At_layer = nn.Linear(hz_dim, dynamic_dim * dynamic_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(hz_dim, dynamic_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(hz_dim, n_obs * dynamic_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(hz_dim, n_obs * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def _compute_matrices(self, z_dyn, z_static, dt):
        inp = torch.cat([z_dyn, z_static, dt], dim=-1)
        hz = self.trans_encoder(inp)
        n_obs = self.num_prob * 2 + self.num_inj
        At = self.At_layer(hz).view(-1, self.dynamic_dim, self.dynamic_dim)
        Bt = self.Bt_layer(hz).view(-1, self.dynamic_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, n_obs, self.dynamic_dim)
        Dt = self.Dt_layer(hz).view(-1, n_obs, self.u_dim)
        return At, Bt, Ct, Dt

    def forward(self, z_dyn, z_static, dt, ut):
        At, Bt, Ct, Dt = self._compute_matrices(z_dyn, z_static, dt)
        ut_dt = ut * dt
        z_dyn_next = (torch.bmm(At, z_dyn.unsqueeze(-1)).squeeze(-1)
                      + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))
        yt = (torch.bmm(Ct, z_dyn_next.unsqueeze(-1)).squeeze(-1)
              + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))
        return z_dyn_next, yt

    def forward_nsteps(self, z_dyn, z_static, dt, U):
        Zt_k, Yt_k = [], []
        for ut in U:
            z_dyn, yt = self.forward(z_dyn, z_static, dt, ut)
            Zt_k.append(z_dyn)
            Yt_k.append(yt)
        return Zt_k, Yt_k


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_branch_encoder(cfg, branch_spec, n_channels, latent_dim):
    enc_type = branch_spec['encoder']['type'].lower()
    if enc_type == 'cnn':
        sub = _build_branch_config(cfg, n_channels, latent_dim)
        return Encoder(_ConfigProxy(sub, cfg))
    if enc_type == 'fno':
        return FNOEncoder(n_channels, latent_dim, cfg.config)
    raise ValueError(f"Unknown encoder type: {enc_type}")


def _build_branch_decoder(cfg, branch_spec, n_channels, in_latent_dim, spatial_shape):
    dec_spec = branch_spec.get('decoder')
    if dec_spec is None:
        return None
    dec_type = dec_spec['type'].lower()
    if dec_type == 'cnn':
        sub = _build_branch_config(cfg, n_channels, in_latent_dim)
        return Decoder(_ConfigProxy(sub, cfg))
    if dec_type == 'fno':
        return FNODecoder(n_channels, in_latent_dim, spatial_shape, cfg.config)
    raise ValueError(f"Unknown decoder type: {dec_type}")


# ---------------------------------------------------------------------------
# MultiEmbeddingMultimodal
# ---------------------------------------------------------------------------
class MultiEmbeddingMultimodal(nn.Module):
    """Per-field encoder/decoder backbones with one shared conditioned transition.

    Drop-in replacement for ``MSE2C`` / ``MultimodalMSE2C`` / ``FNOE2C``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_steps = config['training']['nsteps']

        self.enable_cycle_loss = config['loss'].get('enable_cycle_loss', False)
        self.enable_latent_noise = config['training'].get('enable_latent_noise', False)
        self.enable_scheduled_sampling = config['training'].get('enable_scheduled_sampling', False)

        noise_cfg = config['training'].get('latent_noise', {})
        self._noise_std = noise_cfg.get('noise_std', 0.01)
        self._noise_anneal = noise_cfg.get('anneal', False)
        self._noise_start_std = noise_cfg.get('start_std', 0.05)
        self._noise_end_std = noise_cfg.get('end_std', 0.005)
        self._noise_anneal_epochs = noise_cfg.get('anneal_epochs', 100)
        self._current_noise_std = self._noise_start_std if self._noise_anneal else self._noise_std
        self._teacher_forcing_ratio = 1.0

        mem_cfg = config.config.get('multi_embedding', {})
        branches = mem_cfg.get('branches')
        if not branches:
            raise ValueError(
                "MultiEmbeddingMultimodal requires config.multi_embedding.branches "
                "(list of branch specs). None found."
            )

        self.branches = [self._normalize_branch_spec(b) for b in branches]
        self._validate_branches()

        # Group / index bookkeeping
        self.static_branches = [b for b in self.branches if b['role'] == 'static']
        self.dynamic_branches = [b for b in self.branches if b['role'] == 'dynamic']

        self.static_channels = [ch for b in self.static_branches for ch in b['channels']]
        self.dynamic_channels = [ch for b in self.dynamic_branches for ch in b['channels']]
        self.n_total_channels = len(self.static_channels) + len(self.dynamic_channels)

        self.static_latent_dim = sum(b['latent_dim'] for b in self.static_branches)
        self.dynamic_latent_dim = sum(b['latent_dim'] for b in self.dynamic_branches)
        latent_dim = config['model']['latent_dim']
        if self.static_latent_dim + self.dynamic_latent_dim != latent_dim:
            raise ValueError(
                f"Sum of branch latent_dims ({self.static_latent_dim + self.dynamic_latent_dim}) "
                f"must equal config.model.latent_dim ({latent_dim})."
            )

        # Per-branch slice indices into the concatenated dynamic latent.
        self._dyn_slices = {}
        offset = 0
        for b in self.dynamic_branches:
            self._dyn_slices[b['name']] = (offset, offset + b['latent_dim'])
            offset += b['latent_dim']

        # Per-branch slice indices into the concatenated static latent.
        self._static_slices = {}
        offset = 0
        for b in self.static_branches:
            self._static_slices[b['name']] = (offset, offset + b['latent_dim'])
            offset += b['latent_dim']

        spatial_shape = tuple(config['data']['input_shape'][1:])
        self._spatial_shape = spatial_shape

        # Dashboard / hasattr-compat sentinel: existing visualization code
        # checks ``hasattr(model, 'static_encoder')`` to recognize multimodal
        # layouts. Set it to True to take that path.
        self._multi_embedding_sentinel = True

        # Realization-aware multi-entry static cache (replaces the legacy
        # single-entry tensor-identity cache so cycling through multiple
        # realizations during inference no longer thrashes).  Keyed off
        # the static-channel slice of the full input via the shared
        # ``RealizationCache``.
        self._z_static_cache = RealizationCache(maxlen=64)
        # Optional case-index -> realization-id mapping (see
        # ``set_case_to_realization``).  When set, the cache lookup uses
        # the efficient int-key path identical to the GNN; otherwise it
        # falls back to case_index / tensor-fingerprint paths.
        self._case_to_realization = None

        self._build_model(spatial_shape)

    # ------------------------------------------------------------------ #
    #  Branch spec normalization / validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_branch_spec(b):
        """Normalize a branch spec to the canonical nested form.

        Accepted input shapes (any combination):
            - nested:  ``{'encoder': {'type': 'cnn'}, 'decoder': {'type': 'cnn'}}``
            - flat:    ``{'encoder_type': 'cnn', 'decoder_type': 'cnn'}``
              (this is what ``branch_descriptors`` writes out and what the
              YAML config / saved payload typically use).
        """
        spec = dict(b)
        if 'name' not in spec:
            raise ValueError(f"Branch spec missing 'name': {b}")
        if 'channels' not in spec or not isinstance(spec['channels'], (list, tuple)):
            raise ValueError(f"Branch '{spec['name']}' missing list 'channels'")
        spec['channels'] = list(spec['channels'])
        if 'role' not in spec or spec['role'] not in ('static', 'dynamic'):
            raise ValueError(f"Branch '{spec['name']}' must declare role='static' or 'dynamic'")

        # Accept flat encoder_type/decoder_type and lift to nested form.
        enc = spec.get('encoder')
        if isinstance(enc, str):
            enc = {'type': enc}
        if enc is None and 'encoder_type' in spec:
            enc = {'type': spec['encoder_type']}
        if not isinstance(enc, dict) or 'type' not in enc:
            raise ValueError(f"Branch '{spec['name']}' missing encoder.type")
        spec['encoder'] = enc

        dec = spec.get('decoder')
        if isinstance(dec, str):
            dec = {'type': dec}
        if dec is None and 'decoder_type' in spec:
            dt = spec['decoder_type']
            dec = {'type': dt} if dt else None
        spec['decoder'] = dec

        if spec['role'] == 'static':
            if spec.get('decoder') is not None:
                raise ValueError(
                    f"Static branch '{spec['name']}' must have decoder=None "
                    "(static channels are copied through at reassembly)."
                )
        else:
            if spec.get('decoder') is None or 'type' not in spec['decoder']:
                raise ValueError(
                    f"Dynamic branch '{spec['name']}' must declare decoder.type"
                )
        if 'latent_dim' not in spec:
            raise ValueError(f"Branch '{spec['name']}' missing latent_dim")
        return spec

    def _validate_branches(self):
        names = [b['name'] for b in self.branches]
        if len(set(names)) != len(names):
            raise ValueError(f"Branch names must be unique: {names}")
        seen = []
        for b in self.branches:
            seen.extend(b['channels'])
        if len(set(seen)) != len(seen):
            raise ValueError(f"Branch channel sets must not overlap: {seen}")

    # ------------------------------------------------------------------ #
    #  Build encoders / decoders / transition
    # ------------------------------------------------------------------ #
    def _build_model(self, spatial_shape):
        latent_dim = self.config['model']['latent_dim']

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for b in self.branches:
            n_ch = len(b['channels'])
            enc = _build_branch_encoder(self.config, b, n_ch, b['latent_dim'])
            self.encoders[b['name']] = enc

            if b['role'] == 'dynamic':
                # Each dynamic decoder consumes concat(z_static_all, z_b_next)
                # and outputs only this branch's channels.
                in_latent = self.static_latent_dim + b['latent_dim']
                dec = _build_branch_decoder(self.config, b, n_ch, in_latent, spatial_shape)
                if dec is None:
                    raise ValueError(
                        f"Dynamic branch '{b['name']}' produced None decoder."
                    )
                self.decoders[b['name']] = dec

        # Sentinel attribute for dashboard / visualization detection
        # (mirrors FNOE2C.fno_encoder = True).
        if any(b['encoder']['type'].lower() == 'fno' for b in self.branches):
            self.fno_encoder = True

        # --- shared conditioned transition on z_dyn / z_static ---
        transition_type = self.config['transition'].get('type', 'linear').lower()
        _conditioned_map = {
            'clru': ('model.models.clru_transition', 'ConditionedCLRUTransition'),
            's4d': ('model.models.s4d_transition', 'ConditionedS4DTransition'),
            's4d_dplr': ('model.models.s4d_dplr_transition', 'ConditionedS4DPLRTransition'),
            's5': ('model.models.s5_transition', 'ConditionedS5Transition'),
            'koopman': ('model.models.koopman_transition', 'ConditionedKoopmanTransition'),
            'ct_koopman': ('model.models.ct_koopman_transition', 'ConditionedCTKoopmanTransition'),
            'nonlinear': ('model.models.nonlinear_transition', 'ConditionedNonlinearTransition'),
            'mamba': ('model.models.mamba_transition', 'ConditionedMambaTransition'),
            'mamba2': ('model.models.mamba2_transition', 'ConditionedMamba2Transition'),
            'stable_koopman': ('model.models.stable_koopman_transition', 'ConditionedStableKoopmanTransition'),
            'deep_koopman': ('model.models.deep_koopman_transition', 'ConditionedDeepKoopmanTransition'),
            'gru': ('model.models.gru_transition', 'ConditionedGRUTransition'),
            'lstm': ('model.models.lstm_transition', 'ConditionedLSTMTransition'),
            'hamiltonian': ('model.models.hamiltonian_transition', 'ConditionedHamiltonianTransition'),
            'skolr': ('model.models.skolr_transition', 'ConditionedSKOLRTransition'),
            'ren': ('model.models.ren_transition', 'ConditionedRENTransition'),
            'koopman_aft': ('model.models.koopman_aft_transition', 'ConditionedKoopmanAFTTransition'),
            'dissipative_koopman': ('model.models.dissipative_koopman_transition', 'ConditionedDissipativeKoopmanTransition'),
            'bilinear_koopman': ('model.models.bilinear_koopman_transition', 'ConditionedBilinearKoopmanTransition'),
            'isfno': ('model.models.isfno_transition', 'ConditionedISFNOTransition'),
            'sindy': ('model.models.sindy_transition', 'ConditionedSINDyTransition'),
            'neural_cde': ('model.models.neural_cde_transition', 'ConditionedNeuralCDETransition'),
            'latent_sde': ('model.models.latent_sde_transition', 'ConditionedLatentSDETransition'),
            'transformer': ('model.models.transformer_transition', 'ConditionedTransformerTransition'),
            'deeponet': ('model.models.deeponet_transition', 'ConditionedDeepONetTransition'),
        }
        if transition_type in _conditioned_map:
            mod_path, cls_name = _conditioned_map[transition_type]
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            self.transition = cls(self.config, self.dynamic_latent_dim, self.static_latent_dim)
        else:
            self.transition = _ConditionedLinearTransition(
                self.config, self.dynamic_latent_dim, self.static_latent_dim
            )
        self.transition_mode = 'latent'

        if getattr(self.config.loss, 'enable_adversarial_loss', False):
            self.discriminator = Discriminator3D(self.config)
        else:
            self.discriminator = None

        if getattr(self.config.runtime, 'verbose', False):
            print("[MEM-E2C] branches:")
            for b in self.branches:
                dec_t = (b['decoder']['type'] if b.get('decoder') else 'none')
                print(f"   - {b['name']:>10s} role={b['role']:<7s} ch={b['channels']} "
                      f"enc={b['encoder']['type']} dec={dec_t} z={b['latent_dim']}")
            print(f"[MEM-E2C] z_static={self.static_latent_dim}, "
                  f"z_dynamic={self.dynamic_latent_dim}, total={latent_dim}")
            print(f"[MEM-E2C] Transition: {transition_type} on z_dynamic "
                  f"conditioned on z_static")

    # ------------------------------------------------------------------ #
    #  Public introspection
    # ------------------------------------------------------------------ #
    @property
    def branch_descriptors(self):
        out = []
        for b in self.branches:
            out.append({
                'name': b['name'],
                'role': b['role'],
                'encoder_type': b['encoder']['type'],
                'decoder_type': (b['decoder']['type'] if b.get('decoder') else None),
                'channels': list(b['channels']),
                'latent_dim': b['latent_dim'],
            })
        return out

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _slice_branch(self, x, branch):
        return x[:, branch['channels'], :, :, :]

    def _encode_branch(self, branch, x_branch):
        return self.encoders[branch['name']](x_branch)

    def set_case_to_realization(self, mapping):
        """Install a precomputed case-index -> realization-id lookup so
        the static cache can use the efficient int-key path (matching
        the GNN's behaviour).

        Pass ``None`` to clear and revert to the case-index /
        tensor-fingerprint fallbacks.
        """
        self._case_to_realization = mapping
        self._z_static_cache.clear()

    def _encode_static_cached(self, x_full, case_indices=None):
        """Encode every static branch from the full input and concat in declaration order.

        Uses the multi-entry realization cache so multiple distinct
        realizations can be served simultaneously without thrashing.
        Cache key derivation prefers ``case_indices`` (+ optional
        ``case_to_realization`` mapping) and falls back to per-sample
        tensor fingerprinting on the static-channel slice when
        ``case_indices`` are not available.
        """
        # No static branches: nothing to encode or cache.
        if not self.static_branches:
            return x_full.new_zeros(x_full.shape[0], 0)

        if self.training:
            self._z_static_cache.clear()
            parts = []
            for b in self.static_branches:
                z_b, _, _ = self._encode_branch(b, self._slice_branch(x_full, b))
                parts.append(z_b)
            return torch.cat(parts, dim=-1)

        # Inference: derive keys via the unified helper.  When
        # ``case_indices`` are provided we skip the costly tensor hash
        # entirely; the fallback hashes only the static-channel slice
        # so dynamic content does not pollute the cache key.
        static_chs = [ch for b in self.static_branches for ch in b['channels']]
        x_static_slice = x_full[:, static_chs, ...]
        keys = self._z_static_cache.derive_keys(
            x_static=x_static_slice,
            case_indices=case_indices,
            case_to_realization=self._case_to_realization,
            batch_size=x_full.shape[0],
        )
        z_static, miss_idx, miss_keys = self._z_static_cache.lookup(
            keys, self.static_latent_dim, x_full.device, dtype=x_full.dtype
        )
        if miss_idx.numel() > 0:
            x_miss = x_full.index_select(0, miss_idx)
            parts = []
            for b in self.static_branches:
                z_b, _, _ = self._encode_branch(b, self._slice_branch(x_miss, b))
                parts.append(z_b)
            z_miss = torch.cat(parts, dim=-1)
            z_static.index_copy_(0, miss_idx, z_miss)
            self._z_static_cache.store(miss_keys, z_miss)
        return z_static

    def _encode_dynamic(self, x_full):
        """Encode every dynamic branch and concat in declaration order.

        Returns:
            z_dyn (B, dynamic_latent_dim), mu_dyn or None, logvar_dyn or None.
        """
        z_parts, mu_parts, logvar_parts = [], [], []
        any_vae = False
        for b in self.dynamic_branches:
            z_b, mu_b, logvar_b = self._encode_branch(b, self._slice_branch(x_full, b))
            z_parts.append(z_b)
            if mu_b is not None and logvar_b is not None:
                any_vae = True
                mu_parts.append(mu_b)
                logvar_parts.append(logvar_b)
            else:
                mu_parts.append(z_b)
                logvar_parts.append(None)
        z_dyn = torch.cat(z_parts, dim=-1)
        if any_vae and all(lv is not None for lv in logvar_parts):
            return z_dyn, torch.cat(mu_parts, dim=-1), torch.cat(logvar_parts, dim=-1)
        return z_dyn, None, None

    def _decode_dynamic(self, z_static, z_dyn):
        """Decode each dynamic branch independently, then assemble dynamic-only tensor.

        Returns:
            Dict {branch_name: decoded tensor (B, len(b.channels), Nx, Ny, Nz)}.
        """
        out = {}
        for b in self.dynamic_branches:
            lo, hi = self._dyn_slices[b['name']]
            z_b = z_dyn[:, lo:hi]
            z_in = torch.cat([z_static, z_b], dim=-1)
            out[b['name']] = self.decoders[b['name']](z_in)
        return out

    def _reassemble(self, dyn_preds_by_branch, x_full_input):
        """Build a full (n_total_channels) tensor from per-branch decoded slices
        plus static channels copied verbatim from the input.
        """
        batch = x_full_input.shape[0]
        spatial = x_full_input.shape[2:]
        full = x_full_input.new_zeros(batch, self.n_total_channels, *spatial)
        for b in self.dynamic_branches:
            pred = dyn_preds_by_branch[b['name']]
            for i, ch in enumerate(b['channels']):
                full[:, ch, ...] = pred[:, i, ...]
        for b in self.static_branches:
            for ch in b['channels']:
                full[:, ch, ...] = x_full_input[:, ch, ...]
        return full

    def _split_dynamic_channels_from_full(self, x_full):
        """Extract dynamic channels from a full-channel tensor in branch declaration order."""
        chs = self.dynamic_channels
        return x_full[:, chs, :, :, :]

    def _re_encode_dynamic_from_full(self, x_full):
        """Re-encode each dynamic branch's channels from a full-channel tensor and concat.

        Used for cycle-consistency loss.
        """
        z_parts = []
        for b in self.dynamic_branches:
            x_b = self._slice_branch(x_full, b)
            z_b, _, _ = self._encode_branch(b, x_b)
            z_parts.append(z_b)
        return torch.cat(z_parts, dim=-1)

    # ------------------------------------------------------------------ #
    #  Public training / inference hooks
    # ------------------------------------------------------------------ #
    def set_teacher_forcing_ratio(self, ratio):
        self._teacher_forcing_ratio = ratio

    def update_noise_std(self, epoch):
        if self._noise_anneal and self.enable_latent_noise:
            progress = min(epoch / max(self._noise_anneal_epochs, 1), 1.0)
            self._current_noise_std = (
                self._noise_start_std + (self._noise_end_std - self._noise_start_std) * progress
            )

    # ------------------------------------------------------------------ #
    #  Invertibility loss (FNO branches only)
    # ------------------------------------------------------------------ #
    def get_invertibility_loss(self):
        """Return scalar = sum_{b: enc=fno AND dec=fno} || enc_b(dec_b(z)) - z ||^2.

        Called by ``customized_loss.py`` via the ``hasattr`` integration. The
        function is evaluated only during training; otherwise returns None.
        """
        if not self.training or not hasattr(self, '_last_z_dyn'):
            return None

        z_dyn = self._last_z_dyn.detach()
        z_static = self._last_z_static.detach()

        total = None
        for b in self.dynamic_branches:
            if b['encoder']['type'].lower() != 'fno':
                continue
            if b.get('decoder') is None or b['decoder']['type'].lower() != 'fno':
                continue
            lo, hi = self._dyn_slices[b['name']]
            z_b = z_dyn[:, lo:hi]
            z_in = torch.cat([z_static, z_b], dim=-1)
            with torch.no_grad():
                x_rec = self.decoders[b['name']](z_in)
            z_re, _, _ = self.encoders[b['name']](x_rec)
            term = torch.mean((z_re - z_b) ** 2)
            total = term if total is None else total + term

        return total

    # ------------------------------------------------------------------ #
    #  Forward (training)
    # ------------------------------------------------------------------ #
    def forward(self, inputs):
        # Inputs may carry an optional ``case_indices`` 5-tuple element
        # so the static cache can use the realization-keyed fast path.
        if len(inputs) == 5:
            X, U, Y, dt, case_indices = inputs
        else:
            X, U, Y, dt = inputs
            case_indices = None

        if len(X) != self.n_steps:
            raise ValueError(f"Expected {self.n_steps} states but got {len(X)}")
        if len(U) != self.n_steps - 1:
            raise ValueError(f"Expected {self.n_steps - 1} controls but got {len(U)}")
        if len(Y) != self.n_steps - 1:
            raise ValueError(f"Expected {self.n_steps - 1} observations but got {len(Y)}")

        x0_full = X[0]

        z_static = self._encode_static_cached(x0_full, case_indices=case_indices)
        z_dyn_0, mu0_d, logvar0_d = self._encode_dynamic(x0_full)

        # cache for invertibility loss
        self._last_z_static = z_static
        self._last_z_dyn = z_dyn_0

        z_dyn_for_model = z_dyn_0
        if self.training and self.enable_latent_noise:
            z_dyn_for_model = z_dyn_0 + self._current_noise_std * torch.randn_like(z_dyn_0)

        # Initial reconstruction (assembled full-channel)
        dyn_preds_0 = self._decode_dynamic(z_static, z_dyn_for_model)
        x0_rec_full = self._reassemble(dyn_preds_0, x0_full)

        # z0 returned to loss for L2-reg etc.
        z0_for_dec = torch.cat([z_static, z_dyn_for_model], dim=-1)

        mu_list = [mu0_d] if mu0_d is not None else None
        logvar_list = [logvar0_d] if logvar0_d is not None else None

        X_next_full_targets = [X[i + 1] for i in range(len(X) - 1)]

        X_next_pred = []
        Z_next = []
        Z_next_pred = []
        Z_re_encoded = []

        if self.enable_cycle_loss and self.training:
            z_re_0 = self._re_encode_dynamic_from_full(x0_rec_full)
            Z_re_encoded.append(z_re_0)

        Z_dyn_pred_list, Y_pred_list = self.transition.forward_nsteps(
            z_dyn_for_model, z_static, dt, U
        )
        Y_next_pred = Y_pred_list

        for i_step in range(len(Z_dyn_pred_list)):
            z_dyn_pred = Z_dyn_pred_list[i_step]

            dyn_preds = self._decode_dynamic(z_static, z_dyn_pred)
            x_next_pred_full = self._reassemble(dyn_preds, x0_full)

            use_teacher = (
                not self.enable_scheduled_sampling
                or not self.training
                or torch.rand(1).item() < self._teacher_forcing_ratio
            )
            if use_teacher:
                z_dyn_true, mu_t, logvar_t = self._encode_dynamic(X[i_step + 1])
            else:
                z_dyn_true, mu_t, logvar_t = self._encode_dynamic(x_next_pred_full)

            X_next_pred.append(x_next_pred_full)
            Z_next_pred.append(z_dyn_pred)
            Z_next.append(z_dyn_true)

            if mu_list is not None:
                mu_list.append(mu_t)
                logvar_list.append(logvar_t)

            if self.enable_cycle_loss and self.training:
                z_re = self._re_encode_dynamic_from_full(x_next_pred_full)
                Z_re_encoded.append(z_re)

        if not Z_re_encoded:
            Z_re_encoded = None

        # X_next: full ground-truth tensors (matches MSE2C-style 4-channel
        # reconstruction loss; static channels contribute 0 by construction).
        return (X_next_pred, X_next_full_targets, Z_next_pred, Z_next,
                Y_next_pred, Y, z0_for_dec, x0_full, x0_rec_full,
                mu_list, logvar_list, Z_re_encoded)

    # ------------------------------------------------------------------ #
    #  Predict (inference, single step)
    # ------------------------------------------------------------------ #
    def predict(self, inputs):
        # Accept either 4-tuple or 5-tuple with case_indices.
        if len(inputs) == 5:
            xt_full, ut, yt, dt, case_indices = inputs
        else:
            xt_full, ut, yt, dt = inputs
            case_indices = None

        z_static = self._encode_static_cached(xt_full, case_indices=case_indices)
        z_dyn, _, _ = self._encode_dynamic(xt_full)

        z_dyn_next, yt_next = self.transition(z_dyn, z_static, dt, ut)

        dyn_preds = self._decode_dynamic(z_static, z_dyn_next)
        xt_next_full = self._reassemble(dyn_preds, xt_full)
        return xt_next_full, yt_next

    def encode_initial(self, xt_full, case_indices=None):
        """Encode a full-channel state into the concatenated latent.

        ``case_indices`` is optional and, when supplied, lets the static
        cache use the efficient int-key fast path.

        Returns:
            (B, static_latent_dim + dynamic_latent_dim) tensor.
        """
        z_static = self._encode_static_cached(xt_full, case_indices=case_indices)
        z_dyn, _, _ = self._encode_dynamic(xt_full)
        return torch.cat([z_static, z_dyn], dim=-1)

    def predict_latent(self, zt, dt, ut):
        """Pure latent-space prediction (no spatial encode/decode)."""
        z_static = zt[:, :self.static_latent_dim]
        z_dyn = zt[:, self.static_latent_dim:]
        z_dyn_next, yt_next = self.transition(z_dyn, z_static, dt, ut)
        zt_next = torch.cat([z_static, z_dyn_next], dim=-1)
        return zt_next, yt_next

    def decode_latent_to_spatial(self, zt):
        """Decode a concatenated latent back to a full-channel spatial tensor.

        Static channels are reconstructed by routing each static branch's
        latent slice back through... well, they can't be (no static decoder).
        This helper therefore requires the original full input for the static
        copy-through. For pure-latent rollouts where no input is available,
        callers should keep a reference to the initial spatial state and pass
        it as ``x_static_ref``.

        Use ``decode_latent_with_static_ref`` instead for that case.
        """
        raise NotImplementedError(
            "MultiEmbeddingMultimodal cannot decode static channels from latent "
            "(static branches have no decoder by design). Use "
            "decode_latent_with_static_ref(zt, x_static_ref) instead."
        )

    def decode_latent_with_static_ref(self, zt, x_full_ref):
        """Decode a concatenated latent back to a full-channel spatial tensor,
        copying static channels from ``x_full_ref``.
        """
        z_static = zt[:, :self.static_latent_dim]
        z_dyn = zt[:, self.static_latent_dim:]
        dyn_preds = self._decode_dynamic(z_static, z_dyn)
        return self._reassemble(dyn_preds, x_full_ref)

    # ------------------------------------------------------------------ #
    #  Weight I/O
    # ------------------------------------------------------------------ #
    def save_weights_to_file(self, encoder_file, decoder_file, transition_file):
        encoder_payload = {
            '_multi_embedding': True,
            'branches': self.branch_descriptors,
            'encoders': {name: enc.state_dict() for name, enc in self.encoders.items()},
        }
        decoder_payload = {
            '_multi_embedding': True,
            'branches': self.branch_descriptors,
            'decoders': {name: dec.state_dict() for name, dec in self.decoders.items()},
        }
        torch.save(encoder_payload, win_long_path(encoder_file))
        torch.save(decoder_payload, win_long_path(decoder_file))
        torch.save(self.transition.state_dict(), win_long_path(transition_file))

    def load_weights_from_file(self, encoder_file, decoder_file, transition_file):
        # Backward-compatible full loader; delegates to the per-component
        # methods so the warm-start path uses identical validation.
        self._load_encoder_payload(encoder_file)
        self._load_decoder_payload(decoder_file)
        self._load_transition_payload(transition_file)

    # --- Per-component loaders (used by warm-start paths) ---
    @staticmethod
    def _torch_device():
        return torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    def _load_encoder_payload(self, encoder_file):
        encoder_file = win_long_path(encoder_file)
        device = self._torch_device()
        encoder_payload = torch.load(
            encoder_file, map_location=device, weights_only=False
        )
        if not isinstance(encoder_payload, dict) or not encoder_payload.get('_multi_embedding'):
            raise ValueError(
                f"Encoder file {encoder_file} is not a multi_embedding "
                f"payload (missing '_multi_embedding' sentinel)."
            )
        encoders_sd = encoder_payload.get('encoders', {})
        for name, enc in self.encoders.items():
            if name not in encoders_sd:
                raise KeyError(
                    f"MultiEmbedding encoder payload {encoder_file} missing "
                    f"branch '{name}'."
                )
            try:
                enc.load_state_dict(encoders_sd[name])
            except RuntimeError as e:
                raise RuntimeError(
                    f"MultiEmbedding encoder branch '{name}' weights in "
                    f"{encoder_file} do not match the current model "
                    f"architecture: {e}"
                ) from e

    def _load_decoder_payload(self, decoder_file):
        decoder_file = win_long_path(decoder_file)
        device = self._torch_device()
        decoder_payload = torch.load(
            decoder_file, map_location=device, weights_only=False
        )
        if not (isinstance(decoder_payload, dict) and decoder_payload.get('_multi_embedding')):
            raise ValueError(
                f"Decoder file {decoder_file} is not a multi_embedding "
                f"payload (missing '_multi_embedding' sentinel)."
            )
        decoders_sd = decoder_payload.get('decoders', {})
        for name, dec in self.decoders.items():
            if name not in decoders_sd:
                raise KeyError(
                    f"MultiEmbedding decoder payload {decoder_file} missing "
                    f"branch '{name}'."
                )
            try:
                dec.load_state_dict(decoders_sd[name])
            except RuntimeError as e:
                raise RuntimeError(
                    f"MultiEmbedding decoder branch '{name}' weights in "
                    f"{decoder_file} do not match the current model "
                    f"architecture: {e}"
                ) from e

    def _load_transition_payload(self, transition_file):
        transition_file = win_long_path(transition_file)
        device = self._torch_device()
        try:
            self.transition.load_state_dict(
                torch.load(transition_file, map_location=device, weights_only=False)
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"MultiEmbedding transition weights in {transition_file} do "
                f"not match the current transition architecture: {e}"
            ) from e

    def find_and_load_weights(self, models_dir=None,
                              base_pattern='e2co', specific_pattern=None):
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'saved_models'
            )
        encoder_files = sorted(glob.glob(os.path.join(models_dir, f"{base_pattern}_encoder*.h5")))
        decoder_files = sorted(glob.glob(os.path.join(models_dir, f"{base_pattern}_decoder*.h5")))
        transition_files = sorted(glob.glob(os.path.join(models_dir, f"{base_pattern}_transition*.h5")))

        if not encoder_files or not decoder_files or not transition_files:
            raise FileNotFoundError(f"No matching model files found in {models_dir}")

        if specific_pattern:
            encoder_files = [f for f in encoder_files if specific_pattern in f]
            decoder_files = [f for f in decoder_files if specific_pattern in f]
            transition_files = [f for f in transition_files if specific_pattern in f]

        self.load_weights_from_file(encoder_files[0], decoder_files[0], transition_files[0])
        return {
            'encoder': encoder_files[0],
            'decoder': decoder_files[0],
            'transition': transition_files[0],
        }
