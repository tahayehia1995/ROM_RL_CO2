"""
FNO-based Embed-to-Control Architecture
==========================================
Two-branch FNO encoder that separates static spatial properties (porosity,
permeability) from dynamic spatial properties (saturation, pressure),
with approximately-invertible spectral convolution blocks.

Architecture:
    Branch 1 (Static):  PERMI + POROS -> FNOEncoder -> z_static
    Branch 2 (Dynamic): SG + PRES    -> FNOEncoder -> z_dynamic

    Transition operates on z_dynamic ONLY, conditioned on z_static.
    Decoder: FNODecoder(concat(z_static, z_dynamic)) -> dynamic channels.

The invertible spectral blocks follow FINE (arXiv:2505.15329): each layer
is y = x + alpha*F(x), making encode(decode(z)) ~ z by construction.
An invertibility loss reinforces this during training.

Drop-in replacement for MultimodalMSE2C with the same interface contract.
"""

import torch
import torch.nn as nn
import importlib

from .fno_encoder import FNOEncoder
from .fno_decoder import FNODecoder
from model.models.transition_utils import create_trans_encoder
from model.losses.spatial_enhancements import Discriminator3D
from model.utils.initialization import weights_init
from model.utils import win_long_path


class _ConditionedLinearTransition(nn.Module):
    """Fallback conditioned linear transition (same as multimodal's default)."""

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


class FNOE2C(nn.Module):
    """FNO-based Embed-to-Control model.

    Interface-compatible with MultimodalMSE2C and GNNE2C.
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

        mm_cfg = config.config.get('multimodal', {})
        self.static_channels = mm_cfg.get('static_channels', [2, 3])
        self.dynamic_channels = mm_cfg.get('dynamic_channels', [0, 1])
        self.static_latent_dim = mm_cfg.get('static_latent_dim', 32)
        self.dynamic_latent_dim = mm_cfg.get('dynamic_latent_dim', 96)
        latent_dim = config['model']['latent_dim']

        if self.static_latent_dim + self.dynamic_latent_dim != latent_dim:
            raise ValueError(
                f"static_latent_dim ({self.static_latent_dim}) + "
                f"dynamic_latent_dim ({self.dynamic_latent_dim}) must equal "
                f"latent_dim ({latent_dim})"
            )

        spatial_shape = tuple(config['data']['input_shape'][1:])
        self._spatial_shape = spatial_shape

        self._cached_static_input = None
        self._cached_z_static = None

        self._build_model(spatial_shape)

    def _build_model(self, spatial_shape):
        n_static = len(self.static_channels)
        n_dynamic = len(self.dynamic_channels)
        latent_dim = self.config['model']['latent_dim']
        cfg_dict = self.config.config

        self.static_encoder = FNOEncoder(
            n_static, self.static_latent_dim, cfg_dict)
        self.dynamic_encoder = FNOEncoder(
            n_dynamic, self.dynamic_latent_dim, cfg_dict)

        self.fno_encoder = True  # sentinel for dashboard detection

        self.decoder = FNODecoder(
            n_dynamic, latent_dim, spatial_shape, cfg_dict)

        # --- conditioned transition ---
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
            print(f"[FNO-E2C] static_channels={self.static_channels}, "
                  f"dynamic_channels={self.dynamic_channels}")
            print(f"[FNO-E2C] z_static={self.static_latent_dim}, "
                  f"z_dynamic={self.dynamic_latent_dim}, total={self.config['model']['latent_dim']}")
            print(f"[FNO-E2C] Transition: {transition_type} on z_dynamic "
                  f"conditioned on z_static")

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _encode_static_cached(self, x_static):
        """Encode static channels with caching. Cache is bypassed during training."""
        if self.training:
            z_static, _, _ = self.static_encoder(x_static)
            self._cached_static_input = None
            self._cached_z_static = None
            return z_static

        if (self._cached_z_static is not None
                and self._cached_static_input is not None
                and self._cached_static_input.shape == x_static.shape
                and torch.equal(self._cached_static_input, x_static)):
            return self._cached_z_static

        z_static, _, _ = self.static_encoder(x_static)
        self._cached_static_input = x_static.detach().clone()
        self._cached_z_static = z_static.detach()
        return z_static

    def _split_channels(self, x):
        x_static = x[:, self.static_channels, :, :, :]
        x_dynamic = x[:, self.dynamic_channels, :, :, :]
        return x_static, x_dynamic

    def _concat_for_decoder(self, z_static, z_dynamic):
        return torch.cat([z_static, z_dynamic], dim=-1)

    def _reassemble(self, x_dynamic_pred, x_static_ref):
        batch = x_dynamic_pred.shape[0]
        total_ch = len(self.static_channels) + len(self.dynamic_channels)
        spatial = x_dynamic_pred.shape[2:]
        full = x_dynamic_pred.new_zeros(batch, total_ch, *spatial)
        for i, ch in enumerate(self.dynamic_channels):
            full[:, ch, ...] = x_dynamic_pred[:, i, ...]
        for i, ch in enumerate(self.static_channels):
            full[:, ch, ...] = x_static_ref[:, i, ...]
        return full

    def set_teacher_forcing_ratio(self, ratio):
        self._teacher_forcing_ratio = ratio

    def update_noise_std(self, epoch):
        if self._noise_anneal and self.enable_latent_noise:
            progress = min(epoch / max(self._noise_anneal_epochs, 1), 1.0)
            self._current_noise_std = (
                self._noise_start_std + (self._noise_end_std - self._noise_start_std) * progress
            )

    # --------------------------------------------------------------------- #
    #  Invertibility loss
    # --------------------------------------------------------------------- #
    def get_invertibility_loss(self):
        """Compute ||encode(decode(z)) - z||^2 for the dynamic branch.

        Called by customized_loss.py via hasattr pattern.
        """
        if not self.training or not hasattr(self, '_last_z_dynamic'):
            return None
        z = self._last_z_dynamic.detach()
        z_for_dec = self._concat_for_decoder(
            self._last_z_static.detach(), z)
        with torch.no_grad():
            x_rec = self.decoder(z_for_dec)
        z_re, _, _ = self.dynamic_encoder(x_rec)
        return torch.mean((z_re - z) ** 2)

    # --------------------------------------------------------------------- #
    #  Forward (training)
    # --------------------------------------------------------------------- #
    def forward(self, inputs):
        X, U, Y, dt = inputs

        if len(X) != self.n_steps:
            raise ValueError(f"Expected {self.n_steps} states but got {len(X)}")

        x0_full = X[0]
        x0_static, x0_dynamic = self._split_channels(x0_full)

        z_static = self._encode_static_cached(x0_static)
        z_dynamic_0, mu0_d, logvar0_d = self.dynamic_encoder(x0_dynamic)

        self._last_z_static = z_static
        self._last_z_dynamic = z_dynamic_0

        z_dyn_for_model = z_dynamic_0
        if self.training and self.enable_latent_noise:
            z_dyn_for_model = z_dynamic_0 + self._current_noise_std * torch.randn_like(z_dynamic_0)

        z0_for_dec = self._concat_for_decoder(z_static, z_dyn_for_model)
        x0_rec = self.decoder(z0_for_dec)

        mu_list = [mu0_d] if mu0_d is not None else None
        logvar_list = [logvar0_d] if logvar0_d is not None else None

        X_next_dynamic = []
        for i_step in range(len(X) - 1):
            _, x_dyn = self._split_channels(X[i_step + 1])
            X_next_dynamic.append(x_dyn)

        X_next_pred = []
        Z_next = []
        Z_next_pred = []
        Y_next_pred = []
        Z_re_encoded = []

        if self.enable_cycle_loss and self.training:
            dyn_ch = len(self.dynamic_channels)
            z_dyn_re, _, _ = self.dynamic_encoder(
                x0_rec[:, :dyn_ch] if x0_rec.shape[1] > dyn_ch else x0_rec)
            Z_re_encoded.append(z_dyn_re)

        Z_dyn_pred_list, Y_pred_list = self.transition.forward_nsteps(
            z_dyn_for_model, z_static, dt, U
        )
        Y_next_pred = Y_pred_list

        for i_step in range(len(Z_dyn_pred_list)):
            z_dyn_pred = Z_dyn_pred_list[i_step]

            z_for_dec = self._concat_for_decoder(z_static, z_dyn_pred)
            x_next_pred = self.decoder(z_for_dec)

            use_teacher = (
                not self.enable_scheduled_sampling
                or not self.training
                or torch.rand(1).item() < self._teacher_forcing_ratio
            )
            if use_teacher:
                _, x_dyn_true = self._split_channels(X[i_step + 1])
                z_dyn_true, mu_t, logvar_t = self.dynamic_encoder(x_dyn_true)
            else:
                dyn_ch = len(self.dynamic_channels)
                x_pred_dyn = (x_next_pred[:, :dyn_ch]
                              if x_next_pred.shape[1] > dyn_ch else x_next_pred)
                z_dyn_true, mu_t, logvar_t = self.dynamic_encoder(x_pred_dyn)

            X_next_pred.append(x_next_pred)
            Z_next_pred.append(z_dyn_pred)
            Z_next.append(z_dyn_true)

            if mu_list is not None:
                mu_list.append(mu_t)
                logvar_list.append(logvar_t)

            if self.enable_cycle_loss and self.training:
                dyn_ch = len(self.dynamic_channels)
                x_pred_dyn_ch = (x_next_pred[:, :dyn_ch]
                                 if x_next_pred.shape[1] > dyn_ch else x_next_pred)
                z_re, _, _ = self.dynamic_encoder(x_pred_dyn_ch)
                Z_re_encoded.append(z_re)

        if not Z_re_encoded:
            Z_re_encoded = None

        return (X_next_pred, X_next_dynamic, Z_next_pred, Z_next,
                Y_next_pred, Y, z0_for_dec, x0_dynamic, x0_rec,
                mu_list, logvar_list, Z_re_encoded)

    # --------------------------------------------------------------------- #
    #  Predict (inference, single step)
    # --------------------------------------------------------------------- #
    def predict(self, inputs):
        xt_full, ut, yt, dt = inputs

        x_static, x_dynamic = self._split_channels(xt_full)

        z_static = self._encode_static_cached(x_static)
        z_dynamic, _, _ = self.dynamic_encoder(x_dynamic)

        z_dyn_next, yt_next = self.transition(z_dynamic, z_static, dt, ut)

        z_for_dec = self._concat_for_decoder(z_static, z_dyn_next)
        x_dyn_pred = self.decoder(z_for_dec)

        xt_next_full = self._reassemble(x_dyn_pred, x_static)
        return xt_next_full, yt_next

    def encode_initial(self, xt_full):
        """Encode full spatial state to concatenated latent vector."""
        x_static, x_dynamic = self._split_channels(xt_full)
        z_static = self._encode_static_cached(x_static)
        z_dynamic, _, _ = self.dynamic_encoder(x_dynamic)
        return torch.cat([z_static, z_dynamic], dim=-1)

    def predict_latent(self, zt, dt, ut):
        """Pure latent-space prediction (no spatial encode/decode)."""
        z_static = zt[:, :self.static_latent_dim]
        z_dynamic = zt[:, self.static_latent_dim:]

        z_dyn_next, yt_next = self.transition(z_dynamic, z_static, dt, ut)

        zt_next = torch.cat([z_static, z_dyn_next], dim=-1)
        return zt_next, yt_next

    # --------------------------------------------------------------------- #
    #  Weight I/O
    # --------------------------------------------------------------------- #
    def save_weights_to_file(self, encoder_file, decoder_file, transition_file):
        encoder_payload = {
            'static_encoder': self.static_encoder.state_dict(),
            'dynamic_encoder': self.dynamic_encoder.state_dict(),
            '_fno': True,
        }
        torch.save(encoder_payload, win_long_path(encoder_file))
        torch.save(self.decoder.state_dict(), win_long_path(decoder_file))
        torch.save(self.transition.state_dict(), win_long_path(transition_file))

    def load_weights_from_file(self, encoder_file, decoder_file, transition_file):
        encoder_file = win_long_path(encoder_file)
        decoder_file = win_long_path(decoder_file)
        transition_file = win_long_path(transition_file)

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        encoder_payload = torch.load(encoder_file, map_location=device,
                                     weights_only=False)

        if not isinstance(encoder_payload, dict) or '_fno' not in encoder_payload:
            raise ValueError(
                "Encoder file does not contain FNO weights. "
                "It may have been saved with a different model type."
            )

        self.static_encoder.load_state_dict(encoder_payload['static_encoder'])
        self.dynamic_encoder.load_state_dict(encoder_payload['dynamic_encoder'])
        self.decoder.load_state_dict(
            torch.load(decoder_file, map_location=device, weights_only=False)
        )
        self.transition.load_state_dict(
            torch.load(transition_file, map_location=device, weights_only=False)
        )

    def find_and_load_weights(self, models_dir=None,
                              base_pattern='e2co', specific_pattern=None):
        import os
        import glob
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', 'saved_models')
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
        return {'encoder': encoder_files[0], 'decoder': decoder_files[0],
                'transition': transition_files[0]}
