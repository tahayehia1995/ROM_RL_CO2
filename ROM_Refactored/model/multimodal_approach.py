"""
Multimodal E2C Architecture
============================
Two-branch encoder that separates static spatial properties (porosity,
permeability) from dynamic spatial properties (saturation, pressure).

Architecture:
    Branch 1 (Static):  PERMI + POROS  -> Conv3D pipeline -> z_static
    Branch 2 (Dynamic): SG + PRES     -> Conv3D pipeline -> z_dynamic

    Transition operates on z_dynamic ONLY, conditioned on z_static:
        hz = TransEncoder(concat(z_dynamic, z_static, dt))
        A(dyn_dim x dyn_dim) = A_layer(hz)
        z_dynamic_next = A * z_dynamic + B * (u*dt)

    Decoder receives concat(z_static, z_dynamic) = latent_dim:
        x_pred = Decoder(concat(z_static, z_dynamic_next))

Static channels are never modified by the model during prediction.
"""

import torch
import torch.nn as nn
import copy

from model.models.encoder import Encoder
from model.models.decoder import Decoder
from model.models.transition_utils import create_trans_encoder
from model.losses.spatial_enhancements import Discriminator3D
from model.utils.initialization import weights_init
from model.utils import win_long_path


def _build_branch_config(base_config, n_channels, latent_dim):
    """
    Create a deep-copied config dict adjusted for a branch that processes
    *n_channels* input channels and maps to *latent_dim*.
    """
    cfg = copy.deepcopy(base_config.config)

    cfg['data']['input_shape'] = [n_channels] + list(base_config.config['data']['input_shape'][1:])
    cfg['model']['latent_dim'] = latent_dim

    if 'encoder' in cfg and 'conv_layers' in cfg['encoder']:
        if 'conv1' in cfg['encoder']['conv_layers']:
            conv1 = cfg['encoder']['conv_layers']['conv1']
            if isinstance(conv1, list) and len(conv1) > 0:
                conv1[0] = n_channels

    if 'decoder' in cfg and 'deconv_layers' in cfg['decoder']:
        if 'final_conv' in cfg['decoder']['deconv_layers']:
            fc = cfg['decoder']['deconv_layers']['final_conv']
            if isinstance(fc, list) and len(fc) > 1:
                fc[1] = n_channels

    return cfg


class _ConfigProxy:
    """Lightweight wrapper so a plain dict behaves like the Config object
    that Encoder / Decoder expect."""

    def __init__(self, d, parent_config):
        object.__setattr__(self, '_d', d)
        object.__setattr__(self, '_parent', parent_config)
        object.__setattr__(self, 'config', d)

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getattr__(self, name):
        d = object.__getattribute__(self, '_d')
        if name in d:
            v = d[name]
            if isinstance(v, dict):
                return _DictAttr(v)
            return v
        parent = object.__getattribute__(self, '_parent')
        return getattr(parent, name)


class _DictAttr(dict):
    """Dict subclass with attribute access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# ---------------------------------------------------------------------------
# Conditioned Transition Model
# ---------------------------------------------------------------------------
class ConditionedLinearTransition(nn.Module):
    """Linear state-space transition that operates on z_dynamic only,
    conditioned on z_static.

    trans_encoder input:  concat(z_dynamic, z_static, dt)  → hz
    A(dyn_dim × dyn_dim), B(dyn_dim × u_dim)               from hz
    z_dynamic_next = A * z_dynamic + B * (u * dt)

    Observation equation uses z_dynamic only:
    y = C * z_dynamic + D * (u * dt)
    """

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        n_obs = self.num_prob * 2 + self.num_inj

        encoder_hidden_dims = config['transition'].get('encoder_hidden_dims', [200, 200])

        # Input: z_dynamic + z_static + dt
        total_input = dynamic_dim + static_dim + 1
        self.trans_encoder = create_trans_encoder(total_input, encoder_hidden_dims)
        self.trans_encoder.apply(weights_init)

        # hz output dim = total_input - 1 (convention from create_trans_encoder)
        hz_dim = total_input - 1  # dynamic_dim + static_dim

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
# MultimodalMSE2C
# ---------------------------------------------------------------------------
class MultimodalMSE2C(nn.Module):
    """
    Multimodal Embed-to-Control model.

    Drop-in replacement for ``MSE2C``.  Splits a 4-channel input into
    static (porosity, permeability) and dynamic (saturation, pressure)
    branches.

    Key design:
    - Transition operates on z_dynamic (96-dim), conditioned on z_static (32-dim)
    - Decoder receives concat(z_static, z_dynamic) = 128-dim  (no FC+ReLU fusion)
    - Transition loss is on z_dynamic only (96-dim)
    """

    def __init__(self, config):
        super(MultimodalMSE2C, self).__init__()
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

        self._build_model()

    def _build_model(self):
        n_static = len(self.static_channels)
        n_dynamic = len(self.dynamic_channels)
        latent_dim = self.config['model']['latent_dim']

        # --- static encoder ---
        static_cfg_dict = _build_branch_config(self.config, n_static, self.static_latent_dim)
        static_proxy = _ConfigProxy(static_cfg_dict, self.config)
        self.static_encoder = Encoder(static_proxy)

        # --- dynamic encoder ---
        dyn_cfg_dict = _build_branch_config(self.config, n_dynamic, self.dynamic_latent_dim)
        dyn_proxy = _ConfigProxy(dyn_cfg_dict, self.config)
        self.dynamic_encoder = Encoder(dyn_proxy)

        # --- decoder (takes concat(z_static, z_dynamic) = latent_dim) ---
        dec_cfg_dict = _build_branch_config(self.config, n_dynamic, latent_dim)
        dec_proxy = _ConfigProxy(dec_cfg_dict, self.config)

        decoder_type = self.config['decoder'].get('type', 'standard')
        if decoder_type == 'smooth':
            from model.models.decoder_smooth import DecoderSmooth
            self.decoder = DecoderSmooth(dec_proxy)
        elif decoder_type == 'smooth_generic':
            from model.models.decoder_smooth import DecoderSmoothGeneric
            self.decoder = DecoderSmoothGeneric(dec_proxy)
        else:
            self.decoder = Decoder(dec_proxy)

        # --- conditioned transition (operates on z_dynamic, conditioned on z_static) ---
        transition_type = self.config['transition'].get('type', 'linear')
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
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            self.transition = cls(self.config, self.dynamic_latent_dim, self.static_latent_dim)
        else:
            self.transition = ConditionedLinearTransition(
                self.config, self.dynamic_latent_dim, self.static_latent_dim
            )
        self.transition_mode = 'latent'

        # --- discriminator (adversarial, optional) ---
        if getattr(self.config.loss, 'enable_adversarial_loss', False):
            self.discriminator = Discriminator3D(self.config)
        else:
            self.discriminator = None

        if getattr(self.config.runtime, 'verbose', False):
            print(f"🔀 MULTIMODAL E2C: static_channels={self.static_channels}, "
                  f"dynamic_channels={self.dynamic_channels}")
            print(f"🔀 MULTIMODAL E2C: z_static={self.static_latent_dim}, "
                  f"z_dynamic={self.dynamic_latent_dim}, "
                  f"decoder input={latent_dim}")
            print(f"🔀 MULTIMODAL E2C: Transition on z_dynamic ({self.dynamic_latent_dim}-dim) "
                  f"conditioned on z_static ({self.static_latent_dim}-dim)")
            print(f"🔀 MULTIMODAL E2C: Decoder outputs {n_dynamic} dynamic channels")

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _split_channels(self, x):
        x_static = x[:, self.static_channels, :, :, :]
        x_dynamic = x[:, self.dynamic_channels, :, :, :]
        return x_static, x_dynamic

    def _concat_for_decoder(self, z_static, z_dynamic):
        """Build the decoder input: concat(z_static, z_dynamic) = latent_dim."""
        return torch.cat([z_static, z_dynamic], dim=-1)

    def _reassemble(self, x_dynamic_pred, x_static_ref):
        """Build a full 4-channel tensor from predicted dynamic + original static."""
        batch = x_dynamic_pred.shape[0]
        total_ch = len(self.static_channels) + len(self.dynamic_channels)
        spatial = x_dynamic_pred.shape[2:]
        full = x_dynamic_pred.new_zeros(batch, total_ch, *spatial)
        for i, ch in enumerate(self.dynamic_channels):
            full[:, ch, ...] = x_dynamic_pred[:, i, ...]
        for i, ch in enumerate(self.static_channels):
            full[:, ch, ...] = x_static_ref[:, i, ...]
        return full

    # --------------------------------------------------------------------- #
    #  Forward (training)
    # --------------------------------------------------------------------- #
    def set_teacher_forcing_ratio(self, ratio):
        self._teacher_forcing_ratio = ratio

    def update_noise_std(self, epoch):
        if self._noise_anneal and self.enable_latent_noise:
            progress = min(epoch / max(self._noise_anneal_epochs, 1), 1.0)
            self._current_noise_std = (
                self._noise_start_std + (self._noise_end_std - self._noise_start_std) * progress
            )

    def forward(self, inputs):
        X, U, Y, dt = inputs

        if len(X) != self.n_steps:
            raise ValueError(f"Expected {self.n_steps} states but got {len(X)}")
        if len(U) != self.n_steps - 1:
            raise ValueError(f"Expected {self.n_steps - 1} controls but got {len(U)}")
        if len(Y) != self.n_steps - 1:
            raise ValueError(f"Expected {self.n_steps - 1} observations but got {len(Y)}")

        x0_full = X[0]
        x0_static, x0_dynamic = self._split_channels(x0_full)

        z_static, _, _ = self.static_encoder(x0_static)
        z_dynamic_0, mu0_d, logvar0_d = self.dynamic_encoder(x0_dynamic)

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
            z_dyn_re, _, _ = self.dynamic_encoder(x0_rec[:, :len(self.dynamic_channels), :, :, :] if x0_rec.shape[1] > len(self.dynamic_channels) else x0_rec)
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
                x_pred_dyn = x_next_pred[:, :len(self.dynamic_channels), :, :, :] if x_next_pred.shape[1] > len(self.dynamic_channels) else x_next_pred
                z_dyn_true, mu_t, logvar_t = self.dynamic_encoder(x_pred_dyn)

            X_next_pred.append(x_next_pred)
            Z_next_pred.append(z_dyn_pred)
            Z_next.append(z_dyn_true)

            if mu_list is not None:
                mu_list.append(mu_t)
                logvar_list.append(logvar_t)

            if self.enable_cycle_loss and self.training:
                x_pred_dyn_ch = x_next_pred[:, :len(self.dynamic_channels), :, :, :] if x_next_pred.shape[1] > len(self.dynamic_channels) else x_next_pred
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

        z_static, _, _ = self.static_encoder(x_static)
        z_dynamic, _, _ = self.dynamic_encoder(x_dynamic)

        z_dyn_next, yt_next = self.transition(z_dynamic, z_static, dt, ut)

        z_for_dec = self._concat_for_decoder(z_static, z_dyn_next)
        x_dyn_pred = self.decoder(z_for_dec)

        xt_next_full = self._reassemble(x_dyn_pred, x_static)
        return xt_next_full, yt_next

    def encode_initial(self, xt_full):
        """Encode a full spatial state into a concatenated latent vector.

        Returns:
            (B, static_latent_dim + dynamic_latent_dim) tensor
        """
        x_static, x_dynamic = self._split_channels(xt_full)
        z_static, _, _ = self.static_encoder(x_static)
        z_dynamic, _, _ = self.dynamic_encoder(x_dynamic)
        return torch.cat([z_static, z_dynamic], dim=-1)

    def predict_latent(self, zt, dt, ut):
        """Pure latent-space prediction (no decode/re-encode).

        Mirrors GNNE2C.predict_latent: split the concatenated latent into
        static and dynamic parts, advance z_dynamic via the transition,
        and return the updated latent plus observations.
        """
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
            '_multimodal': True,
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

        encoder_payload = torch.load(encoder_file, map_location=device, weights_only=False)

        if not isinstance(encoder_payload, dict) or '_multimodal' not in encoder_payload:
            raise ValueError(
                "Encoder file does not contain multimodal weights. "
                "It may have been saved with the standard MSE2C model."
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
        import os, glob
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved_models')
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
