"""
Hybrid GNN Embed-to-Control Model (CNN static + GNN dynamic)
=============================================================
Drop-in replacement for MultimodalMSE2C with one of the two encoders
upgraded to a Graph Neural Network.

Architecture:
- Static branch: 3D CNN ``Encoder`` (identical to MultimodalMSE2C) over
  ``(B, len(static_channels), Nx, Ny, Nz)`` -> ``z_static``.  Routed
  through the shared :class:`RealizationCache` keyed by
  ``case_to_realization`` (efficient path) for realization-level reuse.
- Dynamic branch: GNN over per-active-node features
  ``[SG, PRES, PERMI, POROS, is_inj, is_prod, well_one_hot, control]``
  followed by GridPooling (scatter -> Conv3D -> Linear) -> ``z_dynamic``.
  Geology is now locally available at every message-passing step.

Key features:
- CNN static encoder shares weight-I/O conventions with the multimodal
  family; cache thrashing avoided via ``RealizationCache``.
- Vectorised control injection (single scatter, no Python loop).
- Vectorised well-node readout (no per-sample / per-well loop).
- Memory-efficient dynamic batching by realization.

Requires: torch_geometric (PyG)
"""

import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool

from model.models.encoder import Encoder
from model.models.transition_utils import create_trans_encoder
from model.utils.initialization import weights_init
from model.utils import win_long_path
from model.utils.realization_cache import RealizationCache

from model.multimodal_approach import _build_branch_config, _ConfigProxy
from model.models.decoder import Decoder

from .graph_construction import GraphManager
from .gnn_encoder import GNNDynamicEncoder


class _ConditionedLinearTransition(nn.Module):
    """
    Identical to multimodal_approach.ConditionedLinearTransition.
    Duplicated here to avoid circular imports.
    """

    def __init__(self, config, dynamic_dim, static_dim):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        n_obs = self.num_prob * 2 + self.num_inj

        encoder_hidden_dims = config['transition'].get(
            'encoder_hidden_dims', [200, 200]
        )
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


def _build_batched_graph(gs, count, device):
    """
    Replicate a single realization's graph ``count`` times into a PyG-style
    mini-batch (disjoint union of identical graphs).
    """
    N = gs.num_active_nodes
    E = gs.edge_index.shape[1]

    offsets = torch.arange(count, device=device) * N

    ei = gs.edge_index.unsqueeze(2).expand(2, E, count)
    ei = ei + offsets.view(1, 1, count)
    edge_index = ei.reshape(2, E * count)

    edge_attr = gs.edge_attr.unsqueeze(0).expand(count, E, -1).reshape(count * E, -1)

    batch_vec = torch.arange(count, device=device).unsqueeze(1).expand(count, N).reshape(-1)

    return edge_index, edge_attr, batch_vec


class GNNE2C(nn.Module):
    """
    Graph Neural Network Embed-to-Control model.

    Drop-in replacement for ``MultimodalMSE2C`` with three enhancements:
    1. Well indicators + control values as dynamic node features
    2. Well-node readout for auxiliary observation prediction
    3. GridPooling (scatter → Conv3D) for spatially-structured latent
    """

    def __init__(self, config):
        super(GNNE2C, self).__init__()
        self.config = config
        self.n_steps = config['training']['nsteps']

        self.enable_latent_noise = config['training'].get('enable_latent_noise', False)
        noise_cfg = config['training'].get('latent_noise', {})
        self._noise_std = noise_cfg.get('noise_std', 0.01)
        self._noise_anneal = noise_cfg.get('anneal', False)
        self._noise_start_std = noise_cfg.get('start_std', 0.05)
        self._noise_end_std = noise_cfg.get('end_std', 0.005)
        self._noise_anneal_epochs = noise_cfg.get('anneal_epochs', 100)
        self._current_noise_std = self._noise_start_std if self._noise_anneal else self._noise_std
        self._teacher_forcing_ratio = 1.0
        self.enable_scheduled_sampling = config['training'].get('enable_scheduled_sampling', False)

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

        gnn_cfg = config.config.get('gnn', {})
        grid_shape = tuple(config['data']['input_shape'][1:])
        self.grid_shape = grid_shape

        device_str = config.runtime.get('device', 'auto')
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)

        import os as _os
        config_dir = _os.path.dirname(_os.path.abspath(
            getattr(config, 'config_path', 'config.yaml')
        ))

        def _resolve(p: str) -> str:
            if not _os.path.isabs(p):
                return _os.path.normpath(_os.path.join(config_dir, p))
            return p

        mask_path = _resolve(gnn_cfg.get(
            'mask_file_path',
            config.loss.get('mask_file_path',
                            'sr3_batch_output/inactive_cell_locations.h5')
        ))
        spatial_dir = _resolve(config.paths.get('data_dir', 'sr3_batch_output/'))
        well_locations = config['data'].get('well_locations', {})

        self.graph_manager = GraphManager(
            mask_path=mask_path,
            spatial_data_dir=spatial_dir,
            well_locations=well_locations,
            grid_shape=grid_shape,
            device=device,
            verbose=config.runtime.get('verbose', False),
        )

        enc_cfg = gnn_cfg.get('encoder', {})
        conv_type = gnn_cfg.get('conv_type', 'gine')
        num_inj = len(well_locations.get('injectors', {}))
        num_prod = len(well_locations.get('producers', {}))
        num_wells = num_inj + num_prod
        self.num_inj = num_inj
        self.num_prod = num_prod

        # Dynamic encoder per-node features:
        # [dynamic spatial channels, static spatial channels (geology),
        #  is_inj, is_prod, well_one_hot, control]
        # Geology (PERMI/POROS) is now per-node so the GNN's message
        # passing has direct local access to it at every layer.
        n_dyn_channels = len(self.dynamic_channels)
        n_static_channels = len(self.static_channels)
        n_well_indicator_cols = 2 + num_wells + 1  # is_inj, is_prod, well_ids, control
        dyn_in_channels = n_dyn_channels + n_static_channels + n_well_indicator_cols
        self._n_dyn_channels = n_dyn_channels
        self._n_static_channels = n_static_channels
        self._n_well_indicator_cols = n_well_indicator_cols
        # Combined channel slice into the full reservoir tensor used by
        # the dynamic encoder (dynamic first, then static).
        self._dyn_input_channels = list(self.dynamic_channels) + list(self.static_channels)

        # Build well name → control index mapping
        # Control order from training data: BHP[P1,P2,P3], GASRATSC[I1,I2,I4]
        # Well order in graph: injectors sorted, then producers sorted
        self._well_control_map = {}
        inj_names = sorted(well_locations.get('injectors', {}).keys())
        prod_names = sorted(well_locations.get('producers', {}).keys())
        for i, wname in enumerate(inj_names):
            self._well_control_map[wname] = num_prod + i  # GASRATSC starts after BHP
        for i, wname in enumerate(prod_names):
            self._well_control_map[wname] = i  # BHP is first
        self._well_order = inj_names + prod_names

        # --- Static encoder: CNN, identical to MultimodalMSE2C's ---
        # Reads (B, n_static_channels, Nx, Ny, Nz) and produces z_static.
        # The global ``encoder:`` block in config.yaml drives this CNN
        # (the same block multimodal uses).
        static_cfg_dict = _build_branch_config(
            config, n_static_channels, self.static_latent_dim
        )
        static_proxy = _ConfigProxy(static_cfg_dict, config)
        self.static_encoder = Encoder(static_proxy)

        # --- Dynamic encoder: GNN with geology-augmented per-node feats ---
        self.dynamic_encoder = GNNDynamicEncoder(
            in_channels=dyn_in_channels,
            latent_dim=self.dynamic_latent_dim,
            hidden_dim=enc_cfg.get('hidden_dim', 64),
            num_layers=enc_cfg.get('num_layers', 3),
            heads=enc_cfg.get('num_heads', 2),
            edge_dim=7,
            dropout=enc_cfg.get('dropout', 0.1),
            conv_type=conv_type,
            grid_shape=grid_shape,
            num_inj=num_inj,
            num_prod=num_prod,
        )

        n_dynamic = len(self.dynamic_channels)
        dec_cfg_dict = _build_branch_config(config, n_dynamic, latent_dim)
        dec_proxy = _ConfigProxy(dec_cfg_dict, config)

        decoder_type = config['decoder'].get('type', 'standard')
        if decoder_type == 'smooth':
            from model.models.decoder_smooth import DecoderSmooth
            self.decoder = DecoderSmooth(dec_proxy)
        elif decoder_type == 'smooth_generic':
            from model.models.decoder_smooth import DecoderSmoothGeneric
            self.decoder = DecoderSmoothGeneric(dec_proxy)
        else:
            self.decoder = Decoder(dec_proxy)

        if gnn_cfg.get('torch_compile', False):
            import sys as _sys
            _can_compile = True
            if _sys.platform == 'win32':
                _can_compile = False
            else:
                try:
                    import triton  # noqa: F401
                except ImportError:
                    _can_compile = False
            if _can_compile:
                # Only the dynamic GNN benefits from torch.compile; the
                # static branch is a stock CNN whose backend is already
                # cuDNN-optimised.
                try:
                    self.dynamic_encoder.gnn = torch.compile(self.dynamic_encoder.gnn)
                except Exception:
                    pass

        transition_type = config['transition'].get('type', 'linear').lower()
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
            import importlib
            mod_path, cls_name = _conditioned_map[transition_type]
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            self.transition = cls(config, self.dynamic_latent_dim, self.static_latent_dim)
        else:
            self.transition = _ConditionedLinearTransition(
                config, self.dynamic_latent_dim, self.static_latent_dim
            )
        self.transition_mode = 'latent'
        self.discriminator = None

        self._max_sub_batch = gnn_cfg.get('max_sub_batch', 16)

        # Shared multi-entry, realization-keyed static cache.  Identical
        # behaviour to the cache used by Multimodal and MultiEmbedding.
        self._z_static_cache = RealizationCache(
            maxlen=max(64, self.graph_manager.num_realizations + 8)
        )

        # Cached PyG mini-batch tensors for the dynamic GNN, keyed by
        # (realization_id, count).
        self._batched_graph_cache = {}

        # Vectorised control-injection / well-readout precomputed buffers,
        # keyed by realization_id.  Built lazily on first dynamic forward.
        self._well_inject_cache = {}
        self._well_readout_cache = {}

        self.well_readout_lambda = gnn_cfg.get('well_readout_lambda', 1.0)

        if config.runtime.get('verbose', False):
            total_params = sum(p.numel() for p in self.parameters())
            conv_label = 'GINEConv (fast)' if conv_type == 'gine' else 'GATv2Conv (attention)'
            print(f"Hybrid GNN E2C: {self.graph_manager.num_realizations} realizations, "
                  f"z_static={self.static_latent_dim} (CNN), "
                  f"z_dynamic={self.dynamic_latent_dim} (GNN), "
                  f"conv={conv_label}, "
                  f"dyn_in={dyn_in_channels} "
                  f"(dyn={n_dyn_channels}+geo={n_static_channels}+well_ind+ctrl), "
                  f"pooling=GridPooling, "
                  f"well_readout=True (vectorised), "
                  f"total params={total_params:,}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_case_to_realization(self, mapping):
        """Override the ``GraphManager``'s case-to-realization mapping.

        Most callers do NOT need this: the GNN already builds and uses
        ``GraphManager.case_to_realization`` automatically.  Provided
        for API symmetry with :class:`MultimodalMSE2C` and
        :class:`MultiEmbeddingMultimodal` so the same caller code can
        run against any of the three model families.

        Pass ``None`` to revert to the GraphManager's mapping.
        """
        # The GNN sources its mapping from the GraphManager; allow an
        # external override but invalidate the static cache when it
        # changes.
        if mapping is not None:
            # Replace the GraphManager's case_to_realization in-place
            # so downstream `_get_real_ids` calls see the new mapping
            # without any other code changes.
            self.graph_manager.case_to_realization = mapping
        self._z_static_cache.clear()

    def _get_real_ids(self, case_indices, B, device):
        """Per-sample realization id tensor.  Falls back to all-zeros
        when ``case_indices`` is None (single-realization assumption,
        same as the legacy behaviour)."""
        if case_indices is None:
            return torch.zeros(B, dtype=torch.long, device=device)
        return torch.tensor(
            [self.graph_manager.get_realization(int(ci)) for ci in case_indices],
            dtype=torch.long, device=device,
        )

    def _realization_keys_from_real_ids(self, real_ids):
        """Convert a (B,) real_ids tensor into a Python list of keys
        suitable for :class:`RealizationCache`.
        """
        return [int(r) for r in real_ids]

    def _group_by_realization(self, real_ids):
        groups = {}
        for b in range(real_ids.shape[0]):
            r = int(real_ids[b])
            if r not in groups:
                groups[r] = []
            groups[r].append(b)
        return {r: torch.tensor(indices, dtype=torch.long, device=real_ids.device)
                for r, indices in groups.items()}

    def _get_cached_batched_graph(self, gs, r, count, device):
        key = (r, count)
        if key not in self._batched_graph_cache:
            edge_index, edge_attr, batch_vec = _build_batched_graph(
                gs, count, device
            )
            N = gs.num_active_nodes
            positions = gs.node_positions.unsqueeze(0).expand(
                count, N, -1
            ).reshape(count * N, -1)
            self._batched_graph_cache[key] = (
                edge_index, edge_attr, batch_vec, positions
            )
        return self._batched_graph_cache[key]

    def _get_well_inject_buffers(self, r, gs, device):
        """Return precomputed (well_node_ids, well_ctrl_indices) tensors
        for vectorised control injection at well perforations.

        Built once per realization, cached on ``self._well_inject_cache``.
        """
        if r in self._well_inject_cache:
            return self._well_inject_cache[r]

        node_ids = []
        ctrl_indices = []
        for wname in self._well_order:
            winfo = gs.well_node_indices.get(wname)
            if winfo is None:
                continue
            ctrl_idx = self._well_control_map[wname]
            for nid in winfo['node_ids']:
                node_ids.append(int(nid))
                ctrl_indices.append(int(ctrl_idx))

        if node_ids:
            node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=device)
            ctrl_idx_t = torch.tensor(ctrl_indices, dtype=torch.long, device=device)
        else:
            node_ids_t = torch.zeros(0, dtype=torch.long, device=device)
            ctrl_idx_t = torch.zeros(0, dtype=torch.long, device=device)
        self._well_inject_cache[r] = (node_ids_t, ctrl_idx_t)
        return self._well_inject_cache[r]

    def _get_well_readout_buffers(self, r, gs, device):
        """Return precomputed padded ``(num_wells, max_perfs)`` index +
        mask tensors used for batched well-node readout pooling.

        Wells with zero perforations contribute an all-zero mean (the
        mask sums clamp at 1 to avoid division by zero).
        """
        if r in self._well_readout_cache:
            return self._well_readout_cache[r]

        per_well_ids = []
        for wname in self._well_order:
            winfo = gs.well_node_indices.get(wname)
            if winfo and len(winfo['node_ids']) > 0:
                per_well_ids.append([int(n) for n in winfo['node_ids']])
            else:
                per_well_ids.append([])

        max_perfs = max((len(ids) for ids in per_well_ids), default=0)
        num_wells = len(per_well_ids)

        if max_perfs == 0:
            idx_padded = torch.zeros(num_wells, 1, dtype=torch.long, device=device)
            valid_mask = torch.zeros(num_wells, 1, dtype=torch.float32, device=device)
        else:
            idx_padded = torch.zeros(num_wells, max_perfs, dtype=torch.long, device=device)
            valid_mask = torch.zeros(num_wells, max_perfs, dtype=torch.float32, device=device)
            for w_i, ids in enumerate(per_well_ids):
                if len(ids) > 0:
                    idx_padded[w_i, :len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=device
                    )
                    valid_mask[w_i, :len(ids)] = 1.0

        self._well_readout_cache[r] = (idx_padded, valid_mask)
        return self._well_readout_cache[r]

    # ------------------------------------------------------------------
    # Static encoding (CNN + realization-aware cache)
    # ------------------------------------------------------------------

    def _encode_static_grouped(self, x_full, real_ids, B, device):
        """Encode static channels via a 3D CNN, with realization-keyed
        multi-entry caching.

        Args:
            x_full: (B, C, Nx, Ny, Nz) full reservoir state.  Static
                    channels are sliced internally.
            real_ids: (B,) int64 realization indices used as cache keys
                      (efficient path; matches the GNN's existing
                      per-realization grouping).
            B: batch size.
            device: torch device.

        Returns:
            z_static: (B, static_latent_dim) tensor.
        """
        x_static = x_full[:, self.static_channels, :, :, :]

        if self.training:
            self._z_static_cache.clear()
            return self.static_encoder(x_static)[0]

        keys = self._realization_keys_from_real_ids(real_ids)
        z_static, miss_idx, miss_keys = self._z_static_cache.lookup(
            keys, self.static_latent_dim, device, dtype=x_static.dtype
        )
        if miss_idx.numel() > 0:
            x_miss = x_static.index_select(0, miss_idx)
            z_miss = self.static_encoder(x_miss)[0]
            z_static.index_copy_(0, miss_idx, z_miss)
            # Deduplicate keys: a single inference batch can include
            # multiple samples from the same realization, so pair the
            # *first* per-key encoded latent with each unique key.
            seen = {}
            unique_keys = []
            unique_z = []
            for i, k in enumerate(miss_keys):
                if k in seen:
                    continue
                seen[k] = True
                unique_keys.append(k)
                unique_z.append(z_miss[i])
            if unique_keys:
                self._z_static_cache.store(unique_keys, torch.stack(unique_z, dim=0))
        return z_static

    # ------------------------------------------------------------------
    # Dynamic encoding with controls + well indicators + grid pooling
    # ------------------------------------------------------------------

    def _encode_dynamic_grouped(self, grid_tensor, real_ids, B, device,
                                controls=None):
        """
        Encode dynamic features with geology, well indicators, controls,
        GridPooling, and (vectorised) well-node readout.

        Per-node feature layout (in order):
            [dynamic spatial channels (SG, PRES),
             static spatial channels (PERMI, POROS),
             is_inj, is_prod, well_one_hot, control]

        Args:
            grid_tensor: (B, C, Nx, Ny, Nz)
            real_ids: (B,) realization indices
            B: batch size
            device: torch device
            controls: (B, u_dim) or None — well controls to inject at well nodes

        Returns:
            z_dynamic: (B, dynamic_latent_dim)
            y_well_aux: (B, n_obs) or None — auxiliary well observation predictions
        """
        z_dynamic = torch.empty(B, self.dynamic_latent_dim, device=device)
        collect_well_readout = controls is not None
        y_well_aux = None
        if collect_well_readout:
            n_obs = self.num_inj + self.num_prod * 2
            y_well_aux = torch.zeros(B, n_obs, device=device)
        groups = self._group_by_realization(real_ids)
        hidden_dim = self.dynamic_encoder.hidden_dim
        Nx, Ny, Nz = self.grid_shape

        for r, indices in groups.items():
            gs = self.graph_manager.get_graph(r)
            N = gs.num_active_nodes
            coords = gs.node_to_grid

            for chunk_start in range(0, len(indices), self._max_sub_batch):
                chunk_idx = indices[chunk_start:chunk_start + self._max_sub_batch]
                count = len(chunk_idx)

                # Extract dynamic + static (geology) channels at active nodes
                # in one gather.  Channel order:
                # [dynamic_channels..., static_channels...]
                sub_grid = grid_tensor[chunk_idx]
                dyn_feats = sub_grid[:, self._dyn_input_channels][
                    :, :, coords[:, 0], coords[:, 1], coords[:, 2]
                ]  # (count, n_dyn+n_stat, N)
                dyn_feats = dyn_feats.permute(0, 2, 1)  # (count, N, n_dyn+n_stat)

                # Append well indicators (is_inj, is_prod, well_ids, control=0)
                well_ind = gs.well_indicator_features.unsqueeze(0).expand(
                    count, N, -1
                ).clone()  # (count, N, n_well_indicator_cols)

                # --- Vectorised control injection at well perforations ---
                if controls is not None:
                    sub_controls = controls[chunk_idx]  # (count, u_dim)
                    well_node_ids, well_ctrl_idx = self._get_well_inject_buffers(
                        r, gs, device
                    )
                    if well_node_ids.numel() > 0:
                        ctrl_col = self._n_well_indicator_cols - 1
                        # ctrl_vals: (count, total_well_nodes)
                        ctrl_vals = sub_controls.index_select(1, well_ctrl_idx)
                        # In-place scatter into the control column at the
                        # right node ids.  Equivalent to the legacy
                        # double-loop but a single op.
                        well_ind[:, well_node_ids, ctrl_col] = ctrl_vals

                # Concatenate: [dyn, geo, is_inj, is_prod, well_ids, control]
                flat_feats = torch.cat([dyn_feats, well_ind], dim=-1)
                flat_feats = flat_feats.reshape(count * N, -1)

                edge_index, edge_attr, batch_vec, _ = \
                    self._get_cached_batched_graph(gs, r, count, device)
                h = self.dynamic_encoder.gnn(
                    flat_feats, edge_index, edge_attr=edge_attr
                )  # (count * N, hidden_dim)

                h_reshaped = h.view(count, N, hidden_dim)

                # --- Vectorised well-node readout ---
                if collect_well_readout:
                    idx_padded, valid_mask = self._get_well_readout_buffers(
                        r, gs, device
                    )
                    # gather: (count, num_wells, max_perfs, hidden_dim)
                    gathered = h_reshaped[:, idx_padded, :]
                    mask = valid_mask.unsqueeze(0).unsqueeze(-1)  # (1, num_wells, max_perfs, 1)
                    summed = (gathered * mask).sum(dim=2)
                    counts = valid_mask.sum(dim=1).clamp(min=1.0)  # (num_wells,)
                    well_pooled = summed / counts.view(1, -1, 1)
                    # well_pooled: (count, num_wells, hidden_dim)
                    y_chunk = self.dynamic_encoder.well_readout(well_pooled)
                    y_well_aux[chunk_idx] = y_chunk

                # --- GridPooling: scatter to grid → Conv3D → latent ---
                grid_3d = torch.zeros(
                    count, hidden_dim, Nx, Ny, Nz,
                    device=device, dtype=h.dtype
                )
                grid_3d[:, :, coords[:, 0], coords[:, 1], coords[:, 2]] = \
                    h_reshaped.permute(0, 2, 1)
                z_chunk = self.dynamic_encoder.grid_pool(grid_3d)
                z_dynamic[chunk_idx] = z_chunk

        return z_dynamic, y_well_aux

    def _decode_cnn(self, z_batch):
        return self.decoder(z_batch)

    def _reassemble(self, x_dynamic_pred, x_static_ref):
        B = x_dynamic_pred.shape[0]
        total_ch = len(self.static_channels) + len(self.dynamic_channels)
        spatial = x_dynamic_pred.shape[2:]
        full = x_dynamic_pred.new_zeros(B, total_ch, *spatial)
        for i, ch in enumerate(self.dynamic_channels):
            full[:, ch, ...] = x_dynamic_pred[:, i, ...]
        for i, ch in enumerate(self.static_channels):
            full[:, ch, ...] = x_static_ref[:, i, ...]
        return full

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def set_teacher_forcing_ratio(self, ratio):
        self._teacher_forcing_ratio = ratio

    def update_noise_std(self, epoch):
        if self._noise_anneal and self.enable_latent_noise:
            progress = min(epoch / max(self._noise_anneal_epochs, 1), 1.0)
            self._current_noise_std = (
                self._noise_start_std + (self._noise_end_std - self._noise_start_std) * progress
            )

    def forward(self, inputs):
        if len(inputs) == 5:
            X, U, Y, dt, case_indices = inputs
        else:
            X, U, Y, dt = inputs
            case_indices = None

        if len(X) != self.n_steps:
            raise ValueError(f"Expected {self.n_steps} states but got {len(X)}")

        x0_full = X[0]
        B = x0_full.shape[0]
        device = x0_full.device

        real_ids = self._get_real_ids(case_indices, B, device)
        z_static = self._encode_static_grouped(x0_full, real_ids, B, device)

        ctrl_for_encoding = U[0] if len(U) > 0 else None
        z_dynamic_0, y_well_aux_0 = self._encode_dynamic_grouped(
            x0_full, real_ids, B, device, controls=ctrl_for_encoding
        )

        z_dyn_for_model = z_dynamic_0
        if self.training and self.enable_latent_noise:
            z_dyn_for_model = z_dynamic_0 + self._current_noise_std * torch.randn_like(z_dynamic_0)

        z0_for_dec = torch.cat([z_static, z_dyn_for_model], dim=-1)
        x0_rec = self._decode_cnn(z0_for_dec)

        X_next_dynamic = []
        for i_step in range(len(X) - 1):
            X_next_dynamic.append(X[i_step + 1][:, self.dynamic_channels, :, :, :])

        Z_dyn_pred_list, Y_pred_list = self.transition.forward_nsteps(
            z_dyn_for_model, z_static, dt, U
        )

        X_next_pred = []
        Z_next_pred = []
        Z_next = []

        for i_step in range(len(Z_dyn_pred_list)):
            z_dyn_pred = Z_dyn_pred_list[i_step]
            z_for_dec = torch.cat([z_static, z_dyn_pred], dim=-1)
            x_next_pred = self._decode_cnn(z_for_dec)

            use_teacher = (
                not self.enable_scheduled_sampling
                or not self.training
                or torch.rand(1).item() < self._teacher_forcing_ratio
            )
            if use_teacher:
                z_dyn_true, _ = self._encode_dynamic_grouped(
                    X[i_step + 1], real_ids, B, device, controls=None
                )
            else:
                z_dyn_true, _ = self._encode_dynamic_grouped(
                    x_next_pred, real_ids, B, device, controls=None
                )

            X_next_pred.append(x_next_pred)
            Z_next_pred.append(z_dyn_pred)
            Z_next.append(z_dyn_true)

        x0_dynamic = x0_full[:, self.dynamic_channels, :, :, :]

        self._well_readout_aux = None
        if y_well_aux_0 is not None and len(Y) > 0:
            y_true = Y[0]
            aux_loss = nn.functional.mse_loss(y_well_aux_0, y_true)
            self._well_readout_aux = aux_loss

        return (X_next_pred, X_next_dynamic, Z_next_pred, Z_next,
                Y_pred_list, Y, z0_for_dec, x0_dynamic, x0_rec,
                None, None, None)

    # ------------------------------------------------------------------
    # Predict (inference, single step)
    # ------------------------------------------------------------------

    def predict(self, inputs):
        # Accept either 4-tuple ``(xt_full, ut, yt, dt)`` or 5-tuple
        # ``(xt_full, ut, yt, dt, case_indices)``.  Without
        # ``case_indices`` the static cache falls back to assuming a
        # single realization (legacy behaviour); with them every sample
        # is mapped to its own realization id.
        if len(inputs) == 5:
            xt_full, ut, yt, dt, case_indices = inputs
        else:
            xt_full, ut, yt, dt = inputs
            case_indices = None

        B = xt_full.shape[0]
        device = xt_full.device

        real_ids = self._get_real_ids(case_indices, B, device)

        z_static = self._encode_static_grouped(xt_full, real_ids, B, device)
        z_dynamic, _ = self._encode_dynamic_grouped(
            xt_full, real_ids, B, device, controls=ut
        )

        z_dyn_next, yt_next = self.transition(z_dynamic, z_static, dt, ut)

        z_for_dec = torch.cat([z_static, z_dyn_next], dim=-1)
        x_dyn_pred = self._decode_cnn(z_for_dec)

        x_static = xt_full[:, self.static_channels, :, :, :]
        xt_next_full = self._reassemble(x_dyn_pred, x_static)
        return xt_next_full, yt_next

    def encode_initial(self, xt_full, case_indices=None):
        B = xt_full.shape[0]
        device = xt_full.device
        real_ids = self._get_real_ids(case_indices, B, device)

        z_static = self._encode_static_grouped(xt_full, real_ids, B, device)
        z_dynamic, _ = self._encode_dynamic_grouped(
            xt_full, real_ids, B, device, controls=None
        )
        return torch.cat([z_static, z_dynamic], dim=-1)

    def predict_latent(self, zt, dt, ut):
        z_static = zt[:, :self.static_latent_dim]
        z_dynamic = zt[:, self.static_latent_dim:]

        z_dyn_next, yt_next = self.transition(z_dynamic, z_static, dt, ut)

        zt_next = torch.cat([z_static, z_dyn_next], dim=-1)
        return zt_next, yt_next

    def get_well_readout_loss(self):
        """Return the auxiliary well-readout loss (computed during forward)."""
        return self._well_readout_aux

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    # Hybrid GNN payload format version.  Bumped when the encoder
    # architecture changes incompatibly with prior checkpoints.  The
    # state-dict KEY name remains ``'static_encoder'`` for the static
    # branch even though its underlying class changed from a GNN to a
    # CNN ``Encoder``.  Keeping the same key (matching the model
    # attribute name and the multimodal/FNO convention) lets all
    # existing dashboards / extractors that already understand
    # multimodal payloads also load hybrid-GNN payloads with no code
    # change -- ``Encoder.fc_mean.weight`` is what they look for when
    # summing per-branch latent dims.
    GNN_PAYLOAD_VERSION = 2

    def save_weights_to_file(self, encoder_file, decoder_file, transition_file):
        encoder_payload = {
            # Hybrid layout: CNN static (still under the historical
            # 'static_encoder' key for downstream tooling compatibility)
            # + GNN dynamic.
            'static_encoder': self.static_encoder.state_dict(),
            'dynamic_encoder': self.dynamic_encoder.state_dict(),
            '_gnn': True,
            '_gnn_version': self.GNN_PAYLOAD_VERSION,
        }
        torch.save(encoder_payload, win_long_path(encoder_file))
        torch.save(self.decoder.state_dict(), win_long_path(decoder_file))
        torch.save(self.transition.state_dict(), win_long_path(transition_file))

    def load_weights_from_file(self, encoder_file, decoder_file, transition_file):
        # Backward-compatible full loader; delegates to the per-component
        # methods so the warm-start path uses identical validation.
        self._load_encoder_payload(encoder_file)
        self._load_decoder_payload(decoder_file)
        self._load_transition_payload(transition_file)

    # --- Per-component loaders (used by warm-start paths) ---
    def _torch_device(self):
        if len(list(self.parameters())) > 0:
            return next(self.parameters()).device
        return torch.device('cpu')

    def _load_encoder_payload(self, encoder_file):
        encoder_file = win_long_path(encoder_file)
        device = self._torch_device()
        encoder_payload = torch.load(
            encoder_file, map_location=device, weights_only=False
        )
        if not isinstance(encoder_payload, dict) or '_gnn' not in encoder_payload:
            raise ValueError(
                f"Encoder file {encoder_file} does not contain GNN weights "
                f"(missing '_gnn' sentinel)."
            )
        payload_version = encoder_payload.get('_gnn_version', 1)
        if payload_version < self.GNN_PAYLOAD_VERSION:
            raise ValueError(
                f"Legacy GNN checkpoint detected (_gnn_version={payload_version}). "
                f"This release uses the hybrid CNN-static + GNN-dynamic layout "
                f"(_gnn_version={self.GNN_PAYLOAD_VERSION}). Old checkpoints "
                f"have an incompatible static-encoder architecture and cannot "
                f"be loaded; please retrain with the new model."
            )
        # Accept either the canonical key ('static_encoder', matching the
        # multimodal/FNO convention and the model attribute name) or the
        # interim 'static_cnn' key used in pre-release hybrid checkpoints.
        static_sd = (
            encoder_payload.get('static_encoder')
            or encoder_payload.get('static_cnn')
        )
        if static_sd is None or 'dynamic_encoder' not in encoder_payload:
            raise ValueError(
                f"Hybrid GNN encoder payload {encoder_file} is missing the "
                f"static branch ('static_encoder' or 'static_cnn') or "
                f"'dynamic_encoder' state dicts."
            )
        try:
            self.static_encoder.load_state_dict(static_sd)
            self.dynamic_encoder.load_state_dict(encoder_payload['dynamic_encoder'])
        except RuntimeError as e:
            raise RuntimeError(
                f"GNN encoder weights in {encoder_file} do not match the "
                f"current model architecture: {e}"
            ) from e

    def _load_decoder_payload(self, decoder_file):
        decoder_file = win_long_path(decoder_file)
        device = self._torch_device()
        try:
            self.decoder.load_state_dict(
                torch.load(decoder_file, map_location=device, weights_only=False)
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"GNN decoder weights in {decoder_file} do not match the "
                f"current model architecture: {e}"
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
                f"GNN transition weights in {transition_file} do not match "
                f"the current transition architecture: {e}"
            ) from e

    def find_and_load_weights(self, models_dir=None,
                              base_pattern='e2co', specific_pattern=None):
        import os
        import glob as globmod
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'saved_models')
        encoder_files = sorted(
            globmod.glob(os.path.join(models_dir, f"{base_pattern}_encoder*.h5"))
        )
        decoder_files = sorted(
            globmod.glob(os.path.join(models_dir, f"{base_pattern}_decoder*.h5"))
        )
        transition_files = sorted(
            globmod.glob(os.path.join(models_dir, f"{base_pattern}_transition*.h5"))
        )

        if not encoder_files or not decoder_files or not transition_files:
            raise FileNotFoundError(
                f"No matching model files found in {models_dir}"
            )

        if specific_pattern:
            encoder_files = [f for f in encoder_files if specific_pattern in f]
            decoder_files = [f for f in decoder_files if specific_pattern in f]
            transition_files = [f for f in transition_files if specific_pattern in f]

        self.load_weights_from_file(
            encoder_files[0], decoder_files[0], transition_files[0]
        )
        return {
            'encoder': encoder_files[0],
            'decoder': decoder_files[0],
            'transition': transition_files[0],
        }
