"""
Digital Twin Engine
===================
Wraps the trained ROM for interactive reservoir simulation.

Model selection follows the same pattern as the testing dashboard
(``ROM_Refactored/testing/dashboard.py``):

1. ``scan_models()`` scans ``ROM_Refactored/saved_models/`` for complete
   encoder+decoder+transition triples.
2. ``load_model(model_info)`` detects model type (CNN / multimodal / GNN)
   and transition type directly from the weight files, configures
   ``ROMWithE2C`` accordingly, and calls each model class's **native**
   ``load_weights_from_file``.
3. ``step()`` supports both *state-based* and *latent* prediction modes.
"""

import os
import re
import sys
import json
import glob as globmod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import torch
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_DIGITAL_TWIN_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _DIGITAL_TWIN_DIR.parent
_ROM_DIR = _PROJECT_ROOT / "ROM_Refactored"
_RL_DIR = _PROJECT_ROOT / "RL_Refactored"
for _p in [str(_PROJECT_ROOT), str(_RL_DIR), str(_ROM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ROM_Refactored.utilities.config_loader import Config
from ROM_Refactored.model.training.rom_wrapper import ROMWithE2C

from .state_manager import StateManager
from .corner_point_parser import load_or_parse_corners

_CHANNEL_VARS = ["SG", "PRES", "PERMI", "POROS"]


class DigitalTwinEngine:
    """High-level controller wrapping a ROM for interactive simulation."""

    def __init__(self, dt_config_path: str = None, rom_weights_path: str = None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        if dt_config_path is None:
            dt_config_path = str(_DIGITAL_TWIN_DIR / "config.yaml")
        with open(dt_config_path, "r") as f:
            self.dt_cfg = yaml.safe_load(f)

        rom_cfg_path = str((_DIGITAL_TWIN_DIR / self.dt_cfg["rom_config_path"]).resolve())
        self._rom_cfg_path = rom_cfg_path

        rl_cfg_path = str((_DIGITAL_TWIN_DIR / self.dt_cfg["rl_config_path"]).resolve())
        self.rl_config = Config(rl_cfg_path)

        rl_model = self.rl_config.config.get("rl_model", {})
        self.restricted_action_ranges = {
            "producer_bhp": {"min": 1087.78, "max": 1305.34},
            "gas_injection": {"min": 6180072.5, "max": 100646896.0},
        }
        self.dt_value = rl_model.get("reservoir", {}).get("time_step_normalized", 1.0)
        self.time_step_days = rl_model.get("reservoir", {}).get("time_step_days", 365)
        self.max_steps = rl_model.get("environment", {}).get("max_episode_steps", 30)

        self.channel_names = self.dt_cfg.get(
            "channels",
            {0: "gas_saturation", 1: "pressure", 2: "permeability", 3: "porosity"},
        )

        # Stable references read from ROM config once
        tmp_cfg = Config(rom_cfg_path)
        shape = tmp_cfg.data["input_shape"]
        self.n_channels, self.nx, self.ny, self.nz = shape
        self.well_locations = tmp_cfg.data.get("well_locations", {})
        self.num_prod = tmp_cfg.data.get("num_prod", 3)
        self.num_inj = tmp_cfg.data.get("num_inj", 3)
        self.u_dim = tmp_cfg.model["u_dim"]

        # Normalization
        self.norm_params: Dict = {}
        self._load_normalization_parameters()
        self._spatial_denorm = self._build_spatial_denorm()

        # State manager (economics from RL config)
        econ = rl_model.get("economics", {})
        self.state_mgr = StateManager(
            n_channels=self.n_channels, nx=self.nx, ny=self.ny, nz=self.nz,
            num_prod=self.num_prod, num_inj=self.num_inj,
            economics_config=econ, time_step_days=self.time_step_days,
        )

        # Inactive-cell mask (full 4D: cases x Nx x Ny x Nz)
        self._active_mask_4d: Optional[np.ndarray] = None
        self.active_mask: Optional[np.ndarray] = None
        self._load_inactive_mask(tmp_cfg)

        # All initial states from training data (N x C x Nx x Ny x Nz)
        self._all_initial_states: Optional[torch.Tensor] = None
        self._default_initial_state: Optional[torch.Tensor] = None
        self._current_case_idx = 0
        self.num_cases = 0
        self._load_initial_state()

        # Corner-point grid (parsed from CMG .dat file)
        self.corner_point_grid = None
        self._load_corner_point_grid()

        # ROM (populated by load_model)
        self.rom = None
        self.rom_config = None
        self._loaded_model_label = "None"

        # Prediction mode – latent avoids encode-decode error accumulation
        self.prediction_mode = "latent"

        # Runtime state
        self.current_spatial_state: Optional[torch.Tensor] = None
        self.current_latent: Optional[torch.Tensor] = None
        self._initial_static_channels: Optional[torch.Tensor] = None
        self.step_index = 0

        # RL agent (lazy)
        self.rl_agent = None
        self._rl_agent_loaded = False
        self._rl_algorithm_type: Optional[str] = None

        # Scan and auto-load first model
        models = self.scan_models()
        if rom_weights_path:
            cfg = Config(self._rom_cfg_path)
            cfg.config.setdefault("loss", {})["enable_inactive_masking"] = False
            self.rom_config = cfg
            self.rom = ROMWithE2C(cfg)
            self.rom.model.to(self.device).eval()
            sd = torch.load(rom_weights_path, map_location=self.device, weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            self.rom.model.load_state_dict(sd, strict=False)
            self._loaded_model_label = Path(rom_weights_path).stem
            print(f"[DT-Engine] Loaded weights from {rom_weights_path}")
        elif models:
            self.load_model(models[0])
        else:
            cfg = Config(self._rom_cfg_path)
            cfg.config.setdefault("gnn", {})["enable"] = False
            cfg.config.setdefault("multimodal", {})["enable"] = False
            cfg.config.setdefault("loss", {})["enable_inactive_masking"] = False
            self.rom_config = cfg
            self.rom = ROMWithE2C(cfg)
            self.rom.model.to(self.device).eval()
            self._loaded_model_label = "No weights"
            print("[DT-Engine] No model files found -- using random weights")

    # ==================================================================
    # Model scanning  (mirrors testing dashboard _scan_available_models)
    # ==================================================================
    @staticmethod
    def scan_models(models_dir: str = None) -> List[Dict]:
        """Scan ``saved_models/`` for complete encoder+decoder+transition sets."""
        if models_dir is None:
            models_dir = str(_ROM_DIR / "saved_models")
        if not os.path.exists(models_dir):
            return []

        enc_files = globmod.glob(os.path.join(models_dir, "e2co_encoder_grid_*.h5"))
        enc_files += globmod.glob(os.path.join(models_dir, "e2co_encoder*.h5"))
        enc_files = list(dict.fromkeys(enc_files))

        dec_files = globmod.glob(os.path.join(models_dir, "e2co_decoder*.h5"))
        trn_files = globmod.glob(os.path.join(models_dir, "e2co_transition*.h5"))

        dec_set = {os.path.basename(f): f for f in dec_files}
        trn_set = {os.path.basename(f): f for f in trn_files}

        results = []
        seen = set()
        for enc_path in enc_files:
            enc_name = os.path.basename(enc_path)
            dec_name = enc_name.replace("_encoder", "_decoder")
            trn_name = enc_name.replace("_encoder", "_transition")
            dec_path = dec_set.get(dec_name)
            trn_path = trn_set.get(trn_name)
            if not dec_path:
                continue
            key = enc_name
            if key in seen:
                continue
            seen.add(key)

            # Detect type from filename
            fname_lower = enc_name.lower()
            if "_gnn" in fname_lower and "_mm" not in fname_lower:
                mtype = "gnn"
            elif "_fno" in fname_lower:
                mtype = "fno"
            elif "_mm" in fname_lower:
                mtype = "multimodal"
            else:
                mtype = "cnn"

            # Detect transition type from filename
            ttype = "linear"
            for suffix, tname in [
                ("_trnNONLINEAR", "nonlinear"), ("_trnS4D_DPLR", "s4d_dplr"),
                ("_trnS4D", "s4d"), ("_trnS5", "s5"), ("_trnCLRU", "clru"),
                ("_trnDEEP_KOOPMAN", "deep_koopman"), ("_trnKOOPMAN", "koopman"),
                ("_trnCT_KOOPMAN", "ct_koopman"), ("_trnMAMBA2", "mamba2"),
                ("_trnMAMBA", "mamba"), ("_trnGRU", "gru"), ("_trnLSTM", "lstm"),
                ("_trnHAMILTONIAN", "hamiltonian"),
            ]:
                if suffix.lower() in fname_lower or suffix in enc_name:
                    ttype = tname
                    break

            label = f"{mtype.upper()} | {ttype}"
            m = re.search(r"run(\d+)", enc_name)
            if m:
                label = f"Run {m.group(1)} | {label}"

            results.append({
                "label": label,
                "encoder": enc_path,
                "decoder": dec_path,
                "transition": trn_path,
                "model_type": mtype,
                "transition_type": ttype,
            })

        results.sort(key=lambda x: x["label"])
        return results

    # ==================================================================
    # Model loading  (mirrors testing dashboard _load_model_handler)
    # ==================================================================
    def load_model(self, model_info: Dict) -> str:
        """Build the correct model type and load weights natively.

        Returns a status message string.
        """
        enc_path = model_info["encoder"]
        dec_path = model_info["decoder"]
        trn_path = model_info.get("transition")

        # Detect model type from encoder payload
        mtype = model_info.get("model_type", "cnn")
        try:
            payload = torch.load(enc_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                if payload.get("_multimodal"):
                    mtype = "multimodal"
                elif payload.get("_gnn"):
                    mtype = "gnn"
                elif payload.get("_fno"):
                    mtype = "fno"
        except Exception:
            pass

        # Detect transition type from transition state_dict keys
        ttype = model_info.get("transition_type", "linear")
        if trn_path and os.path.exists(trn_path):
            try:
                tsd = torch.load(trn_path, map_location="cpu", weights_only=False)
                ttype = self._detect_transition_type_from_keys(tsd)
            except Exception:
                pass

        # Build fresh config with correct overrides
        cfg = Config(self._rom_cfg_path)
        gnn_cfg = cfg.config.setdefault("gnn", {})
        fno_cfg = cfg.config.setdefault("fno", {})
        mm_cfg = cfg.config.setdefault("multimodal", {})

        gnn_cfg["enable"] = (mtype == "gnn")
        fno_cfg["enable"] = (mtype == "fno")
        mm_cfg["enable"] = (mtype == "multimodal")
        cfg.config.setdefault("transition", {})["type"] = ttype
        cfg.config.setdefault("loss", {})["enable_inactive_masking"] = False

        print(f"[DT-Engine] Building {mtype} model with {ttype} transition ...")

        self.rom_config = cfg
        self.rom = ROMWithE2C(cfg)
        self.rom.model.to(self.device).eval()

        # Load weights using each model type's native method
        try:
            if trn_path and os.path.exists(trn_path):
                self.rom.model.load_weights_from_file(enc_path, dec_path, trn_path)
            else:
                self.rom.model.load_weights_from_file(enc_path, dec_path, enc_path)
            self._loaded_model_label = model_info.get("label", "Loaded")
            msg = f"Loaded: {self._loaded_model_label}"
            print(f"[DT-Engine] {msg}")
        except Exception as exc:
            msg = f"Weight load failed: {exc}"
            self._loaded_model_label = "Load failed"
            print(f"[DT-Engine] {msg}")

        self.reset()
        return msg

    @staticmethod
    def _detect_transition_type_from_keys(sd: dict) -> str:
        """Detect transition type from state_dict keys (ported from testing dashboard)."""
        keys = set(sd.keys()) if isinstance(sd, dict) else set()
        has = lambda pat: any(pat in k for k in keys)

        if "sindy_coefficients" in keys:
            return "sindy"
        if has("cde_func.net."):
            return "neural_cde"
        if has("drift_net.") and has("diffusion_net."):
            return "latent_sde"
        if has("temporal_transformer."):
            return "transformer"
        if has("branch_net.") and has("trunk_net."):
            return "deeponet"
        if has("spectral_gates.") and has("rnn_lambda_real."):
            return "skolr"
        if "ren_H" in keys:
            return "ren"
        if "K" in keys and has("aft.aft_W_q"):
            return "koopman_aft"
        if "U_skew_params" in keys and "sigma_raw" in keys:
            return "dissipative_koopman"
        if "A_bilinear" in keys:
            return "bilinear_koopman"
        if has("lift_net.") and "exp_r_real" in keys:
            return "isfno"
        if "A_skew_params" in keys:
            return "ct_koopman"
        if has("H_net.") and "B_ctrl" in keys:
            return "hamiltonian"
        if has("lstm_cells."):
            return "lstm"
        if has("gru_cells."):
            return "gru"
        if "A_log" in keys and "delta_proj.weight" in keys and "out_proj.weight" in keys:
            return "mamba2"
        if "A_log" in keys and "delta_proj.weight" in keys:
            return "mamba"
        if has("selector.") and "Kt_layer.weight" in keys:
            return "deep_koopman"
        if ("eig_raw" in keys or "eig_mag_raw" in keys) and not has("selector."):
            return "stable_koopman"
        if "K" in keys and "At_layer.weight" not in keys:
            return "koopman"
        if has("selector.") and "alpha_layer.weight" in keys and "V_real" in keys:
            return "s5"
        if has("selector.") and "alpha_layer.weight" in keys and "U_real_layer.weight" in keys:
            return "s4d_dplr"
        if has("selector.") and "alpha_layer.weight" in keys:
            return "s4d"
        if has("selector.") and "nu_layer.weight" in keys:
            return "clru"
        if has("ode_func.net."):
            return "nonlinear"
        if has("trans_encoder.") and "At_layer.weight" in keys:
            return "linear"
        return "linear"

    # ==================================================================
    # Spatial denormalization
    # ==================================================================
    def _build_spatial_denorm(self) -> List[Optional[Dict]]:
        specs: List[Optional[Dict]] = [None] * self.n_channels
        for ch_idx, var_name in enumerate(_CHANNEL_VARS):
            if ch_idx >= self.n_channels:
                break
            if var_name in self.norm_params:
                specs[ch_idx] = self.norm_params[var_name]
        return specs

    def denormalize_state(self, state_norm: np.ndarray) -> np.ndarray:
        out = state_norm.copy()
        for ch, spec in enumerate(self._spatial_denorm):
            if spec is None or ch >= out.shape[0]:
                continue
            ntype = spec.get("type", "minmax")
            if ntype == "log":
                log_val = out[ch] * (spec["log_max"] - spec["log_min"]) + spec["log_min"]
                out[ch] = np.exp(log_val) - spec.get("epsilon", 1e-8) + spec.get("data_shift", 0)
            elif ntype != "none":
                out[ch] = out[ch] * (spec["max"] - spec["min"]) + spec["min"]
        np.clip(out, 0.0, None, out=out)
        # Saturations (ch 0 = SG) and porosity (ch 3) are fractions in [0, 1]
        for ch_idx in (0, 3):
            if ch_idx < out.shape[0]:
                np.clip(out[ch_idx], 0.0, 1.0, out=out[ch_idx])
        return out

    # ==================================================================
    # Inactive mask / initial state / normalization
    # ==================================================================
    def _load_inactive_mask(self, cfg):
        gnn_cfg = cfg.config.get("gnn", {})
        mask_rel = gnn_cfg.get("mask_file_path", "sr3_batch_output/inactive_cell_locations.h5")
        mask_path = None
        for c in [_ROM_DIR / mask_rel, _PROJECT_ROOT / mask_rel]:
            if c.exists():
                mask_path = c
                break
        if mask_path is None:
            loss_rel = cfg.config.get("loss", {}).get("mask_file_path", "")
            if loss_rel:
                alt = _ROM_DIR / loss_rel
                if alt.exists():
                    mask_path = alt
        if mask_path is None:
            return
        try:
            with h5py.File(str(mask_path), "r") as f:
                raw = f["active_mask"][...]
            if raw.ndim == 4:
                self._active_mask_4d = raw.astype(bool)
                self.active_mask = self._active_mask_4d[0]
                print(f"[DT-Engine] Mask: {raw.shape[0]} cases, {int(self.active_mask.sum())}/{self.active_mask.size} active (case 0)")
            else:
                self.active_mask = raw.astype(bool)
                print(f"[DT-Engine] Mask: {int(self.active_mask.sum())}/{self.active_mask.size} active")
        except Exception as exc:
            print(f"[DT-Engine] Mask load failed: {exc}")

    def _load_corner_point_grid(self):
        """Parse corner-point geometry from the CMG .dat file if configured."""
        cp_cfg = self.dt_cfg.get("corner_point", {})
        if not cp_cfg.get("enabled", False):
            return
        dat_rel = cp_cfg.get("dat_file", "simulation_data_file.dat")
        dat_path = _DIGITAL_TWIN_DIR / dat_rel
        if not dat_path.exists():
            print(f"[DT-Engine] Corner-point .dat not found: {dat_path}")
            return
        try:
            cache_rel = cp_cfg.get("cache_file", ".corner_point_cache.npy")
            cache_path = str(_DIGITAL_TWIN_DIR / cache_rel)
            z_exag = cp_cfg.get("z_exaggeration", 30.0)

            corners = load_or_parse_corners(
                str(dat_path), cache_path=cache_path,
                ni=self.nx, nj=self.ny, nk=self.nz,
            )
            from DigitalTwin.visualization.corner_point_grid import CornerPointGrid
            self.corner_point_grid = CornerPointGrid(
                corners, z_exaggeration=z_exag, negate_y=True,
            )
            bb = self.corner_point_grid.bounding_box
            print(f"[DT-Engine] Corner-point grid loaded: "
                  f"X [{bb['x'][0]:.0f}, {bb['x'][1]:.0f}] "
                  f"Y [{bb['y'][0]:.0f}, {bb['y'][1]:.0f}] "
                  f"Z [{bb['z'][0]:.0f}, {bb['z'][1]:.0f}] (z_exag={z_exag})")
        except Exception as exc:
            print(f"[DT-Engine] Corner-point grid failed: {exc}")

    def set_case(self, case_idx: int):
        """Update active mask and initial state for a specific simulation case."""
        self._current_case_idx = case_idx
        if self._active_mask_4d is not None:
            idx = min(case_idx, self._active_mask_4d.shape[0] - 1)
            self.active_mask = self._active_mask_4d[idx]
        if self._all_initial_states is not None:
            idx = min(case_idx, self._all_initial_states.shape[0] - 1)
            self._default_initial_state = self._all_initial_states[idx:idx+1]

    def _load_initial_state(self):
        """Load timestep-0 from raw spatial H5 files for all 1000 simulation cases.

        Reads each channel file, takes ``[:, 0, :, :, :]`` (all cases, first
        timestep), normalizes with the same parameters used during training,
        and stacks into ``(num_cases, n_channels, Nx, Ny, Nz)``.
        """
        raw_dir = _ROM_DIR / "sr3_batch_output"
        # Channel files in training order: SG(0), PRES(1), PERMI(2), POROS(3)
        channel_files = [
            ("SG",    raw_dir / "batch_spatial_properties_SG.h5"),
            ("PRES",  raw_dir / "batch_spatial_properties_PRES.h5"),
            ("PERMI", raw_dir / "batch_spatial_properties_PERMI.h5"),
            ("POROS", raw_dir / "batch_spatial_properties_POROS.h5"),
        ]

        # Check all files exist
        missing = [name for name, path in channel_files if not path.exists()]
        if missing:
            print(f"[DT-Engine] Raw files missing for: {missing} -- falling back to processed data")
            self._load_initial_state_from_processed()
            return

        try:
            channels = []
            for var_name, fpath in channel_files:
                with h5py.File(str(fpath), "r") as f:
                    raw = f["data"][:, 0, :, :, :]  # (1000, Nx, Ny, Nz) -- timestep 0
                # Normalize exactly as training pipeline does
                raw = self._normalize_raw(raw, var_name)
                channels.append(raw)

            # Stack: (1000, 4, Nx, Ny, Nz)
            stacked = np.stack(channels, axis=1).astype(np.float32)
            self._all_initial_states = torch.tensor(stacked)
            self.num_cases = stacked.shape[0]
            self._default_initial_state = self._all_initial_states[0:1]
            print(f"[DT-Engine] {self.num_cases} simulation cases loaded from raw data")
        except Exception as exc:
            print(f"[DT-Engine] Raw data load failed: {exc} -- falling back to processed data")
            self._load_initial_state_from_processed()

    def _normalize_raw(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize raw data using the same method as training."""
        if var_name not in self.norm_params:
            return data
        p = self.norm_params[var_name]
        ntype = p.get("type", "minmax")
        if ntype == "log":
            eps = p.get("epsilon", 1e-8)
            shift = p.get("data_shift", 0)
            log_data = np.log(data - shift + eps)
            return (log_data - p["log_min"]) / (p["log_max"] - p["log_min"])
        elif ntype != "none":
            return (data - p["min"]) / (p["max"] - p["min"])
        return data

    def _load_initial_state_from_processed(self):
        """Fallback: load from processed H5 (shuffled training samples)."""
        proc_dir = _ROM_DIR / "processed_data"
        if not proc_dir.exists():
            return
        h5_files = sorted(proc_dir.glob("processed_data_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not h5_files:
            return
        try:
            with h5py.File(str(h5_files[0]), "r") as f:
                if "train" in f and "STATE" in f["train"]:
                    grp = f["train"]["STATE"]
                    key0 = sorted(grp.keys())[0]
                    data = grp[key0][...]
                    self._all_initial_states = torch.tensor(data, dtype=torch.float32)
                    self.num_cases = min(data.shape[0], 1000)
                    self._default_initial_state = self._all_initial_states[0:1]
                    print(f"[DT-Engine] Fallback: {self.num_cases} states from processed data")
        except Exception as exc:
            print(f"[DT-Engine] Processed data load failed: {exc}")

    def _load_normalization_parameters(self):
        """Load and merge normalization parameters from all available files.

        The ROM training file (from ``processed_data/``) contains the correct
        spatial-channel parameters (SG, PRES, PERMI, POROS) that match what the
        model was trained with.  The RL file may have control/observation
        variables (BHP, GASRATSC, WATRATSC) that the ROM file lacks.

        Strategy: load ALL files oldest-first so that the ROM training file
        (which is older) provides the spatial baseline, then newer RL files
        can ADD control/observation variables without overwriting spatial
        channels.  Within spatial_channels specifically, we prefer the file
        that covers ALL 4 required channels.
        """
        search_dirs = [str(_ROM_DIR / "processed_data"), str(_RL_DIR), str(_PROJECT_ROOT)]
        json_files = []
        for d in search_dirs:
            json_files.extend(globmod.glob(str(Path(d) / "normalization_parameters_*.json")))
        json_files = list(dict.fromkeys(str(Path(f).resolve()) for f in json_files))
        if not json_files:
            return

        required_spatial = {"SG", "PRES", "PERMI", "POROS"}

        # Find the best file for spatial channels (the one that has all 4)
        best_spatial_file = None
        for fpath in json_files:
            with open(fpath, "r") as f:
                cfg = json.load(f)
            spatial_vars = set(cfg.get("spatial_channels", {}).keys())
            if required_spatial.issubset(spatial_vars):
                if best_spatial_file is None:
                    best_spatial_file = fpath
                elif Path(fpath).stat().st_mtime > Path(best_spatial_file).stat().st_mtime:
                    best_spatial_file = fpath

        # Load spatial channels from the best file first
        if best_spatial_file:
            with open(best_spatial_file, "r") as f:
                cfg = json.load(f)
            for var_name, info in cfg.get("spatial_channels", {}).items():
                self.norm_params[var_name] = self._to_numbers(info["parameters"])
            for section in ("control_variables", "observation_variables"):
                for var_name, info in cfg.get(section, {}).items():
                    self.norm_params[var_name] = self._to_numbers(info["parameters"])
            print(f"[DT-Engine] Spatial normalization from {best_spatial_file}")

        # Merge control/observation variables from other files (don't overwrite spatial)
        json_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        for fpath in json_files:
            if fpath == best_spatial_file:
                continue
            with open(fpath, "r") as f:
                cfg = json.load(f)
            for section in ("control_variables", "observation_variables"):
                for var_name, info in cfg.get(section, {}).items():
                    if var_name not in self.norm_params:
                        self.norm_params[var_name] = self._to_numbers(info["parameters"])
            # Only fill in MISSING spatial channels (never overwrite)
            for var_name, info in cfg.get("spatial_channels", {}).items():
                if var_name not in self.norm_params:
                    self.norm_params[var_name] = self._to_numbers(info["parameters"])
            if not best_spatial_file:
                print(f"[DT-Engine] Normalization fallback from {fpath}")

        loaded = set(self.norm_params.keys())
        missing = required_spatial - loaded
        if missing:
            print(f"[DT-Engine] WARNING: missing spatial norm params: {missing}")

    @staticmethod
    def _to_numbers(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v) if isinstance(v, str) else v
            except (ValueError, TypeError):
                out[k] = v
        return out

    # ==================================================================
    # Action mapping / observation denormalization
    # ==================================================================
    def _map_action_to_rom(self, action_01: torch.Tensor,
                           rl_source: bool = False) -> torch.Tensor:
        """Map [0,1] controls to ROM-normalized space.

        Two paths depending on the source of the action:

        * **Manual sliders** (``rl_source=False``): the slider already
          represents the full ROM training range, so its value IS the
          ROM-normalized control (identity mapping).

        * **RL agent** (``rl_source=True``): the policy was trained with
          restricted action ranges.  Its [0,1] output maps to a narrow
          physical sub-range first, then gets re-normalized to the full
          ROM training range — exactly matching the RL environment's
          ``_map_dashboard_action_to_rom_input``.
        """
        if not rl_source:
            return action_01.clone()

        bhp_lo = self.restricted_action_ranges["producer_bhp"]["min"]
        bhp_hi = self.restricted_action_ranges["producer_bhp"]["max"]
        gas_lo = self.restricted_action_ranges["gas_injection"]["min"]
        gas_hi = self.restricted_action_ranges["gas_injection"]["max"]

        physical = action_01.clone()
        physical[:, 0:3] = action_01[:, 0:3] * (bhp_hi - bhp_lo) + bhp_lo
        physical[:, 3:6] = action_01[:, 3:6] * (gas_hi - gas_lo) + gas_lo

        rom = physical.clone()
        if "BHP" in self.norm_params:
            p = self.norm_params["BHP"]
            rom[:, 0:3] = (physical[:, 0:3] - p["min"]) / (p["max"] - p["min"])
        if "GASRATSC" in self.norm_params:
            p = self.norm_params["GASRATSC"]
            rom[:, 3:6] = (physical[:, 3:6] - p["min"]) / (p["max"] - p["min"])
        return rom

    def _action_to_physical(self, action_01: torch.Tensor,
                            rl_source: bool = False) -> torch.Tensor:
        """Convert [0,1] actions to physical units for display/logging.

        Uses the same source distinction as ``_map_action_to_rom``:
        manual sliders map across the full ROM training range; RL
        actions map across the restricted RL range.
        """
        physical = action_01.clone()
        if rl_source:
            bhp_lo = self.restricted_action_ranges["producer_bhp"]["min"]
            bhp_hi = self.restricted_action_ranges["producer_bhp"]["max"]
            gas_lo = self.restricted_action_ranges["gas_injection"]["min"]
            gas_hi = self.restricted_action_ranges["gas_injection"]["max"]
        else:
            bhp_lo = float(self.norm_params.get("BHP", {}).get("min", self.restricted_action_ranges["producer_bhp"]["min"]))
            bhp_hi = float(self.norm_params.get("BHP", {}).get("max", self.restricted_action_ranges["producer_bhp"]["max"]))
            gas_lo = float(self.norm_params.get("GASRATSC", {}).get("min", self.restricted_action_ranges["gas_injection"]["min"]))
            gas_hi = float(self.norm_params.get("GASRATSC", {}).get("max", self.restricted_action_ranges["gas_injection"]["max"]))
        physical[:, 0:3] = action_01[:, 0:3] * (bhp_hi - bhp_lo) + bhp_lo
        physical[:, 3:6] = action_01[:, 3:6] * (gas_hi - gas_lo) + gas_lo
        return physical

    def _denorm_obs(self, yobs: torch.Tensor) -> torch.Tensor:
        yobs = torch.clamp(yobs, 0.0, 1.0)
        out = yobs.clone()
        mapping = [
            (range(0, self.num_inj), "BHP"),
            (range(self.num_inj, self.num_inj + self.num_prod), "GASRATSC"),
            (range(self.num_inj + self.num_prod, self.num_inj + 2 * self.num_prod), "WATRATSC"),
        ]
        for idx_range, key in mapping:
            if key not in self.norm_params:
                continue
            p = self.norm_params[key]
            ntype = p.get("type", "minmax")
            for i in idx_range:
                if ntype == "log":
                    log_val = out[:, i] * (p["log_max"] - p["log_min"]) + p["log_min"]
                    out[:, i] = torch.exp(log_val) - p.get("epsilon", 1e-8) + p.get("data_shift", 0)
                elif ntype != "none":
                    out[:, i] = out[:, i] * (p["max"] - p["min"]) + p["min"]
        return torch.clamp(out, min=0.0)

    # ==================================================================
    # RL Agent
    # ==================================================================
    _AGENT_MAP = {
        'SAC':  ('RL_Refactored.agent.sac_agent',  'SAC'),
        'DDPG': ('RL_Refactored.agent.ddpg_agent', 'DDPG'),
        'TD3':  ('RL_Refactored.agent.td3_agent',  'TD3'),
        'PPO':  ('RL_Refactored.agent.ppo_agent',  'PPO'),
    }

    def scan_rl_policies(self) -> list:
        """Scan RL_Refactored/checkpoints/ for saved agent checkpoints.

        Returns a list of dicts with keys: path, label, algorithm, filename.
        """
        ckpt_dir = _RL_DIR / "checkpoints"
        policies: list = []
        if not ckpt_dir.exists():
            return policies

        for fpath in sorted(ckpt_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if fpath.is_dir() or fpath.suffix == '.pkl':
                continue
            algo = self._detect_algorithm(fpath)
            if algo is None:
                continue
            label = f"{algo} | {fpath.name}"
            policies.append({
                "path": str(fpath),
                "label": label,
                "algorithm": algo,
                "filename": fpath.name,
            })
        return policies

    @staticmethod
    def _detect_algorithm(fpath: Path) -> Optional[str]:
        """Detect algorithm type from checkpoint file."""
        fname_lower = fpath.name.lower()
        for prefix, algo in [('sac_', 'SAC'), ('ddpg_', 'DDPG'), ('td3_', 'TD3'), ('ppo_', 'PPO')]:
            if fname_lower.startswith(prefix):
                return algo

        try:
            ckpt = torch.load(str(fpath), map_location='cpu', weights_only=False)
            arch = ckpt.get('architecture', {})
            algo_str = arch.get('algorithm_type', '').upper()
            if algo_str in ('SAC', 'DDPG', 'TD3', 'PPO'):
                return algo_str
            if 'policy_state_dict' in ckpt:
                if 'critic_target_state_dict' in ckpt and 'policy_target_state_dict' not in ckpt:
                    return 'SAC'
                if 'policy_target_state_dict' in ckpt:
                    return 'DDPG'
                return 'SAC'
        except Exception:
            pass
        return None

    def load_rl_agent(self, algorithm_type: str = 'SAC',
                      checkpoint_path: str = None) -> bool:
        """Load an RL agent of the given algorithm type from a checkpoint."""
        algo_key = algorithm_type.upper()
        if algo_key not in self._AGENT_MAP:
            print(f"[DT-Engine] Unknown RL algorithm: {algorithm_type}")
            return False

        mod_path, cls_name = self._AGENT_MAP[algo_key]
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            AgentClass = getattr(mod, cls_name)
            latent_dim = self.rom_config.model["latent_dim"]
            self.rl_agent = AgentClass(latent_dim, self.u_dim, self.rl_config)
            self._rl_agent_loaded = True
            self._rl_algorithm_type = algo_key
            if checkpoint_path and Path(checkpoint_path).exists():
                self.rl_agent.load_checkpoint(str(checkpoint_path), evaluate=True)
            print(f"[DT-Engine] {algo_key} agent loaded from {checkpoint_path}")
            return True
        except Exception as exc:
            print(f"[DT-Engine] RL agent failed ({algo_key}): {exc}")
            import traceback; traceback.print_exc()
            self.rl_agent = None
            self._rl_agent_loaded = False
            self._rl_algorithm_type = None
            return False

    def get_rl_action(self) -> Optional[np.ndarray]:
        if self.rl_agent is None or self.current_latent is None:
            return None
        with torch.no_grad():
            action = self.rl_agent.select_action(self.current_latent, evaluate=True)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return np.clip(action.flatten()[:self.u_dim], 0.0, 1.0)

    # ==================================================================
    # Encode helper (works for all model types)
    # ==================================================================
    def _encode_to_latent(self, spatial: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.rom.model, "encode_initial"):
                return self.rom.model.encode_initial(spatial)
            enc_out = self.rom.model.encoder(spatial)
            return enc_out[0] if isinstance(enc_out, tuple) else enc_out

    def _decode_to_spatial(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to spatial.

        Handles both Multimodal and GNN models (which have
        ``static_latent_dim``) as well as plain CNN models.  Static
        channels (permeability, porosity) are taken from the **initial**
        state captured at ``reset()``.
        """
        with torch.no_grad():
            model = self.rom.model
            if hasattr(model, "static_latent_dim"):
                z_static = latent[:, :model.static_latent_dim]
                z_dynamic = latent[:, model.static_latent_dim:]
                if hasattr(model, "_concat_for_decoder"):
                    z_for_dec = model._concat_for_decoder(z_static, z_dynamic)
                else:
                    z_for_dec = torch.cat([z_static, z_dynamic], dim=-1)
                if hasattr(model, "_decode_cnn"):
                    x_dyn = model._decode_cnn(z_for_dec)
                else:
                    x_dyn = model.decoder(z_for_dec)
                if hasattr(model, "_reassemble"):
                    if self._initial_static_channels is not None:
                        return model._reassemble(x_dyn, self._initial_static_channels)
                    if self.current_spatial_state is not None:
                        x_static, _ = model._split_channels(self.current_spatial_state)
                        return model._reassemble(x_dyn, x_static)
                return x_dyn
            return model.decoder(latent)

    # ==================================================================
    # Public API
    # ==================================================================
    def reset(self, initial_spatial: torch.Tensor = None) -> np.ndarray:
        self.step_index = 0
        self.state_mgr.reset()

        if initial_spatial is None:
            if self._default_initial_state is not None:
                initial_spatial = self._default_initial_state.clone()
            else:
                initial_spatial = torch.zeros(1, self.n_channels, self.nx, self.ny, self.nz)
        if initial_spatial.dim() == 4:
            initial_spatial = initial_spatial.unsqueeze(0)

        self.current_spatial_state = initial_spatial.to(self.device)
        self.current_latent = self._encode_to_latent(self.current_spatial_state)

        if self.rom is not None and hasattr(self.rom.model, "static_channels"):
            self._initial_static_channels = self.current_spatial_state[
                :, self.rom.model.static_channels, :, :, :
            ].clone()
        else:
            self._initial_static_channels = None

        state_phys = self.denormalize_state(self.current_spatial_state[0].detach().cpu().numpy())
        self.state_mgr.record(state_phys, np.zeros(self.num_inj + 2 * self.num_prod), np.zeros(self.u_dim))
        return state_phys

    def step(self, controls_01: np.ndarray, rl_source: bool = False) -> Dict:
        if self.current_spatial_state is None:
            raise RuntimeError("Call reset() before step().")

        self.step_index += 1
        action_01 = torch.tensor(controls_01, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_rom = self._map_action_to_rom(action_01, rl_source=rl_source)
        action_phys = self._action_to_physical(action_01, rl_source=rl_source)

        dt = torch.ones(1, 1, dtype=torch.float32, device=self.device) * self.dt_value
        dummy_obs = torch.zeros(1, self.num_inj + 2 * self.num_prod, device=self.device)

        with torch.no_grad():
            if self.prediction_mode == "latent" and self.current_latent is not None:
                next_latent, yobs_norm = self.rom.model.predict_latent(
                    self.current_latent, dt, action_rom
                )
                self.current_latent = next_latent
                next_spatial = self._decode_to_spatial(next_latent)
                self.current_spatial_state = next_spatial
            else:
                inputs = (self.current_spatial_state, action_rom, dummy_obs, dt)
                next_spatial, yobs_norm = self.rom.predict(inputs)
                self.current_spatial_state = next_spatial
                self.current_latent = self._encode_to_latent(next_spatial)

        yobs_phys = self._denorm_obs(yobs_norm)
        next_spatial_viz = torch.clamp(next_spatial, min=0.0)
        state_phys = self.denormalize_state(next_spatial_viz[0].detach().cpu().numpy())
        obs_np = yobs_phys[0].detach().cpu().numpy()
        ctrl_np = action_phys[0].detach().cpu().numpy()

        sg = state_phys[0]
        pr = state_phys[1]
        print(f"[DT Step {self.step_index}/{self.max_steps}] "
              f"SG=[{sg.min():.3f},{sg.max():.3f}] "
              f"PRES=[{pr.min():.0f},{pr.max():.0f}] "
              f"mode={self.prediction_mode}")

        self.state_mgr.record(state_phys, obs_np, ctrl_np)

        return {
            "state_3d": state_phys,
            "well_observations": obs_np,
            "controls_physical": ctrl_np,
            "kpis": self.state_mgr.compute_kpis(),
            "step": self.step_index,
            "done": self.step_index >= self.max_steps,
        }

    def get_current_state(self) -> Optional[np.ndarray]:
        if self.current_spatial_state is None:
            return None
        return self.denormalize_state(self.current_spatial_state[0].detach().cpu().numpy())

    @property
    def grid_spacing(self) -> Tuple[float, float, float]:
        g = self.dt_cfg.get("grid", {})
        return (g.get("dx", 20.0), g.get("dy", 10.0), g.get("dz", 5.0))

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)
