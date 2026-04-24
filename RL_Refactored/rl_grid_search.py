#%%!/usr/bin/env python3
"""
RL Grid Search Script
=====================
Runs grid search over RL Training, RL Evaluation, and Classical Optimizers
(GA, DE, PSO, CMA-ES, LS-SQP-StoSAG, L-BFGS-B, Adam, Dual-Annealing, Basin-Hopping)
with configurable parameter grids, ROM model selection, metadata collection,
and independent plotting.

Cell Structure:
  Cell 1 - Setup & ROM Model Selection
  Cell 2 - Grid Search Parameter Definitions & Execution
  Cell 3 - Plotting (independent of runs, loads from saved JSON)
"""

# =============================================================================
# CELL 1: SETUP & ROM MODEL SELECTION
# =============================================================================
#%%
import os
import sys
import re
import json
import gc
import time
import glob
import itertools
import numpy as np
import torch
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

project_root = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
rom_refactored_path = project_root / 'ROM_Refactored'
if str(rom_refactored_path) not in sys.path:
    sys.path.insert(0, str(rom_refactored_path))

from RL_Refactored.utilities import print_hardware_info, Config
from ROM_Refactored.model.training.rom_wrapper import ROMWithE2C
from ROM_Refactored.utilities.config_loader import Config as ROMConfig

_raw_device = print_hardware_info()


def _check_cuda_conv3d() -> bool:
    """Test whether Conv3d works on CUDA (fails on some Windows PyTorch builds)."""
    if not torch.cuda.is_available():
        return False
    try:
        conv = torch.nn.Conv3d(1, 1, 3, padding=1).cuda()
        x = torch.randn(1, 1, 4, 4, 4).cuda()
        _ = conv(x)
        del conv, x
        torch.cuda.empty_cache()
        return True
    except (RuntimeError, NotImplementedError):
        return False


if _raw_device.type == 'cuda' and not _check_cuda_conv3d():
    print("\nWARNING: CUDA Conv3d is not supported on this PyTorch build.")
    print("         Falling back to CPU for all operations.\n")
    DEVICE = torch.device('cpu')
else:
    DEVICE = _raw_device

# Output paths
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_search_results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# ROM Model Discovery (adapted from optimizers/dashboard_config.py)
# ---------------------------------------------------------------------------

def scan_rom_models(rom_folder: str) -> List[Dict[str, Any]]:
    """Discover ROM model triplets (encoder/decoder/transition) in a folder."""
    if not os.path.exists(rom_folder):
        print(f"ROM folder not found: {rom_folder}")
        return []

    models = {}
    for filename in os.listdir(rom_folder):
        if not filename.endswith('.h5') or 'encoder' not in filename:
            continue
        decoder_file = os.path.join(rom_folder, filename.replace('encoder', 'decoder'))
        transition_file = os.path.join(rom_folder, filename.replace('encoder', 'transition'))
        if os.path.exists(decoder_file) and os.path.exists(transition_file):
            info = _parse_model_filename(filename)
            base = filename.replace('encoder', 'MODEL')
            models[base] = {
                'name': _model_display_name(filename, info),
                'encoder': os.path.join(rom_folder, filename),
                'decoder': decoder_file,
                'transition': transition_file,
                'info': info,
            }
    return list(models.values())


def _parse_model_filename(filename: str) -> Dict:
    info = {}
    for param, pattern in {
        'latent_dim': r'ld(\d+)', 'batch_size': r'bs(\d+)',
        'nsteps': r'ns(\d+)', 'channels': r'ch(\d+)',
        'run': r'run(\d+)', 'residual_blocks': r'_rb(\d+)',
    }.items():
        m = re.search(pattern, filename)
        if m:
            info[param] = int(m.group(1))
    ehd = re.search(r'_ehd([\d-]+)', filename)
    if ehd:
        try:
            info['encoder_hidden_dims'] = [int(d) for d in ehd.group(1).split('-')]
        except ValueError:
            pass
    if '_mmT' in filename:
        info['multimodal'] = True
    elif '_mmF' in filename:
        info['multimodal'] = False
    if '_gnnT' in filename:
        info['gnn'] = True
    elif '_gnnF' in filename:
        info['gnn'] = False
    if '_fnoT' in filename:
        info['fno'] = True
    elif '_fnoF' in filename:
        info['fno'] = False
    if '_memT' in filename or '_mem-' in filename:
        info['multi_embedding'] = True
        m_preset = re.search(r'_mem-([a-zA-Z0-9-]+)', filename)
        if m_preset:
            # Restore underscores that were replaced with hyphens at write time
            info['mem_preset'] = m_preset.group(1).replace('-', '_')
    norm_m = re.search(r'_norm(ba|gd)', filename)
    if norm_m:
        info['norm_type'] = 'batchnorm' if norm_m.group(1) == 'ba' else 'gdn'
    trn_m = re.search(r'_trn([A-Z]+)', filename)
    if trn_m:
        info['transition_type'] = trn_m.group(1).lower()
    return info


def _model_display_name(filename: str, info: Dict) -> str:
    parts = []
    if info.get('gnn'):
        parts.append("GNN")
    if info.get('fno'):
        parts.append("FNO")
    if 'batch_size' in info:
        parts.append(f"bs={info['batch_size']}")
    if 'latent_dim' in info:
        parts.append(f"ld={info['latent_dim']}")
    if 'channels' in info:
        parts.append(f"ch={info['channels']}")
    if 'run' in info:
        parts.append(f"run={info['run']}")
    return f"Model ({', '.join(parts)})" if parts else filename


# ---------------------------------------------------------------------------
# ROM Loading
# ---------------------------------------------------------------------------

def _detect_multimodal(encoder_file: str) -> bool:
    try:
        payload = torch.load(encoder_file, map_location='cpu', weights_only=False)
        return isinstance(payload, dict) and payload.get('_multimodal', False)
    except Exception:
        return False


def _detect_gnn(encoder_file: str) -> bool:
    try:
        payload = torch.load(encoder_file, map_location='cpu', weights_only=False)
        return isinstance(payload, dict) and payload.get('_gnn', False)
    except Exception:
        return False


def _detect_fno(encoder_file: str) -> bool:
    try:
        payload = torch.load(encoder_file, map_location='cpu', weights_only=False)
        return isinstance(payload, dict) and payload.get('_fno', False)
    except Exception:
        return False


def _detect_multi_embedding(encoder_file: str) -> bool:
    try:
        payload = torch.load(encoder_file, map_location='cpu', weights_only=False)
        return isinstance(payload, dict) and bool(payload.get('_multi_embedding', False))
    except Exception:
        return False


def _read_multi_embedding_branches(encoder_file: str):
    try:
        payload = torch.load(encoder_file, map_location='cpu', weights_only=False)
        if isinstance(payload, dict) and payload.get('_multi_embedding'):
            return payload.get('branches', None)
    except Exception:
        pass
    return None


def _detect_vae(encoder_file: str) -> bool:
    try:
        payload = torch.load(encoder_file, map_location='cpu', weights_only=False)
        if isinstance(payload, dict) and '_multimodal' in payload:
            return 'fc_logvar.weight' in payload.get('dynamic_encoder', {})
        return 'fc_logvar.weight' in payload
    except Exception:
        return False


def _detect_decoder_type(decoder_file: str) -> Optional[str]:
    try:
        sd = torch.load(decoder_file, map_location='cpu', weights_only=False)
        keys = set(sd.keys())
        has_deconv = any(k.startswith('deconv1') for k in keys)
        has_res = any(k.startswith('res_layers') for k in keys)
        if has_deconv and not has_res:
            return 'standard'
        if has_res and not has_deconv:
            return 'smooth'
    except Exception:
        pass
    return None


def _detect_transition_type(transition_file: str) -> str:
    """Detect transition model type from saved weight keys."""
    try:
        sd = torch.load(transition_file, map_location='cpu', weights_only=False)
        has_selector = any(k.startswith('selector.') for k in sd)
        has_nu_layer = 'nu_layer.weight' in sd
        has_alpha_layer = 'alpha_layer.weight' in sd
        has_At_layer = 'At_layer.weight' in sd
        has_V_real = 'V_real' in sd
        has_U_real = 'U_real_layer.weight' in sd
        has_K_param = 'K' in sd and not has_At_layer
        has_A_ct = 'A_skew_params' in sd
        has_attractor = any(k.startswith('attractor_net.') for k in sd)
        if has_attractor and has_alpha_layer:
            return 'accss'
        if has_A_ct:
            return 'ct_koopman'
        if has_K_param and not has_selector:
            return 'koopman'
        if has_selector and has_alpha_layer and has_V_real:
            return 's5'
        if has_selector and has_alpha_layer and has_U_real:
            return 's4d_dplr'
        if has_selector and has_alpha_layer:
            return 's4d'
        if has_selector and has_nu_layer:
            return 'clru'
    except Exception:
        pass
    return 'linear'


def _infer_encoder_hidden_dims(transition_file: str) -> List[int]:
    try:
        sd = torch.load(transition_file, map_location='cpu', weights_only=False)
        dims, idx = [], 0
        # Linear transition uses trans_encoder.{idx}.0.weight
        while f'trans_encoder.{idx}.0.weight' in sd:
            dims.append(sd[f'trans_encoder.{idx}.0.weight'].shape[0])
            idx += 1
        if dims:
            return dims[:-1] if len(dims) > 1 else dims
        # CLRU transition uses selector.{idx}.0.weight (last layer is plain Linear)
        while f'selector.{idx}.0.weight' in sd:
            dims.append(sd[f'selector.{idx}.0.weight'].shape[0])
            idx += 1
        if dims:
            return dims
        return [200, 200]
    except Exception:
        return [200, 200]


def load_rom_model(model_entry: Dict, device: torch.device) -> Tuple[ROMWithE2C, ROMConfig]:
    """Load a ROM model from encoder/decoder/transition files."""
    rom_config_path = str(project_root / 'ROM_Refactored' / 'config.yaml')
    rom_config = ROMConfig(rom_config_path)
    info = model_entry.get('info', {})

    n_channels = info.get('channels', 2)
    rom_config.model['n_channels'] = n_channels
    if 'data' in rom_config.config and 'input_shape' in rom_config.config['data']:
        if isinstance(rom_config.config['data']['input_shape'], list):
            rom_config.config['data']['input_shape'][0] = n_channels
    if 'encoder' in rom_config.config and 'conv_layers' in rom_config.config['encoder']:
        if 'conv1' in rom_config.config['encoder']['conv_layers']:
            rom_config.config['encoder']['conv_layers']['conv1'][0] = n_channels
    if 'decoder' in rom_config.config and 'deconv_layers' in rom_config.config['decoder']:
        fc = rom_config.config['decoder']['deconv_layers'].get('final_conv')
        if fc and isinstance(fc, list) and len(fc) > 1 and fc[1] is not None:
            fc[1] = n_channels

    if 'latent_dim' in info:
        rom_config.model['latent_dim'] = info['latent_dim']
    if 'residual_blocks' in info:
        rom_config.config.setdefault('encoder', {})['residual_blocks'] = info['residual_blocks']
    ehd = info.get('encoder_hidden_dims') or _infer_encoder_hidden_dims(model_entry['transition'])
    rom_config.config.setdefault('transition', {})['encoder_hidden_dims'] = ehd

    trn_type = info.get('transition_type', _detect_transition_type(model_entry['transition']))
    rom_config.config.setdefault('transition', {})['type'] = trn_type

    if 'norm_type' in info:
        rom_config.config.setdefault('encoder', {})['norm_type'] = info['norm_type']
        rom_config.config.setdefault('decoder', {})['norm_type'] = info['norm_type']

    is_mem = info.get('multi_embedding', _detect_multi_embedding(model_entry['encoder']))
    is_gnn = info.get('gnn', _detect_gnn(model_entry['encoder']))
    is_fno = info.get('fno', _detect_fno(model_entry['encoder']))
    is_mm = info.get('multimodal', _detect_multimodal(model_entry['encoder']))

    if is_mem:
        # Multi-embedding multimodal takes priority and forces every other
        # backbone flag off. Always 4-channel by construction.
        is_gnn = False
        is_fno = False
        is_mm = False
        n_channels = 4
        rom_config.model['n_channels'] = n_channels
        if 'data' in rom_config.config and 'input_shape' in rom_config.config['data']:
            if isinstance(rom_config.config['data']['input_shape'], list):
                rom_config.config['data']['input_shape'][0] = n_channels
    elif is_gnn:
        is_mm = False
        is_fno = False
        n_channels = 4
        rom_config.model['n_channels'] = n_channels
        if 'data' in rom_config.config and 'input_shape' in rom_config.config['data']:
            if isinstance(rom_config.config['data']['input_shape'], list):
                rom_config.config['data']['input_shape'][0] = n_channels
    elif is_fno:
        is_mm = False
        n_channels = 4
        rom_config.model['n_channels'] = n_channels
        if 'data' in rom_config.config and 'input_shape' in rom_config.config['data']:
            if isinstance(rom_config.config['data']['input_shape'], list):
                rom_config.config['data']['input_shape'][0] = n_channels

    rom_config.config.setdefault('multi_embedding', {})['enable'] = is_mem
    rom_config.config.setdefault('gnn', {})['enable'] = is_gnn
    rom_config.config.setdefault('fno', {})['enable'] = is_fno
    rom_config.config.setdefault('multimodal', {})['enable'] = is_mm
    if is_mem:
        # Pull branch metadata from the encoder payload so the model can
        # rebuild the right per-branch encoders/decoders.
        branches = _read_multi_embedding_branches(model_entry['encoder'])
        if branches:
            rom_config.config['multi_embedding']['branches'] = branches
            # latent_dim must equal sum of branch latent_dims
            rom_config.model['latent_dim'] = sum(int(b.get('latent_dim', 0)) for b in branches)
            if any(b.get('encoder', {}).get('type') == 'fno'
                   or (b.get('decoder') and b['decoder'].get('type') == 'fno')
                   for b in branches):
                rom_config.config.setdefault('loss', {})['enable_invertibility_loss'] = True
    elif is_mm or is_fno:
        ld = rom_config.model.get('latent_dim', 128)
        sld = rom_config.config['multimodal'].get('static_latent_dim', 32)
        rom_config.config['multimodal']['dynamic_latent_dim'] = ld - sld

    dec_type = _detect_decoder_type(model_entry['decoder'])
    if dec_type:
        rom_config.config.setdefault('decoder', {})['type'] = dec_type

    vae = _detect_vae(model_entry['encoder'])
    rom_config.config.setdefault('model', {})['enable_vae'] = vae
    rom_config.config.setdefault('loss', {})['enable_kl_loss'] = vae

    rom_config._resolve_dynamic_values()
    rom_config.device = device

    my_rom = ROMWithE2C(rom_config).to(device)
    my_rom.model.load_weights_from_file(
        model_entry['encoder'], model_entry['decoder'], model_entry['transition']
    )
    my_rom.eval()
    print(f"ROM model loaded: {model_entry['name']}")
    return my_rom, rom_config


# ---------------------------------------------------------------------------
# Normalization Parameters
# ---------------------------------------------------------------------------

def load_normalization_params() -> Dict:
    norm_dir = project_root / 'ROM_Refactored' / 'processed_data'
    files = sorted(norm_dir.glob('normalization_parameters_*.json'), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No normalization parameter files found")
        return {}
    latest = files[-1]
    with open(latest) as f:
        raw = json.load(f)
    params: Dict[str, Dict] = {}
    for section, prefix in [('spatial_channels', ''), ('control_variables', 'ctrl_'), ('observation_variables', 'obs_')]:
        block = raw.get(section, {})
        if isinstance(block, dict):
            for name, vd in block.items():
                p = vd.get('parameters', vd)
                params[prefix + name] = {'min': float(p.get('min', 0)), 'max': float(p.get('max', 1)),
                                         'type': vd.get('normalization_type', 'minmax')}
        elif isinstance(block, list):
            for entry in block:
                name = entry.get('name', entry.get('variable', ''))
                p = entry.get('parameters', entry)
                params[prefix + name] = {'min': float(p.get('min', 0)), 'max': float(p.get('max', 1))}
    print(f"Loaded normalization params from {latest.name} ({len(params)} variables)")
    return params


# ---------------------------------------------------------------------------
# Z0 Generation
# ---------------------------------------------------------------------------

def generate_z0_options(rom_model: ROMWithE2C, n_channels: int,
                        norm_params: Dict, device: torch.device,
                        max_cases: int = 1000) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Encode all reservoir cases to latent space to get Z0 options."""
    state_folder = str(project_root / 'ROM_Refactored' / 'sr3_batch_output')
    ch_map = {2: ['SG', 'PRES'], 4: ['SG', 'PRES', 'PERMI', 'POROS']}
    channel_order = ch_map.get(n_channels, ['SG', 'PRES'])

    all_data = {}
    for sn in channel_order:
        sf = os.path.join(state_folder, f'batch_spatial_properties_{sn}.h5')
        with h5py.File(sf, 'r') as f:
            all_data[sn] = f['data'][:max_cases, 0, :, :, :]

    num_cases = min(max_cases, next(iter(all_data.values())).shape[0])
    batch_size = 100
    all_z0, all_spatial = [], []

    for start in range(0, num_cases, batch_size):
        end = min(start + batch_size, num_cases)
        channels = []
        for sn in channel_order:
            d = all_data[sn][start:end].astype(np.float32)
            if sn in norm_params:
                p = norm_params[sn]
                d = (d - p['min']) / (p['max'] - p['min'] + 1e-8)
            channels.append(torch.tensor(d).unsqueeze(1))
        spatial = torch.cat(channels, dim=1).to(device)
        all_spatial.append(spatial.cpu())
        with torch.no_grad():
            if hasattr(rom_model.model, 'encode_initial'):
                z0 = rom_model.model.encode_initial(spatial)
            elif hasattr(rom_model.model, 'static_encoder'):
                sc = rom_model.model.static_channels
                dc = rom_model.model.dynamic_channels
                zs, _, _ = rom_model.model.static_encoder(spatial[:, sc])
                zd, _, _ = rom_model.model.dynamic_encoder(spatial[:, dc])
                z0 = torch.cat([zs, zd], dim=-1)
            else:
                enc_out = rom_model.model.encoder(spatial)
                z0 = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        all_z0.append(z0.cpu())

    z0_all = torch.cat(all_z0, dim=0).to(device)
    spatial_all = torch.cat(all_spatial, dim=0).to(device)
    print(f"Generated {z0_all.shape[0]} Z0 options  (shape {z0_all.shape}, device {device})")
    return z0_all, spatial_all


# ---------------------------------------------------------------------------
# Action-range auto-detection
# ---------------------------------------------------------------------------

def auto_detect_action_ranges() -> Dict:
    data_dir = str(project_root / 'ROM_Refactored' / 'sr3_batch_output')
    ranges = {'producer_bhp': {'min': 1087.78, 'max': 1305.34},
              'gas_injection': {'min': 24720290.0, 'max': 100646896.0}}
    bhp_f = os.path.join(data_dir, 'batch_timeseries_data_BHP.h5')
    if os.path.exists(bhp_f):
        with h5py.File(bhp_f, 'r') as f:
            d = f['data'][:]
            if d.shape[2] >= 6:
                ranges['producer_bhp'] = {'min': float(np.min(d[:, :, 3:6])),
                                          'max': float(np.max(d[:, :, 3:6]))}
    gas_f = os.path.join(data_dir, 'batch_timeseries_data_GIWELL.h5')
    if os.path.exists(gas_f):
        with h5py.File(gas_f, 'r') as f:
            d = f['data'][:]
            if d.shape[2] >= 3:
                ranges['gas_injection'] = {'min': float(np.min(d[:, :, :3])),
                                           'max': float(np.max(d[:, :, :3]))}
    print(f"Action ranges: BHP [{ranges['producer_bhp']['min']:.2f}, {ranges['producer_bhp']['max']:.2f}] psi | "
          f"Gas [{ranges['gas_injection']['min']:.0f}, {ranges['gas_injection']['max']:.0f}] ft3/day")
    return ranges


# ---------------------------------------------------------------------------
# Interactive ROM Selection
# ---------------------------------------------------------------------------

def select_rom_model_interactive() -> Dict:
    """Present a simple numbered menu to select a ROM model."""
    rom_folder = str(project_root / 'ROM_Refactored' / 'saved_models')
    models = scan_rom_models(rom_folder)
    if not models:
        raise FileNotFoundError(f"No ROM models found in {rom_folder}")

    print("\n" + "=" * 60)
    print("Available ROM Models")
    print("=" * 60)
    for i, m in enumerate(models):
        print(f"  [{i}] {m['name']}")
    print("=" * 60)

    if len(models) == 1:
        choice = 0
        print(f"Auto-selecting the only available model: [{choice}]")
    else:
        try:
            choice = int(input(f"Select model [0-{len(models)-1}]: "))
        except (ValueError, EOFError):
            choice = 0
            print(f"Defaulting to model [{choice}]")

    selected = models[min(choice, len(models) - 1)]
    print(f"\nSelected: {selected['name']}")
    return selected


# ---- Run the selection ----
print("\n>>> CELL 1: Setup & ROM Model Selection\n")

NORM_PARAMS = load_normalization_params()
ACTION_RANGES = auto_detect_action_ranges()
SELECTED_ROM_ENTRY = select_rom_model_interactive()
ROM_MODEL, ROM_CONFIG = load_rom_model(SELECTED_ROM_ENTRY, DEVICE)
N_CHANNELS = SELECTED_ROM_ENTRY['info'].get('channels', 2)
Z0_OPTIONS, SPATIAL_STATES = generate_z0_options(ROM_MODEL, N_CHANNELS, NORM_PARAMS, DEVICE)

RL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
RL_CONFIG = Config(RL_CONFIG_PATH)

print("\nSetup complete.")


# =============================================================================
# CELL 2: GRID SEARCH PARAMETER DEFINITIONS & EXECUTION
# =============================================================================
#%%
"""
INSTRUCTIONS:
  - Edit the grids below to customise which optimizers and parameters to sweep.
  - Each value must be a *list* so that itertools.product works.
  - Run this cell to execute all grid combinations.
"""

# ---------------------------------------------------------------------------
# 2a  Parameter Grid Definitions
# ---------------------------------------------------------------------------

OPTIMIZER_GRID = {
    'optimizer_type': ['RL_Training'],
    # All available options:
    # 'RL_Training'           - SAC reinforcement learning training
    # 'RL_Evaluation'         - Evaluate trained RL checkpoint
    # 'GA'                    - Genetic Algorithm
    # 'Differential-Evolution'- Differential Evolution
    # 'PSO'                   - Particle Swarm Optimization
    # 'CMA-ES'               - Covariance Matrix Adaptation
    # 'LS-SQP-StoSAG'        - Line-Search SQP with stochastic gradients
    # 'L-BFGS-B'             - Quasi-Newton with limited-memory Hessian
    # 'Adam'                  - Adaptive Moment Estimation (per-variable adaptive LR)
    # 'Dual-Annealing'       - Simulated Annealing + local search
    # 'Basin-Hopping'        - Monte Carlo + local optimization
    # 'Hybrid'               - Two-stage: global search + local gradient refinement
}

SHARED_GRID = {
    # --- Shared across all optimizers ---
    'num_steps': [30],                            # Simulation timesteps (10-100)
    'init_strategy': ['midpoint'],                # 'midpoint', 'random', 'low', 'naive_zero', 'naive_max',
                                                  # 'naive_low_bhp_high_gas', 'naive_high_bhp_low_gas'
}

ECONOMIC_GRID = {
    # --- Economic Prices (Configuration Dashboard > Economics Tab) ---
    'gas_injection_revenue': [50.0],              # Gas injection credit ($/ton)
    'gas_injection_cost': [10.0],                 # Gas injection operating cost ($/ton)
    'water_production_penalty': [5.0],            # Water production penalty ($/barrel)
    'gas_production_penalty': [50.0],             # Gas production penalty ($/ton)
    'scale_factor': [1000000.0],                  # Reward normalization scale factor

    # --- Capital / Project Economics (Configuration Dashboard > Economics Tab) ---
    'years_before_project_start': [5],            # Pre-project development years
    'capital_cost_per_year': [100000000.0],        # Annual capital cost during pre-project ($)
}

# Per-optimizer parameter grids
RL_TRAINING_GRID = {
    # --- Algorithm Selection ---
    'algorithm_type': ['SAC'],                    # 'SAC', 'TD3', 'DDPG', 'PPO'

    # --- Network Architecture (Configuration Dashboard > RL Hyperparameters) ---
    'hidden_dim': [192],                          # Hidden layer size for Q-networks and policy
    'policy_type': ['gaussian'],                  # 'deterministic' or 'gaussian'
    'output_activation': ['sigmoid'],             # 'sigmoid' or 'tanh'

    # --- SAC Algorithm (Configuration Dashboard > RL Hyperparameters) ---
    'learning_rate_critic': [0.0001],             # Critic (Q-network) learning rate
    'learning_rate_policy': [0.0003],             # Policy (actor) learning rate
    'discount_factor': [0.986],                   # Gamma - discount for future rewards
    'soft_update_tau': [0.005],                   # Tau - soft update rate for target networks
    'entropy_alpha': [0.2],                       # Entropy regularization weight
    'automatic_entropy_tuning': [True],           # Auto-tune entropy coefficient
    'gradient_clipping': [True],                  # Enable gradient clipping
    'gradient_clip_max_norm': [10.0],             # Max gradient norm

    # --- Training (Configuration Dashboard > Training + Training Dashboard) ---
    'max_episodes': [100],                        # Total training episodes
    'max_steps_per_episode': [30],                # Steps per episode (= simulation timesteps)
    'replay_batch_size': [256],                   # Replay memory sampling batch size
    'replay_capacity': [100000],                  # Replay memory capacity
    'initial_exploration_steps': [0],             # Random steps before learning (config.yaml training.exploration_steps default)
    'updates_per_step': [1],                      # Gradient updates per env step

    # --- Action Variation (Configuration Dashboard > Action Variation Tab) ---
    'action_variation_enabled': [True],           # Enable exploration noise
    'action_variation_mode': ['adaptive'],        # 'adaptive', 'exploration', 'exploitation', 'minimal'
    'noise_decay_rate': [0.995],                  # Noise decay per episode
    'max_noise_std': [0.25],                      # Max noise std for early exploration
    'min_noise_std': [0.01],                      # Min noise std for fine-tuning
    'step_variation_amplitude': [0.15],           # Amplitude for within-episode variation

    # --- Enhanced Gaussian Policy (Configuration Dashboard > Action Variation Tab) ---
    'enhanced_gaussian_enabled': [False],         # Use Gaussian policy instead of deterministic
    'log_std_min': [-1.0],                        # Min log std for Gaussian policy
    'log_std_max': [1.0],                         # Max log std for Gaussian policy
    'enhanced_entropy_weight': [0.2],             # Entropy weight for Gaussian policy

    # --- Environment (config.yaml > rl_model.environment) ---
    'prediction_mode': ['state_based'],           # 'state_based' or 'latent_based'
    'env_noise_enable': [False],                  # Enable state noise
    'env_noise_std': [0.10],                      # Noise std for stochastic environment

    # --- Evaluation-Based Checkpoint (Training Dashboard) ---
    'eval_checkpoint_enabled': [False],           # Save checkpoints based on deterministic eval
    'eval_interval': [5],                         # Evaluate every N episodes
    'eval_num_cases': [10],                       # Z0 cases per evaluation
}

RL_EVALUATION_GRID = {
    'num_eval_cases': [100],                      # Number of Z0 cases to evaluate
    'deterministic': [True],                      # Deterministic (mean) actions vs stochastic
    # 'checkpoint_path': auto-discovered from checkpoints/ folder
    'baselines': [['Random', 'Midpoint', 'NaiveMaxGas', 'NaiveLowGas']],  # Baselines to compare
}

GA_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > GA)
    'population_size': [50],                      # Number of individuals in population (20-200)
    'crossover_prob': [0.7],                      # Crossover probability (0.0-1.0)
    'mutation_prob': [0.2],                       # Mutation probability per gene (0.01-0.5)
    'elitism': [2],                               # Elite individuals to preserve (0-10)
    'max_iterations': [100],                      # Maximum generations (10-500)
    'tolerance': [1e-6],                          # Convergence tolerance
}

DE_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > Differential Evolution)
    'popsize': [15],                              # Population size multiplier (5-50)
    'mutation': [0.8],                            # Mutation constant F (0.1-2.0)
    'recombination': [0.7],                       # Crossover probability CR (0.0-1.0)
    'strategy': ['best1bin'],                     # 'best1bin', 'best2bin', 'rand1bin', 'currenttobest1bin'
    'max_iterations': [100],                      # Maximum generations (10-500)
    'tolerance': [1e-6],                          # Convergence tolerance
}

PSO_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > PSO)
    'n_particles': [30],                          # Number of particles in swarm (10-100)
    'c1': [2.0],                                  # Cognitive parameter - personal best (0.5-4.0)
    'c2': [2.0],                                  # Social parameter - global best (0.5-4.0)
    'w': [0.7],                                   # Inertia weight / momentum (0.1-1.0)
    'max_iterations': [100],                      # Maximum iterations (10-500)
    'tolerance': [1e-6],                          # Convergence tolerance
}

CMAES_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > CMA-ES)
    'sigma0': [0.3],                              # Initial step size (0.1-0.5)
    'popsize': [0],                               # Population size (0 = automatic)
    'max_iterations': [100],                      # Maximum generations (10-500)
    'tolerance': [1e-6],                          # Convergence tolerance
}

LSSQP_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > LS-SQP-StoSAG)
    'gradient_type': ['spsa'],                    # 'spsa', 'stosag', 'fd_forward', 'fd_central'
    'perturbation_size': [0.01],                  # Relative perturbation for gradient (0.001-0.1)
    'spsa_num_samples': [5],                      # SPSA samples to average (1-20)
    'max_iterations': [100],                      # Maximum optimization iterations (10-1000)
    'tolerance': [1e-6],                          # Convergence tolerance
}

DUAL_ANNEALING_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > Dual Annealing)
    'initial_temp': [5230.0],                     # Initial temperature (100-50000)
    'restart_temp_ratio': [2e-5],                 # Restart temperature ratio (1e-6 - 1e-3)
    'visit': [2.62],                              # Visiting distribution parameter (1.0-3.0)
    'accept': [-5.0],                             # Acceptance distribution parameter (-10 to -1)
    'max_iterations': [100],                      # Maximum iterations (10-1000)
}

BASIN_HOPPING_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > Basin Hopping)
    'niter': [100],                               # Basin hopping iterations (10-500)
    'T': [1.0],                                   # Temperature for Metropolis criterion (0.1-10)
    'stepsize': [0.5],                            # Initial step size for displacement (0.1-1.0)
}

LBFGSB_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > L-BFGS-B)
    'gradient_type': ['spsa'],                    # 'spsa', 'stosag', 'fd_forward', 'fd_central'
    'perturbation_size': [0.01],                  # Relative perturbation for gradient (0.001-0.1)
    'spsa_num_samples': [5],                      # SPSA samples to average (1-20)
    'maxcor': [10],                               # Limited-memory corrections (5-20)
    'ftol': [1e-10],                              # Function convergence tolerance
    'gtol': [1e-6],                               # Projected gradient convergence tolerance
    'max_iterations': [100],                      # Maximum L-BFGS-B iterations (10-500)
    'tolerance': [1e-6],                          # Convergence tolerance
}

ADAM_GRID = {
    # (Optimizer Dashboard > Optimizer Tab > Adam)
    'learning_rate': [0.01],                      # Step size alpha (0.001-0.1)
    'beta1': [0.9],                               # First moment decay rate (0.8-0.99)
    'beta2': [0.999],                             # Second moment decay rate (0.99-0.9999)
    'epsilon': [1e-8],                            # Numerical stability constant
    'lr_decay': [1.0],                            # LR decay per iteration (0.99-1.0)
    'warmup_iterations': [0],                     # Linear warmup iterations (0-20)
    'gradient_type': ['spsa'],                    # 'spsa', 'stosag', 'fd_forward', 'fd_central'
    'perturbation_size': [0.01],                  # Relative perturbation for gradient (0.001-0.1)
    'spsa_num_samples': [5],                      # SPSA samples to average (1-20)
    'n_stagnation': [50],                         # Early stop after N iters without improvement (20-100)
    'max_iterations': [200],                      # Maximum Adam iterations (50-1000)
    'tolerance': [1e-6],                          # Convergence tolerance
}

HYBRID_GRID = {
    # Stage 1 global source and Stage 2 local refiner
    'stage1_source': ['PSO'],                     # 'RL', 'PSO', 'GA', 'CMA-ES', 'Differential-Evolution'
    'stage2_type': ['L-BFGS-B'],                  # 'L-BFGS-B', 'Adam', 'LS-SQP-StoSAG'
    # Stage 1 params (used when stage1 is a classical optimizer)
    'stage1_max_iterations': [50],                # Stage 1 iteration budget
    # Stage 2 params
    'stage2_gradient_type': ['spsa'],             # Gradient estimation for Stage 2
    'stage2_perturbation_size': [0.01],           # Perturbation size
    'stage2_spsa_num_samples': [5],               # SPSA samples
    'stage2_max_iterations': [100],               # Stage 2 iteration budget
}

OPTIMIZER_PARAM_GRIDS = {
    'RL_Training': RL_TRAINING_GRID,
    'RL_Evaluation': RL_EVALUATION_GRID,
    'GA': GA_GRID,
    'Differential-Evolution': DE_GRID,
    'PSO': PSO_GRID,
    'CMA-ES': CMAES_GRID,
    'LS-SQP-StoSAG': LSSQP_GRID,
    'L-BFGS-B': LBFGSB_GRID,
    'Adam': ADAM_GRID,
    'Dual-Annealing': DUAL_ANNEALING_GRID,
    'Basin-Hopping': BASIN_HOPPING_GRID,
    'Hybrid': HYBRID_GRID,
}


# ---------------------------------------------------------------------------
# RL config builder (replaces dashboard configuration for headless runs)
# ---------------------------------------------------------------------------

CHANNEL_STATE_MAP = {
    2: ['SG', 'PRES'],
    4: ['SG', 'PRES', 'PERMI', 'POROS'],
}


def build_rl_config_from_action_ranges(action_ranges: Dict,
                                        n_channels: int = 2) -> Dict:
    """
    Build an rl_config dict that matches the dashboard format expected by
    the environment and orchestrator.  Includes selected_states so that
    downstream components know which spatial channels are active.
    """
    bhp_min = action_ranges['producer_bhp']['min']
    bhp_max = action_ranges['producer_bhp']['max']
    gas_min = action_ranges['gas_injection']['min']
    gas_max = action_ranges['gas_injection']['max']

    selected_states = CHANNEL_STATE_MAP.get(n_channels, ['SG', 'PRES'])

    return {
        'selected_states': selected_states,
        'action_ranges': {
            'bhp': {
                'P1': {'min': bhp_min, 'max': bhp_max},
                'P2': {'min': bhp_min, 'max': bhp_max},
                'P3': {'min': bhp_min, 'max': bhp_max},
            },
            'gas_injection': {
                'I1': {'min': gas_min, 'max': gas_max},
                'I2': {'min': gas_min, 'max': gas_max},
                'I3': {'min': gas_min, 'max': gas_max},
            },
        },
    }


# ---------------------------------------------------------------------------
# Device override helper
# ---------------------------------------------------------------------------

def _override_env_device(env, device: torch.device):
    """Ensure all environment tensors are on the target device."""
    env.device = device
    env.z0_options = env.z0_options.to(device)
    env.noise = env.noise.to(device)
    env.dt = env.dt.to(device)
    if env.state is not None:
        env.state = env.state.to(device)
    if hasattr(env, 'original_spatial_states') and env.original_spatial_states is not None:
        env.original_spatial_states = env.original_spatial_states.to(device)
    if hasattr(env, 'current_spatial_state') and env.current_spatial_state is not None:
        env.current_spatial_state = env.current_spatial_state.to(device)


# ---------------------------------------------------------------------------
# 2b  Standalone RL Training Runner
# ---------------------------------------------------------------------------

def run_rl_training(params: Dict, config: Config, z0_options: torch.Tensor,
                    rom_model: ROMWithE2C, rl_config_dict: Optional[Dict],
                    spatial_states: Optional[torch.Tensor],
                    device: torch.device) -> Dict:
    """Run one RL training session programmatically (no dashboard)."""
    import random as _random
    from RL_Refactored.agent import create_rl_agent, ReplayMemory
    from RL_Refactored.environment.reservoir_env import create_environment
    from RL_Refactored.training import EnhancedTrainingOrchestrator
    from RL_Refactored.configuration import (
        get_action_scaling_params, create_rl_reward_function, update_config_with_dashboard
    )

    # Set seeds for reproducibility (matches config.yaml seeds)
    seeds = config.rl_model['training']['seeds']
    torch.manual_seed(seeds['torch'])
    np.random.seed(seeds['numpy'])
    _random.seed(seeds['torch'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds['torch'])

    max_episodes = params.get('max_episodes', 100)
    max_steps = params.get('max_steps_per_episode', 30)
    batch_size = params.get('replay_batch_size', 256)
    updates_per_step = params.get('updates_per_step', 1)
    exploration_steps = params.get('initial_exploration_steps', 0)
    eval_ckpt = params.get('eval_checkpoint_enabled', False)
    eval_interval = params.get('eval_interval', 5)
    eval_num_cases = params.get('eval_num_cases', 10)

    # --- Network Architecture ---
    config.rl_model['networks']['hidden_dim'] = params.get('hidden_dim', 192)
    config.rl_model['networks']['policy']['type'] = params.get('policy_type', 'gaussian')
    config.rl_model['networks']['policy']['output_activation'] = params.get('output_activation', 'sigmoid')

    # --- SAC Algorithm ---
    config.rl_model['sac']['learning_rates']['critic'] = params.get('learning_rate_critic', 0.0001)
    config.rl_model['sac']['learning_rates']['policy'] = params.get('learning_rate_policy', 0.0003)
    config.rl_model['sac']['discount_factor'] = params.get('discount_factor', 0.986)
    config.rl_model['sac']['soft_update_tau'] = params.get('soft_update_tau', 0.005)
    config.rl_model['sac']['entropy']['alpha'] = params.get('entropy_alpha', 0.2)
    config.rl_model['sac']['entropy']['automatic_tuning'] = params.get('automatic_entropy_tuning', True)
    config.rl_model['sac']['gradient_clipping']['enable'] = params.get('gradient_clipping', True)
    config.rl_model['sac']['gradient_clipping']['policy_max_norm'] = params.get('gradient_clip_max_norm', 10.0)

    # --- Training ---
    config.rl_model['training']['max_episodes'] = max_episodes
    config.rl_model['training']['max_steps_per_episode'] = max_steps
    config.rl_model['training']['updates_per_step'] = updates_per_step
    config.rl_model['training']['exploration_steps'] = exploration_steps
    config.rl_model['replay_memory']['batch_size'] = batch_size
    config.rl_model['replay_memory']['capacity'] = params.get('replay_capacity', 100000)

    # --- Action Variation ---
    config.rl_model['action_variation']['enabled'] = params.get('action_variation_enabled', True)
    config.rl_model['action_variation']['mode'] = params.get('action_variation_mode', 'adaptive')
    config.rl_model['action_variation']['noise_decay_rate'] = params.get('noise_decay_rate', 0.995)
    config.rl_model['action_variation']['max_noise_std'] = params.get('max_noise_std', 0.25)
    config.rl_model['action_variation']['min_noise_std'] = params.get('min_noise_std', 0.01)
    config.rl_model['action_variation']['step_variation_amplitude'] = params.get('step_variation_amplitude', 0.15)

    # --- Enhanced Gaussian Policy ---
    config.rl_model['action_variation'].setdefault('enhanced_gaussian_policy', {})
    config.rl_model['action_variation']['enhanced_gaussian_policy']['enabled'] = params.get('enhanced_gaussian_enabled', False)
    config.rl_model['gaussian_policy']['log_std_bounds']['min'] = params.get('log_std_min', -1.0)
    config.rl_model['gaussian_policy']['log_std_bounds']['max'] = params.get('log_std_max', 1.0)

    # --- Environment ---
    config.rl_model['environment']['prediction_mode'] = params.get('prediction_mode', 'state_based')
    config.rl_model['environment']['noise']['enable'] = params.get('env_noise_enable', False)
    config.rl_model['environment']['noise']['std'] = params.get('env_noise_std', 0.10)

    if rl_config_dict is None:
        rl_config_dict = build_rl_config_from_action_ranges(ACTION_RANGES, N_CHANNELS)

    # Set algorithm type from grid params
    algo_type = params.get('algorithm_type', 'SAC')
    config.rl_model.setdefault('algorithm', {})['type'] = algo_type

    agent = create_rl_agent(config, rl_config_dict, rom_model=rom_model)
    is_ppo = algo_type.upper() == 'PPO'

    env = create_environment(z0_options, config, rom_model, rl_config_dict,
                             spatial_states=spatial_states)
    _override_env_device(env, device)
    memory = ReplayMemory(config.rl_model['replay_memory']['capacity'],
                          config.rl_model['training']['seeds']['replay_memory'])
    orchestrator = EnhancedTrainingOrchestrator(config, rl_config_dict)
    orchestrator.set_environment(env)

    episode_rewards = []
    best_reward = -np.inf
    best_eval_reward = -np.inf
    global_step = 0
    total_numsteps = 0
    start_time = time.time()
    best_controls = None
    best_observations = None
    best_step_npvs = None

    for episode in range(max_episodes):
        state = env.reset(z0_options)
        orchestrator.start_new_episode()
        episode_reward = 0.0
        ep_controls = []
        ep_observations = []

        for step in range(max_steps):
            if is_ppo:
                action = agent.select_action(state, evaluate=False)
            else:
                action = orchestrator.select_enhanced_action(agent, state, episode, step, exploration_steps, total_numsteps)
            action = action.to(device)
            next_state, reward, done = env.step(action)
            obs = getattr(env, 'last_observation', None)
            orchestrator.record_step_data(step=step, action=action, observation=obs, reward=reward, state=state)
            total_numsteps += 1

            if is_ppo:
                agent.collect_step(state, action, reward, done)
            else:
                memory.push(state, action, reward, next_state)

            ep_controls.append(action.detach().cpu().numpy().flatten().tolist())
            if obs is not None:
                ep_observations.append(obs.detach().cpu().numpy().flatten().tolist())

            state = next_state
            episode_reward += reward.item()

            if not is_ppo and len(memory) > batch_size:
                for _ in range(updates_per_step):
                    agent.update_parameters(memory, batch_size, global_step)
                    global_step += 1

            if done:
                break

        # PPO: episode-level update
        if is_ppo:
            agent.update_from_buffer(state)
            global_step += 1

        op_reward = orchestrator.finalize_episode(episode, total_reward=episode_reward)
        final_reward = op_reward if op_reward is not None else episode_reward
        episode_rewards.append(final_reward)

        if final_reward > best_reward:
            best_reward = final_reward
            best_controls = ep_controls
            best_observations = ep_observations

        if eval_ckpt and episode % eval_interval == 0:
            agent.policy.eval()
            eval_npvs = []
            nc = min(eval_num_cases, z0_options.shape[0])
            idxs = np.random.choice(z0_options.shape[0], nc, replace=False)
            for ci in idxs:
                z0 = z0_options[ci:ci + 1]
                s = env.reset(z0_options=z0)
                er = 0.0
                for _ in range(max_steps):
                    with torch.no_grad():
                        a = agent.select_action(s, evaluate=True)
                    ns, r, d = env.step(a)
                    er += r.item()
                    s = ns
                    if d:
                        break
                eval_npvs.append(er)
            agent.policy.train()
            mean_eval = float(np.mean(eval_npvs))
            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval

        if episode % 10 == 0:
            avg10 = np.mean(episode_rewards[-10:])
            print(f"  Ep {episode}/{max_episodes} | Reward {final_reward:.3f} | Avg10 {avg10:.3f} | Best {best_reward:.3f}")

    elapsed = time.time() - start_time

    # Extract per-step rewards from the best episode.
    # Each RL step reward = daily_cashflow / scale_factor, which is the same
    # format as the classical optimizer step_npvs.  Using the reward values
    # directly avoids any mismatch between the reward function and the
    # orchestrator's independent economic breakdown calculation.
    step_npvs = None
    econ_breakdown = {}
    best_ep_data = orchestrator.get_best_episode_data()
    if best_ep_data is not None:
        best_step_rewards = best_ep_data.get('rewards', [])
        if best_step_rewards:
            step_npvs = list(best_step_rewards)
            econ_breakdown = {
                'step_npvs': step_npvs,
                'total_npv': sum(step_npvs),
                'num_steps': len(step_npvs),
            }

    if step_npvs is None and best_controls:
        step_npvs = [best_reward / len(best_controls)] * len(best_controls)

    # Diagnostic: verify step_npvs consistency with best_reward
    if step_npvs:
        _snpv_sum = sum(step_npvs)
        _snpv_annual = sum(v * 365 for v in step_npvs)
        print(f"  [DIAG] best_reward={best_reward:.4f}  sum(step_npvs)={_snpv_sum:.4f}  "
              f"sum(annual)={_snpv_annual:.1f}  n_steps={len(step_npvs)}  "
              f"step_npvs[:3]={step_npvs[:3]}")
        if best_ep_data is not None:
            _ep_rew = best_ep_data.get('rewards', [])
            _ep_total = best_ep_data.get('total_reward', None)
            _ep_sum = f"{sum(_ep_rew):.4f}" if _ep_rew else "N/A"
            _ep_tot_str = f"{_ep_total:.4f}" if isinstance(_ep_total, (int, float)) else str(_ep_total)
            print(f"  [DIAG] best_ep total_reward={_ep_tot_str}  "
                  f"sum(ep_rewards)={_ep_sum}  "
                  f"ep_rewards[:3]={_ep_rew[:3] if _ep_rew else 'N/A'}")

    return {
        'best_reward_or_npv': best_reward,
        'final_converged_objective': episode_rewards[-1] if episode_rewards else 0.0,
        'initial_objective': episode_rewards[0] if episode_rewards else 0.0,
        'improvement_ratio': (best_reward - episode_rewards[0]) / abs(episode_rewards[0]) if episode_rewards and episode_rewards[0] != 0 else 0.0,
        'running_time_seconds': elapsed,
        'num_iterations': max_episodes,
        'num_function_evaluations': total_numsteps,
        'convergence_achieved': True,
        'optimal_controls': best_controls,
        'optimal_observations': best_observations,
        'step_npvs': step_npvs,
        'economic_breakdown': econ_breakdown,
        'objective_history': episode_rewards,
        'best_eval_npv': best_eval_reward if eval_ckpt else None,
    }


# ---------------------------------------------------------------------------
# 2c  RL Evaluation Runner
# ---------------------------------------------------------------------------

def run_rl_evaluation(params: Dict, config: Config, z0_options: torch.Tensor,
                      rom_model: ROMWithE2C, rl_config_dict: Optional[Dict],
                      spatial_states: Optional[torch.Tensor],
                      device: torch.device) -> Dict:
    """Evaluate a trained RL checkpoint across multiple Z0 cases."""
    from RL_Refactored.agent import create_rl_agent
    from RL_Refactored.environment.reservoir_env import create_environment
    from RL_Refactored.evaluation.evaluator import PolicyEvaluator

    if rl_config_dict is None:
        rl_config_dict = build_rl_config_from_action_ranges(ACTION_RANGES, N_CHANNELS)

    agent = create_rl_agent(config, rl_config_dict, rom_model=rom_model)
    env = create_environment(z0_options, config, rom_model, rl_config_dict,
                             spatial_states=spatial_states)
    _override_env_device(env, device)

    evaluator = PolicyEvaluator(agent, env, config, rl_config=rl_config_dict)

    ckpt_path = params.get('checkpoint_path', None)
    if ckpt_path is None:
        ckpts = PolicyEvaluator.discover_checkpoints('checkpoints')
        if ckpts:
            best = [c for c in ckpts if c['type'] == 'best']
            ckpt_path = best[0]['path'] if best else ckpts[0]['path']
        else:
            return {
                'best_reward_or_npv': 0.0, 'final_converged_objective': 0.0,
                'initial_objective': 0.0, 'improvement_ratio': 0.0,
                'running_time_seconds': 0.0, 'num_iterations': 0,
                'num_function_evaluations': 0, 'convergence_achieved': False,
                'optimal_controls': None, 'optimal_observations': None,
                'step_npvs': None, 'economic_breakdown': {},
                'objective_history': [], 'error': 'No checkpoint found',
            }

    evaluator.load_checkpoint(ckpt_path, evaluate=True)
    num_cases = params.get('num_eval_cases', 100)
    deterministic = params.get('deterministic', True)

    start_time = time.time()
    results = evaluator.evaluate_multiple_cases(z0_options, num_cases=num_cases,
                                                 deterministic=deterministic, verbose=False)
    elapsed = time.time() - start_time

    npvs = [ep.total_npv for ep in results.all_episodes]
    best_ep = max(results.all_episodes, key=lambda e: e.total_npv)

    controls = [[v for v in a.values()] for a in best_ep.actions] if best_ep.actions else None
    observations = [[v for v in o.values()] for o in best_ep.observations] if best_ep.observations else None

    # Use per-step rewards directly as step_npvs (same format as classical
    # optimizer: daily_cashflow / scale_factor).
    step_npvs = list(best_ep.step_rewards) if best_ep.step_rewards else None
    econ_breakdown = {}
    if step_npvs:
        econ_breakdown = {
            'step_npvs': step_npvs,
            'total_npv': sum(step_npvs),
            'num_steps': len(step_npvs),
        }

    return {
        'best_reward_or_npv': float(np.max(npvs)),
        'final_converged_objective': float(np.mean(npvs)),
        'initial_objective': float(np.min(npvs)),
        'improvement_ratio': (float(np.max(npvs)) - float(np.min(npvs))) / abs(float(np.min(npvs))) if np.min(npvs) != 0 else 0.0,
        'running_time_seconds': elapsed,
        'num_iterations': num_cases,
        'num_function_evaluations': num_cases * config.rl_model['training']['max_steps_per_episode'],
        'convergence_achieved': True,
        'optimal_controls': controls,
        'optimal_observations': observations,
        'step_npvs': step_npvs,
        'economic_breakdown': econ_breakdown,
        'objective_history': npvs,
        'checkpoint_path': ckpt_path,
        'mean_npv': float(np.mean(npvs)),
        'std_npv': float(np.std(npvs)),
    }


# ---------------------------------------------------------------------------
# 2d  Classical Optimizer Runner
# ---------------------------------------------------------------------------

def run_classical_optimizer(optimizer_type: str, params: Dict, config: Config,
                            z0_options: torch.Tensor, rom_model: ROMWithE2C,
                            norm_params: Dict, action_ranges: Dict,
                            spatial_states: Optional[torch.Tensor],
                            device: torch.device) -> Dict:
    """Run a classical optimizer (GA, DE, PSO, etc.) programmatically."""
    from RL_Refactored.optimizers import create_optimizer, run_optimization

    opt_config = {
        'optimizer_type': optimizer_type,
        'rom_model': rom_model,
        'config': config,
        'norm_params': norm_params,
        'device': device,
        'z0_options': z0_options,
        'action_ranges': action_ranges,
        'init_strategy': params.get('init_strategy', 'midpoint'),
        'spatial_states': spatial_states,
        'optimizer_params': {k: v for k, v in params.items() if k not in ('init_strategy', 'num_steps')},
    }

    if optimizer_type == 'LS-SQP-StoSAG':
        opt_config['stosag_params'] = {
            'num_realizations': 1,
            'perturbation_size': params.get('perturbation_size', 0.01),
            'gradient_type': params.get('gradient_type', 'spsa'),
            'spsa_num_samples': params.get('spsa_num_samples', 5),
        }
        opt_config['sqp_params'] = {
            'max_iterations': params.get('max_iterations', 100),
            'tolerance': params.get('tolerance', 1e-6),
        }

    if optimizer_type == 'Hybrid':
        opt_config['hybrid_config'] = {
            'stage1_source': params.get('stage1_source', 'PSO'),
            'stage1_params': {
                'max_iterations': params.get('stage1_max_iterations', 50),
                'init_strategy': params.get('init_strategy', 'midpoint'),
            },
            'stage2_type': params.get('stage2_type', 'L-BFGS-B'),
            'stage2_params': {
                'gradient_type': params.get('stage2_gradient_type', 'spsa'),
                'perturbation_size': params.get('stage2_perturbation_size', 0.01),
                'spsa_num_samples': params.get('stage2_spsa_num_samples', 5),
                'max_iterations': params.get('stage2_max_iterations', 100),
            },
        }
        opt_config['optimizer_params'] = {}

    num_steps = params.get('num_steps', 30)

    start_time = time.time()
    optimizer = create_optimizer(opt_config)
    result = run_optimization(optimizer, z0_options=z0_options, num_steps=num_steps)
    elapsed = time.time() - start_time

    controls = result.optimal_controls.tolist() if result.optimal_controls is not None else None
    observations = None
    if result.optimal_observations is not None:
        obs = result.optimal_observations
        if obs.ndim == 3:
            obs = obs.squeeze(1) if obs.shape[1] == 1 else obs.mean(axis=1)
        observations = obs.tolist()

    breakdown = result.economic_breakdown or {}
    step_npvs = breakdown.get('step_npvs', None)

    return {
        'best_reward_or_npv': float(result.optimal_objective),
        'final_converged_objective': float(result.optimal_objective),
        'initial_objective': float(result.initial_objective),
        'improvement_ratio': float(result.improvement_ratio()),
        'running_time_seconds': elapsed,
        'num_iterations': result.num_iterations,
        'num_function_evaluations': result.num_function_evaluations,
        'convergence_achieved': result.convergence_achieved,
        'optimal_controls': controls,
        'optimal_observations': observations,
        'step_npvs': step_npvs,
        'economic_breakdown': {k: v for k, v in breakdown.items() if k != 'step_npvs'},
        'objective_history': result.objective_history,
    }


# ---------------------------------------------------------------------------
# 2e  Grid Search Loop
# ---------------------------------------------------------------------------

def _product_dict(d: Dict[str, list]) -> List[Dict[str, Any]]:
    """Cartesian product of a dict of lists."""
    keys = list(d.keys())
    vals = list(d.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def apply_economic_params(config: Config, econ: Dict):
    """Patch config with economic grid values (prices + project economics)."""
    prices = config.rl_model['economics']['prices']
    prices['gas_injection_revenue'] = econ.get('gas_injection_revenue', prices['gas_injection_revenue'])
    prices['gas_injection_cost'] = econ.get('gas_injection_cost', prices['gas_injection_cost'])
    prices['water_production_penalty'] = econ.get('water_production_penalty', prices['water_production_penalty'])
    prices['gas_production_penalty'] = econ.get('gas_production_penalty', prices['gas_production_penalty'])
    config.rl_model['economics']['scale_factor'] = econ.get('scale_factor', config.rl_model['economics']['scale_factor'])
    config.rl_model['economics']['years_before_project_start'] = econ.get('years_before_project_start', 5)
    config.rl_model['economics']['capital_cost_per_year'] = econ.get('capital_cost_per_year', 100000000.0)


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def save_results_json(all_results: List[Dict], path: str):
    def _convert(obj):
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"Results saved: {path}")


def run_grid_search():
    """Main grid search orchestration."""
    print("\n" + "=" * 70)
    print("RL GRID SEARCH")
    print("=" * 70)

    optimizer_combos = _product_dict(OPTIMIZER_GRID)
    shared_combos = _product_dict(SHARED_GRID)
    econ_combos = _product_dict(ECONOMIC_GRID)

    all_runs: List[Dict] = []
    run_idx = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f'rl_grid_search_results_{timestamp}.json')
    latest_file = os.path.join(RESULTS_DIR, 'rl_grid_search_results_latest.json')

    for opt_combo in optimizer_combos:
        opt_type = opt_combo['optimizer_type']
        opt_param_grid = OPTIMIZER_PARAM_GRIDS.get(opt_type, {})
        opt_param_combos = _product_dict(opt_param_grid) if opt_param_grid else [{}]

        for opt_params in opt_param_combos:
            for shared in shared_combos:
                for econ in econ_combos:
                    run_idx += 1
                    merged = {**shared, **opt_params}
                    run_id = f"{opt_type}_run{run_idx:04d}"

                    print(f"\n[{run_idx}] {run_id}")
                    print(f"    Optimizer: {opt_type} | Params: {opt_params}")

                    force_cleanup()

                    # Fresh config copy for each run
                    cfg = Config(RL_CONFIG_PATH)
                    apply_economic_params(cfg, econ)
                    cfg.config['runtime']['device'] = str(DEVICE)
                    cfg.device = DEVICE

                    run_meta: Dict[str, Any] = {
                        'run_id': run_id,
                        'optimizer_type': opt_type,
                        'params': merged,
                        'economic_params': econ,
                        'rom_model_name': SELECTED_ROM_ENTRY['name'],
                    }

                    try:
                        if opt_type == 'RL_Training':
                            result = run_rl_training(
                                merged, cfg, Z0_OPTIONS, ROM_MODEL, None,
                                SPATIAL_STATES, DEVICE)
                        elif opt_type == 'RL_Evaluation':
                            result = run_rl_evaluation(
                                merged, cfg, Z0_OPTIONS, ROM_MODEL, None,
                                SPATIAL_STATES, DEVICE)
                        else:
                            result = run_classical_optimizer(
                                opt_type, merged, cfg, Z0_OPTIONS, ROM_MODEL,
                                NORM_PARAMS, ACTION_RANGES, SPATIAL_STATES, DEVICE)

                        run_meta.update(result)
                        run_meta['status'] = 'success'
                        print(f"    Result: NPV={result['best_reward_or_npv']:.4f} | "
                              f"Time={result['running_time_seconds']:.1f}s | "
                              f"Iters={result['num_iterations']}")

                    except Exception as e:
                        run_meta['status'] = 'failed'
                        run_meta['error'] = str(e)
                        print(f"    FAILED: {e}")
                        import traceback
                        traceback.print_exc()

                    all_runs.append(run_meta)
                    save_results_json(all_runs, latest_file)

    save_results_json(all_runs, results_file)

    # Print summary
    successful = [r for r in all_runs if r.get('status') == 'success']
    failed = [r for r in all_runs if r.get('status') == 'failed']
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(all_runs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")
    if successful:
        best = max(successful, key=lambda r: r.get('best_reward_or_npv', -np.inf))
        print(f"\nBest run: {best['run_id']}")
        print(f"  Optimizer:  {best['optimizer_type']}")
        print(f"  Best NPV:   {best['best_reward_or_npv']:.6f}")
        print(f"  Time:       {best['running_time_seconds']:.1f}s")
        print(f"  Iterations: {best['num_iterations']}")
    if failed:
        print(f"\nFailed runs:")
        for r in failed[:5]:
            print(f"  {r['run_id']}: {r.get('error', 'unknown')}")
    print("=" * 70)
    return all_runs


# ---- Execute ----
print("\n>>> CELL 2: Grid Search Execution\n")
ALL_RESULTS = run_grid_search()


# =============================================================================
# CELL 3: PLOTTING (independent of runs)
# =============================================================================
#%%
"""
INSTRUCTIONS:
  - This cell loads results from the latest saved JSON file.
  - Edit and re-run this cell without re-running the grid search.
  - Modify RESULTS_JSON_PATH to load a specific results file.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, 'rl_grid_search_results_latest.json')

def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def _successful_runs(results: List[Dict]) -> List[Dict]:
    return [r for r in results if r.get('status') == 'success']


# ---------------------------------------------------------------------------
# 3a  Project Lifecycle NCF Plot (all runs overlaid)
# ---------------------------------------------------------------------------

def plot_lifecycle_ncf(results: List[Dict], save_path: str):
    """Plot cumulative project lifecycle NCF for all successful runs."""
    runs = _successful_runs(results)
    if not runs:
        print("No successful runs to plot.")
        return

    years_before = 5
    capital_cost_per_year = 100000000.0
    scale_factor = 1000000.0

    try:
        cfg = Config(RL_CONFIG_PATH)
        econ = cfg.rl_model.get('economics', {})
        years_before = econ.get('years_before_project_start', 5)
        capital_cost_per_year = econ.get('capital_cost_per_year', 100000000.0)
        scale_factor = econ.get('scale_factor', 1000000.0)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.tab10

    for idx, run in enumerate(runs):
        step_npvs = run.get('step_npvs')
        if step_npvs is None:
            bd = run.get('economic_breakdown', {})
            if isinstance(bd, dict) and 'step_npvs' in bd:
                step_npvs = bd['step_npvs']
        if step_npvs is None:
            obj_hist = run.get('objective_history', [])
            if not obj_hist:
                continue
            total_obj = obj_hist[-1] if isinstance(obj_hist[-1], (int, float)) else run.get('best_reward_or_npv', 0)
            num_s = run.get('params', {}).get('num_steps', 30)
            step_npvs = [total_obj / num_s] * num_s

        step_annual = [npv * 365 for npv in step_npvs]

        run_econ = run.get('economic_params', {})
        run_yb = run_econ.get('years_before_project_start', years_before)
        run_cc = run_econ.get('capital_cost_per_year', capital_cost_per_year)
        run_sf = run_econ.get('scale_factor', scale_factor)

        # Timeline: year 0 = project start (cumulative 0), years 1..yb = CAPEX,
        # year yb = operations start, years yb+1..yb+N = operational cashflows
        pre_cashflows = [0.0] + [-run_cc / run_sf] * run_yb
        all_cashflows = pre_cashflows + step_annual

        pre_years = list(range(0, run_yb + 1))
        op_years = list(range(run_yb + 1, run_yb + 1 + len(step_annual)))
        all_years = pre_years + op_years

        cum_npv = np.cumsum(all_cashflows)
        color = cmap(idx % 10)
        label = f"{run['optimizer_type']} ({run['run_id']})"
        ax.plot(all_years, cum_npv, '-', linewidth=2, color=color, label=label)

    ax.axvline(x=years_before, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Operations Start')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Break-even Line')
    ax.set_xlabel('Project Year', fontsize=12)
    ax.set_ylabel('Cumulative NPV ($M)', fontsize=12)
    ax.set_title(f'Project Lifecycle NCF - All Runs\n'
                 f'(Capital: ${years_before * capital_cost_per_year / 1e6:.0f}M over {years_before} years)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3b  Controls Plot (all runs overlaid, 6 subplots)
# ---------------------------------------------------------------------------

def plot_controls_all_runs(results: List[Dict], save_path: str):
    """Plot 6 action/control subplots with all runs overlaid."""
    runs = _successful_runs(results)
    runs_with_ctrl = [r for r in runs
                      if r.get('optimal_controls') is not None
                      and len(r.get('optimal_controls', [])) > 0]
    if not runs_with_ctrl:
        print("No runs with control data to plot.")
        return

    try:
        ar = auto_detect_action_ranges()
        bhp_min, bhp_max = ar['producer_bhp']['min'], ar['producer_bhp']['max']
        gas_min, gas_max = ar['gas_injection']['min'], ar['gas_injection']['max']
    except Exception:
        bhp_min, bhp_max = 1087.78, 1305.34
        gas_min, gas_max = 24720290.0, 100646896.0

    num_prod = 3
    num_inj = 3
    well_names_prod = ['P1', 'P2', 'P3']
    well_names_inj = ['I1', 'I2', 'I3']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Controls Across All Runs', fontsize=14, fontweight='bold')
    cmap = plt.cm.tab10

    for idx, run in enumerate(runs_with_ctrl):
        controls = np.array(run['optimal_controls'])
        timesteps = np.arange(controls.shape[0])
        num_controls = controls.shape[1]
        color = cmap(idx % 10)
        label = f"{run['optimizer_type']} ({run['run_id']})"
        is_normalized = np.all(controls >= -0.1) and np.all(controls <= 1.1)

        for i in range(min(num_prod, num_controls)):
            ax = axes[0, i]
            vals = controls[:, i].copy()
            if is_normalized:
                vals = vals * (bhp_max - bhp_min) + bhp_min
            ax.plot(timesteps, vals, '-', linewidth=1.5, markersize=3, color=color,
                    alpha=0.7, label=label if i == 0 else None)

        for i in range(min(num_inj, max(0, num_controls - num_prod))):
            ax = axes[1, i]
            vals = controls[:, num_prod + i].copy()
            if is_normalized:
                vals = vals * (gas_max - gas_min) + gas_min
            ax.plot(timesteps, vals / 1e6, '-', linewidth=1.5, markersize=3, color=color,
                    alpha=0.7)

    for i in range(num_prod):
        axes[0, i].set_xlabel('Timestep')
        axes[0, i].set_ylabel('BHP (psi)')
        axes[0, i].set_title(f'{well_names_prod[i]} - Producer BHP')
        axes[0, i].grid(True, alpha=0.3)
    for i in range(num_inj):
        axes[1, i].set_xlabel('Timestep')
        axes[1, i].set_ylabel('Gas Injection (MMscf/day)')
        axes[1, i].set_title(f'{well_names_inj[i]} - Gas Injection')
        axes[1, i].grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(5, len(handles)),
                   fontsize=7, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3c  Observation Plots (9 subplots, all runs overlaid)
# ---------------------------------------------------------------------------

def plot_observations_all_runs(results: List[Dict], save_path: str):
    """Plot 9 observation subplots (all runs overlaid)."""
    runs = _successful_runs(results)
    runs_with_obs = [r for r in runs if r.get('optimal_observations') is not None and len(r.get('optimal_observations', [])) > 0]

    if not runs_with_obs:
        print("No runs with observation data to plot.")
        return

    obs_labels = [
        ('I1 BHP', 'psi'), ('I2 BHP', 'psi'), ('I3 BHP', 'psi'),
        ('P1 Gas Prod', 'ft3/day'), ('P2 Gas Prod', 'ft3/day'), ('P3 Gas Prod', 'ft3/day'),
        ('P1 Water Prod', 'ft3/day'), ('P2 Water Prod', 'ft3/day'), ('P3 Water Prod', 'ft3/day'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Observations Across All Runs', fontsize=14, fontweight='bold')
    cmap = plt.cm.tab10

    for idx, run in enumerate(runs_with_obs):
        obs = np.array(run['optimal_observations'])
        timesteps = np.arange(obs.shape[0])
        color = cmap(idx % 10)
        label = f"{run['optimizer_type']} ({run['run_id']})"

        num_obs = min(obs.shape[1], 9)
        for oi in range(num_obs):
            row, col = divmod(oi, 3)
            ax = axes[row, col]
            scale = 1e-6 if oi >= 3 else 1.0
            ax.plot(timesteps, obs[:, oi] * scale, '-', linewidth=1.5, color=color,
                    alpha=0.7, label=label if oi == 0 else None)

    for oi in range(min(9, len(obs_labels))):
        row, col = divmod(oi, 3)
        ax = axes[row, col]
        lbl, unit = obs_labels[oi]
        display_unit = 'MMscf/day' if oi >= 3 else unit
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel(display_unit)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(5, len(handles)),
                   fontsize=7, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---- Execute Plots ----
print("\n>>> CELL 3: Plotting\n")

if os.path.exists(RESULTS_JSON_PATH):
    LOADED_RESULTS = load_results(RESULTS_JSON_PATH)
    print(f"Loaded {len(LOADED_RESULTS)} runs from {RESULTS_JSON_PATH}")

    plot_lifecycle_ncf(
        LOADED_RESULTS,
        os.path.join(PLOTS_DIR, 'lifecycle_ncf_all_runs.png'))

    plot_controls_all_runs(
        LOADED_RESULTS,
        os.path.join(PLOTS_DIR, 'controls_all_runs.png'))

    plot_observations_all_runs(
        LOADED_RESULTS,
        os.path.join(PLOTS_DIR, 'observations_all_runs.png'))

    # Print metadata summary
    successful = _successful_runs(LOADED_RESULTS)
    if successful:
        print("\n" + "=" * 70)
        print("RUN METADATA SUMMARY")
        print("=" * 70)
        print(f"{'Run ID':<30} {'Optimizer':<20} {'Best NPV':>12} {'Time (s)':>10} {'Iters':>8} {'Converged':>10}")
        print("-" * 90)
        for r in sorted(successful, key=lambda x: x.get('best_reward_or_npv', 0), reverse=True):
            print(f"{r['run_id']:<30} {r['optimizer_type']:<20} "
                  f"{r.get('best_reward_or_npv', 0):>12.4f} "
                  f"{r.get('running_time_seconds', 0):>10.1f} "
                  f"{r.get('num_iterations', 0):>8} "
                  f"{str(r.get('convergence_achieved', '')):>10}")
        print("=" * 70)
else:
    print(f"No results file found at {RESULTS_JSON_PATH}")
    print("Run Cell 2 first to generate results.")

# %%
