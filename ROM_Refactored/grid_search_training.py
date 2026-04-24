"""
Main Run File for E2C Model
==========================
This file orchestrates the two main parts of the ROM project:
1. Data Preprocessing Dashboard
2. Testing & Visualization Dashboard

Model training is handled separately via grid_search_training.py.
Each part is independent with its own configuration and functionality.
"""


#%%
import torch
# ============================================================================
# Hardware Configuration
# ============================================================================
print("=" * 70)
print("Hardware Configuration")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")
print("=" * 70)

# ============================================================================
# STEP 1: DATA PREPROCESSING DASHBOARD
# ============================================================================
# Import and create data preprocessing dashboard
from data_preprocessing import create_data_preprocessing_dashboard

# Create and display the interactive dashboard
preprocessing_dashboard = create_data_preprocessing_dashboard()


# ============================================================================
# STEP 2: HYPERPARAMETER GRID SEARCH TRAINING
# ============================================================================
#%%
import os
import sys
import json
import csv
import itertools
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from numpy import False_
import torch

# Add parent directory to path for imports
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.path.abspath(os.path.join(os.getcwd(), 'ROM_Refactored'))
sys.path.insert(0, _this_dir)

from utilities.config_loader import Config
from utilities.timing import Timer, collect_training_metadata
from utilities.wandb_integration import create_wandb_logger
from data_preprocessing import load_processed_data
from model.training.rom_wrapper import ROMWithE2C


# ============================================================================
# HYPERPARAMETER GRID DEFINITION
# ============================================================================

# Define hyperparameter ranges
# batch_size, latent_dim, n_steps, and n_channels are varied
# learning_rate is fixed (uses config default)
HYPERPARAMETER_GRID = {
    # Existing parameters
    'batch_size': [16],
    'n_steps': [2],  # Available processed data files
    'n_channels': [4],  # Number of channels (2 for SW/SG, 4 for SW/SG/PRES/PERMI, etc.)
    'latent_dim': [128],
    # Learning rate scheduler
    'lr_scheduler_type': ['fixed'], #, 'reduce_on_plateau', 'exponential_decay', 'step_decay', 'cosine_annealing'],
    
    # Architecture parameters
    'residual_blocks': [3],  # Number of residual blocks in encoder/decoder
    'encoder_hidden_dims': [[300, 300, 300]],  # Transition encoder hidden dimensions
    
    # Normalization/Activation type
    # - 'batchnorm': BatchNorm + ReLU (default, good for physics accuracy)
    # - 'gdn': GDN + GELU (better for perceptual quality/smooth reconstructions)
    'norm_type': ['batchnorm'],  # Options: 'batchnorm', 'gdn'
    
    # Dynamic loss weighting
    'dynamic_loss_weighting_enable': [False],
    'dynamic_loss_weighting_method': ['gradnorm', 'uncertainty', 'dwa'],  # Only when enabled=True
    
    # Adversarial training
    'adversarial_enable': [False],
    
    # VAE (Variational Autoencoder) configuration
    'enable_vae': [False_],                    # Enable VAE mode (reparameterization trick)
    'enable_kl_loss': [False],                # Enable KL divergence loss (requires enable_vae=True)
    'lambda_kl_loss': [0.00001],             # KL loss weight (beta in beta-VAE)
    'enable_kl_annealing': [False],          # Enable step-wise KL annealing schedule
    
    # FFT Loss configuration - frequency domain loss for better reconstruction
    'enable_fft_loss': [True],              # Enable FFT loss (captures low & high frequency errors)
    'lambda_fft_loss': [1.0],                # FFT loss weight
    
    # Inactive Cell Masking - exclude non-reservoir cells from reconstruction loss
    'enable_inactive_masking': [True],      # Enable inactive cell masking
    
    # Multimodal (two-branch fused encoder) - requires n_channels=4
    'enable_multimodal': [False],           # Separate static (PERMI/POROS) from dynamic (SG/PRES) branches. Must be False if enable_gnn is True
    
    # GNN (Graph Neural Network encoder/decoder) - requires n_channels=4, torch_geometric
    'enable_gnn':[False],                 # Replace CNN encoder/decoder with GNN (GATv2-based)
    
    # FNO (Fourier Neural Operator encoder/decoder) - invertible spectral convolution, requires n_channels=4
    'enable_fno': [False],                # Replace CNN encoder/decoder with FNO (spectral-based)
    'fno_norm_type': ['batchnorm'],       # FNO normalization: 'batchnorm' (default), 'instancenorm', 'none'
    
    # Multi-embedding multimodal: per-field encoder/decoder backbone choice (CNN or FNO per branch),
    # one shared conditioned transition. Requires n_channels=4. Mutually exclusive with enable_gnn / enable_fno.
    # When True, enable_multimodal is auto-disabled (this option subsumes it).
    'enable_multi_embedding_multimodal': [True],   # NEW
    'mem_branches_preset': ['cnn_sg__fno_pres'],    # Named preset (see MEM_PRESETS below)
    
    # Transition model type
    'transition_type': ['Linear'],        # Options: 's4d', 's4d_dplr', 's5', 'koopman', 'ct_koopman', 'clru', 'linear', 'nonlinear', 'mamba', 'mamba2', 'stable_koopman', 'deep_koopman', 'gru', 'lstm', 'hamiltonian', 'skolr', 'ren', 'koopman_aft', 'dissipative_koopman', 'bilinear_koopman', 'isfno', 'sindy', 'neural_cde', 'latent_sde', 'transformer', 'deeponet'
    
    # Encoder enhancement strategies (reduce re-encoding error accumulation)
    'enable_jacobian_loss': [False],      # Contractive encoder via Jacobian regularization
    'lambda_jacobian_loss': [0.01],       # Jacobian loss weight
    'enable_cycle_loss': [False],         # Cycle-consistency loss: encode(decode(z)) ≈ z
    'lambda_cycle_loss': [0.1],           # Cycle loss weight
    'enable_scheduled_sampling': [False], # Gradually replace teacher forcing with self-sampling
    'enable_latent_noise': [False],       # Inject noise into latent space during training
}

# Multi-embedding multimodal presets. Each entry is a list of branch specs in
# *declaration order*. Latent dim of each branch must sum to the global
# `latent_dim` hyperparameter. Channel sets across branches must partition
# {0,1,2,3} (SG, PRES, POROS, PERMI) exactly.
MEM_PRESETS = {
    # Default ask: CNN for static + saturation, FNO for pressure
    'cnn_sg__fno_pres': [
        {'name': 'static',     'channels': [2, 3], 'role': 'static',
         'encoder': {'type': 'cnn'}, 'decoder': None,                'latent_dim': 32},
        {'name': 'saturation', 'channels': [0],    'role': 'dynamic',
         'encoder': {'type': 'cnn'}, 'decoder': {'type': 'cnn'},     'latent_dim': 48},
        {'name': 'pressure',   'channels': [1],    'role': 'dynamic',
         'encoder': {'type': 'fno'}, 'decoder': {'type': 'fno'},     'latent_dim': 48},
    ],
    # All-CNN (sanity check; should match current MultimodalMSE2C behavior)
    'cnn_all': [
        {'name': 'static',  'channels': [2, 3], 'role': 'static',
         'encoder': {'type': 'cnn'}, 'decoder': None,             'latent_dim': 32},
        {'name': 'dynamic', 'channels': [0, 1], 'role': 'dynamic',
         'encoder': {'type': 'cnn'}, 'decoder': {'type': 'cnn'},  'latent_dim': 96},
    ],
    # All-FNO
    'fno_all': [
        {'name': 'static',  'channels': [2, 3], 'role': 'static',
         'encoder': {'type': 'fno'}, 'decoder': None,             'latent_dim': 32},
        {'name': 'dynamic', 'channels': [0, 1], 'role': 'dynamic',
         'encoder': {'type': 'fno'}, 'decoder': {'type': 'fno'},  'latent_dim': 96},
    ],
}


# Nested parameter grids (applied conditionally based on parent parameter values)
LR_SCHEDULER_PARAMS = {
    'reduce_on_plateau': {
        'factor': [0.5],
        #'patience': [5],
       # 'threshold': [1e-4],
        #'cooldown': [3],
       # 'min_lr': [1e-7]
    },
    'exponential_decay': {
        'gamma': [0.9]
    },
    'step_decay': {
        'step_size': [50],
        #'gamma': [0.5]
    },
    'cosine_annealing': {
        'T_max': [20],
        #'eta_min': [1e-6]
    },
    'cyclic': {
        'base_lr': [1e-5],
        #'max_lr': [1e-3],
        #'step_size_up': [500],
        #'gamma': [1.0],
       # 'base_momentum': [0.8],
       # 'max_momentum': [0.9]
    },
    'one_cycle': {
        'max_lr': [1e-3],
        #'pct_start': [0.3],
        #'div_factor': [25.0],
        #'final_div_factor': [1e4],
        #'base_momentum': [0.85],
        #'max_momentum': [0.95]
    }
}

DYNAMIC_LOSS_WEIGHTING_PARAMS = {
    'gradnorm': {
        'alpha': [0.12, 0.25],
        'learning_rate': [0.025, 0.05]
    },
    'uncertainty': {
        'log_variance_init': [0.0, -1.0]
    },
    'dwa': {
        'temperature': [2.0, 3.0],
        'window_size': [10, 15]
    },
    'yoto': {
        'alpha': [0.5, 0.7],
        'beta': [0.5, 0.7]
    },
    'adaptive_curriculum': {
        'initial_weights': [[1.0, 1.0, 1.0, 1.0, 1.0]],
        'adaptation_rate': [0.1, 0.2]
    }
}

ADVERSARIAL_PARAMS = {
    'discriminator_learning_rate': [1e-5],
    'discriminator_update_frequency': [2]
}

# Channel names mapping based on n_channels
# Maps number of channels to list of channel names in order
CHANNEL_NAMES_MAP = {
    2: ['SG', 'PRES'],
    4: ['SG', 'PRES','POROS', 'PERMI']
    # Add more mappings as needed
}

# Output directories (anchored to ROM_Refactored regardless of CWD)
OUTPUT_DIR = os.path.join(_this_dir, 'saved_models', 'grid_search')
SUMMARY_DIR = os.path.join(_this_dir, 'grid_search_results')
TIMING_LOG_DIR = os.path.join(_this_dir, 'timing_logs')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_processed_data_file(n_steps: int, n_channels: int, data_dir: str = None) -> Optional[str]:
    """
    Find the processed data file for a specific n_steps and n_channels value.
    
    Args:
        n_steps: Number of steps to find data for
        n_channels: Number of channels to find data for
        data_dir: Directory to search for processed data files
        
    Returns:
        Path to the processed data file, or None if not found
    """
    import glob
    import re
    
    if data_dir is None:
        data_dir = os.path.join(_this_dir, 'processed_data')
    
    # Resolve relative path
    if not os.path.isabs(data_dir):
        if not os.path.exists(data_dir):
            current_file_dir = _this_dir
            processed_data_path = os.path.join(current_file_dir, 'processed_data')
            processed_data_path = os.path.normpath(processed_data_path)
            if os.path.exists(processed_data_path):
                data_dir = processed_data_path
    
    if not os.path.exists(data_dir):
        return None
    
    # Find all processed data files
    pattern = os.path.join(data_dir, 'processed_data_*.h5')
    files = glob.glob(pattern)
    
    # Find file matching both n_steps AND n_channels
    for filepath in files:
        filename = os.path.basename(filepath)
        # Check for n_steps pattern (format: ..._nsteps{N}_...)
        nsteps_match = re.search(r'nsteps(\d+)_', filename)
        if not nsteps_match:
            continue
        
        file_n_steps = int(nsteps_match.group(1))
        if file_n_steps != n_steps:
            continue
        
        # Check for n_channels pattern (format: ..._ch{N}_...)
        ch_match = re.search(r'_ch(\d+)_', filename)
        if not ch_match:
            continue
        
        file_n_channels = int(ch_match.group(1))
        if file_n_channels == n_channels:
            return filepath
    
    return None


def validate_processed_data_file(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a processed data file can be opened and read.
    
    Args:
        filepath: Path to the processed data file
        
    Returns:
        Tuple of (is_valid, error_message)
        is_valid: True if file is valid, False otherwise
        error_message: Error message if file is invalid, None if valid
    """
    import h5py
    
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    
    try:
        # Try to open and read basic structure
        with h5py.File(filepath, 'r') as hf:
            # Check for required groups
            if 'metadata' not in hf:
                return False, "Missing 'metadata' group"
            if 'train' not in hf:
                return False, "Missing 'train' group"
            if 'eval' not in hf:
                return False, "Missing 'eval' group"
            
            # Try to access metadata attributes
            if 'metadata' in hf:
                _ = hf['metadata'].attrs.get('nsteps', None)
        
        return True, None
        
    except (OSError, IOError, ValueError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def get_channel_names(n_channels: int) -> List[str]:
    """
    Get channel names for a given number of channels.
    
    Args:
        n_channels: Number of channels
        
    Returns:
        List of channel names
    """
    if n_channels in CHANNEL_NAMES_MAP:
        return CHANNEL_NAMES_MAP[n_channels]
    else:
        # Fallback: generate generic names
        return [f'Channel_{i}' for i in range(n_channels)]


def validate_hyperparameter_combination(hyperparams: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a hyperparameter combination is valid.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate scheduler parameters are only included when scheduler type is not 'fixed'
    scheduler_type = hyperparams.get('lr_scheduler_type', 'fixed')
    if scheduler_type == 'fixed':
        # Fixed scheduler doesn't need additional parameters
        pass
    elif scheduler_type not in LR_SCHEDULER_PARAMS:
        return False, f"Unknown scheduler type: {scheduler_type}"
    
    # Validate dynamic loss weighting parameters
    dlw_enable = hyperparams.get('dynamic_loss_weighting_enable', False)
    dlw_method = hyperparams.get('dynamic_loss_weighting_method', 'gradnorm')
    if dlw_enable and dlw_method not in DYNAMIC_LOSS_WEIGHTING_PARAMS:
        return False, f"Unknown dynamic loss weighting method: {dlw_method}"
    
    # Validate adversarial parameters
    adv_enable = hyperparams.get('adversarial_enable', False)
    if adv_enable:
        # Adversarial parameters will be applied from ADVERSARIAL_PARAMS
        pass
    
    # Validate multimodal requires n_channels == 4
    mm_enable = hyperparams.get('enable_multimodal', False)
    if mm_enable and hyperparams.get('n_channels', 2) != 4:
        return False, "Multimodal mode requires n_channels=4 (SG, PRES, PERMI, POROS)"
    
    # Validate GNN requires n_channels == 4
    gnn_enable = hyperparams.get('enable_gnn', False)
    if gnn_enable and hyperparams.get('n_channels', 2) != 4:
        return False, "GNN mode requires n_channels=4 (SG, PRES, PERMI, POROS)"
    
    # GNN includes static/dynamic split internally — auto-override multimodal
    if gnn_enable and mm_enable:
        hyperparams['enable_multimodal'] = False
    
    # Validate FNO
    fno_enable = hyperparams.get('enable_fno', False)
    if fno_enable and hyperparams.get('n_channels', 2) != 4:
        return False, "FNO mode requires n_channels=4 (SG, PRES, PERMI, POROS)"
    if fno_enable and gnn_enable:
        return False, "FNO and GNN cannot both be enabled"
    # FNO includes static/dynamic split internally — auto-override multimodal
    if fno_enable and mm_enable:
        hyperparams['enable_multimodal'] = False
    
    # Validate multi_embedding_multimodal: highest-priority, mutually exclusive with FNO/GNN
    mem_enable = hyperparams.get('enable_multi_embedding_multimodal', False)
    if mem_enable:
        if hyperparams.get('enable_gnn', False):
            return False, "multi_embedding_multimodal cannot be combined with enable_gnn"
        if hyperparams.get('enable_fno', False):
            return False, "multi_embedding_multimodal cannot be combined with enable_fno (it composes FNO internally)"
        # multi_embedding subsumes the legacy two-branch multimodal mode
        if hyperparams.get('enable_multimodal', False):
            hyperparams['enable_multimodal'] = False
        if hyperparams.get('n_channels', 2) != 4:
            return False, "multi_embedding_multimodal requires n_channels=4 (SG, PRES, POROS, PERMI)"
        preset_name = hyperparams.get('mem_branches_preset')
        if preset_name not in MEM_PRESETS:
            return False, (f"Unknown mem_branches_preset: {preset_name}. "
                           f"Available: {list(MEM_PRESETS)}")
        branches = MEM_PRESETS[preset_name]
        total_branch_latent = sum(b['latent_dim'] for b in branches)
        if total_branch_latent != hyperparams.get('latent_dim', total_branch_latent):
            return False, (f"Sum of branch latent_dims ({total_branch_latent}) must equal "
                           f"latent_dim ({hyperparams.get('latent_dim')})")
        seen = []
        for b in branches:
            seen.extend(b['channels'])
        if sorted(seen) != [0, 1, 2, 3] or len(set(seen)) != 4:
            return False, (f"Branch channels must partition {{0,1,2,3}} exactly; got {seen}")
    
    return True, None


def generate_hyperparameter_combinations() -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters, including conditional nested parameters.
    
    Returns:
        List of dictionaries, each containing a hyperparameter set with nested parameters expanded
    """
    # Create a filtered grid that excludes dynamic_loss_weighting_method when enable is False
    filtered_grid = HYPERPARAMETER_GRID.copy()
    
    # Check if dynamic_loss_weighting_enable contains any True values
    dlw_enable_values = filtered_grid.get('dynamic_loss_weighting_enable', [False])
    has_enabled_dlw = any(dlw_enable_values) if isinstance(dlw_enable_values, list) else bool(dlw_enable_values)
    
    # If dynamic loss weighting is never enabled, remove method from grid to avoid unnecessary combinations
    if not has_enabled_dlw and 'dynamic_loss_weighting_method' in filtered_grid:
        # Remove method from grid, but keep a default value for when enable=False
        filtered_grid = {k: v for k, v in filtered_grid.items() if k != 'dynamic_loss_weighting_method'}
    
    # Base grid keys and values
    base_keys = list(filtered_grid.keys())
    base_values = list(filtered_grid.values())
    
    combinations = []
    
    # Generate base combinations
    for base_combination in itertools.product(*base_values):
        base_hyperparams = dict(zip(base_keys, base_combination))
        
        # If dynamic_loss_weighting_method was removed, set a default when enable=False
        if 'dynamic_loss_weighting_method' not in base_hyperparams:
            base_hyperparams['dynamic_loss_weighting_method'] = 'gradnorm'  # Default, won't be used when enable=False
        
        # Expand conditional parameters based on base hyperparameters
        expanded_combinations = expand_conditional_parameters(base_hyperparams)
        
        # Validate and add each expanded combination
        for expanded in expanded_combinations:
            is_valid, error_msg = validate_hyperparameter_combination(expanded)
            if is_valid:
                combinations.append(expanded)
            else:
                print(f"⚠️ Skipping invalid combination: {error_msg}")
    
    return combinations


def expand_conditional_parameters(base_hyperparams: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand conditional nested parameters based on base hyperparameters.
    
    Args:
        base_hyperparams: Base hyperparameter dictionary
        
    Returns:
        List of expanded hyperparameter dictionaries
    """
    expanded = [base_hyperparams.copy()]
    
    # Expand scheduler parameters
    scheduler_type = base_hyperparams.get('lr_scheduler_type', 'fixed')
    if scheduler_type != 'fixed' and scheduler_type in LR_SCHEDULER_PARAMS:
        scheduler_params = LR_SCHEDULER_PARAMS[scheduler_type]
        scheduler_keys = list(scheduler_params.keys())
        scheduler_values = list(scheduler_params.values())
        
        scheduler_combinations = []
        for combo in itertools.product(*scheduler_values):
            scheduler_dict = dict(zip(scheduler_keys, combo))
            scheduler_combinations.append(scheduler_dict)
        
        # Create new expanded combinations with scheduler params
        new_expanded = []
        for base in expanded:
            for scheduler_combo in scheduler_combinations:
                new_base = base.copy()
                new_base['lr_scheduler_params'] = scheduler_combo
                new_expanded.append(new_base)
        expanded = new_expanded
    
    # Expand dynamic loss weighting parameters
    dlw_enable = base_hyperparams.get('dynamic_loss_weighting_enable', False)
    dlw_method = base_hyperparams.get('dynamic_loss_weighting_method', 'gradnorm')
    
    if dlw_enable and dlw_method in DYNAMIC_LOSS_WEIGHTING_PARAMS:
        dlw_params = DYNAMIC_LOSS_WEIGHTING_PARAMS[dlw_method]
        dlw_keys = list(dlw_params.keys())
        dlw_values = list(dlw_params.values())
        
        dlw_combinations = []
        for combo in itertools.product(*dlw_values):
            dlw_dict = dict(zip(dlw_keys, combo))
            dlw_combinations.append(dlw_dict)
        
        # Create new expanded combinations with DLW params
        new_expanded = []
        for base in expanded:
            for dlw_combo in dlw_combinations:
                new_base = base.copy()
                new_base['dynamic_loss_weighting_params'] = dlw_combo
                new_expanded.append(new_base)
        expanded = new_expanded
    
    # Expand adversarial parameters
    adv_enable = base_hyperparams.get('adversarial_enable', False)
    
    if adv_enable:
        adv_keys = list(ADVERSARIAL_PARAMS.keys())
        adv_values = list(ADVERSARIAL_PARAMS.values())
        
        adv_combinations = []
        for combo in itertools.product(*adv_values):
            adv_dict = dict(zip(adv_keys, combo))
            adv_combinations.append(adv_dict)
        
        # Create new expanded combinations with adversarial params
        new_expanded = []
        for base in expanded:
            for adv_combo in adv_combinations:
                new_base = base.copy()
                new_base['adversarial_params'] = adv_combo
                new_expanded.append(new_base)
        expanded = new_expanded
    
    return expanded


def format_encoder_hidden_dims(encoder_hidden_dims: List[int]) -> str:
    """Format encoder hidden dims list as string."""
    return '-'.join(map(str, encoder_hidden_dims))


def format_scheduler_abbreviation(scheduler_type: str) -> str:
    """Get abbreviation for scheduler type."""
    abbrev_map = {
        'fixed': 'fix',
        'reduce_on_plateau': 'rop',
        'exponential_decay': 'exp',
        'step_decay': 'step',
        'cosine_annealing': 'cos',
        'cyclic': 'cyc',
        'one_cycle': '1cyc'
    }
    return abbrev_map.get(scheduler_type, scheduler_type[:4])


def format_dlw_abbreviation(method: str) -> str:
    """Get abbreviation for dynamic loss weighting method."""
    abbrev_map = {
        'gradnorm': 'grad',
        'uncertainty': 'unc',
        'dwa': 'dwa',
        'yoto': 'yoto',
        'adaptive_curriculum': 'acur'
    }
    return abbrev_map.get(method, method[:4])


def create_run_id(hyperparams: Dict[str, Any], run_index: int) -> str:
    """
    Create a unique run ID based on hyperparameters.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        run_index: Index of the run
        
    Returns:
        String run ID
    """
    parts = [f"run{run_index:04d}"]
    
    # Base parameters
    parts.append(f"bs{hyperparams['batch_size']}")
    parts.append(f"ld{hyperparams['latent_dim']}")
    parts.append(f"ns{hyperparams['n_steps']}")
    parts.append(f"ch{hyperparams['n_channels']}")
    
    # Scheduler
    scheduler_type = hyperparams.get('lr_scheduler_type', 'fixed')
    parts.append(f"sch{format_scheduler_abbreviation(scheduler_type)}")
    
    # Residual blocks
    if 'residual_blocks' in hyperparams:
        parts.append(f"rb{hyperparams['residual_blocks']}")
    
    # Normalization type (batchnorm or gdn)
    norm_type = hyperparams.get('norm_type', 'batchnorm')
    parts.append(f"norm{norm_type[:2]}")  # 'ba' for batchnorm, 'gd' for gdn
    
    # Encoder hidden dims
    if 'encoder_hidden_dims' in hyperparams:
        ehd_str = format_encoder_hidden_dims(hyperparams['encoder_hidden_dims'])
        parts.append(f"ehd{ehd_str}")
    
    # Only include optional flags when enabled to keep filenames short
    dlw_enable = hyperparams.get('dynamic_loss_weighting_enable', False)
    if dlw_enable:
        dlw_method = hyperparams.get('dynamic_loss_weighting_method', 'gradnorm')
        parts.append(f"dlw{format_dlw_abbreviation(dlw_method)}")
    
    adv_enable = hyperparams.get('adversarial_enable', False)
    if adv_enable:
        parts.append("adv")
    
    fft_enable = hyperparams.get('enable_fft_loss', False)
    if fft_enable:
        parts.append("fft")
    
    mask_enable = hyperparams.get('enable_inactive_masking', False)
    if mask_enable:
        parts.append("mask")
    
    mm_enable = hyperparams.get('enable_multimodal', False)
    if mm_enable:
        parts.append("mm")
    
    gnn_enable = hyperparams.get('enable_gnn', False)
    if gnn_enable:
        parts.append("gnn")
    
    fno_enable = hyperparams.get('enable_fno', False)
    if fno_enable:
        parts.append("fno")
    
    # Multi-embedding multimodal: include preset name and a fixed token used for
    # downstream filename-based detection (testing dashboard, RL, DigitalTwin).
    mem_enable = hyperparams.get('enable_multi_embedding_multimodal', False)
    if mem_enable:
        preset = hyperparams.get('mem_branches_preset', 'unknown')
        # Sanitize for filename: replace underscores with hyphens to keep _ as
        # a token separator in run_id parsing.
        preset_tag = preset.replace('_', '-')
        parts.append(f"mem-{preset_tag}")
        parts.append("memT")
    
    # Transition type
    trn = hyperparams.get('transition_type', 'linear')
    if trn != 'linear':
        parts.append(f"trn{trn.upper()}")
    
    return '_'.join(parts)


def create_run_name(hyperparams: Dict[str, Any], run_index: int,
                    epochs: int = None, learning_rate: float = None) -> str:
    """
    Create a human-readable run name for wandb.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        run_index: Index of the run
        epochs: Number of training epochs (from config)
        learning_rate: Learning rate value (from config)
        
    Returns:
        String run name
    """
    parts = []
    
    # Transition model name (always first for clarity)
    trn = hyperparams.get('transition_type', 'linear')
    parts.append(f"trn{trn.upper()}")
    
    # Base parameters
    parts.append(f"bs{hyperparams['batch_size']}")
    parts.append(f"ld{hyperparams['latent_dim']}")
    parts.append(f"ns{hyperparams['n_steps']}")
    parts.append(f"ch{hyperparams['n_channels']}")
    
    # Epochs
    if epochs is not None:
        parts.append(f"ep{epochs}")
    
    # Learning rate
    if learning_rate is not None:
        parts.append(f"lr{learning_rate:.0e}")
    
    # Scheduler (abbreviated)
    scheduler_type = hyperparams.get('lr_scheduler_type', 'fixed')
    parts.append(f"sch{format_scheduler_abbreviation(scheduler_type)}")
    
    # Residual blocks
    if 'residual_blocks' in hyperparams:
        parts.append(f"rb{hyperparams['residual_blocks']}")
    
    # Normalization type
    norm_type = hyperparams.get('norm_type', 'batchnorm')
    parts.append(f"norm{norm_type[:2]}")
    
    # Encoder hidden dims
    if 'encoder_hidden_dims' in hyperparams:
        ehd_str = format_encoder_hidden_dims(hyperparams['encoder_hidden_dims'])
        parts.append(f"ehd{ehd_str}")
    
    # Dynamic loss weighting (only if enabled)
    dlw_enable = hyperparams.get('dynamic_loss_weighting_enable', False)
    if dlw_enable:
        dlw_method = hyperparams.get('dynamic_loss_weighting_method', 'gradnorm')
        parts.append(f"dlw{format_dlw_abbreviation(dlw_method)}")
    
    # Adversarial (only if enabled)
    adv_enable = hyperparams.get('adversarial_enable', False)
    if adv_enable:
        parts.append("adv")
    
    # FFT Loss (only if enabled)
    fft_enable = hyperparams.get('enable_fft_loss', False)
    if fft_enable:
        parts.append("fft")
    
    # Inactive Cell Masking (only if enabled)
    mask_enable = hyperparams.get('enable_inactive_masking', False)
    if mask_enable:
        parts.append("mask")
    
    # Multimodal (only if enabled)
    mm_enable = hyperparams.get('enable_multimodal', False)
    if mm_enable:
        parts.append("mm")
    
    # GNN (only if enabled)
    gnn_enable = hyperparams.get('enable_gnn', False)
    if gnn_enable:
        parts.append("gnn")
    
    # FNO (only if enabled)
    fno_enable = hyperparams.get('enable_fno', False)
    if fno_enable:
        parts.append("fno")
    
    # Multi-embedding multimodal (only if enabled)
    mem_enable = hyperparams.get('enable_multi_embedding_multimodal', False)
    if mem_enable:
        preset = hyperparams.get('mem_branches_preset', 'unknown')
        parts.append(f"mem-{preset.replace('_', '-')}")
    
    return '_'.join(parts)


def create_run_model_filename(component: str, hyperparams: Dict[str, Any], 
                              num_train: int, num_well: int, run_id: str) -> str:
    """
    Create model filename for a specific run.
    
    Args:
        component: Model component ('encoder', 'decoder', 'transition')
        hyperparams: Dictionary of hyperparameters
        num_train: Number of training samples
        num_well: Number of wells
        run_id: Unique run ID (already includes all parameters)
        
    Returns:
        Formatted filename string
    """
    return f"e2co_{component}_grid_{run_id}.h5"


def update_config_with_hyperparams(config_path: str, hyperparams: Dict[str, Any]) -> Config:
    """
    Create a new config with hyperparameters applied.
    
    Args:
        config_path: Path to config file
        hyperparams: Dictionary of hyperparameters to apply
        
    Returns:
        New Config object with hyperparameters applied
    """
    # Load fresh config for each run (this gets default learning_rate and lr_scheduler)
    config = Config(config_path)
    
    # Validate that learning_rate is fixed (not in hyperparams)
    if 'learning_rate' in hyperparams:
        raise ValueError("learning_rate should not be in hyperparams - it must remain fixed")
    
    # Ensure learning rate is fixed (read from config, do not modify)
    original_lr = config.training['learning_rate']
    
    # Update training hyperparameters
    config.set('training.batch_size', hyperparams['batch_size'])
    config.set('training.nsteps', hyperparams['n_steps'])
    
    # Update model hyperparameters
    config.set('model.latent_dim', hyperparams['latent_dim'])
    
    # Update n_channels related config (matching dashboard logic)
    n_channels = hyperparams['n_channels']
    if 'model' not in config.config:
        config.config['model'] = {}
    config.config['model']['n_channels'] = n_channels
    
    # Update data.input_shape[0] to match n_channels
    if 'data' in config.config and 'input_shape' in config.config['data']:
        if isinstance(config.config['data']['input_shape'], list) and len(config.config['data']['input_shape']) > 0:
            config.config['data']['input_shape'][0] = n_channels
    
    # Update encoder.conv_layers.conv1[0] if it exists (first conv layer input channels)
    if 'encoder' in config.config and 'conv_layers' in config.config['encoder']:
        if 'conv1' in config.config['encoder']['conv_layers']:
            conv1 = config.config['encoder']['conv_layers']['conv1']
            if isinstance(conv1, list) and len(conv1) > 0:
                conv1[0] = n_channels
    
    # Update decoder final_conv output channels if exists
    if 'decoder' in config.config and 'deconv_layers' in config.config['decoder']:
        if 'final_conv' in config.config['decoder']['deconv_layers']:
            final_conv_config = config.config['decoder']['deconv_layers']['final_conv']
            if isinstance(final_conv_config, list) and len(final_conv_config) > 1:
                if final_conv_config[1] is not None:
                    final_conv_config[1] = n_channels
    
    # Update learning rate scheduler
    scheduler_type = hyperparams.get('lr_scheduler_type', 'fixed')
    if scheduler_type == 'fixed':
        scheduler_type = 'constant'  # Map 'fixed' to 'constant' for config compatibility
    
    config.set('learning_rate_scheduler.enable', scheduler_type != 'constant')
    config.set('learning_rate_scheduler.type', scheduler_type)
    
    # Apply scheduler-specific parameters
    if scheduler_type != 'constant' and 'lr_scheduler_params' in hyperparams:
        scheduler_params = hyperparams['lr_scheduler_params']
        
        if scheduler_type == 'reduce_on_plateau':
            if 'factor' in scheduler_params:
                config.set('learning_rate_scheduler.reduce_on_plateau.factor', scheduler_params['factor'])
            if 'patience' in scheduler_params:
                config.set('learning_rate_scheduler.reduce_on_plateau.patience', scheduler_params['patience'])
            if 'threshold' in scheduler_params:
                config.set('learning_rate_scheduler.reduce_on_plateau.threshold', scheduler_params['threshold'])
            if 'cooldown' in scheduler_params:
                config.set('learning_rate_scheduler.reduce_on_plateau.cooldown', scheduler_params['cooldown'])
            if 'min_lr' in scheduler_params:
                config.set('learning_rate_scheduler.reduce_on_plateau.min_lr', scheduler_params['min_lr'])
        
        elif scheduler_type == 'exponential_decay':
            if 'gamma' in scheduler_params:
                config.set('learning_rate_scheduler.exponential_decay.gamma', scheduler_params['gamma'])
        
        elif scheduler_type == 'step_decay':
            if 'step_size' in scheduler_params:
                config.set('learning_rate_scheduler.step_decay.step_size', scheduler_params['step_size'])
            if 'gamma' in scheduler_params:
                config.set('learning_rate_scheduler.step_decay.gamma', scheduler_params['gamma'])
        
        elif scheduler_type == 'cosine_annealing':
            if 'T_max' in scheduler_params:
                config.set('learning_rate_scheduler.cosine_annealing.T_max', scheduler_params['T_max'])
            if 'eta_min' in scheduler_params:
                config.set('learning_rate_scheduler.cosine_annealing.eta_min', scheduler_params['eta_min'])
        
        elif scheduler_type == 'cyclic':
            if 'base_lr' in scheduler_params:
                config.set('learning_rate_scheduler.cyclic.base_lr', scheduler_params['base_lr'])
            if 'max_lr' in scheduler_params:
                config.set('learning_rate_scheduler.cyclic.max_lr', scheduler_params['max_lr'])
            if 'step_size_up' in scheduler_params:
                config.set('learning_rate_scheduler.cyclic.step_size_up', scheduler_params['step_size_up'])
            if 'gamma' in scheduler_params:
                config.set('learning_rate_scheduler.cyclic.gamma', scheduler_params['gamma'])
            if 'base_momentum' in scheduler_params:
                config.set('learning_rate_scheduler.cyclic.base_momentum', scheduler_params['base_momentum'])
            if 'max_momentum' in scheduler_params:
                config.set('learning_rate_scheduler.cyclic.max_momentum', scheduler_params['max_momentum'])
        
        elif scheduler_type == 'one_cycle':
            if 'max_lr' in scheduler_params:
                config.set('learning_rate_scheduler.one_cycle.max_lr', scheduler_params['max_lr'])
            if 'pct_start' in scheduler_params:
                config.set('learning_rate_scheduler.one_cycle.pct_start', scheduler_params['pct_start'])
            if 'div_factor' in scheduler_params:
                config.set('learning_rate_scheduler.one_cycle.div_factor', scheduler_params['div_factor'])
            if 'final_div_factor' in scheduler_params:
                config.set('learning_rate_scheduler.one_cycle.final_div_factor', scheduler_params['final_div_factor'])
            if 'base_momentum' in scheduler_params:
                config.set('learning_rate_scheduler.one_cycle.base_momentum', scheduler_params['base_momentum'])
            if 'max_momentum' in scheduler_params:
                config.set('learning_rate_scheduler.one_cycle.max_momentum', scheduler_params['max_momentum'])
    
    # Update residual blocks
    if 'residual_blocks' in hyperparams:
        if 'encoder' not in config.config:
            config.config['encoder'] = {}
        config.config['encoder']['residual_blocks'] = hyperparams['residual_blocks']
    
    # Update normalization type (batchnorm or gdn)
    if 'norm_type' in hyperparams:
        norm_type = hyperparams['norm_type']
        if 'encoder' not in config.config:
            config.config['encoder'] = {}
        if 'decoder' not in config.config:
            config.config['decoder'] = {}
        # Set norm_type for both encoder and decoder
        config.config['encoder']['norm_type'] = norm_type
        config.config['decoder']['norm_type'] = norm_type
    
    # Update encoder hidden dimensions
    if 'encoder_hidden_dims' in hyperparams:
        if 'transition' not in config.config:
            config.config['transition'] = {}
        config.config['transition']['encoder_hidden_dims'] = hyperparams['encoder_hidden_dims']
    
    # Update transition model type
    if 'transition_type' in hyperparams:
        if 'transition' not in config.config:
            config.config['transition'] = {}
        config.config['transition']['type'] = hyperparams['transition_type']
    
    # Update dynamic loss weighting
    dlw_enable = hyperparams.get('dynamic_loss_weighting_enable', False)
    if 'dynamic_loss_weighting' not in config.config:
        config.config['dynamic_loss_weighting'] = {}
    
    config.config['dynamic_loss_weighting']['enable'] = dlw_enable
    
    if dlw_enable:
        dlw_method = hyperparams.get('dynamic_loss_weighting_method', 'gradnorm')
        config.config['dynamic_loss_weighting']['method'] = dlw_method
        
        # Apply method-specific parameters
        if 'dynamic_loss_weighting_params' in hyperparams:
            dlw_params = hyperparams['dynamic_loss_weighting_params']
            
            if dlw_method == 'gradnorm':
                if 'alpha' in dlw_params:
                    config.config['dynamic_loss_weighting']['gradnorm'] = config.config['dynamic_loss_weighting'].get('gradnorm', {})
                    config.config['dynamic_loss_weighting']['gradnorm']['alpha'] = dlw_params['alpha']
                if 'learning_rate' in dlw_params:
                    config.config['dynamic_loss_weighting']['gradnorm'] = config.config['dynamic_loss_weighting'].get('gradnorm', {})
                    config.config['dynamic_loss_weighting']['gradnorm']['learning_rate'] = dlw_params['learning_rate']
            
            elif dlw_method == 'uncertainty':
                if 'log_variance_init' in dlw_params:
                    config.config['dynamic_loss_weighting']['uncertainty'] = config.config['dynamic_loss_weighting'].get('uncertainty', {})
                    config.config['dynamic_loss_weighting']['uncertainty']['log_variance_init'] = dlw_params['log_variance_init']
            
            elif dlw_method == 'dwa':
                if 'temperature' in dlw_params:
                    config.config['dynamic_loss_weighting']['dwa'] = config.config['dynamic_loss_weighting'].get('dwa', {})
                    config.config['dynamic_loss_weighting']['dwa']['temperature'] = dlw_params['temperature']
                if 'window_size' in dlw_params:
                    config.config['dynamic_loss_weighting']['dwa'] = config.config['dynamic_loss_weighting'].get('dwa', {})
                    config.config['dynamic_loss_weighting']['dwa']['window_size'] = dlw_params['window_size']
            
            elif dlw_method == 'yoto':
                if 'alpha' in dlw_params:
                    config.config['dynamic_loss_weighting']['yoto'] = config.config['dynamic_loss_weighting'].get('yoto', {})
                    config.config['dynamic_loss_weighting']['yoto']['alpha'] = dlw_params['alpha']
                if 'beta' in dlw_params:
                    config.config['dynamic_loss_weighting']['yoto'] = config.config['dynamic_loss_weighting'].get('yoto', {})
                    config.config['dynamic_loss_weighting']['yoto']['beta'] = dlw_params['beta']
            
            elif dlw_method == 'adaptive_curriculum':
                if 'initial_weights' in dlw_params:
                    config.config['dynamic_loss_weighting']['adaptive_curriculum'] = config.config['dynamic_loss_weighting'].get('adaptive_curriculum', {})
                    config.config['dynamic_loss_weighting']['adaptive_curriculum']['initial_weights'] = dlw_params['initial_weights']
                if 'adaptation_rate' in dlw_params:
                    config.config['dynamic_loss_weighting']['adaptive_curriculum'] = config.config['dynamic_loss_weighting'].get('adaptive_curriculum', {})
                    config.config['dynamic_loss_weighting']['adaptive_curriculum']['adaptation_rate'] = dlw_params['adaptation_rate']
    
    # Update adversarial training
    adv_enable = hyperparams.get('adversarial_enable', False)
    if 'adversarial' not in config.config:
        config.config['adversarial'] = {}
    
    config.config['adversarial']['enable'] = adv_enable
    
    if adv_enable:
        # Apply adversarial parameters
        if 'adversarial_params' in hyperparams:
            adv_params = hyperparams['adversarial_params']
            
            if 'discriminator_learning_rate' in adv_params:
                config.config['adversarial']['discriminator_learning_rate'] = adv_params['discriminator_learning_rate']
            if 'discriminator_update_frequency' in adv_params:
                config.config['adversarial']['discriminator_update_frequency'] = adv_params['discriminator_update_frequency']
        
        # Enable discriminator in config
        if 'discriminator' not in config.config:
            config.config['discriminator'] = {}
        config.config['discriminator']['enable'] = True
    
    # Update VAE (Variational Autoencoder) configuration
    if 'enable_vae' in hyperparams:
        if 'model' not in config.config:
            config.config['model'] = {}
        config.config['model']['enable_vae'] = hyperparams['enable_vae']
    
    if 'enable_kl_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_kl_loss'] = hyperparams['enable_kl_loss']
    
    if 'lambda_kl_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['lambda_kl_loss'] = hyperparams['lambda_kl_loss']
    
    if 'enable_kl_annealing' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_kl_annealing'] = hyperparams['enable_kl_annealing']
    
    # Update FFT Loss configuration
    if 'enable_fft_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_fft_loss'] = hyperparams['enable_fft_loss']
    
    if 'lambda_fft_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['lambda_fft_loss'] = hyperparams['lambda_fft_loss']
    
    # Update Inactive Cell Masking configuration
    if 'enable_inactive_masking' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_inactive_masking'] = hyperparams['enable_inactive_masking']
    
    # Update Multimodal configuration
    if hyperparams.get('enable_multimodal', False):
        if 'multimodal' not in config.config:
            config.config['multimodal'] = {}
        config.config['multimodal']['enable'] = True
        latent_dim = hyperparams.get('latent_dim', config.config['model']['latent_dim'])
        static_ld = config.config.get('multimodal', {}).get('static_latent_dim', 32)
        dynamic_ld = config.config.get('multimodal', {}).get('dynamic_latent_dim', 96)
        if static_ld + dynamic_ld != latent_dim:
            dynamic_ld = latent_dim - static_ld
            config.config['multimodal']['dynamic_latent_dim'] = dynamic_ld
    else:
        if 'multimodal' not in config.config:
            config.config['multimodal'] = {}
        config.config['multimodal']['enable'] = False
    
    # Update GNN configuration
    if hyperparams.get('enable_gnn', False):
        if 'gnn' not in config.config:
            config.config['gnn'] = {}
        config.config['gnn']['enable'] = True
        # GNN handles static/dynamic split internally, disable multimodal
        config.config['multimodal']['enable'] = False
        # GNN requires inactive masking
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_inactive_masking'] = True
    else:
        if 'gnn' not in config.config:
            config.config['gnn'] = {}
        config.config['gnn']['enable'] = False
    
    # Update FNO configuration
    if hyperparams.get('enable_fno', False):
        if 'fno' not in config.config:
            config.config['fno'] = {}
        config.config['fno']['enable'] = True
        config.config['multimodal']['enable'] = False
        config.config['gnn']['enable'] = False
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_invertibility_loss'] = True
        fno_norm = hyperparams.get('fno_norm_type', 'batchnorm')
        if 'encoder' not in config.config['fno']:
            config.config['fno']['encoder'] = {}
        if 'decoder' not in config.config['fno']:
            config.config['fno']['decoder'] = {}
        config.config['fno']['encoder']['norm_type'] = fno_norm
        config.config['fno']['decoder']['norm_type'] = fno_norm
    else:
        if 'fno' not in config.config:
            config.config['fno'] = {}
        config.config['fno']['enable'] = False
    
    # Update Multi-Embedding Multimodal configuration
    if hyperparams.get('enable_multi_embedding_multimodal', False):
        if 'multi_embedding' not in config.config:
            config.config['multi_embedding'] = {}
        config.config['multi_embedding']['enable'] = True
        preset_name = hyperparams['mem_branches_preset']
        config.config['multi_embedding']['preset'] = preset_name
        # Deep-copy the preset to avoid mutating the global table later
        import copy as _copy
        config.config['multi_embedding']['branches'] = _copy.deepcopy(MEM_PRESETS[preset_name])
        # Force the legacy switches off so the model factory dispatches to the new class
        config.config.setdefault('multimodal', {})['enable'] = False
        config.config.setdefault('gnn', {})['enable']        = False
        config.config.setdefault('fno', {})['enable']        = False
        # If any branch uses FNO, enable invertibility loss by default (per-branch in model)
        if any(b['encoder']['type'] == 'fno' or (b.get('decoder') and b['decoder']['type'] == 'fno')
               for b in MEM_PRESETS[preset_name]):
            config.config.setdefault('loss', {})['enable_invertibility_loss'] = True
    else:
        if 'multi_embedding' not in config.config:
            config.config['multi_embedding'] = {}
        config.config['multi_embedding']['enable'] = False
    
    # Encoder enhancement: Jacobian regularization
    if 'enable_jacobian_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_jacobian_loss'] = hyperparams['enable_jacobian_loss']
    if 'lambda_jacobian_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['lambda_jacobian_loss'] = hyperparams['lambda_jacobian_loss']

    # Encoder enhancement: Cycle-consistency loss
    if 'enable_cycle_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['enable_cycle_loss'] = hyperparams['enable_cycle_loss']
    if 'lambda_cycle_loss' in hyperparams:
        if 'loss' not in config.config:
            config.config['loss'] = {}
        config.config['loss']['lambda_cycle_loss'] = hyperparams['lambda_cycle_loss']

    # Encoder enhancement: Scheduled sampling
    if 'enable_scheduled_sampling' in hyperparams:
        if 'training' not in config.config:
            config.config['training'] = {}
        config.config['training']['enable_scheduled_sampling'] = hyperparams['enable_scheduled_sampling']

    # Encoder enhancement: Latent noise injection
    if 'enable_latent_noise' in hyperparams:
        if 'training' not in config.config:
            config.config['training'] = {}
        config.config['training']['enable_latent_noise'] = hyperparams['enable_latent_noise']

    # Verify learning rate was not accidentally modified
    if config.training['learning_rate'] != original_lr:
        raise ValueError(f"Learning rate was modified! Expected {original_lr}, got {config.training['learning_rate']}")
    
    # Re-resolve dynamic values after changes
    config._resolve_dynamic_values()
    
    return config


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(config: Config, loaded_data: Dict[str, Any], 
                run_id: str, run_name: str, output_dir: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Train a model with given configuration.
    
    Args:
        config: Configuration object
        loaded_data: Dictionary containing training/eval data
        run_id: Unique run identifier
        run_name: Human-readable run name
        output_dir: Directory to save models
        
    Returns:
        Tuple of (results_dict, model_paths_dict)
    """
    # Extract data
    STATE_train = loaded_data['STATE_train']
    BHP_train = loaded_data['BHP_train']
    Yobs_train = loaded_data['Yobs_train']
    STATE_eval = loaded_data['STATE_eval']
    BHP_eval = loaded_data['BHP_eval']
    Yobs_eval = loaded_data['Yobs_eval']
    dt_train = loaded_data['dt_train']
    dt_eval = loaded_data['dt_eval']
    
    # Extract case indices for mask lookup (None if not available in older data files)
    case_indices_train = loaded_data.get('case_indices_train', None)
    case_indices_eval = loaded_data.get('case_indices_eval', None)
    
    metadata = loaded_data['metadata']
    num_train = metadata.get('num_train', 0)
    num_well = metadata.get('num_well', 0)
    
    # Validate n_steps
    loaded_nsteps = metadata.get('nsteps', None)
    config_nsteps = config.training['nsteps']
    if loaded_nsteps is not None and loaded_nsteps != config_nsteps:
        raise ValueError(
            f"Data preprocessing used n_steps={loaded_nsteps}, but training config has n_steps={config_nsteps}."
        )
    
    # Get device
    device = config.runtime.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    config.device = torch.device(device)
    
    # Create model filenames
    os.makedirs(output_dir, exist_ok=True)
    hyperparams_dict = {
        'batch_size': config.training['batch_size'], 
        'latent_dim': config.model['latent_dim'],
        'n_steps': config.training['nsteps'],
        'n_channels': config.model.get('n_channels', 2)  # Default to 2 if not set
    }
    encoder_file = os.path.join(output_dir, create_run_model_filename('encoder', 
        hyperparams_dict, num_train, num_well, run_id))
    decoder_file = os.path.join(output_dir, create_run_model_filename('decoder', 
        hyperparams_dict, num_train, num_well, run_id))
    transition_file = os.path.join(output_dir, create_run_model_filename('transition', 
        hyperparams_dict, num_train, num_well, run_id))
    
    # Initialize WandB logger
    wandb_logger = create_wandb_logger(config)
    
    # Initialize model
    my_rom = ROMWithE2C(config).to(config.device)
    wandb_logger.watch_model(my_rom)
    
    # Setup schedulers
    num_batch = int(num_train / config.training['batch_size'])
    total_training_steps = num_batch * config.training['epoch']
    my_rom.setup_schedulers_with_steps(total_training_steps)
    
    # Training loop
    best_loss = 1.0e9
    best_observation_loss = 1.0e9
    best_reconstruction_loss = 1.0e9
    best_model_criterion = config.runtime.get('best_model_criterion', 'total_loss')
    global_step = 0
    
    for e in range(config.training['epoch']):
        # Update KL annealing at the start of each epoch
        current_kl_lambda = my_rom.update_kl_annealing(e)
        if current_kl_lambda is not None and e % 50 == 0:
            print(f"   📊 KL Lambda: {current_kl_lambda:.6e}")

        # Encoder enhancements: update scheduled sampling and noise annealing
        model = my_rom.model
        if hasattr(model, 'update_noise_std'):
            model.update_noise_std(e)
        if hasattr(model, 'set_teacher_forcing_ratio') and config['training'].get('enable_scheduled_sampling', False):
            ss_cfg = config['training'].get('scheduled_sampling', {})
            stype = ss_cfg.get('schedule_type', 'exponential')
            start_e = ss_cfg.get('start_epoch', 0)
            end_e = ss_cfg.get('end_epoch', 150)
            p_start = ss_cfg.get('start_p', 1.0)
            p_end = ss_cfg.get('end_p', 0.1)
            if e < start_e:
                ratio = p_start
            elif e >= end_e:
                ratio = p_end
            else:
                progress = (e - start_e) / max(end_e - start_e, 1)
                if stype == 'linear':
                    ratio = p_start + (p_end - p_start) * progress
                elif stype == 'inverse_sigmoid':
                    k = 5.0
                    ratio = p_end + (p_start - p_end) * (k / (k + pow(2.718, (progress * 2 * k - k))))
                else:
                    ratio = p_start * pow(p_end / max(p_start, 1e-8), progress)
            model.set_teacher_forcing_ratio(ratio)
        
        for ib in range(num_batch):
            ind0 = ib * config.training['batch_size']
            ind1 = ind0 + config.training['batch_size']
            
            X_batch = [state[ind0:ind1, ...] for state in STATE_train]
            U_batch = [bhp[ind0:ind1, ...] for bhp in BHP_train]
            Y_batch = [yobs[ind0:ind1, ...] for yobs in Yobs_train]
            dt_batch = dt_train[ind0:ind1, ...]
            
            # Get case indices for this batch (for case-specific mask lookup)
            batch_case_indices = case_indices_train[ind0:ind1] if case_indices_train is not None else None
            
            inputs = (X_batch, U_batch, Y_batch, dt_batch, batch_case_indices)
            my_rom.update(inputs)
            
            # SINDy sequential thresholding: zero out small coefficients periodically
            transition_ref = getattr(my_rom.model, 'transition', None)
            if transition_ref is None:
                transition_ref = getattr(my_rom.model, 'conditioned_transition', None)
            if transition_ref is not None and hasattr(transition_ref, 'apply_threshold'):
                sindy_cfg = config.transition.get('sindy', {})
                threshold_freq = sindy_cfg.get('threshold_frequency', 500)
                threshold_val = sindy_cfg.get('coefficient_threshold', 0.1)
                if global_step % threshold_freq == 0 and global_step > 0:
                    transition_ref.apply_threshold(threshold_val)
            
            global_step += 1
            wandb_logger.log_training_step(my_rom, e+1, ib+1, global_step)
            
            if ib % config.runtime.get('print_interval', 10) == 0:
                # Evaluate
                X_batch_eval = [state for state in STATE_eval]
                U_batch_eval = [bhp for bhp in BHP_eval]
                Y_batch_eval = [yobs for yobs in Yobs_eval]
                test_inputs = (X_batch_eval, U_batch_eval, Y_batch_eval, dt_eval, case_indices_eval)
                my_rom.evaluate(test_inputs)
                
                wandb_logger.log_evaluation_step(my_rom, e+1, global_step)
        
        # Step scheduler
        current_eval_loss = my_rom.test_loss.item() if hasattr(my_rom.test_loss, 'item') else float(my_rom.test_loss)
        my_rom.step_scheduler_on_epoch(validation_loss=current_eval_loss)
        
        # Save best model
        if config.runtime.get('save_best_model', True):
            should_save = False
            if best_model_criterion == 'observation_loss':
                current_obs_loss = my_rom.get_test_observation_loss()
                if current_obs_loss < best_observation_loss:
                    best_observation_loss = current_obs_loss
                    should_save = True
            elif best_model_criterion == 'reconstruction_loss':
                current_recon_loss = my_rom.get_test_reconstruction_loss()
                if current_recon_loss < best_reconstruction_loss:
                    best_reconstruction_loss = current_recon_loss
                    should_save = True
            else:  # total_loss
                if my_rom.test_loss < best_loss:
                    best_loss = my_rom.test_loss
                    should_save = True
            
            if should_save:
                my_rom.model.save_weights_to_file(encoder_file, decoder_file, transition_file)
    
    # Collect results
    results = {
        'final_loss': float(my_rom.test_loss.item() if hasattr(my_rom.test_loss, 'item') else float(my_rom.test_loss)),
        'final_reconstruction_loss': float(my_rom.get_test_reconstruction_loss()),
        'final_transition_loss': float(my_rom.get_test_transition_loss()),
        'final_observation_loss': float(my_rom.get_test_observation_loss()),
        'best_loss': float(best_loss.item() if hasattr(best_loss, 'item') else float(best_loss)),
        'best_observation_loss': float(best_observation_loss),
        'best_reconstruction_loss': float(best_reconstruction_loss),
    }
    
    model_paths = {
        'encoder': encoder_file,
        'decoder': decoder_file,
        'transition': transition_file
    }
    
    # Finish wandb run
    wandb_logger.finish()
    
    return results, model_paths


def run_single_training(config_path: str, hyperparams: Dict[str, Any], run_index: int, 
                       output_dir: str, processed_data_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    Run a single training with given hyperparameters.
    
    Args:
        config_path: Path to base config file
        hyperparams: Dictionary of hyperparameters (includes n_steps)
        run_index: Index of the run
        output_dir: Directory to save models
        processed_data_dir: Directory containing processed data files
        
    Returns:
        Dictionary with run results or None if failed
    """
    run_id = create_run_id(hyperparams, run_index)
    run_name = create_run_name(hyperparams, run_index)
    
    try:
        # Find and load the appropriate processed data file for this n_steps and n_channels
        n_steps = hyperparams['n_steps']
        n_channels = hyperparams['n_channels']
        data_filepath = find_processed_data_file(n_steps, n_channels, processed_data_dir)
        
        if data_filepath is None:
            error_msg = f"No processed data file found for n_steps={n_steps}, n_channels={n_channels} in {processed_data_dir}"
            print(f"❌ Run {run_id} failed: {error_msg}")
            return {
                'run_id': run_id,
                'run_name': run_name,
                'status': 'failed',
                'error': error_msg,
                **hyperparams,
                'channel_names': get_channel_names(n_channels),
                'learning_rate': 'N/A',
                'lr_scheduler': 'N/A'
            }
        
        # Load the data file
        loaded_data = load_processed_data(filepath=data_filepath, n_channels=n_channels)
        
        if loaded_data is None:
            error_msg = f"Failed to load processed data from {data_filepath}"
            print(f"❌ Run {run_id} failed: {error_msg}")
            return {
                'run_id': run_id,
                'run_name': run_name,
                'status': 'failed',
                'error': error_msg,
                **hyperparams,
                'learning_rate': 'N/A',
                'lr_scheduler': 'N/A'
            }
        
        # Create config with hyperparameters
        config = update_config_with_hyperparams(config_path, hyperparams)
        
        # Re-create run_name with epochs and learning rate from config
        cfg_epochs = config.training.get('epoch', None)
        cfg_lr = config.training.get('learning_rate', None)
        run_name = create_run_name(hyperparams, run_index,
                                   epochs=cfg_epochs, learning_rate=cfg_lr)
        
        # Update wandb config for this run
        if 'wandb' not in config.config['runtime']:
            config.config['runtime']['wandb'] = {}
        config.config['runtime']['wandb']['name'] = run_name
        config.config['runtime']['wandb']['enable'] = True
        
        # Run training with timing
        with Timer("training", log_dir=TIMING_LOG_DIR) as timer:
            results, model_paths = train_model(config, loaded_data, run_id, run_name, output_dir)
            
            # Collect metadata for timing log
            metadata = collect_training_metadata(config, loaded_data)
            metadata.update(results)
            metadata.update(hyperparams)
            metadata['run_id'] = run_id
            metadata['run_name'] = run_name
            metadata['data_file'] = os.path.basename(data_filepath)
            timer.metadata = metadata
        
        # Combine results with hyperparameters and metadata
        # Add learning_rate and lr_scheduler from config for reference
        run_result = {
            'run_id': run_id,
            'run_name': run_name,
            'status': 'success',
            **hyperparams,  # Includes all hyperparameters including nested ones
            'channel_names': get_channel_names(n_channels),
            'learning_rate': config.training['learning_rate'],  # From config default (FIXED)
            'lr_scheduler': config.learning_rate_scheduler.get('type', 'constant'),
            'data_file': os.path.basename(data_filepath),
            **results,
            **model_paths
        }
        
        # Ensure nested parameters are included in results (for CSV serialization)
        if 'lr_scheduler_params' not in run_result and 'lr_scheduler_params' in hyperparams:
            run_result['lr_scheduler_params'] = hyperparams['lr_scheduler_params']
        if 'dynamic_loss_weighting_params' not in run_result and 'dynamic_loss_weighting_params' in hyperparams:
            run_result['dynamic_loss_weighting_params'] = hyperparams['dynamic_loss_weighting_params']
        if 'adversarial_params' not in run_result and 'adversarial_params' in hyperparams:
            run_result['adversarial_params'] = hyperparams['adversarial_params']
        
        return run_result
        
    except Exception as e:
        print(f"❌ Run {run_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'run_id': run_id,
            'run_name': run_name,
            'status': 'failed',
            'error': str(e),
            **hyperparams,
            'channel_names': get_channel_names(hyperparams.get('n_channels', 2)),
            'learning_rate': 'N/A',
            'lr_scheduler': 'N/A'
        }


# ============================================================================
# MAIN GRID SEARCH LOOP
# ============================================================================

def main():
    """Main grid search orchestration"""
    print("=" * 80)
    print("🔍 HYPERPARAMETER GRID SEARCH TRAINING")
    print("=" * 80)
    
    # Config path (use _this_dir so it works from Jupyter notebooks too)
    config_path = os.path.join(_this_dir, 'config.yaml')
    
    # Load base config to get paths
    print("\n📖 Loading configuration...")
    base_config = Config(config_path)
    
    # Get processed data directory
    processed_data_dir = base_config.paths.get('processed_data_dir', os.path.join(_this_dir, 'processed_data'))
    
    # Verify processed data files exist and are valid for all n_steps and n_channels combinations
    print("📂 Checking processed data files...")
    n_steps_values = HYPERPARAMETER_GRID['n_steps'].copy()
    n_channels_values = HYPERPARAMETER_GRID['n_channels'].copy()
    missing_files = []
    corrupted_files = []
    valid_combinations = []
    
    for n_steps in n_steps_values:
        for n_channels in n_channels_values:
            data_file = find_processed_data_file(n_steps, n_channels, processed_data_dir)
            if data_file is None:
                missing_files.append((n_steps, n_channels))
                print(f"   ⚠️  No data file found for n_steps={n_steps}, n_channels={n_channels}")
            else:
                # Validate the file can be opened
                is_valid, error_msg = validate_processed_data_file(data_file)
                if is_valid:
                    valid_combinations.append((n_steps, n_channels))
                    print(f"   ✅ Found valid data file for n_steps={n_steps}, n_channels={n_channels}: {os.path.basename(data_file)}")
                else:
                    corrupted_files.append((n_steps, n_channels, data_file, error_msg))
                    print(f"   ❌ Corrupted data file for n_steps={n_steps}, n_channels={n_channels}: {os.path.basename(data_file)}")
                    print(f"      Error: {error_msg}")
    
    # Filter out invalid combinations from the grid
    if missing_files or corrupted_files:
        print(f"\n⚠️  Filtering invalid combinations from grid:")
        if missing_files:
            print(f"   Missing files: {missing_files}")
        if corrupted_files:
            print(f"   Corrupted files: {[(n, ch) for n, ch, _, _ in corrupted_files]}")
        
        # Filter combinations to only include valid ones
        valid_n_steps = list(set([n for n, _ in valid_combinations]))
        valid_n_channels = list(set([ch for _, ch in valid_combinations]))
        
        if not valid_combinations:
            print(f"\n❌ ERROR: No valid processed data files found!")
            print(f"   Please preprocess data for at least one n_steps/n_channels combination.")
            print(f"   Expected directory: {processed_data_dir}")
            return
        
        # Update the grid to only include valid values
        HYPERPARAMETER_GRID['n_steps'] = valid_n_steps
        HYPERPARAMETER_GRID['n_channels'] = valid_n_channels
        
        print(f"   ✅ Continuing with valid n_steps: {valid_n_steps}")
        print(f"   ✅ Continuing with valid n_channels: {valid_n_channels}")
    else:
        print("✅ All required data files found and validated")
    
    # Generate hyperparameter combinations
    print("\n🔢 Generating hyperparameter combinations...")
    combinations = generate_hyperparameter_combinations()
    total_runs = len(combinations)
    print(f"   Total combinations: {total_runs}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    failed_runs = []
    
    # Run grid search
    print("\n🚀 Starting grid search...")
    print("=" * 80)
    
    # Get default learning_rate and epochs from config for display
    default_lr = base_config.training['learning_rate']
    default_epochs = base_config.training.get('epoch', None)
    
    print(f"\n📋 Using fixed hyperparameters:")
    print(f"   Learning rate: {default_lr:.0e} (FIXED - from config, will not vary)")
    print(f"\n🔄 Varying hyperparameters:")
    print(f"   Batch sizes: {HYPERPARAMETER_GRID.get('batch_size', [])}")
    print(f"   Latent dimensions: {HYPERPARAMETER_GRID.get('latent_dim', [])}")
    print(f"   N-steps: {HYPERPARAMETER_GRID.get('n_steps', [])}")
    print(f"   N-channels: {HYPERPARAMETER_GRID.get('n_channels', [])}")
    print(f"   LR Scheduler types: {HYPERPARAMETER_GRID.get('lr_scheduler_type', [])}")
    print(f"   Residual blocks: {HYPERPARAMETER_GRID.get('residual_blocks', [])}")
    print(f"   Normalization type: {HYPERPARAMETER_GRID.get('norm_type', [])}")
    print(f"   Encoder hidden dims: {HYPERPARAMETER_GRID.get('encoder_hidden_dims', [])}")
    print(f"   Dynamic loss weighting enable: {HYPERPARAMETER_GRID.get('dynamic_loss_weighting_enable', [])}")
    print(f"   Dynamic loss weighting methods: {HYPERPARAMETER_GRID.get('dynamic_loss_weighting_method', [])}")
    print(f"   Adversarial training enable: {HYPERPARAMETER_GRID.get('adversarial_enable', [])}")
    print(f"   Multi-embedding multimodal enable: {HYPERPARAMETER_GRID.get('enable_multi_embedding_multimodal', [])}")
    print(f"   Multi-embedding multimodal presets: {HYPERPARAMETER_GRID.get('mem_branches_preset', [])}")
    print(f"   Transition type: {HYPERPARAMETER_GRID.get('transition_type', ['linear'])}")
    print(f"   Channel names mapping: {CHANNEL_NAMES_MAP}")
    print("=" * 80)
    
    for idx, hyperparams in enumerate(combinations, 1):
        print(f"\n[{idx}/{total_runs}] Running: {create_run_name(hyperparams, idx, epochs=default_epochs, learning_rate=default_lr)}")
        print(f"   Batch size: {hyperparams['batch_size']}, "
              f"Latent dim: {hyperparams['latent_dim']}, "
              f"N-steps: {hyperparams['n_steps']}, "
              f"N-channels: {hyperparams['n_channels']}, "
              f"LR: {default_lr:.0e} (fixed), "
              f"Scheduler: {hyperparams.get('lr_scheduler_type', 'fixed')}")
        
        # Print additional parameters if present
        if 'residual_blocks' in hyperparams:
            print(f"   Residual blocks: {hyperparams['residual_blocks']}")
        if 'norm_type' in hyperparams:
            print(f"   Normalization type: {hyperparams['norm_type']}")
        if 'encoder_hidden_dims' in hyperparams:
            print(f"   Encoder hidden dims: {hyperparams['encoder_hidden_dims']}")
        if hyperparams.get('dynamic_loss_weighting_enable', False):
            print(f"   Dynamic loss weighting: {hyperparams.get('dynamic_loss_weighting_method', 'N/A')}")
        if hyperparams.get('adversarial_enable', False):
            print(f"   Adversarial training: enabled")
        
        result = run_single_training(config_path, hyperparams, idx, OUTPUT_DIR, processed_data_dir)
        
        if result:
            all_results.append(result)
            if result['status'] == 'success':
                print(f"   ✅ Completed - Best loss: {result.get('best_loss', 'N/A'):.6f}")
            else:
                failed_runs.append(result)
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Save intermediate results every 10 runs
        if idx % 10 == 0:
            save_results(all_results, failed_runs, idx, total_runs)
    
    # Save final results
    print("\n💾 Saving final results...")
    save_results(all_results, failed_runs, total_runs, total_runs)
    
    # Print summary
    print_summary(all_results, failed_runs, total_runs)


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj


def save_results(all_results: List[Dict], failed_runs: List[Dict], 
                 current_run: int, total_runs: int):
    """Save results to JSON and CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types to Python native types
    all_results_clean = numpy_to_python(all_results)
    failed_runs_clean = numpy_to_python(failed_runs)
    
    # Save JSON
    json_file = os.path.join(SUMMARY_DIR, f'grid_search_results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump({
            'total_runs': total_runs,
            'completed_runs': current_run,
            'successful_runs': len([r for r in all_results_clean if r.get('status') == 'success']),
            'failed_runs': len(failed_runs_clean),
            'results': all_results_clean,
            'failed': failed_runs_clean
        }, f, indent=2)
    
    # Save CSV
    if all_results_clean:
        csv_file = os.path.join(SUMMARY_DIR, f'grid_search_results_{timestamp}.csv')
        # Extended fieldnames to include new parameters
        fieldnames = [
            'run_id', 'run_name', 'status', 
            'batch_size', 'latent_dim', 'n_steps', 'n_channels',
            'lr_scheduler_type', 'lr_scheduler_params',
            'residual_blocks', 'norm_type', 'encoder_hidden_dims',
            'transition_type',
            'dynamic_loss_weighting_enable', 'dynamic_loss_weighting_method', 'dynamic_loss_weighting_params',
            'adversarial_enable', 'adversarial_params',
            'enable_multimodal', 'enable_gnn', 'enable_fno',
            'enable_multi_embedding_multimodal', 'mem_branches_preset',
            'learning_rate', 'data_file', 
            'best_loss', 'best_observation_loss', 'best_reconstruction_loss',
            'final_loss', 'final_observation_loss', 'final_reconstruction_loss', 'final_transition_loss',
            'encoder', 'decoder', 'transition'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results_clean:
                row = {}
                for k in fieldnames:
                    value = result.get(k, '')
                    # Convert lists and dicts to strings for CSV
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value) if value else ''
                    row[k] = value
                writer.writerow(row)
        
        print(f"   💾 Results saved: {json_file}, {csv_file}")


def print_summary(all_results: List[Dict], failed_runs: List[Dict], total_runs: int):
    """Print summary of grid search results"""
    successful = [r for r in all_results if r.get('status') == 'success']
    
    print("\n" + "=" * 80)
    print("📊 GRID SEARCH SUMMARY")
    print("=" * 80)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed_runs)}")
    
    if successful:
        # Find best models
        best_total = min(successful, key=lambda x: x.get('best_loss', float('inf')))
        best_obs = min(successful, key=lambda x: x.get('best_observation_loss', float('inf')))
        best_recon = min(successful, key=lambda x: x.get('best_reconstruction_loss', float('inf')))
        
        print("\n🏆 BEST MODELS:")
        print(f"   Best Total Loss: {best_total['run_name']} - Loss: {best_total['best_loss']:.6f}")
        print(f"   Best Observation Loss: {best_obs['run_name']} - Loss: {best_obs['best_observation_loss']:.6f}")
        print(f"   Best Reconstruction Loss: {best_recon['run_name']} - Loss: {best_recon['best_reconstruction_loss']:.6f}")
        
        # Print hyperparameter ranges used
        print("\n📋 HYPERPARAMETER RANGES:")
        print(f"   Batch sizes: {HYPERPARAMETER_GRID.get('batch_size', [])}")
        print(f"   Latent dimensions: {HYPERPARAMETER_GRID.get('latent_dim', [])}")
        print(f"   N-steps: {HYPERPARAMETER_GRID.get('n_steps', [])}")
        print(f"   N-channels: {HYPERPARAMETER_GRID.get('n_channels', [])}")
        print(f"   LR Schedulers: {HYPERPARAMETER_GRID.get('lr_scheduler_type', [])}")
        print(f"   Residual blocks: {HYPERPARAMETER_GRID.get('residual_blocks', [])}")
        print(f"   Normalization type: {HYPERPARAMETER_GRID.get('norm_type', [])}")
        print(f"   Encoder hidden dims: {HYPERPARAMETER_GRID.get('encoder_hidden_dims', [])}")
        print(f"   Dynamic loss weighting: {HYPERPARAMETER_GRID.get('dynamic_loss_weighting_enable', [])}")
        print(f"   Adversarial training: {HYPERPARAMETER_GRID.get('adversarial_enable', [])}")
        print(f"   Transition type: {HYPERPARAMETER_GRID.get('transition_type', ['linear'])}")
        
        print("\n📁 Model files saved to:", OUTPUT_DIR)
        print("📊 Results saved to:", SUMMARY_DIR)
    
    if failed_runs:
        print(f"\n❌ Failed Runs ({len(failed_runs)}):")
        for failed in failed_runs[:5]:  # Show first 5
            print(f"   {failed['run_id']}: {failed.get('error', 'Unknown')}")
        if len(failed_runs) > 5:
            print(f"   ... and {len(failed_runs) - 5} more")
    
    print("=" * 80)


# ============================================================================
# VALIDATION AND TESTING FUNCTIONS
# ============================================================================

def validate_setup(config_path: str = None, test_single_run: bool = False) -> bool:
    """
    Validate that everything is set up correctly before running grid search.
    
    Args:
        config_path: Path to config file
        test_single_run: If True, test a single training run (1 epoch, small batch)
        
    Returns:
        True if validation passes, False otherwise
    """
    if config_path is None:
        config_path = os.path.join(_this_dir, 'config.yaml')
    
    print("=" * 80)
    print("🔍 VALIDATING GRID SEARCH SETUP")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # 1. Test imports
    print("\n1️⃣ Testing imports...")
    try:
        import torch
        print("   ✅ torch")
    except ImportError as e:
        errors.append(f"torch import failed: {e}")
        print(f"   ❌ torch: {e}")
    
    try:
        from utilities.config_loader import Config
        print("   ✅ Config")
    except ImportError as e:
        errors.append(f"Config import failed: {e}")
        print(f"   ❌ Config: {e}")
    
    try:
        from utilities.timing import Timer, collect_training_metadata
        print("   ✅ Timer, collect_training_metadata")
    except ImportError as e:
        errors.append(f"Timing imports failed: {e}")
        print(f"   ❌ Timing: {e}")
    
    try:
        from utilities.wandb_integration import create_wandb_logger
        print("   ✅ WandB logger")
    except ImportError as e:
        warnings.append(f"WandB import failed (optional): {e}")
        print(f"   ⚠️ WandB logger: {e} (optional)")
    
    try:
        from data_preprocessing import load_processed_data
        print("   ✅ load_processed_data")
    except ImportError as e:
        errors.append(f"Data preprocessing import failed: {e}")
        print(f"   ❌ load_processed_data: {e}")
    
    try:
        from model.training.rom_wrapper import ROMWithE2C
        print("   ✅ ROMWithE2C")
    except ImportError as e:
        errors.append(f"ROMWithE2C import failed: {e}")
        print(f"   ❌ ROMWithE2C: {e}")
    
    # 2. Test config loading
    print("\n2️⃣ Testing config loading...")
    try:
        config = Config(config_path)
        print(f"   ✅ Config loaded from: {config_path}")
        
        # Check required fields
        required_fields = [
            ('training', 'learning_rate'),
            ('training', 'batch_size'),
            ('training', 'epoch'),
            ('training', 'nsteps'),
            ('model', 'latent_dim'),
            ('model', 'n_channels'),
            ('learning_rate_scheduler', 'type'),
        ]
        
        for section, field in required_fields:
            if section not in config.config:
                errors.append(f"Missing config section: {section}")
                print(f"   ❌ Missing section: {section}")
            elif field not in config.config[section]:
                errors.append(f"Missing config field: {section}.{field}")
                print(f"   ❌ Missing field: {section}.{field}")
            else:
                print(f"   ✅ {section}.{field}: {config.config[section][field]}")
                
    except Exception as e:
        errors.append(f"Config loading failed: {e}")
        print(f"   ❌ Config loading failed: {e}")
        return False
    
    # 3. Test data loading
    print("\n3️⃣ Testing data loading...")
    try:
        processed_data_dir = config.paths.get('processed_data_dir', os.path.join(_this_dir, 'processed_data'))
        
        # Check that data files exist for all n_steps and n_channels combinations in grid
        n_steps_values = HYPERPARAMETER_GRID.get('n_steps', [])
        n_channels_values = HYPERPARAMETER_GRID.get('n_channels', [])
        if n_steps_values and n_channels_values:
            print(f"   Checking data files for n_steps: {n_steps_values}, n_channels: {n_channels_values}")
            missing_files = []
            for n_steps in n_steps_values:
                for n_channels in n_channels_values:
                    data_file = find_processed_data_file(n_steps, n_channels, processed_data_dir)
                    if data_file is None:
                        missing_files.append((n_steps, n_channels))
                        print(f"   ❌ No data file found for n_steps={n_steps}, n_channels={n_channels}")
                    else:
                        print(f"   ✅ Found data file for n_steps={n_steps}, n_channels={n_channels}: {os.path.basename(data_file)}")
            
            if missing_files:
                errors.append(f"Missing processed data files for combinations: {missing_files}")
            
            # Test loading one file
            if not missing_files:
                test_n_steps = n_steps_values[0]
                test_n_channels = n_channels_values[0]
                test_file = find_processed_data_file(test_n_steps, test_n_channels, processed_data_dir)
                loaded_data = load_processed_data(filepath=test_file, n_channels=test_n_channels)
                
                if loaded_data is None:
                    errors.append(f"Failed to load test data file for n_steps={test_n_steps}")
                    print(f"   ❌ Failed to load test data file")
                else:
                    print(f"   ✅ Successfully loaded test data file for n_steps={test_n_steps}")
                    
                    # Check required data keys
                    required_keys = ['STATE_train', 'BHP_train', 'Yobs_train', 
                                   'STATE_eval', 'BHP_eval', 'Yobs_eval', 
                                   'dt_train', 'dt_eval', 'metadata']
                    
                    for key in required_keys:
                        if key not in loaded_data:
                            errors.append(f"Missing data key: {key}")
                            print(f"   ❌ Missing data key: {key}")
                        else:
                            print(f"   ✅ {key}: {type(loaded_data[key])}")
        else:
            # Fallback to old behavior if n_steps not in grid
            loaded_data = load_processed_data(data_dir=processed_data_dir)
            if loaded_data is None:
                errors.append("No processed data found")
                print(f"   ❌ No processed data found in: {processed_data_dir}")
            else:
                print(f"   ✅ Data loaded from: {processed_data_dir}")
                    
    except Exception as e:
        errors.append(f"Data loading failed: {e}")
        print(f"   ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Test hyperparameter generation
    print("\n4️⃣ Testing hyperparameter generation...")
    try:
        combinations = generate_hyperparameter_combinations()
        total = len(combinations)
        print(f"   ✅ Generated {total} combinations")
        
        if total == 0:
            errors.append("No hyperparameter combinations generated")
            print("   ❌ No combinations generated!")
        else:
            # Test first combination
            first_combo = combinations[0]
            print(f"   ✅ First combination: {first_combo}")
            
            # Test run ID and name generation
            run_id = create_run_id(first_combo, 1)
            _ep = config.training.get('epoch', None)
            _lr = config.training.get('learning_rate', None)
            run_name = create_run_name(first_combo, 1, epochs=_ep, learning_rate=_lr)
            print(f"   ✅ Run ID: {run_id}")
            print(f"   ✅ Run name: {run_name}")
            
            # Test filename generation (use dummy values if no loaded_data)
            num_train = 1000
            num_well = 10
            if 'loaded_data' in locals() and loaded_data and 'metadata' in loaded_data:
                num_train = loaded_data['metadata'].get('num_train', 1000)
                num_well = loaded_data['metadata'].get('num_well', 10)
            filename = create_run_model_filename('encoder', first_combo, num_train, num_well, run_id)
            print(f"   ✅ Sample filename: {filename}")
                
    except Exception as e:
        errors.append(f"Hyperparameter generation failed: {e}")
        print(f"   ❌ Hyperparameter generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Test config update
    print("\n5️⃣ Testing config update...")
    try:
        test_hyperparams = {
            'batch_size': 16,
            'latent_dim': 64,
            'n_steps': 2,
            'n_channels': 2,
            'lr_scheduler_type': 'step_decay',
            'lr_scheduler_params': {'step_size': 50, 'gamma': 0.5},
            'residual_blocks': 3,
            'encoder_hidden_dims': [200, 200],
            'dynamic_loss_weighting_enable': False,
            'dynamic_loss_weighting_method': 'gradnorm',
            'adversarial_enable': False
        }
        
        updated_config = update_config_with_hyperparams(config_path, test_hyperparams)
        
        # Verify updates
        if updated_config.training['batch_size'] != test_hyperparams['batch_size']:
            errors.append("Batch size not updated correctly")
            print(f"   ❌ Batch size update failed")
        else:
            print(f"   ✅ Batch size updated: {updated_config.training['batch_size']}")
            
        if updated_config.model['latent_dim'] != test_hyperparams['latent_dim']:
            errors.append("Latent dim not updated correctly")
            print(f"   ❌ Latent dim update failed")
        else:
            print(f"   ✅ Latent dim updated: {updated_config.model['latent_dim']}")
        
        if updated_config.training['nsteps'] != test_hyperparams['n_steps']:
            errors.append("N-steps not updated correctly")
            print(f"   ❌ N-steps update failed")
        else:
            print(f"   ✅ N-steps updated: {updated_config.training['nsteps']}")
        
        # Verify new parameters
        if updated_config.encoder.get('residual_blocks') != test_hyperparams['residual_blocks']:
            errors.append("Residual blocks not updated correctly")
            print(f"   ❌ Residual blocks update failed")
        else:
            print(f"   ✅ Residual blocks updated: {updated_config.encoder.get('residual_blocks')}")
        
        if updated_config.transition.get('encoder_hidden_dims') != test_hyperparams['encoder_hidden_dims']:
            errors.append("Encoder hidden dims not updated correctly")
            print(f"   ❌ Encoder hidden dims update failed")
        else:
            print(f"   ✅ Encoder hidden dims updated: {updated_config.transition.get('encoder_hidden_dims')}")
        
        scheduler_type = updated_config.learning_rate_scheduler.get('type', 'constant')
        if scheduler_type != test_hyperparams['lr_scheduler_type']:
            errors.append("Scheduler type not updated correctly")
            print(f"   ❌ Scheduler type update failed")
        else:
            print(f"   ✅ Scheduler type updated: {scheduler_type}")
        
        # Verify learning_rate uses default
        print(f"   ✅ Learning rate (default): {updated_config.training['learning_rate']}")
            
    except Exception as e:
        errors.append(f"Config update failed: {e}")
        print(f"   ❌ Config update failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Test output directories
    print("\n6️⃣ Testing output directories...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"   ✅ Output directory: {OUTPUT_DIR}")
        
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        print(f"   ✅ Summary directory: {SUMMARY_DIR}")
        
        os.makedirs(TIMING_LOG_DIR, exist_ok=True)
        print(f"   ✅ Timing log directory: {TIMING_LOG_DIR}")
        
    except Exception as e:
        errors.append(f"Directory creation failed: {e}")
        print(f"   ❌ Directory creation failed: {e}")
    
    # 7. Test single run (if requested)
    if test_single_run and not errors:
        print("\n7️⃣ Testing single training run (1 epoch, reduced batch)...")
        try:
            # Use first n_steps value from grid, or default to 2
            test_n_steps = HYPERPARAMETER_GRID.get('n_steps', [2])[0]
            test_n_channels = HYPERPARAMETER_GRID.get('n_channels', [2])[0]
            test_hyperparams = {
                'batch_size': 8,  # Small batch for testing
                'latent_dim': 32,  # Small latent dim for testing
                'n_steps': test_n_steps,
                'n_channels': test_n_channels,
                'lr_scheduler_type': 'fixed',  # Use fixed scheduler for test
                'residual_blocks': 2,  # Small number for testing
                'encoder_hidden_dims': [100, 100],  # Small dimensions for testing
                'dynamic_loss_weighting_enable': False,  # Disable for test
                'dynamic_loss_weighting_method': 'gradnorm',
                'adversarial_enable': False  # Disable for test
            }
            
            # Temporarily reduce epochs for testing
            original_epochs = config.training['epoch']
            config.set('training.epoch', 1)
            
            print(f"   Running test with: {test_hyperparams}")
            processed_data_dir = config.paths.get('processed_data_dir', os.path.join(_this_dir, 'processed_data'))
            result = run_single_training(config_path, test_hyperparams, 9999, OUTPUT_DIR, processed_data_dir)
            
            # Restore original epochs
            config.set('training.epoch', original_epochs)
            
            if result and result.get('status') == 'success':
                print(f"   ✅ Test run successful!")
                print(f"      Best loss: {result.get('best_loss', 'N/A')}")
            else:
                errors.append("Test run failed")
                print(f"   ❌ Test run failed: {result.get('error', 'Unknown error') if result else 'No result'}")
                
        except Exception as e:
            errors.append(f"Test run failed: {e}")
            print(f"   ❌ Test run failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print(f"❌ Found {len(errors)} error(s):")
        for error in errors:
            print(f"   • {error}")
        print("\n⚠️ Please fix errors before running grid search!")
        return False
    else:
        print("✅ All validations passed!")
        
    if warnings:
        print(f"\n⚠️ Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   • {warning}")
        print("\n💡 Warnings are non-critical but should be reviewed.")
    
    print("\n🚀 Ready to run grid search!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search Training')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation checks before training')
    parser.add_argument('--test-run', action='store_true',
                       help='Test a single training run (requires --validate)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    # Use parse_known_args to ignore IPython/Jupyter arguments (like --f=...)
    # This works in both regular Python and IPython/Jupyter environments
    args, unknown = parser.parse_known_args()
    
    if args.validate or args.test_run:
        success = validate_setup(args.config, test_single_run=args.test_run)
        if not success:
            print("\n❌ Validation failed. Exiting.")
            sys.exit(1)
        
        if args.test_run:
            print("\n✅ Test run completed. You can now run full grid search.")
            sys.exit(0)
        
        if args.validate:
            print("\n✅ Validation passed. Run without --validate to start grid search.")
            sys.exit(0)
    
    # Run main grid search
    main()




#%%
# ============================================================================
# STEP 3: TESTING & VISUALIZATION DASHBOARD
# ============================================================================
# Import and create testing dashboard
import os
os.chdir(r'c:\Users\taha.yehia\Desktop\ROM-Optimization\ROM_Refactored')
print(os.getcwd())

from testing import create_testing_dashboard

# Create and display the testing dashboard
testing_dashboard = create_testing_dashboard()

# %%
