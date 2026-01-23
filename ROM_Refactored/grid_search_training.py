#%%!/usr/bin/env python3
"""
Hyperparameter Grid Search Training Script
==========================================
Runs multiple training runs with different hyperparameter combinations.
Saves best model from each run with descriptive filenames.
Integrates wandb logging and timing logs.

Usage:
    python grid_search_training.py
"""

import os
import sys
import json
import csv
import itertools
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
# learning_rate and lr_scheduler use default values from config
HYPERPARAMETER_GRID = {
    'batch_size': [8,16,32,64,128,256],
    'latent_dim': [16,32,64,128,256],
    'n_steps': [2],  # Available processed data files
    'n_channels': [2,4]  # Number of channels (2 for SW/SG, 4 for SW/SG/PRES/PERMI, etc.)
    # learning_rate and lr_scheduler will use config defaults
}

# Channel names mapping based on n_channels
# Maps number of channels to list of channel names in order
CHANNEL_NAMES_MAP = {
    2: ['SG', 'PRES'],
    4: ['SG', 'PRES','POROS', 'PERMI']
    # Add more mappings as needed
}

# Output directories
OUTPUT_DIR = './saved_models/grid_search/'
SUMMARY_DIR = './grid_search_results/'
TIMING_LOG_DIR = './timing_logs/'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_processed_data_file(n_steps: int, n_channels: int, data_dir: str = './processed_data/') -> Optional[str]:
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
    
    # Resolve relative path
    if not os.path.isabs(data_dir):
        if not os.path.exists(data_dir):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
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


def generate_hyperparameter_combinations() -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters.
    
    Returns:
        List of dictionaries, each containing a hyperparameter set
    """
    keys = HYPERPARAMETER_GRID.keys()
    values = HYPERPARAMETER_GRID.values()
    
    combinations = []
    for combination in itertools.product(*values):
        hyperparams = dict(zip(keys, combination))
        combinations.append(hyperparams)
    
    return combinations


def create_run_id(hyperparams: Dict[str, Any], run_index: int) -> str:
    """
    Create a unique run ID based on hyperparameters.
    
    Args:
        hyperparams: Dictionary of hyperparameters (batch_size, latent_dim, n_steps, n_channels)
        run_index: Index of the run
        
    Returns:
        String run ID
    """
    return (f"run{run_index:04d}_bs{hyperparams['batch_size']}_"
            f"ld{hyperparams['latent_dim']}_ns{hyperparams['n_steps']}_ch{hyperparams['n_channels']}")


def create_run_name(hyperparams: Dict[str, Any], run_index: int) -> str:
    """
    Create a human-readable run name for wandb.
    
    Args:
        hyperparams: Dictionary of hyperparameters (batch_size, latent_dim, n_steps, n_channels)
        run_index: Index of the run
        
    Returns:
        String run name
    """
    return (f"bs{hyperparams['batch_size']}_ld{hyperparams['latent_dim']}_"
            f"ns{hyperparams['n_steps']}_ch{hyperparams['n_channels']}")


def create_run_model_filename(component: str, hyperparams: Dict[str, Any], 
                              num_train: int, num_well: int, run_id: str) -> str:
    """
    Create model filename for a specific run.
    
    Args:
        component: Model component ('encoder', 'decoder', 'transition')
        hyperparams: Dictionary of hyperparameters (batch_size, latent_dim, n_steps, n_channels)
        num_train: Number of training samples
        num_well: Number of wells
        run_id: Unique run ID
        
    Returns:
        Formatted filename string
    """
    return (f"e2co_{component}_grid_"
            f"bs{hyperparams['batch_size']}_"
            f"ld{hyperparams['latent_dim']}_"
            f"ns{hyperparams['n_steps']}_"
            f"ch{hyperparams['n_channels']}_"
            f"{run_id}_"
            f"bs{hyperparams['batch_size']}_"
            f"ld{hyperparams['latent_dim']}_"
            f"ns{hyperparams['n_steps']}_"
            f"ch{hyperparams['n_channels']}.h5")


def update_config_with_hyperparams(config_path: str, hyperparams: Dict[str, Any]) -> Config:
    """
    Create a new config with hyperparameters applied.
    
    Args:
        config_path: Path to config file
        hyperparams: Dictionary of hyperparameters to apply (batch_size, latent_dim, n_steps, n_channels)
        
    Returns:
        New Config object with hyperparameters applied
    """
    # Load fresh config for each run (this gets default learning_rate and lr_scheduler)
    config = Config(config_path)
    
    # Validate that learning_rate is fixed (not in hyperparams)
    if 'learning_rate' in hyperparams:
        raise ValueError("learning_rate should not be in hyperparams - it must remain fixed")
    
    # Update training hyperparameters
    # Update batch_size and n_steps (learning_rate uses config default and is FIXED)
    config.set('training.batch_size', hyperparams['batch_size'])
    config.set('training.nsteps', hyperparams['n_steps'])
    
    # Ensure learning rate is fixed (read from config, do not modify)
    original_lr = config.training['learning_rate']
    
    # Update model hyperparameters
    config.set('model.latent_dim', hyperparams['latent_dim'])
    
    # Update n_channels related config (matching dashboard logic)
    n_channels = hyperparams['n_channels']
    # Update model.n_channels
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
                # Update first element which is input channels
                conv1[0] = n_channels
    
    # Update decoder final_conv output channels if exists (matching dashboard logic)
    if 'decoder' in config.config and 'deconv_layers' in config.config['decoder']:
        if 'final_conv' in config.config['decoder']['deconv_layers']:
            final_conv_config = config.config['decoder']['deconv_layers']['final_conv']
            if isinstance(final_conv_config, list) and len(final_conv_config) > 0:
                # Check if second element is null (which means n_channels) - keep as None
                if len(final_conv_config) > 1 and final_conv_config[1] is None:
                    # Keep as None, it will be auto-filled
                    pass
                elif len(final_conv_config) > 1:
                    # Update output channels to n_channels
                    final_conv_config[1] = n_channels
    
    # Validate scheduler is not step_decay (ensure fixed learning rate)
    scheduler_type = config.learning_rate_scheduler.get('type', 'step_decay')
    if scheduler_type == 'step_decay':
        # Override to constant scheduler to ensure fixed learning rate
        config.set('learning_rate_scheduler.type', 'constant')
        scheduler_type = 'constant'
    
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
        for ib in range(num_batch):
            ind0 = ib * config.training['batch_size']
            
            X_batch = [state[ind0:ind0+config.training['batch_size'], ...] for state in STATE_train]
            U_batch = [bhp[ind0:ind0+config.training['batch_size'], ...] for bhp in BHP_train]
            Y_batch = [yobs[ind0:ind0+config.training['batch_size'], ...] for yobs in Yobs_train]
            dt_batch = dt_train[ind0:ind0+config.training['batch_size'], ...]
            
            inputs = (X_batch, U_batch, Y_batch, dt_batch)
            my_rom.update(inputs)
            
            global_step += 1
            wandb_logger.log_training_step(my_rom, e+1, ib+1, global_step)
            
            if ib % config.runtime.get('print_interval', 10) == 0:
                # Evaluate
                X_batch_eval = [state for state in STATE_eval]
                U_batch_eval = [bhp for bhp in BHP_eval]
                Y_batch_eval = [yobs for yobs in Yobs_eval]
                test_inputs = (X_batch_eval, U_batch_eval, Y_batch_eval, dt_eval)
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
                       output_dir: str, processed_data_dir: str = './processed_data/') -> Optional[Dict[str, Any]]:
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
            **hyperparams,
            'channel_names': get_channel_names(n_channels),
            'learning_rate': config.training['learning_rate'],  # From config default (FIXED)
            'lr_scheduler': config.learning_rate_scheduler.get('type', 'constant'),  # Overridden to constant if was step_decay
            'data_file': os.path.basename(data_filepath),
            **results,
            **model_paths
        }
        
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
    
    # Config path
    config_path = 'config.yaml'
    
    # Load base config to get paths
    print("\n📖 Loading configuration...")
    base_config = Config(config_path)
    
    # Get processed data directory
    processed_data_dir = base_config.paths.get('processed_data_dir', './processed_data/')
    
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
    
    # Get default learning_rate and lr_scheduler from config for display
    default_lr = base_config.training['learning_rate']
    default_scheduler = base_config.learning_rate_scheduler.get('type', 'step_decay')
    
    # Ensure scheduler is not step_decay (will be overridden to constant in update_config_with_hyperparams)
    if default_scheduler == 'step_decay':
        print(f"\n⚠️  Note: Config has step_decay scheduler, but will be overridden to 'constant' to ensure fixed learning rate")
        default_scheduler = 'constant'
    
    print(f"\n📋 Using fixed hyperparameters:")
    print(f"   Learning rate: {default_lr:.0e} (FIXED - from config, will not vary)")
    print(f"   LR scheduler: {default_scheduler} (FIXED - ensures constant learning rate)")
    print(f"\n🔄 Varying hyperparameters:")
    print(f"   Batch sizes: {HYPERPARAMETER_GRID['batch_size']}")
    print(f"   Latent dimensions: {HYPERPARAMETER_GRID['latent_dim']}")
    print(f"   N-steps: {HYPERPARAMETER_GRID['n_steps']}")
    print(f"   N-channels: {HYPERPARAMETER_GRID['n_channels']}")
    print(f"   Channel names mapping: {CHANNEL_NAMES_MAP}")
    print("=" * 80)
    
    for idx, hyperparams in enumerate(combinations, 1):
        print(f"\n[{idx}/{total_runs}] Running: {create_run_name(hyperparams, idx)}")
        print(f"   Batch size: {hyperparams['batch_size']}, "
              f"Latent dim: {hyperparams['latent_dim']}, "
              f"N-steps: {hyperparams['n_steps']}, "
              f"LR: {default_lr:.0e} (default), "
              f"Scheduler: {default_scheduler} (default)")
        
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


def save_results(all_results: List[Dict], failed_runs: List[Dict], 
                 current_run: int, total_runs: int):
    """Save results to JSON and CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = os.path.join(SUMMARY_DIR, f'grid_search_results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump({
            'total_runs': total_runs,
            'completed_runs': current_run,
            'successful_runs': len([r for r in all_results if r.get('status') == 'success']),
            'failed_runs': len(failed_runs),
            'results': all_results,
            'failed': failed_runs
        }, f, indent=2)
    
    # Save CSV
    if all_results:
        csv_file = os.path.join(SUMMARY_DIR, f'grid_search_results_{timestamp}.csv')
        fieldnames = ['run_id', 'run_name', 'status', 'batch_size', 'latent_dim', 'n_steps',
                     'learning_rate', 'lr_scheduler', 'data_file', 'best_loss', 'best_observation_loss',
                     'best_reconstruction_loss', 'final_loss', 'final_observation_loss',
                     'final_reconstruction_loss', 'final_transition_loss', 'encoder', 'decoder', 'transition']
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                row = {k: result.get(k, '') for k in fieldnames}
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

def validate_setup(config_path: str = 'config.yaml', test_single_run: bool = False) -> bool:
    """
    Validate that everything is set up correctly before running grid search.
    
    Args:
        config_path: Path to config file
        test_single_run: If True, test a single training run (1 epoch, small batch)
        
    Returns:
        True if validation passes, False otherwise
    """
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
        processed_data_dir = config.paths.get('processed_data_dir', './processed_data/')
        
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
            run_name = create_run_name(first_combo, 1)
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
            'n_steps': 2
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
        
        # Verify learning_rate and scheduler use defaults
        print(f"   ✅ Learning rate (default): {updated_config.training['learning_rate']}")
        # Get scheduler type (will be 'constant' if was 'step_decay')
        scheduler_type = updated_config.learning_rate_scheduler.get('type', 'constant')
        print(f"   ✅ Scheduler: {scheduler_type} (overridden to constant if was step_decay)")
            
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
            test_hyperparams = {
                'batch_size': 8,  # Small batch for testing
                'latent_dim': 32,  # Small latent dim for testing
                'n_steps': test_n_steps
            }
            
            # Temporarily reduce epochs for testing
            original_epochs = config.training['epoch']
            config.set('training.epoch', 1)
            
            print(f"   Running test with: {test_hyperparams}")
            processed_data_dir = config.paths.get('processed_data_dir', './processed_data/')
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


# %%
