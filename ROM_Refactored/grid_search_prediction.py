#!/usr/bin/env python3
"""
Grid Search Batch Metrics Calculation Script
============================================
Processes all saved grid search models, generates predictions for both training 
and testing cases, and calculates overall performance metrics using dashboard logic.

Usage:
    python grid_search_batch_metrics.py
"""
#%%
import os
import sys
import json
import glob
import re
import gc
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utilities.config_loader import Config
from utilities.timing import Timer
from data_preprocessing import load_processed_data
from model.training.rom_wrapper import ROMWithE2C
from testing.dashboard import TestingDashboard
from testing.prediction.predictor import generate_test_visualization_standalone
from testing.visualization.dashboard import InteractiveVisualizationDashboard

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration options (similar to grid_search_training.py)
SPATIAL_FIELDS = None  # None = all fields, or list of field names to calculate
TIMESERIES_GROUPS = None  # None = all groups, or list of group names to calculate
SPATIAL_LAYER = None  # None = all layers, or int layer index (0 to Nz-1) to calculate
TIMESERIES_WELL = None  # None = all wells in group, or well name (e.g., 'BHP1', 'Gas Prod1') to calculate
NUM_TSTEP = 30  # Number of timesteps for prediction (default from dashboard)
METRICS_CALCULATION_MODE = 'Averaged'  # 'Aggregated' or 'Averaged' - how to calculate metrics

# Output directories
OUTPUT_DIR = './timing_logs/'
MODELS_DIR = './saved_models/'
DATA_DIR = 'sr3_batch_output/'  # Directory containing raw data files (batch_spatial_properties_*.h5, batch_timeseries_data_*.h5)
PROCESSED_DATA_DIR = './processed_data/'  # Directory containing processed data files
CONFIG_PATH = 'config.yaml'

# Timeseries groups mapping (includes well names for individual well selection)
TIMESERIES_GROUPS_MAP = {
    'BHP (All Injectors)': (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'),
    'Gas Production (All Producers)': (list(range(3, 6)), ['Gas Prod1', 'Gas Prod2', 'Gas Prod3'], 'ft3/day'),
    'Water Production (All Producers)': (list(range(6, 9)), ['Water Prod1', 'Water Prod2', 'Water Prod3'], 'ft3/day')
}


# ============================================================================
# MODEL DISCOVERY
# ============================================================================

def _parse_model_filename_with_channels(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse model filename to extract hyperparameters including n_channels.
    Supports both old format (without ch) and new format (with ch).
    
    Args:
        filename: Model filename (e.g., 'e2co_encoder_grid_bs32_ld64_ns2_ch4_run0001_bs32_ld64_ns2_ch4.h5')
        
    Returns:
        Dict with batch_size, latent_dim, n_steps, n_channels, run_id, component, or None if parsing fails
    """
    try:
        # New format with n_channels: e2co_{component}_grid_bs{bs}_ld{ld}_ns{ns}_ch{ch}_run{id}_bs{bs}_ld{ld}_ns{ns}_ch{ch}.h5
        pattern_with_ch = r'e2co_(encoder|decoder|transition)_grid_bs(\d+)_ld(\d+)_ns(\d+)_ch(\d+)_run(\d+)_bs\d+_ld\d+_ns\d+_ch\d+\.h5'
        match = re.match(pattern_with_ch, filename)
        
        if match:
            component = match.group(1)
            batch_size = int(match.group(2))
            latent_dim = int(match.group(3))
            n_steps = int(match.group(4))
            n_channels = int(match.group(5))
            run_id = match.group(6)
            
            return {
                'component': component,
                'batch_size': batch_size,
                'latent_dim': latent_dim,
                'n_steps': n_steps,
                'n_channels': n_channels,
                'run_id': run_id
            }
        
        # Old format without n_channels: e2co_{component}_grid_bs{bs}_ld{ld}_ns{ns}_run{id}_bs{bs}_ld{ld}_ns{ns}.h5
        # Default to n_channels=2 for backward compatibility
        pattern_old = r'e2co_(encoder|decoder|transition)_grid_bs(\d+)_ld(\d+)_ns(\d+)_run(\d+)_bs\d+_ld\d+_ns\d+\.h5'
        match = re.match(pattern_old, filename)
        
        if match:
            component = match.group(1)
            batch_size = int(match.group(2))
            latent_dim = int(match.group(3))
            n_steps = int(match.group(4))
            run_id = match.group(5)
            
            return {
                'component': component,
                'batch_size': batch_size,
                'latent_dim': latent_dim,
                'n_steps': n_steps,
                'n_channels': 2,  # Default for old format
                'run_id': run_id
            }
        
        return None
    except Exception as e:
        return None


def discover_models_with_channels(model_dir: str = './saved_models/') -> List[Dict[str, Any]]:
    """
    Discover all available models with n_channels support.
    Groups models by (run_id, batch_size, latent_dim, n_steps, n_channels).
    
    Args:
        model_dir: Directory to scan for model files
        
    Returns:
        List of complete model sets with n_channels included
    """
    print("🔍 Discovering models with n_channels support...")
    
    if not os.path.exists(model_dir):
        print(f"⚠️ Model directory not found: {model_dir}")
        return []
    
    # Find all model files matching pattern
    encoder_files = glob.glob(os.path.join(model_dir, 'e2co_encoder_grid_*.h5'))
    decoder_files = glob.glob(os.path.join(model_dir, 'e2co_decoder_grid_*.h5'))
    transition_files = glob.glob(os.path.join(model_dir, 'e2co_transition_grid_*.h5'))
    
    # Also check grid_search subdirectory
    grid_search_dir = os.path.join(model_dir, 'grid_search')
    if os.path.exists(grid_search_dir):
        encoder_files.extend(glob.glob(os.path.join(grid_search_dir, 'e2co_encoder_grid_*.h5')))
        decoder_files.extend(glob.glob(os.path.join(grid_search_dir, 'e2co_decoder_grid_*.h5')))
        transition_files.extend(glob.glob(os.path.join(grid_search_dir, 'e2co_transition_grid_*.h5')))
    
    # Group models by composite key: (run_id, batch_size, latent_dim, n_steps, n_channels)
    # This ensures models with same hyperparameters but different n_channels are treated separately
    model_sets = {}
    
    for encoder_file in encoder_files:
        filename = os.path.basename(encoder_file)
        parsed = _parse_model_filename_with_channels(filename)
        if parsed:
            # Use composite key including n_channels
            model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'], parsed['n_channels'])
            if model_key not in model_sets:
                model_sets[model_key] = {
                    'run_id': parsed['run_id'],
                    'batch_size': parsed['batch_size'],
                    'latent_dim': parsed['latent_dim'],
                    'n_steps': parsed['n_steps'],
                    'n_channels': parsed['n_channels'],
                    'encoder': None,
                    'decoder': None,
                    'transition': None
                }
            model_sets[model_key]['encoder'] = encoder_file
    
    for decoder_file in decoder_files:
        filename = os.path.basename(decoder_file)
        parsed = _parse_model_filename_with_channels(filename)
        if parsed:
            model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'], parsed['n_channels'])
            if model_key in model_sets:
                model_sets[model_key]['decoder'] = decoder_file
    
    for transition_file in transition_files:
        filename = os.path.basename(transition_file)
        parsed = _parse_model_filename_with_channels(filename)
        if parsed:
            model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'], parsed['n_channels'])
            if model_key in model_sets:
                model_sets[model_key]['transition'] = transition_file
    
    # Filter to only complete sets (all three components)
    complete_sets = []
    for model_key, model_set in model_sets.items():
        if model_set['encoder'] and model_set['decoder'] and model_set['transition']:
            complete_sets.append(model_set)
    
    # Sort by run_id, then batch_size, latent_dim, n_steps, n_channels
    complete_sets.sort(key=lambda x: (x['run_id'], x['batch_size'], x['latent_dim'], x['n_steps'], x['n_channels']))
    
    if not complete_sets:
        print(f"⚠️ No complete model sets found in {model_dir}")
        return []
    
    print(f"✅ Found {len(complete_sets)} complete model set(s)")
    for model_set in complete_sets:
        print(f"   Run {model_set['run_id']}: bs={model_set['batch_size']}, "
              f"ld={model_set['latent_dim']}, ns={model_set['n_steps']}, ch={model_set['n_channels']}")
    
    return complete_sets


def discover_models(model_dir: str = './saved_models/') -> List[Dict[str, Any]]:
    """
    Discover all available models using dashboard's scanning logic.
    DEPRECATED: Use discover_models_with_channels() instead.
    
    Args:
        model_dir: Directory to scan for model files
        
    Returns:
        List of complete model sets (same as dashboard dropdown)
    """
    return discover_models_with_channels(model_dir)


# ============================================================================
# DATA FILE MATCHING
# ============================================================================

def find_matching_data_file(model_n_steps: int, model_n_channels: int, data_dir: str) -> Optional[str]:
    """
    Find processed data file matching model's n_steps and n_channels.
    
    Args:
        model_n_steps: n_steps from model hyperparameters
        model_n_channels: n_channels extracted from model weights
        data_dir: Directory containing processed data files
        
    Returns:
        Path to matching data file (normalized absolute path), or None if not found
    """
    # Resolve relative path
    if not os.path.isabs(data_dir):
        if not os.path.exists(data_dir):
            # Try resolving relative to script directory
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            processed_data_path = os.path.join(current_file_dir, 'processed_data')
            processed_data_path = os.path.normpath(processed_data_path)
            if os.path.exists(processed_data_path):
                data_dir = processed_data_path
    
    if not os.path.exists(data_dir):
        return None
    
    # Normalize the data directory path
    data_dir = os.path.normpath(os.path.abspath(data_dir))
    
    # Find all processed data files
    pattern = os.path.join(data_dir, 'processed_data_*.h5')
    data_files = glob.glob(pattern)
    
    if not data_files:
        return None
    
    # Extract n_steps and n_channels from filenames
    matching_files = []
    
    for filepath in data_files:
        filename = os.path.basename(filepath)
        
        # Extract n_steps (look for nsteps{N} pattern)
        nsteps_match = re.search(r'nsteps(\d+)', filename)
        if not nsteps_match:
            continue
        
        file_n_steps = int(nsteps_match.group(1))
        
        # Extract n_channels (look for ch{N} pattern)
        ch_match = re.search(r'_ch(\d+)_', filename)
        if not ch_match:
            continue
        
        file_n_channels = int(ch_match.group(1))
        
        # Check if matches
        if file_n_steps == model_n_steps and file_n_channels == model_n_channels:
            # Normalize filepath for consistent handling across operating systems
            normalized_filepath = os.path.normpath(os.path.abspath(filepath))
            matching_files.append((normalized_filepath, os.path.getmtime(filepath)))
    
    if not matching_files:
        return None
    
    # Return most recent file if multiple matches
    matching_files.sort(key=lambda x: x[1], reverse=True)
    return matching_files[0][0]


# ============================================================================
# MODEL LOADING
# ============================================================================

def extract_n_channels_from_weights(encoder_file: str) -> Optional[int]:
    """
    Extract n_channels from encoder weights.
    
    Args:
        encoder_file: Path to encoder weight file
        
    Returns:
        n_channels (int) or None if extraction fails
    """
    try:
        # Load to CPU since we only need to read tensor shapes, not run inference
        # This prevents CUDA memory accumulation when processing many models
        state_dict = torch.load(encoder_file, map_location='cpu', weights_only=False)
        
        if 'conv1.0.weight' in state_dict:
            n_channels = state_dict['conv1.0.weight'].shape[1]
            return n_channels
        return None
    except Exception as e:
        print(f"   ⚠️ Error extracting n_channels from {encoder_file}: {e}")
        return None


def extract_latent_dim_from_weights(encoder_file: str) -> Optional[int]:
    """
    Extract latent_dim from encoder weights.
    
    Args:
        encoder_file: Path to encoder weight file
        
    Returns:
        latent_dim (int) or None if extraction fails
    """
    try:
        # Load to CPU since we only need to read tensor shapes, not run inference
        # This prevents CUDA memory accumulation when processing many models
        state_dict = torch.load(encoder_file, map_location='cpu', weights_only=False)
        
        if 'fc_mean.weight' in state_dict:
            latent_dim = state_dict['fc_mean.weight'].shape[0]
            return latent_dim
        return None
    except Exception as e:
        print(f"   ⚠️ Error extracting latent_dim from {encoder_file}: {e}")
        return None


def load_model_and_config(model_info: Dict[str, Any], config_path: str, 
                          n_channels: int, latent_dim: int) -> Tuple[Optional[ROMWithE2C], Optional[Config]]:
    """
    Load model and configure it with hyperparameters (using dashboard logic).
    
    Args:
        model_info: Dictionary with model file paths and hyperparameters
        config_path: Path to base config file
        n_channels: Number of channels extracted from model weights
        latent_dim: Latent dimension extracted from model weights
        
    Returns:
        Tuple of (ROMWithE2C model, Config object) or (None, None) if loading fails
    """
    try:
        # Load base config
        config = Config(config_path)
        
        # Update config with hyperparameters (matching dashboard's _update_config_from_model exactly)
        # Update basic parameters - use actual values from weights
        config.set('model.latent_dim', latent_dim)  # Use extracted value
        config.set('training.nsteps', model_info['n_steps'])
        config.set('training.batch_size', model_info['batch_size'])
        
        # Update n_channels related config (matching dashboard logic)
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
        
        # Re-resolve dynamic values
        config._resolve_dynamic_values()
        
        # Get device
        device_config = config.runtime.get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            device = torch.device(device_config)
        config.device = device
        
        # Initialize model
        my_rom = ROMWithE2C(config).to(device)
        
        # Load weights
        my_rom.model.load_weights_from_file(
            model_info['encoder'],
            model_info['decoder'],
            model_info['transition']
        )
        
        return my_rom, config
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# PREDICTION GENERATION
# ============================================================================

def generate_predictions_with_timing(my_rom: ROMWithE2C, loaded_data: Dict[str, Any], 
                                     device: torch.device, data_dir: str, 
                                     num_tstep: int = 24) -> Tuple[Optional[InteractiveVisualizationDashboard], float, float]:
    """
    Generate predictions using dashboard's generate_test_visualization_standalone function.
    
    Args:
        my_rom: Trained ROM model
        loaded_data: Dictionary from load_processed_data()
        device: PyTorch device
        data_dir: Directory containing raw data files
        num_tstep: Number of time steps for prediction
        
    Returns:
        Tuple of (dashboard_instance, total_time_seconds, time_per_case_seconds)
    """
    try:
        # Use Timer to track prediction time
        with Timer("prediction", log_dir='./timing_logs/') as timer:
            dashboard = generate_test_visualization_standalone(
                loaded_data, my_rom, device, data_dir, num_tstep=num_tstep
            )
        
        total_time = timer.get_elapsed()
        
        # Calculate per-case time
        # Get number of cases from loaded_data
        num_test_cases = loaded_data.get('metadata', {}).get('num_eval', 0)
        num_train_cases = loaded_data.get('metadata', {}).get('num_train', 0)
        total_cases = num_test_cases + num_train_cases
        
        time_per_case = total_time / total_cases if total_cases > 0 else 0.0
        
        return dashboard, total_time, time_per_case
        
    except Exception as e:
        print(f"   ❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, 0.0


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_all_metrics(dashboard: InteractiveVisualizationDashboard, 
                          spatial_fields: Optional[List[str]] = None,
                          timeseries_groups: Optional[List[str]] = None,
                          spatial_layer: Optional[int] = None,
                          timeseries_well: Optional[str] = None,
                          use_averaged_metrics: bool = False) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculate overall performance metrics for all spatial fields and timeseries groups.
    
    Args:
        dashboard: InteractiveVisualizationDashboard instance with predictions
        spatial_fields: List of field names to calculate (None = all fields)
        timeseries_groups: List of group names to calculate (None = all groups)
        spatial_layer: Layer index to calculate (None = calculate for ALL layers per field, int = specific layer only)
        timeseries_well: Well name to calculate (None = calculate for ALL wells per group, str = specific well only)
        use_averaged_metrics: If True, calculate metrics for each individual case/layer/timestep combination and average them.
                             If False, aggregate all data points and calculate metrics (default).
        
    Returns:
        Tuple of (spatial_metrics_dict, timeseries_metrics_dict)
        Each dict has structure: {'training': {...}, 'testing': {...}}
        When spatial_layer=None, results include metrics for each layer: {'training': {'Field - Layer 0': {...}, 'Field - Layer 1': {...}, ...}}
        When timeseries_well=None, results include metrics for each well: {'training': {'Group - Well1': {...}, 'Group - Well2': {...}, ...}}
    """
    selected_metrics = ['r2', 'mse', 'rmse', 'mae']
    
    # Get field names from dashboard
    # Use field_names (display names) if available, otherwise use channel_names
    dashboard_field_names = None
    if hasattr(dashboard, 'field_names') and dashboard.field_names:
        dashboard_field_names = dashboard.field_names
    elif hasattr(dashboard, 'channel_names') and dashboard.channel_names:
        dashboard_field_names = dashboard.channel_names
    else:
        # Fallback: determine from state_pred shape
        n_channels = dashboard.state_pred.shape[2]
        dashboard_field_names = [f"Channel_{i}" for i in range(n_channels)]
    
    # Determine which fields to calculate
    if spatial_fields is None:
        spatial_fields = dashboard_field_names
    else:
        # Filter to only fields that exist in dashboard
        spatial_fields = [f for f in spatial_fields if f in dashboard_field_names]
        if not spatial_fields:
            print(f"   ⚠️ Warning: None of the specified fields found, using all fields")
            spatial_fields = dashboard_field_names
    
    # Get timeseries groups to calculate
    if timeseries_groups is None:
        timeseries_groups = list(TIMESERIES_GROUPS_MAP.keys())
    
    # Initialize results dictionaries
    spatial_metrics = {'training': {}, 'testing': {}}
    timeseries_metrics = {'training': {}, 'testing': {}}
    
    # Calculate spatial metrics for each field
    print(f"   📊 Calculating spatial metrics for {len(spatial_fields)} fields...")
    
    # Determine which layers to process
    if spatial_layer is not None:
        # Single layer specified
        layers_to_process = [spatial_layer]
    else:
        # Calculate for ALL layers per field
        layers_to_process = list(range(dashboard.Nz))
    
    for field_name in spatial_fields:
        # Find field index using field_names (display names)
        field_idx = None
        if hasattr(dashboard, 'field_names') and dashboard.field_names:
            try:
                field_idx = dashboard.field_names.index(field_name)
            except ValueError:
                pass
        
        # If not found in field_names, try channel_names
        if field_idx is None and hasattr(dashboard, 'channel_names') and dashboard.channel_names:
            try:
                field_idx = dashboard.channel_names.index(field_name)
            except ValueError:
                pass
        
        # Final fallback: use index from spatial_fields list
        if field_idx is None:
            try:
                field_idx = dashboard_field_names.index(field_name)
            except ValueError:
                print(f"   ⚠️ Warning: Field '{field_name}' not found, skipping")
                continue
        
        # Calculate metrics for each layer
        for layer_idx in layers_to_process:
            # Build field key for results (always include layer info)
            field_key = f"{field_name} - Layer {layer_idx}"
            
            # Calculate metrics for testing
            try:
                test_metrics, _ = dashboard._calculate_overall_spatial_metrics_optimized(
                    case_indices=None,  # Use all cases
                    selected_metrics=selected_metrics,
                    field_idx=field_idx,
                    use_training_data=False,
                    layer_idx=layer_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                if test_metrics:
                    spatial_metrics['testing'][field_key] = {
                        'r2': test_metrics.get('r2', 0.0),
                        'mse': test_metrics.get('mse', 0.0),
                        'rmse': test_metrics.get('rmse', 0.0),
                        'mae': test_metrics.get('mae', 0.0)
                    }
            except Exception as e:
                print(f"   ⚠️ Warning: Error calculating test metrics for {field_key}: {e}")
                spatial_metrics['testing'][field_key] = {'r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
            
            # Calculate metrics for training
            try:
                train_metrics, _ = dashboard._calculate_overall_spatial_metrics_optimized(
                    case_indices=None,  # Use all cases
                    selected_metrics=selected_metrics,
                    field_idx=field_idx,
                    use_training_data=True,
                    layer_idx=layer_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                if train_metrics:
                    spatial_metrics['training'][field_key] = {
                        'r2': train_metrics.get('r2', 0.0),
                        'mse': train_metrics.get('mse', 0.0),
                        'rmse': train_metrics.get('rmse', 0.0),
                        'mae': train_metrics.get('mae', 0.0)
                    }
            except Exception as e:
                print(f"   ⚠️ Warning: Error calculating training metrics for {field_key}: {e}")
                spatial_metrics['training'][field_key] = {'r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
    
    # Calculate timeseries metrics for each group
    print(f"   📊 Calculating timeseries metrics for {len(timeseries_groups)} groups...")
    for group_name in timeseries_groups:
        if group_name not in TIMESERIES_GROUPS_MAP:
            print(f"   ⚠️ Warning: Unknown timeseries group '{group_name}', skipping")
            continue
        
        obs_indices, well_names, unit = TIMESERIES_GROUPS_MAP[group_name]
        
        # Determine which wells to process
        if timeseries_well is not None and timeseries_well in well_names:
            # Single well specified
            well_idx = well_names.index(timeseries_well)
            wells_to_process = [(well_names[well_idx], obs_indices[well_idx])]
        else:
            # Calculate for ALL wells per group
            wells_to_process = list(zip(well_names, obs_indices))
        
        # Calculate metrics for each well
        for well_name, obs_idx in wells_to_process:
            group_key = f"{group_name} - {well_name}"
            
            # Calculate metrics for testing
            try:
                test_metrics_list, _ = dashboard._calculate_overall_timeseries_metrics_optimized(
                    case_indices=None,  # Use all cases
                    selected_metrics=selected_metrics,
                    obs_group_indices=None,  # Not using group averaging, using single obs
                    use_training_data=False,
                    obs_idx=obs_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                if test_metrics_list and len(test_metrics_list) > 0:
                    # For single observation, metrics_list contains one dict
                    test_metrics = test_metrics_list[0] if isinstance(test_metrics_list, list) else test_metrics_list
                    timeseries_metrics['testing'][group_key] = {
                        'r2': test_metrics.get('r2', 0.0),
                        'mse': test_metrics.get('mse', 0.0),
                        'rmse': test_metrics.get('rmse', 0.0),
                        'mae': test_metrics.get('mae', 0.0)
                    }
            except Exception as e:
                print(f"   ⚠️ Warning: Error calculating test metrics for {group_key}: {e}")
                timeseries_metrics['testing'][group_key] = {'r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
            
            # Calculate metrics for training
            try:
                train_metrics_list, _ = dashboard._calculate_overall_timeseries_metrics_optimized(
                    case_indices=None,  # Use all cases
                    selected_metrics=selected_metrics,
                    obs_group_indices=None,  # Not using group averaging, using single obs
                    use_training_data=True,
                    obs_idx=obs_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                if train_metrics_list and len(train_metrics_list) > 0:
                    # For single observation, metrics_list contains one dict
                    train_metrics = train_metrics_list[0] if isinstance(train_metrics_list, list) else train_metrics_list
                    timeseries_metrics['training'][group_key] = {
                        'r2': train_metrics.get('r2', 0.0),
                        'mse': train_metrics.get('mse', 0.0),
                        'rmse': train_metrics.get('rmse', 0.0),
                        'mae': train_metrics.get('mae', 0.0)
                    }
            except Exception as e:
                print(f"   ⚠️ Warning: Error calculating training metrics for {group_key}: {e}")
                timeseries_metrics['training'][group_key] = {'r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
    
    return spatial_metrics, timeseries_metrics


# ============================================================================
# MEMORY CLEANUP
# ============================================================================

def force_cuda_cleanup():
    """
    Force aggressive CUDA memory cleanup.
    This should be called proactively before loading new models/data.
    """
    try:
        if torch.cuda.is_available():
            # Synchronize all CUDA operations
            torch.cuda.synchronize()
            # Clear cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            # Clear cache again after GC
            torch.cuda.empty_cache()
    except Exception as e:
        # Don't let cleanup errors break execution
        pass


def cleanup_memory(my_rom=None, dashboard=None, loaded_data=None, config=None):
    """
    Clean up memory by deleting objects, clearing CUDA cache, and forcing garbage collection.
    More aggressive cleanup that handles CUDA OOM errors gracefully.
    
    Args:
        my_rom: ROM model instance to delete
        dashboard: Dashboard instance to delete
        loaded_data: Loaded data dictionary to delete
        config: Config object to delete
    """
    try:
        # Delete model if it exists - move to CPU first if possible
        if my_rom is not None:
            try:
                if hasattr(my_rom, 'model') and hasattr(my_rom.model, 'cpu'):
                    my_rom.model.cpu()
            except:
                pass
            try:
                del my_rom
            except:
                pass
        
        # Delete dashboard if it exists
        if dashboard is not None:
            try:
                # Try to clear dashboard's CUDA tensors if possible
                if hasattr(dashboard, 'state_pred'):
                    del dashboard.state_pred
                if hasattr(dashboard, 'state_true'):
                    del dashboard.state_true
                if hasattr(dashboard, 'obs_pred'):
                    del dashboard.obs_pred
            except:
                pass
            try:
                del dashboard
            except:
                pass
        
        # Delete loaded data if it exists
        if loaded_data is not None:
            try:
                # Clear large tensors explicitly
                for key in ['STATE_train', 'STATE_eval', 'BHP_train', 'BHP_eval', 
                           'Yobs_train', 'Yobs_eval', 'dt_train', 'dt_eval']:
                    if key in loaded_data:
                        del loaded_data[key]
            except:
                pass
            try:
                del loaded_data
            except:
                pass
        
        # Delete config if it exists
        if config is not None:
            try:
                del config
            except:
                pass
        
        # Force garbage collection before clearing CUDA cache
        gc.collect()
        
        # Clear CUDA cache if available - handle OOM errors gracefully
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Force another GC after clearing cache
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                # CUDA OOM during cleanup - try to recover
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
        
    except Exception as e:
        # Don't let cleanup errors break the main loop
        # Try to at least clear CUDA cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


# ============================================================================
# RESULTS STORAGE
# ============================================================================

def save_results(results: Dict[str, Any], output_dir: str, use_timestamp: bool = True):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary with all results
        output_dir: Directory to save results
        use_timestamp: If True, append timestamp to filename. If False, use consistent filename for incremental saves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(output_dir, f'grid_search_batch_metrics_{timestamp}.json')
    else:
        # Use consistent filename for incremental saves
        json_file = os.path.join(output_dir, 'grid_search_batch_metrics_latest.json')
    
    try:
        # Convert NaN and inf values to None/null for JSON serialization
        def convert_nan_to_none(obj):
            """Recursively convert NaN and inf values to None for JSON serialization"""
            if isinstance(obj, dict):
                return {k: convert_nan_to_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan_to_none(item) for item in obj]
            elif isinstance(obj, (float, np.floating)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            else:
                return obj
        
        # Clean results before saving
        cleaned_results = convert_nan_to_none(results)
        
        with open(json_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
        print(f"💾 Results saved to: {json_file}")
    except Exception as e:
        print(f"\n❌ Error saving results to {json_file}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main script orchestration"""
    print("=" * 80)
    print("🚀 GRID SEARCH BATCH METRICS CALCULATION")
    print("=" * 80)
    print(f"📊 Metrics calculation mode: {METRICS_CALCULATION_MODE}")
    print("=" * 80)
    
    # Discover models
    models = discover_models(MODELS_DIR)
    
    if not models:
        print("❌ No complete models found. Exiting.")
        return
    
    # Initialize results storage
    all_results = {}
    errors = []
    
    # Process each model
    total_runs = len(models)
    print(f"\n📊 Processing {total_runs} model runs...")
    print("=" * 80)
    
    for idx, model_info in enumerate(models, 1):
        # Proactively clear CUDA cache before processing each model
        # This helps prevent OOM errors from accumulating
        force_cuda_cleanup()
        
        # Create unique model ID (include n_channels if available)
        n_channels = model_info.get('n_channels', None)
        if n_channels is not None:
            model_id = f"bs{model_info['batch_size']}_ld{model_info['latent_dim']}_ns{model_info['n_steps']}_ch{n_channels}_{model_info['run_id']}"
        else:
            model_id = f"bs{model_info['batch_size']}_ld{model_info['latent_dim']}_ns{model_info['n_steps']}_{model_info['run_id']}"
        
        print(f"\n[{idx}/{total_runs}] Processing {model_id}...")
        print(f"   Hyperparameters: bs={model_info['batch_size']}, "
              f"ld={model_info['latent_dim']}, "
              f"ns={model_info['n_steps']}, "
              f"ch={n_channels if n_channels is not None else 'N/A'}, "
              f"run_id={model_info['run_id']}")
        
        # Initialize variables for cleanup
        my_rom = None
        dashboard = None
        loaded_data = None
        config = None
        
        try:
            # Get n_channels from model_info if available (from filename parsing), otherwise extract from weights
            encoder_file = model_info['encoder']
            n_channels = model_info.get('n_channels', None)
            latent_dim = extract_latent_dim_from_weights(encoder_file)
            
            # If n_channels not in model_info, extract from weights (for backward compatibility)
            if n_channels is None:
                n_channels = extract_n_channels_from_weights(encoder_file)
                if n_channels is None:
                    error_msg = f"Could not extract n_channels from encoder weights"
                    print(f"   ❌ {error_msg}")
                    errors.append({'model_id': model_id, 'error': error_msg})
                    continue
                print(f"   ✅ Extracted n_channels={n_channels} from model weights (not in filename)")
            else:
                print(f"   ✅ Using n_channels={n_channels} from filename")
            
            if latent_dim is None:
                error_msg = f"Could not extract latent_dim from encoder weights"
                print(f"   ❌ {error_msg}")
                errors.append({'model_id': model_id, 'error': error_msg})
                continue
            
            print(f"   ✅ Extracted latent_dim={latent_dim} from model weights")
            
            # Find matching data file
            n_steps = model_info['n_steps']
            data_file = find_matching_data_file(n_steps, n_channels, PROCESSED_DATA_DIR)
            
            if data_file is None:
                error_msg = f"No matching data file found for n_steps={n_steps}, n_channels={n_channels}"
                print(f"   ❌ {error_msg}")
                errors.append({'model_id': model_id, 'error': error_msg})
                continue
            
            print(f"   ✅ Found matching data file: {os.path.basename(data_file)}")
            
            # Load data
            print(f"   📂 Loading processed data...")
            loaded_data = load_processed_data(filepath=data_file, n_channels=n_channels)
            
            if loaded_data is None:
                error_msg = f"Failed to load data from {data_file}"
                print(f"   ❌ {error_msg}")
                errors.append({'model_id': model_id, 'error': error_msg})
                # Clean up any partial state
                cleanup_memory(my_rom=None, dashboard=None, loaded_data=None, config=None)
                continue
            
            print(f"   ✅ Data loaded successfully")
            
            # Load model
            print(f"   🤖 Loading model...")
            my_rom, config = load_model_and_config(
                model_info,
                CONFIG_PATH,
                n_channels,
                latent_dim
            )
            
            if my_rom is None:
                error_msg = f"Failed to load model"
                print(f"   ❌ {error_msg}")
                errors.append({'model_id': model_id, 'error': error_msg})
                # Clean up any partial state
                cleanup_memory(my_rom=None, dashboard=None, loaded_data=loaded_data, config=config)
                continue
            
            print(f"   ✅ Model loaded successfully")
            
            # Generate predictions
            # Note: DATA_DIR should point to directory containing raw data files (batch_timeseries_data_*.h5, etc.)
            # This is used by generate_test_visualization_standalone to load spatial and timeseries data
            print(f"   🔮 Generating predictions (num_tstep={NUM_TSTEP})...")
            dashboard, prediction_time, time_per_case = generate_predictions_with_timing(
                my_rom, loaded_data, config.device, DATA_DIR, num_tstep=NUM_TSTEP
            )
            
            if dashboard is None:
                error_msg = f"Failed to generate predictions"
                print(f"   ❌ {error_msg}")
                errors.append({'model_id': model_id, 'error': error_msg})
                # Clean up memory before continuing
                cleanup_memory(my_rom=my_rom, dashboard=None, loaded_data=loaded_data, config=config)
                continue
            
            print(f"   ✅ Predictions generated (time: {prediction_time:.2f}s, {time_per_case:.4f}s per case)")
            
            # Calculate metrics
            use_averaged_metrics = (METRICS_CALCULATION_MODE == 'Averaged')
            print(f"   📊 Calculating metrics (mode: {METRICS_CALCULATION_MODE})...")
            if SPATIAL_LAYER is not None:
                print(f"   📍 Spatial layer filter: Layer {SPATIAL_LAYER}")
            if TIMESERIES_WELL is not None:
                print(f"   📍 Timeseries well filter: {TIMESERIES_WELL}")
            spatial_metrics, timeseries_metrics = calculate_all_metrics(
                dashboard, 
                spatial_fields=SPATIAL_FIELDS,
                timeseries_groups=TIMESERIES_GROUPS,
                spatial_layer=SPATIAL_LAYER,
                timeseries_well=TIMESERIES_WELL,
                use_averaged_metrics=use_averaged_metrics
            )
            
            print(f"   ✅ Metrics calculated")
            
            # Store results
            all_results[model_id] = {
                'run_id': model_info['run_id'],
                'hyperparameters': {
                    'batch_size': model_info['batch_size'],
                    'latent_dim': model_info['latent_dim'],
                    'n_steps': model_info['n_steps']
                },
                'n_channels': n_channels,
                'data_file': os.path.basename(data_file),
                'spatial_metrics': spatial_metrics,
                'timeseries_metrics': timeseries_metrics,
                'prediction_time_seconds': prediction_time,
                'prediction_time_per_case_seconds': time_per_case
            }
            
            print(f"   ✅ {model_id} completed successfully")
            
            # Save results after each run (incremental save - overwrites same file)
            save_results(all_results, OUTPUT_DIR, use_timestamp=False)
            
            # Clean up memory after each successful run
            cleanup_memory(my_rom=my_rom, dashboard=dashboard, loaded_data=loaded_data, config=config)
            
        except RuntimeError as e:
            # Handle CUDA OOM errors specifically
            if "out of memory" in str(e) or "CUDA" in str(e):
                error_msg = f"CUDA out of memory: {str(e)}"
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Attempting aggressive memory cleanup...")
                errors.append({'model_id': model_id, 'error': error_msg})
                # Aggressive cleanup for OOM errors
                cleanup_memory(my_rom=my_rom, dashboard=dashboard, loaded_data=loaded_data, config=config)
                # Wait a moment and try clearing again
                time.sleep(1)
                force_cuda_cleanup()
            else:
                error_msg = f"Runtime error: {str(e)}"
                print(f"   ❌ {error_msg}")
                errors.append({'model_id': model_id, 'error': error_msg})
                cleanup_memory(my_rom=my_rom, dashboard=dashboard, loaded_data=loaded_data, config=config)
            import traceback
            traceback.print_exc()
            continue
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"   ❌ {error_msg}")
            errors.append({'model_id': model_id, 'error': error_msg})
            import traceback
            traceback.print_exc()
            
            # Clean up memory even on error
            cleanup_memory(my_rom=my_rom, dashboard=dashboard, loaded_data=loaded_data, config=config)
            continue
    
    # Save results
    print("\n" + "=" * 80)
    print("💾 Saving results...")
    save_results(all_results, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total runs processed: {total_runs}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {len(errors)}")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors:
            print(f"   {error['model_id']}: {error['error']}")
    
    # Save error log
    if errors:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = os.path.join(OUTPUT_DIR, f'grid_search_batch_metrics_errors_{timestamp}.log')
        with open(error_log_file, 'w') as f:
            f.write("Grid Search Batch Metrics Calculation Errors\n")
            f.write("=" * 80 + "\n\n")
            for error in errors:
                f.write(f"Model ID: {error['model_id']}\n")
                f.write(f"Error: {error['error']}\n")
                f.write("-" * 80 + "\n")
        print(f"\n📝 Error log saved to: {error_log_file}")
    
    print("=" * 80)
    print("✅ Batch metrics calculation completed!")


if __name__ == "__main__":
    main()


# %%

