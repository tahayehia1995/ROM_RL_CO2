"""
Runtime Timing Utility
Tracks execution times for data preprocessing, training, and prediction phases.
Provides context manager interface and JSON logging with metadata.
"""

import time
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np


class Timer:
    """
    Context manager for tracking execution time of code blocks.
    Automatically prints timing information and saves to JSON log file.
    """
    
    def __init__(self, phase_name: str, log_dir: str = './timing_logs/'):
        """
        Initialize Timer.
        
        Args:
            phase_name: Name of the phase being timed (e.g., 'processing', 'training', 'prediction')
            log_dir: Directory to save timing log files
        """
        self.phase_name = phase_name
        self.log_dir = log_dir
        self.start_time = None
        self.end_time = None
        self.metadata = {}
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def __enter__(self):
        """Start timing when entering context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and save results when exiting context manager."""
        self.stop()
        if exc_type is None:  # Only save if no exception occurred
            self.print_timing(self.metadata)
            self.save_to_json(self.metadata)
        return False  # Don't suppress exceptions
    
    def start(self):
        """Manually start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Manually stop timing."""
        self.end_time = time.time()
    
    def get_elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds, or 0 if timing hasn't started/stopped
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to HH:MM:SS format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted string in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def print_timing(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Print formatted timing information.
        
        Args:
            metadata: Optional metadata dictionary to include in output
        """
        elapsed = self.get_elapsed()
        formatted = self._format_duration(elapsed)
        
        print(f"\n{'='*70}")
        print(f"⏱️  {self.phase_name.upper()} TIMING")
        print(f"{'='*70}")
        print(f"Duration: {elapsed:.4f} seconds ({formatted})")
        
        if metadata:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"  • {key}: {value}")
                elif isinstance(value, dict):
                    print(f"  • {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, str, bool)):
                            print(f"    - {sub_key}: {sub_value}")
        print(f"{'='*70}\n")
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types and other non-JSON-serializable objects to native Python types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        # Check for numpy arrays first
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Check for numpy integer types (compatible with NumPy 1.x and 2.x)
        # Use specific types instead of abstract base classes for NumPy 2.0 compatibility
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # Check for numpy floating types (compatible with NumPy 1.x and 2.x)
        # Use specific types instead of abstract base classes for NumPy 2.0 compatibility
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        
        # Check for numpy boolean
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Check for abstract base classes (NumPy 1.x only, wrapped in try-except)
        # This provides backward compatibility but won't fail in NumPy 2.0
        try:
            if hasattr(np, 'integer') and isinstance(obj, np.integer):
                return int(obj)
        except (AttributeError, TypeError):
            pass
        
        try:
            if hasattr(np, 'floating') and isinstance(obj, np.floating):
                return float(obj)
        except (AttributeError, TypeError):
            pass
        
        # Recursively handle dictionaries and lists
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def save_to_json(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Save timing data with metadata to JSON log file.
        
        Args:
            metadata: Optional metadata dictionary to include in log entry
        """
        elapsed = self.get_elapsed()
        formatted = self._format_duration(elapsed)
        
        # Convert metadata to JSON-serializable format
        serializable_metadata = self._convert_to_json_serializable(metadata) if metadata else {}
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase_name,
            "duration_seconds": elapsed,
            "duration_formatted": formatted,
            "metadata": serializable_metadata
        }
        
        # Log file name: timing_log_YYYY-MM-DD.json
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"timing_log_{date_str}.json")
        
        # Append to log file (one JSON object per line for easy parsing)
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')


def collect_processing_metadata(dashboard) -> Dict[str, Any]:
    """
    Collect metadata from DataPreprocessingDashboard for timing log.
    
    Args:
        dashboard: DataPreprocessingDashboard instance
        
    Returns:
        Dictionary containing processing metadata
    """
    metadata = {}
    
    try:
        # Basic information
        metadata['data_directory'] = getattr(dashboard, 'data_dir', 'N/A')
        metadata['n_steps'] = getattr(dashboard, 'nsteps', 'N/A')
        
        # Channel information
        if hasattr(dashboard, 'selected_training_channels'):
            metadata['num_channels'] = len(dashboard.selected_training_channels)
            metadata['selected_channels'] = [ch for ch in dashboard.selected_training_channels]
        else:
            metadata['num_channels'] = 'N/A'
        
        # Normalization settings
        if hasattr(dashboard, 'normalization_settings'):
            metadata['normalization_settings'] = dashboard.normalization_settings
        else:
            metadata['normalization_settings'] = {}
        
        # Train/test split
        if hasattr(dashboard, 'train_test_split_ratio'):
            metadata['train_test_split_ratio'] = dashboard.train_test_split_ratio
        else:
            metadata['train_test_split_ratio'] = 'N/A'
        
        # Data dimensions
        if hasattr(dashboard, 'Nx'):
            nx = dashboard.Nx
            ny = dashboard.Ny
            nz = dashboard.Nz
            # Convert numpy types to native Python types
            if isinstance(nx, (np.integer, np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
                nx = int(nx)
            if isinstance(ny, (np.integer, np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
                ny = int(ny)
            if isinstance(nz, (np.integer, np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
                nz = int(nz)
            metadata['spatial_dimensions'] = {
                'Nx': nx,
                'Ny': ny,
                'Nz': nz
            }
        
        # Number of cases
        if hasattr(dashboard, 'STATE_train') and dashboard.STATE_train is not None:
            if isinstance(dashboard.STATE_train, list) and len(dashboard.STATE_train) > 0:
                num_train = dashboard.STATE_train[0].shape[0]
                # Convert numpy int64 to native Python int
                if isinstance(num_train, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                         np.uint32, np.uint64)):
                    num_train = int(num_train)
                metadata['num_training_cases'] = num_train
            else:
                metadata['num_training_cases'] = 'N/A'
        else:
            metadata['num_training_cases'] = 'N/A'
        
        if hasattr(dashboard, 'STATE_eval') and dashboard.STATE_eval is not None:
            if isinstance(dashboard.STATE_eval, list) and len(dashboard.STATE_eval) > 0:
                num_eval = dashboard.STATE_eval[0].shape[0]
                # Convert numpy int64 to native Python int
                if isinstance(num_eval, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                         np.uint32, np.uint64)):
                    num_eval = int(num_eval)
                metadata['num_evaluation_cases'] = num_eval
            else:
                metadata['num_evaluation_cases'] = 'N/A'
        else:
            metadata['num_evaluation_cases'] = 'N/A'
        
        # Number of wells
        if hasattr(dashboard, 'num_well'):
            num_well = dashboard.num_well
            # Convert numpy int64 to native Python int
            if isinstance(num_well, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                     np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                     np.uint32, np.uint64)):
                num_well = int(num_well)
            metadata['num_wells'] = num_well
        else:
            metadata['num_wells'] = 'N/A'
        
        # Controls and observations
        if hasattr(dashboard, 'selected_controls'):
            metadata['num_controls'] = sum(len(config.get('wells', [])) for config in dashboard.selected_controls.values())
        else:
            metadata['num_controls'] = 'N/A'
        
        if hasattr(dashboard, 'selected_observations'):
            metadata['num_observations'] = sum(len(config.get('wells', [])) for config in dashboard.selected_observations.values())
        else:
            metadata['num_observations'] = 'N/A'
            
    except Exception as e:
        metadata['metadata_collection_error'] = str(e)
    
    return metadata


def collect_training_metadata(config, loaded_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Collect metadata from training config and loaded data for timing log.
    
    Args:
        config: Config object containing training configuration
        loaded_data: Optional dictionary containing loaded training data
        
    Returns:
        Dictionary containing training metadata
    """
    metadata = {}
    
    try:
        # Training parameters
        if hasattr(config, 'training'):
            metadata['epochs'] = config.training.get('epoch', 'N/A')
            metadata['batch_size'] = config.training.get('batch_size', 'N/A')
            metadata['learning_rate'] = config.training.get('learning_rate', 'N/A')
        
        # Model architecture
        if hasattr(config, 'model'):
            metadata['model_method'] = config.model.get('method', 'N/A')
            metadata['latent_dimension'] = config.model.get('latent_dim', 'N/A')
            metadata['n_channels'] = config.model.get('n_channels', 'N/A')
        
        # Transition type
        if hasattr(config, 'transition'):
            metadata['transition_type'] = config.transition.get('type', 'N/A')
        else:
            metadata['transition_type'] = 'N/A'
        
        # Device
        if hasattr(config, 'device'):
            metadata['device'] = str(config.device)
        elif hasattr(config, 'runtime'):
            metadata['device'] = config.runtime.get('device', 'N/A')
        else:
            metadata['device'] = 'N/A'
        
        # Scheduler
        if hasattr(config, 'learning_rate_scheduler'):
            metadata['scheduler_type'] = config.learning_rate_scheduler.get('type', 'N/A')
            metadata['scheduler_params'] = {k: v for k, v in config.learning_rate_scheduler.items() if k != 'type'}
        else:
            metadata['scheduler_type'] = 'N/A'
        
        # Loss weights
        if hasattr(config, 'loss'):
            loss_weights = {}
            for key in ['reconstruction_weight', 'transition_weight', 'observation_weight', 
                       'flux_weight', 'well_weight', 'non_negative_weight']:
                if key in config.loss:
                    loss_weights[key] = config.loss[key]
            if loss_weights:
                metadata['loss_weights'] = loss_weights
        
        # Best model criterion
        if hasattr(config, 'runtime'):
            metadata['best_model_criterion'] = config.runtime.get('best_model_criterion', 'N/A')
        
        # Data information from loaded_data
        if loaded_data:
            if 'metadata' in loaded_data:
                meta = loaded_data['metadata']
                num_train = meta.get('num_train', 'N/A')
                num_eval = meta.get('num_eval', 'N/A')
                num_well = meta.get('num_well', 'N/A')
                # Convert numpy types to native Python types
                if isinstance(num_train, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                         np.uint32, np.uint64)):
                    num_train = int(num_train)
                if isinstance(num_eval, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                         np.uint32, np.uint64)):
                    num_eval = int(num_eval)
                if isinstance(num_well, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                         np.uint32, np.uint64)):
                    num_well = int(num_well)
                metadata['num_training_cases'] = num_train
                metadata['num_evaluation_cases'] = num_eval
                metadata['num_wells'] = num_well
                
                nx = meta.get('Nx', 'N/A')
                ny = meta.get('Ny', 'N/A')
                nz = meta.get('Nz', 'N/A')
                # Convert numpy types
                if isinstance(nx, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                  np.uint32, np.uint64)):
                    nx = int(nx)
                if isinstance(ny, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                  np.uint32, np.uint64)):
                    ny = int(ny)
                if isinstance(nz, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                  np.uint32, np.uint64)):
                    nz = int(nz)
                metadata['spatial_dimensions'] = {
                    'Nx': nx,
                    'Ny': ny,
                    'Nz': nz
                }
        
        # Final loss values (if available from model)
        # Note: These will be added after training completes
        
    except Exception as e:
        metadata['metadata_collection_error'] = str(e)
    
    return metadata


def collect_prediction_metadata(num_cases: int, num_timesteps: int, 
                                 device: Optional[str] = None,
                                 model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Collect metadata for prediction timing log.
    
    Args:
        num_cases: Number of test cases
        num_timesteps: Number of time steps for prediction
        device: Device used for prediction (optional)
        model_info: Additional model information dictionary (optional)
        
    Returns:
        Dictionary containing prediction metadata
    """
    metadata = {
        'num_cases': num_cases,
        'num_timesteps': num_timesteps
    }
    
    if device:
        metadata['device'] = str(device)
    
    if model_info:
        metadata.update(model_info)
    
    return metadata

