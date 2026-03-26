"""
Simple Configuration Loader for E2C Model
=========================================
This module provides a lightweight configuration management system for the E2C model.
It loads YAML configuration files and makes values easily accessible throughout the codebase.
"""

import yaml
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from typing import Any, Dict, List, Union, Optional

class Config:
    """
    Simple configuration loader and manager for E2C model.
    
    Features:
    - Load YAML configuration files
    - Automatic device detection
    - Dynamic value resolution (e.g., null -> n_channels)
    - Easy nested value access with dot notation
    - Configuration validation
    
    Usage:
        config = Config('config.yaml')
        latent_dim = config.model['latent_dim']
        device = config.device
        learning_rate = config.get('training.learning_rate')
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file (default: 'config.yaml' for unified config)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._normalize_config_structure()
        self._setup_device()
        self._resolve_dynamic_values()
        self._validate_config()
        
        # Only print verbose message if runtime section exists
        if 'runtime' in self.config and self.config.get('runtime', {}).get('verbose', False):
            print(f"Configuration loaded: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    
    def _normalize_config_structure(self):
        """Normalize config structure to handle unified config format."""
        # If config has data_preprocessing section, merge it into top level for backward compatibility
        if 'data_preprocessing' in self.config:
            data_prep = self.config['data_preprocessing']
            # Merge paths if they exist
            if 'paths' in data_prep:
                if 'paths' not in self.config:
                    self.config['paths'] = {}
                self.config['paths'].update(data_prep['paths'])
            # Merge other sections if needed
            if 'processing' in data_prep:
                self.config['data_preprocessing_processing'] = data_prep['processing']
            if 'normalization' in data_prep:
                self.config['data_preprocessing_normalization'] = data_prep['normalization']
        
        # If config has testing section, ensure it's accessible
        if 'testing' in self.config:
            # Testing section is already at top level, no action needed
            pass
    
    def _setup_device(self):
        """Set up computation device based on config and availability."""
        # Handle configs that may not have runtime section (e.g., testing configs)
        if 'runtime' in self.config and 'device' in self.config['runtime']:
            device_config = self.config['runtime']['device']
        else:
            device_config = 'auto'  # Default to auto if not specified
        
        if device_config == 'auto':
            # Automatic device selection
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            # Manual device selection
            self.device = torch.device(device_config)
    
    def _resolve_dynamic_values(self):
        """Resolve dynamic values like 'null' -> n_channels."""
        # Only resolve if this is a model config
        if 'model' not in self.config or 'n_channels' not in self.config['model']:
            return
        
        n_channels = self.config['model']['n_channels']
        
        # Replace null values in encoder
        if 'encoder' in self.config and 'conv_layers' in self.config['encoder']:
            if 'conv1' in self.config['encoder']['conv_layers']:
                if self.config['encoder']['conv_layers']['conv1'][0] is None:
                    self.config['encoder']['conv_layers']['conv1'][0] = n_channels
        
        # Replace null values in decoder
        if 'decoder' in self.config and 'deconv_layers' in self.config['decoder']:
            if 'final_conv' in self.config['decoder']['deconv_layers']:
                if self.config['decoder']['deconv_layers']['final_conv'][1] is None:
                    self.config['decoder']['deconv_layers']['final_conv'][1] = n_channels
    
    # ===== ADD TYPE CONVERSION METHODS =====
    def _convert_scheduler_types(self):
        """Convert scheduler configuration values to proper types."""
        if 'learning_rate_scheduler' not in self.config:
            return
        
        lr_config = self.config['learning_rate_scheduler']
        
        # Type conversion mappings
        type_conversions = {
            'reduce_on_plateau': {
                'factor': float,
                'patience': int,
                'threshold': float,
                'cooldown': int,
                'min_lr': float,
                'eps': float,
                'verbose': bool
            },
            'exponential_decay': {
                'gamma': float
            },
            'step_decay': {
                'step_size': int,
                'gamma': float
            },
            'cosine_annealing': {
                'T_max': int,
                'eta_min': float
            },
            'cyclic': {
                'base_lr': float,
                'max_lr': float,
                'step_size_up': int,
                'step_size_down': int,
                'gamma': float,
                'base_momentum': float,
                'max_momentum': float,
                'cycle_momentum': bool
            },
            'one_cycle': {
                'max_lr': float,
                'total_steps': int,
                'epochs': int,
                'steps_per_epoch': int,
                'pct_start': float,
                'cycle_momentum': bool,
                'base_momentum': float,
                'max_momentum': float,
                'div_factor': float,
                'final_div_factor': float
            }
        }
        
        # Apply conversions
        for scheduler_type, params in type_conversions.items():
            if scheduler_type in lr_config:
                scheduler_config = lr_config[scheduler_type]
                for param_name, param_type in params.items():
                    if param_name in scheduler_config:
                        try:
                            # Handle None values and type conversion
                            value = scheduler_config[param_name]
                            if value is not None:
                                if param_type == bool:
                                    # Handle various boolean representations
                                    if isinstance(value, str):
                                        scheduler_config[param_name] = value.lower() in ('true', '1', 'yes', 'on')
                                    else:
                                        scheduler_config[param_name] = bool(value)
                                else:
                                    scheduler_config[param_name] = param_type(value)
                        except (ValueError, TypeError) as e:
                            pass

    # ===== FINAL SCHEDULER VALIDATION =====
    def _final_scheduler_validation(self):
        """Final validation and type conversion for scheduler parameters."""
        if 'learning_rate_scheduler' not in self.config:
            return
        
        lr_config = self.config['learning_rate_scheduler']
        
        # Check if scheduler is enabled
        enable_value = lr_config.get('enable', False)
        if isinstance(enable_value, str):
            lr_config['enable'] = enable_value.lower() in ('true', '1', 'yes', 'on')
        
        # Check scheduler type
        scheduler_type = lr_config.get('type', 'fixed')
        if scheduler_type == 'fixed' or not lr_config.get('enable', False):
            return
        
        # Final parameter type check for all schedulers
        critical_params = {
            'reduce_on_plateau': ['factor', 'patience', 'threshold', 'cooldown', 'min_lr', 'eps'],
            'exponential_decay': ['gamma'],
            'step_decay': ['step_size', 'gamma'],
            'cosine_annealing': ['T_max', 'eta_min'],
            'cyclic': ['base_lr', 'max_lr', 'step_size_up', 'step_size_down', 'gamma', 'base_momentum', 'max_momentum'],
            'one_cycle': ['max_lr', 'total_steps', 'epochs', 'steps_per_epoch', 'pct_start', 'div_factor', 'final_div_factor', 'base_momentum', 'max_momentum']
        }
        
        if scheduler_type in critical_params and scheduler_type in lr_config:
            config_section = lr_config[scheduler_type]
            for param in critical_params[scheduler_type]:
                if param in config_section:
                    value = config_section[param]
                    if value is not None and isinstance(value, str):
                        try:
                            # Try to convert string to appropriate numeric type
                            if param in ['patience', 'cooldown', 'step_size', 'step_size_up', 'step_size_down', 'T_max', 'total_steps', 'epochs', 'steps_per_epoch']:
                                config_section[param] = int(float(value))  # Convert to int via float to handle "5.0" strings
                            else:
                                config_section[param] = float(value)
                        except (ValueError, TypeError):
                            pass

    def _validate_config(self):
        """Basic validation of configuration values."""
        # Check required sections only if this appears to be a model config
        # (has at least one of the model-specific sections)
        has_model_sections = any(section in self.config for section in ['model', 'training', 'encoder', 'decoder', 'transition'])
        
        if has_model_sections:
            # This is a model config - validate all required sections
            required_sections = ['model', 'data', 'training', 'loss', 'encoder', 'decoder', 'transition', 'paths', 'runtime']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Validate data dimensions (only for model configs)
            if 'data' in self.config and 'input_shape' in self.config['data']:
                input_shape = self.config['data']['input_shape']
                if len(input_shape) != 4:
                    raise ValueError(f"input_shape must be 4D [channels, X, Y, Z], got: {input_shape}")
                
                if 'model' in self.config and 'n_channels' in self.config['model']:
                    if input_shape[0] != self.config['model']['n_channels']:
                        raise ValueError(f"input_shape channels ({input_shape[0]}) must match n_channels ({self.config['model']['n_channels']})")
            
            # Validate model parameters (only for model configs)
            if 'model' in self.config and 'latent_dim' in self.config['model']:
                if self.config['model']['latent_dim'] <= 0:
                    raise ValueError("latent_dim must be positive")
            
            if 'training' in self.config and 'batch_size' in self.config['training']:
                if self.config['training']['batch_size'] <= 0:
                    raise ValueError("batch_size must be positive")
            
            # ===== ADD LOSS PARAMETER VALIDATION =====
            if 'loss' in self.config:
                self._validate_loss_parameters()
            
            # ===== ADD SCHEDULER TYPE VALIDATION =====
            if 'learning_rate_scheduler' in self.config:
                self._convert_scheduler_types()
                self._final_scheduler_validation()
    
    def _validate_loss_parameters(self):
        """Validate loss configuration parameters."""
        loss_config = self.config.get('loss', {})
        
        # Validate reconstruction_variance parameter
        reconstruction_variance = loss_config.get('reconstruction_variance', 0.1)
        
        try:
            reconstruction_variance = float(reconstruction_variance)
        except (ValueError, TypeError):
            raise ValueError(f"reconstruction_variance must be a number, got: {reconstruction_variance}")
        
        if reconstruction_variance <= 0:
            raise ValueError(f"reconstruction_variance must be positive, got: {reconstruction_variance}")
        
        if reconstruction_variance > 1.0:
            print(f"⚠️ Warning: reconstruction_variance ({reconstruction_variance:.4f}) is quite large (>1.0)")
            print(f"   This makes reconstruction loss very forgiving. Consider values 0.01-0.5 for most applications.")
        
        if reconstruction_variance < 0.001:
            print(f"⚠️ Warning: reconstruction_variance ({reconstruction_variance:.6f}) is very small (<0.001)")
            print(f"   This makes reconstruction loss extremely strict. May cause training instability.")
        
        # Update the config with validated value
        loss_config['reconstruction_variance'] = reconstruction_variance
        
        # Validate per-element normalization setting
        enable_per_element_norm = loss_config.get('enable_per_element_normalization', False)
        if not isinstance(enable_per_element_norm, bool):
            try:
                enable_per_element_norm = str(enable_per_element_norm).lower() in ('true', '1', 'yes', 'on')
                loss_config['enable_per_element_normalization'] = enable_per_element_norm
            except:
                print(f"⚠️ Warning: Invalid enable_per_element_normalization value. Using False.")
                loss_config['enable_per_element_normalization'] = False
        
        # Warning about lambda weights with normalization
        if enable_per_element_norm:
            high_lambdas = []
            if loss_config.get('lambda_trans_loss', 1.0) > 5.0:
                high_lambdas.append(f"lambda_trans_loss={loss_config.get('lambda_trans_loss')}")
            if loss_config.get('lambda_yobs_loss', 1.0) > 5.0:
                high_lambdas.append(f"lambda_yobs_loss={loss_config.get('lambda_yobs_loss')}")
            
            if high_lambdas:
                print(f"⚠️ Note: Per-element normalization is enabled with high lambda weights:")
                for lamb in high_lambdas:
                    print(f"   • {lamb} (consider reducing to ~1-2 for balanced training)")
                print(f"   💡 With normalization, lambda weights should typically be 0.5-2.0")
        
        # Validate other loss weights while we're here
        loss_weights = ['lambda_reconstruction_loss', 'lambda_flux_loss', 'lambda_bhp_loss', 
                       'lambda_trans_loss', 'lambda_yobs_loss', 'lambda_non_negative_loss']
        
        for weight_name in loss_weights:
            if weight_name in loss_config:
                try:
                    weight_value = float(loss_config[weight_name])
                    if weight_value < 0:
                        print(f"⚠️ Warning: {weight_name} ({weight_value}) is negative. Loss weights should be non-negative.")
                    loss_config[weight_name] = weight_value
                except (ValueError, TypeError):
                    print(f"⚠️ Warning: {weight_name} value '{loss_config[weight_name]}' is not a valid number.")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to value (e.g., 'model.latent_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            latent_dim = config.get('model.latent_dim')
            batch_size = config.get('training.batch_size', 4)
        """
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to value
            value: Value to set
        """
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the final value
        config_section[keys[-1]] = value
    
    def save_config_snapshot(self, output_path: str):
        """
        Save current configuration to file for experiment reproducibility.
        
        Args:
            output_path: Path to save configuration snapshot
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"📸 Configuration snapshot saved to: {output_path}")
    
    def create_model_filename(self, component: str, num_train: int, num_well: int) -> str:
        """
        Create standardized model filename based on configuration.
        
        Args:
            component: Model component ('encoder', 'decoder', 'transition')
            num_train: Number of training samples
            num_well: Number of wells
            
        Returns:
            Formatted filename string
        """
        return (f"{self.paths['model_prefix']}_{component}_3D_native_"
                f"nt{num_train}_l{self.model['latent_dim']}_"
                f"lr{self.training['learning_rate']:.0e}_"
                f"ep{self.training['epoch']}_"
                f"steps{self.training['nsteps']}_"
                f"channels{self.model['n_channels']}_"
                f"wells{num_well}{self.paths['file_extension']}")
    
    def get_well_locations(self, well_type: str) -> List[List[int]]:
        """
        Get well locations for specified well type.
        
        Args:
            well_type: 'producers' or 'injectors'
            
        Returns:
            List of [X, Y, Z] coordinates for each well
        """
        well_locations = self.data['well_locations'][well_type]
        locations_list = []
        
        # Sort wells by name to ensure consistent ordering
        sorted_wells = sorted(well_locations.items())
        
        for well_name, coords in sorted_wells:
            locations_list.append(coords)
            
        return locations_list
    

    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: Optional[int] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: The optimizer to schedule
            num_training_steps: Total number of training steps (needed for some schedulers)
            
        Returns:
            Scheduler object or None if scheduling is disabled
        """
        # Check if scheduling is enabled
        if not self.config.get('learning_rate_scheduler', {}).get('enable', False):
            if self.config.get('runtime', {}).get('verbose', True):
                print("📅 Learning rate scheduling disabled - using fixed learning rate")
            return None
        
        scheduler_config = self.config['learning_rate_scheduler']
        scheduler_type = scheduler_config.get('type', 'fixed').lower()
        
        if self.runtime.get('verbose', True):
            print(f"📅 Creating {scheduler_type} learning rate scheduler")
        
        try:
            return self._create_scheduler_by_type(optimizer, scheduler_type, scheduler_config, num_training_steps)
        except Exception as e:
            print(f"❌ Error creating scheduler: {e}")
            print("⚠️  Falling back to fixed learning rate")
            return None
    
    def _create_scheduler_by_type(self, optimizer: torch.optim.Optimizer, scheduler_type: str, 
                                 scheduler_config: Dict[str, Any], num_training_steps: Optional[int] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create specific scheduler type with robust type handling."""
        
        if scheduler_type == 'fixed':
            return None
            
        elif scheduler_type == 'reduce_on_plateau':
            config = scheduler_config.get('reduce_on_plateau', {})
            
            # Safe parameter extraction with type conversion
            def safe_get(key, default, convert_type):
                try:
                    value = config.get(key, default)
                    return convert_type(value) if value is not None else default
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid {key} value '{value}', using default {default}")
                    return default
            
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get('mode', 'min'),
                factor=safe_get('factor', 0.5, float),
                patience=safe_get('patience', 5, int),
                threshold=safe_get('threshold', 1e-4, float),
                threshold_mode=config.get('threshold_mode', 'rel'),
                cooldown=safe_get('cooldown', 3, int),
                min_lr=safe_get('min_lr', 1e-7, float),
                eps=safe_get('eps', 1e-8, float),
                verbose=safe_get('verbose', True, bool)
            )
            
        elif scheduler_type == 'exponential_decay':
            config = scheduler_config.get('exponential_decay', {})
            
            def safe_get(key, default, convert_type):
                try:
                    value = config.get(key, default)
                    return convert_type(value) if value is not None else default
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid {key} value '{value}', using default {default}")
                    return default
            
            return lr_scheduler.ExponentialLR(
                optimizer,
                gamma=safe_get('gamma', 0.95, float)
            )
            
        elif scheduler_type == 'step_decay':
            config = scheduler_config.get('step_decay', {})
            
            def safe_get(key, default, convert_type):
                try:
                    value = config.get(key, default)
                    return convert_type(value) if value is not None else default
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid {key} value '{value}', using default {default}")
                    return default
            
            return lr_scheduler.StepLR(
                optimizer,
                step_size=safe_get('step_size', 10, int),
                gamma=safe_get('gamma', 0.1, float)
            )
            
        elif scheduler_type == 'cosine_annealing':
            config = scheduler_config.get('cosine_annealing', {})
            
            def safe_get(key, default, convert_type):
                try:
                    value = config.get(key, default)
                    return convert_type(value) if value is not None else default
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid {key} value '{value}', using default {default}")
                    return default
            
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=safe_get('T_max', self.training['epoch'], int),
                eta_min=safe_get('eta_min', 1e-6, float)
            )
            
        elif scheduler_type == 'cyclic':
            config = scheduler_config.get('cyclic', {})
            
            def safe_get(key, default, convert_type):
                try:
                    value = config.get(key, default)
                    return convert_type(value) if value is not None else default
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid {key} value '{value}', using default {default}")
                    return default
            
            # Validate cyclic LR parameters with safe type conversion
            base_lr = safe_get('base_lr', 1e-5, float)
            max_lr = safe_get('max_lr', 1e-3, float)
            if base_lr >= max_lr:
                print(f"⚠️ base_lr ({base_lr}) must be less than max_lr ({max_lr}), adjusting automatically")
                base_lr = max_lr * 0.1  # Set base_lr to 10% of max_lr
            
            step_size_up = safe_get('step_size_up', 500, int)
            step_size_down = safe_get('step_size_down', step_size_up, int)
            
            return lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode=config.get('mode', 'triangular'),
                gamma=safe_get('gamma', 1.0, float),
                scale_fn=config.get('scale_fn', None),
                scale_mode=config.get('scale_mode', 'cycle'),
                cycle_momentum=safe_get('cycle_momentum', True, bool),
                base_momentum=safe_get('base_momentum', 0.8, float),
                max_momentum=safe_get('max_momentum', 0.9, float)
            )
            
        elif scheduler_type == 'one_cycle':
            config = scheduler_config.get('one_cycle', {})
            
            def safe_get(key, default, convert_type):
                try:
                    value = config.get(key, default)
                    return convert_type(value) if value is not None else default
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid {key} value '{value}', using default {default}")
                    return default
            
            # Calculate total steps if not provided with safe type conversion
            if num_training_steps is None:
                epochs = safe_get('epochs', self.training['epoch'], int)
                steps_per_epoch = safe_get('steps_per_epoch', None, int)
                if steps_per_epoch is None:
                    # This will be calculated later in the training script
                    raise ValueError("num_training_steps must be provided for OneCycleLR scheduler")
                total_steps = epochs * steps_per_epoch
            else:
                total_steps = safe_get('total_steps', num_training_steps, int)
            
            return lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=safe_get('max_lr', 1e-3, float),
                total_steps=total_steps,
                pct_start=safe_get('pct_start', 0.3, float),
                anneal_strategy=config.get('anneal_strategy', 'cos'),
                cycle_momentum=safe_get('cycle_momentum', True, bool),
                base_momentum=safe_get('base_momentum', 0.85, float),
                max_momentum=safe_get('max_momentum', 0.95, float),
                div_factor=safe_get('div_factor', 25.0, float),
                final_div_factor=safe_get('final_div_factor', 1e4, float)
            )
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Get scheduler configuration information."""
        scheduler_config = self.config.get('learning_rate_scheduler', {})
        
        if not scheduler_config.get('enable', False):
            return {'enabled': False, 'type': 'fixed'}
        
        return {
            'enabled': True,
            'type': scheduler_config.get('type', 'fixed'),
            'requires_validation_loss': scheduler_config.get('type', 'fixed') == 'reduce_on_plateau',
            'step_on_batch': scheduler_config.get('type', 'fixed') in ['cyclic', 'one_cycle'],
            'logging': scheduler_config.get('logging', {})
        }
    
    def validate_scheduler_config(self) -> List[str]:
        """
        Validate scheduler configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if no issues)
        """
        issues = []
        
        scheduler_config = self.config.get('learning_rate_scheduler', {})
        if not scheduler_config.get('enable', False):
            return issues  # No validation needed for fixed LR
        
        scheduler_type = scheduler_config.get('type', 'fixed')
        
        # Safe parameter extraction with type validation
        def safe_validate(config, key, default, convert_type, validator=None):
            try:
                value = config.get(key, default)
                typed_value = convert_type(value) if value is not None else default
                if validator and not validator(typed_value):
                    return False, typed_value
                return True, typed_value
            except (ValueError, TypeError):
                issues.append(f"{scheduler_type}: Invalid {key} value '{value}', must be {convert_type.__name__}")
                return False, default
        
        # Type-specific validation
        if scheduler_type == 'cyclic':
            config = scheduler_config.get('cyclic', {})
            
            valid_base, base_lr = safe_validate(config, 'base_lr', 1e-5, float, lambda x: x > 0)
            valid_max, max_lr = safe_validate(config, 'max_lr', 1e-3, float, lambda x: x > 0)
            
            if valid_base and valid_max and base_lr >= max_lr:
                issues.append(f"Cyclic LR: base_lr ({base_lr}) must be less than max_lr ({max_lr})")
            
            valid_step, step_size_up = safe_validate(config, 'step_size_up', 500, int, lambda x: x > 0)
            if not valid_step:
                issues.append(f"Cyclic LR: step_size_up must be positive, got {step_size_up}")
        
        elif scheduler_type == 'reduce_on_plateau':
            config = scheduler_config.get('reduce_on_plateau', {})
            
            valid_factor, factor = safe_validate(config, 'factor', 0.5, float, lambda x: 0 < x < 1)
            if not valid_factor:
                issues.append(f"ReduceLROnPlateau: factor must be between 0 and 1, got {factor}")
            
            valid_patience, patience = safe_validate(config, 'patience', 5, int, lambda x: x > 0)
            if not valid_patience:
                issues.append(f"ReduceLROnPlateau: patience must be positive, got {patience}")
        
        elif scheduler_type == 'step_decay':
            config = scheduler_config.get('step_decay', {})
            
            valid_step, step_size = safe_validate(config, 'step_size', 10, int, lambda x: x > 0)
            if not valid_step:
                issues.append(f"StepLR: step_size must be positive, got {step_size}")
            
            valid_gamma, gamma = safe_validate(config, 'gamma', 0.1, float, lambda x: 0 < x <= 1)
            if not valid_gamma:
                issues.append(f"StepLR: gamma should be between 0 and 1, got {gamma}")
        
        elif scheduler_type == 'one_cycle':
            config = scheduler_config.get('one_cycle', {})
            
            valid_lr, max_lr = safe_validate(config, 'max_lr', 1e-3, float, lambda x: x > 0)
            if not valid_lr:
                issues.append(f"OneCycleLR: max_lr must be positive, got {max_lr}")
            
            valid_pct, pct_start = safe_validate(config, 'pct_start', 0.3, float, lambda x: 0 < x < 1)
            if not valid_pct:
                issues.append(f"OneCycleLR: pct_start must be between 0 and 1, got {pct_start}")
        
        elif scheduler_type not in ['fixed', 'exponential_decay', 'cosine_annealing']:
            issues.append(f"Unknown scheduler type: {scheduler_type}")
        
        return issues

    def print_summary(self):
        """Print a summary of key configuration parameters."""
        print("E2C Model Configuration:")
        print(f"  Method: {self.model['method']}, Latent: {self.model['latent_dim']}, Channels: {self.model['n_channels']}")
        print(f"  Wells: {self.data['num_prod']} prod, {self.data['num_inj']} inj")
        print(f"  Training: {self.training['epoch']} epochs, batch={self.training['batch_size']}, lr={self.training['learning_rate']}")
        
        # Add scheduler information
        scheduler_info = self.get_scheduler_info()
        print(f"\nLearning Rate Scheduling:")
        if scheduler_info['enabled']:
            print(f"  - Type: {scheduler_info['type']}")
            print(f"  - Requires Validation Loss: {scheduler_info['requires_validation_loss']}")
            print(f"  - Steps on Batch: {scheduler_info['step_on_batch']}")
            # Validate configuration
            issues = self.validate_scheduler_config()
            if issues:
                print(f"  - ⚠️  Configuration Issues:")
                for issue in issues:
                    print(f"      • {issue}")
        else:
            print(f"  - Type: Fixed (no scheduling)")
        
        print(f"\nLoss Configuration:")
        print(f"  - Flux Loss: {'Enabled' if self.loss.get('enable_flux_loss', False) else 'Disabled'} (λ={self.loss['lambda_flux_loss']})")
        print(f"  - BHP Loss: {'Enabled' if self.loss.get('enable_bhp_loss', False) else 'Disabled'} (λ={self.loss['lambda_bhp_loss']})")
        print(f"  - Reconstruction Loss Weight: {self.loss.get('lambda_reconstruction_loss', 1.0)}")
        print(f"  - Reconstruction Variance: {self.loss.get('reconstruction_variance', 0.1):.4f} ({'Strict' if self.loss.get('reconstruction_variance', 0.1) < 0.05 else 'Balanced' if self.loss.get('reconstruction_variance', 0.1) <= 0.2 else 'Forgiving'})")
        print(f"  - Per-Element Normalization: {'ENABLED' if self.loss.get('enable_per_element_normalization', False) else 'DISABLED'}")
        if self.loss.get('enable_per_element_normalization', False):
            print(f"    ↳ Losses normalized by degrees of freedom for balanced training")
        print(f"  - Transition Loss Weight: {self.loss['lambda_trans_loss']}")
        print(f"  - Observation Loss Weight: {self.loss['lambda_yobs_loss']}")
        print(f"\nHardware:")
        print(f"  - Device: {self.device}")
        print(f"\nPaths:")
        print(f"  - Output Dir: {self.paths['output_dir']}")
        print(f"  - Data Dir: {self.paths['data_dir']}")
        print("="*60)
    
    def __getattr__(self, name: str) -> Any:
        """Allow direct access to top-level config sections."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Configuration section '{name}' not found")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator for checking config keys."""
        return key in self.config

# Convenience function for quick config loading
def load_config(config_path: str = 'config.yaml') -> Config:
    """
    Quick config loader function.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)

# Example usage
if __name__ == "__main__":
    # Demo configuration loading
    try:
        config = Config('config.yaml')
        config.print_summary()
        
        # Example access methods
        print(f"\nExample access methods:")
        print(f"Direct: config.model['latent_dim'] = {config.model['latent_dim']}")
        print(f"Dot notation: config.get('training.learning_rate') = {config.get('training.learning_rate')}")
        print(f"Device: config.device = {config.device}")
        
    except FileNotFoundError:
        print("❌ config.yaml not found. Please create configuration file first.")
    except Exception as e:
        print(f"❌ Error loading config: {e}") 