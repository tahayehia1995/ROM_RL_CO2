"""
Reservoir Environment for RL Training
"""
import math
import torch
import numpy as np
import glob
import json
import pickle
from pathlib import Path

from .reward import reward_fun


class ReservoirEnvironment(object):
    def __init__(self, state0, config, my_rom):
        """
        Enhanced RL Environment with restricted action mapping for conservative operation
        while maintaining full E2C ROM compatibility
        
        Args:
            state0: Initial state options (can be single state or multiple Z0 options for random sampling)
            config: Configuration object
            my_rom: ROM model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Store Z0 options for random sampling (don't set self.state here - wait for reset())
        if state0 is not None:
            self.z0_options = state0.clone().to(self.device)
            # Get reference shape for noise initialization from first option
            if state0.dim() == 1:
                reference_shape = state0.unsqueeze(0).shape  # (1, latent_dim)
            elif state0.dim() == 2:
                reference_shape = state0[0:1].shape  # (1, latent_dim)
            else:
                raise ValueError(f"Unexpected state0 shape: {state0.shape}")
        else:
            # Fallback if no state0 provided
            latent_dim = config.model['latent_dim']
            self.z0_options = torch.zeros((1, latent_dim), device=self.device)
            reference_shape = (1, latent_dim)
        
        self.state = None  # Will be set during reset()
        self.config = config
        self.rom = my_rom
        
        # Episode configuration
        self.nsteps = config.rl_model['environment']['max_episode_steps']
        self.istep = 0
        
        # Well configuration
        self.num_prod = config.rl_model['reservoir']['num_producers']
        self.num_inj = config.rl_model['reservoir']['num_injectors']
        
        # RL prediction mode configuration
        prediction_mode = config.rl_model['environment']['prediction_mode']
        if prediction_mode not in ['state_based', 'latent']:
            print(f"⚠️ Invalid prediction mode '{prediction_mode}'. Using 'state_based'")
            prediction_mode = 'state_based'
        self.prediction_mode = prediction_mode
        print(f"🎯 Environment using {self.prediction_mode} prediction mode")
        
        # Noise configuration with safety defaults
        if 'environment' in config.rl_model and 'noise' in config.rl_model['environment']:
            self.noise_config = config.rl_model['environment']['noise']
        else:
            # Default noise configuration if not specified
            self.noise_config = {
                'enable': False,
                'std': 0.01
            }
            print("⚠️ Noise configuration not found in config, using defaults: disabled")
        
        # Initialize noise and time stepping using reference shape
        self.noise = torch.zeros(reference_shape).to(self.device)
        self.dt = torch.tensor(np.ones((reference_shape[0], 1)), dtype=torch.float32).to(self.device)
        
        # 🎯 CRITICAL UPDATE: Use IDENTICAL normalization parameter loading as E2C evaluation
        # Load the EXACT SAME normalization parameters that E2C evaluation uses
        self.norm_params = {}  # Will store the EXACT same structure as E2C evaluation
        self.normalization_file_loaded = False
        self.has_authentic_norm_params = False
        
        # Dashboard Action Range Configuration
        # Initialize with default ranges - will be updated with dashboard configuration
        self.restricted_action_ranges = {
            'producer_bhp': {
                'min': 1087.784912109375,  # psi - Default minimum
                'max': 1305.3419189453125   # psi - Default maximum  
            },
            'gas_injection': {
                'min': 6180072.5,          # ft³/day - Default minimum
                'max': 100646896.0          # ft³/day - Default maximum
            }
        }
        
        # Initialize with attempt to load latest normalization parameters automatically
        self._load_normalization_parameters_automatically()

    def _map_agent_action_to_rom_input(self, action_01):
        """
        🎮 CORE FUNCTION: Map agent's [0,1] actions to optimal ROM structure
        
        Optimal control order: [Producer_BHP(0-2), Gas_Injection(3-5)]
        This matches EXACTLY the structure proven optimal in corrected_model_test.py
        
        Args:
            action_01: Agent's actions in [0,1] range
            
        Returns:
            actions_for_rom: Actions in optimal order with training normalization
        """
        # Step 1: Convert agent [0,1] to restricted physical ranges
        actions_restricted = action_01.clone()
        
        # ✅ CORRECTED: Map Producer BHP (first 3 actions) to restricted range
        bhp_min = self.restricted_action_ranges['producer_bhp']['min']
        bhp_max = self.restricted_action_ranges['producer_bhp']['max']
        actions_restricted[:, 0:3] = action_01[:, 0:3] * (bhp_max - bhp_min) + bhp_min
        
        # ✅ CORRECTED: Map Gas Injection (last 3 actions) to restricted range  
        gas_min = self.restricted_action_ranges['gas_injection']['min']
        gas_max = self.restricted_action_ranges['gas_injection']['max']
        actions_restricted[:, 3:6] = action_01[:, 3:6] * (gas_max - gas_min) + gas_min
        
        # Step 2: Normalize using TRAINING-ONLY parameters for ROM compatibility
        actions_for_rom = actions_restricted.clone()
        
        # ✅ CORRECTED: Normalize Producer BHP using training parameters
        if 'BHP' in self.norm_params:
            bhp_params = self.norm_params['BHP']
            full_bhp_min = float(bhp_params['min'])
            full_bhp_max = float(bhp_params['max'])
            actions_for_rom[:, 0:3] = (actions_restricted[:, 0:3] - full_bhp_min) / (full_bhp_max - full_bhp_min)
        
        # ✅ CORRECTED: Normalize Gas Injection using training parameters
        if 'GASRATSC' in self.norm_params:
            gas_params = self.norm_params['GASRATSC']
            full_gas_min = float(gas_params['min'])
            full_gas_max = float(gas_params['max'])
            actions_for_rom[:, 3:6] = (actions_restricted[:, 3:6] - full_gas_min) / (full_gas_max - full_gas_min)
        
        return actions_for_rom

    def _map_dashboard_action_to_rom_input(self, action_01):
        """
        🎯 NEW: Map dashboard-constrained actions to ROM input
        
        Policy now outputs [0,1] where [0,1] corresponds to dashboard ranges directly.
        We need to convert to physical units using dashboard ranges, then normalize for ROM.
        
        Args:
            action_01: Agent's actions in [0,1] range (corresponding to dashboard ranges)
            
        Returns:
            actions_for_rom: Actions normalized for ROM using global training parameters
        """
        # Step 1: Convert [0,1] to dashboard physical ranges
        actions_physical = action_01.clone()
        
        # Map Producer BHP (first 3 actions) from [0,1] to dashboard BHP range
        bhp_min = self.restricted_action_ranges['producer_bhp']['min']
        bhp_max = self.restricted_action_ranges['producer_bhp']['max']
        actions_physical[:, 0:3] = action_01[:, 0:3] * (bhp_max - bhp_min) + bhp_min
        
        # Map Gas Injection (last 3 actions) from [0,1] to dashboard gas range  
        gas_min = self.restricted_action_ranges['gas_injection']['min']
        gas_max = self.restricted_action_ranges['gas_injection']['max']
        actions_physical[:, 3:6] = action_01[:, 3:6] * (gas_max - gas_min) + gas_min
        
        # Step 2: Normalize using GLOBAL training parameters for ROM compatibility
        actions_for_rom = actions_physical.clone()
        
        # Normalize Producer BHP using global training parameters
        if 'BHP' in self.norm_params:
            bhp_params = self.norm_params['BHP']
            full_bhp_min = float(bhp_params['min'])
            full_bhp_max = float(bhp_params['max'])
            actions_for_rom[:, 0:3] = (actions_physical[:, 0:3] - full_bhp_min) / (full_bhp_max - full_bhp_min)
        
        # Normalize Gas Injection using global training parameters
        if 'GASRATSC' in self.norm_params:
            gas_params = self.norm_params['GASRATSC']
            full_gas_min = float(gas_params['min'])
            full_gas_max = float(gas_params['max'])
            actions_for_rom[:, 3:6] = (actions_physical[:, 3:6] - full_gas_min) / (full_gas_max - full_gas_min)
        
        # Debug info for first few steps
        if self.istep <= 3:
            print(f"      📊 Dashboard → Physical: BHP=[{actions_physical[0,0]:.1f},{actions_physical[0,1]:.1f},{actions_physical[0,2]:.1f}] psi")
            print(f"      📊 Dashboard → Physical: Gas=[{actions_physical[0,3]:.0f},{actions_physical[0,4]:.0f},{actions_physical[0,5]:.0f}] ft³/day")
            print(f"      🔧 Physical → ROM: [{actions_for_rom.min().item():.3f}, {actions_for_rom.max().item():.3f}]")
        
        return actions_for_rom

    def _convert_dashboard_action_to_physical(self, action_01):
        """
        🎯 NEW: Convert dashboard [0,1] actions to physical units for reward calculation
        
        Args:
            action_01: Agent's actions in [0,1] range (corresponding to dashboard ranges)
            
        Returns:
            actions_physical: Actions in physical units using dashboard ranges
        """
        actions_physical = action_01.clone()
        
        # Convert Producer BHP (first 3 actions) from [0,1] to dashboard BHP range
        bhp_min = self.restricted_action_ranges['producer_bhp']['min']
        bhp_max = self.restricted_action_ranges['producer_bhp']['max']
        actions_physical[:, 0:3] = action_01[:, 0:3] * (bhp_max - bhp_min) + bhp_min
        
        # Convert Gas Injection (last 3 actions) from [0,1] to dashboard gas range  
        gas_min = self.restricted_action_ranges['gas_injection']['min']
        gas_max = self.restricted_action_ranges['gas_injection']['max']
        actions_physical[:, 3:6] = action_01[:, 3:6] * (gas_max - gas_min) + gas_min
        
        return actions_physical

    def _load_normalization_parameters_automatically(self):
        """
        Load normalization parameters using IDENTICAL method as E2C evaluation
        This ensures 100% consistency with the evaluation process
        """
        print("🔄 Loading normalization parameters using IDENTICAL E2C evaluation method...")
        
        # Try to find the EXACT same normalization files that E2C evaluation uses
        # Pattern 1: JSON files with timestamp pattern (same as E2C evaluation)
        json_files = glob.glob("normalization_parameters_*.json")
        pkl_files = glob.glob("normalization_parameters_*.pkl")
        
        # Sort by modification time to get the latest (same logic as E2C evaluation)
        json_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        pkl_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        loaded_successfully = False
        
        # Try to load the latest JSON file first (same priority as E2C evaluation)
        if json_files:
            latest_json = json_files[0]
            try:
                print(f"📖 Loading normalization parameters from: {latest_json}")
                with open(latest_json, 'r') as f:
                    norm_config = json.load(f)
                
                # Extract norm_params in EXACT same format as E2C evaluation
                self.norm_params = {}
                
                # Load spatial channel parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('spatial_channels', {}).items():
                    self.norm_params[var_name] = self._convert_strings_to_numbers(info['parameters'])
                
                # Load control variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('control_variables', {}).items():
                    self.norm_params[var_name] = self._convert_strings_to_numbers(info['parameters'])
                    
                # Load observation variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('observation_variables', {}).items():
                    self.norm_params[var_name] = self._convert_strings_to_numbers(info['parameters'])
                
                # Store the full configuration for reference (same as E2C evaluation)
                self.loaded_norm_config = norm_config
                self.normalization_file_loaded = True
                self.has_authentic_norm_params = True
                loaded_successfully = True
                
                print(f"✅ Loaded IDENTICAL normalization parameters as E2C evaluation!")
                print(f"   📊 Available parameters: {list(self.norm_params.keys())}")
                
            except Exception as e:
                print(f"❌ Error loading JSON file {latest_json}: {e}")
        
        # Try pickle file as fallback (same fallback logic as E2C evaluation)
        if not loaded_successfully and pkl_files:
            latest_pkl = pkl_files[0]
            try:
                print(f"📖 Loading normalization parameters from: {latest_pkl}")
                with open(latest_pkl, 'rb') as f:
                    norm_config = pickle.load(f)
                
                # Extract norm_params in EXACT same format as E2C evaluation
                self.norm_params = {}
                
                # Load spatial channel parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('spatial_channels', {}).items():
                    self.norm_params[var_name] = self._convert_strings_to_numbers(info['parameters'])
                
                # Load control variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('control_variables', {}).items():
                    self.norm_params[var_name] = self._convert_strings_to_numbers(info['parameters'])
                    
                # Load observation variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('observation_variables', {}).items():
                    self.norm_params[var_name] = self._convert_strings_to_numbers(info['parameters'])
                
                # Store the full configuration for reference (same as E2C evaluation)
                self.loaded_norm_config = norm_config
                self.normalization_file_loaded = True
                self.has_authentic_norm_params = True
                loaded_successfully = True
                
                print(f"✅ Loaded IDENTICAL normalization parameters as E2C evaluation!")
                print(f"   📊 Available parameters: {list(self.norm_params.keys())}")
                
            except Exception as e:
                print(f"❌ Error loading pickle file {latest_pkl}: {e}")
        
        if not loaded_successfully:
            print("❌ Could not load normalization parameters using E2C evaluation method")
            print("   🔍 Available files:")
            all_norm_files = json_files + pkl_files
            if all_norm_files:
                for file in all_norm_files:
                    print(f"      {file}")
            else:
                print("      No normalization_parameters_*.json or *.pkl files found")
            raise ValueError("❌ CRITICAL: No normalization parameters found! Run dashboard configuration first to generate parameters.")
        
        # Final validation - ensure all required parameters are available
        required_params = ['BHP', 'GASRATSC', 'WATRATSC']
        missing_params = [p for p in required_params if p not in self.norm_params]
        if missing_params:
            raise ValueError(f"❌ CRITICAL: Missing required normalization parameters: {missing_params}. Available: {list(self.norm_params.keys())}")
        
        print("✅ Environment automatically configured with:")
        print("   🎯 TRAINING-ONLY normalization parameters (NO data leakage)")
        print("   📊 Optimal structure from configuration")
        print("   🔧 Optimal action/observation mappings")
    
    def _convert_strings_to_numbers(self, params_dict):
        """
        Convert string numeric values to floats when loading from JSON
        This fixes the TypeError when JSON loads numbers as strings
        
        Args:
            params_dict: Dictionary with potentially string numeric values
            
        Returns:
            Dictionary with proper numeric values
        """
        converted_params = {}
        
        for key, value in params_dict.items():
            if isinstance(value, str):
                try:
                    # Try to convert string to float
                    converted_params[key] = float(value)
                except (ValueError, TypeError):
                    # If conversion fails, keep as string
                    converted_params[key] = value
            else:
                # Keep non-string values as-is
                converted_params[key] = value
        
        return converted_params
    
    def set_normalization_parameters(self, norm_params: dict):
        """
        Set normalization parameters using IDENTICAL format as E2C evaluation
        
        Args:
            norm_params: Dictionary with same structure as E2C evaluation uses
        """
        self.norm_params = norm_params
        self.has_authentic_norm_params = True

    # 🎯 CRITICAL UPDATE: Use IDENTICAL denormalization functions as E2C evaluation
    
    def _denormalize_observations_rom(self, yobs_normalized):
        """
        Denormalize observations using optimal ROM structure
        
        Optimal observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
        This matches EXACTLY the structure proven optimal in corrected_model_test.py
        
        Args:
            yobs_normalized: Normalized observations from ROM
            
        Returns:
            Physical observations using optimal structure and training normalization
        """
        yobs_physical = yobs_normalized.clone()
        
        # ✅ CORRECTED: Use optimal observation order from corrected_model_test.py
        # [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
        
        # Denormalize Injector BHP (first 3 observations)
        for obs_idx in range(self.num_inj):
            yobs_physical[:, obs_idx] = self._denormalize_single_observation(
                yobs_normalized[:, obs_idx], obs_idx
            )
        
        # ✅ CORRECTED: Denormalize Gas Production (next 3 observations)
        for obs_idx in range(self.num_inj, self.num_inj + self.num_prod):
            yobs_physical[:, obs_idx] = self._denormalize_single_observation(
                yobs_normalized[:, obs_idx], obs_idx
            )
        
        # ✅ CORRECTED: Denormalize Water Production (last 3 observations)
        for obs_idx in range(self.num_inj + self.num_prod, self.num_inj + self.num_prod * 2):
            yobs_physical[:, obs_idx] = self._denormalize_single_observation(
                yobs_normalized[:, obs_idx], obs_idx
            )
        
        return yobs_physical
    
    def _denormalize_single_observation(self, data, obs_idx):
        """
        Denormalize single observation using optimal ROM structure
        
        Optimal order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
        
        Args:
            data: Normalized observation data
            obs_idx: Observation index
            
        Returns:
            Denormalized data using optimal structure and training parameters
        """
        if obs_idx < 3:  # Injector BHP (0-2)
            if 'BHP' in self.norm_params:
                norm_params = self.norm_params['BHP']
                if norm_params.get('type') == 'none':
                    return data
                elif norm_params.get('type') == 'log':
                    log_min = float(norm_params['log_min']) if isinstance(norm_params['log_min'], str) else norm_params['log_min']
                    log_max = float(norm_params['log_max']) if isinstance(norm_params['log_max'], str) else norm_params['log_max']
                    log_data = data * (log_max - log_min) + log_min
                    epsilon = float(norm_params.get('epsilon', 1e-8)) if isinstance(norm_params.get('epsilon', 1e-8), str) else norm_params.get('epsilon', 1e-8)
                    data_shift = float(norm_params.get('data_shift', 0)) if isinstance(norm_params.get('data_shift', 0), str) else norm_params.get('data_shift', 0)
                    return torch.exp(log_data) - epsilon + data_shift
                else:
                    obs_min = float(norm_params['min']) if isinstance(norm_params['min'], str) else norm_params['min']
                    obs_max = float(norm_params['max']) if isinstance(norm_params['max'], str) else norm_params['max']
                    return data * (obs_max - obs_min) + obs_min
        elif obs_idx < 6:  # ✅ CORRECTED: Gas production (3-5)
            if 'GASRATSC' in self.norm_params:
                norm_params = self.norm_params['GASRATSC']
                if norm_params.get('type') == 'none':
                    return data
                elif norm_params.get('type') == 'log':
                    log_min = float(norm_params['log_min']) if isinstance(norm_params['log_min'], str) else norm_params['log_min']
                    log_max = float(norm_params['log_max']) if isinstance(norm_params['log_max'], str) else norm_params['log_max']
                    log_data = data * (log_max - log_min) + log_min
                    epsilon = float(norm_params.get('epsilon', 1e-8)) if isinstance(norm_params.get('epsilon', 1e-8), str) else norm_params.get('epsilon', 1e-8)
                    data_shift = float(norm_params.get('data_shift', 0)) if isinstance(norm_params.get('data_shift', 0), str) else norm_params.get('data_shift', 0)
                    return torch.exp(log_data) - epsilon + data_shift
                else:
                    obs_min = float(norm_params['min']) if isinstance(norm_params['min'], str) else norm_params['min']
                    obs_max = float(norm_params['max']) if isinstance(norm_params['max'], str) else norm_params['max']
                    return data * (obs_max - obs_min) + obs_min
        else:  # ✅ CORRECTED: Water production (6-8)
            if 'WATRATSC' in self.norm_params:
                norm_params = self.norm_params['WATRATSC']
                if norm_params.get('type') == 'none':
                    return data
                elif norm_params.get('type') == 'log':
                    log_min = float(norm_params['log_min']) if isinstance(norm_params['log_min'], str) else norm_params['log_min']
                    log_max = float(norm_params['log_max']) if isinstance(norm_params['log_max'], str) else norm_params['log_max']
                    log_data = data * (log_max - log_min) + log_min
                    epsilon = float(norm_params.get('epsilon', 1e-8)) if isinstance(norm_params.get('epsilon', 1e-8), str) else norm_params.get('epsilon', 1e-8)
                    data_shift = float(norm_params.get('data_shift', 0)) if isinstance(norm_params.get('data_shift', 0), str) else norm_params.get('data_shift', 0)
                    return torch.exp(log_data) - epsilon + data_shift
                else:
                    obs_min = float(norm_params['min']) if isinstance(norm_params['min'], str) else norm_params['min']
                    obs_max = float(norm_params['max']) if isinstance(norm_params['max'], str) else norm_params['max']
                    return data * (obs_max - obs_min) + obs_min
            
        # NO FALLBACKS - if we reach here, something is wrong with parameter loading
        raise ValueError(f"❌ Missing normalization parameters for observation {obs_idx}! Available params: {list(self.norm_params.keys())}")
    
    def step(self, action):
        self.istep += 1
        
        # Handle dashboard-constrained actions from policy
        # Policy outputs [0,1] where [0,1] corresponds to dashboard ranges directly
        # Convert these to physical units using dashboard ranges, then normalize for ROM
        action_restricted = self._map_dashboard_action_to_rom_input(action)
        
        if self.istep <= 3:
            # Show normalized actions for ROM input (all individual values)
            if action_restricted.shape[0] > 0:
                # Action order: [Producer_BHP(0-2), Gas_Injection(3-5)]
                producer_bhp_norm = action_restricted[0, 0:self.num_prod].detach().cpu().numpy()  # Producer BHP normalized
                gas_injection_norm = action_restricted[0, self.num_prod:self.num_prod+self.num_inj].detach().cpu().numpy()  # Gas Injection normalized
                
                # Format and print normalized actions
                bhp_norm_str = ", ".join([f"{val:.3f}" for val in producer_bhp_norm])
                gas_norm_str = ", ".join([f"{val:.3f}" for val in gas_injection_norm])
                
                print(f"   🔧 Step {self.istep}: Actions normalized for ROM input → Producer_BHP=[{bhp_norm_str}], Gas_Injection=[{gas_norm_str}]")
            
        # Store original action for reward calculation (which needs physical units)
        action_for_reward = action.clone()
        # Use restricted action for ROM prediction
        action = action_restricted
        
        # 🔬 Dual prediction mode implementation
        try:
            if self.prediction_mode == 'state_based':
                # 🎓 EXACT TRAINING DASHBOARD METHOD: Use rom.predict() with spatial states
                # This is the SAME method used in the training dashboard that shows excellent results
                
                if not hasattr(self, 'current_spatial_state'):
                    # If we don't have spatial state, decode current latent first
                    self.current_spatial_state = self.rom.model.decoder(self.state)
                
                # Create dummy observation (training dashboard uses ground truth, we'll use zeros)
                # Structure: [Injector_BHP(3), Gas_production(3), Water_production(3)] = 9 observations
                dummy_obs = torch.zeros(self.current_spatial_state.shape[0], 9).to(self.device)
                
                # 🎓 EXACT TRAINING DASHBOARD INPUTS: (spatial_state, controls, observations, dt)
                inputs = (self.current_spatial_state, action, dummy_obs, self.dt)
                
                
                next_spatial_state, yobs = self.rom.predict(inputs)
                
                # Update spatial state for next iteration
                self.current_spatial_state = next_spatial_state
                
                # Encode next spatial state to latent (for RL state representation)
                with torch.no_grad():
                    self.state = self.rom.model.encoder(next_spatial_state)
                
                # Debug logging for first few steps
                if self.istep <= 3:
                    print(f"   🎓 Step {self.istep}: Using EXACT training dashboard method (rom.predict)")
                
            else:
                # Latent-based prediction mode: Pure latent evolution
                self.state, yobs = self.rom.predict_latent(self.state, self.dt, action)


            
            # Check for NaN outputs
            if torch.isnan(self.state).any() or torch.isnan(yobs).any():
                print("⚠️ ROM predicted NaN values")
                
        except Exception as e:
            print(f"❌ ROM prediction failed: {e}")
            raise

        # 📊 Apply ROM-based observation denormalization: [Injector_BHP(3), Producer_Gas(3), Producer_Water(3)]
        # 🎯 NO CONSTRAINTS - Show RAW ROM predictions to understand the actual output
        
        # Always apply ROM normalization parameters - NO FALLBACKS
        # 📊 Show RAW ROM outputs without any modification
        yobs_original = yobs.clone()
        
        # Print normalized observations from ROM (before denormalization)
        if self.istep <= 3:
            # Observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
            if yobs_original.shape[0] > 0:
                # Extract normalized observation values
                injector_bhp_norm = yobs_original[0, 0:self.num_inj].detach().cpu().numpy()  # Injector BHP normalized
                gas_production_norm = yobs_original[0, self.num_inj:self.num_inj+self.num_prod].detach().cpu().numpy()  # Gas Production normalized
                water_production_norm = yobs_original[0, self.num_inj+self.num_prod:self.num_inj+self.num_prod*2].detach().cpu().numpy()  # Water Production normalized
                
                # Format and print normalized observations
                bhp_norm_str = ", ".join([f"{val:.3f}" for val in injector_bhp_norm])
                gas_norm_str = ", ".join([f"{val:.3f}" for val in gas_production_norm])
                water_norm_str = ", ".join([f"{val:.3f}" for val in water_production_norm])
                
                print(f"   🔧 Step {self.istep}: Observations normalized from ROM output → Injector_BHP=[{bhp_norm_str}], Gas_Production=[{gas_norm_str}], Water_Production=[{water_norm_str}]")
        
        # Apply ROM-based denormalization directly
        yobs_denorm = self._denormalize_observations_rom(yobs_original)
        
        # 🚫 Ensure non-negative observations: Round any negative predictions to zero
        yobs_denorm = torch.clamp(yobs_denorm, min=0.0)
        
        yobs = yobs_denorm

        # 🔬 Store last observation for scientific visualization
        self.last_observation = yobs.clone()
        
        # 📊 Print predicted observations in physical units (similar to actions)
        if self.istep <= 3:
            # Observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
            if yobs.shape[0] > 0:
                # Extract observation values
                injector_bhp = yobs[0, 0:self.num_inj].detach().cpu().numpy()  # Injector BHP (psi)
                gas_production = yobs[0, self.num_inj:self.num_inj+self.num_prod].detach().cpu().numpy()  # Gas Production (ft³/day)
                water_production = yobs[0, self.num_inj+self.num_prod:self.num_inj+self.num_prod*2].detach().cpu().numpy()  # Water Production (ft³/day)
                
                # Format and print observations
                bhp_str = ", ".join([f"{val:.1f}" for val in injector_bhp])
                gas_str = ", ".join([f"{val:.0f}" for val in gas_production])
                water_str = ", ".join([f"{val:.0f}" for val in water_production])
                
                print(f"   📊 Step {self.istep}: Predicted observations → Injector_BHP=[{bhp_str}] psi, Gas_Production=[{gas_str}] ft³/day, Water_Production=[{water_str}] ft³/day")

        # Calculate reward with physical actions (normalization parameters always available)
        try:
            # Convert actions to physical units for reward calculation
            # action_for_reward now contains [0,1] actions corresponding to dashboard ranges
            # Convert to physical units using dashboard ranges
            action_physical = self._convert_dashboard_action_to_physical(action_for_reward)
            
            # Print physical actions and observations for first few steps
            if self.istep <= 3:
                # Action order: [Producer_BHP(0-2), Gas_Injection(3-5)]
                producer_bhp = action_physical[0, 0:self.num_prod].detach().cpu().numpy()  # Producer BHP (psi)
                gas_injection = action_physical[0, self.num_prod:self.num_prod+self.num_inj].detach().cpu().numpy()  # Gas Injection (ft³/day)
                
                # Format and print actions
                bhp_str = ", ".join([f"{val:.1f}" for val in producer_bhp])
                gas_str = ", ".join([f"{val:.0f}" for val in gas_injection])
                
                print(f"   🎯 Step {self.istep}: Policy actions → Producer_BHP=[{bhp_str}] psi, Gas_Injection=[{gas_str}] ft³/day")
            
            # Use physical actions for reward calculation
            reward = reward_fun(yobs, action_physical, self.num_prod, self.num_inj, self.config)
            
        except Exception as e:
            print(f"❌ Reward calculation failed: {e}")
            raise
        
        done = self.istep == self.nsteps
        
        # Minimal logging only on completion or errors
        if done:
            print(f"Episode completed: {self.istep} steps, final reward: {reward.item():.2f}")
                
        return self.state, reward, done
    
    def reset(self, z0_options=None):
        """
        Reset environment with random initial state sampling
        
        Args:
            z0_options: Multiple Z0 options tensor (num_cases, latent_dim) for random sampling.
                       Also supports single Z0 tensor (latent_dim,) or (1, latent_dim) for compatibility.
                       If None, uses the Z0 options stored in constructor.
        
        Returns:
            z00: Selected initial state tensor with batch dimension (1, latent_dim)
        """
        self.istep = 0
        
        # Use provided z0_options or fall back to stored options from constructor
        if z0_options is None:
            z0_options = self.z0_options
        
        # Handle different Z0 input formats
        if z0_options.dim() == 1:
            # Single Z0 provided: (latent_dim,) -> (1, latent_dim)
            z00 = z0_options.unsqueeze(0)
        elif z0_options.dim() == 2:
            if z0_options.shape[0] == 1:
                # Single case provided: (1, latent_dim)
                z00 = z0_options
            else:
                # Multiple Z0 options provided: (num_cases, latent_dim) - RANDOM SAMPLING
                num_cases = z0_options.shape[0]
                
                # Random sampling: Select random case index
                random_case_idx = torch.randint(0, num_cases, (1,)).item()
                z00 = z0_options[random_case_idx:random_case_idx+1]  # Keep batch dimension
                
                # Log random sampling (only occasionally to avoid spam)
                if not hasattr(self, '_sampling_count'):
                    self._sampling_count = 0
                self._sampling_count += 1
                
                if self._sampling_count <= 5 or self._sampling_count % 20 == 0:
                    print(f"🎲 Random sampling: Selected case {random_case_idx}/{num_cases-1} for episode reset")
                    if self._sampling_count == 5:
                        print("   (Further random sampling messages will be shown every 20 episodes)")
        else:
            # Unexpected shape
            raise ValueError(f"Invalid Z0 options shape: {z0_options.shape}. Expected (latent_dim,), (1, latent_dim), or (num_cases, latent_dim)")
        
        # Apply state noise if enabled
        if self.noise_config['enable']:
            noise = self.noise.normal_(0., std=self.noise_config['std'])
            z00 = z00 + noise
        
        self.state = z00
        
        # For state-based mode, initialize spatial state from latent
        if self.prediction_mode == 'state_based':
            try:
                self.current_spatial_state = self.rom.model.decoder(z00)
                if self.istep == 0:  # Only print on first reset
                    print("🎯 State-based mode: Initialized spatial state from random Z0")
            except Exception as e:
                print(f"⚠️ Failed to decode Z0 to spatial state: {e}")
                print("   State-based mode may not work properly")
        
        return z00
    
    def sample_action(self):
        # ✅ CORRECTED: Generate random actions using consistent policy order
        # Order: [Producer_BHP(3), Gas_Injection(3)] - matches policy output
        action_bhp = torch.rand(1, self.num_prod).to(self.device)  # Producer BHP actions [0,1]
        action_rate = torch.rand(1, self.num_inj).to(self.device)  # Gas injection actions [0,1]
        action = torch.cat((action_bhp, action_rate), dim=1)  # Order: [BHP(3), Gas(3)]
        return action

    def update_action_ranges_from_dashboard(self, rl_config):
        """
        🎯 NEW: Update environment action ranges using DASHBOARD configuration
        This ensures the interactive dashboard selections are actually used!
        
        Args:
            rl_config: Dashboard configuration dictionary
        """
        print(f"🌍 Updating environment with DASHBOARD action ranges...")
        
        if not rl_config:
            print(f"   ❌ No dashboard configuration provided - using default ranges")
            return
        
        action_ranges = rl_config.get('action_ranges', {})
        if not action_ranges:
            print(f"   ❌ No action ranges in dashboard config - using default ranges")
            return
        
        # Extract BHP ranges from dashboard
        bhp_ranges = action_ranges.get('bhp', {})
        if bhp_ranges:
            bhp_mins = [ranges['min'] for ranges in bhp_ranges.values()]
            bhp_maxs = [ranges['max'] for ranges in bhp_ranges.values()]
            
            if bhp_mins and bhp_maxs:
                dashboard_bhp_min = min(bhp_mins)
                dashboard_bhp_max = max(bhp_maxs)
                
                # Update environment with DASHBOARD selections
                self.restricted_action_ranges['producer_bhp']['min'] = dashboard_bhp_min
                self.restricted_action_ranges['producer_bhp']['max'] = dashboard_bhp_max
                
                print(f"   ✅ Producer BHP updated to DASHBOARD: [{dashboard_bhp_min:.2f}, {dashboard_bhp_max:.2f}] psi")
            else:
                print(f"   ⚠️ Empty BHP ranges in dashboard config")
        else:
            print(f"   ⚠️ No BHP ranges in dashboard config")
        
        # Extract Gas Injection ranges from dashboard
        gas_ranges = action_ranges.get('gas_injection', {})
        if gas_ranges:
            gas_mins = [ranges['min'] for ranges in gas_ranges.values()]
            gas_maxs = [ranges['max'] for ranges in gas_ranges.values()]
            
            if gas_mins and gas_maxs:
                dashboard_gas_min = min(gas_mins)
                dashboard_gas_max = max(gas_maxs)
                
                # Update environment with DASHBOARD selections
                self.restricted_action_ranges['gas_injection']['min'] = dashboard_gas_min
                self.restricted_action_ranges['gas_injection']['max'] = dashboard_gas_max
                
                print(f"   ✅ Gas Injection updated to DASHBOARD: [{dashboard_gas_min:.0f}, {dashboard_gas_max:.0f}] ft³/day")
            else:
                print(f"   ⚠️ Empty gas ranges in dashboard config")
        else:
            print(f"   ⚠️ No gas injection ranges in dashboard config")
        
        print(f"   🎯 DASHBOARD ACTION RANGES APPLIED TO ENVIRONMENT!")
        print(f"   ✅ Your interactive selections are now being used for action mapping")

    def verify_dashboard_action_mapping(self, sample_actions=None):
        """Verify dashboard action mapping is working correctly"""
        
        # Use test actions if none provided
        if sample_actions is None:
            test_actions = torch.tensor([[0.0, 0.5, 1.0, 0.0, 0.5, 1.0]], device=self.device)
        else:
            test_actions = sample_actions
        
        try:
            physical_actions = self._convert_dashboard_action_to_physical(test_actions)
            
            # Check if values are within dashboard ranges
            bhp_min = self.restricted_action_ranges['producer_bhp']['min']
            bhp_max = self.restricted_action_ranges['producer_bhp']['max']
            gas_min = self.restricted_action_ranges['gas_injection']['min']
            gas_max = self.restricted_action_ranges['gas_injection']['max']
            
            bhp_physical = physical_actions[0, 0:3].detach().cpu().numpy()
            gas_physical = physical_actions[0, 3:6].detach().cpu().numpy()
            
            bhp_in_range = all(bhp_min <= val <= bhp_max for val in bhp_physical)
            gas_in_range = all(gas_min <= val <= gas_max for val in gas_physical)
            
            if not (bhp_in_range and gas_in_range):
                print("⚠️ Action mapping verification failed")
                
        except Exception as e:
            print(f"❌ Action mapping verification error: {e}")


def create_environment(state0, config, rom, rl_config=None):
    """
    Create environment with config parameters and dashboard configuration
    
    Args:
        state0: Initial state options (single state or multiple Z0 options for random sampling)
        config: Main configuration object
        rom: ROM model
        rl_config: Dashboard configuration (optional)
    
    Returns:
        ReservoirEnvironment: Configured environment instance
    """
    environment = ReservoirEnvironment(state0, config, rom)
    
    # 🎯 NEW: Update environment with dashboard configuration if provided
    if rl_config:
        print("🌍 Applying dashboard configuration to environment...")
        environment.update_action_ranges_from_dashboard(rl_config)
    else:
        print("⚠️ No dashboard configuration provided to environment")
    
    return environment
