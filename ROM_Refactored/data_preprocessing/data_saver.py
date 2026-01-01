"""
Data saving functions for processed data and normalization parameters
Handles saving to H5 and JSON formats
"""

import h5py
import json
import os
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional


def save_normalization_parameters(norm_params: Dict[str, Any], data_dir: str, nsteps: int,
                                  all_spatial_properties: Optional[Dict] = None,
                                  selected_training_channels: Optional[List] = None,
                                  selected_controls: Optional[Dict] = None,
                                  selected_observations: Optional[Dict] = None,
                                  normalization_preferences: Optional[Dict] = None,
                                  control_normalization_preferences: Optional[Dict] = None,
                                  observation_normalization_preferences: Optional[Dict] = None,
                                  training_channel_names: Optional[List] = None,
                                  training_channel_mapping: Optional[Dict] = None,
                                  save_dir: str = './processed_data/') -> str:
    """
    Save normalization parameters to a JSON file for later use.
    
    Args:
        norm_params: Dictionary of normalization parameters
        data_dir: Original data directory
        nsteps: Number of time steps
        all_spatial_properties: Dictionary of all spatial properties
        selected_training_channels: List of selected training channel names
        selected_controls: Dictionary of selected control configurations
        selected_observations: Dictionary of selected observation configurations
        normalization_preferences: Normalization preferences for spatial channels
        control_normalization_preferences: Normalization preferences for controls
        observation_normalization_preferences: Normalization preferences for observations
        training_channel_names: List of training channel names in order
        training_channel_mapping: Mapping of training channels
        save_dir: Directory to save the file
        
    Returns:
        Path to saved JSON file
    """
    print("Saving normalization parameters...")
    
    # Create comprehensive normalization configuration
    norm_config = {
        'metadata': {
            'created_timestamp': datetime.now().isoformat(),
            'data_directory': data_dir,
            'n_steps': nsteps,
            'total_channels': len(all_spatial_properties) if all_spatial_properties else 0,
            'selected_channels': len(selected_training_channels) if selected_training_channels else 0,
            'total_controls': sum(len(config['wells']) for config in selected_controls.values()) if selected_controls else 0,
            'total_observations': sum(len(config['wells']) for config in selected_observations.values()) if selected_observations else 0
        },
        'spatial_channels': {},
        'control_variables': {},
        'observation_variables': {},
        'channel_mapping': {},
        'selection_summary': {}
    }
    
    # Store spatial channel normalization parameters
    if all_spatial_properties:
        for var_name, filename in all_spatial_properties.items():
            if var_name in norm_params:
                norm_config['spatial_channels'][var_name] = {
                    'filename': filename,
                    'normalization_type': normalization_preferences.get(var_name, 'minmax') if normalization_preferences else 'minmax',
                    'parameters': norm_params[var_name],
                    'selected_for_training': var_name in selected_training_channels if selected_training_channels else False
                }
                
                # Add training position if selected
                if training_channel_names and var_name in training_channel_names:
                    training_position = training_channel_names.index(var_name)
                    norm_config['spatial_channels'][var_name]['training_position'] = training_position
    
    # Store control variable normalization parameters
    if selected_controls:
        for var_name, config in selected_controls.items():
            if var_name in norm_params:
                norm_config['control_variables'][var_name] = {
                    'filename': config.get('filename', ''),
                    'selected_wells': config.get('wells', []),
                    'normalization_type': control_normalization_preferences.get(var_name, 'minmax') if control_normalization_preferences else 'minmax',
                    'parameters': norm_params[var_name],
                    'variable_type': 'control'
                }
    
    # Store observation variable normalization parameters
    if selected_observations:
        for var_name, config in selected_observations.items():
            if var_name in norm_params:
                norm_config['observation_variables'][var_name] = {
                    'filename': config.get('filename', ''),
                    'selected_wells': config.get('wells', []),
                    'normalization_type': observation_normalization_preferences.get(var_name, 'minmax') if observation_normalization_preferences else 'minmax',
                    'parameters': norm_params[var_name],
                    'variable_type': 'observation'
                }
    
    # Store channel mapping for training tensor reconstruction
    if training_channel_mapping:
        norm_config['channel_mapping'] = training_channel_mapping
    
    # Store selection summary
    norm_config['selection_summary'] = {
        'training_channels': list(selected_training_channels) if selected_training_channels else [],
        'control_wells_by_variable': {var: config.get('wells', []) for var, config in selected_controls.items()} if selected_controls else {},
        'observation_wells_by_variable': {var: config.get('wells', []) for var, config in selected_observations.items()} if selected_observations else {}
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON
    json_filename = f"normalization_parameters_{timestamp}.json"
    json_filepath = os.path.join(save_dir, json_filename)
    
    try:
        with open(json_filepath, 'w') as f:
            json.dump(norm_config, f, indent=2, default=str)
        print(f"Normalization parameters saved to: {json_filepath}")
        return json_filepath
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        raise


def save_processed_data(STATE_train: List[torch.Tensor], STATE_eval: List[torch.Tensor],
                       BHP_train: List[torch.Tensor], BHP_eval: List[torch.Tensor],
                       Yobs_train: List[torch.Tensor], Yobs_eval: List[torch.Tensor],
                       nsteps: int, n_channels: int, num_well: int, num_prod: int, num_inj: int,
                       Nx: int, Ny: int, Nz: int, data_dir: str,
                       norm_params: Optional[Dict] = None,
                       data_selections: Optional[Dict] = None,
                       save_dir: str = './processed_data/') -> str:
    """
    Save all processed data (states, controls, observations) to H5 file.
    
    Args:
        STATE_train, STATE_eval: Lists of state tensors
        BHP_train, BHP_eval: Lists of control tensors
        Yobs_train, Yobs_eval: Lists of observation tensors
        nsteps: Number of time steps
        n_channels: Number of channels
        num_well, num_prod, num_inj: Well configuration
        Nx, Ny, Nz: Spatial dimensions
        data_dir: Original data directory
        norm_params: Normalization parameters dictionary
        data_selections: Data selection metadata
        save_dir: Directory to save the file
        
    Returns:
        Path to saved H5 file
    """
    print("💾 Saving processed data to .h5 file...")
    
    # Calculate metadata for filename
    num_train = STATE_train[0].shape[0] if STATE_train else 0
    num_eval = STATE_eval[0].shape[0] if STATE_eval else 0
    num_states = len(STATE_train) if STATE_train else 0
    
    if BHP_train and len(BHP_train) > 0:
        num_controls = BHP_train[0].shape[1]
    else:
        num_controls = 0
    
    if Yobs_train and len(Yobs_train) > 0:
        num_observations = Yobs_train[0].shape[1]
    else:
        num_observations = 0
    
    # Create filename with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (f"processed_data_nstates{num_states}_ncontrols{num_controls}_nobs{num_observations}_"
                f"nsteps{nsteps}_ntrain{num_train}_neval{num_eval}_ch{n_channels}_wells{num_well}_{timestamp}.h5")
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with h5py.File(filepath, 'w') as hf:
            # Save metadata
            metadata_group = hf.create_group('metadata')
            metadata_group.attrs['num_train'] = num_train
            metadata_group.attrs['num_eval'] = num_eval
            metadata_group.attrs['num_states'] = num_states
            metadata_group.attrs['num_controls'] = num_controls
            metadata_group.attrs['num_observations'] = num_observations
            metadata_group.attrs['nsteps'] = nsteps
            metadata_group.attrs['n_channels'] = n_channels
            metadata_group.attrs['num_well'] = num_well
            metadata_group.attrs['num_prod'] = num_prod
            metadata_group.attrs['num_inj'] = num_inj
            metadata_group.attrs['Nx'] = Nx
            metadata_group.attrs['Ny'] = Ny
            metadata_group.attrs['Nz'] = Nz
            metadata_group.attrs['created_timestamp'] = timestamp
            metadata_group.attrs['data_dir'] = data_dir
            
            # Save training data
            train_group = hf.create_group('train')
            
            # Save STATE_train (list of tensors)
            state_train_group = train_group.create_group('STATE')
            for i, state_tensor in enumerate(STATE_train):
                state_data = state_tensor.cpu().numpy() if hasattr(state_tensor, 'cpu') else state_tensor
                state_train_group.create_dataset(f'step_{i}', data=state_data)
            
            # Save BHP_train (list of tensors)
            if BHP_train:
                bhp_train_group = train_group.create_group('BHP')
                for i, bhp_tensor in enumerate(BHP_train):
                    bhp_data = bhp_tensor.cpu().numpy() if hasattr(bhp_tensor, 'cpu') else bhp_tensor
                    bhp_train_group.create_dataset(f'step_{i}', data=bhp_data)
            
            # Save Yobs_train (list of tensors)
            if Yobs_train:
                yobs_train_group = train_group.create_group('Yobs')
                for i, yobs_tensor in enumerate(Yobs_train):
                    yobs_data = yobs_tensor.cpu().numpy() if hasattr(yobs_tensor, 'cpu') else yobs_tensor
                    yobs_train_group.create_dataset(f'step_{i}', data=yobs_data)
            
            # Save dt_train
            dt_train = torch.tensor(np.ones((num_train, 1)), dtype=torch.float32).to(device)
            train_group.create_dataset('dt', data=dt_train.cpu().numpy())
            
            # Save evaluation data
            eval_group = hf.create_group('eval')
            
            # Save STATE_eval (list of tensors)
            state_eval_group = eval_group.create_group('STATE')
            for i, state_tensor in enumerate(STATE_eval):
                state_data = state_tensor.cpu().numpy() if hasattr(state_tensor, 'cpu') else state_tensor
                state_eval_group.create_dataset(f'step_{i}', data=state_data)
            
            # Save BHP_eval (list of tensors)
            if BHP_eval:
                bhp_eval_group = eval_group.create_group('BHP')
                for i, bhp_tensor in enumerate(BHP_eval):
                    bhp_data = bhp_tensor.cpu().numpy() if hasattr(bhp_tensor, 'cpu') else bhp_tensor
                    bhp_eval_group.create_dataset(f'step_{i}', data=bhp_data)
            
            # Save Yobs_eval (list of tensors)
            if Yobs_eval:
                yobs_eval_group = eval_group.create_group('Yobs')
                for i, yobs_tensor in enumerate(Yobs_eval):
                    yobs_data = yobs_tensor.cpu().numpy() if hasattr(yobs_tensor, 'cpu') else yobs_tensor
                    yobs_eval_group.create_dataset(f'step_{i}', data=yobs_data)
            
            # Save dt_eval
            dt_eval = torch.tensor(np.ones((num_eval, 1)), dtype=torch.float32).to(device)
            eval_group.create_dataset('dt', data=dt_eval.cpu().numpy())
            
            # Save normalization parameters if provided
            if norm_params:
                norm_group = hf.create_group('normalization')
                norm_group.attrs['params_json'] = json.dumps(norm_params, default=str)
            
            # Save data selections if provided
            if data_selections:
                selections_group = hf.create_group('data_selections')
                selections_group.attrs['selections_json'] = json.dumps(data_selections, default=str)
        
        print(f"✅ Processed data saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving processed data: {e}")
        import traceback
        traceback.print_exc()
        raise

