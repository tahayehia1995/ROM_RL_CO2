"""
Test Prediction Functions
Generate predictions and prepare data for visualization
"""

import h5py
import numpy as np
import torch
import os
from testing.visualization.utils import create_visualization_dashboard
from utilities.timing import Timer, collect_prediction_metadata


def generate_test_visualization_standalone(loaded_data, my_rom, device, data_dir, num_tstep=24):
    """
    Standalone function to generate test predictions and launch visualization dashboard.
    Works independently of the dashboard object by using loaded data and selections.
    
    Args:
        loaded_data: Dictionary returned from load_processed_data() containing:
            - norm_params: Normalization parameters
            - data_selections: Data selection metadata (spatial properties, controls, observations, channel names)
            - metadata: Metadata including Nx, Ny, Nz, etc.
        my_rom: Trained ROM model
        device: PyTorch device
        data_dir: Directory containing raw data files
        num_tstep: Number of time steps for prediction
    
    Returns:
        Visualization dashboard object or None if failed
    """
    import h5py
    import os
    import numpy as np
    import torch
    
    print("🎨 Generating test visualization with loaded data selections...")
    
    # Extract required data from loaded_data
    norm_params = loaded_data.get('norm_params')
    data_selections = loaded_data.get('data_selections')
    metadata = loaded_data.get('metadata', {})
    
    if norm_params is None:
        print("❌ No normalization parameters found in loaded data!")
        return None
    
    if data_selections is None:
        print("❌ No data selection metadata found in loaded data!")
        print("   This file may have been created before selections were saved.")
        print("   Please reprocess data using Step 1 to include selection metadata.")
        return None
    
    # Extract spatial properties (prefer all_spatial_properties, fallback to selected_states)
    spatial_properties_to_load = data_selections.get('all_spatial_properties') or data_selections.get('selected_states', {})
    
    if not spatial_properties_to_load:
        print("  ❌ No spatial property configuration found!")
        return None
    
    # Extract selections
    selected_controls = data_selections.get('selected_controls', {})
    selected_observations = data_selections.get('selected_observations', {})
    training_channel_names = data_selections.get('training_channel_names', [])
    
    # Extract metadata
    Nx = metadata.get('Nx', 0)
    Ny = metadata.get('Ny', 0)
    Nz = metadata.get('Nz', 0)
    
    # Step 1: Load and normalize test data using SAME user selections
    print("🔄 Loading test data with user-selected normalization...")
    print("Note: Using the same normalization settings as training")
    
    # Load spatial data with SAME files and normalization as training
    test_spatial_data = {}
    
    for var_name, filename in spatial_properties_to_load.items():
        print(f"  📊 Loading test {var_name} from {filename}...")
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"    ⚠️ Warning: File not found: {filepath}")
            continue
            
        with h5py.File(filepath, 'r') as hf:
            raw_data = np.array(hf['data'])
        
        # Apply SAME normalization as training using stored parameters
        if var_name not in norm_params:
            print(f"    ⚠️ Warning: No normalization params for {var_name}, skipping")
            continue
            
        norm_params_var = norm_params[var_name]
        if norm_params_var.get('type') == 'log':
            # Apply log normalization using stored parameters
            epsilon = norm_params_var['epsilon']
            data_shift = norm_params_var['data_shift']
            
            data_shifted = raw_data - data_shift + epsilon
            log_data = np.log(data_shifted)
            normalized_data = (log_data - norm_params_var['log_min']) / (norm_params_var['log_max'] - norm_params_var['log_min'])
            print(f"    🔢 Applied LOG normalization to {var_name}")
        else:
            # Apply min-max normalization using stored parameters  
            normalized_data = (raw_data - norm_params_var['min']) / (norm_params_var['max'] - norm_params_var['min'])
            print(f"    📏 Applied MIN-MAX normalization to {var_name}")
            
        test_spatial_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
    
    # Load timeseries data with SAME normalization as training
    test_timeseries_data = {}
    all_timeseries_vars = set(list(selected_controls.keys()) + list(selected_observations.keys()))
    
    for var_name in all_timeseries_vars:
        filename = f"batch_timeseries_data_{var_name}.h5"
        filepath = os.path.join(data_dir, filename)
        print(f"  📈 Loading test {var_name}...")
        
        if not os.path.exists(filepath):
            print(f"    ⚠️ Warning: File not found: {filepath}")
            continue
        
        with h5py.File(filepath, 'r') as hf:
            raw_data = np.array(hf['data'])
        
        # Apply SAME normalization as training
        if var_name not in norm_params:
            print(f"    ⚠️ Warning: No normalization params for {var_name}, skipping")
            continue
            
        norm_params_var = norm_params[var_name]
        normalized_data = (raw_data - norm_params_var['min']) / (norm_params_var['max'] - norm_params_var['min'])
        test_timeseries_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
    
    # Step 2: Extract controls and observations using SAME user selections
    print("🔧 Extracting controls and observations using user selections...")
    
    # Create control tensor using SAME logic as training
    control_components = []
    for var_name, config in selected_controls.items():
        if var_name not in test_timeseries_data:
            continue
        data = test_timeseries_data[var_name]
        selected_data = data[:, :, config['wells']]
        
        for well_idx in range(selected_data.shape[2]):
            control_components.append(selected_data[:, :, well_idx])
    
    bhp_test = torch.stack(control_components, dim=2) if control_components else torch.zeros(0)
    
    # Create observation tensor using SAME logic as training  
    obs_components = []
    for var_name, config in selected_observations.items():
        if var_name not in test_timeseries_data:
            continue
        data = test_timeseries_data[var_name]
        selected_data = data[:, :, config['wells']]
        
        for well_idx in range(selected_data.shape[2]):
            obs_components.append(selected_data[:, :, well_idx])
    
    yobs_test = torch.stack(obs_components, dim=2) if obs_components else torch.zeros(0)
    
    print(f"📊 Test control data shape: {bhp_test.shape}")
    print(f"📊 Test observation data shape: {yobs_test.shape}")
    
    # Step 3: Organize spatial data in SAME channel order as training
    print("🏗️ Organizing spatial data in training channel order...")
    
    # Get spatial data in the same order as training channels
    spatial_channels = []
    channel_names = []
    
    # Use training_channel_names if available (new structure), otherwise use spatial_properties_to_load keys (old structure)
    if training_channel_names:
        for var_name in training_channel_names:
            if var_name in test_spatial_data:
                spatial_channels.append(test_spatial_data[var_name])
                channel_names.append(var_name)
            else:
                print(f"  ⚠️ Warning: Training channel '{var_name}' not found in test data")
    else:
        # Fallback: use all available spatial properties
        for var_name in spatial_properties_to_load.keys():
            if var_name in test_spatial_data:
                spatial_channels.append(test_spatial_data[var_name])
                channel_names.append(var_name)
    
    # Stack into state tensor: (n_sample, timesteps, channels, Nx, Ny, Nz)
    if spatial_channels:
        state_test = torch.stack(spatial_channels, dim=2)
        n_sample, timesteps, n_channels, Nx, Ny, Nz = state_test.shape
        print(f"✅ Test state tensor: {state_test.shape}")
    else:
        print("❌ No spatial data available")
        return None
    
    # Step 4: Generate test cases and run predictions
    print("🎯 Setting up test cases...")
    
    available_cases = n_sample  # Use ALL available cases automatically
    test_case_indices = np.arange(available_cases)
    num_case = len(test_case_indices)
    
    print(f"📊 Processing {num_case} test cases with {num_tstep} time steps")
    
    # Initialize prediction arrays
    state_pred = torch.zeros((num_case, num_tstep, n_channels, Nx, Ny, Nz), dtype=torch.float32).to(device)
    yobs_pred = torch.zeros((num_case, num_tstep, yobs_test.shape[2]), dtype=torch.float32).to(device)
    
    # Time step configuration
    t_steps = np.arange(0, 200, 200//num_tstep)
    dt = 10
    t_steps1 = (t_steps + dt).astype(int)
    indt_del = t_steps1 - t_steps
    indt_del = indt_del / max(indt_del)
    
    tmp = np.array(range(num_tstep)) - 1
    tmp1 = np.array(range(num_tstep))
    tmp[0] = 0
    
    # Prepare control and observation sequences
    bhp_tt1 = bhp_test[:, tmp1, :]
    bhp_t = torch.swapaxes(bhp_tt1, 1, 2).to(device)
    bhp_seq = bhp_t[test_case_indices, :, :]
    
    yobs_t_seq = torch.swapaxes(yobs_test[test_case_indices, ...], 1, 2).to(device)
    
    # Initial state preparation - use first timestep, rearrange to (batch, channels, X, Y, Z)
    initial_state = state_test[test_case_indices, 0, :, :, :, :].to(device)  # (num_case, channels, Nx, Ny, Nz)
    state_t_seq = initial_state
    
    print(f"🔍 Sequential Prediction Setup:")
    print(f"Initial state shape: {state_t_seq.shape}")
    print(f"Control sequence shape: {bhp_seq.shape}")
    print(f"Observation sequence shape: {yobs_t_seq.shape}")
    
    # Step 5: Run sequential predictions
    print(f"\n🚀 Running sequential predictions...")
    
    # Time the prediction phase
    with Timer("prediction", log_dir='./timing_logs/') as timer:
        for i_tstep in range(num_tstep):
            # Store current state prediction
            state_pred[:, i_tstep, ...] = state_t_seq
            
            # Time step for current iteration
            dt_seq = torch.tensor(np.ones((num_case, 1)) * indt_del[i_tstep], dtype=torch.float32).to(device)
            
            # Prepare inputs for model
            inputs = (state_t_seq, bhp_seq[:, :, i_tstep], yobs_t_seq[:, :, i_tstep], dt_seq)
            
            # Predict next state
            state_t1_seq, yobs_t1_seq = my_rom.predict(inputs)
            
            # Update state for next iteration
            state_t_seq = state_t1_seq
            
            # Store well output predictions
            yobs_pred[:, i_tstep, :] = yobs_t1_seq
            
            # Progress indicator
            if (i_tstep + 1) % 5 == 0:
                print(f"  Step {i_tstep + 1}/{num_tstep} completed")
        
        # Calculate average time per case (get current elapsed time)
        total_time = timer.get_elapsed()
        avg_time_per_case = total_time / num_case if num_case > 0 else 0.0
        
        # Collect metadata for timing log
        model_info = {}
        if hasattr(my_rom, 'config') and my_rom.config:
            if hasattr(my_rom.config, 'model'):
                model_info['model_method'] = my_rom.config.model.get('method', 'N/A')
                model_info['latent_dimension'] = my_rom.config.model.get('latent_dim', 'N/A')
            else:
                model_info['model_method'] = 'N/A'
                model_info['latent_dimension'] = 'N/A'
        else:
            model_info['model_method'] = 'N/A'
            model_info['latent_dimension'] = 'N/A'
        metadata = collect_prediction_metadata(num_case, num_tstep, str(device), model_info)
        metadata['average_time_per_case'] = avg_time_per_case
        timer.metadata = metadata
    
    # Step 6: Prepare data for visualization
    print("🎨 Preparing visualization data...")
    
    # Get true sequences for comparison - rearrange to match prediction format
    state_seq_true = torch.zeros((num_case, n_channels, timesteps, Nx, Ny, Nz))
    for i, var_name in enumerate(channel_names):
        state_seq_true[:, i, :, :, :, :] = test_spatial_data[var_name][test_case_indices, ...]
    
    # Align time dimensions
    state_seq_true_aligned = state_seq_true[:, :, :num_tstep, :, :, :]
    
    print(f"📊 Final shapes:")
    print(f"Predicted state: {state_pred.shape}")
    print(f"True state: {state_seq_true_aligned.shape}")
    print(f"Predicted observations: {yobs_pred.shape}")
    
    # Step 7: Generate training predictions if training data is available
    # Training predictions should always be generated upfront (like test predictions) when data is available
    train_state_pred = None
    train_state_seq_true_aligned = None
    train_yobs_pred = None
    train_yobs_seq_true = None
    train_case_indices = None
    
    STATE_train = loaded_data.get('STATE_train')
    BHP_train = loaded_data.get('BHP_train')
    Yobs_train = loaded_data.get('Yobs_train')
    num_train = metadata.get('num_train', 0)
    
    # Ensure training predictions are generated whenever training data is available
    # This matches the behavior of test predictions - always generated upfront
    if STATE_train and len(STATE_train) > 0 and num_train > 0:
        print("\n🎓 Generating training predictions...")
        
        # Organize training data: STATE_train is a list of tensors, one per timestep
        # Each tensor has shape (num_train, channels, Nx, Ny, Nz)
        # We need to stack them into (num_train, timesteps, channels, Nx, Ny, Nz)
        num_train_timesteps = len(STATE_train)
        train_state_data = torch.stack(STATE_train, dim=1)  # (num_train, timesteps, channels, Nx, Ny, Nz)
        
        # Get BHP and Yobs data similarly
        if BHP_train and len(BHP_train) > 0:
            train_bhp_data = torch.stack(BHP_train, dim=1)  # (num_train, timesteps, n_controls)
        else:
            train_bhp_data = None
            
        if Yobs_train and len(Yobs_train) > 0:
            train_yobs_data = torch.stack(Yobs_train, dim=1)  # (num_train, timesteps, n_observations)
        else:
            train_yobs_data = None
        
        # Use all training cases
        train_case_indices = np.arange(num_train)
        num_train_case = len(train_case_indices)
        
        print(f"📊 Processing {num_train_case} training cases with {num_tstep} time steps")
        
        # Initialize training prediction arrays
        train_state_pred = torch.zeros((num_train_case, num_tstep, n_channels, Nx, Ny, Nz), dtype=torch.float32).to(device)
        if train_yobs_data is not None:
            train_yobs_pred = torch.zeros((num_train_case, num_tstep, train_yobs_data.shape[2]), dtype=torch.float32).to(device)
        else:
            train_yobs_pred = torch.zeros((num_train_case, num_tstep, yobs_test.shape[2]), dtype=torch.float32).to(device)
        
        # Prepare control and observation sequences for training
        if train_bhp_data is not None:
            train_bhp_tt1 = train_bhp_data[:, tmp1, :]
            train_bhp_t = torch.swapaxes(train_bhp_tt1, 1, 2).to(device)
            train_bhp_seq = train_bhp_t[train_case_indices, :, :]
        else:
            # Fallback: use zeros if BHP not available - use shape from test BHP as reference
            train_bhp_seq = torch.zeros((num_train_case, bhp_seq.shape[1], bhp_seq.shape[2]), dtype=torch.float32).to(device)
        
        if train_yobs_data is not None:
            train_yobs_t_seq = torch.swapaxes(train_yobs_data[train_case_indices, ...], 1, 2).to(device)
        else:
            # Fallback: use zeros if Yobs not available
            train_yobs_t_seq = torch.zeros((num_train_case, yobs_t_seq.shape[1], yobs_t_seq.shape[2]), dtype=torch.float32).to(device)
        
        # Initial state preparation - use first timestep
        train_initial_state = train_state_data[train_case_indices, 0, :, :, :, :].to(device)  # (num_train_case, channels, Nx, Ny, Nz)
        train_state_t_seq = train_initial_state
        
        print(f"🔍 Training Sequential Prediction Setup:")
        print(f"Initial state shape: {train_state_t_seq.shape}")
        print(f"Control sequence shape: {train_bhp_seq.shape}")
        print(f"Observation sequence shape: {train_yobs_t_seq.shape}")
        
        # Run sequential predictions for training cases
        print(f"\n🚀 Running sequential predictions for training cases...")
        with Timer("training_prediction", log_dir='./timing_logs/') as train_timer:
            for i_tstep in range(num_tstep):
                # Store current state prediction
                train_state_pred[:, i_tstep, ...] = train_state_t_seq
                
                # Time step for current iteration
                dt_seq = torch.tensor(np.ones((num_train_case, 1)) * indt_del[i_tstep], dtype=torch.float32).to(device)
                
                # Prepare inputs for model
                inputs = (train_state_t_seq, train_bhp_seq[:, :, i_tstep], train_yobs_t_seq[:, :, i_tstep], dt_seq)
                
                # Predict next state
                train_state_t1_seq, train_yobs_t1_seq = my_rom.predict(inputs)
                
                # Update state for next iteration
                train_state_t_seq = train_state_t1_seq
                
                # Store well output predictions
                train_yobs_pred[:, i_tstep, :] = train_yobs_t1_seq
                
                # Progress indicator
                if (i_tstep + 1) % 5 == 0:
                    print(f"  Training Step {i_tstep + 1}/{num_tstep} completed")
        
        # Prepare training true sequences for comparison
        train_state_seq_true = torch.zeros((num_train_case, n_channels, num_train_timesteps, Nx, Ny, Nz))
        for i in range(n_channels):
            train_state_seq_true[:, i, :, :, :, :] = train_state_data[train_case_indices, :, i, :, :, :]
        
        # Align time dimensions
        train_state_seq_true_aligned = train_state_seq_true[:, :, :num_tstep, :, :, :]
        
        print(f"📊 Training prediction shapes:")
        print(f"Predicted state: {train_state_pred.shape}")
        print(f"True state: {train_state_seq_true_aligned.shape}")
        print(f"Predicted observations: {train_yobs_pred.shape}")
        print("✅ Training predictions generated successfully!")
    else:
        # Training data is not available - this is expected if only test data was loaded
        # Training predictions will be None, and the dashboard will skip training metrics gracefully
        print("⚠️ Training data not available - training predictions will not be generated")
        if not STATE_train or len(STATE_train) == 0:
            print("   Reason: STATE_train is missing or empty in loaded_data")
        elif num_train == 0:
            print(f"   Reason: num_train is 0 (metadata indicates no training cases)")
        print("   Note: Training metrics will be skipped in the dashboard (this is expected behavior)")
    
    # Step 8: Launch visualization dashboard
    print("\n🚀 Launching Interactive Visualization Dashboard...")
    
    visualization_dashboard = create_visualization_dashboard(
        state_pred=state_pred,
        state_seq_true_aligned=state_seq_true_aligned,
        yobs_pred=yobs_pred,
        yobs_seq_true=yobs_t_seq,
        test_case_indices=test_case_indices,
        norm_params=norm_params,
        Nx=Nx, Ny=Ny, Nz=Nz,
        num_tstep=num_tstep,
        channel_names=channel_names,  # Pass the channel names for visualization
        my_rom=my_rom,  # Pass ROM model for comparison predictions
        test_controls=bhp_seq,  # Pass test controls for comparison predictions
        test_observations=yobs_t_seq,  # Pass test observations for comparison predictions
        device=device,  # Pass device for computation
        train_state_pred=train_state_pred,  # Pass training predictions
        train_state_seq_true_aligned=train_state_seq_true_aligned,
        train_yobs_pred=train_yobs_pred,
        train_yobs_seq_true=train_yobs_t_seq if train_state_pred is not None else None,
        train_case_indices=train_case_indices,
        loaded_data=loaded_data  # Pass loaded_data for on-demand training prediction generation
    )
    
    print("Test visualization completed")
    
    return visualization_dashboard


