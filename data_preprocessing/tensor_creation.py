"""
Tensor creation and sliding window functions for data preprocessing
Handles sliding window application and train/eval split
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def apply_sliding_window(state_tensor: np.ndarray, control_tensor: Optional[np.ndarray], 
                        observation_tensor: Optional[np.ndarray], nsteps: int) -> Tuple:
    """
    Apply sliding window to state, control, and observation tensors.
    
    Args:
        state_tensor: State tensor with shape (n_sample, timesteps, n_channels, Nx, Ny, Nz)
        control_tensor: Control tensor with shape (n_sample, timesteps, n_controls) or None
        observation_tensor: Observation tensor with shape (n_sample, timesteps, n_observations) or None
        nsteps: Number of time steps for sliding window
        
    Returns:
        Tuple of (channel_data_slt, BHP_slt, Yobs_slt, num_t_slt, Nx, Ny, Nz, n_channels)
    """
    n_sample, steps_slt, n_channels, Nx, Ny, Nz = state_tensor.shape
    
    # Create sliding window indices
    indt = np.array(range(0, steps_slt - (nsteps - 1)))
    
    # Apply sliding window to all tensors
    # Create dynamic channel lists based on selected channels
    channel_data_slt = []  # List of lists for each channel
    for i in range(n_channels):
        channel_data_slt.append([])
    
    # Legacy compatibility names (for backward compatibility with existing code)
    SW_slt = []
    SG_slt = [] 
    PRES_slt = []
    BHP_slt = []
    Yobs_slt = []
    
    # Track if we have any data for each legacy channel
    has_sw = False
    has_sg = False
    has_pres = False
    
    for k in range(nsteps):
        indt_k = indt + k
        
        # State data (split by channels dynamically)
        for channel_idx in range(n_channels):
            channel_data = state_tensor[:, indt_k, channel_idx, :, :, :]
            channel_data_slt[channel_idx].append(channel_data)
        
        # Control and observation data for prediction steps
        if k < nsteps - 1:
            if control_tensor is not None:
                ctrl_indt_k = np.minimum(indt_k, control_tensor.shape[1] - 1)
                BHP_slt.append(control_tensor[:, ctrl_indt_k, :])
            
            if observation_tensor is not None:
                obs_indt_k = np.minimum(indt_k, observation_tensor.shape[1] - 1)
                Yobs_slt.append(observation_tensor[:, obs_indt_k, :])
    
    num_t_slt = len(indt)
    
    return channel_data_slt, BHP_slt, Yobs_slt, num_t_slt, Nx, Ny, Nz, n_channels


def train_split_data(SW_slt: List, SG_slt: List, PRES_slt: List, BHP_slt: List, 
                    Yobs_slt: List, num_t_slt: int, Nx: int, Ny: int, Nz: int, 
                    num_well: int, num_prod: int, num_inj: int, n_channels: int, 
                    device: torch.device, channel_data_slt: Optional[List] = None) -> Tuple:
    """
    Split data into training and evaluation sets for 3D CNN processing.
    
    Args:
        SW_slt, SG_slt, PRES_slt: Legacy channel data lists (for backward compatibility)
        BHP_slt: Control data list
        Yobs_slt: Observation data list
        num_t_slt: Number of time steps after sliding window
        Nx, Ny, Nz: Spatial dimensions
        num_well, num_prod, num_inj: Well configuration
        n_channels: Number of channels
        device: PyTorch device
        channel_data_slt: Dynamic channel data list (preferred over legacy channels)
        
    Returns:
        Tuple of (STATE_train, BHP_train, Yobs_train, STATE_eval, BHP_eval, Yobs_eval)
    """
    # Check if we have any data in the legacy channels
    if not SW_slt or not SG_slt or not PRES_slt:
        if not channel_data_slt or len(channel_data_slt) == 0:
            print("❌ No channel data available - cannot proceed with training")
            return None, None, None, None, None, None
    
    # Get the number of cases from the first available data
    if len(SW_slt) > 0:
        num_all = SW_slt[0].shape[0]
    elif len(SG_slt) > 0:
        num_all = SG_slt[0].shape[0]
    elif len(PRES_slt) > 0:
        num_all = PRES_slt[0].shape[0]
    elif channel_data_slt and len(channel_data_slt) > 0 and len(channel_data_slt[0]) > 0:
        num_all = channel_data_slt[0][0].shape[0]
    else:
        print("❌ No data available to determine number of cases")
        return None, None, None, None, None, None
    
    split_ratio = int(num_all / 100)  # How many sets of 100 cases
    num_run_per_case = 75  # 75% for training
    num_run_eval = 100 - num_run_per_case  # 25% for evaluation
    
    # Adjust for actual number of cases if less than 100
    if num_all < 100:
        split_ratio = 1
        actual_train = int(num_all * 0.75)  # 75% of available cases
        actual_eval = num_all - actual_train  # Remaining 25%
    else:
        actual_train = num_run_per_case * split_ratio
        actual_eval = num_run_eval * split_ratio
    
    print(f"📊 Total cases: {num_all}")
    print(f"📊 Actual training cases: {actual_train}")
    print(f"📊 Actual evaluation cases: {actual_eval}")
    
    # Initialize arrays for training and evaluation data
    sw_t_train = np.zeros((actual_train, num_t_slt, Nx, Ny, Nz))
    sg_t_train = np.zeros((actual_train, num_t_slt, Nx, Ny, Nz))
    pres_t_train = np.zeros((actual_train, num_t_slt, Nx, Ny, Nz))
    # BHP_slt[0] has shape (n_cases, n_timesteps, n_controls), so use shape[2] for controls
    bhp_t_train = np.zeros((actual_train, num_t_slt, BHP_slt[0].shape[2])) if BHP_slt and len(BHP_slt) > 0 else None
    yobs_t_train = np.zeros((actual_train, num_t_slt, Yobs_slt[0].shape[2])) if Yobs_slt and len(Yobs_slt) > 0 else None

    sw_t_eval = np.zeros((actual_eval, num_t_slt, Nx, Ny, Nz))
    sg_t_eval = np.zeros((actual_eval, num_t_slt, Nx, Ny, Nz))
    pres_t_eval = np.zeros((actual_eval, num_t_slt, Nx, Ny, Nz))
    bhp_t_eval = np.zeros((actual_eval, num_t_slt, BHP_slt[0].shape[2])) if BHP_slt and len(BHP_slt) > 0 else None
    yobs_t_eval = np.zeros((actual_eval, num_t_slt, Yobs_slt[0].shape[2])) if Yobs_slt and len(Yobs_slt) > 0 else None
    
    # Create shuffling indices
    num_train_samples = actual_train * num_t_slt
    shuffle_ind_train = np.random.default_rng(seed=1010).permutation(num_train_samples)
    num_eval_samples = actual_eval * num_t_slt
    shuffle_ind_eval = np.random.default_rng(seed=1010).permutation(num_eval_samples)
    
    STATE_train = []
    BHP_train = []
    Yobs_train = []
    STATE_eval = []
    BHP_eval = []
    Yobs_eval = []
    
    print(f"Processing {len(SW_slt)} time steps...")
    
    for i_step in range(len(SW_slt)):
        # Split data for this time step
        if num_all < 100:
            # Simple split for smaller datasets
            sw_t_train[:] = SW_slt[i_step][:actual_train]
            sg_t_train[:] = SG_slt[i_step][:actual_train]
            pres_t_train[:] = PRES_slt[i_step][:actual_train]
            
            sw_t_eval[:] = SW_slt[i_step][actual_train:actual_train + actual_eval]
            sg_t_eval[:] = SG_slt[i_step][actual_train:actual_train + actual_eval]
            pres_t_eval[:] = PRES_slt[i_step][actual_train:actual_train + actual_eval]
            
            if i_step < len(BHP_slt) and BHP_slt:
                bhp_t_train[:] = BHP_slt[i_step][:actual_train]
                yobs_t_train[:] = Yobs_slt[i_step][:actual_train]
                bhp_t_eval[:] = BHP_slt[i_step][actual_train:actual_train + actual_eval]
                yobs_t_eval[:] = Yobs_slt[i_step][actual_train:actual_train + actual_eval]
        else:
            # Original splitting logic for larger datasets
            for k in range(split_ratio):
                ind0 = k * num_run_per_case
                sw_t_train[ind0:ind0 + num_run_per_case] = SW_slt[i_step][k*100:k*100 + num_run_per_case]
                sg_t_train[ind0:ind0 + num_run_per_case] = SG_slt[i_step][k*100:k*100 + num_run_per_case]
                pres_t_train[ind0:ind0 + num_run_per_case] = PRES_slt[i_step][k*100:k*100 + num_run_per_case]
                
                if i_step < len(BHP_slt) and BHP_slt:
                    bhp_t_train[ind0:ind0 + num_run_per_case] = BHP_slt[i_step][k*100:k*100 + num_run_per_case]
                    yobs_t_train[ind0:ind0 + num_run_per_case] = Yobs_slt[i_step][k*100:k*100 + num_run_per_case]
                
                # Evaluation set
                ind1 = k * num_run_eval
                sw_t_eval[ind1:ind1 + num_run_eval] = SW_slt[i_step][k*100 + num_run_per_case:k*100 + 100]
                sg_t_eval[ind1:ind1 + num_run_eval] = SG_slt[i_step][k*100 + num_run_per_case:k*100 + 100]
                pres_t_eval[ind1:ind1 + num_run_eval] = PRES_slt[i_step][k*100 + num_run_per_case:k*100 + 100]
                
                if i_step < len(BHP_slt) and BHP_slt:
                    bhp_t_eval[ind1:ind1 + num_run_eval] = BHP_slt[i_step][k*100 + num_run_per_case:k*100 + 100]
                    yobs_t_eval[ind1:ind1 + num_run_eval] = Yobs_slt[i_step][k*100 + num_run_per_case:k*100 + 100]

        # Reshape for 3D CNN: (batch, channels, Nx, Ny, Nz)
        SW_t_train = sw_t_train.reshape((actual_train * num_t_slt, 1, Nx, Ny, Nz))
        SG_t_train = sg_t_train.reshape((actual_train * num_t_slt, 1, Nx, Ny, Nz))
        PRES_t_train = pres_t_train.reshape((actual_train * num_t_slt, 1, Nx, Ny, Nz))
        
        SW_t_eval = sw_t_eval.reshape((actual_eval * num_t_slt, 1, Nx, Ny, Nz))
        SG_t_eval = sg_t_eval.reshape((actual_eval * num_t_slt, 1, Nx, Ny, Nz))
        PRES_t_eval = pres_t_eval.reshape((actual_eval * num_t_slt, 1, Nx, Ny, Nz))
        
        if i_step < len(BHP_slt) and BHP_slt:
            # BHP_slt[0] has shape (n_cases, n_timesteps, n_controls), so use shape[2] for controls
            BHP_t_train = bhp_t_train.reshape((actual_train * num_t_slt, BHP_slt[0].shape[2]))
            Yobs_t_train = yobs_t_train.reshape((actual_train * num_t_slt, Yobs_slt[0].shape[2]))
            BHP_t_eval = bhp_t_eval.reshape((actual_eval * num_t_slt, BHP_slt[0].shape[2]))
            Yobs_t_eval = yobs_t_eval.reshape((actual_eval * num_t_slt, Yobs_slt[0].shape[2]))

        # Create state tensors with dynamic channel support
        train_channels = []
        eval_channels = []
        
        # Use the dynamic channel data from sliding window if available
        if channel_data_slt and len(channel_data_slt) >= n_channels:
            for channel_idx in range(n_channels):
                # Get channel data for this step
                channel_data = channel_data_slt[channel_idx][i_step]
                
                # Split into train and eval
                if num_all < 100:
                    train_data = channel_data[:actual_train]
                    eval_data = channel_data[actual_train:actual_train + actual_eval]
                else:
                    train_data = np.zeros((actual_train, num_t_slt, Nx, Ny, Nz))
                    eval_data = np.zeros((actual_eval, num_t_slt, Nx, Ny, Nz))
                    
                    for k in range(split_ratio):
                        ind0 = k * num_run_per_case
                        train_data[ind0:ind0 + num_run_per_case] = channel_data[k*100:k*100 + num_run_per_case]
                        
                        ind1 = k * num_run_eval
                        eval_data[ind1:ind1 + num_run_eval] = channel_data[k*100 + num_run_per_case:k*100 + 100]
                
                # Reshape for CNN: (batch, 1, Nx, Ny, Nz)
                train_data_reshaped = train_data.reshape((actual_train * num_t_slt, 1, Nx, Ny, Nz))
                eval_data_reshaped = eval_data.reshape((actual_eval * num_t_slt, 1, Nx, Ny, Nz))
                
                train_channels.append(train_data_reshaped)
                eval_channels.append(eval_data_reshaped)
        else:
            # Fallback to legacy data structure
            legacy_channels = []
            if len(SW_slt) > i_step:
                legacy_channels.append(('SW', SW_t_train, SW_t_eval))
            if len(SG_slt) > i_step:
                legacy_channels.append(('SG', SG_t_train, SG_t_eval))
            if len(PRES_slt) > i_step:
                legacy_channels.append(('PRES', PRES_t_train, PRES_t_eval))
            
            # Use available channels or create zeros
            for channel_idx in range(n_channels):
                if channel_idx < len(legacy_channels):
                    _, train_data, eval_data = legacy_channels[channel_idx]
                    train_channels.append(train_data)
                    eval_channels.append(eval_data)
                else:
                    zero_channel_train = np.zeros((actual_train * num_t_slt, 1, Nx, Ny, Nz))
                    zero_channel_eval = np.zeros((actual_eval * num_t_slt, 1, Nx, Ny, Nz))
                    train_channels.append(zero_channel_train)
                    eval_channels.append(zero_channel_eval)
        
        # Concatenate all channels
        if train_channels and eval_channels:
            STATE_t_train = torch.tensor(
                np.concatenate(train_channels, axis=1), 
                dtype=torch.float32
            ).to(device)
            STATE_t_eval = torch.tensor(
                np.concatenate(eval_channels, axis=1), 
                dtype=torch.float32
            ).to(device)
        else:
            print(f"    ❌ Error: No channel data available for n_channels={n_channels}")
            return None, None, None, None, None, None
        
        # Apply shuffling
        STATE_t_train = STATE_t_train[shuffle_ind_train]
        if i_step < len(BHP_slt) and BHP_slt:
            BHP_t_train = torch.tensor(BHP_t_train[shuffle_ind_train], dtype=torch.float32).to(device)
            Yobs_t_train = torch.tensor(Yobs_t_train[shuffle_ind_train], dtype=torch.float32).to(device)

        STATE_t_eval = STATE_t_eval[shuffle_ind_eval]
        if i_step < len(BHP_slt) and BHP_slt:
            BHP_t_eval = torch.tensor(BHP_t_eval[shuffle_ind_eval], dtype=torch.float32).to(device)
            Yobs_t_eval = torch.tensor(Yobs_t_eval[shuffle_ind_eval], dtype=torch.float32).to(device)

        # Store processed data
        STATE_train.append(STATE_t_train)
        STATE_eval.append(STATE_t_eval)
        if i_step < len(BHP_slt) and BHP_slt:
            BHP_train.append(BHP_t_train)
            BHP_eval.append(BHP_t_eval)
            Yobs_train.append(Yobs_t_train)
            Yobs_eval.append(Yobs_t_eval)

    print(f"Data splitting complete: {len(STATE_train)} training batches")
    
    return STATE_train, BHP_train, Yobs_train, STATE_eval, BHP_eval, Yobs_eval

