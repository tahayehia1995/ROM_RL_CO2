import numpy as np

def normalize_dataset_inplace(data, dataset_name, normalization_type='minmax'):
    """
    Apply normalization to a dataset and return normalization parameters
    
    Args:
        data: Input data array
        dataset_name: Name of the dataset for logging
        normalization_type: 'minmax', 'log', or 'none' normalization
    """
    if data.size == 0:
        return data, {'type': normalization_type, 'min': 0.0, 'max': 1.0}
    
        # Handle 'none' normalization type (keep original values)
    if normalization_type == 'none':
        pass  # Keeping original values
        norm_params = {
            'type': 'none',
            'min': np.min(data),
            'max': np.max(data)
        }
        print(f"    📊 Original range: [{norm_params['min']:.6f}, {norm_params['max']:.6f}]")
        return data.copy(), norm_params
    
    # Determine normalization strategy based on dataset structure
    shape = data.shape
    pass  # Dataset loaded
    
    if len(shape) == 5:  # Spatial data: (cases, timesteps, x, y, z)
        pass  # Spatial data structure
    elif len(shape) == 3:  # Time series data: (cases, timesteps, wells)
        pass  # Time series data structure
    else:
        pass  # Unknown structure
    
    if normalization_type == 'log':
        pass  # Log normalization
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        data_shifted = data + epsilon
        
        # Check for negative values
        if np.any(data < 0):
            print(f"    ⚠️ Negative values detected. Shifting data to positive range...")
            data_shifted = data - np.min(data) + epsilon
        
        # Apply log transformation
        log_data = np.log(data_shifted)
        
        # Then apply min-max normalization to log-transformed data
        log_min = np.min(log_data)
        log_max = np.max(log_data)
        
        if log_max == log_min:
            print(f"    ⚠️ All log values are the same ({log_min}), returning zeros")
            return np.zeros_like(data), {
                'type': 'log',
                'min': log_min,
                'max': log_max,
                'epsilon': epsilon,
                'data_shift': np.min(data) if np.any(data < 0) else 0
            }
        
        normalized = (log_data - log_min) / (log_max - log_min)
        
        norm_params = {
            'type': 'log',
            'log_min': log_min,
            'log_max': log_max,
            'epsilon': epsilon,
            'data_shift': np.min(data) if np.any(data < 0) else 0
        }
        
        pass  # Log range applied
        
    else:  # minmax normalization (default)
        pass  # Min-max normalization
        # Use minimum positive value instead of absolute minimum
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            data_min = np.min(positive_data)  # Minimum positive value
        else:
            data_min = np.min(data)  # Fallback to absolute minimum if no positive values
        
        data_max = np.max(data)
        
        # Avoid division by zero
        if data_max == data_min:
            print(f"    ⚠️ All values are the same ({data_min}), returning zeros")
            return np.zeros_like(data), {'type': 'minmax', 'min': data_min, 'max': data_max}
        
        # Apply min-max normalization
        normalized = (data - data_min) / (data_max - data_min)
        
        norm_params = {
            'type': 'minmax',
            'min': data_min,
            'max': data_max
        }
        
        pass  # Min-max range applied
    
    return normalized, norm_params

def denormalize_data(normalized_data, norm_params):
    """
    Denormalize data using stored normalization parameters
    
    Args:
        normalized_data: Normalized data array
        norm_params: Normalization parameters dictionary
    """
    norm_type = norm_params.get('type', 'minmax')
    
    if norm_type == 'none':
        return normalized_data
    
    elif norm_type == 'log':
        # Reverse log normalization
        log_min = norm_params['log_min']
        log_max = norm_params['log_max']
        epsilon = norm_params.get('epsilon', 1e-8)
        data_shift = norm_params.get('data_shift', 0)
        
        # Reverse min-max on log space
        log_data = normalized_data * (log_max - log_min) + log_min
        
        # Reverse log transformation
        data_shifted = np.exp(log_data)
        
        # Reverse data shift
        denormalized = data_shifted - epsilon + data_shift
        
        return denormalized
    
    else:  # minmax
        # Reverse min-max normalization
        data_min = norm_params['min']
        data_max = norm_params['max']
        
        denormalized = normalized_data * (data_max - data_min) + data_min
        
        return denormalized

