"""
Data loading functions for H5 files
Handles scanning directories, loading spatial and timeseries data files
"""

import h5py
import numpy as np
import os
import glob
from typing import List, Dict, Tuple, Optional

# Canonical order for spatial properties to ensure consistency
CANONICAL_SPATIAL_ORDER = ['SW', 'SG', 'PRES', 'PERMI', 'POROS', 'PERMJ', 'PERMK']


def scan_h5_files(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Scan directory for H5 files and categorize into spatial and timeseries files.
    
    Args:
        data_dir: Directory containing H5 files
        
    Returns:
        Tuple of (spatial_files, timeseries_files) with deterministic ordering
    """
    if not data_dir.endswith('/'):
        data_dir += '/'
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Find H5 files with deterministic sorting
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    
    if not h5_files:
        return [], []
    
    # Categorize files with deterministic ordering
    spatial_files_dict = {}
    timeseries_files = []
    
    for file_path in h5_files:
        filename = os.path.basename(file_path)
        if 'spatial_properties' in filename:
            # Extract variable name for canonical ordering
            var_name = filename.replace('batch_spatial_properties_', '').replace('.h5', '')
            spatial_files_dict[var_name] = filename
        elif 'timeseries_data' in filename:
            timeseries_files.append(filename)
    
    # Sort spatial files in canonical order for consistency
    spatial_files = []
    
    # First, add files in canonical order
    for canonical_var in CANONICAL_SPATIAL_ORDER:
        if canonical_var in spatial_files_dict:
            spatial_files.append(spatial_files_dict[canonical_var])
    
    # Then, add any additional files not in canonical order (alphabetically sorted)
    remaining_vars = set(spatial_files_dict.keys()) - set(CANONICAL_SPATIAL_ORDER)
    for var_name in sorted(remaining_vars):
        spatial_files.append(spatial_files_dict[var_name])
    
    # Sort timeseries files alphabetically for consistency
    timeseries_files = sorted(timeseries_files)
    
    return spatial_files, timeseries_files


def load_spatial_file(filepath: str) -> np.ndarray:
    """
    Load spatial property data from H5 file.
    
    Args:
        filepath: Full path to spatial property H5 file
        
    Returns:
        Numpy array with shape (cases, timesteps, Nx, Ny, Nz)
    """
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf['data'])
    return data


def load_timeseries_file(filepath: str) -> np.ndarray:
    """
    Load timeseries data from H5 file.
    
    Args:
        filepath: Full path to timeseries H5 file
        
    Returns:
        Numpy array with shape (cases, timesteps, wells)
    """
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf['data'])
    return data


def get_file_info(filepath: str) -> Dict:
    """
    Get information about an H5 file (shape, dtype, etc.).
    
    Args:
        filepath: Full path to H5 file
        
    Returns:
        Dictionary with file information
    """
    with h5py.File(filepath, 'r') as hf:
        if 'data' in hf:
            data = hf['data']
            return {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size': data.size
            }
        else:
            return {'error': 'No data dataset found'}


def validate_data_shapes(spatial_data: Dict[str, np.ndarray], 
                        timeseries_data: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
    """
    Validate that all data files have consistent shapes.
    
    Args:
        spatial_data: Dictionary of spatial property arrays
        timeseries_data: Dictionary of timeseries arrays
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check spatial data consistency
    if spatial_data:
        first_spatial = list(spatial_data.values())[0]
        expected_spatial_shape = first_spatial.shape[:2]  # (cases, timesteps)
        
        for var_name, data in spatial_data.items():
            if data.shape[:2] != expected_spatial_shape:
                errors.append(f"Spatial data {var_name} has inconsistent shape: {data.shape[:2]} vs {expected_spatial_shape}")
    
    # Check timeseries data consistency
    if timeseries_data:
        first_timeseries = list(timeseries_data.values())[0]
        expected_timeseries_shape = first_timeseries.shape[:2]  # (cases, timesteps)
        
        for var_name, data in timeseries_data.items():
            if data.shape[:2] != expected_timeseries_shape:
                errors.append(f"Timeseries data {var_name} has inconsistent shape: {data.shape[:2]} vs {expected_timeseries_shape}")
    
    return len(errors) == 0, errors

