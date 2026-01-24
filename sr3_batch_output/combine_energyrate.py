"""
Combine separate ENERGYRATE producer files into a single file matching BHP structure.

This script:
1. Loads BHP file to determine well configuration (cases, timesteps, num_wells)
2. Finds all ENERGYRATE producer files matching pattern *ENERGYRATE prod-*.h5
3. Extracts non-zero well data from each producer file
4. Combines into single array with zeros for injectors and actual data for producers
5. Saves as batch_timeseries_data_ENERGYRATE.h5
"""

#%%
import h5py
import numpy as np
import os
import glob
import re
from typing import List, Tuple, Optional
import argparse


def load_bhp_structure(bhp_filepath: str) -> Tuple[int, int, int]:
    """
    Load BHP file and extract shape information.
    
    Args:
        bhp_filepath: Path to batch_timeseries_data_BHP.h5 file
        
    Returns:
        Tuple of (num_cases, num_timesteps, num_wells)
        
    Raises:
        FileNotFoundError: If BHP file doesn't exist
        KeyError: If 'data' dataset not found in file
    """
    if not os.path.exists(bhp_filepath):
        raise FileNotFoundError(f"BHP file not found: {bhp_filepath}")
    
    with h5py.File(bhp_filepath, 'r') as hf:
        if 'data' not in hf:
            raise KeyError(f"No 'data' dataset found in {bhp_filepath}")
        
        data = hf['data']
        shape = data.shape
        
        if len(shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {shape}")
        
        num_cases, num_timesteps, num_wells = shape
        print(f"[INFO] BHP file structure: {num_cases} cases, {num_timesteps} timesteps, {num_wells} wells")
        
        return num_cases, num_timesteps, num_wells


def find_energyrate_files(data_dir: str, pattern: str = "*ENERGYRATE prod-*.h5") -> List[str]:
    """
    Find and sort ENERGYRATE producer files matching the pattern.
    
    Args:
        data_dir: Directory to search for ENERGYRATE files
        pattern: Glob pattern to match ENERGYRATE files (default: "*ENERGYRATE prod-*.h5")
        
    Returns:
        Sorted list of file paths (sorted by producer number: prod-1, prod-2, etc.)
        
    Raises:
        FileNotFoundError: If no ENERGYRATE files found
    """
    search_pattern = os.path.join(data_dir, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No ENERGYRATE files found matching pattern: {pattern}")
    
    # Sort by producer number (extract number from filename)
    def extract_producer_number(filepath: str) -> int:
        filename = os.path.basename(filepath)
        match = re.search(r'prod-(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    files_sorted = sorted(files, key=extract_producer_number)
    
    print(f"[INFO] Found {len(files_sorted)} ENERGYRATE producer files:")
    for i, filepath in enumerate(files_sorted, 1):
        print(f"   {i}. {os.path.basename(filepath)}")
    
    return files_sorted


def extract_producer_data(energyrate_filepath: str, num_cases: int, num_timesteps: int) -> Tuple[np.ndarray, int]:
    """
    Extract non-zero well data from a single ENERGYRATE file.
    
    Args:
        energyrate_filepath: Path to ENERGYRATE producer file
        num_cases: Expected number of cases (for validation)
        num_timesteps: Expected number of timesteps (for validation)
        
    Returns:
        Tuple of (producer_data, well_index) where:
        - producer_data: Array of shape (num_cases, num_timesteps)
        - well_index: The well index that contained the non-zero data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If shape mismatch or no/multiple non-zero wells found
    """
    if not os.path.exists(energyrate_filepath):
        raise FileNotFoundError(f"ENERGYRATE file not found: {energyrate_filepath}")
    
    with h5py.File(energyrate_filepath, 'r') as hf:
        if 'data' not in hf:
            raise KeyError(f"No 'data' dataset found in {energyrate_filepath}")
        
        data = np.array(hf['data'])
        
        # Validate shape
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {data.shape}")
        
        file_cases, file_timesteps, file_wells = data.shape
        
        if file_cases != num_cases or file_timesteps != num_timesteps:
            raise ValueError(
                f"Shape mismatch in {energyrate_filepath}: "
                f"expected ({num_cases}, {num_timesteps}, _), got {data.shape}"
            )
        
        # Find which well index has non-zero data
        non_zero_wells = []
        for well_idx in range(file_wells):
            if np.any(data[:, :, well_idx] != 0):
                non_zero_wells.append(well_idx)
        
        if len(non_zero_wells) == 0:
            raise ValueError(f"No non-zero data found in {energyrate_filepath}")
        
        if len(non_zero_wells) > 1:
            raise ValueError(
                f"Multiple non-zero wells found in {energyrate_filepath}: {non_zero_wells}. "
                f"Expected exactly one well with data."
            )
        
        well_index = non_zero_wells[0]
        producer_data = data[:, :, well_index].copy()
        
        filename = os.path.basename(energyrate_filepath)
        print(f"   [OK] {filename}: extracted data from well index {well_index}")
        
        return producer_data, well_index


def combine_energyrate_files(
    bhp_filepath: str,
    energyrate_files: List[str],
    output_filepath: str,
    num_injectors: Optional[int] = None,
    num_producers: Optional[int] = None
) -> None:
    """
    Combine ENERGYRATE producer files into a single file matching BHP structure.
    
    Args:
        bhp_filepath: Path to batch_timeseries_data_BHP.h5 file
        energyrate_files: List of paths to ENERGYRATE producer files (sorted by producer number)
        output_filepath: Path where combined file will be saved
        num_injectors: Number of injector wells (auto-detected from BHP if None)
        num_producers: Number of producer wells (auto-detected if None)
        
    Raises:
        ValueError: If configuration is invalid or data inconsistencies found
    """
    # Load BHP structure
    num_cases, num_timesteps, num_wells = load_bhp_structure(bhp_filepath)
    
    # Auto-detect injectors/producers if not specified
    if num_injectors is None or num_producers is None:
        if num_injectors is None and num_producers is None:
            # Try to infer from number of ENERGYRATE files
            num_producers = len(energyrate_files)
            num_injectors = num_wells - num_producers
            print(f"[INFO] Auto-detected: {num_injectors} injectors, {num_producers} producers")
        elif num_injectors is None:
            num_injectors = num_wells - num_producers
        else:
            num_producers = num_wells - num_injectors
    
    # Validate configuration
    if num_injectors + num_producers != num_wells:
        raise ValueError(
            f"Invalid well configuration: {num_injectors} injectors + {num_producers} producers "
            f"!= {num_wells} total wells"
        )
    
    if len(energyrate_files) != num_producers:
        raise ValueError(
            f"Mismatch: found {len(energyrate_files)} ENERGYRATE files but expected {num_producers} producers"
        )
    
    # Initialize combined array with zeros
    combined_data = np.zeros((num_cases, num_timesteps, num_wells), dtype=np.float64)
    
    print(f"\n[INFO] Extracting data from ENERGYRATE files...")
    
    # Extract and map each producer's data
    for producer_idx, energyrate_file in enumerate(energyrate_files):
        producer_data, source_well_idx = extract_producer_data(
            energyrate_file, num_cases, num_timesteps
        )
        
        # Map to correct producer well index
        target_well_idx = num_injectors + producer_idx
        
        if target_well_idx >= num_wells:
            raise ValueError(
                f"Target well index {target_well_idx} exceeds total wells {num_wells}"
            )
        
        combined_data[:, :, target_well_idx] = producer_data
        print(f"   [MAP] Mapped producer {producer_idx + 1} data to well index {target_well_idx}")
    
    # Verify injector wells are zeros
    injector_data = combined_data[:, :, :num_injectors]
    if np.any(injector_data != 0):
        print(f"[WARNING] Non-zero values found in injector wells (0-{num_injectors-1})")
    else:
        print(f"[OK] Verified: Injector wells (0-{num_injectors-1}) contain zeros")
    
    # Save combined file
    print(f"\n[INFO] Saving combined ENERGYRATE file to: {output_filepath}")
    with h5py.File(output_filepath, 'w') as hf:
        hf.create_dataset('data', data=combined_data, compression='gzip', compression_opts=9)
    
    print(f"[SUCCESS] Successfully created combined ENERGYRATE file!")
    print(f"   Shape: {combined_data.shape}")
    print(f"   Injector wells (0-{num_injectors-1}): zeros")
    print(f"   Producer wells ({num_injectors}-{num_wells-1}): energy rate data")


def main():
    """Command-line interface for combining ENERGYRATE files."""
    parser = argparse.ArgumentParser(
        description="Combine separate ENERGYRATE producer files into a single file matching BHP structure",
        allow_abbrev=False  # Prevent abbreviation conflicts with Jupyter's --f argument
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing BHP and ENERGYRATE files (default: current directory)'
    )
    parser.add_argument(
        '--bhp-file',
        type=str,
        default='batch_timeseries_data_BHP.h5',
        help='BHP file name (default: batch_timeseries_data_BHP.h5)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='batch_timeseries_data_ENERGYRATE.h5',
        help='Output file name (default: batch_timeseries_data_ENERGYRATE.h5)'
    )
    parser.add_argument(
        '--num-injectors',
        type=int,
        default=None,
        help='Number of injector wells (auto-detected if not specified)'
    )
    parser.add_argument(
        '--num-producers',
        type=int,
        default=None,
        help='Number of producer wells (auto-detected if not specified)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*ENERGYRATE prod-*.h5',
        help='Pattern to match ENERGYRATE files (default: *ENERGYRATE prod-*.h5)'
    )
    
    # Use parse_known_args to ignore unknown arguments (like Jupyter's --f argument)
    args, unknown = parser.parse_known_args()
    
    # Warn about unknown arguments if any (but don't fail)
    if unknown:
        print(f"[WARNING] Ignoring unknown arguments: {unknown}")
    
    # Construct full paths
    bhp_filepath = os.path.join(args.data_dir, args.bhp_file)
    output_filepath = os.path.join(args.data_dir, args.output_file)
    
    try:
        # Find ENERGYRATE files
        energyrate_files = find_energyrate_files(args.data_dir, args.pattern)
        
        # Combine files
        combine_energyrate_files(
            bhp_filepath=bhp_filepath,
            energyrate_files=energyrate_files,
            output_filepath=output_filepath,
            num_injectors=args.num_injectors,
            num_producers=args.num_producers
        )
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


# %%
