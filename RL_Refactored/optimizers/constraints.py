"""
Constraints Module for Classical Optimizers
============================================

Provides constraint handling for reservoir optimization problems:
- Bound constraints (well control limits)
- Nonlinear constraints (cumulative limits, rate limits, etc.)

Matches the constraint structure used in RL action ranges.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import NonlinearConstraint, LinearConstraint


def get_bounds_from_config(
    config: Dict,
    num_steps: int,
    num_prod: int = 3,
    num_inj: int = 3
) -> List[Tuple[float, float]]:
    """
    Get scipy-compatible bounds from configuration.
    
    Matches the action_ranges structure from RL dashboard.
    
    Args:
        config: Configuration dictionary with action_ranges
        num_steps: Number of control timesteps
        num_prod: Number of production wells
        num_inj: Number of injection wells
        
    Returns:
        List of (min, max) tuples for each control variable
        Order: [BHP_P1_t0, BHP_P2_t0, BHP_P3_t0, GAS_I1_t0, ..., GAS_I3_tN]
    """
    # Extract action ranges from config
    action_ranges = config.get('action_ranges', {})
    
    # Default ranges if not specified
    bhp_min = 1087.784912109375
    bhp_max = 5050.42871094
    gas_min = 0.0
    gas_max = 143972480.0
    
    # Override with config values if available
    if 'producer_bhp' in action_ranges:
        bhp_min = action_ranges['producer_bhp'].get('min', bhp_min)
        bhp_max = action_ranges['producer_bhp'].get('max', bhp_max)
    elif 'bhp' in action_ranges:
        # Handle per-well BHP ranges
        bhp_values = action_ranges['bhp']
        if isinstance(bhp_values, dict):
            all_mins = [v.get('min', bhp_min) for v in bhp_values.values()]
            all_maxs = [v.get('max', bhp_max) for v in bhp_values.values()]
            bhp_min = min(all_mins)
            bhp_max = max(all_maxs)
    
    if 'gas_injection' in action_ranges:
        gas_min = action_ranges['gas_injection'].get('min', gas_min)
        gas_max = action_ranges['gas_injection'].get('max', gas_max)
    
    # Build bounds list
    bounds = []
    num_controls = num_prod + num_inj
    
    for t in range(num_steps):
        # Producer BHP bounds (first num_prod controls per step)
        for p in range(num_prod):
            bounds.append((bhp_min, bhp_max))
        
        # Gas injection bounds (next num_inj controls per step)
        for i in range(num_inj):
            bounds.append((gas_min, gas_max))
    
    return bounds


def get_per_well_bounds(
    config: Dict,
    num_steps: int,
    num_prod: int = 3,
    num_inj: int = 3
) -> List[Tuple[float, float]]:
    """
    Get per-well bounds if different wells have different constraints.
    
    Args:
        config: Configuration with per-well action_ranges
        num_steps: Number of control timesteps
        num_prod: Number of producers
        num_inj: Number of injectors
        
    Returns:
        List of (min, max) tuples for each control variable
    """
    action_ranges = config.get('action_ranges', {})
    bounds = []
    
    # Get per-well BHP bounds
    bhp_bounds = {}
    if 'bhp' in action_ranges:
        for well, ranges in action_ranges['bhp'].items():
            bhp_bounds[well] = (ranges.get('min', 1000), ranges.get('max', 5000))
    
    # Get per-well gas injection bounds
    gas_bounds = {}
    if 'gas_injection' in action_ranges:
        for well, ranges in action_ranges['gas_injection'].items():
            gas_bounds[well] = (ranges.get('min', 0), ranges.get('max', 1e8))
    
    # Default bounds
    default_bhp = (1087.78, 5050.43)
    default_gas = (0, 143972480)
    
    for t in range(num_steps):
        # Producer bounds
        for p in range(num_prod):
            well_name = f'P{p+1}'
            bounds.append(bhp_bounds.get(well_name, default_bhp))
        
        # Injector bounds
        for i in range(num_inj):
            well_name = f'I{i+1}'
            bounds.append(gas_bounds.get(well_name, default_gas))
    
    return bounds


def create_nonlinear_constraints(
    config: Dict,
    num_steps: int,
    num_prod: int = 3,
    num_inj: int = 3
) -> List[NonlinearConstraint]:
    """
    Create nonlinear constraints for the optimization problem.
    
    Example constraints:
    - Maximum cumulative gas injection
    - Maximum rate of change between timesteps
    - Minimum field production rates
    
    Args:
        config: Configuration with constraint parameters
        num_steps: Number of control timesteps
        num_prod: Number of producers
        num_inj: Number of injectors
        
    Returns:
        List of scipy NonlinearConstraint objects
    """
    constraints = []
    num_controls = num_prod + num_inj
    constraint_config = config.get('optimization_constraints', {})
    
    # Cumulative gas injection constraint
    if constraint_config.get('max_cumulative_gas_injection'):
        max_cum_gas = constraint_config['max_cumulative_gas_injection']
        
        def cumulative_gas_constraint(x):
            """Sum of all gas injection across all timesteps and wells."""
            x_reshaped = x.reshape(num_steps, num_controls)
            gas_injection = x_reshaped[:, num_prod:]  # Gas injection columns
            return np.sum(gas_injection)
        
        constraints.append(NonlinearConstraint(
            cumulative_gas_constraint,
            lb=-np.inf,
            ub=max_cum_gas
        ))
    
    # Rate of change constraint (smooth controls)
    if constraint_config.get('max_control_rate_of_change'):
        max_roc = constraint_config['max_control_rate_of_change']
        
        def rate_of_change_constraint(x):
            """Maximum absolute change between consecutive timesteps."""
            x_reshaped = x.reshape(num_steps, num_controls)
            diffs = np.abs(np.diff(x_reshaped, axis=0))
            return np.max(diffs)
        
        constraints.append(NonlinearConstraint(
            rate_of_change_constraint,
            lb=-np.inf,
            ub=max_roc
        ))
    
    # Minimum field gas injection per step (for CO2 storage targets)
    if constraint_config.get('min_field_gas_injection_per_step'):
        min_gas = constraint_config['min_field_gas_injection_per_step']
        
        def min_gas_constraint(x):
            """Minimum total gas injection at each timestep."""
            x_reshaped = x.reshape(num_steps, num_controls)
            gas_per_step = np.sum(x_reshaped[:, num_prod:], axis=1)
            return gas_per_step
        
        constraints.append(NonlinearConstraint(
            min_gas_constraint,
            lb=min_gas * np.ones(num_steps),
            ub=np.inf * np.ones(num_steps)
        ))
    
    return constraints


def create_linear_constraints(
    config: Dict,
    num_steps: int,
    num_prod: int = 3,
    num_inj: int = 3
) -> List[LinearConstraint]:
    """
    Create linear constraints for faster optimization.
    
    Linear constraints can be solved more efficiently than nonlinear ones.
    
    Args:
        config: Configuration with constraint parameters
        num_steps: Number of control timesteps
        num_prod: Number of producers
        num_inj: Number of injectors
        
    Returns:
        List of scipy LinearConstraint objects
    """
    constraints = []
    num_controls = num_prod + num_inj
    total_vars = num_steps * num_controls
    constraint_config = config.get('optimization_constraints', {})
    
    # Example: Cumulative gas injection limit (this is linear)
    if constraint_config.get('max_cumulative_gas_injection'):
        max_cum_gas = constraint_config['max_cumulative_gas_injection']
        
        # Build coefficient matrix for sum of all gas injection
        A = np.zeros((1, total_vars))
        for t in range(num_steps):
            for i in range(num_inj):
                idx = t * num_controls + num_prod + i
                A[0, idx] = 1.0
        
        constraints.append(LinearConstraint(
            A,
            lb=-np.inf,
            ub=max_cum_gas
        ))
    
    return constraints


def validate_controls(
    controls: np.ndarray,
    bounds: List[Tuple[float, float]],
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    Validate that controls satisfy all bound constraints.
    
    Args:
        controls: Flattened control array
        bounds: List of (min, max) tuples
        tolerance: Numerical tolerance for constraint checking
        
    Returns:
        is_valid: True if all constraints satisfied
        violations: List of violation descriptions
    """
    is_valid = True
    violations = []
    
    for i, (val, (lb, ub)) in enumerate(zip(controls, bounds)):
        if val < lb - tolerance:
            is_valid = False
            violations.append(f"Control {i}: {val:.4f} < lower bound {lb:.4f}")
        if val > ub + tolerance:
            is_valid = False
            violations.append(f"Control {i}: {val:.4f} > upper bound {ub:.4f}")
    
    return is_valid, violations


def project_to_bounds(
    controls: np.ndarray,
    bounds: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Project controls to satisfy bound constraints.
    
    Args:
        controls: Flattened control array
        bounds: List of (min, max) tuples
        
    Returns:
        Projected control array within bounds
    """
    projected = controls.copy()
    
    for i, (lb, ub) in enumerate(bounds):
        projected[i] = np.clip(projected[i], lb, ub)
    
    return projected


def get_constraint_summary(
    controls: np.ndarray,
    bounds: List[Tuple[float, float]],
    num_steps: int,
    num_prod: int = 3,
    num_inj: int = 3
) -> Dict:
    """
    Get summary of how controls relate to constraints.
    
    Useful for analysis and dashboard display.
    
    Args:
        controls: Optimal control array
        bounds: Bound constraints
        num_steps: Number of timesteps
        num_prod: Number of producers
        num_inj: Number of injectors
        
    Returns:
        Summary dictionary with constraint statistics
    """
    num_controls = num_prod + num_inj
    controls_reshaped = controls.reshape(num_steps, num_controls)
    
    summary = {
        'producer_bhp': {
            'mean': [],
            'min': [],
            'max': [],
            'at_lower_bound': [],
            'at_upper_bound': []
        },
        'gas_injection': {
            'mean': [],
            'min': [],
            'max': [],
            'at_lower_bound': [],
            'at_upper_bound': []
        }
    }
    
    # Analyze producer BHP
    bhp_controls = controls_reshaped[:, :num_prod]
    for p in range(num_prod):
        well_bhp = bhp_controls[:, p]
        bound_idx = p  # First timestep, producer p
        lb, ub = bounds[bound_idx]
        
        summary['producer_bhp']['mean'].append(np.mean(well_bhp))
        summary['producer_bhp']['min'].append(np.min(well_bhp))
        summary['producer_bhp']['max'].append(np.max(well_bhp))
        summary['producer_bhp']['at_lower_bound'].append(
            np.sum(np.isclose(well_bhp, lb, rtol=0.01)) / num_steps
        )
        summary['producer_bhp']['at_upper_bound'].append(
            np.sum(np.isclose(well_bhp, ub, rtol=0.01)) / num_steps
        )
    
    # Analyze gas injection
    gas_controls = controls_reshaped[:, num_prod:]
    for i in range(num_inj):
        well_gas = gas_controls[:, i]
        bound_idx = num_prod + i  # First timestep, injector i
        lb, ub = bounds[bound_idx]
        
        summary['gas_injection']['mean'].append(np.mean(well_gas))
        summary['gas_injection']['min'].append(np.min(well_gas))
        summary['gas_injection']['max'].append(np.max(well_gas))
        summary['gas_injection']['at_lower_bound'].append(
            np.sum(np.isclose(well_gas, lb, rtol=0.01)) / num_steps
        )
        summary['gas_injection']['at_upper_bound'].append(
            np.sum(np.isclose(well_gas, ub, rtol=0.01)) / num_steps
        )
    
    return summary
