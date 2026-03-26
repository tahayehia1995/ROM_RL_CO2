"""
Objective Function for Classical Optimizers
============================================

Provides objective function computation that exactly matches the RL reward function.
This ensures fair comparison between RL and classical optimization approaches.

The objective function computes Net Present Value (NPV) based on:
- Gas injection net revenue (revenue - cost)
- Water production penalty
- Gas production penalty
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_step_reward(
    yobs: torch.Tensor,
    action: torch.Tensor,
    num_prod: int,
    num_inj: int,
    config
) -> torch.Tensor:
    """
    Compute single-step reward matching RL reward function.
    
    This is identical to the reward_fun in environment/reward.py to ensure
    consistency between RL and classical optimization.
    
    Observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
    Action order: [Producer_BHP(0-2), Gas_Injection(3-5)]
    
    Args:
        yobs: Observations in physical units, shape (batch, 9)
        action: Actions in physical units, shape (batch, 6)
        num_prod: Number of production wells (3)
        num_inj: Number of injection wells (3)
        config: Configuration object with economic parameters
        
    Returns:
        Economic value (reward/NPV) for this timestep
    """
    econ_config = config.rl_model['economics']
    
    # Unit conversion factors from config
    gas_conversion_factor = (
        econ_config['conversion']['lf3_to_intermediate'] * 
        econ_config['conversion']['intermediate_to_ton']
    )
    water_conversion_factor = econ_config['conversion']['ft3_to_barrel']
    
    # Economic parameters from config
    prices = econ_config['prices']
    gas_injection_net = prices['gas_injection_revenue'] - prices['gas_injection_cost']
    
    # Extract observations using optimal order
    # Gas production (indices num_inj to num_inj+num_prod)
    gas_production_ft3_day = torch.sum(yobs[:, num_inj:num_inj+num_prod], dim=1)
    
    # Water production (indices num_inj+num_prod to num_inj+2*num_prod)
    water_production_ft3_day = torch.sum(yobs[:, num_inj+num_prod:num_inj+num_prod*2], dim=1)
    water_production_bbl_day = water_production_ft3_day / water_conversion_factor
    
    # Extract actions using optimal order
    # Gas injection (indices num_prod to num_prod+num_inj)
    # Note: For optimizer, actions may already be in physical units
    if action.shape[1] == num_prod + num_inj:
        gas_injection_ft3_day = torch.sum(action[:, num_prod:num_prod+num_inj], dim=1)
    else:
        gas_injection_ft3_day = torch.sum(action[:, num_prod:], dim=1)
    
    # Calculate economic value:
    # - Gas injection revenue: ($/ton) × (ft³/day → tons/day) [POSITIVE]
    # - Water production penalty: ($/bbl) × (bbl/day) [NEGATIVE]
    # - Gas production penalty: ($/ton) × (ft³/day → tons/day) [NEGATIVE]
    PV = (
        gas_injection_net * gas_conversion_factor * gas_injection_ft3_day - 
        prices['water_production_penalty'] * water_production_bbl_day - 
        prices['gas_production_penalty'] * gas_conversion_factor * gas_production_ft3_day
    ) / econ_config['scale_factor']
    
    return PV


def compute_objective(
    observations: List[np.ndarray],
    actions: np.ndarray,
    config,
    num_prod: int = 3,
    num_inj: int = 3
) -> float:
    """
    Compute total objective (NPV) over trajectory.
    
    This is the main objective function for optimizers.
    Sum of rewards over all timesteps.
    
    Args:
        observations: List of observation arrays, each shape (batch, 9)
        actions: Control sequence, shape (num_steps, num_controls)
        config: Configuration object with economic parameters
        num_prod: Number of production wells
        num_inj: Number of injection wells
        
    Returns:
        Total NPV (scalar)
    """
    total_npv = 0.0
    
    for t, obs in enumerate(observations):
        # Convert to tensors
        yobs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_tensor = torch.tensor(actions[t:t+1], dtype=torch.float32)
        
        # Compute step reward
        step_reward = compute_step_reward(
            yobs_tensor, action_tensor, num_prod, num_inj, config
        )
        
        total_npv += float(step_reward.sum().item())
    
    return total_npv


def compute_trajectory_npv(
    observations: List[np.ndarray],
    actions: np.ndarray,
    config,
    num_prod: int = 3,
    num_inj: int = 3,
    discount_rate: float = 0.0
) -> Tuple[float, Dict]:
    """
    Compute NPV with economic breakdown.
    
    Args:
        observations: List of observation arrays
        actions: Control sequence
        config: Configuration object
        num_prod: Number of production wells
        num_inj: Number of injection wells
        discount_rate: Annual discount rate (0 = no discounting)
        
    Returns:
        total_npv: Total discounted NPV
        breakdown: Dictionary with economic components
    """
    econ_config = config.rl_model['economics']
    
    # Unit conversion factors
    gas_conversion = (
        econ_config['conversion']['lf3_to_intermediate'] * 
        econ_config['conversion']['intermediate_to_ton']
    )
    water_conversion = econ_config['conversion']['ft3_to_barrel']
    
    prices = econ_config['prices']
    gas_net = prices['gas_injection_revenue'] - prices['gas_injection_cost']
    scale_factor = econ_config['scale_factor']
    
    # Track components
    total_gas_injection_revenue = 0.0
    total_water_penalty = 0.0
    total_gas_production_penalty = 0.0
    step_npvs = []
    
    for t, obs in enumerate(observations):
        # Discount factor (assuming 1 year per step)
        discount = 1.0 / (1.0 + discount_rate) ** t if discount_rate > 0 else 1.0
        
        # Extract values
        gas_production = np.sum(obs[0, num_inj:num_inj+num_prod])
        water_production = np.sum(obs[0, num_inj+num_prod:num_inj+2*num_prod])
        gas_injection = np.sum(actions[t, num_prod:num_prod+num_inj])
        
        # Compute components
        gas_rev = gas_net * gas_conversion * gas_injection * discount
        water_pen = prices['water_production_penalty'] * (water_production / water_conversion) * discount
        gas_pen = prices['gas_production_penalty'] * gas_conversion * gas_production * discount
        
        step_npv = (gas_rev - water_pen - gas_pen) / scale_factor
        
        total_gas_injection_revenue += gas_rev / scale_factor
        total_water_penalty += water_pen / scale_factor
        total_gas_production_penalty += gas_pen / scale_factor
        step_npvs.append(step_npv)
    
    total_npv = sum(step_npvs)
    
    breakdown = {
        'total_npv': total_npv,
        'gas_injection_revenue': total_gas_injection_revenue,
        'water_production_penalty': total_water_penalty,
        'gas_production_penalty': total_gas_production_penalty,
        'step_npvs': step_npvs,
        'num_steps': len(observations)
    }
    
    return total_npv, breakdown


def compute_economic_breakdown_from_controls(
    controls: np.ndarray,
    rom_model,
    z0: torch.Tensor,
    config,
    norm_params: Dict,
    device: torch.device,
    num_prod: int = 3,
    num_inj: int = 3
) -> Dict:
    """
    Compute full economic breakdown given controls.
    
    This runs the ROM simulation and computes detailed economics.
    
    Args:
        controls: Control sequence, shape (num_steps, num_controls)
        rom_model: ROMWithE2C model
        z0: Initial latent state
        config: Configuration object
        norm_params: Normalization parameters
        device: PyTorch device
        num_prod: Number of producers
        num_inj: Number of injectors
        
    Returns:
        Dictionary with full economic analysis
    """
    # This function would be called by the optimizer after finding optimal controls
    # to get detailed economic breakdown for results dashboard
    
    econ_config = config.rl_model['economics']
    
    breakdown = {
        'per_step': [],
        'per_well': {
            'producers': {f'P{i+1}': {'gas': [], 'water': []} for i in range(num_prod)},
            'injectors': {f'I{i+1}': {'gas': [], 'bhp': []} for i in range(num_inj)}
        },
        'totals': {
            'gas_injection_revenue': 0.0,
            'water_penalty': 0.0,
            'gas_penalty': 0.0,
            'npv': 0.0
        }
    }
    
    # Run simulation and collect per-step data
    # (Implementation would call rom_rollout and process results)
    
    return breakdown
