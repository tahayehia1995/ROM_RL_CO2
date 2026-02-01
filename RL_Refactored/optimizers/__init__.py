"""
Classical Optimizers Package for ROM-based Reservoir Optimization
==================================================================

This package provides classical optimization methods for comparison with RL approaches.
All optimizers use the same ROM model, objective function, and constraints as the RL agent.

Available Optimizers:
- LS-SQP with StoSAG: Line-Search Sequential Quadratic Programming with 
                      Stochastic Simplex Approximate Gradients for robust optimization

Usage:
    from RL_Refactored.optimizers import (
        launch_optimizer_config_dashboard,
        create_optimizer,
        run_optimization,
        launch_optimizer_results_dashboard
    )
    
    # Step 1: Configure optimizer
    optimizer_config = launch_optimizer_config_dashboard()
    
    # Step 2: Create and run optimizer
    optimizer = create_optimizer(optimizer_config)
    result = run_optimization(optimizer)
    
    # Step 3: Visualize results
    viz = launch_optimizer_results_dashboard(result)
"""

# Import dashboard launch functions
from .dashboard_config import (
    launch_optimizer_config_dashboard,
    OptimizerConfigDashboard
)

from .dashboard_results import (
    launch_optimizer_results_dashboard,
    OptimizerResultsDashboard
)

# Import optimizer classes
from .base_optimizer import BaseOptimizer, OptimizationResult
from .ls_sqp_stosag import LSSQPStoSAGOptimizer

# Import utility functions
from .objective import compute_objective, compute_trajectory_npv
from .constraints import get_bounds_from_config, create_nonlinear_constraints


def create_optimizer(config):
    """
    Factory function to create optimizer based on configuration.
    
    Args:
        config: OptimizerConfigDashboard configuration dictionary
        
    Returns:
        BaseOptimizer instance
    """
    optimizer_type = config.get('optimizer_type', 'LS-SQP-StoSAG')
    
    # Extract action ranges from dashboard config
    action_ranges = config.get('action_ranges', None)
    stosag_params = config.get('stosag_params', {})
    sqp_params = config.get('sqp_params', {})
    
    if optimizer_type == 'LS-SQP-StoSAG':
        return LSSQPStoSAGOptimizer(
            rom_model=config['rom_model'],
            config=config['config'],
            norm_params=config['norm_params'],
            device=config['device'],
            num_realizations=stosag_params.get('num_realizations', 1),
            perturbation_size=stosag_params.get('perturbation_size', 0.01),
            max_iterations=sqp_params.get('max_iterations', 100),
            tolerance=sqp_params.get('tolerance', 1e-6),
            action_ranges=action_ranges,
            # New fast gradient options
            gradient_type=stosag_params.get('gradient_type', 'spsa'),  # SPSA is fast!
            spsa_num_samples=stosag_params.get('spsa_num_samples', 5)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def run_optimization(optimizer, z0_options=None, num_steps=None):
    """
    Run optimization with the configured optimizer.
    
    Args:
        optimizer: BaseOptimizer instance
        z0_options: Optional tensor of initial state options (uses optimizer's default if None)
        num_steps: Number of control timesteps (uses optimizer's default if None)
        
    Returns:
        OptimizationResult with optimal controls and performance data
    """
    return optimizer.optimize(z0_options=z0_options, num_steps=num_steps)


__all__ = [
    # Dashboard functions
    'launch_optimizer_config_dashboard',
    'launch_optimizer_results_dashboard',
    'OptimizerConfigDashboard',
    'OptimizerResultsDashboard',
    
    # Optimizer classes
    'BaseOptimizer',
    'OptimizationResult',
    'LSSQPStoSAGOptimizer',
    
    # Factory functions
    'create_optimizer',
    'run_optimization',
    
    # Utility functions
    'compute_objective',
    'compute_trajectory_npv',
    'get_bounds_from_config',
    'create_nonlinear_constraints',
]
