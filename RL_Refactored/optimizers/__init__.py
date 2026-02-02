"""
Classical Optimizers Package for ROM-based Reservoir Optimization
==================================================================

This package provides classical optimization methods for comparison with RL approaches.
All optimizers use the same ROM model, objective function, and constraints as the RL agent.

Available Optimizers:
- LS-SQP with StoSAG: Line-Search Sequential Quadratic Programming with 
                      Stochastic Simplex Approximate Gradients (gradient-based)
- Differential Evolution: Population-based global optimization (scipy)
- Dual Annealing: Simulated annealing with local search (scipy)
- Basin Hopping: Monte Carlo + local optimization (scipy)
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy (requires 'cma' library)
- PSO: Particle Swarm Optimization (built-in or pyswarms)
- GA: Genetic Algorithm (built-in or DEAP)

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
from .scipy_optimizer import ScipyOptimizer
from .cmaes_optimizer import CMAESOptimizer, check_cma_available
from .pso_optimizer import PSOOptimizer, check_pyswarms_available
from .ga_optimizer import GAOptimizer, check_deap_available

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
        
    Supported optimizer types:
        - 'LS-SQP-StoSAG': Gradient-based with SPSA/StoSAG gradients
        - 'Differential-Evolution': Population-based global search
        - 'Dual-Annealing': Simulated annealing with local refinement
        - 'Basin-Hopping': Monte Carlo + local optimization
        - 'CMA-ES': Covariance Matrix Adaptation (requires 'cma' library)
        - 'PSO': Particle Swarm Optimization (built-in or pyswarms)
        - 'GA': Genetic Algorithm (built-in or DEAP)
    """
    optimizer_type = config.get('optimizer_type', 'LS-SQP-StoSAG')
    
    # Extract common parameters from dashboard config
    action_ranges = config.get('action_ranges', None)
    optimizer_params = config.get('optimizer_params', {})
    stosag_params = config.get('stosag_params', {})
    sqp_params = config.get('sqp_params', {})
    
    # Common arguments for all optimizers
    common_args = {
        'rom_model': config['rom_model'],
        'config': config['config'],
        'norm_params': config['norm_params'],
        'device': config['device'],
        'action_ranges': action_ranges
    }
    
    if optimizer_type == 'LS-SQP-StoSAG':
        return LSSQPStoSAGOptimizer(
            **common_args,
            num_realizations=stosag_params.get('num_realizations', 1),
            perturbation_size=stosag_params.get('perturbation_size', 0.01),
            max_iterations=sqp_params.get('max_iterations', 100),
            tolerance=sqp_params.get('tolerance', 1e-6),
            gradient_type=stosag_params.get('gradient_type', 'spsa'),
            spsa_num_samples=stosag_params.get('spsa_num_samples', 5)
        )
    
    elif optimizer_type == 'Differential-Evolution':
        return ScipyOptimizer(
            **common_args,
            method='differential_evolution',
            max_iterations=optimizer_params.get('max_iterations', 100),
            tolerance=optimizer_params.get('tolerance', 1e-6),
            popsize=optimizer_params.get('popsize', 15),
            mutation=optimizer_params.get('mutation', 0.8),
            recombination=optimizer_params.get('recombination', 0.7),
            strategy=optimizer_params.get('strategy', 'best1bin')
        )
    
    elif optimizer_type == 'Dual-Annealing':
        return ScipyOptimizer(
            **common_args,
            method='dual_annealing',
            max_iterations=optimizer_params.get('max_iterations', 100),
            initial_temp=optimizer_params.get('initial_temp', 5230.0),
            restart_temp_ratio=optimizer_params.get('restart_temp_ratio', 2e-5),
            visit=optimizer_params.get('visit', 2.62),
            accept=optimizer_params.get('accept', -5.0)
        )
    
    elif optimizer_type == 'Basin-Hopping':
        return ScipyOptimizer(
            **common_args,
            method='basinhopping',
            niter_basinhopping=optimizer_params.get('niter', 100),
            T=optimizer_params.get('T', 1.0),
            stepsize=optimizer_params.get('stepsize', 0.5)
        )
    
    elif optimizer_type == 'CMA-ES':
        # Check if CMA library is available
        if not check_cma_available():
            raise ImportError(
                "CMA-ES requires the 'cma' library. Install with: pip install cma\n"
                "Alternatively, select Differential Evolution or Dual Annealing."
            )
        
        popsize = optimizer_params.get('popsize', 0)
        return CMAESOptimizer(
            **common_args,
            max_iterations=optimizer_params.get('max_iterations', 100),
            tolerance=optimizer_params.get('tolerance', 1e-6),
            sigma0=optimizer_params.get('sigma0', 0.3),
            popsize=popsize if popsize > 0 else None  # 0 means automatic
        )
    
    elif optimizer_type == 'PSO':
        return PSOOptimizer(
            **common_args,
            max_iterations=optimizer_params.get('max_iterations', 100),
            tolerance=optimizer_params.get('tolerance', 1e-6),
            n_particles=optimizer_params.get('n_particles', 30),
            c1=optimizer_params.get('c1', 2.0),
            c2=optimizer_params.get('c2', 2.0),
            w=optimizer_params.get('w', 0.7)
        )
    
    elif optimizer_type == 'GA':
        return GAOptimizer(
            **common_args,
            max_iterations=optimizer_params.get('max_iterations', 100),
            tolerance=optimizer_params.get('tolerance', 1e-6),
            population_size=optimizer_params.get('population_size', 50),
            crossover_prob=optimizer_params.get('crossover_prob', 0.7),
            mutation_prob=optimizer_params.get('mutation_prob', 0.2),
            elitism=optimizer_params.get('elitism', 2)
        )
    
    else:
        raise ValueError(
            f"Unknown optimizer type: '{optimizer_type}'. "
            f"Available types: LS-SQP-StoSAG, Differential-Evolution, "
            f"Dual-Annealing, Basin-Hopping, CMA-ES, PSO, GA"
        )


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
    'ScipyOptimizer',
    'CMAESOptimizer',
    'PSOOptimizer',
    'GAOptimizer',
    
    # Factory functions
    'create_optimizer',
    'run_optimization',
    
    # Utility functions
    'compute_objective',
    'compute_trajectory_npv',
    'get_bounds_from_config',
    'create_nonlinear_constraints',
    'check_cma_available',
    'check_pyswarms_available',
    'check_deap_available',
]
