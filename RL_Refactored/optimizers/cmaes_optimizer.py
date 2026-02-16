"""
CMA-ES Optimizer
=================

Covariance Matrix Adaptation Evolution Strategy optimizer for reservoir optimization.
Uses the `cma` library for the core algorithm.

CMA-ES is particularly effective for:
- Continuous, nonlinear optimization
- Medium-dimensional problems (10-1000 variables)
- Problems with complex, multimodal landscapes
- Cases where gradients are not available or unreliable

References:
- Hansen & Ostermeier (2001) - Completely Derandomized Self-Adaptation in Evolution Strategies
- Hansen (2016) - The CMA Evolution Strategy: A Tutorial
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .base_optimizer import BaseOptimizer, OptimizationResult

# Check if cma library is available
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    cma = None


class CMAESOptimizer(BaseOptimizer):
    """
    CMA-ES optimizer for reservoir production optimization.
    
    Uses the same objective function, bounds, and ROM rollout as other optimizers,
    ensuring fair comparison with RL approaches.
    
    Key Features:
    - Adaptive step size control
    - Covariance matrix adaptation (learns variable correlations)
    - Robust to local optima
    - No gradient information required
    - RL-like random case sampling from Z0 pool
    """
    
    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        action_ranges: Optional[Dict] = None,
        # CMA-ES specific parameters
        sigma0: float = 0.3,
        popsize: Optional[int] = None,  # None = auto (4 + 3*ln(n))
        seed: Optional[int] = None,
        # Stopping criteria
        tolx: float = 1e-11,
        tolfun: float = 1e-11,
        tolfunhist: float = 1e-12,
        # Display
        verbose: int = 1,
        init_strategy: str = 'midpoint'
    ):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            max_iterations: Maximum generations
            tolerance: Convergence tolerance
            action_ranges: Optional well control bounds
            
            CMA-ES specific:
                sigma0: Initial step size (standard deviation)
                        Recommended: 0.2-0.5 for [0,1] bounded problems
                popsize: Population size (None = automatic)
                seed: Random seed for reproducibility
            
            Stopping criteria:
                tolx: Tolerance on x changes
                tolfun: Tolerance on function value changes
                tolfunhist: Tolerance on function history
            
            verbose: Verbosity level (0=silent, 1=summary, 3=full)
            init_strategy: Initialization strategy ('midpoint', 'random', 'naive_zero', etc.)
        """
        super().__init__(rom_model, config, norm_params, device, action_ranges, init_strategy)
        
        if not CMA_AVAILABLE:
            raise ImportError(
                "CMA-ES requires the 'cma' library. Install with: pip install cma\n"
                "Alternatively, use Differential Evolution or Dual Annealing which are built into scipy."
            )
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.sigma0 = sigma0
        self.popsize = popsize
        self.seed = seed
        self.verbose = verbose
        
        # Stopping criteria
        self.tolx = tolx
        self.tolfun = tolfun
        self.tolfunhist = tolfunhist
        
        # Storage for optimization history
        self.history = {
            'objectives': [],
            'best_objectives': [],
            'sigma': [],
            'controls': [],
            'sampled_indices': []
        }
        
        # For RL-like sampling
        self.z0_ensemble = None
        self.total_samples_count = {}
        self.current_iteration = 0
    
    def _get_random_realization(self) -> Tuple[torch.Tensor, int]:
        """
        Get a single random realization from the Z0 ensemble.
        
        This implements RL-like sampling: each function evaluation uses
        one randomly selected case from the pool.
        
        Returns:
            z0: Single initial state tensor (1, latent_dim)
            idx: Index of selected case
        """
        num_cases = self.z0_ensemble.shape[0]
        idx = np.random.randint(0, num_cases)
        z0 = self.z0_ensemble[idx:idx+1]
        
        # Track sampling statistics
        if idx not in self.total_samples_count:
            self.total_samples_count[idx] = 0
        self.total_samples_count[idx] += 1
        
        return z0, idx
    
    def _objective_wrapper(self, x: np.ndarray) -> float:
        """
        Objective function wrapper for CMA-ES.
        
        Uses RL-like sampling: picks one random case per evaluation.
        
        Args:
            x: Control vector in normalized [0,1] space
            
        Returns:
            Negative objective (CMA-ES minimizes, we want to maximize NPV)
        """
        # Clip to bounds (CMA-ES can sometimes exceed bounds slightly)
        x = np.clip(x, 0.0, 1.0)
        
        # Get random realization (RL-like sampling)
        z0, idx = self._get_random_realization()
        
        # Reshape controls
        controls = x.reshape(self.num_steps, self.num_controls)
        
        # Evaluate objective
        obj, _ = self.evaluate_objective(controls, z0)
        
        # Track sampling
        self.history['sampled_indices'].append(idx)
        
        # Return negative (CMA-ES minimizes)
        return -obj
    
    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> OptimizationResult:
        """
        Run CMA-ES optimization.
        
        Args:
            z0_options: Tensor of initial states (num_cases, latent_dim)
            num_steps: Number of control timesteps (default: 30)
            
        Returns:
            OptimizationResult with optimal controls and performance data
        """
        start_time = time.time()
        self.reset_counters()
        self.history = {
            'objectives': [],
            'best_objectives': [],
            'sigma': [],
            'controls': [],
            'sampled_indices': []
        }
        self.total_samples_count = {}
        self.current_iteration = 0
        
        # Setup
        num_steps = num_steps or 30
        self.num_steps = num_steps
        
        if z0_options is None:
            raise ValueError("z0_options must be provided")
        
        self.z0_ensemble = z0_options
        actual_cases = z0_options.shape[0]
        num_vars = num_steps * self.num_controls
        
        # Print header
        print(f"\n{'='*60}")
        print(f"CMA-ES Optimization (RL-like Sampling)")
        print(f"{'='*60}")
        print(f"Cases available: {actual_cases}")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables: {num_vars}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Initial sigma: {self.sigma0}")
        
        # Determine population size
        if self.popsize is None:
            auto_popsize = int(4 + 3 * np.log(num_vars))
            print(f"Population size: {auto_popsize} (automatic)")
        else:
            print(f"Population size: {self.popsize}")
        
        print(f"\nPhysical Ranges:")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, {self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, {self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'='*60}\n")
        
        # Initial guess using configured strategy
        x0 = self.generate_initial_guess(num_steps, strategy=self.init_strategy)
        
        # Evaluate initial objective
        z0_first = self.z0_ensemble[0:1]
        initial_obj, _ = self.evaluate_objective(x0.reshape(num_steps, self.num_controls), z0_first)
        print(f"Initial objective (midpoint): {initial_obj:.6f}")
        
        # CMA-ES options
        opts = {
            'maxiter': self.max_iterations,
            'tolfun': self.tolfun,
            'tolx': self.tolx,
            'tolfunhist': self.tolfunhist,
            'bounds': [0, 1],  # Bounds in [0, 1]
            'seed': self.seed if self.seed else np.random.randint(1, 10000),
            'verbose': self.verbose,
            'verb_disp': 1 if self.verbose > 0 else 0,
            'verb_log': 0  # Disable file logging
        }
        
        if self.popsize is not None:
            opts['popsize'] = self.popsize
        
        print(f"\nStarting CMA-ES optimization...\n")
        
        # Create CMA-ES instance
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        
        # Run optimization
        iteration = 0
        while not es.stop():
            # Get candidate solutions
            solutions = es.ask()
            
            # Evaluate all solutions
            fitnesses = [self._objective_wrapper(x) for x in solutions]
            
            # Update CMA-ES with evaluations
            es.tell(solutions, fitnesses)
            
            # Track history
            self.history['objectives'].append(-min(fitnesses))  # Best this generation
            self.history['best_objectives'].append(-es.result.fbest)  # Best overall
            self.history['sigma'].append(es.sigma)
            
            iteration += 1
            self.current_iteration = iteration
            
            # Progress reporting
            if self.verbose > 0 and (iteration <= 5 or iteration % 10 == 0):
                print(f"Generation {iteration:4d}: Best = {-es.result.fbest:.6f}, "
                      f"Gen Best = {-min(fitnesses):.6f}, σ = {es.sigma:.4f}")
                print(f"   Unique cases sampled: {len(self.total_samples_count)}/{actual_cases}")
        
        # Get best solution
        result = es.result
        optimal_x = result.xbest
        
        # Ensure within bounds
        optimal_x = np.clip(optimal_x, 0.0, 1.0)
        
        # Extract optimal solution
        optimal_controls_normalized = optimal_x.reshape(num_steps, self.num_controls)
        optimal_controls_physical = self.controls_normalized_to_physical(optimal_controls_normalized)
        initial_controls_physical = self.controls_normalized_to_physical(x0.reshape(num_steps, self.num_controls))
        
        # Final objective
        final_obj = -result.fbest
        
        # Get trajectory for visualization
        _, trajectory = self.evaluate_objective(optimal_controls_normalized, z0_first, return_trajectory=True)
        
        # Decode spatial states
        spatial_states = self.decode_spatial_states(trajectory['states'])
        
        # Compute economic breakdown
        from .objective import compute_trajectory_npv
        observations_array = np.array(trajectory['observations'])
        _, economic_breakdown = compute_trajectory_npv(
            trajectory['observations'],
            optimal_controls_physical,
            self.config,
            self.num_prod,
            self.num_inj
        )
        
        total_time = time.time() - start_time
        
        # Build result
        optimization_result = OptimizationResult(
            optimal_controls=optimal_controls_physical,
            optimal_objective=final_obj,
            optimal_states=torch.stack(trajectory['states']),
            optimal_spatial_states=spatial_states,
            optimal_observations=observations_array,
            objective_history=self.history['best_objectives'],
            gradient_norm_history=self.history['sigma'],  # Use sigma as proxy
            control_history=self.history['controls'],
            num_iterations=iteration,
            num_function_evaluations=self.function_eval_count,
            num_gradient_evaluations=0,
            total_time_seconds=total_time,
            convergence_achieved=True,
            termination_reason=str(es.stop()),
            optimizer_type='CMA-ES',
            optimizer_params={
                'sigma0': self.sigma0,
                'popsize': self.popsize or int(4 + 3 * np.log(num_vars)),
                'max_iterations': self.max_iterations,
                'final_sigma': es.sigma
            },
            num_realizations=actual_cases,
            initial_controls=initial_controls_physical,
            initial_objective=initial_obj,
            economic_breakdown=economic_breakdown
        )
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(optimization_result.summary())
        print(f"\nCMA-ES Stop Conditions: {es.stop()}")
        self._print_sampling_statistics()
        
        return optimization_result
    
    def _print_sampling_statistics(self):
        """Print RL-like sampling statistics."""
        print(f"\n{'='*60}")
        print(f"Sampling Statistics (RL-like)")
        print(f"{'='*60}")
        print(f"Total cases available: {self.z0_ensemble.shape[0]}")
        print(f"Unique cases sampled: {len(self.total_samples_count)}")
        print(f"Total samples drawn: {sum(self.total_samples_count.values())}")
        
        if self.total_samples_count:
            counts = list(self.total_samples_count.values())
            print(f"Coverage: {100 * len(self.total_samples_count) / self.z0_ensemble.shape[0]:.1f}%")
            print(f"Samples per case: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.2f}")
        print(f"{'='*60}\n")


def check_cma_available():
    """Check if CMA library is available."""
    return CMA_AVAILABLE
