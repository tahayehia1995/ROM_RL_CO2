"""
Scipy Global Optimizers
========================

Unified optimizer class for scipy global optimization methods:
- Differential Evolution (DE)
- Dual Annealing (SA with local search)
- Basin Hopping (global + local)

All optimizers use the same objective function, bounds, and ROM rollout as LS-SQP,
ensuring fair comparison with RL approaches.
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import differential_evolution, dual_annealing, basinhopping, Bounds
from dataclasses import dataclass

from .base_optimizer import BaseOptimizer, OptimizationResult


class ScipyOptimizer(BaseOptimizer):
    """
    Unified optimizer class for scipy global optimization methods.
    
    Supports:
    - differential_evolution: Population-based global search
    - dual_annealing: Simulated annealing with local refinement
    - basinhopping: Monte Carlo + local optimization
    
    All methods use:
    - Same normalized [0,1] control space as LS-SQP
    - Same objective function (NPV) as RL
    - Same ROM rollout for predictions
    - RL-like random case sampling from Z0 pool
    """
    
    SUPPORTED_METHODS = ['differential_evolution', 'dual_annealing', 'basinhopping']
    
    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        method: str = 'differential_evolution',
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        action_ranges: Optional[Dict] = None,
        # Differential Evolution parameters
        popsize: int = 15,
        mutation: float = 0.8,
        recombination: float = 0.7,
        strategy: str = 'best1bin',
        # Dual Annealing parameters
        initial_temp: float = 5230.0,
        restart_temp_ratio: float = 2e-5,
        visit: float = 2.62,
        accept: float = -5.0,
        # Basin Hopping parameters
        niter_basinhopping: int = 100,
        T: float = 1.0,
        stepsize: float = 0.5,
        # Common parameters
        seed: Optional[int] = None,
        workers: int = 1,
        disp: bool = True
    ):
        """
        Initialize Scipy global optimizer.
        
        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            method: Optimization method ('differential_evolution', 'dual_annealing', 'basinhopping')
            max_iterations: Maximum iterations/generations
            tolerance: Convergence tolerance
            action_ranges: Optional well control bounds
            
            Differential Evolution specific:
                popsize: Population size multiplier (actual = popsize * num_vars)
                mutation: Mutation constant [0, 2]
                recombination: Crossover probability [0, 1]
                strategy: DE strategy ('best1bin', 'best2bin', 'rand1bin', etc.)
            
            Dual Annealing specific:
                initial_temp: Initial temperature
                restart_temp_ratio: Restart temperature ratio
                visit: Visiting distribution parameter
                accept: Acceptance distribution parameter
            
            Basin Hopping specific:
                niter_basinhopping: Number of basin hopping iterations
                T: Temperature for Metropolis criterion
                stepsize: Initial step size for random displacement
            
            Common:
                seed: Random seed for reproducibility
                workers: Number of parallel workers (DE only)
                disp: Display progress
        """
        super().__init__(rom_model, config, norm_params, device, action_ranges)
        
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Use one of: {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        self.workers = workers
        self.disp = disp
        
        # Method-specific parameters
        self.de_params = {
            'popsize': popsize,
            'mutation': mutation,
            'recombination': recombination,
            'strategy': strategy
        }
        
        self.da_params = {
            'initial_temp': initial_temp,
            'restart_temp_ratio': restart_temp_ratio,
            'visit': visit,
            'accept': accept
        }
        
        self.bh_params = {
            'niter': niter_basinhopping,
            'T': T,
            'stepsize': stepsize
        }
        
        # Storage for optimization history
        self.history = {
            'objectives': [],
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
        one randomly selected case from the pool, just like RL picks
        one random case at each episode reset.
        
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
        Objective function wrapper for scipy optimizers.
        
        Uses RL-like sampling: picks one random case per evaluation.
        
        Args:
            x: Control vector in normalized [0,1] space
            
        Returns:
            Negative objective (scipy minimizes, we want to maximize NPV)
        """
        # Get random realization (RL-like sampling)
        z0, idx = self._get_random_realization()
        
        # Reshape controls
        controls = x.reshape(self.num_steps, self.num_controls)
        
        # Evaluate objective
        obj, _ = self.evaluate_objective(controls, z0)
        
        # Track history (occasionally)
        if len(self.history['objectives']) == 0 or \
           (len(self.history['objectives']) > 0 and abs(obj - self.history['objectives'][-1]) > 1e-8):
            self.history['objectives'].append(obj)
            self.history['sampled_indices'].append(idx)
        
        # Return negative (scipy minimizes)
        return -obj
    
    def _callback(self, xk, convergence=None):
        """Callback for progress reporting."""
        self.current_iteration += 1
        
        if self.disp:
            if self.current_iteration <= 5 or self.current_iteration % 10 == 0:
                # Evaluate current best
                z0, idx = self._get_random_realization()
                controls = xk.reshape(self.num_steps, self.num_controls)
                obj, _ = self.evaluate_objective(controls, z0)
                
                print(f"Iteration {self.current_iteration:4d}: Objective = {obj:.6f} (Case {idx})")
                print(f"   Unique cases sampled: {len(self.total_samples_count)}/{self.z0_ensemble.shape[0]}")
    
    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> OptimizationResult:
        """
        Run scipy global optimization.
        
        Args:
            z0_options: Tensor of initial states (num_cases, latent_dim)
            num_steps: Number of control timesteps (default: 30)
            
        Returns:
            OptimizationResult with optimal controls and performance data
        """
        start_time = time.time()
        self.reset_counters()
        self.history = {'objectives': [], 'controls': [], 'sampled_indices': []}
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
        method_names = {
            'differential_evolution': 'Differential Evolution',
            'dual_annealing': 'Dual Annealing',
            'basinhopping': 'Basin Hopping'
        }
        
        print(f"\n{'='*60}")
        print(f"{method_names[self.method]} Optimization (RL-like Sampling)")
        print(f"{'='*60}")
        print(f"Cases available: {actual_cases}")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables: {num_vars}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Tolerance: {self.tolerance}")
        print(f"\nPhysical Ranges:")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, {self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, {self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'='*60}\n")
        
        # Get bounds (all in [0, 1])
        bounds_list = self.get_bounds(num_steps)
        
        # Evaluate initial objective (at midpoint)
        x0 = self.generate_initial_guess(num_steps, strategy='midpoint')
        z0_first = self.z0_ensemble[0:1]
        initial_obj, _ = self.evaluate_objective(x0.reshape(num_steps, self.num_controls), z0_first)
        print(f"Initial objective (midpoint): {initial_obj:.6f}")
        
        # Run optimization based on method
        if self.method == 'differential_evolution':
            result = self._run_differential_evolution(bounds_list, num_vars)
        elif self.method == 'dual_annealing':
            result = self._run_dual_annealing(bounds_list, num_vars)
        elif self.method == 'basinhopping':
            result = self._run_basinhopping(bounds_list, num_vars, x0)
        
        # Extract optimal solution
        optimal_controls_normalized = result.x.reshape(num_steps, self.num_controls)
        optimal_controls_physical = self.controls_normalized_to_physical(optimal_controls_normalized)
        initial_controls_physical = self.controls_normalized_to_physical(x0.reshape(num_steps, self.num_controls))
        
        # Evaluate final solution
        final_obj = -result.fun  # Convert back from minimization
        
        # Get trajectory for visualization (use first case)
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
            objective_history=self.history['objectives'],
            gradient_norm_history=[],  # No gradients for these methods
            control_history=self.history['controls'],
            num_iterations=self.current_iteration,
            num_function_evaluations=self.function_eval_count,
            num_gradient_evaluations=0,
            total_time_seconds=total_time,
            convergence_achieved=result.success if hasattr(result, 'success') else True,
            termination_reason=result.message if hasattr(result, 'message') else 'Completed',
            optimizer_type=method_names[self.method],
            optimizer_params=self._get_method_params(),
            num_realizations=actual_cases,
            initial_controls=initial_controls_physical,
            initial_objective=initial_obj,
            economic_breakdown=economic_breakdown
        )
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(optimization_result.summary())
        self._print_sampling_statistics()
        
        return optimization_result
    
    def _run_differential_evolution(self, bounds_list, num_vars):
        """Run scipy differential_evolution."""
        print(f"\nStarting Differential Evolution...")
        print(f"  Population size: {self.de_params['popsize']} × {num_vars} = {self.de_params['popsize'] * num_vars}")
        print(f"  Mutation: {self.de_params['mutation']}")
        print(f"  Recombination: {self.de_params['recombination']}")
        print(f"  Strategy: {self.de_params['strategy']}\n")
        
        result = differential_evolution(
            self._objective_wrapper,
            bounds=bounds_list,
            maxiter=self.max_iterations,
            tol=self.tolerance,
            popsize=self.de_params['popsize'],
            mutation=self.de_params['mutation'],
            recombination=self.de_params['recombination'],
            strategy=self.de_params['strategy'],
            seed=self.seed,
            workers=self.workers,
            disp=self.disp,
            callback=self._callback,
            polish=True,  # Use local optimization at the end
            updating='deferred' if self.workers > 1 else 'immediate'
        )
        
        return result
    
    def _run_dual_annealing(self, bounds_list, num_vars):
        """Run scipy dual_annealing."""
        print(f"\nStarting Dual Annealing...")
        print(f"  Initial temp: {self.da_params['initial_temp']}")
        print(f"  Restart temp ratio: {self.da_params['restart_temp_ratio']}")
        print(f"  Visit: {self.da_params['visit']}")
        print(f"  Accept: {self.da_params['accept']}\n")
        
        result = dual_annealing(
            self._objective_wrapper,
            bounds=bounds_list,
            maxiter=self.max_iterations,
            initial_temp=self.da_params['initial_temp'],
            restart_temp_ratio=self.da_params['restart_temp_ratio'],
            visit=self.da_params['visit'],
            accept=self.da_params['accept'],
            seed=self.seed,
            callback=lambda x, f, context: self._callback(x)
        )
        
        return result
    
    def _run_basinhopping(self, bounds_list, num_vars, x0):
        """Run scipy basinhopping."""
        print(f"\nStarting Basin Hopping...")
        print(f"  Iterations: {self.bh_params['niter']}")
        print(f"  Temperature: {self.bh_params['T']}")
        print(f"  Step size: {self.bh_params['stepsize']}\n")
        
        # Basin hopping needs bounds as minimizer_kwargs
        bounds = [(b[0], b[1]) for b in bounds_list]
        
        result = basinhopping(
            self._objective_wrapper,
            x0,
            niter=self.bh_params['niter'],
            T=self.bh_params['T'],
            stepsize=self.bh_params['stepsize'],
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'bounds': bounds
            },
            callback=lambda x, f, accept: self._callback(x),
            seed=self.seed,
            disp=self.disp
        )
        
        return result
    
    def _get_method_params(self) -> Dict:
        """Get parameters for current method."""
        if self.method == 'differential_evolution':
            return {**self.de_params, 'max_iterations': self.max_iterations, 'tolerance': self.tolerance}
        elif self.method == 'dual_annealing':
            return {**self.da_params, 'max_iterations': self.max_iterations}
        elif self.method == 'basinhopping':
            return {**self.bh_params}
        return {}
    
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
