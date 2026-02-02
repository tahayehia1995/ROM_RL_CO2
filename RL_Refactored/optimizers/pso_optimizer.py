"""
Particle Swarm Optimization (PSO) Optimizer
=============================================

PSO optimizer for reservoir production optimization.
Implements a standard PSO algorithm with optional pyswarms backend.

PSO is particularly effective for:
- Global optimization without gradients
- Problems with many local optima
- Parallelizable objective evaluations
- Continuous optimization problems

References:
- Kennedy & Eberhart (1995) - Particle Swarm Optimization
- Shi & Eberhart (1998) - A Modified Particle Swarm Optimizer
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any

from .base_optimizer import BaseOptimizer, OptimizationResult

# Check if pyswarms library is available
try:
    import pyswarms as ps
    from pyswarms.single.global_best import GlobalBestPSO
    PYSWARMS_AVAILABLE = True
except ImportError:
    PYSWARMS_AVAILABLE = False
    ps = None


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization for reservoir production optimization.
    
    Uses either pyswarms library (if available) or a built-in implementation.
    
    Key Features:
    - Population-based global search
    - No gradient information required
    - Balances exploration (global search) and exploitation (local refinement)
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
        # PSO specific parameters
        n_particles: int = 30,
        c1: float = 2.0,  # Cognitive parameter (personal best attraction)
        c2: float = 2.0,  # Social parameter (global best attraction)
        w: float = 0.7,   # Inertia weight
        w_decay: float = 0.99,  # Inertia decay per iteration
        v_max: float = 0.2,  # Maximum velocity (as fraction of range)
        # Stopping criteria
        n_stagnation: int = 20,  # Stop after N iterations without improvement
        seed: Optional[int] = None,
        verbose: int = 1
    ):
        """
        Initialize PSO optimizer.
        
        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            max_iterations: Maximum iterations/generations
            tolerance: Convergence tolerance
            action_ranges: Optional well control bounds
            
            PSO specific:
                n_particles: Number of particles in swarm
                c1: Cognitive parameter (attraction to personal best)
                c2: Social parameter (attraction to global best)
                w: Initial inertia weight (momentum)
                w_decay: Inertia decay factor per iteration
                v_max: Maximum velocity as fraction of search range
            
            Stopping:
                n_stagnation: Stop if no improvement for N iterations
                
            seed: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=summary)
        """
        super().__init__(rom_model, config, norm_params, device, action_ranges)
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.w_decay = w_decay
        self.v_max = v_max
        self.n_stagnation = n_stagnation
        self.seed = seed
        self.verbose = verbose
        
        # Storage for optimization history
        self.history = {
            'objectives': [],
            'best_objectives': [],
            'mean_objectives': [],
            'sampled_indices': []
        }
        
        # For RL-like sampling
        self.z0_ensemble = None
        self.total_samples_count = {}
        self.current_iteration = 0
    
    def _get_random_realization(self) -> Tuple[torch.Tensor, int]:
        """
        Get a single random realization from the Z0 ensemble.
        
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
    
    def _evaluate_particle(self, x: np.ndarray) -> float:
        """
        Evaluate objective for a single particle.
        
        Uses RL-like sampling: picks one random case per evaluation.
        
        Args:
            x: Control vector in normalized [0,1] space
            
        Returns:
            Negative objective (PSO minimizes, we want to maximize NPV)
        """
        # Get random realization (RL-like sampling)
        z0, idx = self._get_random_realization()
        
        # Reshape controls
        controls = x.reshape(self.num_steps, self.num_controls)
        
        # Evaluate objective
        obj, _ = self.evaluate_objective(controls, z0)
        
        # Track sampling
        self.history['sampled_indices'].append(idx)
        
        # Return negative (PSO minimizes)
        return -obj
    
    def _evaluate_swarm(self, swarm: np.ndarray) -> np.ndarray:
        """
        Evaluate all particles in the swarm.
        
        Args:
            swarm: Array of shape (n_particles, n_dims)
            
        Returns:
            Array of fitness values (n_particles,)
        """
        n_particles = swarm.shape[0]
        fitness = np.zeros(n_particles)
        
        for i in range(n_particles):
            fitness[i] = self._evaluate_particle(swarm[i])
        
        return fitness
    
    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> OptimizationResult:
        """
        Run PSO optimization.
        
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
            'mean_objectives': [],
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
        
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Print header
        print(f"\n{'='*60}")
        print(f"Particle Swarm Optimization (RL-like Sampling)")
        print(f"{'='*60}")
        print(f"Cases available: {actual_cases}")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables: {num_vars}")
        print(f"Particles: {self.n_particles}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Parameters: c1={self.c1}, c2={self.c2}, w={self.w}")
        print(f"\nPhysical Ranges:")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, {self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, {self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'='*60}\n")
        
        # Evaluate initial objective
        x0 = self.generate_initial_guess(num_steps, strategy='midpoint')
        z0_first = self.z0_ensemble[0:1]
        initial_obj, _ = self.evaluate_objective(x0.reshape(num_steps, self.num_controls), z0_first)
        print(f"Initial objective (midpoint): {initial_obj:.6f}")
        
        # Run PSO
        if PYSWARMS_AVAILABLE:
            result = self._run_pyswarms(num_vars)
        else:
            result = self._run_builtin_pso(num_vars)
        
        best_position, best_cost = result
        
        # Extract optimal solution
        optimal_controls_normalized = best_position.reshape(num_steps, self.num_controls)
        optimal_controls_physical = self.controls_normalized_to_physical(optimal_controls_normalized)
        initial_controls_physical = self.controls_normalized_to_physical(x0.reshape(num_steps, self.num_controls))
        
        # Final objective (convert from negative)
        final_obj = -best_cost
        
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
            gradient_norm_history=[],  # No gradients for PSO
            control_history=[],
            num_iterations=self.current_iteration,
            num_function_evaluations=self.function_eval_count,
            num_gradient_evaluations=0,
            total_time_seconds=total_time,
            convergence_achieved=True,
            termination_reason='Max iterations reached' if self.current_iteration >= self.max_iterations else 'Converged',
            optimizer_type='PSO',
            optimizer_params={
                'n_particles': self.n_particles,
                'c1': self.c1,
                'c2': self.c2,
                'w': self.w,
                'max_iterations': self.max_iterations,
                'backend': 'pyswarms' if PYSWARMS_AVAILABLE else 'builtin'
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
        self._print_sampling_statistics()
        
        return optimization_result
    
    def _run_pyswarms(self, num_vars: int) -> Tuple[np.ndarray, float]:
        """Run optimization using pyswarms library."""
        print(f"\nStarting PSO (pyswarms backend)...\n")
        
        # Define bounds
        lb = np.zeros(num_vars)
        ub = np.ones(num_vars)
        bounds = (lb, ub)
        
        # PSO options
        options = {
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w
        }
        
        # Create optimizer
        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=num_vars,
            options=options,
            bounds=bounds
        )
        
        # Define cost function for pyswarms (takes whole swarm)
        def cost_func(swarm):
            return self._evaluate_swarm(swarm)
        
        # Run optimization with progress
        best_cost = float('inf')
        best_position = None
        stagnation_count = 0
        
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            
            # Perform one iteration
            cost, pos = optimizer.optimize(cost_func, iters=1, verbose=False)
            
            # Track history
            self.history['best_objectives'].append(-cost)
            
            # Check improvement
            if cost < best_cost - self.tolerance:
                best_cost = cost
                best_position = pos
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Progress reporting
            if self.verbose > 0 and (iteration < 5 or (iteration + 1) % 10 == 0):
                print(f"Iteration {iteration + 1:4d}: Best = {-best_cost:.6f}")
                print(f"   Unique cases sampled: {len(self.total_samples_count)}/{self.z0_ensemble.shape[0]}")
            
            # Check stagnation
            if stagnation_count >= self.n_stagnation:
                print(f"\nStopped: No improvement for {self.n_stagnation} iterations")
                break
        
        return best_position, best_cost
    
    def _run_builtin_pso(self, num_vars: int) -> Tuple[np.ndarray, float]:
        """Run optimization using built-in PSO implementation."""
        print(f"\nStarting PSO (built-in implementation)...\n")
        
        # Initialize swarm
        positions = np.random.uniform(0, 1, (self.n_particles, num_vars))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.n_particles, num_vars))
        
        # Personal bests
        personal_best_positions = positions.copy()
        personal_best_costs = self._evaluate_swarm(positions)
        
        # Global best
        best_idx = np.argmin(personal_best_costs)
        global_best_position = personal_best_positions[best_idx].copy()
        global_best_cost = personal_best_costs[best_idx]
        
        # Track history
        self.history['best_objectives'].append(-global_best_cost)
        
        # Inertia weight
        w = self.w
        stagnation_count = 0
        
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            
            # Update velocities and positions
            r1 = np.random.random((self.n_particles, num_vars))
            r2 = np.random.random((self.n_particles, num_vars))
            
            # Velocity update
            cognitive = self.c1 * r1 * (personal_best_positions - positions)
            social = self.c2 * r2 * (global_best_position - positions)
            velocities = w * velocities + cognitive + social
            
            # Clamp velocities
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            
            # Position update
            positions = positions + velocities
            
            # Clamp positions to bounds [0, 1]
            positions = np.clip(positions, 0, 1)
            
            # Evaluate new positions
            costs = self._evaluate_swarm(positions)
            
            # Update personal bests
            improved = costs < personal_best_costs
            personal_best_positions[improved] = positions[improved]
            personal_best_costs[improved] = costs[improved]
            
            # Update global best
            best_idx = np.argmin(personal_best_costs)
            if personal_best_costs[best_idx] < global_best_cost - self.tolerance:
                global_best_position = personal_best_positions[best_idx].copy()
                global_best_cost = personal_best_costs[best_idx]
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Track history
            self.history['best_objectives'].append(-global_best_cost)
            self.history['mean_objectives'].append(-np.mean(costs))
            
            # Decay inertia
            w *= self.w_decay
            
            # Progress reporting
            if self.verbose > 0 and (iteration < 5 or (iteration + 1) % 10 == 0):
                print(f"Iteration {iteration + 1:4d}: Best = {-global_best_cost:.6f}, "
                      f"Mean = {-np.mean(costs):.6f}, w = {w:.4f}")
                print(f"   Unique cases sampled: {len(self.total_samples_count)}/{self.z0_ensemble.shape[0]}")
            
            # Check stagnation
            if stagnation_count >= self.n_stagnation:
                print(f"\nStopped: No improvement for {self.n_stagnation} iterations")
                break
        
        return global_best_position, global_best_cost
    
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


def check_pyswarms_available():
    """Check if pyswarms library is available."""
    return PYSWARMS_AVAILABLE
