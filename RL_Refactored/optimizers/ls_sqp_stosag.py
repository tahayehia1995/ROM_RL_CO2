"""
LS-SQP with StoSAG Optimizer
============================

Line-Search Sequential Quadratic Programming with Stochastic Simplex 
Approximate Gradients for robust reservoir production optimization.

Key Features:
- StoSAG gradient estimation for noisy/stochastic objectives
- Multiple geological realizations for robust optimization
- scipy.optimize.minimize with SLSQP backend
- Bound constraints matching RL action ranges
- Optional nonlinear constraints

References:
- Fonseca et al. (2017) - A Stochastic Simplex Approximate Gradient (StoSAG) 
  for optimization under uncertainty
- Volkov & Voskov (2023) - Nonlinearly Constrained Life-Cycle Production 
  Optimization Using SQP with StoSAG
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass

from .base_optimizer import BaseOptimizer, OptimizationResult
from .constraints import get_bounds_from_config, create_nonlinear_constraints, project_to_bounds


class LSSQPStoSAGOptimizer(BaseOptimizer):
    """
    LS-SQP optimizer with StoSAG gradient estimation.
    
    Uses stochastic simplex approximate gradients for robust optimization
    over multiple geological realizations (initial state ensemble).
    """
    
    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        num_realizations: int = 10,
        perturbation_size: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        action_ranges: Optional[Dict] = None,
        use_parallel_realizations: bool = False,
        gradient_type: str = 'stosag',  # 'stosag', 'spsa', 'fd_central', 'fd_forward', 'adjoint'
        line_search_params: Optional[Dict] = None,
        control_parameterization: str = 'full',  # 'full', 'piecewise', 'polynomial'
        num_control_periods: int = 6,  # For piecewise parameterization
        spsa_num_samples: int = 5  # Number of SPSA gradient samples to average
    ):
        """
        Initialize LS-SQP optimizer with StoSAG.
        
        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            num_realizations: Number of geological realizations for robust optimization
            perturbation_size: Relative perturbation size for gradient estimation
            max_iterations: Maximum SQP iterations
            tolerance: Convergence tolerance on objective improvement
            action_ranges: Optional well control bounds
            use_parallel_realizations: If True, compute realizations in parallel
            gradient_type: Type of gradient approximation:
                - 'stosag': Stochastic Simplex Approximate Gradient (default)
                - 'spsa': Simultaneous Perturbation Stochastic Approximation (FAST!)
                - 'fd_central': Central finite differences (accurate but slow)
                - 'fd_forward': Forward finite differences
                - 'adjoint': PyTorch automatic differentiation (fastest, if differentiable)
            line_search_params: Parameters for line search
            control_parameterization: How to represent controls:
                - 'full': Each timestep has independent controls (180 vars)
                - 'piecewise': Controls constant over periods (reduces vars)
                - 'polynomial': Polynomial representation (very few vars)
            num_control_periods: Number of periods for piecewise parameterization
            spsa_num_samples: Number of random perturbations to average in SPSA
        """
        super().__init__(rom_model, config, norm_params, device, action_ranges)
        
        self.num_realizations = num_realizations
        self.perturbation_size = perturbation_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_parallel = use_parallel_realizations
        self.gradient_type = gradient_type
        
        # Control parameterization
        self.control_parameterization = control_parameterization
        self.num_control_periods = num_control_periods
        self.spsa_num_samples = spsa_num_samples
        
        # Line search parameters
        self.line_search_params = line_search_params or {
            'c1': 1e-4,  # Armijo condition parameter
            'alpha_init': 1.0,  # Initial step size
            'alpha_min': 1e-8,  # Minimum step size
            'rho': 0.5,  # Step size reduction factor
            'max_ls_iter': 20  # Maximum line search iterations
        }
        
        # Storage for optimization history
        self.history = {
            'objectives': [],
            'gradient_norms': [],
            'step_sizes': [],
            'controls': []
        }
        
        # Z0 ensemble (set during optimize)
        self.z0_ensemble = None
        self.num_steps = None
        
    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> OptimizationResult:
        """
        Run LS-SQP optimization with StoSAG gradients.
        
        Args:
            z0_options: Tensor of initial states (num_cases, latent_dim)
            num_steps: Number of control timesteps (default: 30)
            
        Returns:
            OptimizationResult with optimal controls and performance data
        """
        start_time = time.time()
        self.reset_counters()
        self.history = {'objectives': [], 'gradient_norms': [], 'step_sizes': [], 'controls': []}
        
        # Setup
        num_steps = num_steps or 30
        self.num_steps = num_steps
        
        if z0_options is None:
            raise ValueError("z0_options must be provided for robust optimization")
        
        # Use provided Z0 (single realization mode or ensemble)
        self.z0_ensemble = z0_options
        actual_realizations = z0_options.shape[0]
        
        print(f"\n{'='*60}")
        print(f"LS-SQP Optimization")
        print(f"{'='*60}")
        if actual_realizations == 1:
            print(f"Mode: Single Case Optimization")
        else:
            print(f"Mode: Robust Optimization ({actual_realizations} realizations)")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables: {self.num_controls * num_steps}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Tolerance: {self.tolerance}")
        
        # Get bounds and print for verification
        bounds_list = self.get_bounds(num_steps)
        lb = np.array([b[0] for b in bounds_list])
        ub = np.array([b[1] for b in bounds_list])
        bounds = Bounds(lb, ub)
        
        # Print bounds summary
        print(f"\nOptimization in Normalized [0,1] Space")
        print(f"Physical Ranges (for reference):")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, {self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, {self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'='*60}\n")
        
        # Run ROM sensitivity test
        self.test_rom_sensitivity(self.z0_ensemble[0:1])
        
        # Generate initial guess
        x0 = self.generate_initial_guess(num_steps, strategy='midpoint')
        
        # Evaluate initial objective
        initial_obj, _ = self.evaluate_robust_objective(
            x0.reshape(num_steps, self.num_controls),
            self.z0_ensemble
        )
        print(f"Initial objective: {initial_obj:.6f}")
        
        # Run optimization using scipy SLSQP with custom gradient
        result = self._run_slsqp_optimization(x0, bounds, num_steps)
        
        # Extract optimal solution (in normalized [0,1] space)
        optimal_controls_normalized = result.x.reshape(num_steps, self.num_controls)
        
        # Convert to physical units for results
        optimal_controls_physical = self.controls_normalized_to_physical(optimal_controls_normalized)
        initial_controls_physical = self.controls_normalized_to_physical(x0.reshape(num_steps, self.num_controls))
        
        # Evaluate final solution with trajectory
        final_obj, individual_objs = self.evaluate_robust_objective(
            optimal_controls_normalized, self.z0_ensemble
        )
        
        # Get trajectory for best realization (for visualization)
        best_realization_idx = np.argmax(individual_objs)
        z0_best = self.z0_ensemble[best_realization_idx:best_realization_idx+1]
        _, trajectory = self.evaluate_objective(optimal_controls_normalized, z0_best, return_trajectory=True)
        
        # Decode spatial states for visualization
        spatial_states = self.decode_spatial_states(trajectory['states'])
        
        # Compute economic breakdown with PHYSICAL controls
        from .objective import compute_trajectory_npv
        observations_array = np.array(trajectory['observations'])
        _, economic_breakdown = compute_trajectory_npv(
            trajectory['observations'],
            optimal_controls_physical,  # Use physical controls
            self.config,
            self.num_prod,
            self.num_inj
        )
        
        total_time = time.time() - start_time
        
        # Build result with PHYSICAL controls
        optimization_result = OptimizationResult(
            optimal_controls=optimal_controls_physical,  # Physical units!
            optimal_objective=final_obj,
            optimal_states=torch.stack(trajectory['states']),
            optimal_spatial_states=spatial_states,
            optimal_observations=observations_array,
            objective_history=self.history['objectives'],
            gradient_norm_history=self.history['gradient_norms'],
            control_history=self.history['controls'],
            num_iterations=result.nit if hasattr(result, 'nit') else len(self.history['objectives']),
            num_function_evaluations=self.function_eval_count,
            num_gradient_evaluations=self.gradient_eval_count,
            total_time_seconds=total_time,
            convergence_achieved=result.success if hasattr(result, 'success') else True,
            termination_reason=result.message if hasattr(result, 'message') else 'Completed',
            optimizer_type='LS-SQP-StoSAG',
            optimizer_params={
                'num_realizations': self.num_realizations,
                'perturbation_size': self.perturbation_size,
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance,
                'gradient_type': self.gradient_type
            },
            num_realizations=self.num_realizations,
            initial_controls=initial_controls_physical,  # Physical units!
            initial_objective=initial_obj,
            economic_breakdown=economic_breakdown
        )
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(optimization_result.summary())
        
        return optimization_result
    
    def _run_slsqp_optimization(
        self,
        x0: np.ndarray,
        bounds: Bounds,
        num_steps: int
    ):
        """
        Run scipy SLSQP with StoSAG gradients.
        
        Args:
            x0: Initial control vector
            bounds: Scipy Bounds object
            num_steps: Number of timesteps
            
        Returns:
            scipy OptimizeResult
        """
        # Objective wrapper for scipy (minimization, so we negate for maximization)
        def objective(x):
            controls = x.reshape(num_steps, self.num_controls)
            obj, _ = self.evaluate_robust_objective(controls, self.z0_ensemble)
            self.history['objectives'].append(obj)
            self.history['controls'].append(x.copy())
            return -obj  # Negate for minimization
        
        # Gradient wrapper
        def gradient(x):
            grad = self._compute_gradient(x, num_steps)
            grad_norm = np.linalg.norm(grad)
            self.history['gradient_norms'].append(grad_norm)
            return -grad  # Negate for minimization
        
        # Callback for progress printing
        iteration_count = [0]
        def callback(xk):
            iteration_count[0] += 1
            if iteration_count[0] % 10 == 0 or iteration_count[0] <= 5:
                obj = -objective(xk)  # Negate back to get actual objective
                print(f"Iteration {iteration_count[0]:4d}: Objective = {obj:.6f}, "
                      f"Grad Norm = {self.history['gradient_norms'][-1]:.6e}")
        
        # Run SLSQP
        print("Starting SLSQP optimization...")
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            callback=callback,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )
        
        return result
    
    def _compute_gradient(
        self,
        x: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """
        Compute gradient using selected method.
        
        Available methods:
        - stosag: Stochastic Simplex Approximate Gradient (accurate, slow)
        - spsa: Simultaneous Perturbation (fast, 2 evals per sample)
        - fd_central: Central finite differences (most accurate, 2N evals)
        - fd_forward: Forward finite differences (N evals)
        - adjoint: PyTorch autodiff (fastest, single backward pass)
        
        Args:
            x: Current control vector
            num_steps: Number of timesteps
            
        Returns:
            Gradient vector
        """
        self.gradient_eval_count += 1
        
        if self.gradient_type == 'spsa':
            return self._spsa_gradient(x, num_steps)
        elif self.gradient_type == 'adjoint':
            return self._adjoint_gradient(x, num_steps)
        elif self.gradient_type == 'fd_central':
            return self._central_difference_gradient(x, num_steps)
        elif self.gradient_type == 'fd_forward':
            return self._forward_difference_gradient(x, num_steps)
        else:  # stosag (default)
            return self._stosag_gradient(x, num_steps)
    
    def _spsa_gradient(
        self,
        x: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """
        Simultaneous Perturbation Stochastic Approximation (SPSA).
        
        MUCH FASTER than finite differences - only 2 function evaluations
        per gradient sample regardless of dimension!
        
        Reference: Spall (1992) - Multivariate Stochastic Approximation
        
        Args:
            x: Current control vector
            num_steps: Number of timesteps
            
        Returns:
            Gradient estimate
        """
        n_vars = len(x)
        bounds_list = self.get_bounds(num_steps)
        
        # Perturbation magnitude (can decay with iterations)
        c_k = self.perturbation_size
        
        all_gradients = []
        
        # Average over multiple random perturbations for robustness
        for sample in range(self.spsa_num_samples):
            # Random perturbation direction (Bernoulli ±1)
            delta = np.random.choice([-1, 1], size=n_vars)
            
            # Create perturbed points
            x_plus = x + c_k * delta
            x_minus = x - c_k * delta
            
            # Project to bounds
            x_plus = project_to_bounds(x_plus, bounds_list)
            x_minus = project_to_bounds(x_minus, bounds_list)
            
            # Evaluate at both points (average over realizations)
            f_plus, _ = self.evaluate_robust_objective(
                x_plus.reshape(num_steps, self.num_controls),
                self.z0_ensemble
            )
            f_minus, _ = self.evaluate_robust_objective(
                x_minus.reshape(num_steps, self.num_controls),
                self.z0_ensemble
            )
            
            # SPSA gradient estimate
            # g_i = (f+ - f-) / (2 * c_k * delta_i)
            grad = (f_plus - f_minus) / (2 * c_k * delta + 1e-10)
            all_gradients.append(grad)
        
        # Average gradients
        gradient = np.mean(all_gradients, axis=0)
        
        # Debug output
        if self.gradient_eval_count <= 2:
            print(f"\n   SPSA Gradient (only {2 * self.spsa_num_samples} ROM evals vs {n_vars} for FD!):")
            print(f"   Samples: {self.spsa_num_samples}, Perturbation: {c_k}")
            print(f"   Gradient norm: {np.linalg.norm(gradient):.6f}")
        
        return gradient
    
    def _adjoint_gradient(
        self,
        x: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """
        Adjoint (automatic differentiation) gradient using PyTorch.
        
        FASTEST method - single forward + backward pass!
        
        Args:
            x: Current control vector
            num_steps: Number of timesteps
            
        Returns:
            Gradient via autodiff
        """
        # Convert to tensor with gradient tracking
        x_tensor = torch.tensor(
            x.reshape(num_steps, self.num_controls), 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        # Average over realizations
        total_obj = 0.0
        
        for r in range(self.num_realizations):
            z0_r = self.z0_ensemble[r:r+1]
            
            # Forward pass with gradient tracking
            obj = self._differentiable_objective(x_tensor, z0_r)
            total_obj = total_obj + obj
        
        avg_obj = total_obj / self.num_realizations
        
        # Backward pass - compute gradients
        avg_obj.backward()
        
        # Extract gradient
        gradient = x_tensor.grad.cpu().numpy().flatten()
        
        # Debug output
        if self.gradient_eval_count <= 2:
            print(f"\n   Adjoint Gradient (single backward pass!):")
            print(f"   Gradient norm: {np.linalg.norm(gradient):.6f}")
        
        return gradient
    
    def _differentiable_objective(
        self,
        controls: torch.Tensor,
        z0: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable objective for adjoint gradients.
        
        Args:
            controls: Control tensor (num_steps, num_controls) with requires_grad=True
            z0: Initial latent state
            
        Returns:
            Objective value as differentiable tensor
        """
        # Convert normalized controls to physical
        controls_physical = self._controls_norm_to_physical_tensor(controls)
        
        z_current = z0
        total_npv = torch.tensor(0.0, device=self.device)
        
        for t in range(controls.shape[0]):
            control_t = controls_physical[t:t+1, :]
            
            # Prepare ROM input
            control_normalized = self._prepare_control_tensor(control_t)
            
            # ROM prediction (must be differentiable)
            rom_input = {'z': z_current, 'u': control_normalized}
            rom_output = self.rom.model(rom_input)
            
            z_next = rom_output['z_next']
            obs_normalized = rom_output.get('obs', rom_output.get('y', None))
            
            if obs_normalized is not None:
                # Denormalize observations
                obs_physical = self._denormalize_obs_tensor(obs_normalized)
                
                # Compute reward (differentiable)
                step_npv = self._compute_differentiable_reward(control_t, obs_physical)
                total_npv = total_npv + step_npv
            
            z_current = z_next
        
        return -total_npv  # Negative because we minimize
    
    def _controls_norm_to_physical_tensor(self, controls_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0,1] controls to physical units (differentiable)."""
        controls_physical = torch.zeros_like(controls_norm)
        
        for i in range(self.num_controls):
            if i < self.num_prod:
                low, high = self.action_ranges['producer_bhp']
            else:
                low, high = self.action_ranges['gas_injection']
            controls_physical[:, i] = controls_norm[:, i] * (high - low) + low
        
        return controls_physical
    
    def _prepare_control_tensor(self, control_physical: torch.Tensor) -> torch.Tensor:
        """Prepare control tensor for ROM input (differentiable)."""
        control_normalized = torch.zeros_like(control_physical)
        
        for i in range(self.num_controls):
            if i < self.num_prod:
                low = self.norm_bounds['BHP']['min']
                high = self.norm_bounds['BHP']['max']
            else:
                low = self.norm_bounds['GASRATSC']['min']
                high = self.norm_bounds['GASRATSC']['max']
            
            control_normalized[:, i] = (control_physical[:, i] - low) / (high - low + 1e-8)
        
        return control_normalized
    
    def _denormalize_obs_tensor(self, obs_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize observations (differentiable)."""
        # Simplified - full implementation would use norm_params
        return obs_norm  # Placeholder - needs proper implementation
    
    def _compute_differentiable_reward(
        self, 
        control: torch.Tensor, 
        obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute step reward in differentiable manner."""
        # Simplified differentiable reward
        # Full implementation would match compute_step_reward
        return torch.sum(obs)  # Placeholder
    
    def _stosag_gradient(
        self,
        x: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """
        Stochastic Simplex Approximate Gradient (StoSAG).
        
        For each realization, uses a simplex-based gradient approximation.
        The gradients are then averaged for robustness.
        
        Args:
            x: Current control vector
            num_steps: Number of timesteps
            
        Returns:
            Robust gradient estimate
        """
        n_vars = len(x)
        all_gradients = []
        
        # Compute perturbation scale based on variable ranges
        bounds_list = self.get_bounds(num_steps)
        perturbation_scales = np.array([
            self.perturbation_size * (b[1] - b[0]) for b in bounds_list
        ])
        
        # Debug: Print perturbation info on first call
        if self.gradient_eval_count <= 1:
            print(f"\n   DEBUG - Gradient Computation (Normalized Space):")
            print(f"   Perturbation size: {self.perturbation_size}")
            print(f"   All perturbations are {self.perturbation_size} in [0,1] space")
        
        # For each realization
        for r in range(self.num_realizations):
            z0_r = self.z0_ensemble[r:r+1]
            
            # Evaluate base point
            controls_base = x.reshape(num_steps, self.num_controls)
            f0, trajectory = self.evaluate_objective(controls_base, z0_r, return_trajectory=True)
            
            # Debug: Print base objective on first call
            if self.gradient_eval_count <= 1 and r == 0:
                print(f"   Base objective: {f0:.6f}")
                print(f"   Base controls normalized (step 0): {controls_base[0, :]}")
                if trajectory and 'controls_physical' in trajectory:
                    phys = trajectory['controls_physical'][0]
                    print(f"   Base controls physical (step 0): BHP={phys[:self.num_prod]}, Gas={phys[self.num_prod:]}")
                if trajectory and 'observations' in trajectory:
                    obs0 = trajectory['observations'][0]
                    print(f"   Base observations (step 0): {obs0[0, :3]} (BHP), {obs0[0, 3:6]} (Gas), {obs0[0, 6:]} (Water)")
            
            # Simplex gradient approximation
            grad_r = np.zeros(n_vars)
            non_zero_grads = 0
            
            for i in range(n_vars):
                x_plus = x.copy()
                x_plus[i] += perturbation_scales[i]
                
                # Project to bounds
                x_plus = project_to_bounds(x_plus, bounds_list)
                
                controls_plus = x_plus.reshape(num_steps, self.num_controls)
                f_plus, _ = self.evaluate_objective(controls_plus, z0_r)
                
                # Forward difference approximation
                delta_x = x_plus[i] - x[i]
                if abs(delta_x) > 1e-10:
                    grad_r[i] = (f_plus - f0) / delta_x
                    if abs(f_plus - f0) > 1e-10:
                        non_zero_grads += 1
                else:
                    grad_r[i] = 0.0
                
                # Debug: First few gradient computations
                if self.gradient_eval_count <= 1 and i < 6:
                    print(f"   Var {i}: f0={f0:.6f}, f+={f_plus:.6f}, delta_f={f_plus-f0:.8f}, delta_x={delta_x:.4f}, grad={grad_r[i]:.8f}")
            
            if self.gradient_eval_count <= 1:
                print(f"   Non-zero gradients: {non_zero_grads}/{n_vars}")
            
            all_gradients.append(grad_r)
        
        # Average gradients across realizations
        robust_gradient = np.mean(all_gradients, axis=0)
        
        # Debug: print gradient statistics for first few calls
        if self.gradient_eval_count <= 3:
            grad_reshaped = robust_gradient.reshape(num_steps, self.num_controls)
            bhp_grad = grad_reshaped[:, :self.num_prod]
            gas_grad = grad_reshaped[:, self.num_prod:]
            print(f"   Gradient stats (eval {self.gradient_eval_count}):")
            print(f"      BHP grad: mean={np.mean(np.abs(bhp_grad)):.6e}, max={np.max(np.abs(bhp_grad)):.6e}")
            print(f"      Gas grad: mean={np.mean(np.abs(gas_grad)):.6e}, max={np.max(np.abs(gas_grad)):.6e}")
        
        return robust_gradient
    
    def _central_difference_gradient(
        self,
        x: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """
        Central difference gradient (more accurate but 2x function evals).
        """
        n_vars = len(x)
        bounds_list = self.get_bounds(num_steps)
        perturbation_scales = np.array([
            self.perturbation_size * (b[1] - b[0]) for b in bounds_list
        ])
        
        all_gradients = []
        
        for r in range(self.num_realizations):
            z0_r = self.z0_ensemble[r:r+1]
            grad_r = np.zeros(n_vars)
            
            for i in range(n_vars):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += perturbation_scales[i]
                x_minus[i] -= perturbation_scales[i]
                
                x_plus = project_to_bounds(x_plus, bounds_list)
                x_minus = project_to_bounds(x_minus, bounds_list)
                
                f_plus, _ = self.evaluate_objective(
                    x_plus.reshape(num_steps, self.num_controls), z0_r
                )
                f_minus, _ = self.evaluate_objective(
                    x_minus.reshape(num_steps, self.num_controls), z0_r
                )
                
                h = x_plus[i] - x_minus[i]
                if abs(h) > 1e-10:
                    grad_r[i] = (f_plus - f_minus) / h
            
            all_gradients.append(grad_r)
        
        return np.mean(all_gradients, axis=0)
    
    def _forward_difference_gradient(
        self,
        x: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """
        Simple forward difference gradient.
        """
        n_vars = len(x)
        bounds_list = self.get_bounds(num_steps)
        perturbation_scales = np.array([
            self.perturbation_size * (b[1] - b[0]) for b in bounds_list
        ])
        
        # Evaluate base point (average over all realizations)
        f0, _ = self.evaluate_robust_objective(
            x.reshape(self.num_steps, self.num_controls), 
            self.z0_ensemble
        )
        
        gradient = np.zeros(n_vars)
        
        for i in range(n_vars):
            x_plus = x.copy()
            x_plus[i] += perturbation_scales[i]
            x_plus = project_to_bounds(x_plus, bounds_list)
            
            f_plus, _ = self.evaluate_robust_objective(
                x_plus.reshape(self.num_steps, self.num_controls),
                self.z0_ensemble
            )
            
            h = x_plus[i] - x[i]
            if abs(h) > 1e-10:
                gradient[i] = (f_plus - f0) / h
        
        return gradient
    
    def stosag_gradient(
        self,
        controls: np.ndarray,
        z0_ensemble: torch.Tensor
    ) -> np.ndarray:
        """
        Public interface for StoSAG gradient computation.
        
        Args:
            controls: Control sequence (may be 2D or flattened)
            z0_ensemble: Ensemble of initial states
            
        Returns:
            Gradient array
        """
        if controls.ndim == 2:
            num_steps = controls.shape[0]
            x = controls.flatten()
        else:
            x = controls
            num_steps = len(x) // self.num_controls
        
        old_ensemble = self.z0_ensemble
        self.z0_ensemble = z0_ensemble
        
        gradient = self._stosag_gradient(x, num_steps)
        
        self.z0_ensemble = old_ensemble
        
        return gradient
