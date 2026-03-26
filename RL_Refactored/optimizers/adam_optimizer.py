"""
Adam (Adaptive Moment Estimation) Optimizer
=============================================

Projected Adam optimizer for reservoir production optimization.

Adam maintains per-variable adaptive learning rates using exponential
moving averages of the gradient (first moment) and squared gradient
(second moment), with bias correction.  Combined with SPSA gradient
estimation, it provides an efficient, robust optimizer for noisy,
high-dimensional reservoir control problems.

Particularly effective for:
- High-dimensional problems with noisy gradient estimates
- Non-stationary objectives (stochastic Z0 sampling)
- Per-variable adaptation (BHP and gas injection have very different
  sensitivities)
- Fast initial progress followed by fine-grained convergence

References:
- Kingma & Ba (2015) - Adam: A Method for Stochastic Optimization
- Oliveira & Reynolds (2022) - An adaptive moment estimation framework
  for well placement optimization, Computational Geosciences
- Spall (1998) - Implementation of the SPSA Method for Stochastic
  Optimization
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any

from .base_optimizer import BaseOptimizer, OptimizationResult
from .constraints import project_to_bounds


class AdamOptimizer(BaseOptimizer):
    """
    Projected Adam optimizer for reservoir production optimization.

    Custom implementation of Adam with bound projection, using pluggable
    stochastic gradient estimators (SPSA, StoSAG, finite differences).

    Key Features:
    - Per-variable adaptive step sizes via first/second moment estimates
    - Bias-corrected moment estimates for stable early iterations
    - Bound projection after each update (box constraints)
    - Warm-up schedule option for learning rate
    - RL-like random case sampling from Z0 pool
    """

    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
        action_ranges: Optional[Dict] = None,
        # Adam specific
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        lr_decay: float = 1.0,
        warmup_iterations: int = 0,
        # Gradient estimation
        gradient_type: str = 'spsa',
        perturbation_size: float = 0.01,
        spsa_num_samples: int = 5,
        # Stopping criteria
        n_stagnation: int = 50,
        # Common
        seed: Optional[int] = None,
        verbose: int = 1,
        init_strategy: str = 'midpoint',
        spatial_states=None,
        prediction_mode: str = 'state_based',
    ):
        """
        Initialize Adam optimizer.

        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            max_iterations: Maximum Adam iterations
            tolerance: Convergence tolerance (relative improvement)
            action_ranges: Optional well control bounds

            Adam specific:
                learning_rate: Step size alpha (default 0.01).
                               For [0,1] bounded problems, 0.005-0.05
                               is typically a good range.
                beta1: Exponential decay rate for first moment (momentum).
                       Default 0.9.
                beta2: Exponential decay rate for second moment (RMSprop).
                       Default 0.999.
                epsilon: Small constant for numerical stability.
                lr_decay: Multiplicative decay per iteration (1.0 = no decay).
                          E.g. 0.999 gives gradual annealing.
                warmup_iterations: Linear warmup from 0 to learning_rate
                                   over this many iterations (0 = disabled).

            Gradient estimation (shared with LS-SQP / L-BFGS-B):
                gradient_type: 'spsa', 'stosag', 'fd_forward', 'fd_central'
                perturbation_size: Relative perturbation for gradient
                spsa_num_samples: SPSA perturbations to average

            Stopping:
                n_stagnation: Stop if best objective hasn't improved for
                              this many iterations.

            seed: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=summary)
            init_strategy: Initialization strategy
        """
        super().__init__(
            rom_model, config, norm_params, device,
            action_ranges, init_strategy, spatial_states, prediction_mode,
        )

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.warmup_iterations = warmup_iterations
        self.gradient_type = gradient_type
        self.perturbation_size = perturbation_size
        self.spsa_num_samples = spsa_num_samples
        self.n_stagnation = n_stagnation
        self.seed = seed
        self.verbose = verbose

        self.history: Dict[str, List] = {
            'objectives': [],
            'best_objectives': [],
            'gradient_norms': [],
            'learning_rates': [],
            'sampled_indices': [],
        }

        self.z0_ensemble: Optional[torch.Tensor] = None
        self.total_samples_count: Dict[int, int] = {}
        self.num_steps: Optional[int] = None
        self.current_iteration = 0

    # ------------------------------------------------------------------
    # RL-like sampling
    # ------------------------------------------------------------------

    def _get_random_realization(self) -> Tuple[torch.Tensor, int]:
        num_cases = self.z0_ensemble.shape[0]
        idx = np.random.randint(0, num_cases)
        z0 = self.z0_ensemble[idx:idx + 1]
        self.total_samples_count[idx] = self.total_samples_count.get(idx, 0) + 1
        return z0, idx

    # ------------------------------------------------------------------
    # Gradient estimation (mirrors LS-SQP / L-BFGS-B)
    # ------------------------------------------------------------------

    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        self.gradient_eval_count += 1

        if self.gradient_type == 'spsa':
            return self._spsa_gradient(x)
        elif self.gradient_type == 'fd_central':
            return self._central_fd_gradient(x)
        elif self.gradient_type == 'fd_forward':
            return self._forward_fd_gradient(x)
        else:  # stosag
            return self._stosag_gradient(x)

    def _spsa_gradient(self, x: np.ndarray) -> np.ndarray:
        n_vars = len(x)
        bounds_list = self.get_bounds(self.num_steps)
        c_k = self.perturbation_size

        current_z0, sampled_idx = self._get_random_realization()
        self.history['sampled_indices'].append(sampled_idx)

        all_grads: List[np.ndarray] = []
        for _ in range(self.spsa_num_samples):
            delta = np.random.choice([-1, 1], size=n_vars).astype(np.float64)
            x_plus = project_to_bounds(x + c_k * delta, bounds_list)
            x_minus = project_to_bounds(x - c_k * delta, bounds_list)

            f_plus, _ = self.evaluate_objective(
                x_plus.reshape(self.num_steps, self.num_controls),
                current_z0, z0_idx=sampled_idx,
            )
            f_minus, _ = self.evaluate_objective(
                x_minus.reshape(self.num_steps, self.num_controls),
                current_z0, z0_idx=sampled_idx,
            )
            grad = (f_plus - f_minus) / (2 * c_k * delta + 1e-10)
            all_grads.append(grad)

        return np.mean(all_grads, axis=0)

    def _stosag_gradient(self, x: np.ndarray) -> np.ndarray:
        n_vars = len(x)
        bounds_list = self.get_bounds(self.num_steps)
        perturbation_scales = np.array([
            self.perturbation_size * (b[1] - b[0]) for b in bounds_list
        ])

        current_z0, sampled_idx = self._get_random_realization()
        self.history['sampled_indices'].append(sampled_idx)

        f0, _ = self.evaluate_objective(
            x.reshape(self.num_steps, self.num_controls),
            current_z0, z0_idx=sampled_idx,
        )

        gradient = np.zeros(n_vars)
        for i in range(n_vars):
            x_plus = x.copy()
            x_plus[i] += perturbation_scales[i]
            x_plus = project_to_bounds(x_plus, bounds_list)

            f_plus, _ = self.evaluate_objective(
                x_plus.reshape(self.num_steps, self.num_controls),
                current_z0, z0_idx=sampled_idx,
            )
            h = x_plus[i] - x[i]
            if abs(h) > 1e-10:
                gradient[i] = (f_plus - f0) / h

        return gradient

    def _forward_fd_gradient(self, x: np.ndarray) -> np.ndarray:
        return self._stosag_gradient(x)  # identical logic

    def _central_fd_gradient(self, x: np.ndarray) -> np.ndarray:
        n_vars = len(x)
        bounds_list = self.get_bounds(self.num_steps)
        perturbation_scales = np.array([
            self.perturbation_size * (b[1] - b[0]) for b in bounds_list
        ])

        current_z0, sampled_idx = self._get_random_realization()
        self.history['sampled_indices'].append(sampled_idx)

        gradient = np.zeros(n_vars)
        for i in range(n_vars):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += perturbation_scales[i]
            x_minus[i] -= perturbation_scales[i]
            x_plus = project_to_bounds(x_plus, bounds_list)
            x_minus = project_to_bounds(x_minus, bounds_list)

            f_plus, _ = self.evaluate_objective(
                x_plus.reshape(self.num_steps, self.num_controls),
                current_z0, z0_idx=sampled_idx,
            )
            f_minus, _ = self.evaluate_objective(
                x_minus.reshape(self.num_steps, self.num_controls),
                current_z0, z0_idx=sampled_idx,
            )
            h = x_plus[i] - x_minus[i]
            if abs(h) > 1e-10:
                gradient[i] = (f_plus - f_minus) / h

        return gradient

    # ------------------------------------------------------------------
    # Learning-rate schedule
    # ------------------------------------------------------------------

    def _effective_lr(self, iteration: int) -> float:
        """Compute effective learning rate with optional warmup and decay."""
        base_lr = self.learning_rate * (self.lr_decay ** iteration)

        if self.warmup_iterations > 0 and iteration < self.warmup_iterations:
            warmup_factor = (iteration + 1) / self.warmup_iterations
            return base_lr * warmup_factor

        return base_lr

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> OptimizationResult:
        start_time = time.time()
        self.reset_counters()
        self.history = {
            'objectives': [], 'best_objectives': [],
            'gradient_norms': [], 'learning_rates': [],
            'sampled_indices': [],
        }
        self.total_samples_count = {}
        self.current_iteration = 0

        num_steps = num_steps or 30
        self.num_steps = num_steps

        if z0_options is None:
            raise ValueError("z0_options must be provided")

        self.z0_ensemble = z0_options
        actual_cases = z0_options.shape[0]
        num_vars = num_steps * self.num_controls

        if self.seed is not None:
            np.random.seed(self.seed)

        # -- Header --
        print(f"\n{'=' * 60}")
        print("Adam Optimization (RL-like Sampling)")
        print(f"{'=' * 60}")
        print(f"Cases available: {actual_cases}")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables: {num_vars}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Beta1: {self.beta1}, Beta2: {self.beta2}")
        print(f"LR decay: {self.lr_decay}")
        if self.warmup_iterations > 0:
            print(f"Warmup iterations: {self.warmup_iterations}")
        print(f"Gradient type: {self.gradient_type}")
        if self.gradient_type == 'spsa':
            print(f"SPSA samples: {self.spsa_num_samples}")
        print(f"Perturbation size: {self.perturbation_size}")
        print(f"Stagnation limit: {self.n_stagnation}")
        print(f"\nPhysical Ranges:")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, "
              f"{self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, "
              f"{self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'=' * 60}\n")

        # -- Initial guess --
        x = self.generate_initial_guess(num_steps, strategy=self.init_strategy).copy()
        z0_first = self.z0_ensemble[0:1]
        initial_obj, _ = self.evaluate_objective(
            x.reshape(num_steps, self.num_controls), z0_first,
        )
        print(f"Initial objective ({self.init_strategy}): {initial_obj:.6f}")

        # -- Adam state --
        m = np.zeros(num_vars)  # first moment estimate
        v = np.zeros(num_vars)  # second moment estimate

        best_obj = initial_obj
        best_x = x.copy()
        stagnation_count = 0

        self.history['best_objectives'].append(best_obj)

        print("\nStarting Adam optimization...\n")

        for t in range(1, self.max_iterations + 1):
            self.current_iteration = t

            # Gradient (we MAXIMIZE NPV, so gradient points uphill)
            grad = self._compute_gradient(x)
            grad_norm = np.linalg.norm(grad)
            self.history['gradient_norms'].append(grad_norm)

            # Adam moment updates
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Effective learning rate
            lr = self._effective_lr(t - 1)
            self.history['learning_rates'].append(lr)

            # Ascent step (+ because we maximize)
            x = x + lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Project to bounds [0, 1]
            x = np.clip(x, 0.0, 1.0)

            # Evaluate current solution
            z0, idx = self._get_random_realization()
            obj, _ = self.evaluate_objective(
                x.reshape(num_steps, self.num_controls), z0, z0_idx=idx,
            )
            self.history['objectives'].append(obj)

            # Track best
            if obj > best_obj + self.tolerance:
                best_obj = obj
                best_x = x.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            self.history['best_objectives'].append(best_obj)

            # Progress reporting
            if self.verbose > 0 and (t <= 5 or t % 10 == 0):
                print(f"Iteration {t:4d}: Obj = {obj:.6f}, Best = {best_obj:.6f}, "
                      f"LR = {lr:.6f}, |g| = {grad_norm:.6e}")
                print(f"   Unique cases sampled: "
                      f"{len(self.total_samples_count)}/{actual_cases}")
                self.log_wandb({
                    'iteration': t,
                    'objective': obj,
                    'best_objective': best_obj,
                    'learning_rate': lr,
                    'grad_norm': grad_norm,
                    'function_evals': self.function_eval_count,
                })

            # Stagnation check
            if stagnation_count >= self.n_stagnation:
                print(f"\nStopped: No improvement for {self.n_stagnation} iterations")
                break

        # -- Post-process --
        optimal_controls_norm = best_x.reshape(num_steps, self.num_controls)
        optimal_controls_phys = self.controls_normalized_to_physical(optimal_controls_norm)
        initial_controls_phys = self.controls_normalized_to_physical(
            self.generate_initial_guess(num_steps, strategy=self.init_strategy)
            .reshape(num_steps, self.num_controls),
        )

        _, trajectory = self.evaluate_objective(
            optimal_controls_norm, z0_first, return_trajectory=True,
        )
        spatial_states = self.decode_spatial_states(trajectory['states'])

        from .objective import compute_trajectory_npv
        observations_array = np.array(trajectory['observations'])
        _, economic_breakdown = compute_trajectory_npv(
            trajectory['observations'],
            optimal_controls_phys,
            self.config, self.num_prod, self.num_inj,
        )

        total_time = time.time() - start_time

        converged = stagnation_count < self.n_stagnation
        if converged:
            term_reason = 'Max iterations reached'
        else:
            term_reason = f'Stagnation ({self.n_stagnation} iterations without improvement)'

        optimization_result = OptimizationResult(
            optimal_controls=optimal_controls_phys,
            optimal_objective=best_obj,
            optimal_states=torch.stack(trajectory['states']),
            optimal_spatial_states=spatial_states,
            optimal_observations=observations_array,
            objective_history=self.history['best_objectives'],
            gradient_norm_history=self.history['gradient_norms'],
            control_history=[],
            num_iterations=self.current_iteration,
            num_function_evaluations=self.function_eval_count,
            num_gradient_evaluations=self.gradient_eval_count,
            total_time_seconds=total_time,
            convergence_achieved=converged,
            termination_reason=term_reason,
            optimizer_type='Adam',
            optimizer_params={
                'learning_rate': self.learning_rate,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'epsilon': self.epsilon,
                'lr_decay': self.lr_decay,
                'warmup_iterations': self.warmup_iterations,
                'gradient_type': self.gradient_type,
                'perturbation_size': self.perturbation_size,
                'spsa_num_samples': self.spsa_num_samples,
                'max_iterations': self.max_iterations,
                'n_stagnation': self.n_stagnation,
            },
            num_realizations=actual_cases,
            initial_controls=initial_controls_phys,
            initial_objective=initial_obj,
            economic_breakdown=economic_breakdown,
        )

        print(f"\n{'=' * 60}")
        print("Optimization Complete!")
        print(f"{'=' * 60}")
        print(optimization_result.summary())
        self._print_sampling_statistics()

        return optimization_result

    def _print_sampling_statistics(self):
        print(f"\n{'=' * 60}")
        print("Sampling Statistics (RL-like)")
        print(f"{'=' * 60}")
        print(f"Total cases available: {self.z0_ensemble.shape[0]}")
        print(f"Unique cases sampled: {len(self.total_samples_count)}")
        print(f"Total samples drawn: {sum(self.total_samples_count.values())}")

        if self.total_samples_count:
            counts = list(self.total_samples_count.values())
            print(f"Coverage: {100 * len(self.total_samples_count) / self.z0_ensemble.shape[0]:.1f}%")
            print(f"Samples per case: min={min(counts)}, max={max(counts)}, "
                  f"mean={np.mean(counts):.2f}")
        print(f"{'=' * 60}\n")
