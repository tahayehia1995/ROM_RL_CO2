"""
L-BFGS-B Optimizer
===================

Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds (L-BFGS-B)
for reservoir production optimization.

L-BFGS-B is a quasi-Newton method that approximates the inverse Hessian
using a limited number of past gradient evaluations, providing
superlinear convergence while respecting box constraints natively.

Particularly effective for:
- Large-scale bounded optimization (100-10000+ variables)
- Smooth or moderately noisy objectives
- Problems where curvature information accelerates convergence
- Combining with stochastic gradient estimators (SPSA, StoSAG)

When paired with SPSA gradient estimation, L-BFGS-B requires only
2*k ROM evaluations per gradient (k = SPSA samples), making it
tractable for high-dimensional reservoir control problems.

References:
- Byrd, Lu, Nocedal & Zhu (1995) - A Limited Memory Algorithm for
  Bound Constrained Optimization
- Zhu, Byrd, Lu & Nocedal (1997) - Algorithm 778: L-BFGS-B
- Fonseca et al. (2017) - StoSAG for optimization under uncertainty
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize, Bounds

from .base_optimizer import BaseOptimizer, OptimizationResult
from .constraints import project_to_bounds


class LBFGSBOptimizer(BaseOptimizer):
    """
    L-BFGS-B optimizer for reservoir production optimization.

    Uses scipy's L-BFGS-B implementation with pluggable stochastic
    gradient estimators (SPSA, StoSAG, finite differences).

    Key Features:
    - Quasi-Newton curvature approximation via limited-memory BFGS
    - Native box-constraint handling (no projection heuristics)
    - Multiple gradient estimation methods (shared with LS-SQP)
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
        # L-BFGS-B specific
        maxcor: int = 10,
        ftol: float = 1e-10,
        gtol: float = 1e-6,
        # Gradient estimation
        gradient_type: str = 'spsa',
        perturbation_size: float = 0.01,
        spsa_num_samples: int = 5,
        # Common
        seed: Optional[int] = None,
        verbose: int = 1,
        init_strategy: str = 'midpoint',
        spatial_states=None,
        prediction_mode: str = 'state_based',
    ):
        """
        Initialize L-BFGS-B optimizer.

        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            max_iterations: Maximum L-BFGS-B iterations
            tolerance: Convergence tolerance on objective improvement
            action_ranges: Optional well control bounds

            L-BFGS-B specific:
                maxcor: Maximum number of variable metric corrections
                        used to define the limited-memory matrix (default 10).
                        Higher values use more memory but capture more
                        curvature information.
                ftol: Convergence criterion on function value change.
                      Iteration stops when
                      (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol.
                gtol: Convergence criterion on projected gradient norm.

            Gradient estimation (shared with LS-SQP):
                gradient_type: 'spsa', 'stosag', 'fd_forward', 'fd_central'
                perturbation_size: Relative perturbation for gradient (0.001-0.1)
                spsa_num_samples: SPSA perturbations to average (1-20)

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
        self.maxcor = maxcor
        self.ftol = ftol
        self.gtol = gtol
        self.gradient_type = gradient_type
        self.perturbation_size = perturbation_size
        self.spsa_num_samples = spsa_num_samples
        self.seed = seed
        self.verbose = verbose

        self.history: Dict[str, List] = {
            'objectives': [],
            'best_objectives': [],
            'gradient_norms': [],
            'controls': [],
            'sampled_indices': [],
        }

        self.z0_ensemble: Optional[torch.Tensor] = None
        self.total_samples_count: Dict[int, int] = {}
        self.num_steps: Optional[int] = None
        self.current_iteration = 0
        self._best_obj = -np.inf
        self._best_x: Optional[np.ndarray] = None

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
    # Gradient estimation (mirrors LS-SQP implementation)
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
            'gradient_norms': [], 'controls': [],
            'sampled_indices': [],
        }
        self.total_samples_count = {}
        self.current_iteration = 0
        self._best_obj = -np.inf
        self._best_x = None

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
        print(f"L-BFGS-B Optimization (RL-like Sampling)")
        print(f"{'=' * 60}")
        print(f"Cases available: {actual_cases}")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables: {num_vars}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Memory corrections (maxcor): {self.maxcor}")
        print(f"Gradient type: {self.gradient_type}")
        if self.gradient_type == 'spsa':
            print(f"SPSA samples: {self.spsa_num_samples}")
        print(f"Perturbation size: {self.perturbation_size}")
        print(f"\nPhysical Ranges:")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, "
              f"{self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, "
              f"{self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'=' * 60}\n")

        # -- Initial guess --
        x0 = self.generate_initial_guess(num_steps, strategy=self.init_strategy)
        z0_first = self.z0_ensemble[0:1]
        initial_obj, _ = self.evaluate_objective(
            x0.reshape(num_steps, self.num_controls), z0_first,
        )
        self._best_obj = initial_obj
        self._best_x = x0.copy()
        print(f"Initial objective ({self.init_strategy}): {initial_obj:.6f}")

        # -- Bounds --
        bounds_list = self.get_bounds(num_steps)
        lb = np.array([b[0] for b in bounds_list])
        ub = np.array([b[1] for b in bounds_list])

        # -- Objective & gradient wrappers (scipy minimizes → negate) --
        def objective(x):
            controls = x.reshape(num_steps, self.num_controls)
            z0, idx = self._get_random_realization()
            obj, _ = self.evaluate_objective(controls, z0, z0_idx=idx)

            self.history['objectives'].append(obj)
            if obj > self._best_obj:
                self._best_obj = obj
                self._best_x = x.copy()
            self.history['best_objectives'].append(self._best_obj)
            return -obj

        def gradient(x):
            grad = self._compute_gradient(x)
            grad_norm = np.linalg.norm(grad)
            self.history['gradient_norms'].append(grad_norm)
            return -grad

        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1
            self.current_iteration = iteration_count[0]
            if self.verbose > 0 and (iteration_count[0] <= 5 or iteration_count[0] % 10 == 0):
                print(f"Iteration {iteration_count[0]:4d}: Best = {self._best_obj:.6f}, "
                      f"Grad Norm = {self.history['gradient_norms'][-1]:.6e}")
                print(f"   Unique cases sampled: "
                      f"{len(self.total_samples_count)}/{actual_cases}")
                self.log_wandb({
                    'iteration': iteration_count[0],
                    'objective': self._best_obj,
                    'grad_norm': self.history['gradient_norms'][-1],
                    'function_evals': self.function_eval_count,
                })

        # -- Run L-BFGS-B --
        print("Starting L-BFGS-B optimization...\n")
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=Bounds(lb, ub),
            callback=callback,
            options={
                'maxiter': self.max_iterations,
                'maxcor': self.maxcor,
                'ftol': self.ftol,
                'gtol': self.gtol,
                'disp': False,
            },
        )

        # -- Post-process --
        optimal_x = self._best_x if self._best_x is not None else np.clip(result.x, 0, 1)
        optimal_controls_norm = optimal_x.reshape(num_steps, self.num_controls)
        optimal_controls_phys = self.controls_normalized_to_physical(optimal_controls_norm)
        initial_controls_phys = self.controls_normalized_to_physical(
            x0.reshape(num_steps, self.num_controls),
        )

        final_obj = self._best_obj

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

        optimization_result = OptimizationResult(
            optimal_controls=optimal_controls_phys,
            optimal_objective=final_obj,
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
            convergence_achieved=result.success if hasattr(result, 'success') else True,
            termination_reason=result.message if hasattr(result, 'message') else 'Completed',
            optimizer_type='L-BFGS-B',
            optimizer_params={
                'maxcor': self.maxcor,
                'ftol': self.ftol,
                'gtol': self.gtol,
                'gradient_type': self.gradient_type,
                'perturbation_size': self.perturbation_size,
                'spsa_num_samples': self.spsa_num_samples,
                'max_iterations': self.max_iterations,
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
