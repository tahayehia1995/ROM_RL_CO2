"""
Hybrid Optimizer
=================

Two-stage optimization pipeline:
  Stage 1 (Global Search) -- produces a good initial control sequence from
      a trained RL policy or a population-based optimizer (PSO, GA, etc.).
  Stage 2 (Local Refinement) -- uses a gradient-based optimizer (L-BFGS-B,
      Adam, LS-SQP-StoSAG) with the Stage 1 controls as the warm-start
      initial guess.

This combines global exploration with local exploitation for superior
reservoir production optimization.
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .base_optimizer import BaseOptimizer, OptimizationResult


# Gradient-based optimizers eligible for Stage 2
GRADIENT_OPTIMIZERS = {'LS-SQP-StoSAG', 'L-BFGS-B', 'Adam'}

# Global / population-based optimizers eligible for Stage 1
GLOBAL_OPTIMIZERS = {
    'PSO', 'GA', 'CMA-ES',
    'Differential-Evolution', 'Dual-Annealing', 'Basin-Hopping',
}


class HybridOptimizer(BaseOptimizer):
    """
    Two-stage hybrid optimizer.

    Stage 1 provides a warm-start control sequence (either from an RL
    training results file or from running a classical global optimizer).
    Stage 2 refines it with a gradient-based method.

    The returned ``OptimizationResult`` has the same format as any other
    optimizer so it works directly with the existing results dashboard.
    """

    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        action_ranges: Optional[Dict] = None,
        spatial_states=None,
        prediction_mode: str = 'state_based',
        # Hybrid-specific
        hybrid_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            rom_model, config, norm_params, device,
            action_ranges, 'midpoint', spatial_states, prediction_mode,
        )
        self.hybrid_config = hybrid_config or {}

    # ------------------------------------------------------------------
    # Stage 1: extract initial controls
    # ------------------------------------------------------------------

    def _extract_rl_controls(self, pkl_path: str, num_steps: int) -> Tuple[np.ndarray, float]:
        """Load RL training results and return best-episode controls in physical units.

        Returns:
            controls_physical: (num_steps, num_controls) in physical units
            objective: total reward of the best episode
        """
        from RL_Refactored.training.orchestrator import EnhancedTrainingOrchestrator

        loaded = EnhancedTrainingOrchestrator.load_results(pkl_path)
        best_ep = loaded.get_best_episode_data()
        if best_ep is None:
            raise ValueError(f"No best episode data found in {pkl_path}")

        actions_list = best_ep.get('actions', [])
        if not actions_list:
            raise ValueError(f"Best episode has no recorded actions in {pkl_path}")

        controls = np.zeros((len(actions_list), self.num_controls))
        for t, action_dict in enumerate(actions_list):
            for i in range(self.num_prod):
                key = f"P{i+1}_BHP_psi"
                controls[t, i] = action_dict.get(key, 0.0)
            for i in range(self.num_inj):
                key = f"I{i+1}_Gas_ft3day"
                controls[t, self.num_prod + i] = action_dict.get(key, 0.0)

        # Truncate or pad to requested num_steps
        if controls.shape[0] > num_steps:
            controls = controls[:num_steps]
        elif controls.shape[0] < num_steps:
            pad = np.tile(controls[-1:], (num_steps - controls.shape[0], 1))
            controls = np.vstack([controls, pad])

        objective = best_ep.get('total_reward', 0.0)
        if objective is None:
            objective = sum(best_ep.get('rewards', [0.0]))

        return controls, float(objective)

    def _run_stage1_classical(
        self, stage1_type: str, stage1_params: Dict,
        z0_options: torch.Tensor, num_steps: int,
    ) -> Tuple[np.ndarray, float]:
        """Run a classical global optimizer and return its optimal controls."""
        from . import create_optimizer

        stage1_config = {
            'optimizer_type': stage1_type,
            'rom_model': self.rom,
            'config': self.config,
            'norm_params': self.norm_params,
            'device': self.device,
            'action_ranges': self.action_ranges,
            'init_strategy': stage1_params.get('init_strategy', 'midpoint'),
            'spatial_states': self.spatial_states,
            'prediction_mode': 'state_based' if self.prediction_mode == 'state_based' else 'latent_based',
            'optimizer_params': stage1_params,
            'stosag_params': {},
            'sqp_params': {},
        }

        optimizer = create_optimizer(stage1_config)
        result = optimizer.optimize(z0_options=z0_options, num_steps=num_steps)

        return result.optimal_controls, float(result.optimal_objective)

    # ------------------------------------------------------------------
    # Stage 2: create gradient-based optimizer with warm-start
    # ------------------------------------------------------------------

    def _create_stage2_optimizer(
        self, stage2_type: str, stage2_params: Dict,
        initial_guess_normalized: np.ndarray,
    ) -> BaseOptimizer:
        """Instantiate the Stage 2 gradient-based optimizer with custom init."""
        from . import create_optimizer

        stage2_config = {
            'optimizer_type': stage2_type,
            'rom_model': self.rom,
            'config': self.config,
            'norm_params': self.norm_params,
            'device': self.device,
            'action_ranges': self.action_ranges,
            'init_strategy': 'custom',
            'spatial_states': self.spatial_states,
            'prediction_mode': 'state_based' if self.prediction_mode == 'state_based' else 'latent_based',
            'optimizer_params': stage2_params,
            'stosag_params': {
                'num_realizations': 1,
                'perturbation_size': stage2_params.get('perturbation_size', 0.01),
                'gradient_type': stage2_params.get('gradient_type', 'spsa'),
                'spsa_num_samples': stage2_params.get('spsa_num_samples', 5),
            },
            'sqp_params': {
                'max_iterations': stage2_params.get('max_iterations', 100),
                'tolerance': stage2_params.get('tolerance', 1e-6),
            },
        }

        optimizer = create_optimizer(stage2_config)
        optimizer.custom_initial_guess = initial_guess_normalized
        return optimizer

    # ------------------------------------------------------------------
    # Main optimize
    # ------------------------------------------------------------------

    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> OptimizationResult:
        total_start = time.time()
        num_steps = num_steps or 30

        hc = self.hybrid_config
        stage1_source = hc.get('stage1_source', 'RL')
        stage1_params = hc.get('stage1_params', {})
        stage2_type = hc.get('stage2_type', 'L-BFGS-B')
        stage2_params = hc.get('stage2_params', {})

        print(f"\n{'=' * 60}")
        print("Hybrid Optimization")
        print(f"{'=' * 60}")
        print(f"Stage 1 (Global): {stage1_source}")
        print(f"Stage 2 (Local):  {stage2_type}")
        print(f"Timesteps: {num_steps}")
        print(f"{'=' * 60}\n")

        # ---- Stage 1 ----
        print("--- Stage 1: Global Search ---")
        stage1_start = time.time()

        if stage1_source == 'RL':
            pkl_path = stage1_params.get('pkl_path', '')
            if not pkl_path or not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"RL training results file not found: {pkl_path}\n"
                    "Run RL training first or select a valid .pkl file."
                )
            controls_physical, stage1_obj = self._extract_rl_controls(pkl_path, num_steps)
            stage1_label = "RL Trained Policy"
            print(f"Loaded RL controls from {Path(pkl_path).name}")
            print(f"Stage 1 objective (RL best episode reward): {stage1_obj:.6f}")
        else:
            if z0_options is None:
                raise ValueError("z0_options required for classical Stage 1")
            controls_physical, stage1_obj = self._run_stage1_classical(
                stage1_source, stage1_params, z0_options, num_steps,
            )
            stage1_label = stage1_source
            print(f"Stage 1 objective ({stage1_source}): {stage1_obj:.6f}")

        stage1_time = time.time() - stage1_start

        # Convert physical controls to normalized [0,1]
        controls_norm = self.controls_physical_to_normalized(controls_physical)
        controls_norm = np.clip(controls_norm, 0.0, 1.0)

        print(f"Stage 1 completed in {stage1_time:.1f}s")
        print(f"Controls shape: {controls_norm.shape}")

        # ---- Stage 2 ----
        print(f"\n--- Stage 2: Local Refinement ({stage2_type}) ---")
        print("Using Stage 1 controls as warm-start initial guess")

        stage2_optimizer = self._create_stage2_optimizer(
            stage2_type, stage2_params, controls_norm,
        )

        stage2_result = stage2_optimizer.optimize(
            z0_options=z0_options, num_steps=num_steps,
        )

        total_time = time.time() - total_start

        # ---- Combine results ----
        hybrid_label = f"Hybrid ({stage1_label} -> {stage2_type})"
        improvement_over_stage1 = 0.0
        if stage1_obj != 0:
            improvement_over_stage1 = (
                (stage2_result.optimal_objective - stage1_obj) / abs(stage1_obj)
            )

        combined_params = {
            'hybrid_mode': True,
            'stage1_source': stage1_source,
            'stage1_label': stage1_label,
            'stage1_objective': stage1_obj,
            'stage1_time_seconds': stage1_time,
            'stage1_params': stage1_params,
            'stage2_type': stage2_type,
            'stage2_params': stage2_params,
            'stage2_objective': stage2_result.optimal_objective,
            'improvement_over_stage1': improvement_over_stage1,
        }
        combined_params.update(stage2_result.optimizer_params)

        final_result = OptimizationResult(
            optimal_controls=stage2_result.optimal_controls,
            optimal_objective=stage2_result.optimal_objective,
            optimal_states=stage2_result.optimal_states,
            optimal_spatial_states=stage2_result.optimal_spatial_states,
            optimal_observations=stage2_result.optimal_observations,
            objective_history=stage2_result.objective_history,
            gradient_norm_history=stage2_result.gradient_norm_history,
            control_history=stage2_result.control_history,
            num_iterations=stage2_result.num_iterations,
            num_function_evaluations=(
                stage2_result.num_function_evaluations
                + getattr(stage2_optimizer, 'function_eval_count', 0)
            ),
            num_gradient_evaluations=stage2_result.num_gradient_evaluations,
            total_time_seconds=total_time,
            convergence_achieved=stage2_result.convergence_achieved,
            termination_reason=stage2_result.termination_reason,
            optimizer_type=hybrid_label,
            optimizer_params=combined_params,
            num_realizations=stage2_result.num_realizations,
            initial_controls=stage2_result.initial_controls,
            initial_objective=stage1_obj,
            economic_breakdown=stage2_result.economic_breakdown,
        )

        print(f"\n{'=' * 60}")
        print("Hybrid Optimization Complete!")
        print(f"{'=' * 60}")
        print(f"Stage 1 ({stage1_label}): {stage1_obj:.6f}")
        print(f"Stage 2 ({stage2_type}):  {stage2_result.optimal_objective:.6f}")
        print(f"Improvement over Stage 1: {improvement_over_stage1 * 100:.2f}%")
        print(f"Total time: {total_time:.1f}s "
              f"(Stage 1: {stage1_time:.1f}s, Stage 2: {total_time - stage1_time:.1f}s)")
        print(f"{'=' * 60}\n")

        return final_result
