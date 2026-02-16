"""
Policy Evaluator
=================

Core evaluation class for running trained RL policies in evaluation mode.
Supports deterministic rollouts, multiple Z0 cases, and baseline comparisons.
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import glob

from .results import EpisodeResult, EvaluationResults, ComparisonResults
from .baselines import BaselinePolicy, get_all_baselines


class PolicyEvaluator:
    """
    Evaluates trained RL policies across multiple initial states.
    
    Features:
    - Load checkpoints and run deterministic policy rollouts
    - Evaluate across multiple Z0 cases for robust performance estimation
    - Compare against baseline policies (random, midpoint, naive strategies)
    - Compute comprehensive statistics and metrics
    
    Usage:
        evaluator = PolicyEvaluator(agent, environment, config)
        evaluator.load_checkpoint("checkpoints/sac_checkpoint_best_model_ep87")
        results = evaluator.evaluate_multiple_cases(z0_options, num_cases=50)
        comparison = evaluator.run_baseline_comparison(z0_options, num_cases=50)
    """
    
    def __init__(self, agent, environment, config, rl_config: Optional[Dict] = None):
        """
        Initialize the policy evaluator.
        
        Args:
            agent: SAC agent instance
            environment: ReservoirEnvironment instance
            config: Configuration object
            rl_config: Optional RL dashboard configuration with action ranges
        """
        self.agent = agent
        self.environment = environment
        self.config = config
        self.rl_config = rl_config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get well configuration
        self.num_producers = config.rl_model['reservoir']['num_producers']
        self.num_injectors = config.rl_model['reservoir']['num_injectors']
        self.num_actions = self.num_producers + self.num_injectors
        
        # Default max steps per episode
        self.max_steps = config.rl_model['training']['max_steps_per_episode']
        
        # Checkpoint info
        self.checkpoint_path = None
        self.checkpoint_loaded = False
        
        print(f"PolicyEvaluator initialized:")
        print(f"  Device: {self.device}")
        print(f"  Actions: {self.num_actions} ({self.num_producers} producers + {self.num_injectors} injectors)")
        print(f"  Max steps: {self.max_steps}")
    
    def load_checkpoint(self, checkpoint_path: str, evaluate: bool = True) -> bool:
        """
        Load a trained checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            evaluate: If True, sets networks to eval mode (deterministic)
            
        Returns:
            True if successful, False otherwise
        """
        success = self.agent.load_checkpoint(checkpoint_path, evaluate=evaluate)
        if success:
            self.checkpoint_path = checkpoint_path
            self.checkpoint_loaded = True
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"  Mode: {'Evaluation (deterministic)' if evaluate else 'Training (stochastic)'}")
        else:
            print(f"Failed to load checkpoint: {checkpoint_path}")
        return success
    
    def evaluate_single_episode(
        self,
        z0: torch.Tensor,
        z0_case_idx: int = 0,
        deterministic: bool = True,
        max_steps: Optional[int] = None,
        record_details: bool = True
    ) -> EpisodeResult:
        """
        Run a single evaluation episode.
        
        Args:
            z0: Initial latent state tensor (1, latent_dim)
            z0_case_idx: Index of the Z0 case (for tracking)
            deterministic: Use deterministic (mean) actions if True
            max_steps: Override default max steps
            record_details: Record detailed actions/observations
            
        Returns:
            EpisodeResult with episode data
        """
        if max_steps is None:
            max_steps = self.max_steps
        
        # Reset environment with specific Z0
        state = self.environment.reset(z0_options=z0)
        
        # Storage for episode data
        step_rewards = []
        actions = []
        observations = []
        economic_breakdown = []
        
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Select action (deterministic or stochastic)
            with torch.no_grad():
                action = self.agent.select_action(state, evaluate=deterministic)
            
            # Step environment
            next_state, reward, done = self.environment.step(action)
            
            # Record data
            reward_value = float(reward.item() if hasattr(reward, 'item') else reward)
            step_rewards.append(reward_value)
            episode_reward += reward_value
            
            if record_details:
                # Convert action to physical units
                action_physical = self._convert_action_to_physical(action)
                actions.append(action_physical)
                
                # Get observation in physical units (already physical from env)
                obs_physical = self._format_observation(self.environment.last_observation)
                observations.append(obs_physical)
                
                # Calculate economic breakdown
                breakdown = self._calculate_economic_breakdown(obs_physical, action_physical)
                economic_breakdown.append(breakdown)
            
            state = next_state
            
            if done:
                break
        
        return EpisodeResult(
            z0_case_idx=z0_case_idx,
            total_npv=episode_reward,
            step_rewards=step_rewards,
            actions=actions,
            observations=observations,
            economic_breakdown=economic_breakdown
        )
    
    def evaluate_multiple_cases(
        self,
        z0_options: torch.Tensor,
        num_cases: int = 50,
        deterministic: bool = True,
        random_sample: bool = True,
        specific_indices: Optional[List[int]] = None,
        verbose: bool = True
    ) -> EvaluationResults:
        """
        Evaluate policy across multiple Z0 cases.
        
        Args:
            z0_options: Tensor of Z0 options (num_cases, latent_dim)
            num_cases: Number of cases to evaluate
            deterministic: Use deterministic actions
            random_sample: If True, randomly sample cases; else use first N
            specific_indices: If provided, use these specific indices
            verbose: Print progress
            
        Returns:
            EvaluationResults with aggregated statistics
        """
        start_time = time.time()
        
        # Determine which cases to evaluate
        total_available = z0_options.shape[0]
        num_cases = min(num_cases, total_available)
        
        if specific_indices is not None:
            case_indices = specific_indices[:num_cases]
        elif random_sample:
            case_indices = np.random.choice(total_available, size=num_cases, replace=False).tolist()
        else:
            case_indices = list(range(num_cases))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Policy: {self.checkpoint_path or 'Current Agent'}")
            print(f"{'='*60}")
            print(f"Cases to evaluate: {num_cases} (of {total_available} available)")
            print(f"Mode: {'Deterministic' if deterministic else 'Stochastic'}")
            print(f"{'='*60}\n")
        
        # Evaluate each case
        all_episodes = []
        for i, case_idx in enumerate(case_indices):
            z0 = z0_options[case_idx:case_idx+1]
            
            episode_result = self.evaluate_single_episode(
                z0=z0,
                z0_case_idx=case_idx,
                deterministic=deterministic,
                record_details=True
            )
            all_episodes.append(episode_result)
            
            if verbose and (i + 1) % 10 == 0:
                npvs_so_far = [ep.total_npv for ep in all_episodes]
                print(f"  Progress: {i+1}/{num_cases} | "
                      f"Mean NPV so far: {np.mean(npvs_so_far):.4f} ± {np.std(npvs_so_far):.4f}")
        
        # Create results object
        results = EvaluationResults(
            policy_name="Trained RL Policy",
            checkpoint_path=self.checkpoint_path,
            all_episodes=all_episodes,
            deterministic=deterministic,
            evaluation_time=time.time() - start_time
        )
        results.compute_statistics()
        
        if verbose:
            print(f"\n{results}")
        
        return results
    
    def evaluate_baseline(
        self,
        baseline: BaselinePolicy,
        z0_options: torch.Tensor,
        case_indices: List[int],
        verbose: bool = False
    ) -> EvaluationResults:
        """
        Evaluate a baseline policy.
        
        Args:
            baseline: Baseline policy instance
            z0_options: Z0 options tensor
            case_indices: Specific case indices to evaluate
            verbose: Print progress
            
        Returns:
            EvaluationResults for the baseline
        """
        start_time = time.time()
        all_episodes = []
        
        for i, case_idx in enumerate(case_indices):
            z0 = z0_options[case_idx:case_idx+1]
            
            # Reset environment and baseline
            state = self.environment.reset(z0_options=z0)
            baseline.reset()
            
            step_rewards = []
            actions = []
            observations = []
            economic_breakdown = []
            episode_reward = 0.0
            
            for step in range(self.max_steps):
                # Get baseline action
                with torch.no_grad():
                    action = baseline.select_action(state)
                
                # Step environment
                next_state, reward, done = self.environment.step(action)
                
                reward_value = float(reward.item() if hasattr(reward, 'item') else reward)
                step_rewards.append(reward_value)
                episode_reward += reward_value
                
                # Record details
                action_physical = self._convert_action_to_physical(action)
                actions.append(action_physical)
                obs_physical = self._format_observation(self.environment.last_observation)
                observations.append(obs_physical)
                breakdown = self._calculate_economic_breakdown(obs_physical, action_physical)
                economic_breakdown.append(breakdown)
                
                state = next_state
                if done:
                    break
            
            all_episodes.append(EpisodeResult(
                z0_case_idx=case_idx,
                total_npv=episode_reward,
                step_rewards=step_rewards,
                actions=actions,
                observations=observations,
                economic_breakdown=economic_breakdown
            ))
            
            if verbose and (i + 1) % 10 == 0:
                npvs_so_far = [ep.total_npv for ep in all_episodes]
                print(f"  {baseline.name}: {i+1}/{len(case_indices)} | "
                      f"Mean NPV: {np.mean(npvs_so_far):.4f}")
        
        results = EvaluationResults(
            policy_name=baseline.name,
            all_episodes=all_episodes,
            deterministic=True,  # Baselines are deterministic
            evaluation_time=time.time() - start_time
        )
        results.compute_statistics()
        
        return results
    
    def run_baseline_comparison(
        self,
        z0_options: torch.Tensor,
        num_cases: int = 50,
        deterministic: bool = True,
        baselines: Optional[List[str]] = None,
        verbose: bool = True
    ) -> ComparisonResults:
        """
        Run comparison between trained policy and baselines.
        
        Args:
            z0_options: Z0 options tensor
            num_cases: Number of cases to evaluate
            deterministic: Use deterministic actions for trained policy
            baselines: List of baseline names to include (default: all)
            verbose: Print progress
            
        Returns:
            ComparisonResults with all comparisons
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running Baseline Comparison")
            print(f"{'='*70}")
        
        # Determine case indices (same for all policies for fair comparison)
        total_available = z0_options.shape[0]
        num_cases = min(num_cases, total_available)
        case_indices = np.random.choice(total_available, size=num_cases, replace=False).tolist()
        
        if verbose:
            print(f"Evaluating on {num_cases} Z0 cases (same for all policies)")
            print(f"Case indices sample: {case_indices[:5]}...")
        
        # 1. Evaluate trained policy
        if verbose:
            print(f"\n[1/5] Evaluating Trained RL Policy...")
        
        trained_results = self.evaluate_multiple_cases(
            z0_options=z0_options,
            num_cases=num_cases,
            deterministic=deterministic,
            specific_indices=case_indices,
            verbose=False
        )
        trained_results.policy_name = "Trained RL"
        
        if verbose:
            print(f"  Trained RL: Mean NPV = {trained_results.mean_npv:.4f} ± {trained_results.std_npv:.4f}")
        
        # 2. Evaluate baselines
        all_baselines = get_all_baselines(
            num_actions=self.num_actions,
            device=self.device,
            num_producers=self.num_producers,
            num_injectors=self.num_injectors
        )
        
        # Filter baselines if specified
        if baselines is not None:
            all_baselines = {k: v for k, v in all_baselines.items() if k in baselines}
        
        baseline_results = {}
        for i, (name, baseline) in enumerate(all_baselines.items()):
            if verbose:
                print(f"\n[{i+2}/{len(all_baselines)+1}] Evaluating {name} baseline...")
            
            results = self.evaluate_baseline(
                baseline=baseline,
                z0_options=z0_options,
                case_indices=case_indices,
                verbose=False
            )
            baseline_results[name] = results
            
            if verbose:
                print(f"  {name}: Mean NPV = {results.mean_npv:.4f} ± {results.std_npv:.4f}")
        
        # Create comparison results
        comparison = ComparisonResults(
            trained_policy=trained_results,
            baselines=baseline_results,
            z0_case_indices=case_indices
        )
        comparison.compute_improvement_ratios()
        
        if verbose:
            print(comparison)
        
        return comparison
    
    def _convert_action_to_physical(self, action: torch.Tensor) -> Dict[str, float]:
        """Convert normalized action to physical units."""
        try:
            action_np = action.detach().cpu().numpy().flatten()
        except:
            action_np = np.array(action).flatten()
        
        action_physical = {}
        
        # Get action ranges from rl_config if available
        if self.rl_config and 'action_ranges' in self.rl_config:
            action_ranges = self.rl_config['action_ranges']
            
            # BHP ranges
            bhp_ranges = action_ranges.get('bhp', {})
            if bhp_ranges:
                bhp_mins = [r['min'] for r in bhp_ranges.values()]
                bhp_maxs = [r['max'] for r in bhp_ranges.values()]
                bhp_min = min(bhp_mins) if bhp_mins else 1087.78
                bhp_max = max(bhp_maxs) if bhp_maxs else 1305.34
            else:
                bhp_min, bhp_max = 1087.78, 1305.34
            
            # Gas ranges
            gas_ranges = action_ranges.get('gas_injection', {})
            if gas_ranges:
                gas_mins = [r['min'] for r in gas_ranges.values()]
                gas_maxs = [r['max'] for r in gas_ranges.values()]
                gas_min = min(gas_mins) if gas_mins else 24720290
                gas_max = max(gas_maxs) if gas_maxs else 100646896
            else:
                gas_min, gas_max = 24720290, 100646896
        else:
            # Default ranges
            bhp_min, bhp_max = 1087.78, 1305.34
            gas_min, gas_max = 24720290, 100646896
        
        # Convert BHP (first num_producers actions)
        for i in range(min(self.num_producers, len(action_np))):
            well_name = f"P{i+1}"
            normalized = action_np[i]
            physical = normalized * (bhp_max - bhp_min) + bhp_min
            action_physical[f"{well_name}_BHP_psi"] = physical
        
        # Convert Gas injection (remaining actions)
        for i in range(min(self.num_injectors, len(action_np) - self.num_producers)):
            well_name = f"I{i+1}"
            idx = self.num_producers + i
            if idx < len(action_np):
                normalized = action_np[idx]
                physical = normalized * (gas_max - gas_min) + gas_min
                action_physical[f"{well_name}_Gas_ft3day"] = physical
        
        return action_physical
    
    def _format_observation(self, observation) -> Dict[str, float]:
        """Format observation into dictionary with physical units."""
        if observation is None:
            return {}
        
        try:
            obs_np = observation.detach().cpu().numpy().flatten() if hasattr(observation, 'detach') else np.array(observation).flatten()
        except:
            return {}
        
        obs_dict = {}
        
        # Observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
        for i in range(min(3, len(obs_np))):
            obs_dict[f"I{i+1}_BHP_psi"] = obs_np[i]
        
        for i in range(min(3, max(0, len(obs_np) - 3))):
            if 3 + i < len(obs_np):
                obs_dict[f"P{i+1}_Gas_ft3day"] = obs_np[3 + i]
        
        for i in range(min(3, max(0, len(obs_np) - 6))):
            if 6 + i < len(obs_np):
                obs_dict[f"P{i+1}_Water_ft3day"] = obs_np[6 + i]
                obs_dict[f"P{i+1}_Water_bblday"] = obs_np[6 + i] / 5.614583
        
        return obs_dict
    
    def _calculate_economic_breakdown(self, obs_physical: Dict, action_physical: Dict) -> Dict[str, float]:
        """Calculate economic breakdown for a single step."""
        econ_config = self.config.rl_model['economics']
        
        breakdown = {
            'gas_injection_revenue': 0.0,
            'gas_injection_cost': 0.0,
            'water_production_penalty': 0.0,
            'gas_production_penalty': 0.0,
            'net_step_cashflow': 0.0
        }
        
        # Gas injection economics
        for key, value in action_physical.items():
            if 'Gas_ft3day' in key:
                gas_tons_per_day = value * econ_config['conversion']['lf3_to_intermediate'] * econ_config['conversion']['intermediate_to_ton']
                annual_revenue = gas_tons_per_day * econ_config['prices']['gas_injection_revenue'] * 365
                annual_cost = gas_tons_per_day * econ_config['prices']['gas_injection_cost'] * 365
                breakdown['gas_injection_revenue'] += annual_revenue
                breakdown['gas_injection_cost'] += annual_cost
        
        # Water production penalty
        for key, value in obs_physical.items():
            if 'Water_bblday' in key:
                annual_penalty = value * econ_config['prices']['water_production_penalty'] * 365
                breakdown['water_production_penalty'] += annual_penalty
        
        # Gas production penalty
        for key, value in obs_physical.items():
            if 'Gas_ft3day' in key and 'P' in key:
                gas_tons_per_day = value * econ_config['conversion']['lf3_to_intermediate'] * econ_config['conversion']['intermediate_to_ton']
                annual_penalty = gas_tons_per_day * econ_config['prices']['gas_production_penalty'] * 365
                breakdown['gas_production_penalty'] += annual_penalty
        
        breakdown['net_step_cashflow'] = (
            breakdown['gas_injection_revenue'] -
            breakdown['gas_injection_cost'] -
            breakdown['water_production_penalty'] -
            breakdown['gas_production_penalty']
        )
        
        return breakdown
    
    @staticmethod
    def discover_checkpoints(checkpoints_dir: str = 'checkpoints', include_architecture: bool = False) -> List[Dict[str, Any]]:
        """
        Discover available checkpoint files.
        
        Args:
            checkpoints_dir: Directory to scan for checkpoints
            include_architecture: If True, inspect each checkpoint for architecture info (slower)
            
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoint_path = Path(checkpoints_dir)
        if not checkpoint_path.exists():
            return []
        
        checkpoints = []
        
        # Find all checkpoint files
        patterns = ['sac_checkpoint_best_model_ep*', 'sac_checkpoint_periodic_ep*', 'sac_checkpoint_*']
        
        found_files = set()
        for pattern in patterns:
            for ckpt_file in checkpoint_path.glob(pattern):
                if ckpt_file not in found_files:
                    found_files.add(ckpt_file)
                    
                    # Parse episode number
                    name = ckpt_file.name
                    episode = -1
                    if '_ep' in name:
                        try:
                            episode = int(name.split('_ep')[-1])
                        except:
                            pass
                    
                    # Determine checkpoint type
                    if 'best_model' in name:
                        ckpt_type = 'best'
                    elif 'periodic' in name:
                        ckpt_type = 'periodic'
                    else:
                        ckpt_type = 'other'
                    
                    ckpt_info = {
                        'path': str(ckpt_file),
                        'name': name,
                        'episode': episode,
                        'type': ckpt_type,
                        'modified': ckpt_file.stat().st_mtime,
                        'architecture': None
                    }
                    
                    # Optionally inspect architecture
                    if include_architecture:
                        arch = PolicyEvaluator.get_checkpoint_architecture(str(ckpt_file))
                        ckpt_info['architecture'] = arch
                    
                    checkpoints.append(ckpt_info)
        
        # Sort by episode number (descending)
        checkpoints.sort(key=lambda x: x['episode'], reverse=True)
        
        return checkpoints
    
    @staticmethod
    def get_checkpoint_architecture(ckpt_path: str) -> Optional[Dict[str, Any]]:
        """
        Get architecture info from a checkpoint file without loading it.
        
        Args:
            ckpt_path: Path to checkpoint file
            
        Returns:
            Architecture info dict or None
        """
        try:
            import torch
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
            
            if 'architecture' in checkpoint:
                return checkpoint['architecture']
            else:
                # Try to infer from state dict
                policy_state = checkpoint.get('policy_state_dict', {})
                arch = {}
                
                if 'linear1.weight' in policy_state:
                    weight_shape = policy_state['linear1.weight'].shape
                    arch['hidden_dim'] = int(weight_shape[0])
                    arch['state_dim'] = int(weight_shape[1])
                
                # Check policy type by looking at layer names
                if 'mean_bhp.weight' in policy_state:
                    arch['policy_type'] = 'deterministic'
                elif 'mean_linear.weight' in policy_state:
                    arch['policy_type'] = 'gaussian'
                
                return arch if arch else None
                
        except Exception as e:
            return None
