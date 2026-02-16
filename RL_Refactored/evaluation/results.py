"""
Evaluation Results Data Classes
================================

Data structures for storing and managing evaluation results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime


@dataclass
class EpisodeResult:
    """
    Results from a single evaluation episode.
    
    Attributes:
        z0_case_idx: Index of the Z0 initial state used
        total_npv: Total NPV (sum of step rewards) for the episode
        step_rewards: List of rewards at each timestep
        actions: List of action dictionaries (physical units) per step
        observations: List of observation dictionaries (physical units) per step
        economic_breakdown: List of economic breakdown dictionaries per step
        cumulative_npv: Cumulative NPV at each timestep
    """
    z0_case_idx: int
    total_npv: float
    step_rewards: List[float] = field(default_factory=list)
    actions: List[Dict[str, float]] = field(default_factory=list)
    observations: List[Dict[str, float]] = field(default_factory=list)
    economic_breakdown: List[Dict[str, float]] = field(default_factory=list)
    cumulative_npv: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute cumulative NPV if not provided."""
        if not self.cumulative_npv and self.step_rewards:
            self.cumulative_npv = list(np.cumsum(self.step_rewards))
    
    def get_final_economic_summary(self) -> Dict[str, float]:
        """Get aggregated economic summary for the episode."""
        if not self.economic_breakdown:
            return {}
        
        summary = {
            'total_gas_injection_revenue': 0.0,
            'total_gas_injection_cost': 0.0,
            'total_water_production_penalty': 0.0,
            'total_gas_production_penalty': 0.0,
            'total_net_cashflow': 0.0
        }
        
        for breakdown in self.economic_breakdown:
            summary['total_gas_injection_revenue'] += breakdown.get('gas_injection_revenue', 0.0)
            summary['total_gas_injection_cost'] += breakdown.get('gas_injection_cost', 0.0)
            summary['total_water_production_penalty'] += breakdown.get('water_production_penalty', 0.0)
            summary['total_gas_production_penalty'] += breakdown.get('gas_production_penalty', 0.0)
            summary['total_net_cashflow'] += breakdown.get('net_step_cashflow', 0.0)
        
        return summary


@dataclass
class EvaluationResults:
    """
    Aggregated results from evaluating a policy across multiple episodes.
    
    Attributes:
        policy_name: Name identifier for the policy
        checkpoint_path: Path to the checkpoint file (if applicable)
        num_episodes: Number of episodes evaluated
        mean_npv: Mean NPV across all episodes
        std_npv: Standard deviation of NPV
        min_npv: Minimum NPV observed
        max_npv: Maximum NPV observed
        median_npv: Median NPV
        all_episodes: List of individual episode results
        evaluation_time: Time taken for evaluation (seconds)
        timestamp: When the evaluation was performed
        deterministic: Whether deterministic actions were used
    """
    policy_name: str
    checkpoint_path: Optional[str] = None
    num_episodes: int = 0
    mean_npv: float = 0.0
    std_npv: float = 0.0
    min_npv: float = 0.0
    max_npv: float = 0.0
    median_npv: float = 0.0
    all_episodes: List[EpisodeResult] = field(default_factory=list)
    evaluation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    deterministic: bool = True
    
    def compute_statistics(self):
        """Compute aggregate statistics from episode results."""
        if not self.all_episodes:
            return
        
        npvs = [ep.total_npv for ep in self.all_episodes]
        self.num_episodes = len(self.all_episodes)
        self.mean_npv = float(np.mean(npvs))
        self.std_npv = float(np.std(npvs))
        self.min_npv = float(np.min(npvs))
        self.max_npv = float(np.max(npvs))
        self.median_npv = float(np.median(npvs))
    
    def get_best_episode(self) -> Optional[EpisodeResult]:
        """Get the episode with highest NPV."""
        if not self.all_episodes:
            return None
        return max(self.all_episodes, key=lambda ep: ep.total_npv)
    
    def get_worst_episode(self) -> Optional[EpisodeResult]:
        """Get the episode with lowest NPV."""
        if not self.all_episodes:
            return None
        return min(self.all_episodes, key=lambda ep: ep.total_npv)
    
    def get_confidence_interval(self, confidence: float = 0.95) -> tuple:
        """
        Compute confidence interval for mean NPV.
        
        Args:
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            (lower_bound, upper_bound) tuple
        """
        if self.num_episodes < 2:
            return (self.mean_npv, self.mean_npv)
        
        from scipy import stats
        npvs = [ep.total_npv for ep in self.all_episodes]
        sem = stats.sem(npvs)  # Standard error of mean
        ci = stats.t.interval(confidence, len(npvs) - 1, loc=self.mean_npv, scale=sem)
        return ci
    
    def get_percentiles(self, percentiles: List[float] = [25, 50, 75]) -> Dict[int, float]:
        """Get NPV percentiles."""
        if not self.all_episodes:
            return {}
        npvs = [ep.total_npv for ep in self.all_episodes]
        return {int(p): float(np.percentile(npvs, p)) for p in percentiles}
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary for display."""
        return {
            'Policy': self.policy_name,
            'Checkpoint': self.checkpoint_path or 'N/A',
            'Episodes': self.num_episodes,
            'Mean NPV': f'{self.mean_npv:.4f}',
            'Std NPV': f'{self.std_npv:.4f}',
            'Min NPV': f'{self.min_npv:.4f}',
            'Max NPV': f'{self.max_npv:.4f}',
            'Median NPV': f'{self.median_npv:.4f}',
            'Deterministic': self.deterministic,
            'Eval Time (s)': f'{self.evaluation_time:.2f}'
        }
    
    def __str__(self) -> str:
        """String representation for printing."""
        lines = [
            f"{'='*60}",
            f"Evaluation Results: {self.policy_name}",
            f"{'='*60}",
            f"Checkpoint: {self.checkpoint_path or 'N/A'}",
            f"Episodes Evaluated: {self.num_episodes}",
            f"Action Mode: {'Deterministic' if self.deterministic else 'Stochastic'}",
            f"",
            f"NPV Statistics:",
            f"  Mean:   {self.mean_npv:>10.4f} ± {self.std_npv:.4f}",
            f"  Median: {self.median_npv:>10.4f}",
            f"  Min:    {self.min_npv:>10.4f}",
            f"  Max:    {self.max_npv:>10.4f}",
            f"",
            f"Evaluation Time: {self.evaluation_time:.2f}s",
            f"{'='*60}"
        ]
        return '\n'.join(lines)


@dataclass
class ComparisonResults:
    """
    Results comparing trained policy against baselines.
    
    Attributes:
        trained_policy: Results for the trained RL policy
        baselines: Dictionary mapping baseline names to their results
        improvement_ratios: Percentage improvement over each baseline
        z0_case_indices: List of Z0 case indices used (same for all policies)
    """
    trained_policy: EvaluationResults
    baselines: Dict[str, EvaluationResults] = field(default_factory=dict)
    improvement_ratios: Dict[str, float] = field(default_factory=dict)
    z0_case_indices: List[int] = field(default_factory=list)
    
    def compute_improvement_ratios(self):
        """Compute percentage improvement over each baseline."""
        self.improvement_ratios = {}
        trained_mean = self.trained_policy.mean_npv
        
        for name, baseline_results in self.baselines.items():
            baseline_mean = baseline_results.mean_npv
            if baseline_mean != 0:
                improvement = ((trained_mean - baseline_mean) / abs(baseline_mean)) * 100
            else:
                improvement = float('inf') if trained_mean > 0 else 0.0
            self.improvement_ratios[name] = improvement
    
    def get_ranking(self) -> List[tuple]:
        """
        Get policies ranked by mean NPV.
        
        Returns:
            List of (policy_name, mean_npv) tuples, sorted descending
        """
        all_policies = [(self.trained_policy.policy_name, self.trained_policy.mean_npv)]
        for name, results in self.baselines.items():
            all_policies.append((name, results.mean_npv))
        return sorted(all_policies, key=lambda x: x[1], reverse=True)
    
    def to_comparison_table(self) -> str:
        """Generate a formatted comparison table."""
        ranking = self.get_ranking()
        
        lines = [
            f"\n{'='*70}",
            f"Policy Comparison Results",
            f"{'='*70}",
            f"{'Policy':<25} {'Mean NPV':>12} {'Std':>10} {'vs Trained':>15}",
            f"{'-'*70}"
        ]
        
        for name, mean_npv in ranking:
            if name == self.trained_policy.policy_name:
                std = self.trained_policy.std_npv
                improvement = "---"
            else:
                std = self.baselines[name].std_npv
                improvement = f"{self.improvement_ratios.get(name, 0):+.1f}%"
            
            lines.append(f"{name:<25} {mean_npv:>12.4f} {std:>10.4f} {improvement:>15}")
        
        lines.append(f"{'='*70}")
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.to_comparison_table()
