"""
RL Policy Evaluation Module
============================

Provides tools for evaluating trained RL policies:
- PolicyEvaluator: Core evaluation class for running deterministic policy rollouts
- Baseline policies: Random, Midpoint, NaiveMaxGas, NaiveLowGas for comparison
- EvaluationDashboard: Interactive dashboard for checkpoint selection and results visualization

Usage:
    from RL_Refactored.evaluation import EvaluationDashboard, PolicyEvaluator
    
    # Launch interactive dashboard
    eval_dashboard = EvaluationDashboard()
    eval_dashboard.display()
    
    # Or use programmatically
    evaluator = PolicyEvaluator(agent, environment, config)
    results = evaluator.evaluate_multiple_cases(z0_options, num_cases=50)
"""

from .results import (
    EpisodeResult,
    EvaluationResults,
    ComparisonResults
)

from .baselines import (
    BaselinePolicy,
    RandomPolicy,
    MidpointPolicy,
    NaiveMaxGasPolicy,
    NaiveLowGasPolicy
)

from .evaluator import PolicyEvaluator

from .dashboard import EvaluationDashboard, launch_evaluation_dashboard

__all__ = [
    # Results
    'EpisodeResult',
    'EvaluationResults',
    'ComparisonResults',
    # Baselines
    'BaselinePolicy',
    'RandomPolicy',
    'MidpointPolicy',
    'NaiveMaxGasPolicy',
    'NaiveLowGasPolicy',
    # Evaluator
    'PolicyEvaluator',
    # Dashboard
    'EvaluationDashboard',
    'launch_evaluation_dashboard'
]
