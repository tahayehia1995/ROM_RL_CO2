"""
Main Run File for RL-SAC Training and Classical Optimization
==============================================================
RL Workflow: Steps 1-4 (Training + Evaluation)
Classical Optimization Workflow: Steps 5-7
"""
#%%
# === SETUP (Run this cell first) ===
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'ROM_Refactored') not in sys.path:
    sys.path.insert(0, str(project_root / 'ROM_Refactored'))

from RL_Refactored.utilities import print_hardware_info
print_hardware_info()

# ==============================================================================
#                           RL TRAINING WORKFLOW
# ==============================================================================

#%% STEP 1: RL Configuration Dashboard
from RL_Refactored.configuration import launch_rl_config_dashboard
config_dashboard = launch_rl_config_dashboard()

#%% STEP 2: RL Training Dashboard
from RL_Refactored.training import create_rl_training_dashboard
training_dashboard = create_rl_training_dashboard(config_path='config.yaml')

#%% STEP 3: RL Visualization Dashboard
from RL_Refactored.visualization import launch_interactive_scientific_analysis
viz_dashboard = launch_interactive_scientific_analysis(training_dashboard=training_dashboard)

#%% STEP 4: RL Policy Evaluation Dashboard
# Evaluate trained policy across multiple Z0 cases and compare against baselines
from RL_Refactored.evaluation import EvaluationDashboard, launch_evaluation_dashboard
eval_dashboard = launch_evaluation_dashboard()

# ==============================================================================
#                      CLASSICAL OPTIMIZATION WORKFLOW
# ==============================================================================

#%% STEP 5: Optimizer Configuration Dashboard
from RL_Refactored.optimizers import launch_optimizer_config_dashboard
optimizer_config_dashboard = launch_optimizer_config_dashboard()

#%% STEP 6: Run Classical Optimization
from RL_Refactored.optimizers import create_optimizer, run_optimization
import builtins
optimizer_config = getattr(builtins, 'optimizer_dashboard_config', None)
if optimizer_config:
    optimizer = create_optimizer(optimizer_config)
    optimizer_result = run_optimization(optimizer, z0_options=optimizer_config['z0_options'], num_steps=optimizer_config['num_steps'])
else:
    print("Run STEP 5 first and click 'Apply Configuration'")

#%% STEP 7: Optimizer Results Dashboard
from RL_Refactored.optimizers import launch_optimizer_results_dashboard
if 'optimizer_result' in dir() and optimizer_result:
    opt_viz = launch_optimizer_results_dashboard(optimizer_result, config=optimizer_config.get('config'))
else:
    print("Run STEP 6 first")

# %%
