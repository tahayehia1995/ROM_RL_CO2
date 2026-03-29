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

from RL_Refactored.training.orchestrator import EnhancedTrainingOrchestrator
from RL_Refactored.visualization import launch_interactive_scientific_analysis
from RL_Refactored.utilities import Config
from pathlib import Path
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
# ==============================================================================
#                           RL VISUALIZATION WORKFLOW
# ==============================================================================
#%% STEP 3b (OPTIONAL): Load & Visualize SAVED Training Results (no re-training needed)
# Use this to visualize results from a previous training run.
# Results are auto-saved after each training in RL_Refactored/training_results/

# Option A: Load the most recent saved results automatically
results_dir = Path(__file__).parent / 'training_results' if '__file__' in dir() else Path('RL_Refactored/training_results')
result_files = sorted(results_dir.glob('rl_training_results_*.pkl')) if results_dir.exists() else []
if result_files:
    loaded_orchestrator = EnhancedTrainingOrchestrator.load_results(result_files[-1])
    config = Config('config.yaml')
    viz_dashboard = launch_interactive_scientific_analysis(training_orchestrator=loaded_orchestrator, config=config)
else:
    print("No saved training results found. Run STEP 2 first.")

#%%
# Option B: Load a specific file (uncomment and set path)
loaded_orchestrator = EnhancedTrainingOrchestrator.load_results('RL_Refactored/training_results/rl_training_results_SAC_MAMBA2_MM_ld_128_ns_2_bs_16_ch_4_20260329_130202.pkl')
config = Config('config.yaml')
viz_dashboard = launch_interactive_scientific_analysis(training_orchestrator=loaded_orchestrator, config=config)


# ==============================================================================
#                           RL POLICY EVALUATION WORKFLOW
# ==============================================================================
#%% STEP 4: RL Policy Evaluation Dashboard
# Evaluate trained policy across multiple Z0 cases and compare against baselines
from RL_Refactored.evaluation import EvaluationDashboard, launch_evaluation_dashboard
eval_dashboard = launch_evaluation_dashboard()

# ==============================================================================
#                      CLASSICAL OPTIMIZATION WORKFLOW
# ==============================================================================
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
