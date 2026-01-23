"""
Main Run File for RL-SAC Training
==================================
This file orchestrates the three main parts of the RL project:
1. Configuration Dashboard - Configure RL parameters, load ROM model, generate Z0
2. Training Dashboard - Train RL agent with SAC algorithm
3. Visualization Dashboard - Analyze training results

Each part is independent with its own configuration and functionality.
"""
#%%
# Setup paths BEFORE importing RL_Refactored modules
import sys
from pathlib import Path

# Get project root (parent of RL_Refactored)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add ROM_Refactored to path
rom_path = project_root / 'ROM_Refactored'
if str(rom_path) not in sys.path:
    sys.path.insert(0, str(rom_path))

# Now import utilities and print hardware info
from RL_Refactored.utilities import print_hardware_info
print_hardware_info()

# Import dashboard launch functions
from RL_Refactored.configuration import launch_rl_config_dashboard
from RL_Refactored.training import create_rl_training_dashboard
from RL_Refactored.visualization import launch_interactive_scientific_analysis

# ============================================================================
# STEP 1: RL CONFIGURATION DASHBOARD
# ============================================================================
config_dashboard = launch_rl_config_dashboard()

# ============================================================================
# STEP 2: RL TRAINING DASHBOARD
# ============================================================================
#%%
training_dashboard = create_rl_training_dashboard(config_path='config.yaml')

# ============================================================================
# STEP 3: RL VISUALIZATION DASHBOARD
# ============================================================================
#%%
viz_dashboard = launch_interactive_scientific_analysis(training_dashboard=training_dashboard)

print("\n" + "=" * 70)
print("RL-SAC Training Workflow Complete")
print("=" * 70)

# %%
