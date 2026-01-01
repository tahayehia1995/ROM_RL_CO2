# RL_Refactored

Refactored Reinforcement Learning (RL) components extracted from `Original_Project`, organized into a modular structure similar to `ROM_Refactored`.

## Structure

```
RL_Refactored/
├── RL_SAC_Interactive_Run_File.py    # Main run file (3 dashboard calls)
├── config.yaml                         # Central RL configuration
├── agent/                              # SAC agent components
│   ├── __init__.py
│   ├── networks.py                     # QNetwork, ValueNetwork, Policy networks
│   ├── sac_agent.py                    # SAC algorithm implementation
│   ├── replay_memory.py                # Experience replay buffer
│   ├── utils.py                        # Utility functions
│   └── factory.py                      # Factory functions for creating components
├── environment/                        # RL environment
│   ├── __init__.py
│   ├── reservoir_env.py               # ReservoirEnvironment class
│   └── reward.py                       # Reward function
├── training/                           # Training components
│   ├── __init__.py
│   ├── orchestrator.py                 # ActionVariationManager, EnhancedTrainingOrchestrator
│   └── dashboard.py                    # Training dashboard (placeholder)
├── configuration/                      # Configuration dashboard
│   ├── __init__.py
│   └── dashboard.py                    # RL configuration dashboard (placeholder)
├── visualization/                      # Visualization dashboard
│   ├── __init__.py
│   └── dashboard.py                    # Scientific visualization dashboard (placeholder)
└── utilities/                          # Shared utilities
    └── __init__.py                     # Re-exports Config and create_wandb_logger from ROM_Refactored
```

## Workflow

The RL training workflow consists of three main dashboards:

1. **Configuration Dashboard** (`configuration/dashboard.py`)
   - Configure RL parameters
   - Load ROM model from `ROM_Refactored`
   - Generate initial states (Z0)

2. **Training Dashboard** (`training/dashboard.py`)
   - Train RL agent using SAC algorithm
   - Monitor training progress
   - Save checkpoints

3. **Visualization Dashboard** (`visualization/dashboard.py`)
   - Analyze training results
   - Visualize RL performance
   - Scientific analysis of episodes

## Usage

Run the main file to launch all three dashboards sequentially:

```python
# Run RL_SAC_Interactive_Run_File.py
# Each dashboard is called in sequence with #%% cell separators
```

## Dependencies

- **ROM_Refactored**: All ROM-related functionality comes from `ROM_Refactored` folder
  - ROM model: `ROM_Refactored.model.training.ROMWithE2C`
  - Data preprocessing: `ROM_Refactored.data_preprocessing.load_processed_data`
  - Utilities: `ROM_Refactored.utilities.Config`, `ROM_Refactored.utilities.wandb_integration`

- **PyTorch**: For neural networks and RL algorithms
- **NumPy**: For numerical operations
- **ipywidgets**: For interactive dashboards (optional)

## Configuration

All RL-specific parameters are configured in `config.yaml`, organized into clear sections:

- `model`: Model architecture parameters (must match ROM model)
- `rl_model`: All RL-specific configuration
  - `reservoir`: Reservoir configuration
  - `economics`: Economic reward function parameters
  - `networks`: Neural network architecture
  - `sac`: SAC algorithm hyperparameters
  - `replay_memory`: Experience replay parameters
  - `training`: Training configuration
  - `environment`: Environment simulation parameters
  - `action_variation`: Action variation enhancement

## Implementation Status

### Completed ✅
- Folder structure and module organization
- Agent components (networks, SAC agent, replay memory, utils, factory)
- Environment components (ReservoirEnvironment, reward function)
- Training orchestrator (ActionVariationManager, EnhancedTrainingOrchestrator)
- Utilities module (re-exports from ROM_Refactored)
- Central config.yaml
- Main run file
- Dashboard structure (placeholders)

### Pending ⚠️
- Full dashboard implementations (copy from `Original_Project`):
  - Configuration dashboard: `Original_Project/rl_config_dashboard.py` (lines 940-2535)
  - Training dashboard: `Original_Project/RL_SAC_Interactive_Training.py` (lines 220-410)
  - Visualization dashboard: `Original_Project/rl_config_dashboard.py` (lines 2958-6628)

## Notes

- All ROM dependencies must come from `ROM_Refactored`, not `Original_Project`
- Dashboard files are currently placeholders with proper structure and imports
- Full implementations need to be copied from `Original_Project` and updated with new imports
- The structure follows the same modular pattern as `ROM_Refactored` for consistency

