"""
Factory functions for creating RL components
"""
import torch
import numpy as np

from .sac_agent import SAC
from .replay_memory import ReplayMemory


def create_sac_agent(config, rl_config=None, rom_model=None):
    """
    Create SAC agent with config parameters and dashboard configuration
    
    Args:
        config: Main configuration object
        rl_config: Dashboard configuration (optional)
        rom_model: ROM model instance (optional, used to extract actual latent_dim)
    """
    # 🎯 Extract latent dimension from ROM model if available, otherwise use config
    if rom_model is not None:
        # Get latent dimension from ROM model's encoder
        try:
            if hasattr(rom_model, 'model') and hasattr(rom_model.model, 'encoder'):
                encoder = rom_model.model.encoder
                if hasattr(encoder, 'fc_mean'):
                    # Extract from encoder's output layer
                    latent_dim = encoder.fc_mean.out_features
                    print(f"   ✅ Using ROM model's latent dimension: {latent_dim}")
                elif hasattr(encoder, 'fc_mean') and hasattr(encoder.fc_mean, 'out_features'):
                    latent_dim = encoder.fc_mean.out_features
                    print(f"   ✅ Using ROM model's latent dimension: {latent_dim}")
                else:
                    # Fallback: try to get from ROM config
                    if hasattr(rom_model, 'config'):
                        latent_dim = rom_model.config.model.get('latent_dim', config.model['latent_dim'])
                        print(f"   ⚠️ Using ROM config latent dimension: {latent_dim}")
                    else:
                        latent_dim = config.model['latent_dim']
                        print(f"   ⚠️ Using RL config latent dimension: {latent_dim} (ROM model structure not accessible)")
            else:
                # Try direct access
                if hasattr(rom_model, 'encoder') and hasattr(rom_model.encoder, 'fc_mean'):
                    latent_dim = rom_model.encoder.fc_mean.out_features
                    print(f"   ✅ Using ROM model's latent dimension: {latent_dim}")
                else:
                    latent_dim = config.model['latent_dim']
                    print(f"   ⚠️ Using RL config latent dimension: {latent_dim} (ROM model structure not accessible)")
        except Exception as e:
            print(f"   ⚠️ Could not extract latent_dim from ROM model: {e}")
            print(f"   ⚠️ Using RL config latent dimension: {config.model['latent_dim']}")
            latent_dim = config.model['latent_dim']
    else:
        latent_dim = config.model['latent_dim']
        print(f"   ⚠️ No ROM model provided, using RL config latent dimension: {latent_dim}")
    
    u_dim = config.model['u_dim']
    agent = SAC(latent_dim, u_dim, config)
    
    # 🎯 NEW: Update agent with dashboard configuration if provided
    if rl_config:
        print("🔧 Applying dashboard configuration to SAC agent...")
        if hasattr(agent, 'update_policy_with_dashboard_config'):
            agent.update_policy_with_dashboard_config(rl_config)
    
    return agent


def create_environment(state0, config, rom, rl_config=None):
    """
    Create environment with config parameters and dashboard configuration
    
    Args:
        state0: Initial state options (single state or multiple Z0 options for random sampling)
        config: Main configuration object
        rom: ROM model
        rl_config: Dashboard configuration (optional)
    """
    from RL_Refactored.environment import ReservoirEnvironment
    environment = ReservoirEnvironment(state0, config, rom)
    
    # 🎯 NEW: Update environment with dashboard configuration if provided
    if rl_config:
        print("🌍 Applying dashboard configuration to environment...")
        environment.update_action_ranges_from_dashboard(rl_config)
    else:
        print("⚠️ No dashboard configuration provided to environment")
    
    return environment


def create_replay_memory(config):
    """Create replay memory with config parameters"""
    capacity = config.rl_model['replay_memory']['capacity']
    seed = config.rl_model['training']['seeds']['replay_memory']
    return ReplayMemory(capacity, seed)


def create_training_orchestrator(config, rl_config=None):
    """Create enhanced training orchestrator with action variation"""
    from RL_Refactored.training import EnhancedTrainingOrchestrator
    return EnhancedTrainingOrchestrator(config, rl_config)


def setup_training_seeds(config):
    """Setup random seeds from config"""
    seeds = config.rl_model['training']['seeds']
    torch.manual_seed(seeds['torch'])
    np.random.seed(seeds['numpy'])

