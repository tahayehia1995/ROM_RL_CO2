"""
Factory functions for creating RL components
"""
import torch
import numpy as np

from .sac_agent import SAC
from .td3_agent import TD3
from .ddpg_agent import DDPG
from .ppo_agent import PPO
from .replay_memory import ReplayMemory

ALGORITHM_CLASSES = {
    'SAC': SAC,
    'TD3': TD3,
    'DDPG': DDPG,
    'PPO': PPO,
}


def _resolve_latent_dim(config, rom_model=None):
    """Extract latent dimension from ROM model or fall back to config."""
    if rom_model is not None:
        try:
            if hasattr(rom_model, 'model') and hasattr(rom_model.model, 'encode_initial'):
                s_dim = rom_model.model.static_latent_dim
                d_dim = rom_model.model.dynamic_latent_dim
                latent_dim = s_dim + d_dim
                print(f"   Using GNN/Multimodal ROM latent dimension: {latent_dim} "
                      f"(static={s_dim}, dynamic={d_dim})")
                return latent_dim
            if hasattr(rom_model, 'model') and hasattr(rom_model.model, 'static_encoder'):
                s_dim = rom_model.model.static_latent_dim
                d_dim = rom_model.model.dynamic_latent_dim
                latent_dim = s_dim + d_dim
                print(f"   Using multimodal ROM latent dimension: {latent_dim} "
                      f"(static={s_dim}, dynamic={d_dim})")
                return latent_dim
            if hasattr(rom_model, 'model') and hasattr(rom_model.model, 'encoder'):
                encoder = rom_model.model.encoder
                if hasattr(encoder, 'fc_mean'):
                    latent_dim = encoder.fc_mean.out_features
                    print(f"   Using ROM model's latent dimension: {latent_dim}")
                    return latent_dim
                if hasattr(rom_model, 'config'):
                    latent_dim = rom_model.config.model.get('latent_dim', config.model['latent_dim'])
                    print(f"   Using ROM config latent dimension: {latent_dim}")
                    return latent_dim
        except Exception as e:
            print(f"   Could not extract latent_dim from ROM model: {e}")

    latent_dim = config.model['latent_dim']
    if rom_model is None:
        print(f"   No ROM model provided, using RL config latent dimension: {latent_dim}")
    else:
        print(f"   Using RL config latent dimension: {latent_dim}")
    return latent_dim


def create_rl_agent(config, rl_config=None, rom_model=None):
    """
    Create an RL agent based on the selected algorithm type.

    The algorithm is read from ``config.rl_model['algorithm']['type']``
    (default ``'SAC'``).  The dashboard stores the user's choice there
    via ``update_config_with_dashboard``.

    Supported algorithms: SAC, TD3, DDPG, PPO.

    Args:
        config: Main configuration object (must have ``rl_model``)
        rl_config: Dashboard configuration dict (optional)
        rom_model: ROM model instance (optional, used to infer latent_dim)

    Returns:
        Agent instance with unified interface
    """
    latent_dim = _resolve_latent_dim(config, rom_model)
    u_dim = config.model['u_dim']

    algo_type = config.rl_model.get('algorithm', {}).get('type', 'SAC').upper()
    agent_cls = ALGORITHM_CLASSES.get(algo_type)
    if agent_cls is None:
        raise ValueError(
            f"Unknown RL algorithm '{algo_type}'. "
            f"Supported: {list(ALGORITHM_CLASSES.keys())}"
        )

    print(f"   Creating {algo_type} agent (state_dim={latent_dim}, action_dim={u_dim})")
    agent = agent_cls(latent_dim, u_dim, config)

    if rl_config:
        print(f"   Applying dashboard configuration to {algo_type} agent...")
        if hasattr(agent, 'update_policy_with_dashboard_config'):
            agent.update_policy_with_dashboard_config(rl_config)

    return agent


# Backward-compatible alias
def create_sac_agent(config, rl_config=None, rom_model=None):
    """Create SAC agent (backward-compatible wrapper around create_rl_agent)."""
    return create_rl_agent(config, rl_config=rl_config, rom_model=rom_model)


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
    
    if rl_config:
        print("   Applying dashboard configuration to environment...")
        environment.update_action_ranges_from_dashboard(rl_config)
    else:
        print("   No dashboard configuration provided to environment")
    
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
