# Agent Module
# RL agent components -- SAC (default), TD3, DDPG, PPO

from .networks import QNetwork, ValueNetwork, DeterministicPolicy, GaussianPolicy
from .sac_agent import SAC
from .td3_agent import TD3
from .ddpg_agent import DDPG
from .ppo_agent import PPO, PPOActorCritic
from .ppo_buffer import RolloutBuffer
from .replay_memory import ReplayMemory
from .utils import (
    create_log_gaussian,
    logsumexp,
    soft_update,
    hard_update,
    weights_init_
)
from .factory import (
    create_rl_agent,
    create_sac_agent,
    create_environment,
    create_replay_memory,
    create_training_orchestrator,
    setup_training_seeds
)

__all__ = [
    # Networks
    'QNetwork',
    'ValueNetwork',
    'DeterministicPolicy',
    'GaussianPolicy',
    # Agents
    'SAC',
    'TD3',
    'DDPG',
    'PPO',
    'PPOActorCritic',
    # Buffers
    'ReplayMemory',
    'RolloutBuffer',
    # Utils
    'create_log_gaussian',
    'logsumexp',
    'soft_update',
    'hard_update',
    'weights_init_',
    # Factory
    'create_rl_agent',
    'create_sac_agent',
    'create_environment',
    'create_replay_memory',
    'create_training_orchestrator',
    'setup_training_seeds',
]
