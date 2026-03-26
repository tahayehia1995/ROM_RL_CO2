"""
Deep Deterministic Policy Gradient (DDPG) Agent

Classic off-policy actor-critic for continuous control.
Uses a single Q-network, deterministic actor, and Ornstein-Uhlenbeck
exploration noise.

Reference: Lillicrap et al. (2016)
           "Continuous control with deep reinforcement learning"
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.optim import Adam

from .networks import DeterministicPolicy
from .utils import hard_update, weights_init_

_RL_DIR = str(Path(__file__).resolve().parent.parent)


class SingleQNetwork(nn.Module):
    """Single Q-network for DDPG (no twin architecture)."""

    def __init__(self, num_inputs, num_actions, config):
        super().__init__()
        hidden_dim = config.rl_model['networks']['hidden_dim']

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(lambda m: weights_init_(m, config))

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class OrnsteinUhlenbeckNoise:
    """OU process for temporally-correlated exploration."""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


class DDPG(object):
    """DDPG agent with the same interface as SAC."""

    def __init__(self, num_inputs, u_dim, config):
        super(DDPG, self).__init__()
        self.config = config

        ddpg_config = config.rl_model.get('ddpg', {})
        sac_config = config.rl_model['sac']

        self.gamma = ddpg_config.get('discount_factor', sac_config['discount_factor'])
        self.tau = ddpg_config.get('soft_update_tau', sac_config['soft_update_tau'])

        ou_theta = ddpg_config.get('ou_theta', 0.15)
        ou_sigma = ddpg_config.get('ou_sigma', 0.2)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        critic_lr = ddpg_config.get('critic_lr', sac_config['learning_rates']['critic'])
        policy_lr = ddpg_config.get('policy_lr', sac_config['learning_rates']['policy'])

        # Single Q-network
        self.critic = SingleQNetwork(num_inputs, u_dim, config).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = SingleQNetwork(num_inputs, u_dim, config).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Deterministic policy (reuse existing DeterministicPolicy)
        self.policy = DeterministicPolicy(num_inputs, u_dim, config).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        self.policy_target = DeterministicPolicy(num_inputs, u_dim, config).to(self.device)
        hard_update(self.policy_target, self.policy)

        # OU noise
        self.ou_noise = OrnsteinUhlenbeckNoise(u_dim, theta=ou_theta, sigma=ou_sigma)

    # ----- unified interface -----

    def select_action(self, state, evaluate=False):
        mean = self.policy.forward(state)
        if evaluate:
            return mean
        noise = torch.tensor(self.ou_noise.sample(), dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.clamp(mean + noise, 0.0, 1.0)

    def update_parameters(self, memory, batch_size, updates):
        if len(memory) < batch_size:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        state, action, reward, next_state = memory.sample(batch_size=batch_size)
        state = torch.clamp(state.detach(), -10.0, 10.0)
        action = torch.clamp(action.detach(), 0.0, 1.0)
        reward = torch.clamp(reward.detach(), -1000.0, 1000.0)
        next_state = torch.clamp(next_state.detach(), -10.0, 10.0)

        # --- critic update ---
        with torch.no_grad():
            next_action = self.policy_target.forward(next_state)
            q_next = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * q_next

        q = self.critic(state, action)
        critic_loss = F.mse_loss(q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        grad_cfg = self.config.rl_model['sac']['gradient_clipping']
        if grad_cfg.get('enable', True):
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=grad_cfg.get('policy_max_norm', 10.0))
        self.critic_optim.step()

        # --- policy update ---
        pi = self.policy.forward(state)
        q_pi = self.critic(state, pi)
        policy_loss = -q_pi.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        if grad_cfg.get('enable', True):
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=grad_cfg.get('policy_max_norm', 10.0))
        self.policy_optim.step()

        # soft-update targets
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)
            for tp, p in zip(self.policy_target.parameters(), self.policy.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

        # Return format: (critic1, critic2, policy, alpha_loss, alpha)
        # DDPG has no twin critic and no entropy -> critic2=0, alpha_loss=0, alpha=0
        return critic_loss.item(), 0.0, policy_loss.item(), 0.0, 0.0

    # ----- checkpoint interface (matches SAC) -----

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        _ckpt_dir = os.path.join(_RL_DIR, 'checkpoints')
        os.makedirs(_ckpt_dir, exist_ok=True)
        if ckpt_path is None:
            ckpt_path = os.path.join(_ckpt_dir, f"ddpg_checkpoint_{env_name}_{suffix}")

        architecture_info = {
            'algorithm_type': 'DDPG',
            'hidden_dim': self.config.rl_model['networks']['hidden_dim'],
            'policy_type': 'deterministic',
            'state_dim': list(self.policy.parameters())[0].shape[1],
            'action_dim': self.config.rl_model['reservoir']['num_producers']
                          + self.config.rl_model['reservoir']['num_injectors'],
            'num_producers': self.config.rl_model['reservoir']['num_producers'],
            'num_injectors': self.config.rl_model['reservoir']['num_injectors'],
        }

        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'architecture': architecture_info,
        }, ckpt_path)
        print(f"  DDPG checkpoint saved: hidden_dim={architecture_info['hidden_dim']}")

    def load_checkpoint(self, ckpt_path, evaluate=False):
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return False
        try:
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            if 'policy_target_state_dict' in checkpoint:
                self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
            else:
                hard_update(self.policy_target, self.policy)
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            mode = 'eval' if evaluate else 'train'
            for net in (self.policy, self.policy_target, self.critic, self.critic_target):
                getattr(net, mode)()
            print(f"DDPG checkpoint loaded ({mode} mode).")
            return True
        except Exception as e:
            print(f"Error loading DDPG checkpoint: {e}")
            return False

    def update_policy_with_dashboard_config(self, rl_config):
        if hasattr(self.policy, 'update_action_parameters_from_dashboard'):
            self.policy.update_action_parameters_from_dashboard(rl_config)
        if hasattr(self.policy_target, 'update_action_parameters_from_dashboard'):
            self.policy_target.update_action_parameters_from_dashboard(rl_config)
