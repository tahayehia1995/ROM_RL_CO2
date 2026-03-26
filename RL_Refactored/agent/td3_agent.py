"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

Improvements over DDPG:
1. Twin critics -- take the minimum Q-value to reduce overestimation
2. Delayed policy updates -- update actor less frequently than critic
3. Target policy smoothing -- add noise to target actions

Reference: Fujimoto, Hoof & Meger (2018)
           "Addressing Function Approximation Error in Actor-Critic Methods"
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import Adam

from .networks import QNetwork, DeterministicPolicy
from .utils import hard_update, weights_init_

_RL_DIR = str(Path(__file__).resolve().parent.parent)


class TD3(object):
    """TD3 agent with the same interface as SAC."""

    def __init__(self, num_inputs, u_dim, config):
        super(TD3, self).__init__()
        self.config = config

        td3_config = config.rl_model.get('td3', {})
        sac_config = config.rl_model['sac']

        self.gamma = td3_config.get('discount_factor', sac_config['discount_factor'])
        self.tau = td3_config.get('soft_update_tau', sac_config['soft_update_tau'])
        self.policy_delay = td3_config.get('policy_delay', 2)
        self.target_noise_std = td3_config.get('target_noise_std', 0.2)
        self.target_noise_clip = td3_config.get('target_noise_clip', 0.5)
        self.exploration_noise_std = td3_config.get('exploration_noise_std', 0.1)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        critic_lr = td3_config.get('critic_lr', sac_config['learning_rates']['critic'])
        policy_lr = td3_config.get('policy_lr', sac_config['learning_rates']['policy'])

        # Twin Q-networks (reuse existing QNetwork which already has Q1+Q2)
        self.critic = QNetwork(num_inputs, u_dim, config).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = QNetwork(num_inputs, u_dim, config).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Deterministic policy
        self.policy = DeterministicPolicy(num_inputs, u_dim, config).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        self.policy_target = DeterministicPolicy(num_inputs, u_dim, config).to(self.device)
        hard_update(self.policy_target, self.policy)

        self._update_count = 0

    # ----- unified interface -----

    def select_action(self, state, evaluate=False):
        mean = self.policy.forward(state)
        if evaluate:
            return mean
        noise = torch.randn_like(mean) * self.exploration_noise_std
        return torch.clamp(mean + noise, 0.0, 1.0)

    def update_parameters(self, memory, batch_size, updates):
        if len(memory) < batch_size:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        self._update_count += 1

        state, action, reward, next_state = memory.sample(batch_size=batch_size)
        state = torch.clamp(state.detach(), -10.0, 10.0)
        action = torch.clamp(action.detach(), 0.0, 1.0)
        reward = torch.clamp(reward.detach(), -1000.0, 1000.0)
        next_state = torch.clamp(next_state.detach(), -10.0, 10.0)

        # --- critic update ---
        with torch.no_grad():
            next_action = self.policy_target.forward(next_state)
            noise = (torch.randn_like(next_action) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = torch.clamp(next_action + noise, 0.0, 1.0)

            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + self.gamma * q_next

        q1, q2 = self.critic(state, action)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        grad_cfg = self.config.rl_model['sac']['gradient_clipping']
        if grad_cfg.get('enable', True):
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=grad_cfg.get('policy_max_norm', 10.0))
        self.critic_optim.step()

        # --- delayed policy update ---
        policy_loss_val = 0.0
        if self._update_count % self.policy_delay == 0:
            pi = self.policy.forward(state)
            q1_pi, _ = self.critic(state, pi)
            policy_loss = -q1_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            if grad_cfg.get('enable', True):
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=grad_cfg.get('policy_max_norm', 10.0))
            self.policy_optim.step()
            policy_loss_val = policy_loss.item()

            # soft-update targets
            with torch.no_grad():
                for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                    tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)
                for tp, p in zip(self.policy_target.parameters(), self.policy.parameters()):
                    tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

        return critic1_loss.item(), critic2_loss.item(), policy_loss_val, 0.0, 0.0

    # ----- checkpoint interface (matches SAC) -----

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        _ckpt_dir = os.path.join(_RL_DIR, 'checkpoints')
        os.makedirs(_ckpt_dir, exist_ok=True)
        if ckpt_path is None:
            ckpt_path = os.path.join(_ckpt_dir, f"td3_checkpoint_{env_name}_{suffix}")

        architecture_info = {
            'algorithm_type': 'TD3',
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
        print(f"  TD3 checkpoint saved: hidden_dim={architecture_info['hidden_dim']}")

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
            print(f"TD3 checkpoint loaded ({mode} mode).")
            return True
        except Exception as e:
            print(f"Error loading TD3 checkpoint: {e}")
            return False

    def update_policy_with_dashboard_config(self, rl_config):
        if hasattr(self.policy, 'update_action_parameters_from_dashboard'):
            self.policy.update_action_parameters_from_dashboard(rl_config)
        if hasattr(self.policy_target, 'update_action_parameters_from_dashboard'):
            self.policy_target.update_action_parameters_from_dashboard(rl_config)
