"""
Proximal Policy Optimization (PPO) Agent

On-policy actor-critic with clipped surrogate objective, GAE, and
separate value network.  PPO collects a full episode of data, then
performs multiple epochs of minibatch updates.

Reference: Schulman et al. (2017)
           "Proximal Policy Optimization Algorithms"
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import Adam
from torch.distributions import Normal

from .utils import weights_init_
from .ppo_buffer import RolloutBuffer

_RL_DIR = str(Path(__file__).resolve().parent.parent)


class PPOActorCritic(nn.Module):
    """Combined actor-critic network for PPO.

    Actor outputs a Gaussian distribution over [0,1] actions.
    Critic outputs a scalar state-value V(s).
    """

    def __init__(self, num_inputs, num_actions, config):
        super().__init__()
        hidden_dim = config.rl_model['networks']['hidden_dim']

        # shared feature layers
        self.shared1 = nn.Linear(num_inputs, hidden_dim)
        self.shared2 = nn.Linear(hidden_dim, hidden_dim)

        # actor head
        self.mean_head = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # critic head
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(lambda m: weights_init_(m, config))
        self.config = config

    def forward(self, state):
        x = torch.clamp(state, -10.0, 10.0)
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        return x

    def get_action_and_value(self, state, action=None):
        """Return action, log_prob, entropy, value."""
        features = self.forward(state)
        mean = torch.sigmoid(self.mean_head(features))  # [0,1]
        std = self.log_std.exp().expand_as(mean)

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        value = self.value_head(features)
        return action, log_prob, entropy, value

    def get_value(self, state):
        features = self.forward(state)
        return self.value_head(features)


class PPO(object):
    """PPO agent with the same external interface as SAC.

    Key differences from off-policy agents:
    * Uses its own RolloutBuffer (not ReplayMemory).
    * ``update_parameters`` accepts the rollout buffer and performs
      multiple epochs of minibatch updates per call.
    * The training dashboard calls ``collect_step`` during the episode,
      then ``update_from_buffer`` at the end.
    * ``select_action`` returns log_prob and value alongside the action
      when ``evaluate=False`` so that the caller can store them.
    """

    def __init__(self, num_inputs, u_dim, config):
        super(PPO, self).__init__()
        self.config = config

        ppo_config = config.rl_model.get('ppo', {})
        sac_config = config.rl_model['sac']

        self.gamma = ppo_config.get('discount_factor', sac_config['discount_factor'])
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.clip_epsilon = ppo_config.get('clip_epsilon', 0.2)
        self.value_loss_coef = ppo_config.get('value_loss_coef', 0.5)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)
        self.num_epochs = ppo_config.get('num_epochs', 4)
        self.num_minibatches = ppo_config.get('num_minibatches', 4)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        lr = ppo_config.get('learning_rate', sac_config['learning_rates']['policy'])

        self.actor_critic = PPOActorCritic(num_inputs, u_dim, config).to(self.device)
        self.optimizer = Adam(self.actor_critic.parameters(), lr=lr)

        # Expose .policy attribute for compatibility (.eval() / .train() switching)
        self.policy = self.actor_critic

        # Rollout buffer
        max_steps = config.rl_model['training']['max_steps_per_episode']
        self.rollout = RolloutBuffer(max_steps, num_inputs, u_dim, self.device)

        # Tracking
        self._last_log_prob = None
        self._last_value = None

    # ----- unified interface -----

    def select_action(self, state, evaluate=False):
        with torch.set_grad_enabled(not evaluate):
            action, log_prob, _, value = self.actor_critic.get_action_and_value(state)
        if evaluate:
            # deterministic: use mean
            with torch.no_grad():
                features = self.actor_critic.forward(state)
                action = torch.sigmoid(self.actor_critic.mean_head(features))
            return action
        self._last_log_prob = log_prob.detach()
        self._last_value = value.detach()
        return action

    def collect_step(self, state, action, reward, done):
        """Store one transition in the rollout buffer."""
        log_prob = self._last_log_prob if self._last_log_prob is not None else torch.zeros(1, device=self.device)
        value = self._last_value if self._last_value is not None else torch.zeros(1, device=self.device)
        r = reward.item() if hasattr(reward, 'item') else reward
        self.rollout.push(state, action, log_prob, r, value, done)

    def update_from_buffer(self, last_state):
        """Run PPO update after a full episode has been collected.

        Returns the same 5-tuple as SAC.update_parameters for compatibility.
        """
        with torch.no_grad():
            last_value = self.actor_critic.get_value(last_state)
        self.rollout.compute_gae(last_value, self.gamma, self.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.num_epochs):
            for states_mb, actions_mb, old_log_probs_mb, returns_mb, advantages_mb in \
                    self.rollout.get_minibatches(self.num_minibatches):

                # normalise advantages
                adv = advantages_mb
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                _, new_log_probs, entropy, new_values = \
                    self.actor_critic.get_action_and_value(states_mb, actions_mb)

                ratio = (new_log_probs - old_log_probs_mb).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, returns_mb)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                n_updates += 1

        self.rollout.reset()

        avg_p = total_policy_loss / max(n_updates, 1)
        avg_v = total_value_loss / max(n_updates, 1)
        return avg_v, 0.0, avg_p, 0.0, 0.0

    def update_parameters(self, memory, batch_size, updates):
        """Compatibility shim -- for PPO the real update happens via
        ``update_from_buffer`` at episode end.  This is a no-op when called
        per-step from the standard off-policy training loop."""
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # ----- checkpoint interface -----

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        _ckpt_dir = os.path.join(_RL_DIR, 'checkpoints')
        os.makedirs(_ckpt_dir, exist_ok=True)
        if ckpt_path is None:
            ckpt_path = os.path.join(_ckpt_dir, f"ppo_checkpoint_{env_name}_{suffix}")

        architecture_info = {
            'algorithm_type': 'PPO',
            'hidden_dim': self.config.rl_model['networks']['hidden_dim'],
            'policy_type': 'gaussian',
            'state_dim': self.actor_critic.shared1.in_features,
            'action_dim': self.config.rl_model['reservoir']['num_producers']
                          + self.config.rl_model['reservoir']['num_injectors'],
            'num_producers': self.config.rl_model['reservoir']['num_producers'],
            'num_injectors': self.config.rl_model['reservoir']['num_injectors'],
        }

        torch.save({
            'policy_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'architecture': architecture_info,
        }, ckpt_path)
        print(f"  PPO checkpoint saved: hidden_dim={architecture_info['hidden_dim']}")

    def load_checkpoint(self, ckpt_path, evaluate=False):
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return False
        try:
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if evaluate:
                self.actor_critic.eval()
            else:
                self.actor_critic.train()
            print(f"PPO checkpoint loaded ({'eval' if evaluate else 'train'} mode).")
            return True
        except Exception as e:
            print(f"Error loading PPO checkpoint: {e}")
            return False

    def update_policy_with_dashboard_config(self, rl_config):
        pass  # PPO actor-critic doesn't store dashboard action ranges
