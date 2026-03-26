"""
Rollout Buffer for PPO (on-policy data collection)

Stores full episode trajectories and computes GAE advantages.
Unlike ReplayMemory (off-policy), this buffer is cleared after each
policy update.
"""
import torch
import numpy as np
from typing import Generator, Tuple


class RolloutBuffer:
    """Fixed-size rollout buffer with GAE computation."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.log_probs = torch.zeros(capacity, 1, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.values = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)

        self.advantages = torch.zeros(capacity, 1, device=device)
        self.returns = torch.zeros(capacity, 1, device=device)

        self.pos = 0
        self.full = False

    def push(self, state, action, log_prob, reward, value, done):
        idx = self.pos
        self.states[idx] = state.detach().squeeze(0)
        self.actions[idx] = action.detach().squeeze(0)
        self.log_probs[idx] = log_prob.detach() if log_prob.dim() == 0 else log_prob.detach().squeeze()
        self.rewards[idx] = reward if isinstance(reward, (int, float)) else reward.detach()
        self.values[idx] = value.detach() if value.dim() == 0 else value.detach().squeeze()
        self.dones[idx] = float(done)

        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        n = self.pos
        last_gae = 0.0
        lv = last_value.detach().squeeze()

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = lv
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_value = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = self.rewards[t].item() + gamma * next_value * next_non_terminal - self.values[t].item()
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_minibatches(self, num_minibatches: int) -> Generator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None
    ]:
        """Yield shuffled minibatches for PPO update epochs."""
        n = self.pos
        indices = np.arange(n)
        np.random.shuffle(indices)
        batch_size = max(1, n // num_minibatches)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            idx_t = torch.tensor(idx, dtype=torch.long, device=self.device)

            yield (
                self.states[idx_t],
                self.actions[idx_t],
                self.log_probs[idx_t],
                self.returns[idx_t],
                self.advantages[idx_t],
            )

    def reset(self):
        self.pos = 0
        self.full = False

    def __len__(self):
        return self.pos
