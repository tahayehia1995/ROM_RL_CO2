"""
Soft Actor-Critic (SAC) Agent
"""
import math
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import QNetwork, DeterministicPolicy, GaussianPolicy
from .utils import hard_update


class SAC(object):
    def __init__(self, num_inputs, u_dim, config):
        super(SAC, self).__init__()
        self.config = config
        
        # Load SAC parameters from config
        sac_config = config.rl_model['sac']
        
        self.gamma = sac_config['discount_factor']
        self.tau = sac_config['soft_update_tau']
        self.alpha = sac_config['entropy']['alpha']
        self.target_update_interval = sac_config['target_update_interval']
        self.automatic_entropy_tuning = sac_config['entropy']['automatic_tuning']

        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Initialize networks with config
        hidden_dim = config.rl_model['networks']['hidden_dim']
        
        self.critic = QNetwork(num_inputs, u_dim, config).to(device=self.device)
        critic_lr = sac_config['learning_rates']['critic']
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = QNetwork(num_inputs, u_dim, config).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Initialize policy based on config
        policy_type = config.rl_model['networks']['policy']['type']
        policy_lr = sac_config['learning_rates']['policy']
        
        if policy_type == 'deterministic':
            self.policy = DeterministicPolicy(num_inputs, u_dim, config).to(device=self.device)
        else:
            self.policy = GaussianPolicy(num_inputs, u_dim, config).to(device=self.device)
            
        self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

    def select_action(self, state, evaluate=False):
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action

    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch):
        """Update only the critic network, completely isolated from policy updates."""
        # Calculate target Q values with no gradient tracking
        with torch.no_grad():
            # Sample actions from the next states
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            
            # Get Q-values from target network
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            # Take the minimum Q-value
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            # Subtract entropy term if using entropy regularization
            if self.alpha > 0:
                min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
                
            # Calculate target using reward and discounted next state value
            next_q_value = reward_batch + self.gamma * min_qf_next_target
        
        # Current critic values (using actual actions taken in the batch)
        qf1, qf2 = self.critic(state_batch, action_batch)
        
        # Compute MSE losses
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Update critics
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        return qf1_loss.item(), qf2_loss.item()
    
    def update_policy(self, state_batch, debug_mode=False):
        """Update policy in a completely isolated computation graph."""
        # Get dimensions from state and action batches for the network
        num_inputs = state_batch.shape[-1]  # Last dimension is the state size
        
        # Calculate action dimension from config
        num_producers = self.config.rl_model['reservoir']['num_producers']
        num_injectors = self.config.rl_model['reservoir']['num_injectors']
        action_dim = num_producers + num_injectors
        
        # IMPORTANT: Make a copy of the critic for policy evaluation
        temp_critic = QNetwork(num_inputs, action_dim, self.config).to(self.device)
        
        with torch.no_grad():
            # Copy parameters without affecting computational history
            for temp_param, critic_param in zip(temp_critic.parameters(), self.critic.parameters()):
                temp_param.data.copy_(critic_param.data)
        
        # Now get actions from the policy in this separate computation context
        state_copy = state_batch.detach().clone().requires_grad_(True)
        pi, log_pi, _ = self.policy.sample(state_copy)
        
        if debug_mode:
            print(f"pi requires_grad: {pi.requires_grad}")
            print(f"state_copy requires_grad: {state_copy.requires_grad}")
        
        # Evaluate Q-values using the temporary critic
        qf1_pi, qf2_pi = temp_critic(state_copy, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        if debug_mode:
            print(f"min_qf_pi requires_grad: {min_qf_pi.requires_grad}")
        
        # Compute policy loss (negative since we want to maximize Q-value)
        policy_loss = torch.mean(-min_qf_pi)
        
        # Add entropy regularization if needed
        if self.alpha > 0:
            alpha_tensor = torch.tensor(self.alpha, device=self.device, requires_grad=False)
            entropy_term = torch.mean(alpha_tensor * log_pi)
            policy_loss = torch.add(policy_loss, entropy_term)
        
        if debug_mode:
            print(f"policy_loss requires_grad: {policy_loss.requires_grad}")
            print(f"policy_loss: {policy_loss}")
        
        # Update policy parameters with gradient clipping from config
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        # Use gradient clipping from config
        gradient_config = self.config.rl_model['sac']['gradient_clipping']
        if gradient_config['enable']:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=gradient_config['policy_max_norm'])
        
        self.policy_optim.step()
        
        # Clean up the temporary network
        del temp_critic
        
        return policy_loss.item()
    
    def update_parameters(self, memory, batch_size, updates):
        """Completely isolated update function with aggressive NaN prevention."""
        
        # NUMERICAL STABILITY CHECK 1: Validate memory has enough samples
        if len(memory) < batch_size:
            log_config = self.config.rl_model.get('logging', {})
            update_frequency = log_config.get('training_update_frequency', 50)
            if updates % update_frequency == 0:
                print(f"⚠️ Not enough samples in memory ({len(memory)} < {batch_size}). Skipping update.")
            return 0.0, 0.0, 0.0, 0.0, self.alpha
        
        # PHASE 1: Update critic with its own isolated batch and computation
        with torch.set_grad_enabled(True):
            # Get a fresh batch for critic
            state_batch_critic, action_batch_critic, reward_batch_critic, next_state_batch_critic = memory.sample(batch_size=batch_size)
            
            # NUMERICAL STABILITY CHECK 2: Validate all inputs
            if torch.isnan(state_batch_critic).any():
                print("🚨 NaN detected in state_batch_critic! Replacing with zeros.")
                state_batch_critic = torch.zeros_like(state_batch_critic)
            if torch.isnan(action_batch_critic).any():
                print("🚨 NaN detected in action_batch_critic! Replacing with 0.5 (mid-range).")
                action_batch_critic = torch.full_like(action_batch_critic, 0.5)
            if torch.isnan(reward_batch_critic).any():
                print("🚨 NaN detected in reward_batch_critic! Replacing with zeros.")
                reward_batch_critic = torch.zeros_like(reward_batch_critic)
            if torch.isnan(next_state_batch_critic).any():
                print("🚨 NaN detected in next_state_batch_critic! Replacing with zeros.")
                next_state_batch_critic = torch.zeros_like(next_state_batch_critic)
                
            # Ensure tensors are fully detached and clipped
            state_batch_critic = torch.clamp(state_batch_critic.detach(), min=-10.0, max=10.0)
            action_batch_critic = torch.clamp(action_batch_critic.detach(), min=0.0, max=1.0)
            reward_batch_critic = torch.clamp(reward_batch_critic.detach(), min=-1000.0, max=1000.0)
            next_state_batch_critic = torch.clamp(next_state_batch_critic.detach(), min=-10.0, max=10.0)
            
            # Update critic
            critic1_loss_val, critic2_loss_val = self.update_critic(
                state_batch_critic, action_batch_critic, reward_batch_critic, next_state_batch_critic
            )
            
            # NUMERICAL STABILITY CHECK 3: Validate critic losses
            if math.isnan(critic1_loss_val) or math.isnan(critic2_loss_val):
                print(f"🚨 NaN detected in critic losses! Q1={critic1_loss_val}, Q2={critic2_loss_val}")
                print("   Skipping this update and resetting networks if needed.")
                return float('nan'), float('nan'), float('nan'), 0.0, self.alpha
        
        # PHASE 2: Clear memory between operations
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # PHASE 3: Update policy with its own isolated batch and computation
        with torch.set_grad_enabled(True):
            # Get a completely fresh batch for policy
            state_batch_policy, _, _, _ = memory.sample(batch_size=batch_size)
            
            # NUMERICAL STABILITY CHECK 4: Validate policy state input
            if torch.isnan(state_batch_policy).any():
                print("🚨 NaN detected in state_batch_policy! Replacing with zeros.")
                state_batch_policy = torch.zeros_like(state_batch_policy)
            
            # Ensure it's a fully new tensor with no history and properly clipped
            state_batch_policy = torch.clamp(state_batch_policy.detach().clone(), min=-10.0, max=10.0).requires_grad_(True)
            
            # Update policy with debug output for the first few updates from config
            debug_episodes = self.config.rl_model['training']['debug_mode_episodes']
            debug_mode = updates < debug_episodes
            policy_loss_val = self.update_policy(state_batch_policy, debug_mode)
            
            # NUMERICAL STABILITY CHECK 5: Validate policy loss
            if math.isnan(policy_loss_val):
                print(f"🚨 NaN detected in policy loss! policy_loss={policy_loss_val}")
                print("   This indicates serious numerical instability in the policy network.")
                print("   Training should be stopped and hyperparameters adjusted.")
                return critic1_loss_val, critic2_loss_val, float('nan'), 0.0, self.alpha
        
        # PHASE 4: Clear memory again
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # PHASE 5: Update target networks
        with torch.no_grad():
            if updates % self.target_update_interval == 0:
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # Create placeholder values for alpha-related returns
        alpha_loss_val = 0.0
        alpha_val = self.alpha
        
        # Return all values
        return critic1_loss_val, critic2_loss_val, policy_loss_val, alpha_loss_val, alpha_val

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        import os
        
        print('Attempting to load models from {}'.format(ckpt_path))
        
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

                if evaluate:
                    self.policy.eval()
                    self.critic.eval()
                    self.critic_target.eval()
                else:
                    self.policy.train()
                    self.critic.train()
                    self.critic_target.train()
                    
                print("Checkpoint loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with randomly initialized models.")
                return False
        else:
            print(f"Checkpoint file not found at {ckpt_path}")
            print("Continuing with randomly initialized models.")
            return False

    def update_policy_with_dashboard_config(self, rl_config):
        """
        🎯 NEW: Update the policy with dashboard configuration
        This ensures your interactive dashboard selections are actually used!
        
        Args:
            rl_config: Dashboard configuration dictionary
        """
        print("🎮 Updating SAC policy with DASHBOARD configuration...")
        
        if hasattr(self.policy, 'update_action_parameters_from_dashboard'):
            self.policy.update_action_parameters_from_dashboard(rl_config)
            print("✅ Policy updated with DASHBOARD action ranges")
        else:
            print("⚠️ Policy does not support dashboard parameter updates")
        
        return True

