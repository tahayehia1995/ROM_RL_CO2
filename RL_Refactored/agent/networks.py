"""
Neural Networks for RL Agent
Q-Networks, Value Networks, and Policy Networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .utils import weights_init_


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, config):
        super(ValueNetwork, self).__init__()
        
        hidden_dim = config.rl_model['networks']['hidden_dim']

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, config.rl_model['networks']['q_network']['num_outputs'])

        self.apply(lambda m: weights_init_(m, config))

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, config):
        super(QNetwork, self).__init__()
        
        hidden_dim = config.rl_model['networks']['hidden_dim']
        num_outputs = config.rl_model['networks']['q_network']['num_outputs']

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_outputs)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_outputs)

        self.apply(lambda m: weights_init_(m, config))

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, config, action_space=None):
        super(DeterministicPolicy, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.config = config
        
        hidden_dim = config.rl_model['networks']['hidden_dim']
        num_producers = config.rl_model['reservoir']['num_producers']
        num_injectors = config.rl_model['reservoir']['num_injectors']
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.mean_bhp = nn.Linear(hidden_dim, num_producers)
        self.mean_rate = nn.Linear(hidden_dim, num_injectors)
        self.noise = torch.Tensor(num_actions).to(self.device)

        self.apply(lambda m: weights_init_(m, config))

        # Store action parameters (will be updated by JSON normalizer)
        # Initialize with default fallback values
        self.injector_gas_min = torch.tensor(6180072, device=self.device)
        self.injector_gas_max = torch.tensor(100646896, device=self.device)
        self.injector_bhp_min = torch.tensor(1087.78, device=self.device)
        self.injector_bhp_max = torch.tensor(5050.43, device=self.device)
        self.producer_bhp_min = torch.tensor(1087.78, device=self.device)
        self.producer_bhp_max = torch.tensor(5050.43, device=self.device)

    def update_action_parameters_from_dashboard(self, rl_config):
        """
        🎯 UPDATE: Store dashboard ranges for reference/logging
        Policy now outputs [0,1] actions; environment applies dashboard ranges
        
        Args:
            rl_config: Dashboard configuration dictionary
        """
        if not rl_config:
            print(f"   ❌ No dashboard configuration provided - using fallback ranges")
            return
        
        action_ranges = rl_config.get('action_ranges', {})
        if not action_ranges:
            print(f"   ❌ No action ranges in dashboard config - using fallback ranges")
            return
        
        # Extract BHP ranges from dashboard - STORE FOR REFERENCE
        bhp_ranges = action_ranges.get('bhp', {})
        if bhp_ranges:
            # Get min/max across all producer wells from dashboard
            bhp_mins = [ranges['min'] for ranges in bhp_ranges.values()]
            bhp_maxs = [ranges['max'] for ranges in bhp_ranges.values()]
            
            if bhp_mins and bhp_maxs:
                dashboard_bhp_min = min(bhp_mins)
                dashboard_bhp_max = max(bhp_maxs)
                
                # Store for reference (not used in forward() anymore)
                self.producer_bhp_min = torch.tensor(dashboard_bhp_min, device=self.device)
                self.producer_bhp_max = torch.tensor(dashboard_bhp_max, device=self.device)
            else:
                print(f"   ⚠️ Empty BHP ranges in dashboard config")
        else:
            print(f"   ⚠️ No BHP ranges in dashboard config")
        
        # Extract Gas Injection ranges from dashboard - STORE FOR REFERENCE
        gas_ranges = action_ranges.get('gas_injection', {})
        if gas_ranges:
            # Get min/max across all injector wells from dashboard
            gas_mins = [ranges['min'] for ranges in gas_ranges.values()]
            gas_maxs = [ranges['max'] for ranges in gas_ranges.values()]
            
            if gas_mins and gas_maxs:
                dashboard_gas_min = min(gas_mins)
                dashboard_gas_max = max(gas_maxs)
                
                # Store for reference (not used in forward() anymore)
                self.injector_gas_min = torch.tensor(dashboard_gas_min, device=self.device)
                self.injector_gas_max = torch.tensor(dashboard_gas_max, device=self.device)
                
                print(f"   ✅ Gas Injection from DASHBOARD: [{dashboard_gas_min:.0f}, {dashboard_gas_max:.0f}] ft³/day")
                print(f"      📊 Dashboard wells: {list(gas_ranges.keys())}")
                print(f"      🔧 Environment will map [0,1] → this range")
            else:
                print(f"   ⚠️ Empty gas ranges in dashboard config")
        else:
            print(f"   ⚠️ No gas injection ranges in dashboard config")
        
        print(f"   🎯 DASHBOARD CONFIGURATION STORED FOR REFERENCE!")
        print(f"   🔧 Policy outputs [0,1] → Environment applies dashboard ranges → ROM normalization")

    def forward(self, state):
        """Forward pass outputting [0,1] actions that will be mapped to dashboard ranges by environment"""
        # Add input validation
        if torch.isnan(state).any():
            print(f"⚠️ NaN detected in input state!")
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values for numerical stability
        state = torch.clamp(state, min=-10.0, max=10.0)
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        # Generate BASE [0,1] actions - let environment apply dashboard constraints
        # Policy outputs raw [0,1] values, environment maps to dashboard ranges
        
        # Get raw neural network outputs
        bhp_raw = torch.clamp(self.mean_bhp(x), min=-10.0, max=10.0)
        rate_raw = torch.clamp(self.mean_rate(x), min=-10.0, max=10.0)
        
        if self.config.rl_model['networks']['policy']['output_activation'] == 'sigmoid':
            # Sigmoid activation for [0,1] output
            base_bhp = torch.sigmoid(bhp_raw)    # [0,1]
            base_rate = torch.sigmoid(rate_raw)  # [0,1]
        else:
            # Tanh activation mapped to [0,1]
            base_bhp = (torch.tanh(bhp_raw) + 1.0) / 2.0   # [-1,1] → [0,1]
            base_rate = (torch.tanh(rate_raw) + 1.0) / 2.0  # [-1,1] → [0,1]
        
        # 🎯 CRITICAL FIX: Output pure [0,1] actions - NO dashboard range application here
        # Environment will map [0,1] → dashboard ranges → ROM normalization
        
        # Combine in environment-expected order: [BHP(3), Gas(3)] - all in [0,1]
        mean = torch.cat((base_bhp, base_rate), dim=-1)
        
        # Final validation and clipping to ensure [0,1] range
        mean = torch.clamp(mean, min=0.0, max=1.0)
        
        # Final NaN check
        if torch.isnan(mean).any():
            print(f"🚨 NaN detected in policy output! Replacing with safe values.")
            mean = torch.nan_to_num(mean, nan=0.5, posinf=1.0, neginf=0.0)
        
        return mean

    def sample(self, state):
        """
        Sample actions from the deterministic policy.
        For a deterministic policy, the action is simply the mean output.
        We create a zero log probability tensor to maintain compatibility with SAC interface.
        """
        # Forward pass to get action means
        mean = self.forward(state)
        
        # In deterministic policy, the action is exactly the mean
        action = mean
        
        # Create log_prob tensor (always 0 for deterministic policy)
        # Make sure it has proper gradient tracking if the input state does
        requires_grad = state.requires_grad
        log_prob = torch.zeros(action.size(0), 1, device=state.device, requires_grad=requires_grad)
        
        # Debug info
        if torch.is_grad_enabled() and requires_grad:
            # Print first time sample is called with grad enabled
            if not hasattr(self, '_printed_grad_info'):
                print(f"[Debug] Sample called with gradient tracking enabled")
                print(f"[Debug] Action requires_grad: {action.requires_grad}")
                print(f"[Debug] State requires_grad: {state.requires_grad}")
                self._printed_grad_info = True
        
        return action, log_prob, mean

    def to(self, device):
        print("Using JSON-based to() method!")
        self.device = device
        self.noise = self.noise.to(device)
        # Update JSON-based tensors
        self.injector_gas_min = self.injector_gas_min.to(device)
        self.injector_gas_max = self.injector_gas_max.to(device)
        self.injector_bhp_min = self.injector_bhp_min.to(device)
        self.injector_bhp_max = self.injector_bhp_max.to(device)
        self.producer_bhp_min = self.producer_bhp_min.to(device)
        self.producer_bhp_max = self.producer_bhp_max.to(device)
        return super(DeterministicPolicy, self).to(device)
    

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, config, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.config = config
        hidden_dim = config.rl_model['networks']['hidden_dim']
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(lambda m: weights_init_(m, config))

        # Action rescaling from config
        if action_space is None:
            default_space = config.rl_model['gaussian_policy']['default_action_space']
            self.action_scale = torch.tensor(default_space['scale'])
            self.action_bias = torch.tensor(default_space['bias'])
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def update_action_parameters_from_dashboard(self, rl_config):
        """
        🎯 NEW: Update action parameters using DASHBOARD configuration for GaussianPolicy
        
        Args:
            rl_config: Dashboard configuration dictionary
        """
        print(f"🎮 Updating GaussianPolicy with DASHBOARD configuration...")
        
        if not rl_config:
            print(f"   ❌ No dashboard configuration provided")
            return
        
        action_ranges = rl_config.get('action_ranges', {})
        if not action_ranges:
            print(f"   ❌ No action ranges in dashboard config")
            return
        
        # For GaussianPolicy, we mainly need to note the ranges for action space scaling
        bhp_ranges = action_ranges.get('bhp', {})
        gas_ranges = action_ranges.get('gas_injection', {})
        
        if bhp_ranges and gas_ranges:
            print(f"   ✅ Dashboard action ranges noted for GaussianPolicy")
            print(f"      🏭 BHP wells: {list(bhp_ranges.keys())}")
            print(f"      💨 Gas wells: {list(gas_ranges.keys())}")
            # GaussianPolicy uses action_scale and action_bias which are typically set from action_space
            # The actual constraining happens in the environment
        else:
            print(f"   ⚠️ Incomplete action ranges in dashboard config")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Use config parameters for bounds
        log_std_bounds = self.config.rl_model['gaussian_policy']['log_std_bounds']
        log_std = torch.clamp(log_std, min=log_std_bounds['min'], max=log_std_bounds['max'])
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        
        if self.config.rl_model['networks']['policy']['output_activation'] == 'sigmoid':
            y_t = torch.sigmoid(x_t)
        else:
            y_t = torch.tanh(x_t)
            
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound using config epsilon
        epsilon = float(self.config.rl_model['gaussian_policy']['epsilon'])
        if self.config.rl_model['networks']['policy']['output_activation'] == 'sigmoid':
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        else:
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            
        log_prob = log_prob.sum(1, keepdim=True)
        
        if self.config.rl_model['networks']['policy']['output_activation'] == 'sigmoid':
            mean = torch.sigmoid(mean)
        else:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        # Check if action_scale is a tensor before calling .to()
        if hasattr(self.action_scale, 'to'):
            self.action_scale = self.action_scale.to(device)
        # Check if action_bias is a tensor before calling .to()
        if hasattr(self.action_bias, 'to'):
            self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

