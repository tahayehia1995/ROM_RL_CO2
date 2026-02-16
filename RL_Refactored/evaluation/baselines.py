"""
Baseline Policies for Comparison
=================================

Simple baseline policies to compare against trained RL policies.
These provide reference points to assess the value of learning.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaselinePolicy(ABC):
    """
    Abstract base class for baseline policies.
    
    All baseline policies output actions in [0, 1] normalized space,
    matching the RL policy output format.
    """
    
    def __init__(self, num_actions: int = 6, device: Optional[torch.device] = None):
        """
        Initialize baseline policy.
        
        Args:
            num_actions: Number of action dimensions (default 6: 3 BHP + 3 Gas)
            device: PyTorch device for tensor output
        """
        self.num_actions = num_actions
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = self.__class__.__name__
    
    @abstractmethod
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select action given current state.
        
        Args:
            state: Current state tensor (ignored by most baselines)
            
        Returns:
            Action tensor in [0, 1] space with shape (1, num_actions)
        """
        pass
    
    def reset(self):
        """Reset any internal state (called at episode start)."""
        pass
    
    def __str__(self) -> str:
        return self.name


class RandomPolicy(BaselinePolicy):
    """
    Random baseline: uniformly random actions at each step.
    
    This represents the lower bound of expected performance.
    Any reasonable policy should beat this baseline.
    """
    
    def __init__(self, num_actions: int = 6, device: Optional[torch.device] = None, seed: Optional[int] = None):
        super().__init__(num_actions, device)
        self.name = "Random"
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Generate uniformly random action in [0, 1]."""
        action = torch.rand(1, self.num_actions, device=self.device)
        return action


class MidpointPolicy(BaselinePolicy):
    """
    Midpoint baseline: all controls at 0.5 (center of range).
    
    This is a simple "do nothing special" baseline that keeps
    all controls at their midpoint values throughout the episode.
    """
    
    def __init__(self, num_actions: int = 6, device: Optional[torch.device] = None):
        super().__init__(num_actions, device)
        self.name = "Midpoint"
        self._action = None
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return midpoint action (0.5 for all controls)."""
        if self._action is None:
            self._action = torch.full((1, self.num_actions), 0.5, device=self.device)
        return self._action
    
    def reset(self):
        """Reset cached action."""
        self._action = None


class NaiveMaxGasPolicy(BaselinePolicy):
    """
    Naive Max Gas baseline: Low BHP + High Gas injection.
    
    This strategy prioritizes CO2 storage by:
    - Setting producer BHP low (0.1) to minimize production pressure
    - Setting gas injection high (0.9) to maximize CO2 injection
    
    Action order: [BHP_P1, BHP_P2, BHP_P3, Gas_I1, Gas_I2, Gas_I3]
    """
    
    def __init__(self, num_actions: int = 6, device: Optional[torch.device] = None,
                 num_producers: int = 3, num_injectors: int = 3,
                 bhp_value: float = 0.1, gas_value: float = 0.9):
        super().__init__(num_actions, device)
        self.name = "NaiveMaxGas"
        self.num_producers = num_producers
        self.num_injectors = num_injectors
        self.bhp_value = bhp_value
        self.gas_value = gas_value
        self._action = None
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return low BHP, high gas injection action."""
        if self._action is None:
            action = torch.zeros(1, self.num_actions, device=self.device)
            # Low BHP for producers (first num_producers actions)
            action[0, :self.num_producers] = self.bhp_value
            # High gas injection for injectors (remaining actions)
            action[0, self.num_producers:] = self.gas_value
            self._action = action
        return self._action
    
    def reset(self):
        """Reset cached action."""
        self._action = None


class NaiveLowGasPolicy(BaselinePolicy):
    """
    Naive Low Gas baseline: High BHP + Low Gas injection.
    
    This is the opposite strategy:
    - Setting producer BHP high (0.9) to maximize production
    - Setting gas injection low (0.1) to minimize injection cost
    
    Action order: [BHP_P1, BHP_P2, BHP_P3, Gas_I1, Gas_I2, Gas_I3]
    """
    
    def __init__(self, num_actions: int = 6, device: Optional[torch.device] = None,
                 num_producers: int = 3, num_injectors: int = 3,
                 bhp_value: float = 0.9, gas_value: float = 0.1):
        super().__init__(num_actions, device)
        self.name = "NaiveLowGas"
        self.num_producers = num_producers
        self.num_injectors = num_injectors
        self.bhp_value = bhp_value
        self.gas_value = gas_value
        self._action = None
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return high BHP, low gas injection action."""
        if self._action is None:
            action = torch.zeros(1, self.num_actions, device=self.device)
            # High BHP for producers (first num_producers actions)
            action[0, :self.num_producers] = self.bhp_value
            # Low gas injection for injectors (remaining actions)
            action[0, self.num_producers:] = self.gas_value
            self._action = action
        return self._action
    
    def reset(self):
        """Reset cached action."""
        self._action = None


class ConstantPolicy(BaselinePolicy):
    """
    Constant baseline: user-specified constant action.
    
    Allows testing any specific control strategy.
    """
    
    def __init__(self, action_values: list, device: Optional[torch.device] = None):
        """
        Initialize with specific action values.
        
        Args:
            action_values: List of action values in [0, 1]
            device: PyTorch device
        """
        num_actions = len(action_values)
        super().__init__(num_actions, device)
        self.name = "Constant"
        self._action = torch.tensor([action_values], dtype=torch.float32, device=self.device)
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return the constant action."""
        return self._action


class LinearRampPolicy(BaselinePolicy):
    """
    Linear ramp baseline: linearly increase/decrease actions over episode.
    
    This tests whether gradual changes in controls are beneficial.
    """
    
    def __init__(self, num_actions: int = 6, device: Optional[torch.device] = None,
                 start_values: Optional[list] = None, end_values: Optional[list] = None,
                 num_steps: int = 30):
        super().__init__(num_actions, device)
        self.name = "LinearRamp"
        self.num_steps = num_steps
        self.current_step = 0
        
        # Default: ramp from low to high
        self.start_values = start_values or [0.2] * num_actions
        self.end_values = end_values or [0.8] * num_actions
        
        self.start_tensor = torch.tensor([self.start_values], dtype=torch.float32, device=self.device)
        self.end_tensor = torch.tensor([self.end_values], dtype=torch.float32, device=self.device)
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return linearly interpolated action based on current step."""
        if self.num_steps <= 1:
            t = 1.0
        else:
            t = self.current_step / (self.num_steps - 1)
        
        action = self.start_tensor + t * (self.end_tensor - self.start_tensor)
        action = torch.clamp(action, 0.0, 1.0)
        
        self.current_step += 1
        return action
    
    def reset(self):
        """Reset step counter for new episode."""
        self.current_step = 0


def get_all_baselines(num_actions: int = 6, device: Optional[torch.device] = None,
                      num_producers: int = 3, num_injectors: int = 3) -> dict:
    """
    Get dictionary of all standard baseline policies.
    
    Args:
        num_actions: Number of action dimensions
        device: PyTorch device
        num_producers: Number of producer wells
        num_injectors: Number of injector wells
        
    Returns:
        Dictionary mapping baseline names to policy instances
    """
    return {
        'Random': RandomPolicy(num_actions, device),
        'Midpoint': MidpointPolicy(num_actions, device),
        'NaiveMaxGas': NaiveMaxGasPolicy(num_actions, device, num_producers, num_injectors),
        'NaiveLowGas': NaiveLowGasPolicy(num_actions, device, num_producers, num_injectors)
    }
