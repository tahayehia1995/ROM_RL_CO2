"""
Dynamic Loss Weighting Strategies for Multi-Task Learning

This module implements state-of-the-art dynamic loss weighting approaches for multi-task learning,
specifically designed for the E2C model with multiple physics-based losses.

Supported strategies:
1. GradNorm: Gradient Normalization for Adaptive Loss Balancing
2. Uncertainty Weighting: Multi-Task Learning Uncertainty
3. DWA: Dynamic Weight Averaging  
4. YOTO: You Only Train Once (learnable loss weights)
5. Adaptive Curriculum: Loss-based adaptive scheduling

References:
- GradNorm: Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing" ICML 2018
- Uncertainty: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" CVPR 2018
- DWA: Liu et al. "Dynamic Weight Averaging for Multi-Task Learning" NeurIPS 2018
- YOTO: Sakaridis "You Only Train Once" arXiv 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class DynamicLossWeighter(ABC):
    """
    Abstract base class for dynamic loss weighting strategies.
    
    All strategies should inherit from this class and implement the required methods.
    """
    
    def __init__(self, 
                 task_names: List[str],
                 initial_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize the dynamic loss weighter.
        
        Args:
            task_names: List of task names (e.g., ['reconstruction', 'transition', 'observation'])
            initial_weights: Initial weights for each task
            device: Device to use for computations
        """
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = device
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in task_names}
        
        self.weights = torch.tensor([initial_weights[name] for name in task_names], 
                                  device=device, dtype=torch.float32)
        
        # Training statistics
        self.step_count = 0
        self.epoch_count = 0
        self.loss_history = {name: [] for name in task_names}
        self.weight_history = {name: [] for name in task_names}
        
    @abstractmethod
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      model: nn.Module, 
                      **kwargs) -> Dict[str, float]:
        """
        Update the loss weights based on the current losses and model state.
        
        Args:
            losses: Dictionary of task losses
            model: The model being trained
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            Dictionary of updated weights
        """
        pass
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights as a dictionary."""
        return {name: float(weight) for name, weight in zip(self.task_names, self.weights)}
    
    def get_weight_tensor(self) -> torch.Tensor:
        """Get current weights as a tensor."""
        return self.weights.clone()
    
    def step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def epoch(self):
        """Increment epoch counter."""
        self.epoch_count += 1
    
    def log_losses_and_weights(self, losses: Dict[str, torch.Tensor]):
        """Log current losses and weights for monitoring."""
        for name, loss in losses.items():
            if name in self.loss_history:
                self.loss_history[name].append(float(loss.detach().cpu()))
        
        for name, weight in self.get_weights().items():
            self.weight_history[name].append(weight)


class GradNormWeighter(DynamicLossWeighter):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    
    Reference: Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing 
    in Deep Multitask Networks" ICML 2018
    """
    
    def __init__(self, 
                 task_names: List[str],
                 alpha: float = 0.12,
                 learning_rate: float = 1e-4,
                 target_layer_name: str = 'last_shared_layer',
                 restoring_force_alpha: float = 0.12,
                 initial_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize GradNorm weighter.
        
        Args:
            task_names: List of task names
            alpha: Restoring force strength (0 = balanced losses, >0 = considers training rates)
            learning_rate: Learning rate for weight updates
            target_layer_name: Name of layer to use for gradient calculation
            restoring_force_alpha: Alpha parameter for restoring force
            initial_weights: Initial task weights
            device: Device for computations
        """
        super().__init__(task_names, initial_weights, device, **kwargs)
        
        self.alpha = alpha
        self.restoring_force_alpha = restoring_force_alpha
        
        # Make weights learnable parameters
        self.weights = nn.Parameter(self.weights)
        self.weight_optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        
        # Track initial losses for relative training rate calculation
        self.initial_losses = None
        self.target_layer_name = target_layer_name
        
        # History for gradient statistics
        self.gradient_norms = {name: [] for name in task_names}
        
    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Find the target layer for gradient calculation."""
        # Try common naming patterns
        for name, module in model.named_modules():
            if (self.target_layer_name in name or 
                'last_shared' in name or 
                'backbone' in name or
                'encoder' in name):
                return module
        
        # Fallback: use the last layer with gradients
        last_layer = None
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                last_layer = module
        
        if last_layer is None:
            warnings.warn("Could not find suitable target layer for GradNorm. Using model parameters.")
            return model
        
        return last_layer
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      model: nn.Module,
                      retain_graph: bool = True,
                      **kwargs) -> Dict[str, float]:
        """
        Update weights using GradNorm algorithm.
        
        Args:
            losses: Dictionary of task losses
            model: Model being trained
            retain_graph: Whether to retain computation graph
            
        Returns:
            Updated weights dictionary
        """
        if self.initial_losses is None:
            # Store initial losses for relative training rate calculation
            self.initial_losses = {name: float(loss.detach()) for name, loss in losses.items()}
        
        # Find target layer
        target_layer = self._find_target_layer(model)
        target_params = list(target_layer.parameters())
        
        if not target_params:
            warnings.warn("No parameters found in target layer. Skipping GradNorm update.")
            return self.get_weights()
        
        # Calculate gradients for each task
        task_gradients = {}
        gradient_norms = torch.zeros(self.num_tasks, device=self.device)
        
        for i, (task_name, loss) in enumerate(losses.items()):
            if task_name not in self.task_names:
                continue
                
            # Clear gradients
            for param in target_params:
                if param.grad is not None:
                    param.grad.zero_()
            
            # Compute gradients for this task
            weighted_loss = self.weights[i] * loss
            grads = torch.autograd.grad(
                weighted_loss, 
                target_params, 
                retain_graph=True, 
                create_graph=True
            )
            
            # Calculate gradient norm
            grad_norm = 0.0
            for grad in grads:
                if grad is not None:
                    grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            
            gradient_norms[i] = grad_norm
            task_gradients[task_name] = grad_norm
            
            # Log gradient norms
            self.gradient_norms[task_name].append(float(grad_norm.detach().cpu()))
        
        # Calculate average gradient norm
        avg_grad_norm = gradient_norms.mean()
        
        # Calculate relative training rates
        if self.epoch_count > 0:  # Skip first epoch
            relative_rates = torch.zeros(self.num_tasks, device=self.device)
            for i, task_name in enumerate(self.task_names):
                if task_name in losses and task_name in self.initial_losses:
                    current_loss = float(losses[task_name].detach())
                    initial_loss = self.initial_losses[task_name]
                    if initial_loss > 0:
                        relative_rates[i] = current_loss / initial_loss
            
            # Calculate target gradient norms using relative training rates
            if relative_rates.sum() > 0:
                avg_relative_rate = relative_rates.mean()
                target_grad_norms = avg_grad_norm * (relative_rates / avg_relative_rate) ** self.alpha
            else:
                target_grad_norms = avg_grad_norm * torch.ones_like(gradient_norms)
        else:
            # For first epoch, target equal gradients
            target_grad_norms = avg_grad_norm * torch.ones_like(gradient_norms)
        
        # Calculate GradNorm loss
        gradnorm_loss = torch.sum(torch.abs(gradient_norms - target_grad_norms.detach()))
        
        # Update weights
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.weight_optimizer.step()
        
        # Renormalize weights to prevent drift
        with torch.no_grad():
            self.weights.data = torch.clamp(self.weights.data, min=0.01)  # Prevent negative weights
            self.weights.data = self.weights.data / self.weights.data.sum() * self.num_tasks
        
        # Log for monitoring
        self.log_losses_and_weights(losses)
        
        return self.get_weights()


class UncertaintyWeighter(DynamicLossWeighter):
    """
    Uncertainty Weighting: Multi-Task Learning Using Uncertainty to Weigh Losses
    
    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics" CVPR 2018
    """
    
    def __init__(self, 
                 task_names: List[str],
                 learning_rate: float = 1e-3,
                 initial_log_vars: Optional[Dict[str, float]] = None,
                 regularization_weight: float = 0.0,
                 initial_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize Uncertainty weighter.
        
        Args:
            task_names: List of task names
            learning_rate: Learning rate for uncertainty parameter updates
            initial_log_vars: Initial log variance for each task
            regularization_weight: L2 regularization on log variances
            initial_weights: Initial task weights (not used in uncertainty weighting)
            device: Device for computations
        """
        super().__init__(task_names, initial_weights, device, **kwargs)
        
        # Initialize log variances (learnable uncertainty parameters)
        if initial_log_vars is None:
            initial_log_vars = {name: 0.0 for name in task_names}
        
        self.log_vars = nn.Parameter(
            torch.tensor([initial_log_vars.get(name, 0.0) for name in task_names], 
                        device=device, dtype=torch.float32)
        )
        
        self.uncertainty_optimizer = torch.optim.Adam([self.log_vars], lr=learning_rate)
        self.regularization_weight = regularization_weight
        
        # Track uncertainty values
        self.uncertainty_history = {name: [] for name in task_names}
    
    def get_uncertainties(self) -> Dict[str, float]:
        """Get current uncertainty values (sigma^2 = exp(log_var))."""
        uncertainties = torch.exp(self.log_vars)
        return {name: float(uncertainty) for name, uncertainty in zip(self.task_names, uncertainties)}
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      model: nn.Module,
                      **kwargs) -> Dict[str, float]:
        """
        Update weights using uncertainty weighting.
        
        The uncertainty weighting formula is:
        L_total = sum_i (1/(2*sigma_i^2)) * L_i + (1/2) * log(sigma_i^2)
        
        Where sigma_i^2 = exp(log_var_i) is the learned uncertainty for task i.
        """
        # Calculate uncertainties
        uncertainties = torch.exp(self.log_vars)  # sigma^2 = exp(log_var)
        
        # Calculate uncertainty-weighted loss
        uncertainty_loss = 0.0
        task_losses = torch.zeros(self.num_tasks, device=self.device)
        
        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                task_loss = losses[task_name]
                task_losses[i] = task_loss
                
                # Uncertainty weighting: (1/(2*sigma^2)) * L + (1/2) * log(sigma^2)
                weighted_term = task_loss / (2 * uncertainties[i]) + self.log_vars[i] / 2
                uncertainty_loss += weighted_term
        
        # Add regularization on log variances
        if self.regularization_weight > 0:
            uncertainty_loss += self.regularization_weight * torch.sum(self.log_vars ** 2)
        
        # Update uncertainty parameters
        self.uncertainty_optimizer.zero_grad()
        uncertainty_loss.backward(retain_graph=True)
        self.uncertainty_optimizer.step()
        
        # Calculate weights as 1/(2*sigma^2)
        with torch.no_grad():
            self.weights = 1.0 / (2 * uncertainties)
            
            # Normalize weights to maintain relative importance
            self.weights = self.weights / self.weights.sum() * self.num_tasks
        
        # Log uncertainties for monitoring
        for name, uncertainty in self.get_uncertainties().items():
            self.uncertainty_history[name].append(uncertainty)
        
        self.log_losses_and_weights(losses)
        
        return self.get_weights()


class DWAWeighter(DynamicLossWeighter):
    """
    Dynamic Weight Averaging (DWA)
    
    Reference: Liu et al. "End-to-End Multi-Task Learning with Attention" CVPR 2019
    """
    
    def __init__(self, 
                 task_names: List[str],
                 temperature: float = 2.0,
                 moving_average_window: int = 2,
                 initial_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize DWA weighter.
        
        Args:
            task_names: List of task names
            temperature: Temperature parameter for softmax (higher = more uniform)
            moving_average_window: Window size for moving average of loss ratios
            initial_weights: Initial task weights
            device: Device for computations
        """
        super().__init__(task_names, initial_weights, device, **kwargs)
        
        self.temperature = temperature
        self.moving_average_window = moving_average_window
        
        # Store recent losses for moving average
        self.recent_losses = {name: [] for name in task_names}
        
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      model: nn.Module,
                      **kwargs) -> Dict[str, float]:
        """
        Update weights using Dynamic Weight Averaging.
        
        DWA calculates weights based on the rate of loss decrease:
        w_i(t) = exp(r_i(t-1) / T) / sum_j exp(r_j(t-1) / T)
        where r_i(t-1) = L_i(t-1) / L_i(t-2) is the loss ratio
        """
        # Store current losses
        current_losses = {}
        for task_name in self.task_names:
            if task_name in losses:
                loss_value = float(losses[task_name].detach().cpu())
                self.recent_losses[task_name].append(loss_value)
                current_losses[task_name] = loss_value
                
                # Keep only recent losses
                if len(self.recent_losses[task_name]) > self.moving_average_window + 1:
                    self.recent_losses[task_name].pop(0)
        
        # Calculate weights if we have enough history
        if all(len(self.recent_losses[name]) >= 2 for name in self.task_names):
            loss_ratios = torch.zeros(self.num_tasks, device=self.device)
            
            for i, task_name in enumerate(self.task_names):
                recent = self.recent_losses[task_name]
                if len(recent) >= 2:
                    # Calculate loss ratio (current / previous)
                    ratio = recent[-1] / (recent[-2] + 1e-8)  # Add epsilon for numerical stability
                    loss_ratios[i] = ratio
                else:
                    loss_ratios[i] = 1.0
            
            # Apply temperature scaling and softmax
            weights_logits = loss_ratios / self.temperature
            weights_softmax = F.softmax(weights_logits, dim=0)
            
            # Scale to maintain the original weight magnitude
            self.weights = weights_softmax * self.num_tasks
        
        self.log_losses_and_weights(losses)
        
        return self.get_weights()


class YOTOWeighter(DynamicLossWeighter):
    """
    YOTO: You Only Train Once - Learnable Loss Weights
    
    Reference: Sakaridis "You Only Train Once" arXiv 2025
    
    This strategy treats loss weights as learnable parameters optimized via gradient descent.
    """
    
    def __init__(self, 
                 task_names: List[str],
                 learning_rate: float = 1e-3,
                 regularization_weight: float = 0.1,
                 uniformity_prior_weight: float = 0.01,
                 initial_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize YOTO weighter.
        
        Args:
            task_names: List of task names
            learning_rate: Learning rate for weight updates
            regularization_weight: L2 regularization on weights
            uniformity_prior_weight: Weight for uniformity prior (encourages balanced weights)
            initial_weights: Initial task weights
            device: Device for computations
        """
        super().__init__(task_names, initial_weights, device, **kwargs)
        
        # Make weights learnable parameters with softmax normalization
        self.raw_weights = nn.Parameter(
            torch.log(self.weights + 1e-8)  # Log to ensure positivity after softmax
        )
        
        self.weight_optimizer = torch.optim.Adam([self.raw_weights], lr=learning_rate)
        self.regularization_weight = regularization_weight
        self.uniformity_prior_weight = uniformity_prior_weight
        
    def _compute_weights(self) -> torch.Tensor:
        """Compute normalized weights using softmax."""
        # Apply softmax to ensure positivity and normalization
        weights = F.softmax(self.raw_weights, dim=0)
        # Scale to maintain original magnitude
        return weights * self.num_tasks
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      model: nn.Module,
                      **kwargs) -> Dict[str, float]:
        """
        Update weights using learnable weight optimization.
        
        The total loss is: L_total = sum_i w_i * L_i + regularization
        where w_i are learnable parameters.
        """
        # Compute current weights
        current_weights = self._compute_weights()
        
        # Calculate weighted loss
        total_loss = 0.0
        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                total_loss += current_weights[i] * losses[task_name]
        
        # Add regularization terms
        regularization_loss = 0.0
        
        # L2 regularization on raw weights
        if self.regularization_weight > 0:
            regularization_loss += self.regularization_weight * torch.sum(self.raw_weights ** 2)
        
        # Uniformity prior (encourage balanced weights)
        if self.uniformity_prior_weight > 0:
            target_weight = torch.ones_like(current_weights) / self.num_tasks
            uniformity_loss = F.mse_loss(current_weights / self.num_tasks, target_weight)
            regularization_loss += self.uniformity_prior_weight * uniformity_loss
        
        total_loss += regularization_loss
        
        # Update weight parameters
        self.weight_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.weight_optimizer.step()
        
        # Update stored weights
        with torch.no_grad():
            self.weights = current_weights
        
        self.log_losses_and_weights(losses)
        
        return self.get_weights()


class AdaptiveCurriculumWeighter(DynamicLossWeighter):
    """
    Adaptive Curriculum Learning for Loss Weighting
    
    This strategy adapts weights based on loss improvement rates and convergence detection.
    Tasks that are improving slowly get higher weights, while converged tasks get lower weights.
    """
    
    def __init__(self, 
                 task_names: List[str],
                 adaptation_rate: float = 0.1,
                 convergence_threshold: float = 1e-4,
                 patience: int = 5,
                 min_weight: float = 0.1,
                 max_weight: float = 5.0,
                 initial_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize Adaptive Curriculum weighter.
        
        Args:
            task_names: List of task names
            adaptation_rate: How quickly to adapt weights
            convergence_threshold: Threshold for detecting convergence
            patience: Number of steps to wait before declaring convergence
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            initial_weights: Initial task weights
            device: Device for computations
        """
        super().__init__(task_names, initial_weights, device, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Track loss improvement rates
        self.loss_improvements = {name: [] for name in task_names}
        self.convergence_counters = {name: 0 for name in task_names}
        
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      model: nn.Module,
                      **kwargs) -> Dict[str, float]:
        """
        Update weights based on loss improvement rates and convergence detection.
        """
        # Calculate loss improvements
        current_losses = {}
        improvements = {}
        
        for task_name in self.task_names:
            if task_name in losses:
                current_loss = float(losses[task_name].detach().cpu())
                current_losses[task_name] = current_loss
                
                if len(self.loss_history[task_name]) > 0:
                    prev_loss = self.loss_history[task_name][-1]
                    improvement = prev_loss - current_loss
                    self.loss_improvements[task_name].append(improvement)
                    
                    # Keep only recent improvements
                    if len(self.loss_improvements[task_name]) > 10:
                        self.loss_improvements[task_name].pop(0)
                    
                    improvements[task_name] = improvement
                    
                    # Check for convergence
                    if abs(improvement) < self.convergence_threshold:
                        self.convergence_counters[task_name] += 1
                    else:
                        self.convergence_counters[task_name] = 0
        
        # Adapt weights based on improvements
        if improvements:
            new_weights = torch.zeros_like(self.weights)
            
            for i, task_name in enumerate(self.task_names):
                if task_name in improvements:
                    # Calculate average improvement rate
                    avg_improvement = np.mean(self.loss_improvements[task_name][-5:])  # Last 5 steps
                    
                    # Check if task has converged
                    has_converged = self.convergence_counters[task_name] >= self.patience
                    
                    if has_converged:
                        # Reduce weight for converged tasks
                        target_weight = self.min_weight
                    else:
                        # Increase weight for tasks with slow improvement
                        if avg_improvement <= 0:  # No improvement or getting worse
                            target_weight = min(self.weights[i] * (1 + self.adaptation_rate), self.max_weight)
                        else:
                            # Decrease weight for tasks improving well
                            target_weight = max(self.weights[i] * (1 - self.adaptation_rate * 0.5), self.min_weight)
                    
                    new_weights[i] = target_weight
                else:
                    new_weights[i] = self.weights[i]
            
            # Smooth transition to new weights
            self.weights = 0.9 * self.weights + 0.1 * new_weights
            
            # Normalize weights
            self.weights = self.weights / self.weights.sum() * self.num_tasks
        
        self.log_losses_and_weights(losses)
        
        return self.get_weights()


def create_dynamic_loss_weighter(strategy: str, 
                                task_names: List[str], 
                                config: dict,
                                device: str = 'cuda') -> DynamicLossWeighter:
    """
    Factory function to create dynamic loss weighters.
    
    Args:
        strategy: Strategy name ('gradnorm', 'uncertainty', 'dwa', 'yoto', 'adaptive_curriculum')
        task_names: List of task names
        config: Configuration dictionary
        device: Device for computations
        
    Returns:
        Configured DynamicLossWeighter instance
    """
    strategy = strategy.lower()
    
    if strategy == 'gradnorm':
        return GradNormWeighter(task_names=task_names, device=device, **config)
    elif strategy == 'uncertainty':
        return UncertaintyWeighter(task_names=task_names, device=device, **config)
    elif strategy == 'dwa':
        return DWAWeighter(task_names=task_names, device=device, **config)
    elif strategy == 'yoto':
        return YOTOWeighter(task_names=task_names, device=device, **config)
    elif strategy == 'adaptive_curriculum':
        return AdaptiveCurriculumWeighter(task_names=task_names, device=device, **config)
    else:
        raise ValueError(f"Unknown dynamic loss weighting strategy: {strategy}")


# Utility functions for monitoring and analysis

def plot_weight_history(weighter: DynamicLossWeighter, save_path: Optional[str] = None):
    """Plot the evolution of loss weights over training."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot weight history
        plt.subplot(2, 1, 1)
        for task_name in weighter.task_names:
            if task_name in weighter.weight_history:
                plt.plot(weighter.weight_history[task_name], label=f'{task_name}_weight')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Weight')
        plt.title('Dynamic Loss Weight Evolution')
        plt.legend()
        plt.grid(True)
        
        # Plot loss history
        plt.subplot(2, 1, 2)
        for task_name in weighter.task_names:
            if task_name in weighter.loss_history:
                plt.plot(weighter.loss_history[task_name], label=f'{task_name}_loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.title('Loss Evolution')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")


def analyze_weight_statistics(weighter: DynamicLossWeighter) -> Dict[str, Dict[str, float]]:
    """Analyze statistics of weight evolution."""
    stats = {}
    
    for task_name in weighter.task_names:
        if task_name in weighter.weight_history and len(weighter.weight_history[task_name]) > 0:
            weights = np.array(weighter.weight_history[task_name])
            stats[task_name] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'final': float(weights[-1]),
                'range': float(np.max(weights) - np.min(weights))
            }
    
    return stats 