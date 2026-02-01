#!/usr/bin/env python3
"""
Weights & Biases Integration for E2C Reservoir Model
===================================================
Simple and effective W&B integration for E2C training.

Author: AI Assistant  
Date: January 2025
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional

class WandBLogger:
    """Simple W&B logger for E2C model training"""
    
    def __init__(self, config):
        """Initialize W&B logger"""
        # Check if W&B is enabled in config
        wandb_config = config.runtime.get('wandb', {})
        self.enabled = wandb_config.get('enable', False)
        
        if not self.enabled:
            print("📝 W&B logging disabled")
            return
        
        try:
            # Initialize W&B
            self.run = wandb.init(
                project=wandb_config.get('project', 'E2C-Reservoir-Model'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('name', None),
                tags=wandb_config.get('tags', ['E2C', 'reservoir-modeling']),
                notes=wandb_config.get('notes', 'E2C model training'),
                config=self._extract_config(config)
            )
            
            # Store settings
            self.log_frequency = wandb_config.get('log_frequency', 100)
            self.best_loss = float('inf')
            self.step_count = 0
            
            print(f"🌟 W&B initialized! Dashboard: {self.run.url}")
            
        except Exception as e:
            print(f"⚠️ W&B init failed: {e}")
            self.enabled = False
    
    def _extract_config(self, config) -> Dict[str, Any]:
        """Extract config for W&B"""
        return {
            # Model
            'method': config.model['method'],
            'latent_dim': config.model['latent_dim'],
            'n_channels': config.model['n_channels'],
            
            # Training
            'epochs': config.training['epoch'],
            'batch_size': config.training['batch_size'],
            'learning_rate': config.training['learning_rate'],
            'nsteps': config.training['nsteps'],
            
            # Loss weights
            'lambda_reconstruction': config.loss['lambda_reconstruction_loss'],
            'lambda_transition': config.loss['lambda_trans_loss'],
            'lambda_observation': config.loss['lambda_yobs_loss'],
            
            # Physics losses
            'enable_flux_loss': config.loss['enable_flux_loss'],
            'enable_bhp_loss': config.loss['enable_bhp_loss'],
        }
    
    def watch_model(self, model):
        """Watch model for gradients"""
        if not self.enabled:
            return
        try:
            wandb.watch(model, log='all', log_freq=self.log_frequency)
            print("👁️ W&B watching model")
        except Exception as e:
            print(f"⚠️ Failed to watch model: {e}")
    
    def log_training_step(self, my_rom, epoch: int, batch: int, step: int):
        """Log training metrics"""
        if not self.enabled:
            return
        
        try:
            metrics = {
                'epoch': epoch,
                'batch': batch,
                'train/total_loss': self._get_value(my_rom.get_train_loss()),
                'train/reconstruction_loss': self._get_value(my_rom.get_train_reconstruction_loss()),
                'train/transition_loss': self._get_value(my_rom.get_train_transition_loss()),
                'train/observation_loss': self._get_value(my_rom.get_train_observation_loss()),
            }
            
            # Add physics losses if available
            flux_loss = self._get_value(my_rom.get_train_flux_loss())
            well_loss = self._get_value(my_rom.get_train_well_loss())
            
            if flux_loss and flux_loss > 0:
                metrics['train/flux_loss'] = flux_loss
            if well_loss and well_loss > 0:
                metrics['train/well_loss'] = well_loss
            
            # Add VAE KL loss if available
            try:
                kl_loss = self._get_value(my_rom.get_train_kl_loss())
                if kl_loss and kl_loss > 0:
                    metrics['train/kl_loss'] = kl_loss
            except Exception:
                pass  # KL loss not available (VAE not enabled)
            
            # Add dynamic loss weights if available
            try:
                dynamic_weights = my_rom.loss_object.getDynamicWeights()
                if dynamic_weights:
                    for task_name, weight in dynamic_weights.items():
                        metrics[f'weights/{task_name}'] = self._get_value(weight)
            except Exception as e:
                pass  # Dynamic weights not available
            
            # Add learning rate information
            if hasattr(my_rom, 'get_current_lr'):
                lr_info = my_rom.get_current_lr()
                if lr_info:
                    metrics.update(self._format_lr_metrics(lr_info))
            
            wandb.log(metrics, step=step)
            
        except Exception as e:
            print(f"⚠️ Failed to log training: {e}")
    
    def _format_lr_metrics(self, lr_info: dict) -> dict:
        """Format learning rate information for logging"""
        formatted_metrics = {}
        
        # Add generator/main optimizer learning rate
        if 'generator_lr' in lr_info:
            formatted_metrics['train/learning_rate'] = lr_info['generator_lr']
        
        # Add discriminator learning rate if available
        if 'discriminator_lr' in lr_info:
            formatted_metrics['train/discriminator_lr'] = lr_info['discriminator_lr']
        
        # Add momentum information if available
        if 'momentum' in lr_info:
            formatted_metrics['train/momentum'] = lr_info['momentum']
        
        # Add scheduler-specific information
        if 'scheduler_info' in lr_info:
            scheduler_info = lr_info['scheduler_info']
            if 'scheduler_type' in scheduler_info:
                formatted_metrics['train/scheduler_type'] = scheduler_info['scheduler_type']
            if 'num_bad_epochs' in scheduler_info:  # For ReduceLROnPlateau
                formatted_metrics['train/plateau_bad_epochs'] = scheduler_info['num_bad_epochs']
        
        return formatted_metrics
    
    def log_scheduler_event(self, event_type: str, details: dict, step: int):
        """Log scheduler events like learning rate reductions"""
        if not self.enabled:
            return
        
        try:
            metrics = {
                'scheduler_events/type': event_type,
                'scheduler_events/step': step
            }
            
            # Add event-specific details
            for key, value in details.items():
                metrics[f'scheduler_events/{key}'] = self._get_value(value)
            
            wandb.log(metrics, step=step)
            
        except Exception as e:
            print(f"⚠️ Failed to log scheduler event: {e}")
    
    def log_evaluation_step(self, my_rom, epoch: int, step: int):
        """Log evaluation metrics"""
        if not self.enabled:
            return
        
        try:
            metrics = {
                'epoch': epoch,
                'eval/total_loss': self._get_value(my_rom.get_test_loss()),
                'eval/reconstruction_loss': self._get_value(my_rom.get_test_reconstruction_loss()),
                'eval/transition_loss': self._get_value(my_rom.get_test_transition_loss()),
                'eval/observation_loss': self._get_value(my_rom.get_test_observation_loss()),
            }
            
            # Add physics losses
            flux_loss = self._get_value(my_rom.get_test_flux_loss())
            well_loss = self._get_value(my_rom.get_test_well_loss())
            
            if flux_loss and flux_loss > 0:
                metrics['eval/flux_loss'] = flux_loss
            if well_loss and well_loss > 0:
                metrics['eval/well_loss'] = well_loss
            
            # Add VAE KL loss if available
            try:
                kl_loss = self._get_value(my_rom.get_test_kl_loss())
                if kl_loss and kl_loss > 0:
                    metrics['eval/kl_loss'] = kl_loss
            except Exception:
                pass  # KL loss not available (VAE not enabled)
            
            # Add current dynamic weights (evaluation uses same weights as training)
            try:
                dynamic_weights = my_rom.loss_object.getDynamicWeights()
                if dynamic_weights:
                    for task_name, weight in dynamic_weights.items():
                        metrics[f'eval_weights/{task_name}'] = self._get_value(weight)
            except Exception as e:
                pass  # Dynamic weights not available
            
            wandb.log(metrics, step=step)
            
            # Track best model based on observation loss (well prediction accuracy)
            current_observation_loss = self._get_value(my_rom.get_test_observation_loss())
            current_total_loss = self._get_value(my_rom.get_test_loss())
            current_reconstruction_loss = self._get_value(my_rom.get_test_reconstruction_loss())
            current_transition_loss = self._get_value(my_rom.get_test_transition_loss())
            
            if current_observation_loss and current_observation_loss < getattr(self, 'best_observation_loss', float('inf')):
                self.best_observation_loss = current_observation_loss
                wandb.log({
                    'eval/best_observation_loss': self.best_observation_loss,
                    'eval/best_model_total_loss': current_total_loss
                }, step=step)
                # Format losses with fallback for None values
                recon_str = f"{current_reconstruction_loss:.6f}" if current_reconstruction_loss is not None else "N/A"
                trans_str = f"{current_transition_loss:.6f}" if current_transition_loss is not None else "N/A"
                total_str = f"{current_total_loss:.6f}" if current_total_loss is not None else "N/A"
                
                # Get KL loss if available
                try:
                    kl_loss = self._get_value(my_rom.get_test_kl_loss())
                    kl_str = f"{kl_loss:.6f}" if kl_loss is not None and kl_loss > 0 else "N/A"
                except Exception:
                    kl_str = "N/A"
                
                print(f"🏆 New best model! Observation Loss: {self.best_observation_loss:.6f} | "
                      f"Reconstruction Loss: {recon_str} | "
                      f"Transition Loss: {trans_str} | "
                      f"KL Loss: {kl_str} | "
                      f"Total: {total_str}")
            
        except Exception as e:
            print(f"⚠️ Failed to log evaluation: {e}")
    
    def _get_value(self, tensor_or_value):
        """Extract value from tensor or return as-is"""
        try:
            if tensor_or_value is None:
                return None
            if torch.is_tensor(tensor_or_value):
                return float(tensor_or_value.detach().cpu().item())
            return float(tensor_or_value)
        except:
            return None
    
    def finish(self):
        """Finish W&B run"""
        if self.enabled:
            try:
                wandb.finish()
                print("🏁 W&B run finished")
            except:
                pass

def create_wandb_logger(config):
    """Create W&B logger from config"""
    return WandBLogger(config) 