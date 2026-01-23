"""
ROMWithE2C training wrapper
Handles model initialization, training loop, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model.models.mse2c import MSE2C
from model.losses.customized_loss import CustomizedLoss


class ROMWithE2C(nn.Module):
    def __init__(self, config):
        super(ROMWithE2C, self).__init__()
        self.config = config
        
        # Select model type from config
        if config.model['method'] == 'E2C':
            self.model = MSE2C(config)
        else:
            raise ValueError(f"Unknown method: {config.model['method']}. Only 'E2C' method is supported.")
        
        # Initialize loss function with config
        self.loss_object = CustomizedLoss(config)
        
        # Set model reference for dynamic loss weighting
        self.loss_object.setModelReference(self.model)
        
        # Initialize optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training['learning_rate'])
        
        # Initialize discriminator optimizer if adversarial training is enabled
        if config.loss.get('enable_adversarial_loss', False) and self.model.discriminator:
            self.discriminator_optimizer = optim.Adam(
                self.model.discriminator.parameters(), 
                lr=config.discriminator['learning_rate']
            )
        else:
            self.discriminator_optimizer = None

        # Initialize learning rate schedulers
        self._setup_schedulers(config)

        self.train_loss = torch.tensor(0.0)
        self.train_reconstruction_loss = torch.tensor(0.0)
        self.train_flux_loss = torch.tensor(0.0)
        self.train_well_loss = torch.tensor(0.0)
        self.train_transition_loss = torch.tensor(0.0)
        self.train_observation_loss = torch.tensor(0.0)
        self.train_non_negative_loss = torch.tensor(0.0)
        self.test_loss = torch.tensor(0.0)
        self.test_reconstruction_loss = torch.tensor(0.0)
        self.test_flux_loss = torch.tensor(0.0)
        self.test_well_loss = torch.tensor(0.0)
        self.test_transition_loss = torch.tensor(0.0)
        self.test_observation_loss = torch.tensor(0.0)
        self.test_non_negative_loss = torch.tensor(0.0)
    
    def _setup_schedulers(self, config):
        """Setup learning rate schedulers based on configuration."""
        self.scheduler = None
        self.discriminator_scheduler = None
        self.scheduler_info = config.get_scheduler_info()
        
        try:
            # Validate scheduler configuration
            validation_issues = config.validate_scheduler_config()
            if validation_issues:
                print("⚠️ Scheduler configuration issues found:")
                for issue in validation_issues:
                    print(f"   • {issue}")
                print("⚠️ Falling back to fixed learning rate")
                return
                
            # Create main optimizer scheduler
            self.scheduler = config.create_scheduler(self.optimizer)
            
            # Create discriminator scheduler if needed
            if (self.discriminator_optimizer and 
                config.config.get('learning_rate_scheduler', {}).get('discriminator', {}).get('enable', False)):
                
                # Use same config as main scheduler for discriminator (can be customized later)
                self.discriminator_scheduler = config.create_scheduler(self.discriminator_optimizer)
                if config.runtime.get('verbose', True):
                    print("📅 Discriminator scheduler created")
            
            # Store scheduler type for logging
            if self.scheduler:
                scheduler_type = config.get_scheduler_info()['type']
                if config.runtime.get('verbose', True):
                    print(f"✅ Learning rate scheduler initialized: {scheduler_type}")
            
        except Exception as e:
            print(f"❌ Error setting up schedulers: {e}")
            print("⚠️ Falling back to fixed learning rate")
            self.scheduler = None
            self.discriminator_scheduler = None
    
    def setup_schedulers_with_steps(self, num_training_steps: int):
        """
        Setup schedulers that require total training steps (OneCycleLR).
        Call this after you know the total number of training steps.
        
        Args:
            num_training_steps: Total number of training steps
        """
        scheduler_config = self.config.config.get('learning_rate_scheduler', {})
        if not scheduler_config.get('enable', False):
            return
            
        scheduler_type = scheduler_config.get('type', 'fixed')
        
        if scheduler_type == 'one_cycle':
            try:
                if self.config.runtime.get('verbose', True):
                    print(f"📅 Reconfiguring OneCycleLR with {num_training_steps} total steps")
                    
                self.scheduler = self.config.create_scheduler(self.optimizer, num_training_steps)
                
                if (self.discriminator_optimizer and 
                    scheduler_config.get('discriminator', {}).get('enable', False)):
                    self.discriminator_scheduler = self.config.create_scheduler(
                        self.discriminator_optimizer, num_training_steps
                    )
                    
                if self.config.runtime.get('verbose', True):
                    print("✅ OneCycleLR scheduler reconfigured with total steps")
                    
            except Exception as e:
                print(f"❌ Error reconfiguring OneCycleLR: {e}")
                print("⚠️ Falling back to fixed learning rate")
                self.scheduler = None
                self.discriminator_scheduler = None

    def predict(self, inputs):
        """
        Single step prediction for 3D reservoir states
        
        Args:
            inputs: (xt, ut, yt, dt) where xt has shape (batch, N, Nx, My, Nz)
            
        Returns:
            xt1_pred: Predicted next state (batch, N, Nx, My, Nz)
            yt1_pred: Predicted well outputs (batch, 2*num_prob + num_inj)
        """
        self.model.eval()
        with torch.no_grad():
            # Updated for 3D CNN: inputs now expect 3D reservoir data
            # Original: xt, ut, yt, dt, perm = inputs
            xt, ut, yt, dt = inputs
            xt1_pred, yt1_pred = self.model.predict(inputs)        
        return xt1_pred, yt1_pred
    
    def predict_latent(self, zt, dt, ut):
        """
        Latent space prediction for efficient multi-step forecasting
        
        Args:
            zt: Current latent state (batch, latent_dim)
            dt: Time step (batch, 1)
            ut: Control inputs (batch, u_dim)
            
        Returns:
            zt_next: Next latent state (batch, latent_dim)
            yt_next: Predicted well outputs (batch, 2*num_prob + num_inj)
        """
        self.model.eval()
        with torch.no_grad():
            zt_next, yt_next = self.model.predict_latent(zt, dt, ut)  

        return zt_next, yt_next

    def evaluate(self, inputs):
        """
        Evaluate model performance on validation/test data
        
        Args:
            inputs: (X, U, Y, dt) where X contains 3D reservoir states
                   X: List of states with shape (batch, 3, 64, 64, 4)
                   U: List of controls (batch, u_dim)
                   Y: List of observations (batch, 2*num_prob + num_inj)
                   dt: Time step (batch, 1)
        """
        self.model.eval()
        with torch.no_grad():
            # Updated for 3D CNN: removed perm-related comments
            # Original: xt1 = labels.float()
            # Original: xt, ut, dt, _, _ = inputs
            
            predictions = self.model(inputs)
            # Parse predictions - consistent with 3D CNN implementation
            # Original: X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec, perm = predictions
            X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec = predictions
            
            # Calculate loss using updated prediction format
            # Original: y_pred = (xt1_pred, zt1_pred, zt1, zt, xt_rec, xt, perm, prod_loc)
            t_loss = self.loss_object(predictions)

            self.test_loss = t_loss
            # Store individual evaluation loss components
            self.test_reconstruction_loss = self.loss_object.getReconstructionLoss()
            self.test_flux_loss = self.loss_object.getFluxLoss()
            self.test_well_loss = self.loss_object.getWellLoss()
            self.test_transition_loss = self.loss_object.getTransitionLoss()
            self.test_observation_loss = self.loss_object.getObservationLoss()
            self.test_non_negative_loss = self.loss_object.getNonNegativeLoss()

    def update(self, inputs):
        """
        Training step with backpropagation for 3D CNN model with spatial enhancements
        
        Args:
            inputs: (X, U, Y, dt) where X contains 3D reservoir states
                   X: List of states with shape (batch, 3, 34, 16, 25)
                   U: List of controls (batch, u_dim)
                   Y: List of observations (batch, 2*num_prob + num_inj)
                   dt: Time step (batch, 1)
        """
        self.model.train()
        X, U, Y, dt = inputs
        
        # ===== GENERATOR UPDATE =====
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec = predictions
        
        # Get discriminator prediction for adversarial loss if enabled
        discriminator_pred = None
        if self.model.discriminator and self.config.loss.get('enable_adversarial_loss', False):
            # Combine all predicted reconstructions for discriminator
            all_pred_states = [x0_rec] + X_next_pred
            combined_pred = torch.cat(all_pred_states, dim=0)
            discriminator_pred = self.model.discriminator(combined_pred)
        
        # Calculate generator loss with spatial enhancements
        loss = self.loss_object(predictions, discriminator_pred)
        loss.backward()
        self.optimizer.step()

        # ===== DISCRIMINATOR UPDATE =====
        if self.discriminator_optimizer and self.model.discriminator:
            self._update_discriminator(X, X_next_pred, x0_rec)

        # Store losses
        self.train_loss = loss
        self.train_flux_loss = self.loss_object.getFluxLoss()
        self.train_reconstruction_loss = self.loss_object.getReconstructionLoss()
        self.train_well_loss = self.loss_object.getWellLoss()
        self.train_transition_loss = self.loss_object.getTransitionLoss()
        self.train_observation_loss = self.loss_object.getObservationLoss()
        self.train_non_negative_loss = self.loss_object.getNonNegativeLoss()
        
        # Step schedulers on batch if needed (for CyclicLR, OneCycleLR)
        self._step_scheduler_on_batch()

    def _step_scheduler_on_batch(self):
        """Step scheduler on each batch for schedulers that require it."""
        if self.scheduler and self.scheduler_info.get('step_on_batch', False):
            try:
                self.scheduler.step()
                if self.discriminator_scheduler:
                    self.discriminator_scheduler.step()
            except Exception as e:
                if self.config.runtime.get('verbose', True):
                    print(f"⚠️ Error stepping scheduler on batch: {e}")
    
    def step_scheduler_on_epoch(self, validation_loss: float = None):
        """
        Step scheduler on epoch end with robust error handling.
        
        Args:
            validation_loss: Validation loss for ReduceLROnPlateau scheduler
        """
        # Update dynamic loss weighter epoch counter
        self.loss_object.epochDynamicWeighter()
        
        if not self.scheduler:
            return
            
        scheduler_type = self.scheduler_info.get('type', 'fixed')
        
        try:
            # Ensure validation_loss is proper type for ReduceLROnPlateau
            if scheduler_type == 'reduce_on_plateau':
                if validation_loss is not None:
                    # Convert validation_loss to float to prevent type errors
                    try:
                        val_loss_float = float(validation_loss)
                        self.scheduler.step(val_loss_float)
                        if self.discriminator_scheduler:
                            self.discriminator_scheduler.step(val_loss_float)
                    except (ValueError, TypeError) as ve:
                        print(f"⚠️ Invalid validation_loss type '{type(validation_loss)}': {validation_loss}")
                        print(f"   ReduceLROnPlateau requires numeric validation loss, skipping scheduler step")
                        return
                else:
                    if self.config.runtime.get('verbose', True):
                        print("⚠️ ReduceLROnPlateau scheduler requires validation_loss but none provided")
                    return
            
            # Other schedulers step without arguments (unless they step on batch)
            elif not self.scheduler_info.get('step_on_batch', False):
                self.scheduler.step()
                if self.discriminator_scheduler:
                    self.discriminator_scheduler.step()
                        
        except Exception as e:
            if self.config.runtime.get('verbose', True):
                print(f"⚠️ Error stepping scheduler on epoch: {e}")
                print(f"   Scheduler type: {scheduler_type}")
                print(f"   Validation loss: {validation_loss} (type: {type(validation_loss)})")
                print(f"   Error type: {type(e).__name__}")
                
                # Try to provide more helpful debugging information
                if hasattr(self.scheduler, 'state_dict'):
                    try:
                        scheduler_state = self.scheduler.state_dict()
                        print(f"   Scheduler state keys: {list(scheduler_state.keys())}")
                    except:
                        pass
                
                print(f"   Disabling scheduler to prevent further errors")
                # Disable the problematic scheduler
                self.scheduler = None
                self.discriminator_scheduler = None
    
    def get_current_lr(self) -> dict:
        """
        Get current learning rate information for logging.
        
        Returns:
            Dictionary with learning rate information
        """
        lr_info = {}
        
        try:
            # Get main optimizer learning rate
            if self.optimizer:
                main_lr = self.optimizer.param_groups[0]['lr']
                lr_info['generator_lr'] = main_lr
                
                # Get momentum if available
                if 'momentum' in self.optimizer.param_groups[0]:
                    lr_info['momentum'] = self.optimizer.param_groups[0]['momentum']
                elif 'betas' in self.optimizer.param_groups[0]:  # Adam optimizer
                    lr_info['momentum'] = self.optimizer.param_groups[0]['betas'][0]
            
            # Get discriminator learning rate
            if self.discriminator_optimizer:
                discriminator_lr = self.discriminator_optimizer.param_groups[0]['lr']
                lr_info['discriminator_lr'] = discriminator_lr
            
            # Add scheduler-specific information
            if self.scheduler:
                scheduler_info = {'scheduler_type': self.scheduler_info.get('type', 'unknown')}
                
                # Add ReduceLROnPlateau specific info
                if hasattr(self.scheduler, 'num_bad_epochs'):
                    scheduler_info['num_bad_epochs'] = self.scheduler.num_bad_epochs
                if hasattr(self.scheduler, 'patience'):
                    scheduler_info['patience'] = self.scheduler.patience
                
                lr_info['scheduler_info'] = scheduler_info
                
        except Exception as e:
            if self.config.runtime.get('verbose', True):
                print(f"⚠️ Error getting learning rate info: {e}")
        
        return lr_info
    
    def log_scheduler_event(self, event_type: str, details: dict, wandb_logger=None, step: int = 0):
        """
        Log scheduler events (like learning rate reductions).
        
        Args:
            event_type: Type of event ('lr_reduction', 'cycle_complete', etc.)
            details: Event details dictionary
            wandb_logger: WandB logger instance
            step: Current training step
        """
        if self.config.runtime.get('verbose', True):
            print(f"📅 Scheduler Event: {event_type}")
            for key, value in details.items():
                print(f"    {key}: {value}")
        
        # Log to WandB if available
        if wandb_logger and hasattr(wandb_logger, 'log_scheduler_event'):
            wandb_logger.log_scheduler_event(event_type, details, step)
        
    def _update_discriminator(self, X_true, X_next_pred, x0_rec):
        """Update discriminator for adversarial training"""
        self.discriminator_optimizer.zero_grad()
        
        # Real data
        all_true_states = X_true  # List of true states
        combined_true = torch.cat(all_true_states, dim=0)
        real_pred = self.model.discriminator(combined_true)
        
        # Fake data  
        all_fake_states = [x0_rec] + X_next_pred
        combined_fake = torch.cat(all_fake_states, dim=0)
        fake_pred = self.model.discriminator(combined_fake.detach())  # Detach to avoid generator gradients
        
        # Discriminator loss
        if self.config.loss['adversarial_loss_type'] == 'gan':
            # Standard GAN discriminator loss
            real_loss = -torch.mean(torch.log(torch.sigmoid(real_pred) + 1e-8))
            fake_loss = -torch.mean(torch.log(1 - torch.sigmoid(fake_pred) + 1e-8))
        elif self.config.loss['adversarial_loss_type'] == 'lsgan':
            # Least squares GAN discriminator loss  
            label_smoothing = self.config.discriminator.get('label_smoothing', 0.1)
            real_loss = torch.mean((real_pred - (1 - label_smoothing)) ** 2)
            fake_loss = torch.mean((fake_pred - 0) ** 2)
        elif self.config.loss['adversarial_loss_type'] == 'wgan':
            # Wasserstein GAN discriminator loss
            real_loss = -torch.mean(real_pred)
            fake_loss = torch.mean(fake_pred)
        
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def get_train_loss(self):
        return self.train_loss.item()

    def get_train_flux_loss(self):
        # Handle case where flux_loss might be an integer (0) instead of tensor
        if isinstance(self.train_flux_loss, torch.Tensor):
            return self.train_flux_loss.item()
        else:
            return float(self.train_flux_loss)

    def get_train_reconstruction_loss(self):
        return self.train_reconstruction_loss.item()

    def get_train_well_loss(self):
        # Handle case where well_loss might be an integer (0) instead of tensor
        if isinstance(self.train_well_loss, torch.Tensor):
            return self.train_well_loss.item()
        else:
            return float(self.train_well_loss)

    def get_test_loss(self):
        return self.test_loss.item()
    
    def get_test_reconstruction_loss(self):
        # Handle case where test_reconstruction_loss might be an integer (0) instead of tensor
        if isinstance(self.test_reconstruction_loss, torch.Tensor):
            return self.test_reconstruction_loss.item()
        else:
            return float(self.test_reconstruction_loss)
    
    def get_test_flux_loss(self):
        # Handle case where test_flux_loss might be an integer (0) instead of tensor
        if isinstance(self.test_flux_loss, torch.Tensor):
            return self.test_flux_loss.item()
        else:
            return float(self.test_flux_loss)
    
    def get_test_well_loss(self):
        # Handle case where test_well_loss might be an integer (0) instead of tensor
        if isinstance(self.test_well_loss, torch.Tensor):
            return self.test_well_loss.item()
        else:
            return float(self.test_well_loss)
    
    def get_train_transition_loss(self):
        # Handle case where train_transition_loss might be an integer (0) instead of tensor
        if isinstance(self.train_transition_loss, torch.Tensor):
            return self.train_transition_loss.item()
        else:
            return float(self.train_transition_loss)
    
    def get_train_observation_loss(self):
        # Handle case where train_observation_loss might be an integer (0) instead of tensor
        if isinstance(self.train_observation_loss, torch.Tensor):
            return self.train_observation_loss.item()
        else:
            return float(self.train_observation_loss)
    
    def get_test_transition_loss(self):
        # Handle case where test_transition_loss might be an integer (0) instead of tensor
        if isinstance(self.test_transition_loss, torch.Tensor):
            return self.test_transition_loss.item()
        else:
            return float(self.test_transition_loss)
    
    def get_test_observation_loss(self):
        # Handle case where test_observation_loss might be an integer (0) instead of tensor
        if isinstance(self.test_observation_loss, torch.Tensor):
            return self.test_observation_loss.item()
        else:
            return float(self.test_observation_loss)
    
    def get_train_non_negative_loss(self):
        # Handle case where train_non_negative_loss might be an integer (0) instead of tensor
        if isinstance(self.train_non_negative_loss, torch.Tensor):
            return self.train_non_negative_loss.item()
        else:
            return float(self.train_non_negative_loss)
    
    def get_test_non_negative_loss(self):
        # Handle case where test_non_negative_loss might be an integer (0) instead of tensor
        if isinstance(self.test_non_negative_loss, torch.Tensor):
            return self.test_non_negative_loss.item()
        else:
            return float(self.test_non_negative_loss)
    
    def save_model_weights(self, encoder_file, decoder_file, transition_file):
        """
        Save model weights to files
        
        Args:
            encoder_file: Path to save encoder weights
            decoder_file: Path to save decoder weights  
            transition_file: Path to save transition model weights
        """
        self.model.save_weights_to_file(encoder_file, decoder_file, transition_file)
    
    def load_model_weights(self, encoder_file, decoder_file, transition_file):
        """
        Load model weights from files
        
        Args:
            encoder_file: Path to encoder weights
            decoder_file: Path to decoder weights
            transition_file: Path to transition model weights
        """
        self.model.load_weights_from_file(encoder_file, decoder_file, transition_file)
    
    def find_and_load_weights(self, models_dir='./saved_models/', base_pattern='e2co', specific_pattern=None):
        """
        Automatically find and load the latest model weights
        
        Args:
            models_dir: Directory containing model files
            base_pattern: Base prefix of model files
            specific_pattern: Optional pattern to further filter model files
            
        Returns:
            Dictionary with paths of loaded model files
        """
        return self.model.find_and_load_weights(models_dir, base_pattern, specific_pattern)

    def smart_load_compatible_weights(self, models_dir='./saved_models/', base_pattern='e2co', verbose=True):
        """
        Intelligently find and load compatible model weights with detailed compatibility analysis.
        
        This method wraps the MSE2C smart loading functionality and provides additional
        compatibility checks for normalization settings and training configurations.
        
        Args:
            models_dir: Directory containing model files
            base_pattern: Base prefix of model files
            verbose: Print detailed loading information
            
        Returns:
            Dictionary with loaded file paths and compatibility information
        """
        # Delegate to the underlying model's smart loading
        result = self.model.smart_load_compatible_weights(models_dir, base_pattern, verbose)
        
        if verbose:
            print("✅ ROMWithE2C: Model loading completed")
            
        return result
    
    def auto_load_weights(self, models_dir='./saved_models/', base_pattern='e2co', fallback_to_smart=True, verbose=True):
        """
        Convenience method that attempts exact loading first, then falls back to smart loading.
        
        This method first tries to load weights using the exact current configuration.
        If that fails (FileNotFoundError), it automatically falls back to smart loading
        to find the most compatible available model.
        
        Args:
            models_dir: Directory containing model files
            base_pattern: Base prefix of model files
            fallback_to_smart: Whether to use smart loading if exact loading fails
            verbose: Print detailed information
            
        Returns:
            Dictionary with loading results and method used
        """
        if verbose:
            print("🤖 Auto Model Loading: Attempting exact configuration match first...")
            
        try:
            # Try exact loading first using find_and_load_weights
            result = self.model.find_and_load_weights(models_dir, base_pattern)
            
            if verbose:
                print("✅ Exact configuration match found and loaded successfully!")
                
            return {
                'method': 'exact',
                'success': True,
                'files': result,
                'message': 'Exact configuration match loaded'
            }
            
        except FileNotFoundError as e:
            if verbose:
                print(f"⚠️ Exact match not found: {e}")
                
            if fallback_to_smart:
                if verbose:
                    print("🧠 Falling back to smart compatible loading...")
                    
                try:
                    result = self.smart_load_compatible_weights(models_dir, base_pattern, verbose)
                    
                    return {
                        'method': 'smart',
                        'success': True,
                        'files': result,
                        'message': 'Compatible model loaded with smart matching'
                    }
                    
                except Exception as smart_error:
                    if verbose:
                        print(f"❌ Smart loading also failed: {smart_error}")
                    raise smart_error
            else:
                raise e
        except Exception as e:
            if verbose:
                print(f"❌ Model loading failed: {e}")
            raise e