"""
Customized loss function that aggregates multiple loss components
Supports dynamic loss weighting and spatial enhancements
"""

import torch
import torch.nn as nn

from .individual_losses import (
    get_reconstruction_loss,
    get_flux_loss,
    get_well_bhp_loss,
    get_l2_reg_loss,
    get_non_negative_loss
)
from .spatial_enhancements import GradientLoss


class CustomizedLoss(nn.Module):
    def __init__(self, config):
        super(CustomizedLoss, self).__init__()
        self.config = config
        
        # Enable/disable flags for physics losses
        self.enable_flux_loss = config.loss.get('enable_flux_loss', False)
        self.enable_bhp_loss = config.loss.get('enable_bhp_loss', False)
        self.enable_non_negative_loss = config.loss.get('enable_non_negative_loss', False)
        
        # Dynamic loss weighting configuration
        self.enable_dynamic_weighting = config.loss.get('enable_dynamic_weighting', False)
        
        if self.enable_dynamic_weighting:
            # Import dynamic loss weighting module
            try:
                from .dynamic_loss_weighting import create_dynamic_loss_weighter
                
                # Define task names and initial weights
                self.task_names = ['reconstruction', 'physics', 'transition', 'observation']
                if self.enable_flux_loss or self.enable_bhp_loss:
                    # Separate physics losses
                    self.task_names = ['reconstruction', 'flux', 'bhp', 'transition', 'observation']
                
                # Initial static weights as fallback
                initial_weights = {
                    'reconstruction': config['loss'].get('lambda_reconstruction_loss', 1.0),
                    'transition': config['loss']['lambda_trans_loss'],
                    'observation': config['loss']['lambda_yobs_loss']
                }
                
                if 'flux' in self.task_names:
                    initial_weights['flux'] = config['loss']['lambda_flux_loss']
                if 'bhp' in self.task_names:
                    initial_weights['bhp'] = config['loss']['lambda_bhp_loss']
                if 'physics' in self.task_names:
                    initial_weights['physics'] = (config['loss']['lambda_flux_loss'] + 
                                                config['loss']['lambda_bhp_loss']) / 2
                
                # Create dynamic loss weighter
                strategy = config.loss.get('dynamic_weighting_strategy', 'gradnorm')
                weighter_config = config.loss.get('dynamic_weighting_config', {})
                
                device = config.runtime.get('device', 'cuda')
                if device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                self.dynamic_weighter = create_dynamic_loss_weighter(
                    strategy=strategy,
                    task_names=self.task_names,
                    config=weighter_config,
                    device=device
                )
                
                if config.runtime.get('verbose', True):
                    print(f"🔧 Dynamic Loss Weighting: Enabled with {strategy} strategy")
                    print(f"   📊 Task names: {self.task_names}")
                    print(f"   ⚖️ Initial weights: {initial_weights}")
                
            except ImportError as e:
                print(f"⚠️ Warning: Could not import dynamic loss weighting module: {e}")
                print("   Falling back to static loss weights.")
                self.enable_dynamic_weighting = False
        
        # Static loss weights (used when dynamic weighting is disabled)
        self.reconstruction_loss_lambda = config['loss'].get('lambda_reconstruction_loss', 1.0)
        self.flux_loss_lambda = config['loss']['lambda_flux_loss']
        self.bhp_loss_lambda = config['loss']['lambda_bhp_loss']
        self.non_negative_loss_lambda = config['loss'].get('lambda_non_negative_loss', 0.1)
        self.trans_loss_weight = config['loss']['lambda_trans_loss']
        self.yobs_loss_weight = config['loss']['lambda_yobs_loss']
        
        # Reconstruction loss variance parameter (configurable noise assumption)
        self.reconstruction_variance = config['loss'].get('reconstruction_variance', 0.1)
        
        # Per-element loss normalization configuration
        self.enable_per_element_normalization = config['loss'].get('enable_per_element_normalization', False)
        
        # Calculate normalization factors based on system dimensions
        if self.enable_per_element_normalization:
            # Spatial normalization: channels × grid cells
            input_shape = config['data']['input_shape']  # [channels, X, Y, Z]
            self.spatial_normalization_factor = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
            
            # Latent normalization: latent dimensions
            self.latent_normalization_factor = config['model']['latent_dim']
            
            # Observation normalization: num_observations × num_timesteps
            num_prod = config['data']['num_prod']
            num_inj = config['data']['num_inj']
            num_timesteps = config['training']['num_tsteps']  # FIXED: Use full episode length (30) not training steps (2)
            self.observation_normalization_factor = (num_prod * 2 + num_inj) * num_timesteps  # BHP + 2 rates per producer + 1 rate per injector
        else:
            # No normalization - set factors to 1
            self.spatial_normalization_factor = 1
            self.latent_normalization_factor = 1
            self.observation_normalization_factor = 1
        
        # Channel mapping for flux loss
        self.channel_mapping = config['loss'].get('channel_mapping', {})
        
        # Well locations from config
        self.prod_well_locations = self._extract_well_locations(config, 'producers')
        self.inj_well_locations = self._extract_well_locations(config, 'injectors')
        
        # Debug information
        if config.runtime.get('verbose', True):
            print(f"🔧 Loss Configuration:")
            print(f"   - BHP Loss: {'Enabled' if self.enable_bhp_loss else 'Disabled'}")
            print(f"   - Flux Loss: {'Enabled' if self.enable_flux_loss else 'Disabled'}")
            print(f"   - Non-Negative Loss: {'Enabled' if self.enable_non_negative_loss else 'Disabled'}")
            print(f"   - Reconstruction Variance: {self.reconstruction_variance:.4f} (std dev ≈ {self.reconstruction_variance**0.5:.3f})")
            print(f"     ↳ {'Strict' if self.reconstruction_variance < 0.05 else 'Balanced' if self.reconstruction_variance <= 0.2 else 'Forgiving'} reconstruction tolerance")
            
            # Per-element normalization information
            print(f"   - Per-Element Normalization: {'ENABLED' if self.enable_per_element_normalization else 'DISABLED'}")
            if self.enable_per_element_normalization:
                print(f"     ↳ Spatial normalization: ÷ {self.spatial_normalization_factor:,} elements (MSE per spatial element)")
                print(f"       • Grid: {config['data']['input_shape'][0]}×{config['data']['input_shape'][1]}×{config['data']['input_shape'][2]}×{config['data']['input_shape'][3]} = {self.spatial_normalization_factor:,}")
                print(f"     ↳ Latent normalization: ÷ {self.latent_normalization_factor} dimensions (MSE per latent dim)")
                print(f"     ↳ Observation normalization: ÷ {self.observation_normalization_factor} elements (MSE per observation)")
                print(f"       • Formula: ({num_prod} producers×2 + {num_inj} injectors) × {config['training']['num_tsteps']} timesteps = {self.observation_normalization_factor}")
                print(f"       • 🔧 FIXED: Now uses num_tsteps={config['training']['num_tsteps']} (full episode) not nsteps={config['training']['nsteps']} (training horizon)")
                print(f"     ↳ 🎯 All losses now balanced! Use λ ≈ 1.0 as starting point")
            else:
                print(f"     ↳ Using original scaling (reconstruction dominates due to {self.spatial_normalization_factor:,} spatial elements)")
                print(f"     ↳ Current λ compensation: trans={self.trans_loss_weight}, obs={self.yobs_loss_weight}")
            
            if self.enable_flux_loss and self.channel_mapping:
                print(f"   - Channel Mapping: {self.channel_mapping}")
            print(f"   - Producer well locations shape: {self.prod_well_locations.shape}")
            print(f"   - Producer well locations: {self.prod_well_locations.tolist()}")
        
        
        # Spatial enhancement losses
        self.enable_gradient_loss = config.loss.get('enable_gradient_loss', False)
        self.gradient_loss_weight = config.loss.get('lambda_gradient_loss', 0.1)
        
        self.enable_adversarial_loss = config.loss.get('enable_adversarial_loss', False)
        self.adversarial_loss_weight = config.loss.get('lambda_adversarial_loss', 0.01)
        self.adversarial_loss_type = config.loss.get('adversarial_loss_type', 'lsgan')
        
        # Initialize gradient loss if enabled
        if self.enable_gradient_loss:
            self.gradient_loss_fn = GradientLoss(
                directions=config.loss.get('gradient_loss_directions', ['x', 'y', 'z'])
            )
    
    def _extract_well_locations(self, config, well_type):
        """Extract well locations from config and convert to tensor"""
        import torch
        
        # Check if well_locations exists in config
        if 'well_locations' not in config['data']:
            print(f"⚠️  Warning: well_locations not found in config for {well_type}")
            return torch.tensor([], dtype=torch.long).reshape(0, 2)
        
        well_locations = config['data']['well_locations'][well_type]
        locations_list = []
        
        # Sort wells by name to ensure consistent ordering
        sorted_wells = sorted(well_locations.items())
        
        # Validate grid dimensions
        input_shape = config['data']['input_shape']
        max_x, max_y, max_z = input_shape[1]-1, input_shape[2]-1, input_shape[3]-1
        
        for well_name, coords in sorted_wells:
            x, y, z = coords[0], coords[1], coords[2]
            
            # Validate coordinates are within bounds
            if x > max_x or y > max_y or z > max_z:
                raise ValueError(f"Well {well_name} coordinates [{x}, {y}, {z}] are out of bounds. "
                               f"Max valid coordinates: [{max_x}, {max_y}, {max_z}]")
            
            # Convert [X, Y, Z] to [X, Y] for 2D indexing (Z penetrates all layers)
            locations_list.append([x, y])
            
        if len(locations_list) == 0:
            return torch.tensor([], dtype=torch.long).reshape(0, 2)
            
        return torch.tensor(locations_list, dtype=torch.long)

    def forward(self, pred, discriminator_pred=None):
        # Parse y_pred
        # X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec, perm = pred
        X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec = pred

        # ===== ORIGINAL LOSS COMPONENTS =====
        loss_rec_t = get_reconstruction_loss(x0, x0_rec, self.reconstruction_variance)
        loss_flux_t = 0
        loss_prod_bhp_t = 0
        loss_rec_t1 = 0
        loss_flux_t1 = 0
        loss_prod_bhp_t1 = 0
        
        # Add flux loss for initial reconstruction if enabled
        if self.enable_flux_loss and self.channel_mapping:
            loss_flux_t += get_flux_loss(x0, x0_rec, self.channel_mapping)
        
        for x_next, x_next_pred in zip(X_next, X_next_pred):
            loss_rec_t1 += get_reconstruction_loss(x_next, x_next_pred, self.reconstruction_variance)
            
            # Add BHP loss if enabled
            if self.enable_bhp_loss:
                if self.prod_well_locations.shape[0] > 0:  # Check if wells are defined
                    pressure_ch = self.channel_mapping.get('pressure', 2) if self.channel_mapping else 2
                    loss_prod_bhp_t1 += get_well_bhp_loss(x_next, x_next_pred, self.prod_well_locations, pressure_ch)
                else:
                    print("⚠️  Warning: BHP loss enabled but no producer wells defined!")
            
            # Add flux loss if enabled
            if self.enable_flux_loss:
                if self.channel_mapping:
                    loss_flux_t1 += get_flux_loss(x_next, x_next_pred, self.channel_mapping)
                else:
                    print("⚠️  Warning: Flux loss enabled but no channel_mapping provided in config!")

        loss_l2_reg = get_l2_reg_loss(z0)
        
        # Apply per-element normalization if enabled
        if self.enable_per_element_normalization:
            # Normalize reconstruction loss by spatial elements
            reconstruction_loss_normalized = (loss_rec_t + loss_rec_t1) / self.spatial_normalization_factor
            reconstruction_loss = self.reconstruction_loss_lambda * reconstruction_loss_normalized
        else:
            # Use original scaling
            reconstruction_loss = self.reconstruction_loss_lambda * (loss_rec_t + loss_rec_t1)
        
        physics_losses = self.bhp_loss_lambda * (loss_prod_bhp_t + loss_prod_bhp_t1) + self.flux_loss_lambda * (loss_flux_t + loss_flux_t1)
        loss_bound = reconstruction_loss + loss_l2_reg + physics_losses
        
        # Transition loss with optional normalization
        loss_trans = 0 
        for z_next, z_next_pred in zip(Z_next, Z_next_pred):
            loss_trans += get_l2_reg_loss(z_next - z_next_pred)
        
        if self.enable_per_element_normalization:
            # Normalize transition loss by latent dimensions
            loss_trans = loss_trans / self.latent_normalization_factor
            
        # Observation loss with optional normalization
        loss_yobs = 0 
        for y_next, y_next_pred in zip(Yobs, Yobs_pred):
            loss_yobs += get_l2_reg_loss(y_next - y_next_pred)
        
        if self.enable_per_element_normalization:
            # Normalize observation loss by observation elements
            loss_yobs = loss_yobs / self.observation_normalization_factor

        # ===== SPATIAL ENHANCEMENT LOSSES =====
        # Option 2: Gradient Loss for spatial detail preservation
        gradient_loss = 0
        if self.enable_gradient_loss:
            # Gradient loss for initial reconstruction
            gradient_loss += self.gradient_loss_fn(x0_rec, x0)
            # Gradient loss for multi-step predictions
            for x_next, x_next_pred in zip(X_next, X_next_pred):
                gradient_loss += self.gradient_loss_fn(x_next_pred, x_next)
            gradient_loss /= (len(X_next) + 1)  # Average over all steps
        
        # Option 4: Adversarial Loss for realistic reconstructions
        adversarial_loss = 0
        if self.enable_adversarial_loss and discriminator_pred is not None:
            if self.adversarial_loss_type == 'gan':
                # Standard GAN loss: -log(D(G(z)))
                adversarial_loss = -torch.mean(torch.log(torch.sigmoid(discriminator_pred) + 1e-8))
            elif self.adversarial_loss_type == 'lsgan':
                # Least squares GAN loss: (D(G(z)) - 1)^2
                adversarial_loss = torch.mean((discriminator_pred - 1) ** 2)
            elif self.adversarial_loss_type == 'wgan':
                # Wasserstein GAN loss: -D(G(z))
                adversarial_loss = -torch.mean(discriminator_pred)

        # ===== NON-NEGATIVE CONSTRAINT LOSS =====
        non_negative_loss = 0
        if self.enable_non_negative_loss:
            # Collect all reconstructed states and predicted observations
            reconstructed_states = [x0_rec] + X_next_pred
            predicted_observations = Yobs_pred
            non_negative_loss = get_non_negative_loss(reconstructed_states, predicted_observations)

        # ===== COMBINE ALL LOSSES =====
        self.flux_loss = loss_flux_t + loss_flux_t1
        
        # Store reconstruction loss (normalized if per-element normalization enabled)
        if self.enable_per_element_normalization:
            self.reconstruction_loss = (loss_rec_t + loss_rec_t1) / self.spatial_normalization_factor
        else:
            self.reconstruction_loss = loss_rec_t + loss_rec_t1
            
        self.well_loss = loss_prod_bhp_t + loss_prod_bhp_t1
        self.transition_loss = loss_trans
        self.observation_loss = loss_yobs
        self.gradient_loss = gradient_loss
        self.adversarial_loss = adversarial_loss
        self.non_negative_loss = non_negative_loss
        
        # Apply dynamic loss weighting if enabled
        if self.enable_dynamic_weighting:
            # Prepare loss dictionary for dynamic weighter
            task_losses = {}
            
            if 'flux' in self.task_names and 'bhp' in self.task_names:
                # Separate flux and BHP losses
                task_losses['reconstruction'] = loss_rec_t + loss_rec_t1
                task_losses['flux'] = self.flux_loss
                task_losses['bhp'] = self.well_loss
                task_losses['transition'] = loss_trans
                task_losses['observation'] = loss_yobs
            else:
                # Combined physics loss
                task_losses['reconstruction'] = loss_rec_t + loss_rec_t1
                task_losses['physics'] = self.flux_loss + self.well_loss
                task_losses['transition'] = loss_trans
                task_losses['observation'] = loss_yobs
            
            # Get model reference (passed via discriminator_pred for now)
            model_ref = getattr(self, '_model_ref', None)
            
            try:
                # Update dynamic weights
                if model_ref is not None:
                    updated_weights = self.dynamic_weighter.update_weights(
                        losses=task_losses,
                        model=model_ref
                    )
                    self.dynamic_weighter.step()
                else:
                    # Fallback: use current weights without updating
                    updated_weights = self.dynamic_weighter.get_weights()
                
                # Apply dynamic weights
                if 'flux' in self.task_names and 'bhp' in self.task_names:
                    weighted_loss = (
                        updated_weights['reconstruction'] * reconstruction_loss +
                        updated_weights['flux'] * self.flux_loss +
                        updated_weights['bhp'] * self.well_loss +
                        updated_weights['transition'] * loss_trans +
                        updated_weights['observation'] * loss_yobs
                    )
                else:
                    weighted_loss = (
                        updated_weights['reconstruction'] * reconstruction_loss +
                        updated_weights['physics'] * (self.flux_loss + self.well_loss) +
                        updated_weights['transition'] * loss_trans +
                        updated_weights['observation'] * loss_yobs
                    )
                
                # Add spatial enhancements (not included in dynamic weighting)
                self.total_loss = (
                    weighted_loss +
                    self.gradient_loss_weight * gradient_loss +
                    self.adversarial_loss_weight * adversarial_loss +
                    self.non_negative_loss_lambda * non_negative_loss
                )
                
                # Store current weights for monitoring
                self.current_dynamic_weights = updated_weights
                
            except Exception as e:
                print(f"⚠️ Warning: Dynamic loss weighting failed: {e}")
                print("   Falling back to static weights.")
                # Fallback to static weighting
                self.total_loss = (
                    reconstruction_loss + 
                    physics_losses +
                    self.trans_loss_weight * loss_trans + 
                    self.yobs_loss_weight * loss_yobs +
                    self.gradient_loss_weight * gradient_loss +
                    self.adversarial_loss_weight * adversarial_loss +
                    self.non_negative_loss_lambda * non_negative_loss
                )
        else:
            # Static loss weighting (original implementation)
            self.total_loss = (
                reconstruction_loss + 
                physics_losses +
                self.trans_loss_weight * loss_trans + 
                self.yobs_loss_weight * loss_yobs +
                self.gradient_loss_weight * gradient_loss +
                self.adversarial_loss_weight * adversarial_loss +
                self.non_negative_loss_lambda * non_negative_loss
            )

        return self.total_loss

    def getFluxLoss(self):
        return self.flux_loss

    def getReconstructionLoss(self):
        return self.reconstruction_loss

    def getWellLoss(self):
        return self.well_loss

    def getTotalLoss(self):
        return self.total_loss
    
    def getTransitionLoss(self):
        return self.transition_loss
    
    def getObservationLoss(self):
        return self.observation_loss
    
    def getNonNegativeLoss(self):
        return self.non_negative_loss
    
    def setModelReference(self, model):
        """Set model reference for dynamic loss weighting (specifically for GradNorm)."""
        self._model_ref = model
        
    def getDynamicWeights(self):
        """Get current dynamic weights if enabled."""
        if self.enable_dynamic_weighting and hasattr(self, 'current_dynamic_weights'):
            return self.current_dynamic_weights
        else:
            # Return static weights
            return {
                'reconstruction': self.reconstruction_loss_lambda,
                'flux': self.flux_loss_lambda,
                'bhp': self.bhp_loss_lambda,
                'transition': self.trans_loss_weight,
                'observation': self.yobs_loss_weight
            }
    
    def getDynamicWeightHistory(self):
        """Get weight evolution history for analysis."""
        if self.enable_dynamic_weighting:
            return self.dynamic_weighter.weight_history
        else:
            return {}
    
    def stepDynamicWeighter(self):
        """Increment step counter for dynamic weighter."""
        if self.enable_dynamic_weighting:
            self.dynamic_weighter.step()
    
    def epochDynamicWeighter(self):
        """Increment epoch counter for dynamic weighter."""
        if self.enable_dynamic_weighting:
            self.dynamic_weighter.epoch()
    
    def saveDynamicWeightPlot(self, save_path: str):
        """Save plot of weight evolution."""
        if self.enable_dynamic_weighting:
            try:
                from model.losses.dynamic_loss_weighting import plot_weight_history
                plot_weight_history(self.dynamic_weighter, save_path)
            except ImportError:
                print("⚠️ Cannot save weight plot: matplotlib not available")
        else:
            print("⚠️ Dynamic weighting not enabled")


