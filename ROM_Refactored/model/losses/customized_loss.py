"""
Customized loss function that aggregates multiple loss components
Supports dynamic loss weighting and spatial enhancements
"""

import os
import torch
import torch.nn as nn

from .individual_losses import (
    get_reconstruction_loss,
    get_flux_loss,
    get_well_bhp_loss,
    get_l2_reg_loss,
    get_non_negative_loss,
    get_kl_divergence_loss,
    get_fft_loss,
    get_masked_reconstruction_loss,
    get_eigloss,
    get_consistency_loss,
    get_energy_conservation_loss,
    get_jacobian_reg_loss,
    get_cycle_consistency_loss,
    get_dissipativity_loss,
    get_reversibility_loss,
    get_sindy_sparsity_loss,
    get_sindy_consistency_loss,
    get_sde_kl_loss,
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
        
        # VAE KL divergence loss configuration
        self.enable_kl_loss = config.loss.get('enable_kl_loss', False)
        self.kl_loss_lambda = config.loss.get('lambda_kl_loss', 0.001)  # beta in beta-VAE
        
        # KL Annealing configuration
        self.enable_kl_annealing = config.loss.get('enable_kl_annealing', False)
        self.kl_annealing_schedule = config.loss.get('kl_annealing_schedule', [])
        self.current_epoch = 0
        
        # If annealing is enabled, start with the first schedule value (typically 0)
        if self.enable_kl_annealing and self.kl_annealing_schedule:
            self.kl_loss_lambda = self.kl_annealing_schedule[0].get('lambda', 0.0)
        
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
        
        # Detect multimodal mode (reconstruction uses dynamic channels only)
        self.is_multimodal = config.config.get('multimodal', {}).get('enable', False)
        
        # Calculate normalization factors based on system dimensions
        if self.enable_per_element_normalization:
            # Spatial normalization: channels × grid cells
            input_shape = config['data']['input_shape']  # [channels, X, Y, Z]
            if self.is_multimodal:
                mm_cfg = config.config.get('multimodal', {})
                n_dynamic_ch = len(mm_cfg.get('dynamic_channels', [0, 1]))
                self.spatial_normalization_factor = n_dynamic_ch * input_shape[1] * input_shape[2] * input_shape[3]
            else:
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
            
            if self.is_multimodal:
                print(f"   - Multimodal Mode: ENABLED (reconstruction uses dynamic channels only)")
                if self.enable_bhp_loss or self.enable_flux_loss:
                    print(f"     ⚠️ BHP/Flux loss channel_mapping may need adjustment for 2-channel dynamic tensors")
            
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
        
        # ===== FFT LOSS CONFIGURATION =====
        self.enable_fft_loss = config.loss.get('enable_fft_loss', False)
        self.fft_loss_lambda = config.loss.get('lambda_fft_loss', 0.1)
        self.fft_loss = torch.tensor(0.0)  # Initialize for tracking
        
        if self.enable_fft_loss and config.runtime.get('verbose', True):
            print(f"   - FFT Loss: ENABLED (λ={self.fft_loss_lambda})")
            print(f"     ↳ Captures both low and high frequency reconstruction errors")
        
        # ===== ENCODER ENHANCEMENT LOSSES =====
        self.enable_jacobian_loss = config.loss.get('enable_jacobian_loss', False)
        self.jacobian_loss_lambda = config.loss.get('lambda_jacobian_loss', 0.01)
        self.jacobian_loss = torch.tensor(0.0)

        self.enable_cycle_loss = config.loss.get('enable_cycle_loss', False)
        self.cycle_loss_lambda = config.loss.get('lambda_cycle_loss', 0.1)
        self.cycle_loss = torch.tensor(0.0)

        # ===== SPECIAL TRANSITION MODEL LOSSES =====
        self.enable_eigloss = config.loss.get('enable_eigloss', False)
        self.eigloss_lambda = config.loss.get('lambda_eigloss', 0.01)
        self.eigloss = torch.tensor(0.0)

        self.enable_consistency_loss = config.loss.get('enable_consistency_loss', False)
        self.consistency_loss_lambda = config.loss.get('lambda_consistency_loss', 0.1)
        self.consistency_loss = torch.tensor(0.0)

        self.enable_energy_loss = config.loss.get('enable_energy_loss', False)
        self.energy_loss_lambda = config.loss.get('lambda_energy_loss', 0.1)
        self.energy_loss = torch.tensor(0.0)

        self.enable_dissipativity_loss = config.loss.get('enable_dissipativity_loss', False)
        self.dissipativity_loss_lambda = config.loss.get('lambda_dissipativity_loss', 0.01)
        self.dissipativity_loss = torch.tensor(0.0)

        self.enable_reversibility_loss = config.loss.get('enable_reversibility_loss', False)
        self.reversibility_loss_lambda = config.loss.get('lambda_reversibility_loss', 0.1)
        self.reversibility_loss = torch.tensor(0.0)

        self.enable_sindy_sparsity_loss = config.loss.get('enable_sindy_sparsity_loss', False)
        self.sindy_sparsity_loss_lambda = config.loss.get('lambda_sindy_sparsity_loss', 0.01)
        self.sindy_sparsity_loss = torch.tensor(0.0)

        self.enable_sindy_consistency_loss = config.loss.get('enable_sindy_consistency_loss', False)
        self.sindy_consistency_loss_lambda = config.loss.get('lambda_sindy_consistency_loss', 0.01)
        self.sindy_consistency_loss = torch.tensor(0.0)

        self.enable_sde_kl_loss = config.loss.get('enable_sde_kl_loss', False)
        self.sde_kl_loss_lambda = config.loss.get('lambda_sde_kl_loss', 0.01)
        self.sde_kl_loss = torch.tensor(0.0)

        # ===== INACTIVE CELL MASKING CONFIGURATION =====
        self.enable_inactive_masking = config.loss.get('enable_inactive_masking', False)
        self.mask_file_path = config.loss.get('mask_file_path', 'sr3_batch_output/inactive_cell_locations.h5')

        # Resolve relative mask path against config file directory
        if not os.path.isabs(self.mask_file_path):
            config_dir = os.path.dirname(os.path.abspath(
                getattr(config, 'config_path', 'config.yaml')
            ))
            resolved = os.path.normpath(os.path.join(config_dir, self.mask_file_path))
            if os.path.exists(resolved):
                self.mask_file_path = resolved

        self.active_mask = None
        
        if self.enable_inactive_masking:
            # Get device for mask
            device = config.runtime.get('device', 'auto')
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load the inactive cell mask
            try:
                self.active_mask = self._load_inactive_mask(self.mask_file_path, device)
                if config.runtime.get('verbose', True):
                    total_cells = self.active_mask.numel()
                    active_cells = self.active_mask.sum().item()
                    inactive_cells = total_cells - active_cells
                    print(f"   - Inactive Cell Masking: ENABLED")
                    print(f"     ↳ Mask loaded from: {self.mask_file_path}")
                    print(f"     ↳ Total cells: {total_cells:,}, Active: {active_cells:,}, Inactive: {inactive_cells:,}")
                    print(f"     ↳ Masking {inactive_cells/total_cells*100:.1f}% of cells from reconstruction loss")
            except Exception as e:
                print(f"   ⚠️ Failed to load inactive mask from {self.mask_file_path}: {e}")
                print(f"     ↳ Falling back to unmasked reconstruction loss")
                self.enable_inactive_masking = False
                self.active_mask = None
    
    def _load_inactive_mask(self, mask_path, device):
        """
        Load inactive cell mask from H5 file.
        
        Args:
            mask_path: Path to the H5 file containing mask data
            device: Device to load mask onto ('cuda', 'cpu', etc.)
            
        Returns:
            Active cell mask tensor (True = active cell)
            - Shape [num_cases, X, Y, Z] for case-specific masks (4D)
            - Shape [X, Y, Z] for global masks (3D)
        """
        import h5py
        import os
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        with h5py.File(mask_path, 'r') as f:
            if 'active_mask' not in f:
                raise KeyError(f"'active_mask' dataset not found in {mask_path}")
            
            active_mask_data = f['active_mask'][...]
            active_mask = torch.tensor(active_mask_data, dtype=torch.bool)
        
        # Keep the original dimensionality
        if active_mask.dim() == 4:
            # Case-specific mask [num_cases, X, Y, Z]
            print(f"     ↳ Case-specific 4D mask loaded: {list(active_mask.shape)}")
            print(f"     ↳ {active_mask.shape[0]} cases, each with {active_mask[0].sum().item()} active cells")
        elif active_mask.dim() == 3:
            # Global mask [X, Y, Z]
            print(f"     ↳ Global 3D mask loaded: {list(active_mask.shape)}")
            print(f"     ↳ {active_mask.sum().item()} active cells")
        else:
            raise ValueError(f"Expected 3D or 4D mask, got shape {active_mask.shape}")
        
        return active_mask.to(device)
    
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

    def forward(self, pred, discriminator_pred=None, case_indices=None):
        """
        Compute the total loss.
        
        Args:
            pred: Model predictions tuple
            discriminator_pred: Discriminator predictions for adversarial loss
            case_indices: Optional tensor [batch] mapping each sample to its original case index
                         Used for case-specific mask lookup when enable_inactive_masking=True
        """
        # Parse predictions tuple (variable length for backwards compatibility)
        Z_re_encoded = None
        if len(pred) == 12:
            X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec, mu_list, logvar_list, Z_re_encoded = pred
        elif len(pred) == 11:
            X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec, mu_list, logvar_list = pred
        else:
            X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec = pred
            mu_list, logvar_list = None, None

        # ===== PREPARE BATCH MASK FOR CASE-SPECIFIC MASKING =====
        batch_mask = None
        if self.enable_inactive_masking and self.active_mask is not None:
            if case_indices is not None and self.active_mask.dim() == 4:
                # Use case_indices to select masks for this batch from full 4D mask
                # active_mask shape: [num_cases, X, Y, Z], case_indices shape: [batch]
                batch_mask = self.active_mask[case_indices]  # Result: [batch, X, Y, Z]
            elif self.active_mask.dim() == 3:
                # Global 3D mask, broadcast to batch
                batch_mask = self.active_mask  # [X, Y, Z]
            else:
                # 4D mask but no case_indices - fallback to first case
                batch_mask = self.active_mask[0]  # [X, Y, Z]

        # ===== ORIGINAL LOSS COMPONENTS =====
        # Compute reconstruction loss with optional inactive cell masking
        if self.enable_inactive_masking and batch_mask is not None:
            # Use masked reconstruction loss - excludes inactive cells
            loss_rec_t = get_masked_reconstruction_loss(x0, x0_rec, batch_mask, self.reconstruction_variance)
        else:
            # Use standard reconstruction loss
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
            # Compute reconstruction loss with optional inactive cell masking
            if self.enable_inactive_masking and batch_mask is not None:
                loss_rec_t1 += get_masked_reconstruction_loss(x_next, x_next_pred, batch_mask, self.reconstruction_variance)
            else:
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

        # ===== VAE KL DIVERGENCE LOSS =====
        kl_loss = 0
        if self.enable_kl_loss and mu_list is not None and logvar_list is not None:
            # Compute KL loss for all encoder outputs
            for mu, logvar in zip(mu_list, logvar_list):
                if mu is not None and logvar is not None:
                    kl_loss += get_kl_divergence_loss(mu, logvar, reduction='mean')
            # Average over all encoded states
            if len(mu_list) > 0:
                kl_loss = kl_loss / len(mu_list)

        # ===== FFT LOSS (FREQUENCY DOMAIN) =====
        fft_loss = 0
        if self.enable_fft_loss:
            # FFT loss for initial reconstruction
            fft_loss += get_fft_loss(x0, x0_rec, normalize=True)
            # FFT loss for multi-step predictions
            for x_next, x_next_pred in zip(X_next, X_next_pred):
                fft_loss += get_fft_loss(x_next, x_next_pred, normalize=True)
            # Average over all steps
            fft_loss /= (len(X_next) + 1)
        
        # Store FFT loss for tracking
        self.fft_loss = fft_loss

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
        self.kl_loss = kl_loss

        # GNN well-node readout auxiliary loss (if available from GNNE2C)
        self.well_readout_loss = 0
        model_ref = getattr(self, '_model_ref', None)
        if model_ref is not None and hasattr(model_ref, 'get_well_readout_loss'):
            aux = model_ref.get_well_readout_loss()
            if aux is not None:
                lam = getattr(model_ref, 'well_readout_lambda', 1.0)
                self.well_readout_loss = lam * aux

        # AC-CSS attractor auxiliary loss
        self.attractor_loss = 0
        if model_ref is not None and hasattr(model_ref, 'transition'):
            transition = model_ref.transition
            if hasattr(transition, '_attractor_targets') and transition._attractor_targets:
                lambda_attr = self.config.loss.get('lambda_attractor_loss', 1.0)
                attr_loss = 0
                targets = transition._attractor_targets
                for i, z_attr in enumerate(targets):
                    if i < len(Z_next):
                        attr_loss = attr_loss + get_l2_reg_loss(z_attr - Z_next[i])
                if len(targets) > 0:
                    self.attractor_loss = lambda_attr * attr_loss / len(targets)
                transition._attractor_targets = []

        # ===== ENCODER ENHANCEMENT LOSSES =====
        jacobian_loss = torch.tensor(0.0, device=x0.device)
        if self.enable_jacobian_loss and x0.requires_grad:
            jacobian_loss = get_jacobian_reg_loss(z0, x0)
        self.jacobian_loss = jacobian_loss

        cycle_loss = torch.tensor(0.0, device=x0.device)
        if self.enable_cycle_loss and Z_re_encoded is not None:
            for z_orig, z_re in zip([z0] + Z_next_pred, Z_re_encoded):
                cycle_loss = cycle_loss + get_cycle_consistency_loss(z_orig, z_re)
        self.cycle_loss = cycle_loss

        # ===== SPECIAL TRANSITION MODEL LOSSES =====
        eigloss = torch.tensor(0.0, device=x0.device)
        consistency_loss = torch.tensor(0.0, device=x0.device)
        energy_loss = torch.tensor(0.0, device=x0.device)
        dissipativity_loss = torch.tensor(0.0, device=x0.device)
        reversibility_loss = torch.tensor(0.0, device=x0.device)
        sindy_sparsity_loss = torch.tensor(0.0, device=x0.device)
        sindy_consistency_loss = torch.tensor(0.0, device=x0.device)
        sde_kl_loss = torch.tensor(0.0, device=x0.device)

        if model_ref is not None and hasattr(model_ref, 'transition'):
            transition = model_ref.transition

            if self.enable_eigloss and hasattr(transition, 'get_eigenvalues'):
                eigvals = transition.get_eigenvalues()
                eigloss = get_eigloss(eigvals)

            if self.enable_energy_loss and hasattr(transition, 'get_trajectory_energies'):
                traj_energies = transition.get_trajectory_energies()
                energy_loss = get_energy_conservation_loss(traj_energies)

            if self.enable_dissipativity_loss and hasattr(transition, 'get_singular_values'):
                sv = transition.get_singular_values()
                if sv is not None:
                    dissipativity_loss = get_dissipativity_loss(sv)

            if self.enable_reversibility_loss and hasattr(transition, 'get_reversibility_residual'):
                rev_res = transition.get_reversibility_residual()
                if rev_res is not None:
                    reversibility_loss = get_reversibility_loss(rev_res)

            if self.enable_sindy_sparsity_loss and hasattr(transition, 'get_sindy_coefficients'):
                coeffs = transition.get_sindy_coefficients()
                if coeffs is not None:
                    sindy_sparsity_loss = get_sindy_sparsity_loss(coeffs)

            if self.enable_sindy_consistency_loss and hasattr(transition, 'get_sindy_velocity'):
                z_dot_sindy = transition.get_sindy_velocity()
                if z_dot_sindy is not None and len(Z_next_pred) > 0:
                    z_dot_model = (Z_next_pred[-1] - Z_next_pred[-2]) if len(Z_next_pred) > 1 else Z_next_pred[0]
                    sindy_consistency_loss = get_sindy_consistency_loss(z_dot_model.detach(), z_dot_sindy)

            if self.enable_sde_kl_loss and hasattr(transition, 'get_diffusion_values'):
                diff_val = transition.get_diffusion_values()
                if diff_val is not None:
                    sde_kl_loss = get_sde_kl_loss(diff_val)

        self.eigloss = eigloss
        self.consistency_loss = consistency_loss
        self.energy_loss = energy_loss
        self.dissipativity_loss = dissipativity_loss
        self.reversibility_loss = reversibility_loss
        self.sindy_sparsity_loss = sindy_sparsity_loss
        self.sindy_consistency_loss = sindy_consistency_loss
        self.sde_kl_loss = sde_kl_loss

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
                
                # Add spatial enhancements, FFT loss, VAE KL loss, and well readout (not in dynamic weighting)
                self.total_loss = (
                    weighted_loss +
                    self.gradient_loss_weight * gradient_loss +
                    self.adversarial_loss_weight * adversarial_loss +
                    self.non_negative_loss_lambda * non_negative_loss +
                    self.kl_loss_lambda * kl_loss +
                    self.fft_loss_lambda * fft_loss +
                    self.well_readout_loss +
                    self.eigloss_lambda * eigloss +
                    self.consistency_loss_lambda * consistency_loss +
                    self.energy_loss_lambda * energy_loss +
                    self.dissipativity_loss_lambda * dissipativity_loss +
                    self.reversibility_loss_lambda * reversibility_loss +
                    self.sindy_sparsity_loss_lambda * sindy_sparsity_loss +
                    self.sindy_consistency_loss_lambda * sindy_consistency_loss +
                    self.sde_kl_loss_lambda * sde_kl_loss +
                    self.jacobian_loss_lambda * jacobian_loss +
                    self.cycle_loss_lambda * cycle_loss
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
                    self.non_negative_loss_lambda * non_negative_loss +
                    self.kl_loss_lambda * kl_loss +
                    self.fft_loss_lambda * fft_loss +
                    self.well_readout_loss +
                    self.eigloss_lambda * eigloss +
                    self.consistency_loss_lambda * consistency_loss +
                    self.energy_loss_lambda * energy_loss +
                    self.dissipativity_loss_lambda * dissipativity_loss +
                    self.reversibility_loss_lambda * reversibility_loss +
                    self.sindy_sparsity_loss_lambda * sindy_sparsity_loss +
                    self.sindy_consistency_loss_lambda * sindy_consistency_loss +
                    self.sde_kl_loss_lambda * sde_kl_loss +
                    self.jacobian_loss_lambda * jacobian_loss +
                    self.cycle_loss_lambda * cycle_loss
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
                self.non_negative_loss_lambda * non_negative_loss +
                self.kl_loss_lambda * kl_loss +
                self.fft_loss_lambda * fft_loss +
                self.well_readout_loss +
                self.attractor_loss +
                self.eigloss_lambda * eigloss +
                self.consistency_loss_lambda * consistency_loss +
                self.energy_loss_lambda * energy_loss +
                self.dissipativity_loss_lambda * dissipativity_loss +
                self.reversibility_loss_lambda * reversibility_loss +
                self.sindy_sparsity_loss_lambda * sindy_sparsity_loss +
                self.sindy_consistency_loss_lambda * sindy_consistency_loss +
                self.sde_kl_loss_lambda * sde_kl_loss +
                self.jacobian_loss_lambda * jacobian_loss +
                self.cycle_loss_lambda * cycle_loss
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

    def getWellReadoutLoss(self):
        return self.well_readout_loss
    
    def getFFTLoss(self):
        """Get FFT (frequency domain) loss."""
        return self.fft_loss
    
    def getJacobianLoss(self):
        """Get Jacobian regularization loss (contractive encoder)."""
        return self.jacobian_loss

    def getCycleLoss(self):
        """Get cycle-consistency loss (encode-decode-re-encode)."""
        return self.cycle_loss

    def getEigLoss(self):
        """Get eigenvalue regularisation loss (Stable Koopman)."""
        return self.eigloss

    def getConsistencyLoss(self):
        """Get temporal consistency loss (Deep Koopman)."""
        return self.consistency_loss

    def getEnergyLoss(self):
        """Get energy conservation loss (Hamiltonian Neural ODE)."""
        return self.energy_loss

    def getDissipativityLoss(self):
        """Get dissipativity loss (Dissipative Koopman)."""
        return self.dissipativity_loss

    def getReversibilityLoss(self):
        """Get reversibility loss (IS-FNO)."""
        return self.reversibility_loss

    def getSINDySparsityLoss(self):
        """Get SINDy coefficient sparsity loss."""
        return self.sindy_sparsity_loss

    def getSINDyConsistencyLoss(self):
        """Get SINDy velocity consistency loss."""
        return self.sindy_consistency_loss

    def getSDEKLLoss(self):
        """Get Latent SDE KL/diffusion regularisation loss."""
        return self.sde_kl_loss

    def getKLLoss(self):
        """Get KL divergence loss (VAE)."""
        return self.kl_loss
    
    def getKLLambda(self):
        """Get current KL loss weight (lambda)."""
        return self.kl_loss_lambda
    
    def update_kl_lambda_for_epoch(self, epoch):
        """
        Update KL lambda based on annealing schedule.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current KL lambda value
        """
        self.current_epoch = epoch
        
        if not self.enable_kl_annealing:
            return self.kl_loss_lambda
        
        # Find the appropriate lambda for this epoch from the schedule
        current_lambda = 0.0
        for entry in self.kl_annealing_schedule:
            if epoch >= entry.get('epoch', 0):
                current_lambda = entry.get('lambda', 0.0)
        
        self.kl_loss_lambda = current_lambda
        return current_lambda
    
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


