"""
FNO (Fourier Neural Operator) transition models for E2C architecture
Operates directly on spatial fields using frequency domain learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """
    3D Fourier layer for learning spatiotemporal operators in frequency domain
    Optimized for reservoir simulation with proper handling of 3D spatial + control inputs
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (x-direction)
        self.modes2 = modes2  # Number of Fourier modes to keep (y-direction)  
        self.modes3 = modes3  # Number of Fourier modes to keep (z-direction)

        self.scale = (1 / (in_channels * out_channels))
        # Initialize complex weights for 3D FFT
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # Complex multiplication for 3D tensors
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3,-2,-1])
        return x

class FNOTransitionModel(nn.Module):
    """
    Fourier Neural Operator-based transition model for E2C architecture
    
    Key Features:
    - Operates directly on spatial fields instead of latent vectors
    - Learns spatiotemporal operators in frequency domain
    - Incorporates control inputs through spatial conditioning
    - Maintains compatibility with existing E2C workflow
    - Configurable modes and architecture depth
    """
    def __init__(self, config):
        super(FNOTransitionModel, self).__init__()
        self.config = config
        
        # Model configuration
        fno_config = config['transition'].get('fno', {})
        # Handle both dict and SimpleNamespace for fno_config
        if isinstance(fno_config, dict):
            self.width = fno_config.get('width', 64)  # Channel width
            self.modes1 = fno_config.get('modes_x', 8)  # Fourier modes in x
            self.modes2 = fno_config.get('modes_y', 8)  # Fourier modes in y  
            self.modes3 = fno_config.get('modes_z', 4)  # Fourier modes in z
            self.n_layers = fno_config.get('n_layers', 4)  # Number of FNO layers
        else:
            self.width = getattr(fno_config, 'width', 64)  # Channel width
            self.modes1 = getattr(fno_config, 'modes_x', 8)  # Fourier modes in x
            self.modes2 = getattr(fno_config, 'modes_y', 8)  # Fourier modes in y  
            self.modes3 = getattr(fno_config, 'modes_z', 4)  # Fourier modes in z
            self.n_layers = getattr(fno_config, 'n_layers', 4)  # Number of FNO layers
        
        # Physical dimensions
        self.num_prod = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.n_channels = config['model']['n_channels']
        self.u_dim = config['model']['u_dim']
        self.spatial_dims = config['data']['input_shape'][1:]  # (Nx, Ny, Nz)
        
        # Control injection method
        if isinstance(fno_config, dict):
            self.control_injection = fno_config.get('control_injection', 'spatial_encoding')
        else:
            self.control_injection = getattr(fno_config, 'control_injection', 'spatial_encoding')
        
        # ===== WELL-SPECIFIC SPATIAL INJECTION CONFIGURATION =====
        self.well_locations_producers = self._extract_well_locations(config, 'producers')
        self.well_locations_injectors = self._extract_well_locations(config, 'injectors')
        
        # Well-specific spatial injection configuration
        if self.control_injection == 'well_specific_spatial':
            well_spatial_config = fno_config.get('well_specific_spatial', {})
            if isinstance(well_spatial_config, dict):
                self.influence_radius = well_spatial_config.get('influence_radius', 3.0)
                self.influence_function = well_spatial_config.get('influence_function', 'gaussian')
                self.influence_strength = well_spatial_config.get('influence_strength', 1.0)
                self.use_3d_influence = well_spatial_config.get('use_3d_influence', True)
                self.z_penetration_mode = well_spatial_config.get('z_penetration_mode', 'full')
                self.normalization = well_spatial_config.get('normalization', 'max_normalize')
                self.per_well_scaling = well_spatial_config.get('per_well_scaling', True)
                self.injector_scaling = well_spatial_config.get('injector_scaling', 1.2)
                self.producer_scaling = well_spatial_config.get('producer_scaling', 0.8)
                self.temporal_weighting = well_spatial_config.get('temporal_weighting', True)
                self.model_interference = well_spatial_config.get('model_interference', False)
                self.interference_threshold = well_spatial_config.get('interference_threshold', 5.0)
            else:
                self.influence_radius = getattr(well_spatial_config, 'influence_radius', 3.0)
                self.influence_function = getattr(well_spatial_config, 'influence_function', 'gaussian')
                self.influence_strength = getattr(well_spatial_config, 'influence_strength', 1.0)
                self.use_3d_influence = getattr(well_spatial_config, 'use_3d_influence', True)
                self.z_penetration_mode = getattr(well_spatial_config, 'z_penetration_mode', 'full')
                self.normalization = getattr(well_spatial_config, 'normalization', 'max_normalize')
                self.per_well_scaling = getattr(well_spatial_config, 'per_well_scaling', True)
                self.injector_scaling = getattr(well_spatial_config, 'injector_scaling', 1.2)
                self.producer_scaling = getattr(well_spatial_config, 'producer_scaling', 0.8)
                self.temporal_weighting = getattr(well_spatial_config, 'temporal_weighting', True)
                self.model_interference = getattr(well_spatial_config, 'model_interference', False)
                self.interference_threshold = getattr(well_spatial_config, 'interference_threshold', 5.0)
            
            # Pre-compute well influence patterns for efficiency
            self._precompute_well_influence_patterns()
        
        # Input projection: (state_channels + control_channels + time) -> width
        control_channels = self._get_control_channels()
        input_channels = self.n_channels + control_channels + 1  # +1 for time
        
        self.fc0 = nn.Linear(input_channels, self.width)
        
        # FNO layers
        self.fno_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()  # Local transformation layers
        
        for i in range(self.n_layers):
            self.fno_layers.append(SpectralConv3d(self.width, self.width, 
                                                 self.modes1, self.modes2, self.modes3))
            self.w_layers.append(nn.Conv3d(self.width, self.width, 1))
        
        # Output projections
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2_state = nn.Linear(128, self.n_channels)  # Next state prediction
        self.fc2_obs = nn.Linear(128, self.num_prod*2 + self.num_inj)  # Observation prediction
        
        # Activation
        self.activation = F.gelu
        
        # Control spatial encoder (for spatial injection method)
        if self.control_injection in ['spatial_encoding', 'well_specific_spatial']:
            self.control_encoder = self._build_control_encoder()
        
        # Well location embeddings for observation extraction
        self.well_embeddings = self._build_well_embeddings()
        
        if config['runtime'].get('verbose', False):
            print(f"🌊 FNO TRANSITION: Input channels {input_channels} → Width {self.width}")
            print(f"🌊 FNO TRANSITION: Modes ({self.modes1}, {self.modes2}, {self.modes3}), Layers {self.n_layers}")
            print(f"🌊 FNO TRANSITION: Control injection method: {self.control_injection}")
            print(f"🌊 FNO TRANSITION: Spatial dims {self.spatial_dims}")
            if self.control_injection == 'well_specific_spatial':
                print(f"🎯 WELL-SPECIFIC INJECTION: Influence radius {self.influence_radius}, Function {self.influence_function}")
                print(f"🎯 WELL-SPECIFIC INJECTION: Producer wells {len(self.well_locations_producers)}, Injector wells {len(self.well_locations_injectors)}")
    
    def _extract_well_locations(self, config, well_type):
        """Extract and validate well locations from config"""
        import torch
        
        if 'well_locations' not in config['data']:
            print(f"⚠️  Warning: well_locations not found in config for {well_type}")
            return torch.tensor([], dtype=torch.long).reshape(0, 3)
        
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
            
            # Store full [X, Y, Z] coordinates for 3D spatial injection
            locations_list.append([x, y, z])
            
        if len(locations_list) == 0:
            return torch.tensor([], dtype=torch.long).reshape(0, 3)
            
        return torch.tensor(locations_list, dtype=torch.long)
    
    def _get_control_channels(self):
        """Calculate number of control channels based on injection method"""
        if self.control_injection == 'spatial_encoding':
            return self.u_dim  # Each control gets a spatial channel
        elif self.control_injection == 'well_specific_spatial':
            return self.u_dim  # Each control gets a well-specific spatial channel
        elif self.control_injection == 'global_conditioning':
            return 1  # Single global conditioning channel
        else:
            return 0  # No additional channels
    
    def _build_control_encoder(self):
        """Build spatial encoder for control inputs"""
        layers = []
        layers.append(nn.Linear(self.u_dim, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 64))
        layers.append(nn.ReLU()) 
        layers.append(nn.Linear(64, self.u_dim))
        return nn.Sequential(*layers)
    
    def _build_well_embeddings(self):
        """Build sophisticated well observation embeddings"""
        embeddings = nn.ModuleDict()
        
        # Enhanced injection well embeddings (for BHP prediction)
        embeddings['injection'] = nn.Sequential(
            nn.Linear(self.width, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_inj)  # One BHP per injector
        )
        
        # Enhanced production well embeddings (for BHP + rates prediction)
        embeddings['production'] = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_prod*2)  # BHP + rate per producer
        )
        
        return embeddings
    
    def _inject_controls_spatial(self, state, controls, dt):
        """Inject control inputs as spatial fields"""
        batch_size = state.shape[0]
        Nx, Ny, Nz = self.spatial_dims
        
        # Encode controls
        controls_encoded = self.control_encoder(controls * dt)  # (batch, u_dim)
        
        # Broadcast to spatial dimensions
        control_fields = []
        for i in range(self.u_dim):
            # Create spatial field for each control
            control_field = controls_encoded[:, i:i+1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            control_field = control_field.expand(batch_size, 1, Nx, Ny, Nz)
            control_fields.append(control_field)
        
        control_spatial = torch.cat(control_fields, dim=1)  # (batch, u_dim, Nx, Ny, Nz)
        return control_spatial
    
    def _inject_controls_global(self, state, controls, dt):
        """Inject control inputs as global conditioning"""
        batch_size = state.shape[0]
        Nx, Ny, Nz = self.spatial_dims
        
        # Simple global conditioning: sum of control magnitudes
        global_control = torch.sum(controls * dt, dim=1, keepdim=True)  # (batch, 1)
        global_field = global_control.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        global_field = global_field.expand(batch_size, 1, Nx, Ny, Nz)
        
        return global_field
    
    def forward(self, state, dt, controls):
        """
        Forward pass through FNO transition model
        
        Args:
            state: Current spatial state (batch, n_channels, Nx, Ny, Nz)
            dt: Time step (batch, 1)
            controls: Control inputs (batch, u_dim)
            
        Returns:
            next_state: Predicted next state (batch, n_channels, Nx, Ny, Nz)
            observations: Predicted observations (batch, 2*num_prod + num_inj)
        """
        batch_size = state.shape[0]
        Nx, Ny, Nz = state.shape[2:]
        
        # Prepare time field
        time_field = dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, Nx, Ny, Nz)
        
        # Inject controls based on chosen method
        if self.control_injection == 'spatial_encoding':
            control_fields = self._inject_controls_spatial(state, controls, dt)
            # Concatenate: state + controls + time
            input_field = torch.cat([state, control_fields, time_field], dim=1)
        elif self.control_injection == 'well_specific_spatial':
            control_fields = self._inject_controls_well_specific_spatial(state, controls, dt)
            # Concatenate: state + well-specific controls + time
            input_field = torch.cat([state, control_fields, time_field], dim=1)
        elif self.control_injection == 'global_conditioning':
            global_field = self._inject_controls_global(state, controls, dt)
            input_field = torch.cat([state, global_field, time_field], dim=1)
        else:
            # No control injection - just state + time
            input_field = torch.cat([state, time_field], dim=1)
        
        # Rearrange for FNO: (batch, channels, x, y, z) -> (batch, x, y, z, channels)
        x = input_field.permute(0, 2, 3, 4, 1)
        
        # Input projection
        x = self.fc0(x)  # (batch, Nx, Ny, Nz, width)
        
        # Rearrange back: (batch, x, y, z, channels) -> (batch, channels, x, y, z)
        x = x.permute(0, 4, 1, 2, 3)
        
        # FNO layers
        for i in range(self.n_layers):
            x1 = self.fno_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = self.activation(x)
        
        # Rearrange for output projection
        x = x.permute(0, 2, 3, 4, 1)  # (batch, Nx, Ny, Nz, width)
        
        # Output projections
        x = self.activation(self.fc1(x))  # (batch, Nx, Ny, Nz, 128)
        
        # State prediction
        next_state_delta = self.fc2_state(x)  # (batch, Nx, Ny, Nz, n_channels)
        next_state_delta = next_state_delta.permute(0, 4, 1, 2, 3)  # (batch, n_channels, Nx, Ny, Nz)
        
        # Residual connection for stability
        next_state = state + next_state_delta
        
        # Observation prediction (extract from actual well locations)
        if self.control_injection == 'well_specific_spatial' and hasattr(self, 'well_locations_injectors'):
            observations = self._extract_well_observations(x)
        else:
            # Fallback to global average pooling
            obs_features = torch.mean(x, dim=[1, 2, 3])  # (batch, 128)
            observations = self.fc2_obs(obs_features)  # (batch, 2*num_prod + num_inj)
        
        return next_state, observations
    
    def forward_nsteps(self, state, dt, controls_sequence):
        """
        Multi-step forward prediction
        
        Args:
            state: Initial state (batch, n_channels, Nx, Ny, Nz)
            dt: Time step (batch, 1)
            controls_sequence: List of control inputs for each step
            
        Returns:
            states_pred: List of predicted states
            observations_pred: List of predicted observations
        """
        states_pred = []
        observations_pred = []
        current_state = state
        
        for controls in controls_sequence:
            next_state, observations = self.forward(current_state, dt, controls)
            states_pred.append(next_state)
            observations_pred.append(observations)
            current_state = next_state
        
        return states_pred, observations_pred

    def _precompute_well_influence_patterns(self):
        """Precompute influence patterns for all wells to optimize performance"""
        import torch
        import numpy as np
        
        Nx, Ny, Nz = self.spatial_dims
        
        # Create coordinate grids
        x_coords = torch.arange(Nx, dtype=torch.float32)
        y_coords = torch.arange(Ny, dtype=torch.float32)
        z_coords = torch.arange(Nz, dtype=torch.float32)
        
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Store influence patterns for each well
        self.well_influence_patterns = {}
        
        # Process injector wells (controls 0, 1, 2 correspond to injectors I1, I2, I4)
        for i, well_loc in enumerate(self.well_locations_injectors):
            well_x, well_y, well_z = well_loc[0].item(), well_loc[1].item(), well_loc[2].item()
            pattern = self._compute_influence_pattern(X, Y, Z, well_x, well_y, well_z, 'injector')
            self.well_influence_patterns[f'injector_{i}'] = pattern
        
        # Process producer wells (controls 3, 4, 5 correspond to producers P1, P2, P3)
        for i, well_loc in enumerate(self.well_locations_producers):
            well_x, well_y, well_z = well_loc[0].item(), well_loc[1].item(), well_loc[2].item()
            pattern = self._compute_influence_pattern(X, Y, Z, well_x, well_y, well_z, 'producer')
            self.well_influence_patterns[f'producer_{i}'] = pattern
        
        print(f"🎯 Precomputed {len(self.well_influence_patterns)} well influence patterns")
    
    def _compute_influence_pattern(self, X, Y, Z, well_x, well_y, well_z, well_type):
        """Compute influence pattern for a single well"""
        import torch
        import math
        
        # Calculate distances from well location
        if self.use_3d_influence and self.z_penetration_mode == 'full':
            # Full 3D influence (wells penetrate all layers)
            dist = torch.sqrt((X - well_x)**2 + (Y - well_y)**2 + ((Z - well_z) * 0.1)**2)  # Reduced Z weight
        elif self.use_3d_influence and self.z_penetration_mode == 'partial':
            # Partial Z influence with exponential decay
            z_weight = torch.exp(-torch.abs(Z - well_z) * 0.2)
            dist = torch.sqrt((X - well_x)**2 + (Y - well_y)**2) * (1.0 / (z_weight + 0.1))
        else:
            # 2D influence only (traditional approach)
            dist = torch.sqrt((X - well_x)**2 + (Y - well_y)**2)
        
        # Apply influence function
        if self.influence_function == 'gaussian':
            # Gaussian influence: exp(-dist^2 / (2 * sigma^2))
            sigma = self.influence_radius / 3.0  # 3-sigma rule
            influence = torch.exp(-dist**2 / (2 * sigma**2))
        elif self.influence_function == 'exponential':
            # Exponential decay: exp(-dist / radius)
            influence = torch.exp(-dist / self.influence_radius)
        elif self.influence_function == 'linear':
            # Linear decay: max(0, 1 - dist/radius)
            influence = torch.clamp(1.0 - dist / self.influence_radius, 0.0, 1.0)
        elif self.influence_function == 'step':
            # Step function: 1 inside radius, 0 outside
            influence = (dist <= self.influence_radius).float()
        else:
            raise ValueError(f"Unknown influence function: {self.influence_function}")
        
        # Apply well-specific scaling
        if self.per_well_scaling:
            if well_type == 'injector':
                influence *= self.injector_scaling
            elif well_type == 'producer':
                influence *= self.producer_scaling
        
        # Apply normalization
        if self.normalization == 'max_normalize':
            influence = influence / (torch.max(influence) + 1e-8)
        elif self.normalization == 'sum_normalize':
            influence = influence / (torch.sum(influence) + 1e-8)
        # 'none' means no normalization
        
        return influence * self.influence_strength
    
    def _inject_controls_well_specific_spatial(self, state, controls, dt):
        """
        Advanced well-specific spatial injection method
        
        This method injects each control input at the specific spatial location
        of its corresponding well with proper influence zones and physics-based
        spreading patterns.
        
        Args:
            state: Current spatial state (batch, n_channels, Nx, Ny, Nz)
            controls: Control inputs (batch, u_dim)
            dt: Time step (batch, 1)
            
        Returns:
            control_spatial: Well-localized control fields (batch, u_dim, Nx, Ny, Nz)
        """
        import torch
        
        batch_size = state.shape[0]
        Nx, Ny, Nz = self.spatial_dims
        device = state.device
        
        # Encode controls with temporal weighting
        if self.temporal_weighting:
            controls_encoded = self.control_encoder(controls * dt)  # (batch, u_dim)
        else:
            controls_encoded = self.control_encoder(controls)  # (batch, u_dim)
        
        # Initialize control fields
        control_fields = torch.zeros(batch_size, self.u_dim, Nx, Ny, Nz, device=device)
        
        # Control mapping based on data structure:
        # Controls 0-2: Gas injection rates for injectors (I1, I2, I4)
        # Controls 3-5: Producer BHP for producers (P1, P2, P3)
        
        # Inject gas injection controls (injectors)
        for i in range(min(3, len(self.well_locations_injectors))):
            if i < controls_encoded.shape[1]:  # Ensure control exists
                pattern_key = f'injector_{i}'
                if pattern_key in self.well_influence_patterns:
                    pattern = self.well_influence_patterns[pattern_key].to(device)
                    control_magnitude = controls_encoded[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    control_fields[:, i, :, :, :] = control_magnitude * pattern
        
        # Inject producer BHP controls (producers)
        for i in range(min(3, len(self.well_locations_producers))):
            control_idx = i + 3  # Producer controls start at index 3
            if control_idx < controls_encoded.shape[1]:  # Ensure control exists
                pattern_key = f'producer_{i}'
                if pattern_key in self.well_influence_patterns:
                    pattern = self.well_influence_patterns[pattern_key].to(device)
                    control_magnitude = controls_encoded[:, control_idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    control_fields[:, control_idx, :, :, :] = control_magnitude * pattern
        
        # Model well interference if enabled
        if self.model_interference:
            control_fields = self._apply_well_interference(control_fields)
        
        return control_fields
    
    def _apply_well_interference(self, control_fields):
        """Apply interference effects between nearby wells"""
        # This is a simplified interference model
        # In a full implementation, you would model pressure interference
        # between wells based on their spatial separation and flow rates
        
        # For now, apply a simple smoothing between nearby wells
        import torch.nn.functional as F
        
        # Apply mild spatial smoothing to represent pressure communication
        kernel_size = 3
        padding = kernel_size // 2
        smoothed_fields = F.conv3d(
            control_fields.view(-1, 1, *control_fields.shape[2:]),
            torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=control_fields.device) / (kernel_size**3),
            padding=padding
        ).view(control_fields.shape)
        
        # Blend original and smoothed fields
        interference_strength = 0.1
        return (1 - interference_strength) * control_fields + interference_strength * smoothed_fields
    
    def _extract_well_observations(self, features):
        """
        Extract observations at specific well locations
        
        This method extracts features at well locations to predict observations,
        providing spatial accuracy consistent with well-specific control injection.
        
        Args:
            features: Spatial features (batch, Nx, Ny, Nz, width)
            
        Returns:
            observations: Well-specific observations (batch, 2*num_prod + num_inj)
        """
        import torch
        
        batch_size = features.shape[0]
        device = features.device
        
        # Initialize observation list
        well_observations = []
        
        # Extract injector BHP observations (at injector locations)
        for i, well_loc in enumerate(self.well_locations_injectors):
            well_x, well_y, well_z = well_loc[0].item(), well_loc[1].item(), well_loc[2].item()
            
            if self.z_penetration_mode == 'full':
                # Average features across all Z layers for penetrating wells
                well_features = torch.mean(features[:, well_x, well_y, :, :], dim=2)  # (batch, width)
            else:
                # Extract features at specific Z layer
                well_features = features[:, well_x, well_y, well_z, :]  # (batch, width)
            
            # Project to injector BHP observation
            well_obs = self.well_embeddings['injection'](well_features)  # Should output 1 value per injector
            well_observations.append(well_obs[:, i:i+1])  # Take only the i-th output for this well
        
        # Extract producer observations (BHP + water rate + gas rate at producer locations)
        for i, well_loc in enumerate(self.well_locations_producers):
            well_x, well_y, well_z = well_loc[0].item(), well_loc[1].item(), well_loc[2].item()
            
            if self.z_penetration_mode == 'full':
                # Average features across all Z layers for penetrating wells
                well_features = torch.mean(features[:, well_x, well_y, :, :], dim=2)  # (batch, width)
            else:
                # Extract features at specific Z layer
                well_features = features[:, well_x, well_y, well_z, :]  # (batch, width)
            
            # Project to producer observations (BHP + rates)
            well_obs = self.well_embeddings['production'](well_features)  # Should output 2 values per producer
            start_idx = i * 2
            end_idx = start_idx + 2
            well_observations.append(well_obs[:, start_idx:end_idx])  # Take BHP + rate for this well
        
        # Concatenate all observations: [Injector_BHP(3), Producer_observations(3*2)]
        observations = torch.cat(well_observations, dim=1)  # (batch, num_inj + num_prod*2)
        
        return observations

