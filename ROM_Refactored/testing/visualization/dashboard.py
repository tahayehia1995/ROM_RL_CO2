"""
Interactive Visualization Dashboard for E2C Model Testing
Provides interactive visualization of model predictions with metrics
"""

import h5py
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
import os
import json
from datetime import datetime
from pathlib import Path

_ROM_DIR = Path(__file__).resolve().parent.parent.parent

# Widget imports
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None

# Config loader
try:
    from utilities.config_loader import load_config
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    try:
        from config_loader import load_config
        CONFIG_LOADER_AVAILABLE = True
    except ImportError:
        CONFIG_LOADER_AVAILABLE = False
        load_config = None

# Import metrics
from .metrics import ModelEvaluationMetrics

# Helper function for formatting numbers to 3 significant digits
def format_3digits(x, pos):
    """Format number to 3 significant digits"""
    if x == 0:
        return '0'
    magnitude = np.floor(np.log10(np.abs(x)))
    factor = 10 ** (2 - magnitude)
    rounded = np.round(x * factor) / factor
    if abs(rounded) >= 1000:
        return f'{rounded:.0f}'
    elif abs(rounded) >= 100:
        return f'{rounded:.1f}'
    elif abs(rounded) >= 10:
        return f'{rounded:.2f}'
    else:
        return f'{rounded:.3f}'


class InteractiveVisualizationDashboard:
    """
    Interactive dashboard for visualizing E2C model predictions with case-specific masking
    Features:
    - Checkbox for inactive cell masking with file path input
    - Two tabs: Spatial predictions and Time series observations
    - Interactive sliders for navigation
    - Real-time plot updates
    - COMPARISON MODE: Side-by-side comparison of state-based vs latent-based predictions
    """
    
    def __init__(self, state_pred, state_seq_true_aligned, yobs_pred, yobs_seq_true, 
                 test_case_indices, norm_params, Nx, Ny, Nz, num_tstep=30, channel_names=None,
                 my_rom=None, test_controls=None, test_observations=None, device=None,
                 train_state_pred=None, train_state_seq_true_aligned=None, 
                 train_yobs_pred=None, train_yobs_seq_true=None, train_case_indices=None,
                 loaded_data=None):
        """
        Initialize the interactive dashboard
        
        Args:
            state_pred: Predicted state data (num_case, num_tstep, 3, Nx, Ny, Nz)
            state_seq_true_aligned: True state data (num_case, 3, num_tstep, Nx, Ny, Nz)  
            yobs_pred: Predicted observations (num_case, num_tstep, 9)
            yobs_seq_true: True observations (num_case, 9, num_tstep)
            test_case_indices: Array of test case indices
            norm_params: Dictionary of normalization parameters
            Nx, Ny, Nz: Grid dimensions
            num_tstep: Number of time steps
            channel_names: Names of state channels (optional)
            my_rom: ROM model for generating comparison predictions (optional)
            test_controls: Test control data for comparison predictions (optional)
            test_observations: Test observation data for comparison predictions (optional)
            device: Device for computation (optional)
            train_state_pred: Training predicted state data (optional)
            train_state_seq_true_aligned: Training true state data (optional)
            train_yobs_pred: Training predicted observations (optional)
            train_yobs_seq_true: Training true observations (optional)
            train_case_indices: Array of training case indices (optional)
            loaded_data: Dictionary containing loaded processed data for on-demand training prediction generation (optional)
        """
        # Check if interactive widgets are available
        if not WIDGETS_AVAILABLE:
            print("⚠️ Interactive widgets not available. Install ipywidgets: pip install ipywidgets")
            self.widgets_available = False
            return
        else:
            self.widgets_available = True
            
        # Store test data
        self.state_pred = state_pred
        self.state_seq_true_aligned = state_seq_true_aligned
        self.yobs_pred = yobs_pred  # Current predictions (default: state-based)
        self.yobs_seq_true = yobs_seq_true
        self.test_case_indices = test_case_indices
        self.norm_params = norm_params
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        
        # Auto-detect num_tstep from data shape if not provided or if default (24) is used
        # state_pred shape: (num_case, num_tstep, n_channels, Nx, Ny, Nz)
        # state_seq_true_aligned shape: (num_case, n_channels, num_tstep, Nx, Ny, Nz)
        if state_pred is not None and num_tstep == 24:  # Default value, try to detect from data
            detected_tstep = state_pred.shape[1] if len(state_pred.shape) > 1 else num_tstep
            if detected_tstep != num_tstep:
                print(f"📊 Auto-detected num_tstep={detected_tstep} from data shape (was {num_tstep})")
                num_tstep = detected_tstep
        elif state_seq_true_aligned is not None and num_tstep == 24:
            # Try to detect from true data shape: (num_case, n_channels, num_tstep, Nx, Ny, Nz)
            detected_tstep = state_seq_true_aligned.shape[2] if len(state_seq_true_aligned.shape) > 2 else num_tstep
            if detected_tstep != num_tstep:
                print(f"📊 Auto-detected num_tstep={detected_tstep} from true data shape (was {num_tstep})")
                num_tstep = detected_tstep
        
        self.num_tstep = num_tstep
        self.num_case = len(test_case_indices)
        
        # Store training data
        self.train_state_pred = train_state_pred
        self.train_state_seq_true_aligned = train_state_seq_true_aligned
        self.train_yobs_pred = train_yobs_pred
        self.train_yobs_seq_true = train_yobs_seq_true
        self.train_case_indices = train_case_indices
        self.num_train_case = len(train_case_indices) if train_case_indices is not None else 0
        
        # Store loaded_data reference for on-demand training prediction generation
        self.loaded_data = loaded_data
        
        # Store ROM and test data for comparison predictions
        self.my_rom = my_rom
        self.test_controls = test_controls
        self.test_observations = test_observations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store latest calculated metrics for saving
        self.latest_spatial_metrics = None
        self.latest_timeseries_metrics = None
        self.latest_selected_metrics = None
        self.latest_case_types = None
        
        # Cache for computed overall metrics to avoid recalculation
        self._overall_metrics_cache = {}
        self._cache_enabled = True
        
        # Cache for numpy conversions of training data (to avoid repeated GPU-to-CPU transfers)
        self._train_state_pred_np_cache = None
        self._train_state_true_np_cache = None
        self._train_yobs_pred_np_cache = None
        self._train_yobs_true_np_cache = None
        
        # Cache for numpy conversions of test data (for consistency)
        self._test_state_pred_np_cache = None
        self._test_state_true_np_cache = None
        self._test_yobs_pred_np_cache = None
        self._test_yobs_true_np_cache = None
        
        # Comparison mode variables
        self.comparison_mode_enabled = False
        self.yobs_pred_state_based = None
        self.yobs_pred_latent_based = None
        self.state_pred_latent_based = None
        self.predictions_generated = False
        self.spatial_latent_generated = False
        
        # Initialize evaluation metrics
        self.metrics_evaluator = ModelEvaluationMetrics(
            state_pred, state_seq_true_aligned, yobs_pred, yobs_seq_true, 
            channel_names=channel_names
        )
        # Set reference to dashboard for inactive cell masking
        self.metrics_evaluator.dashboard_ref = self
        
        # Ensure consistent inactive cell masking across all metrics calculations
        if hasattr(self, '_get_layer_mask'):
            print(f"🎭 Inactive cell masking system initialized and ready")
        
        # Masking variables
        self.masks_loaded_successfully = False
        self.mask_type = None          # 'case_specific', 'global', or None
        self.active_mask_3d = None     # Global 3D mask: (Nx, Ny, Nz)
        self.inactive_mask_3d = None   # Global 3D mask: (Nx, Ny, Nz)
        self.active_mask_4d = None     # Case-specific 4D mask: (cases, Nx, Ny, Nz)
        self.inactive_mask_4d = None   # Case-specific 4D mask: (cases, Nx, Ny, Nz)
        self.inactive_mask_path = ""
        
        # Define visualization parameters
        self.start_year = 2025
        # Use ALL timesteps and layers for full exploration
        self.all_timesteps = list(range(self.num_tstep))
        self.all_layers = list(range(self.Nz))
        self.all_years = [self.start_year + i for i in self.all_timesteps]
        
        # Store channel names
        self.channel_names = channel_names if channel_names else []
        
        # Field names and observation names - update based on channel selection
        if channel_names:
            # Use provided channel names for dynamic visualization
            self.field_keys = [name.upper() for name in channel_names]
            self.field_names = []
            for name in channel_names:
                name_upper = name.upper()
                if name_upper in ['SW', 'SWAT']:
                    self.field_names.append('Water Saturation')
                elif name_upper in ['SG', 'SGAS']:
                    self.field_names.append('Gas Saturation')
                elif name_upper in ['PRES', 'PRESSURE']:
                    self.field_names.append('Pressure')
                elif name_upper.startswith('PERM'):
                    self.field_names.append(f'Permeability {name_upper}')
                elif name_upper in ['PORO', 'POROSITY']:
                    self.field_names.append('Porosity')
                else:
                    self.field_names.append(name_upper.replace('_', ' ').title())
        else:
            # Default names for backward compatibility
            self.field_names = ['Water Saturation', 'Gas Saturation', 'Pressure']
            self.field_keys = ['SW', 'SG', 'PRES']
        self.obs_names = ['Inj1 BHP', 'Inj2 BHP', 'Inj3 BHP', 
                         'Prod1 Gas', 'Prod2 Gas', 'Prod3 Gas',
                         'Prod1 Water', 'Prod2 Water', 'Prod3 Water']
        self.obs_units = ['psi','psi','psi','ft3/day','ft3/day', 'ft3/day', 'ft3/day', 'ft3/day','ft3/day']
        
        # Load well locations from config
        self._load_well_locations()
        
        # Create widgets
        self._create_widgets()
        
    def _load_well_locations(self):
        """Load well locations from config file"""
        self.well_locations = {'injectors': {}, 'producers': {}}
        
        if CONFIG_LOADER_AVAILABLE:
            try:
                config = load_config('config.yaml')
                # Access well_locations under data section
                data_config = config.get('data', {})
                well_config = data_config.get('well_locations', {})
                
                if well_config:
                    # Extract injector locations
                    for well_name, coords in well_config.get('injectors', {}).items():
                        self.well_locations['injectors'][well_name] = coords
                    
                    # Extract producer locations
                    for well_name, coords in well_config.get('producers', {}).items():
                        self.well_locations['producers'][well_name] = coords
                        
                    print(f"✅ Loaded well locations: {len(self.well_locations['injectors'])} injectors, {len(self.well_locations['producers'])} producers")
                else:
                    print("⚠️ No well_locations found in config.yaml under data section")
            except Exception as e:
                print(f"⚠️ Could not load well locations from config: {e}")
        else:
            print("⚠️ Config loader not available - well locations will not be shown")
    
    def _create_widgets(self):
        """Create all interactive widgets"""
        # Inactive cell masking controls
        self.use_masking_checkbox = widgets.Checkbox(
            value=False,
            description='Enable Case-Specific Inactive Cell Masking',
            style={'description_width': 'initial'}
        )
        
        self.mask_file_text = widgets.Text(
            value='sr3_batch_output/inactive_cell_locations.h5',
            placeholder='Enter path to inactive cell file (.h5)',
            description='Mask File Path:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        self.load_mask_button = widgets.Button(
            description='Load Mask File',
            button_style='primary',
            icon='upload'
        )
        
        self.mask_status_label = widgets.Label(value='Mask Status: Not loaded')
        
        # Save predictions controls
        # Get default data directory from config if available
        default_data_dir = str(_ROM_DIR / 'sr3_batch_output')
        try:
            if CONFIG_LOADER_AVAILABLE:
                config = load_config(str(_ROM_DIR / 'config.yaml'))
                default_data_dir = config.get('paths.data_dir', str(_ROM_DIR / 'sr3_batch_output'))
        except:
            pass
        
        self.data_dir_text = widgets.Text(
            value=default_data_dir,
            placeholder='Enter path to data directory (sr3_batch_output)',
            description='Data Directory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        self.output_dir_text = widgets.Text(
            value=os.path.join(default_data_dir, 'predicted/'),
            placeholder='Enter output directory for predicted files',
            description='Output Directory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        self.save_predictions_button = widgets.Button(
            description='💾 Save Predictions',
            button_style='success',
            icon='save',
            layout=widgets.Layout(width='200px')
        )
        
        self.save_status_label = widgets.Label(value='Ready to save predictions')
        
        # CSV export controls - save a single case to CSV for visualization
        self.csv_case_dropdown = widgets.Dropdown(
            options=[(f"Test Case {i} (Sim #{self.test_case_indices[i]})", i) 
                     for i in range(self.num_case)],
            value=0,
            description='Select Case:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.csv_output_dir_text = widgets.Text(
            value=os.path.join(default_data_dir, 'csv_exports/'),
            placeholder='Enter output directory for CSV files',
            description='CSV Output Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        self.save_csv_button = widgets.Button(
            description='📄 Save Case CSV',
            button_style='info',
            icon='file-text',
            layout=widgets.Layout(width='200px')
        )
        
        self.csv_status_label = widgets.Label(value='Ready to export case CSV')
        self.csv_output = widgets.Output()
        
        # Tab selection
        self.tab_widget = widgets.Tab()
        
        # Spatial visualization controls
        self.spatial_case_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.num_case-1,
            step=1,
            description='Test Case (0 to {}):'.format(self.num_case-1),
            style={'description_width': 'initial'}
        )
        
        self.spatial_layer_slider = widgets.IntSlider(
            value=self.Nz//2,  # Start with middle layer
            min=0,
            max=self.Nz-1,  # Use ALL layers (0 to 24)
            step=1,
            description='Layer (0=Top, {}=Bottom):'.format(self.Nz-1),
            style={'description_width': 'initial'}
        )
        
        self.spatial_timestep_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.num_tstep-1,  # Use ALL timesteps (0 to num_tstep-1)
            step=1,
            description='Time Step (0={}, {}={}):'.format(self.start_year, self.num_tstep-1, self.start_year+self.num_tstep-1),
            style={'description_width': 'initial'}
        )
        
        self.spatial_field_dropdown = widgets.Dropdown(
            options=[(name, idx) for idx, name in enumerate(self.field_names)],
            value=0,
            description='Field:',
            style={'description_width': 'initial'}
        )
        
        self.spatial_latent_checkbox = widgets.Checkbox(
            value=False,
            description='🔬 Compare Spatial: State-based vs Latent-based',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        # Time series visualization controls
        self.timeseries_case_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.num_case-1,
            step=1,
            description='Test Case (0 to {}):'.format(self.num_case-1),
            style={'description_width': 'initial'}
        )
        
        self.timeseries_obs_dropdown = widgets.Dropdown(
            options=[(f"{name} ({unit})", idx) for idx, (name, unit) in enumerate(zip(self.obs_names, self.obs_units))],
            value=0,
            description='Observable:',
            style={'description_width': 'initial'}
        )
        
        # Comparison mode checkbox - only create if ROM model is available
        self.comparison_mode_checkbox = widgets.Checkbox(
            value=False,
            description='🔬 Compare State-based vs Latent-based Predictions',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            disabled=(self.my_rom is None)
        )
        
        # Checkbox to show/hide state-based predictions
        self.show_state_based_checkbox = widgets.Checkbox(
            value=True,
            description='👁️ Show State-based Predictions',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            disabled=(self.my_rom is None)
        )
        
        # Output areas
        self.spatial_output = widgets.Output()
        self.timeseries_output = widgets.Output()
        
        # Metrics output areas
        self.spatial_metrics_output = widgets.Output()
        self.timeseries_metrics_output = widgets.Output()
        
        # Overall performance metrics output area
        self.overall_metrics_output = widgets.Output()
        
        # Save predictions output area
        self.save_output = widgets.Output()
        
        # Set up event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Set up event handlers for all widgets"""
        self.load_mask_button.on_click(self._load_mask_file)
        self.save_predictions_button.on_click(self._save_predictions)
        self.save_csv_button.on_click(self._save_case_csv)
        
        # Masking checkbox handler
        self.use_masking_checkbox.observe(self._on_masking_toggle, names='value')
        
        # Spatial visualization handlers
        self.spatial_case_slider.observe(self._update_spatial_plot, names='value')
        self.spatial_layer_slider.observe(self._update_spatial_plot, names='value')
        self.spatial_timestep_slider.observe(self._update_spatial_plot, names='value')
        self.spatial_field_dropdown.observe(self._update_spatial_plot, names='value')
        self.spatial_latent_checkbox.observe(self._on_spatial_latent_toggle, names='value')
        
        # Time series visualization handlers
        self.timeseries_case_slider.observe(self._update_timeseries_plot, names='value')
        self.timeseries_obs_dropdown.observe(self._update_timeseries_plot, names='value')
        
        # Comparison mode handlers
        self.comparison_mode_checkbox.observe(self._on_comparison_mode_toggle, names='value')
        self.show_state_based_checkbox.observe(self._on_show_state_based_toggle, names='value')
        
        # Add handlers for metrics updates
        self.spatial_case_slider.observe(self._update_spatial_metrics, names='value')
        self.spatial_layer_slider.observe(self._update_spatial_metrics, names='value')
        self.spatial_timestep_slider.observe(self._update_spatial_metrics, names='value')
        self.spatial_field_dropdown.observe(self._update_spatial_metrics, names='value')
        
        self.timeseries_case_slider.observe(self._update_timeseries_metrics, names='value')
        self.timeseries_obs_dropdown.observe(self._update_timeseries_metrics, names='value')
    
    def _load_mask_file(self, button):
        """Load inactive cell mask file with support for case-specific masks"""
        try:
            mask_path = self.mask_file_text.value
            with h5py.File(mask_path, 'r') as f:
                # Load mask data and determine structure
                active_mask_raw = f['active_mask'][...]
                inactive_mask_raw = f['inactive_mask'][...]
                
                # Handle both 3D global masks and 4D case-specific masks
                if len(active_mask_raw.shape) == 4:  # Case-specific: (cases, Nx, Ny, Nz)
                    self.mask_type = 'case_specific'
                    self.active_mask_4d = active_mask_raw
                    self.inactive_mask_4d = inactive_mask_raw
                    num_mask_cases, Nx_mask, Ny_mask, Nz_mask = active_mask_raw.shape
                    
                    # Validate mask dimensions match spatial data
                    if (Nx_mask != self.Nx) or (Ny_mask != self.Ny) or (Nz_mask != self.Nz):
                        raise ValueError(f"Mask grid ({Nx_mask}×{Ny_mask}×{Nz_mask}) doesn't match spatial data grid ({self.Nx}×{self.Ny}×{self.Nz})")
                    
                elif len(active_mask_raw.shape) == 3:  # Global: (Nx, Ny, Nz)
                    self.mask_type = 'global'
                    self.active_mask_3d = active_mask_raw
                    self.inactive_mask_3d = inactive_mask_raw
                    Nx_mask, Ny_mask, Nz_mask = active_mask_raw.shape
                    
                    # Validate mask dimensions match spatial data
                    if (Nx_mask != self.Nx) or (Ny_mask != self.Ny) or (Nz_mask != self.Nz):
                        raise ValueError(f"Mask grid ({Nx_mask}×{Ny_mask}×{Nz_mask}) doesn't match spatial data grid ({self.Nx}×{self.Ny}×{self.Nz})")
                    
                else:
                    raise ValueError(f"Unsupported mask dimensions: {active_mask_raw.shape}. Expected 3D (Nx,Ny,Nz) or 4D (cases,Nx,Ny,Nz)")
                
            self.masks_loaded_successfully = True
            
            # Update status message
            if self.mask_type == 'case_specific':
                self.mask_status_label.value = f'Mask Status: ✅ Case-specific masks loaded ({num_mask_cases} cases, {Nx_mask}×{Ny_mask}×{Nz_mask})'
            else:
                self.mask_status_label.value = f'Mask Status: ✅ Global mask loaded ({Nx_mask}×{Ny_mask}×{Nz_mask}, {active_cells} active cells)'
            
            # Update plots if masking is enabled
            if self.use_masking_checkbox.value:
                self._update_spatial_plot()
                self._update_timeseries_plot()
                
        except Exception as e:
            self.masks_loaded_successfully = False
            self.mask_status_label.value = f'Mask Status: ❌ Error loading: {str(e)}'
            # Reset mask type and data
            self.mask_type = None
            if hasattr(self, 'active_mask_4d'):
                delattr(self, 'active_mask_4d')
            if hasattr(self, 'inactive_mask_4d'):
                delattr(self, 'inactive_mask_4d')
            if hasattr(self, 'active_mask_3d'):
                delattr(self, 'active_mask_3d')
            if hasattr(self, 'inactive_mask_3d'):
                delattr(self, 'inactive_mask_3d')
            
    def _on_spatial_latent_toggle(self, change):
        """Handle spatial latent comparison checkbox toggle."""
        if change['new'] and not self.spatial_latent_generated:
            self._generate_latent_spatial_predictions()
        if change['new'] and not self.spatial_latent_generated:
            self.spatial_latent_checkbox.value = False
            return
        self._update_spatial_plot()

    def _generate_latent_spatial_predictions(self):
        """Generate spatial predictions via pure latent rollout + decode at each step."""
        with self.spatial_output:
            clear_output(wait=True)

            if self.my_rom is None or self.test_controls is None:
                print("❌ Cannot generate latent spatial predictions — ROM model or controls not available.")
                return

            try:
                import time as _time
                model = self.my_rom.model
                print("🔬 Generating latent-based spatial predictions (decode at each step)...")
                print(f"   Cases: {self.num_case}, Timesteps: {self.num_tstep}")

                t_steps = np.arange(0, 200, 200 // self.num_tstep)
                dt = 10
                t_steps1 = (t_steps + dt).astype(int)
                indt_del = t_steps1 - t_steps
                indt_del = indt_del / max(indt_del)

                is_multimodal = hasattr(model, 'static_encoder') and not hasattr(model, 'graph_manager')
                is_gnn = hasattr(model, 'graph_manager')
                has_encode_initial = hasattr(model, 'encode_initial')

                state_pred_latent = torch.zeros_like(self.state_pred)

                model.eval()
                t_start = _time.perf_counter()
                with torch.no_grad():
                    initial_spatial = self.state_pred[:, 0, :, :, :, :].to(self.device)

                    if has_encode_initial:
                        latent_state = model.encode_initial(initial_spatial)
                    else:
                        enc_out = model.encoder(initial_spatial)
                        latent_state = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out

                    if is_multimodal or is_gnn:
                        static_dim = model.static_latent_dim
                        x_static_ref = initial_spatial[:, model.static_channels, :, :, :]
                    else:
                        static_dim = 0

                    def _decode_latent(z):
                        if is_multimodal or is_gnn:
                            z_s = z[:, :static_dim]
                            z_d = z[:, static_dim:]
                            x_dyn = model.decoder(torch.cat([z_s, z_d], dim=-1))
                            return model._reassemble(x_dyn, x_static_ref)
                        return model.decoder(z)

                    state_pred_latent[:, 0, :, :, :, :] = initial_spatial.cpu()

                    for i_tstep in range(self.num_tstep - 1):
                        dt_seq = torch.tensor(
                            np.ones((self.num_case, 1)) * indt_del[i_tstep],
                            dtype=torch.float32
                        ).to(self.device)
                        controls = self.test_controls[:, :, i_tstep].to(self.device)

                        latent_state, _ = model.predict_latent(latent_state, dt_seq, controls)
                        state_pred_latent[:, i_tstep + 1, :, :, :, :] = _decode_latent(latent_state).cpu()

                        if (i_tstep + 1) % 5 == 0:
                            print(f"   Step {i_tstep + 1}/{self.num_tstep - 1} completed")

                latent_total = _time.perf_counter() - t_start
                latent_per_case = latent_total / max(self.num_case, 1)

                self.state_pred_latent_based = state_pred_latent
                self.spatial_latent_generated = True
                self.spatial_latent_checkbox.description = '🔬 Compare Spatial: State-based vs Latent-based'
                print("✅ Latent-based spatial predictions generated successfully!")

                # Benchmark state-based prediction for timing comparison
                sb_start = _time.perf_counter()
                with torch.no_grad():
                    sb_state = self.state_pred[:, 0, :, :, :, :].to(self.device)
                    for i_t in range(self.num_tstep):
                        dt_s = torch.tensor(
                            np.ones((self.num_case, 1)) * indt_del[i_t],
                            dtype=torch.float32).to(self.device)
                        ctrls = self.test_controls[:, :, i_t].to(self.device)
                        obs_ph = torch.zeros(self.num_case, 9).to(self.device)
                        sb_state, _ = self.my_rom.predict((sb_state, ctrls, obs_ph, dt_s))
                sb_total = _time.perf_counter() - sb_start
                sb_per_case = sb_total / max(self.num_case, 1)

                speedup = sb_total / max(latent_total, 1e-6)
                self._spatial_timing_info = (
                    f"State-based: {sb_total:.2f}s ({sb_per_case:.4f}s/case) | "
                    f"Latent-based: {latent_total:.2f}s ({latent_per_case:.4f}s/case) | "
                    f"Speedup: {speedup:.1f}x"
                )
                print(f"\n⏱️  {self._spatial_timing_info}")

            except Exception as e:
                print(f"❌ Error generating latent spatial predictions: {e}")
                import traceback
                traceback.print_exc()
                self.spatial_latent_generated = False
                self.spatial_latent_checkbox.description = f'🔬 Spatial Compare — ❌ {str(e)[:60]}'

    def _on_masking_toggle(self, change):
        """Handle masking checkbox toggle"""
        self._update_spatial_plot()
        if hasattr(self, 'timeseries_output'):
            # Time series doesn't use masking but update for consistency
            pass
    
    def _on_comparison_mode_toggle(self, change):
        """Handle comparison mode checkbox toggle"""
        self.comparison_mode_enabled = change['new']
        
        if self.comparison_mode_enabled:
            if not self.predictions_generated:
                self._generate_comparison_predictions()
            
            if not self.predictions_generated:
                self.comparison_mode_enabled = False
                self.comparison_mode_checkbox.value = False
                return
        
        self._update_timeseries_plot()
    
    def _on_show_state_based_toggle(self, change):
        """Handle show state-based predictions checkbox toggle"""
        if self.comparison_mode_enabled:
            self._update_timeseries_plot()
    
    def _generate_comparison_predictions(self):
        """Generate both state-based and latent-based predictions for comparison"""
        with self.timeseries_output:
            clear_output(wait=True)
            
            if self.my_rom is None or self.test_controls is None or self.test_observations is None:
                missing = []
                if self.my_rom is None:
                    missing.append("ROM model")
                if self.test_controls is None:
                    missing.append("test controls")
                if self.test_observations is None:
                    missing.append("test observations")
                error_msg = f"Cannot generate comparison — missing: {', '.join(missing)}"
                print(f"❌ {error_msg}")
                self.comparison_mode_checkbox.description = f'🔬 Compare — ❌ {error_msg}'
                return
            
            try:
                print("🔬 Generating comparison predictions...")
                print(f"   Cases: {self.num_case}, Timesteps: {self.num_tstep}")
                
                self.yobs_pred_state_based = self.yobs_pred.clone()
                
                print("   ⚡ Generating latent-based predictions...")
                self.yobs_pred_latent_based = self._generate_latent_predictions()
                
                self.predictions_generated = True
                self.comparison_mode_checkbox.description = '🔬 Compare State-based vs Latent-based Predictions'
                print("✅ Both prediction types generated successfully!")
                
            except Exception as e:
                print(f"❌ Error generating comparison predictions: {e}")
                import traceback
                traceback.print_exc()
                self.predictions_generated = False
                self.comparison_mode_checkbox.description = f'🔬 Compare — ❌ Error: {str(e)[:60]}'
    
    def _generate_latent_predictions(self):
        """Generate predictions using pure latent-space rollout.

        Unified protocol for ALL model types (base MSE2C, Multimodal, GNN):
        encode the initial spatial state once, then iterate the transition
        model purely in latent space — no decode/re-encode per step.

        This mirrors the RL latent-based optimization loop and gives a fair
        apples-to-apples comparison with the state-based autoregressive mode.
        """
        yobs_pred_latent = torch.zeros_like(self.yobs_pred)

        t_steps = np.arange(0, 200, 200 // self.num_tstep)
        dt = 10
        t_steps1 = (t_steps + dt).astype(int)
        indt_del = t_steps1 - t_steps
        indt_del = indt_del / max(indt_del)

        model = self.my_rom.model
        has_encode_initial = hasattr(model, 'encode_initial')
        model_type = ('GNN' if hasattr(model, 'graph_manager')
                       else 'Multimodal' if hasattr(model, 'static_encoder')
                       else 'Base MSE2C')

        print(f"   Model type: {model_type}")
        print(f"   Using unified latent-space rollout for all models")

        model.eval()
        with torch.no_grad():
            initial_spatial = self.state_pred[:, 0, :, :, :, :].to(self.device)

            if has_encode_initial:
                latent_state = model.encode_initial(initial_spatial)
            else:
                enc_out = model.encoder(initial_spatial)
                latent_state = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out

            for i_tstep in range(self.num_tstep):
                dt_seq = torch.tensor(
                    np.ones((self.num_case, 1)) * indt_del[i_tstep],
                    dtype=torch.float32
                ).to(self.device)
                controls = self.test_controls[:, :, i_tstep].to(self.device)

                latent_state, yobs = model.predict_latent(
                    latent_state, dt_seq, controls
                )
                yobs_pred_latent[:, i_tstep, :] = yobs.cpu()

                if (i_tstep + 1) % 5 == 0:
                    print(f"   Step {i_tstep + 1}/{self.num_tstep} completed")

        return yobs_pred_latent
            
    def _get_layer_mask(self, case_idx, layer_idx, use_training_data=False):
        """
        Get 2D layer mask for specific case and layer.
        
        Args:
            case_idx: Case index (index into test_case_indices or train_case_indices, depending on use_training_data)
            layer_idx: Layer index
            use_training_data: If True, case_idx is an index into train_case_indices; if False, into test_case_indices
        
        Returns:
            2D boolean mask array (Nx, Ny)
        """
        if self.use_masking_checkbox.value and self.masks_loaded_successfully:
            try:
                if self.mask_type == 'case_specific':
                    # Use case-specific mask: (cases, Nx, Ny, Nz)
                    # Map case index to actual case number in the mask data
                    if use_training_data:
                        # For training: case_idx is index into train_case_indices
                        if self.train_case_indices is not None and case_idx < len(self.train_case_indices):
                            actual_case_idx = self.train_case_indices[case_idx]
                        else:
                            # Fallback: assume case_idx is already the actual case number
                            actual_case_idx = case_idx
                    else:
                        # For test: case_idx is index into test_case_indices
                        if case_idx < len(self.test_case_indices):
                            actual_case_idx = self.test_case_indices[case_idx]
                        else:
                            # Fallback: assume case_idx is already the actual case number
                            actual_case_idx = case_idx
                    
                    # Validate case index bounds
                    if actual_case_idx >= self.active_mask_4d.shape[0]:
                        print(f"⚠️ Warning: Case {actual_case_idx} not in mask data (max: {self.active_mask_4d.shape[0]-1}). Using last available mask.")
                        mask_case_idx = self.active_mask_4d.shape[0] - 1
                    else:
                        mask_case_idx = actual_case_idx
                    
                    # Extract 2D layer mask
                    layer_mask_2d = self.active_mask_4d[mask_case_idx, :, :, layer_idx]  # Shape: (Nx, Ny)
                    
                    return layer_mask_2d
                    
                elif self.mask_type == 'global':
                    # Use global mask for all cases: (Nx, Ny, Nz)
                    layer_mask_2d = self.active_mask_3d[:, :, layer_idx]  # Shape: (Nx, Ny)
                    return layer_mask_2d
                    
                else:
                    return np.ones((self.Nx, self.Ny), dtype=bool)
                    
            except Exception as e:
                return np.ones((self.Nx, self.Ny), dtype=bool)
        
        # If no mask loaded or masking disabled, return all active (no masking)
        return np.ones((self.Nx, self.Ny), dtype=bool)
    
    def _get_all_layer_masks_vectorized(self, case_indices, use_training_data=False):
        """
        Get all layer masks for all cases and layers in one vectorized operation.
        This is much faster than calling _get_layer_mask() in a loop.
        
        Args:
            case_indices: Array of case indices (can be indices into test_case_indices or train_case_indices)
            use_training_data: If True, case_indices are indices into train_case_indices; if False, into test_case_indices
        
        Returns:
            4D boolean mask array: (n_cases, Nx, Ny, Nz)
        """
        if not (hasattr(self, 'use_masking_checkbox') and self.use_masking_checkbox.value and 
                hasattr(self, 'masks_loaded_successfully') and self.masks_loaded_successfully):
            # No masking: return all active masks
            n_cases = len(case_indices) if case_indices is not None else 1
            return np.ones((n_cases, self.Nx, self.Ny, self.Nz), dtype=bool)
        
        try:
            n_cases = len(case_indices) if case_indices is not None else 1
            
            if self.mask_type == 'case_specific':
                # Use case-specific mask: (cases, Nx, Ny, Nz)
                case_indices_array = np.array(case_indices)
                max_mask_idx = self.active_mask_4d.shape[0] - 1
                
                # Check if case_indices are already actual case numbers (within mask bounds)
                # If max(case_indices) <= max_mask_idx, they're likely already actual case numbers
                # Otherwise, they might be indices into train/test_case_indices that need mapping
                if len(case_indices_array) > 0 and np.max(case_indices_array) <= max_mask_idx:
                    # case_indices are already actual case numbers - use them directly
                    actual_case_indices = case_indices_array
                else:
                    # case_indices are indices into train/test_case_indices - need to map them
                    if use_training_data:
                        # For training: case_indices are indices into train_case_indices
                        if self.train_case_indices is not None and len(case_indices) > 0:
                            # Fully vectorized mapping using advanced indexing
                            train_indices_array = np.array(self.train_case_indices)
                            # Use advanced indexing: get actual case indices for all cases at once
                            valid_mask = case_indices_array < len(train_indices_array)
                            actual_case_indices = np.where(valid_mask, 
                                                           train_indices_array[case_indices_array], 
                                                           case_indices_array)
                        else:
                            # Fallback: assume case_indices are already actual case numbers
                            actual_case_indices = case_indices_array
                    else:
                        # For test: case_indices are indices into test_case_indices
                        if hasattr(self, 'test_case_indices') and self.test_case_indices is not None and len(case_indices) > 0:
                            # Fully vectorized mapping using advanced indexing
                            test_indices_array = np.array(self.test_case_indices)
                            # Use advanced indexing: get actual case indices for all cases at once
                            valid_mask = case_indices_array < len(test_indices_array)
                            actual_case_indices = np.where(valid_mask,
                                                           test_indices_array[case_indices_array],
                                                           case_indices_array)
                        else:
                            # Fallback: assume case_indices are already actual case numbers
                            actual_case_indices = case_indices_array
                
                # Validate case index bounds (vectorized) - ensure all indices are within valid range
                actual_case_indices = np.clip(actual_case_indices, 0, max_mask_idx)
                
                # Extract all masks at once using advanced indexing: (n_cases, Nx, Ny, Nz)
                all_masks = self.active_mask_4d[actual_case_indices, :, :, :]
                
                return all_masks
                
            elif self.mask_type == 'global':
                # Use global mask for all cases: (Nx, Ny, Nz)
                # Broadcast the same mask to all cases
                global_mask = self.active_mask_3d  # Shape: (Nx, Ny, Nz)
                # Broadcast to (n_cases, Nx, Ny, Nz)
                all_masks = np.broadcast_to(global_mask[np.newaxis, :, :, :], (n_cases, self.Nx, self.Ny, self.Nz)).copy()
                
                return all_masks
                
            else:
                # Unknown mask type: return all active masks
                return np.ones((n_cases, self.Nx, self.Ny, self.Nz), dtype=bool)
                
        except Exception as e:
            # Fallback: return all active masks
            n_cases = len(case_indices) if case_indices is not None else 1
            return np.ones((n_cases, self.Nx, self.Ny, self.Nz), dtype=bool)
        
    def _denormalize_field_data(self, data, field_key):
        """Denormalize field data using stored parameters"""
        if field_key not in self.norm_params:
            print(f"⚠️ Warning: No normalization parameters found for {field_key}")
            return data
            
        norm_params = self.norm_params[field_key]
        
        if norm_params.get('type') == 'none':
            # Data was not normalized, return as-is
            return data
        elif norm_params.get('type') == 'log':
            # Reverse log normalization
            # Step 1: Reverse min-max scaling of log data
            log_min = norm_params['log_min']
            log_max = norm_params['log_max']
            log_data = data * (log_max - log_min) + log_min
            
            # Step 2: Reverse log transform
            exp_data = np.exp(log_data)
            
            # Step 3: Reverse data shifting
            epsilon = norm_params.get('epsilon', 1e-8)
            data_shift = norm_params.get('data_shift', 0)
            original_data = exp_data - epsilon + data_shift
            
            return original_data
        else:
            # Standard min-max denormalization
            field_min = norm_params['min']
            field_max = norm_params['max']
            return data * (field_max - field_min) + field_min
        
    def _denormalize_obs_data(self, data, obs_idx):
        """Denormalize observation data based on observation type with support for 'none' normalization"""
        if obs_idx < 3:  # BHP data
            if 'BHP' in self.norm_params:
                norm_params = self.norm_params['BHP']
                if norm_params.get('type') == 'none':
                    return data  # No normalization was applied, return as-is
                elif norm_params.get('type') == 'log':
                    # Reverse log normalization
                    log_min = norm_params['log_min']
                    log_max = norm_params['log_max']
                    log_data = data * (log_max - log_min) + log_min
                    epsilon = norm_params.get('epsilon', 1e-8)
                    data_shift = norm_params.get('data_shift', 0)
                    return np.exp(log_data) - epsilon + data_shift
                else:
                    # Standard min-max denormalization
                    obs_min = norm_params['min']
                    obs_max = norm_params['max']
                    return data * (obs_max - obs_min) + obs_min
        elif obs_idx < 6:  # Gas production (indices 3-5)
            if 'GASRATSC' in self.norm_params:
                norm_params = self.norm_params['GASRATSC']
                if norm_params.get('type') == 'none':
                    return data  # No normalization was applied, return as-is
                elif norm_params.get('type') == 'log':
                    # Reverse log normalization
                    log_min = norm_params['log_min']
                    log_max = norm_params['log_max']
                    log_data = data * (log_max - log_min) + log_min
                    epsilon = norm_params.get('epsilon', 1e-8)
                    data_shift = norm_params.get('data_shift', 0)
                    return np.exp(log_data) - epsilon + data_shift
                else:
                    # Standard min-max denormalization
                    obs_min = norm_params['min'] 
                    obs_max = norm_params['max']
                    return data * (obs_max - obs_min) + obs_min
        else:  # Water production (indices 6-8)
            if 'WATRATSC' in self.norm_params:
                norm_params = self.norm_params['WATRATSC']
                if norm_params.get('type') == 'none':
                    return data  # No normalization was applied, return as-is
                elif norm_params.get('type') == 'log':
                    # Reverse log normalization
                    log_min = norm_params['log_min']
                    log_max = norm_params['log_max']
                    log_data = data * (log_max - log_min) + log_min
                    epsilon = norm_params.get('epsilon', 1e-8)
                    data_shift = norm_params.get('data_shift', 0)
                    return np.exp(log_data) - epsilon + data_shift
                else:
                    # Standard min-max denormalization
                    obs_min = norm_params['min']
                    obs_max = norm_params['max']
                    return data * (obs_max - obs_min) + obs_min
            
        return data  # Fallback: return data as-is if no normalization params found
    
    def _save_predictions(self, button):
        """Save all denormalized predictions to H5 files matching sr3_batch_output structure"""
        self.save_status_label.value = '💾 Saving...'
        with self.save_output:
            clear_output(wait=True)
            print("💾 Saving predictions...")
            
            try:
                # Get directories from widgets
                data_dir = self.data_dir_text.value.strip()
                output_dir = self.output_dir_text.value.strip()
                
                # Ensure directories end with /
                if not data_dir.endswith('/'):
                    data_dir += '/'
                if not output_dir.endswith('/'):
                    output_dir += '/'
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Validate channel_names
                if not hasattr(self, 'channel_names') or not self.channel_names:
                    print("❌ Error: Channel names not available. Cannot save predictions.")
                    self.save_status_label.value = '❌ Error: Channel names missing'
                    return
                
                # Get total number of cases by loading one original spatial file
                total_cases = None
                first_spatial_file = None
                for var_name in self.channel_names:
                    filename = f"batch_spatial_properties_{var_name}.h5"
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        first_spatial_file = filepath
                        with h5py.File(filepath, 'r') as hf:
                            original_data = np.array(hf['data'])
                            total_cases = original_data.shape[0]
                            num_timesteps_original = original_data.shape[1]
                        print(f"📊 Found {total_cases} total cases in original data")
                        break
                
                if total_cases is None:
                    print(f"❌ Error: Could not find any original spatial files in {data_dir}")
                    print(f"   Looking for files like: batch_spatial_properties_{self.channel_names[0]}.h5")
                    self.save_status_label.value = '❌ Error: Original files not found'
                    return
                
                saved_files = []
                
                # Save spatial predictions for each channel
                print(f"\n📁 Saving spatial predictions...")
                for channel_idx, var_name in enumerate(self.channel_names):
                    # Extract channel data: (num_case, num_tstep, Nx, Ny, Nz)
                    channel_data = self.state_pred[:, :, channel_idx, :, :, :].cpu().detach().numpy()
                    
                    # Denormalize the data
                    denormalized_data = np.zeros_like(channel_data)
                    for case_idx in range(channel_data.shape[0]):
                        for timestep_idx in range(channel_data.shape[1]):
                            field_data = channel_data[case_idx, timestep_idx, :, :, :]
                            denormalized_field = self._denormalize_field_data(field_data, var_name)
                            denormalized_data[case_idx, timestep_idx, :, :, :] = denormalized_field
                    
                    # Create full array for all cases (fill non-predicted with zeros)
                    # Original format: (total_cases, num_timesteps, Nx, Ny, Nz)
                    # We need to match the number of timesteps in original files
                    full_data = np.zeros((total_cases, num_timesteps_original, self.Nx, self.Ny, self.Nz), dtype=denormalized_data.dtype)
                    
                    # Place predictions at the correct case indices
                    test_indices = np.array(self.test_case_indices)
                    num_pred_timesteps = min(denormalized_data.shape[1], num_timesteps_original)
                    # Use advanced indexing to place data correctly
                    for i, case_idx in enumerate(test_indices):
                        full_data[case_idx, :num_pred_timesteps, :, :, :] = denormalized_data[i, :num_pred_timesteps, :, :, :]
                    
                    # Save to H5 file
                    output_filename = f"batch_spatial_properties_{var_name}_predicted.h5"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with h5py.File(output_path, 'w') as hf:
                        hf.create_dataset('data', data=full_data)
                    
                    saved_files.append(output_filename)
                    print(f"  ✅ Saved {output_filename} (shape: {full_data.shape})")
                
                # Save observation predictions
                print(f"\n📈 Saving observation predictions...")
                
                # BHP observations (indices 0-2)
                if 'BHP' in self.norm_params:
                    bhp_pred = self.yobs_pred[:, :, 0:3].cpu().detach().numpy()  # (num_case, num_tstep, 3)
                    
                    # Denormalize
                    denormalized_bhp = np.zeros_like(bhp_pred)
                    for case_idx in range(bhp_pred.shape[0]):
                        for timestep_idx in range(bhp_pred.shape[1]):
                            for well_idx in range(3):
                                obs_data = bhp_pred[case_idx, timestep_idx, well_idx]
                                denormalized_obs = self._denormalize_obs_data(obs_data, well_idx)
                                denormalized_bhp[case_idx, timestep_idx, well_idx] = denormalized_obs
                    
                    # Create full array
                    full_bhp = np.zeros((total_cases, num_timesteps_original, 3), dtype=denormalized_bhp.dtype)
                    test_indices = np.array(self.test_case_indices)
                    num_pred_timesteps = min(denormalized_bhp.shape[1], num_timesteps_original)
                    # Use advanced indexing to place data correctly
                    for i, case_idx in enumerate(test_indices):
                        full_bhp[case_idx, :num_pred_timesteps, :] = denormalized_bhp[i, :num_pred_timesteps, :]
                    
                    # Save BHP
                    output_filename = f"batch_timeseries_data_BHP_predicted.h5"
                    output_path = os.path.join(output_dir, output_filename)
                    with h5py.File(output_path, 'w') as hf:
                        hf.create_dataset('data', data=full_bhp)
                    saved_files.append(output_filename)
                    print(f"  ✅ Saved {output_filename} (shape: {full_bhp.shape})")
                
                # GASRATSC observations (indices 3-5)
                if 'GASRATSC' in self.norm_params:
                    gas_pred = self.yobs_pred[:, :, 3:6].cpu().detach().numpy()  # (num_case, num_tstep, 3)
                    
                    # Denormalize
                    denormalized_gas = np.zeros_like(gas_pred)
                    for case_idx in range(gas_pred.shape[0]):
                        for timestep_idx in range(gas_pred.shape[1]):
                            for well_idx in range(3):
                                obs_data = gas_pred[case_idx, timestep_idx, well_idx]
                                denormalized_obs = self._denormalize_obs_data(obs_data, 3 + well_idx)
                                denormalized_gas[case_idx, timestep_idx, well_idx] = denormalized_obs
                    
                    # Create full array
                    full_gas = np.zeros((total_cases, num_timesteps_original, 3), dtype=denormalized_gas.dtype)
                    test_indices = np.array(self.test_case_indices)
                    num_pred_timesteps = min(denormalized_gas.shape[1], num_timesteps_original)
                    # Use advanced indexing to place data correctly
                    for i, case_idx in enumerate(test_indices):
                        full_gas[case_idx, :num_pred_timesteps, :] = denormalized_gas[i, :num_pred_timesteps, :]
                    
                    # Save GASRATSC
                    output_filename = f"batch_timeseries_data_GASRATSC_predicted.h5"
                    output_path = os.path.join(output_dir, output_filename)
                    with h5py.File(output_path, 'w') as hf:
                        hf.create_dataset('data', data=full_gas)
                    saved_files.append(output_filename)
                    print(f"  ✅ Saved {output_filename} (shape: {full_gas.shape})")
                
                # WATRATSC observations (indices 6-8)
                if 'WATRATSC' in self.norm_params:
                    wat_pred = self.yobs_pred[:, :, 6:9].cpu().detach().numpy()  # (num_case, num_tstep, 3)
                    
                    # Denormalize
                    denormalized_wat = np.zeros_like(wat_pred)
                    for case_idx in range(wat_pred.shape[0]):
                        for timestep_idx in range(wat_pred.shape[1]):
                            for well_idx in range(3):
                                obs_data = wat_pred[case_idx, timestep_idx, well_idx]
                                denormalized_obs = self._denormalize_obs_data(obs_data, 6 + well_idx)
                                denormalized_wat[case_idx, timestep_idx, well_idx] = denormalized_obs
                    
                    # Create full array
                    full_wat = np.zeros((total_cases, num_timesteps_original, 3), dtype=denormalized_wat.dtype)
                    test_indices = np.array(self.test_case_indices)
                    num_pred_timesteps = min(denormalized_wat.shape[1], num_timesteps_original)
                    # Use advanced indexing to place data correctly
                    for i, case_idx in enumerate(test_indices):
                        full_wat[case_idx, :num_pred_timesteps, :] = denormalized_wat[i, :num_pred_timesteps, :]
                    
                    # Save WATRATSC
                    output_filename = f"batch_timeseries_data_WATRATSC_predicted.h5"
                    output_path = os.path.join(output_dir, output_filename)
                    with h5py.File(output_path, 'w') as hf:
                        hf.create_dataset('data', data=full_wat)
                    saved_files.append(output_filename)
                    print(f"  ✅ Saved {output_filename} (shape: {full_wat.shape})")
                
                # Success message
                print(f"\n✅ Successfully saved {len(saved_files)} files to {output_dir}")
                print(f"📋 Saved files:")
                for filename in saved_files:
                    print(f"   • {filename}")
                
                self.save_status_label.value = f'✅ Saved {len(saved_files)} files successfully'
                
            except Exception as e:
                print(f"❌ Error saving predictions: {e}")
                import traceback
                traceback.print_exc()
                self.save_status_label.value = f'❌ Error: {str(e)}'
        
    def _save_case_csv(self, button):
        """Save a single simulation case's predictions to CSV files for visualization.
        
        Produces two CSV files:
        1. Timeseries CSV: rows=timesteps, columns=well observations (BHP, Gas, Water per well)
        2. Spatial CSV: rows=timesteps, columns=Channel_i_j_k for every (channel, cell)
        Both files contain denormalized predicted values with negatives clipped to zero.
        """
        import csv
        
        self.csv_status_label.value = '📄 Saving CSV...'
        with self.csv_output:
            clear_output(wait=True)
            
            try:
                # Get selected case and output directory
                case_idx = self.csv_case_dropdown.value
                actual_case_id = self.test_case_indices[case_idx]
                output_dir = self.csv_output_dir_text.value.strip()
                if not output_dir.endswith('/') and not output_dir.endswith('\\'):
                    output_dir += '/'
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"📄 Exporting Case {case_idx} (Simulation #{actual_case_id}) to CSV...")
                
                saved_files = []
                years = self.all_years  # e.g. [2025, 2026, ..., 2054]
                
                # ── 1. Timeseries (wells) CSV ──────────────────────────────
                print(f"\n📈 Saving timeseries (wells) predictions...")
                
                ts_filename = f"case_{actual_case_id}_timeseries.csv"
                ts_path = os.path.join(output_dir, ts_filename)
                
                ts_header = ['Year']
                for name in self.obs_names:
                    ts_header.append(name)
                
                yobs_pred_case = self.yobs_pred[case_idx].cpu().detach().numpy()   # (num_tstep, 9)
                
                with open(ts_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(ts_header)
                    
                    for t in range(self.num_tstep):
                        row = [years[t]]
                        for obs_idx in range(9):
                            pred_val = float(self._denormalize_obs_data(
                                yobs_pred_case[t, obs_idx], obs_idx))
                            row.append(max(0.0, pred_val))
                        writer.writerow(row)
                
                saved_files.append(ts_filename)
                print(f"  ✅ Saved {ts_filename}  ({self.num_tstep} timesteps × {len(self.obs_names)} wells)")
                
                # ── 2. Spatial (channels) CSV ──────────────────────────────
                # One row per timestep; spatial cells flattened into columns
                print(f"\n🗺️  Saving spatial (channels) predictions...")
                
                sp_filename = f"case_{actual_case_id}_spatial.csv"
                sp_path = os.path.join(output_dir, sp_filename)
                
                n_channels = len(self.channel_names)
                n_cells = self.Nx * self.Ny * self.Nz
                
                # Build header: Year, Channel_i_j_k for every (channel, i, j, k)
                sp_header = ['Year']
                for ch_name in self.channel_names:
                    for ix in range(self.Nx):
                        for iy in range(self.Ny):
                            for iz in range(self.Nz):
                                sp_header.append(f"{ch_name}_{ix}_{iy}_{iz}")
                
                print(f"  Grid: {self.Nx}×{self.Ny}×{self.Nz} = {n_cells} cells × {n_channels} channels = {n_cells * n_channels} columns per timestep")
                
                # Pre-extract and denormalize all channel data, clip negatives
                pred_channels = []   # list of (num_tstep, Nx, Ny, Nz) arrays
                
                for ch_idx, ch_name in enumerate(self.channel_names):
                    field_key = self.field_keys[ch_idx]
                    
                    pred_raw = self.state_pred[case_idx, :, ch_idx, :, :, :].cpu().detach().numpy()
                    
                    pred_denorm = np.zeros_like(pred_raw)
                    for t in range(self.num_tstep):
                        pred_denorm[t] = self._denormalize_field_data(pred_raw[t], field_key)
                    
                    np.clip(pred_denorm, 0.0, None, out=pred_denorm)
                    pred_channels.append(pred_denorm)
                
                # Write spatial CSV — one row per timestep
                with open(sp_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(sp_header)
                    
                    for t in range(self.num_tstep):
                        row = [years[t]]
                        for ch_idx in range(n_channels):
                            row.extend(pred_channels[ch_idx][t].ravel().tolist())
                        writer.writerow(row)
                        
                        if (t + 1) % 10 == 0 or t == self.num_tstep - 1:
                            print(f"    Timestep {t+1}/{self.num_tstep} written...")
                
                saved_files.append(sp_filename)
                print(f"  ✅ Saved {sp_filename}  ({self.num_tstep} rows × {len(sp_header)} columns)")
                
                # ── Summary ────────────────────────────────────────────────
                print(f"\n✅ Successfully exported Case {case_idx} (Simulation #{actual_case_id})")
                print(f"📋 Saved files in {output_dir}:")
                for fname in saved_files:
                    fpath = os.path.join(output_dir, fname)
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    print(f"   • {fname}  ({size_mb:.1f} MB)")
                
                self.csv_status_label.value = f'✅ Exported case #{actual_case_id} ({len(saved_files)} files)'
                
            except Exception as e:
                print(f"❌ Error exporting CSV: {e}")
                import traceback
                traceback.print_exc()
                self.csv_status_label.value = f'❌ Error: {str(e)}'
    
    def _plot_spatial_column(self, axes, pred_masked, true_masked, residual_masked,
                             unified_vmin, unified_vmax, res_vmin, res_vmax,
                             field_idx, label=''):
        """Plot a single column (Predicted, True, Residual) into the given 3 axes."""
        cmap = plt.cm.jet.copy()
        cmap.set_bad('white', alpha=0.3)
        residual_cmap = plt.cm.RdBu_r.copy()
        residual_cmap.set_bad('white', alpha=0.3)

        for ax, data, title, cm, vmin, vmax in [
            (axes[0], pred_masked, f'{label}Predicted', cmap, unified_vmin, unified_vmax),
            (axes[1], true_masked, 'True', cmap, unified_vmin, unified_vmax),
            (axes[2], residual_masked, f'{label}Residual (Pred - True)', residual_cmap, res_vmin, res_vmax),
        ]:
            im = ax.imshow(data.T, origin='lower', cmap=cm, vmin=vmin, vmax=vmax,
                           aspect='equal', interpolation='bilinear')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('I Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('J Index', fontsize=12, fontweight='bold')
            ax.tick_params(labelsize=10, width=1.2)
            cbar_label = f'{self.field_names[field_idx]}' if 'Residual' not in title else 'Residual (Pred - True)'
            cb = plt.colorbar(im, ax=ax, shrink=0.7, aspect=30, pad=0.02)
            cb.set_label(cbar_label, rotation=90, labelpad=12, fontsize=12, fontweight='bold')
            cb.ax.tick_params(labelsize=10, width=1.2)

    def _update_spatial_plot(self, change=None):
        """Update spatial visualization plot with high-quality styling (matching interactive_h5_visualizer.py)"""
        with self.spatial_output:
            clear_output(wait=True)
            
            # Get current selections
            case_idx = self.spatial_case_slider.value
            layer_idx = self.spatial_layer_slider.value  
            timestep_idx = self.spatial_timestep_slider.value
            field_idx = self.spatial_field_dropdown.value
            
            actual_case_idx = self.test_case_indices[case_idx]
            actual_layer = layer_idx
            actual_timestep = timestep_idx
            actual_year = self.all_years[timestep_idx]
            field_key = self.field_keys[field_idx]
            
            layer_mask = self._get_layer_mask(case_idx, actual_layer)
            
            # Get state-based data
            pred_data = self.state_pred[case_idx, actual_timestep, field_idx, :, :, actual_layer].cpu().detach().numpy()
            true_data = self.state_seq_true_aligned[case_idx, field_idx, actual_timestep, :, :, actual_layer].cpu().numpy()
            
            pred_data_denorm = self._denormalize_field_data(pred_data, field_key)
            true_data_denorm = self._denormalize_field_data(true_data, field_key)
            
            pred_data_masked = np.where(layer_mask, pred_data_denorm, np.nan)
            true_data_masked = np.where(layer_mask, true_data_denorm, np.nan)

            # --- COMPARISON MODE: side-by-side state vs latent ---
            show_latent = (self.spatial_latent_checkbox.value
                           and self.spatial_latent_generated
                           and self.state_pred_latent_based is not None)

            if show_latent:
                lat_pred = self.state_pred_latent_based[case_idx, actual_timestep, field_idx, :, :, actual_layer].cpu().detach().numpy()
                lat_pred_denorm = self._denormalize_field_data(lat_pred, field_key)
                lat_pred_masked = np.where(layer_mask, lat_pred_denorm, np.nan)

                residual_sb = pred_data_denorm - true_data_denorm
                residual_sb_masked = np.where(layer_mask, residual_sb, np.nan)
                residual_lb = lat_pred_denorm - true_data_denorm
                residual_lb_masked = np.where(layer_mask, residual_lb, np.nan)

                all_valid = np.concatenate([
                    pred_data_masked[~np.isnan(pred_data_masked)],
                    lat_pred_masked[~np.isnan(lat_pred_masked)],
                    true_data_masked[~np.isnan(true_data_masked)]])
                if len(all_valid) > 0:
                    unified_vmin, unified_vmax = np.percentile(all_valid[all_valid > 0] if np.any(all_valid > 0) else all_valid, [2, 98])
                else:
                    unified_vmin, unified_vmax = 0, 1

                all_res = np.concatenate([
                    residual_sb_masked[~np.isnan(residual_sb_masked)],
                    residual_lb_masked[~np.isnan(residual_lb_masked)]])
                if len(all_res) > 0:
                    abs_max = max(np.abs(np.percentile(all_res, [2, 98])))
                    res_vmax = max(abs_max, np.abs(all_res).max() * 0.95, 0.01)
                else:
                    res_vmax = 1.0
                res_vmin = -res_vmax

                plt.style.use('default')
                fig, axes_2d = plt.subplots(3, 2, figsize=(20, 20), dpi=100)
                timing_line = getattr(self, '_spatial_timing_info', '')
                subtitle = 'State-based (left) vs Latent-based (right)'
                if timing_line:
                    subtitle += f'\n{timing_line}'
                fig.suptitle(
                    f'{self.field_names[field_idx]} — Case {actual_case_idx} — '
                    f'K-Layer {actual_layer+1} — Year {actual_year}\n'
                    f'{subtitle}',
                    fontsize=14, fontweight='bold', y=0.99)

                self._plot_spatial_column(
                    axes_2d[:, 0], pred_data_masked, true_data_masked, residual_sb_masked,
                    unified_vmin, unified_vmax, res_vmin, res_vmax, field_idx, label='State-based ')
                self._plot_spatial_column(
                    axes_2d[:, 1], lat_pred_masked, true_data_masked, residual_lb_masked,
                    unified_vmin, unified_vmax, res_vmin, res_vmax, field_idx, label='Latent-based ')

                self._add_well_overlays(axes_2d[:, 0], actual_layer)
                self._add_well_overlays(axes_2d[:, 1], actual_layer)

                plt.tight_layout()
                plt.subplots_adjust(top=0.91, bottom=0.03, hspace=0.15, wspace=0.20)
                self._add_external_legend(fig)
                display(fig)
                try:
                    plt.close(fig)
                except (AttributeError, RuntimeError):
                    pass
                return
            
            # Calculate RESIDUAL (difference between predicted and true)
            # Residual: pred - true (positive = overprediction, negative = underprediction)
            residual = pred_data_denorm - true_data_denorm
            
            # Apply masking to residual
            residual_masked = np.where(layer_mask, residual, np.nan)
            
            # Enhanced color scaling with outlier handling (matching interactive_h5_visualizer.py approach)
            def get_optimal_color_range(data):
                """Get optimal color range using percentile-based scaling"""
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    # Use active data for color scaling, handle positive values preferentially
                    active_data = valid_data[valid_data > 0] if len(valid_data[valid_data > 0]) > 0 else valid_data
                    if len(active_data) > 0:
                        vmin, vmax = np.percentile(active_data, [2, 98])  # Percentile-based scaling
                    else:
                        vmin, vmax = valid_data.min(), valid_data.max()
                else:
                    vmin, vmax = 0, 1
                return vmin, vmax
            
            # UNIFIED color scale for predicted and actual panels (for easy comparison)
            # Combine both datasets to get unified color range
            combined_valid_pred = pred_data_masked[~np.isnan(pred_data_masked)]
            combined_valid_true = true_data_masked[~np.isnan(true_data_masked)]
            if len(combined_valid_pred) > 0 and len(combined_valid_true) > 0:
                combined_data = np.concatenate([combined_valid_pred, combined_valid_true])
                unified_vmin, unified_vmax = get_optimal_color_range(combined_data.reshape(-1, 1).flatten())
            elif len(combined_valid_pred) > 0:
                unified_vmin, unified_vmax = get_optimal_color_range(combined_valid_pred)
            elif len(combined_valid_true) > 0:
                unified_vmin, unified_vmax = get_optimal_color_range(combined_valid_true)
            else:
                unified_vmin, unified_vmax = 0, 1
            
            # Separate color range for residual
            # Residual can be positive (overprediction) or negative (underprediction)
            # Use symmetric scaling around zero for better visualization
            valid_residual_data = residual_masked[~np.isnan(residual_masked)]
            if len(valid_residual_data) > 0:
                # Use percentile-based scaling for symmetric range around zero
                abs_max = max(np.abs(np.percentile(valid_residual_data, [2, 98])))
                residual_vmax = max(abs_max, np.abs(valid_residual_data).max() * 0.95, 0.01)
                residual_vmin = -residual_vmax  # Symmetric around zero
            else:
                residual_vmin, residual_vmax = -1.0, 1.0
            
            # Create high-resolution plot with modern styling
            plt.style.use('default')  # Clean style
            fig, axes = plt.subplots(3, 1, figsize=(12, 20), dpi=100)  # Vertical layout - much bigger plots
            
            # Enhanced title with minimal spacing
            fig.suptitle(f'{self.field_names[field_idx]} - Case {actual_case_idx} - K-Layer {actual_layer+1} - Year {actual_year}', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # High contrast colormap with NaN handling
            cmap = plt.cm.jet.copy()  # Blue-Red (high contrast) colormap
            cmap.set_bad('white', alpha=0.3)  # Semi-transparent for masked cells
            
            # PREDICTED PANEL
            im1 = axes[0].imshow(pred_data_masked.T,  # Transpose for proper orientation
                               origin='lower',  # Proper grid orientation
                               cmap=cmap, 
                               vmin=unified_vmin, 
                               vmax=unified_vmax,
                               aspect='equal',  # Equal aspect ratio
                               interpolation='bilinear')  # Smooth interpolation
            
            axes[0].set_title('Predicted', fontsize=16, fontweight='bold', pad=15)
            axes[0].set_xlabel('I Index', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('J Index', fontsize=14, fontweight='bold')
            axes[0].tick_params(labelsize=12, width=1.5)
            # Make tick labels bold
            for label in axes[0].get_xticklabels():
                label.set_fontweight('bold')
            for label in axes[0].get_yticklabels():
                label.set_fontweight('bold')
            
            # Enhanced colorbar - exact height match with plot area
            cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.7, aspect=30, pad=0.02)
            cbar1.set_label(f'{self.field_names[field_idx]}', 
                           rotation=90, labelpad=15, fontsize=14, fontweight='bold')
            cbar1.ax.tick_params(labelsize=12, width=1.5)
            # Make colorbar tick labels bold
            for label in cbar1.ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # TRUE PANEL
            im2 = axes[1].imshow(true_data_masked.T,  # Transpose for proper orientation
                               origin='lower',  # Proper grid orientation
                               cmap=cmap, 
                               vmin=unified_vmin, 
                               vmax=unified_vmax,
                               aspect='equal',  # Equal aspect ratio
                               interpolation='bilinear')  # Smooth interpolation
            
            axes[1].set_title('True', fontsize=16, fontweight='bold', pad=15)
            axes[1].set_xlabel('I Index', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('J Index', fontsize=14, fontweight='bold')
            axes[1].tick_params(labelsize=12, width=1.5)
            # Make tick labels bold
            for label in axes[1].get_xticklabels():
                label.set_fontweight('bold')
            for label in axes[1].get_yticklabels():
                label.set_fontweight('bold')
            
            # Enhanced colorbar - exact height match with plot area
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, aspect=30, pad=0.02)
            cbar2.set_label(f'{self.field_names[field_idx]}', 
                           rotation=90, labelpad=15, fontsize=14, fontweight='bold')
            cbar2.ax.tick_params(labelsize=12, width=1.5)
            # Make colorbar tick labels bold
            for label in cbar2.ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # RESIDUAL PANEL (using diverging colormap for positive/negative residuals)
            # Use a diverging colormap (e.g., RdBu_r) where:
            # - Red = positive residual (overprediction: pred > true)
            # - Blue = negative residual (underprediction: pred < true)
            # - White = zero residual (perfect prediction)
            residual_cmap = plt.cm.RdBu_r.copy()  # Red-Blue diverging colormap (reversed)
            residual_cmap.set_bad('white', alpha=0.3)  # White for masked/inactive cells
            
            im3 = axes[2].imshow(residual_masked.T,  # Transpose for proper orientation
                               origin='lower',  # Proper grid orientation
                               cmap=residual_cmap, 
                               vmin=residual_vmin, 
                               vmax=residual_vmax,
                               aspect='equal',  # Equal aspect ratio
                               interpolation='bilinear')  # Smooth interpolation
            
            axes[2].set_title('Residual (Pred - True)', fontsize=16, fontweight='bold', pad=15)
            axes[2].set_xlabel('I Index', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('J Index', fontsize=14, fontweight='bold')
            axes[2].tick_params(labelsize=12, width=1.5)
            # Make tick labels bold
            for label in axes[2].get_xticklabels():
                label.set_fontweight('bold')
            for label in axes[2].get_yticklabels():
                label.set_fontweight('bold')
            
            # Enhanced colorbar for residual - exact height match with plot area
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.7, aspect=30, pad=0.02)
            cbar3.set_label(f'Residual (Pred - True)', 
                            rotation=90, labelpad=15, fontsize=14, fontweight='bold')
            cbar3.ax.tick_params(labelsize=12, width=1.5)
            # Make colorbar tick labels bold
            for label in cbar3.ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Add well location overlays
            self._add_well_overlays(axes, actual_layer)
            
            # Final layout optimization with minimal spacing
            plt.tight_layout()
            plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.08)  # Minimal vertical spacing for compact layout
            
            # Add external legend beneath the plot
            self._add_external_legend(fig)
            
            # Display in widget context
            display(fig)
            try:
                plt.close(fig)
            except (AttributeError, RuntimeError):
                # If closing fails (e.g., manager is None), just continue
                pass
            
            # Also update metrics
            self._update_spatial_metrics()
            
    def _add_well_overlays(self, axes, current_layer):
        """Add well location overlays to all spatial plot panels"""
        # Define well symbols and colors
        injector_style = {'marker': '^', 'color': 'blue', 'markersize': 12, 'markeredgecolor': 'white', 'markeredgewidth': 2}
        producer_style = {'marker': 'o', 'color': 'red', 'markersize': 10, 'markeredgecolor': 'white', 'markeredgewidth': 2}
        
        # Add well locations to each panel
        for ax in axes:
            # Add injector wells
            for well_name, coords in self.well_locations['injectors'].items():
                x, y, z = coords
                # Show well if it penetrates this layer (wells penetrate full depth, so show on all layers)
                ax.plot(x, y, **injector_style, label=f'Injector {well_name}' if ax == axes[0] else "")
                
            # Add producer wells  
            for well_name, coords in self.well_locations['producers'].items():
                x, y, z = coords
                # Show well if it penetrates this layer (wells penetrate full depth, so show on all layers)
                ax.plot(x, y, **producer_style, label=f'Producer {well_name}' if ax == axes[0] else "")
    
    def _add_external_legend(self, fig):
        """Add a legend beneath all plots showing well types"""
        if not self.well_locations['injectors'] and not self.well_locations['producers']:
            return  # No wells to show in legend
            
        # Create legend elements
        legend_elements = []
        
        if self.well_locations['injectors']:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                            markerfacecolor='blue', markersize=12,
                                            markeredgecolor='white', markeredgewidth=2,
                                            label=f'Injectors ({len(self.well_locations["injectors"])})'))
        
        if self.well_locations['producers']:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='red', markersize=10,
                                            markeredgecolor='white', markeredgewidth=2,
                                            label=f'Producers ({len(self.well_locations["producers"])})'))
        
        # Add legend beneath the plots
        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, 0.01), ncol=2, 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
            
    def _update_timeseries_plot(self, change=None):
        """Update time series visualization plot with comparison mode support"""
        with self.timeseries_output:
            clear_output(wait=True)
            
            # Get current selections
            case_idx = self.timeseries_case_slider.value
            obs_idx = self.timeseries_obs_dropdown.value
            
            actual_case_idx = self.test_case_indices[case_idx]
            
            # Get time series data
            years_ts = np.arange(self.start_year, self.start_year + self.num_tstep)
            
            # Extract true data
            true_data = self.yobs_seq_true[case_idx, obs_idx, :self.num_tstep].cpu().numpy()
            true_data_denorm = self._denormalize_obs_data(true_data, obs_idx)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            
            # Always plot ground truth
            ax.plot(years_ts, true_data_denorm, 'b-', label='Ground Truth', linewidth=3, alpha=0.9)
            
            if self.comparison_mode_enabled and self.predictions_generated:
                # COMPARISON MODE: Show both prediction methods (if enabled)
                transition_mode = getattr(self.my_rom.model, 'transition_mode', 'latent')
                is_latent = (transition_mode == 'latent')
                
                alt_label = 'Latent-based Prediction' if is_latent else 'True-initialized Prediction'
                
                # Alternative predictions (always show in comparison mode)
                alt_pred_data = self.yobs_pred_latent_based[case_idx, :self.num_tstep, obs_idx].cpu().detach().numpy()
                alt_pred_denorm = self._denormalize_obs_data(alt_pred_data, obs_idx)
                alt_pred_denorm = np.maximum(alt_pred_denorm, 0.0)
                ax.plot(years_ts, alt_pred_denorm, 'm:', label=alt_label, linewidth=2, alpha=0.8)
                
                # State-based (auto-regressive) predictions (only if checkbox is enabled)
                if self.show_state_based_checkbox.value:
                    state_pred_data = self.yobs_pred_state_based[case_idx, :self.num_tstep, obs_idx].cpu().detach().numpy()
                    state_pred_denorm = self._denormalize_obs_data(state_pred_data, obs_idx)
                    state_pred_denorm = np.maximum(state_pred_denorm, 0.0)
                    ax.plot(years_ts, state_pred_denorm, 'g--', label='State-based Prediction', linewidth=2, alpha=0.8)
                
                title_suffix = " - Comparison Mode"
                
            else:
                # STANDARD MODE: Show default predictions only
                pred_data = self.yobs_pred[case_idx, :self.num_tstep, obs_idx].cpu().detach().numpy()
                pred_data_denorm = self._denormalize_obs_data(pred_data, obs_idx)
                pred_data_denorm = np.maximum(pred_data_denorm, 0.0)
                
                ax.plot(years_ts, pred_data_denorm, 'r--', label='Prediction', linewidth=2, alpha=0.8)
                
                title_suffix = ""
            
            ax.set_title(f'{self.obs_names[obs_idx]} - Case {actual_case_idx}{title_suffix}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{self.obs_names[obs_idx]} ({self.obs_units[obs_idx]})', fontsize=14, fontweight='bold')
            
            # Make legend text bold and larger
            legend = ax.legend(fontsize=12)
            for text in legend.get_texts():
                text.set_fontweight('bold')
            
            # Make grid bolder and tick labels bold and larger
            ax.grid(True, alpha=0.3, linewidth=1.2)
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            
            plt.tight_layout()
            display(fig)
            # Close figure with error handling to prevent backend issues
            try:
                plt.close(fig)
            except (AttributeError, RuntimeError):
                # If closing fails (e.g., manager is None), just continue
                pass
            
            # Also update metrics
            self._update_timeseries_metrics()
    
    def _update_spatial_metrics(self, change=None):
        """Update spatial metrics visualization"""
        with self.spatial_metrics_output:
            clear_output(wait=True)
            
            case_idx = self.spatial_case_slider.value
            layer_idx = self.spatial_layer_slider.value  
            timestep_idx = self.spatial_timestep_slider.value
            field_idx = self.spatial_field_dropdown.value

            show_latent = (self.spatial_latent_checkbox.value
                           and self.spatial_latent_generated
                           and self.state_pred_latent_based is not None)

            if show_latent:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                self.metrics_evaluator.plot_spatial_metrics(
                    case_idx, field_idx, layer_idx, timestep_idx,
                    ax=ax1, norm_params=self.norm_params, dashboard=self
                )
                ax1.set_title('State-based — ' + ax1.get_title(), fontsize=11, fontweight='bold')

                self.metrics_evaluator.plot_spatial_metrics(
                    case_idx, field_idx, layer_idx, timestep_idx,
                    ax=ax2, norm_params=self.norm_params, dashboard=self,
                    state_pred_override=self.state_pred_latent_based
                )
                ax2.set_title('Latent-based — ' + ax2.get_title(), fontsize=11, fontweight='bold')
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                self.metrics_evaluator.plot_spatial_metrics(
                    case_idx, field_idx, layer_idx, timestep_idx, 
                    ax=ax, norm_params=self.norm_params, dashboard=self
                )
            
            plt.tight_layout()
            display(fig)
            try:
                plt.close(fig)
            except (AttributeError, RuntimeError):
                pass
    
    def _update_timeseries_metrics(self, change=None):
        """Update timeseries metrics visualization based on active prediction method"""
        with self.timeseries_metrics_output:
            clear_output(wait=True)
            
            case_idx = self.timeseries_case_slider.value
            obs_idx = self.timeseries_obs_dropdown.value
            
            if self.comparison_mode_enabled and self.predictions_generated:
                show_state = self.show_state_based_checkbox.value
                transition_mode = getattr(self.my_rom.model, 'transition_mode', 'latent')
                is_latent = (transition_mode == 'latent')
                alt_label = 'Latent-based' if is_latent else 'True-initialized'
                
                if show_state:
                    # Both visible → show side-by-side metrics
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    self.metrics_evaluator.plot_timeseries_metrics(
                        case_idx, obs_idx, ax=ax1, norm_params=self.norm_params,
                        yobs_pred_override=self.yobs_pred_state_based
                    )
                    ax1.set_title(ax1.get_title().replace(f'Case {case_idx}', f'Case {case_idx} (State-based)'), fontsize=11, fontweight='bold')
                    
                    self.metrics_evaluator.plot_timeseries_metrics(
                        case_idx, obs_idx, ax=ax2, norm_params=self.norm_params,
                        yobs_pred_override=self.yobs_pred_latent_based
                    )
                    ax2.set_title(ax2.get_title().replace(f'Case {case_idx}', f'Case {case_idx} ({alt_label})'), fontsize=11, fontweight='bold')
                else:
                    # Only latent/true-initialized visible
                    fig, ax = plt.subplots(figsize=(10, 5))
                    self.metrics_evaluator.plot_timeseries_metrics(
                        case_idx, obs_idx, ax=ax, norm_params=self.norm_params,
                        yobs_pred_override=self.yobs_pred_latent_based
                    )
                    ax.set_title(ax.get_title().replace(f'Case {case_idx}', f'Case {case_idx} ({alt_label})'), fontsize=11, fontweight='bold')
            else:
                # Standard mode
                fig, ax = plt.subplots(figsize=(10, 5))
                self.metrics_evaluator.plot_timeseries_metrics(
                    case_idx, obs_idx, ax=ax, norm_params=self.norm_params
                )
            
            plt.tight_layout()
            
            # Display in widget context
            display(fig)
            # Close figure with error handling to prevent backend issues
            try:
                plt.close(fig)
            except (AttributeError, RuntimeError):
                pass
    
    def _create_overall_metrics_tab(self):
        """Create the overall performance metrics tab"""
        # Metric selection checkboxes
        self.metric_r2_checkbox = widgets.Checkbox(
            value=True,
            description='R²',
            style={'description_width': 'initial'}
        )
        self.metric_mse_checkbox = widgets.Checkbox(
            value=True,
            description='MSE',
            style={'description_width': 'initial'}
        )
        self.metric_rmse_checkbox = widgets.Checkbox(
            value=True,
            description='RMSE',
            style={'description_width': 'initial'}
        )
        self.metric_ae_checkbox = widgets.Checkbox(
            value=True,
            description='AE (Absolute Error)',
            style={'description_width': 'initial'}
        )
        
        # Case selection checkboxes
        self.calc_train_checkbox = widgets.Checkbox(
            value=True,
            description='Training Cases',
            style={'description_width': 'initial'}
        )
        self.calc_test_checkbox = widgets.Checkbox(
            value=True,
            description='Testing Cases',
            style={'description_width': 'initial'}
        )
        
        # Metrics calculation mode selection
        self.metrics_mode_dropdown = widgets.Dropdown(
            options=['Aggregated', 'Averaged'],
            value='Aggregated',
            description='Metrics Calculation:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Graph type selection (spatial or timeseries)
        self.graph_type_dropdown = widgets.Dropdown(
            options=['Spatial', 'Timeseries'],
            value='Spatial',
            description='Graph Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        
        # Spatial graph dropdown (shared between training/testing)
        self.spatial_graph_dropdown = widgets.Dropdown(
            options=self.field_names if hasattr(self, 'field_names') else ['Gas Saturation', 'Pressure'],
            value=self.field_names[0] if hasattr(self, 'field_names') and len(self.field_names) > 0 else 'Gas Saturation',
            description='Spatial Graph:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Timeseries graph dropdown (shared between training/testing)
        timeseries_options = ['BHP (All Injectors)', 'Gas Production (All Producers)', 'Water Production (All Producers)']
        self.timeseries_graph_dropdown = widgets.Dropdown(
            options=timeseries_options,
            value=timeseries_options[0],
            description='Timeseries Graph:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Spatial layer dropdown (for filtering by specific layer)
        layer_options = ['All Layers'] + [f'Layer {i}' for i in range(self.Nz)]
        self.spatial_layer_dropdown = widgets.Dropdown(
            options=layer_options,
            value='All Layers',
            description='Layer:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Timeseries well dropdown (for filtering by specific well)
        # Will be populated dynamically based on selected group
        self.timeseries_well_dropdown = widgets.Dropdown(
            options=['All Wells'],
            value='All Wells',
            description='Well:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Calculate & Plot button
        calculate_button = widgets.Button(
            description='📊 Calculate & Plot Overall Metrics',
            button_style='primary',
            layout=widgets.Layout(width='300px', height='40px')
        )
        
        # Button click handler
        def on_calculate_click(button):
            with self.overall_metrics_output:
                self.overall_metrics_output.clear_output(wait=True)
                
                # Check if at least one metric is selected
                selected_metrics = []
                if self.metric_r2_checkbox.value:
                    selected_metrics.append('R²')
                if self.metric_mse_checkbox.value:
                    selected_metrics.append('MSE')
                if self.metric_rmse_checkbox.value:
                    selected_metrics.append('RMSE')
                if self.metric_ae_checkbox.value:
                    selected_metrics.append('AE')
                
                if not selected_metrics:
                    print("❌ Please select at least one metric to calculate!")
                    return
                
                # Check if at least one case type is selected
                if not self.calc_train_checkbox.value and not self.calc_test_checkbox.value:
                    print("❌ Please select at least one case type (Training or Testing)!")
                    return
                
                print("🔄 Calculating overall performance metrics...")
                print(f"   Selected metrics: {', '.join(selected_metrics)}")
                case_types = []
                if self.calc_train_checkbox.value:
                    case_types.append('Training')
                if self.calc_test_checkbox.value:
                    case_types.append('Testing')
                print(f"   Case types: {', '.join(case_types)}")
                metrics_mode = self.metrics_mode_dropdown.value
                print(f"   Metrics calculation mode: {metrics_mode}")
                print("   Optimized calculation in progress...")
                
                self._plot_overall_performance_metrics()
                print("✅ Overall metrics calculation completed!")
        
        calculate_button.on_click(on_calculate_click)
        
        # Helper function to update well dropdown options based on selected group
        def update_well_dropdown_options(group_name):
            obs_groups_map = {
                'BHP (All Injectors)': (list(range(3)), ['BHP1', 'BHP2', 'BHP3']),
                'Gas Production (All Producers)': (list(range(3, 6)), ['Gas Prod1', 'Gas Prod2', 'Gas Prod3']),
                'Water Production (All Producers)': (list(range(6, 9)), ['Water Prod1', 'Water Prod2', 'Water Prod3'])
            }
            
            if group_name in obs_groups_map:
                obs_indices, well_names = obs_groups_map[group_name]
                well_options = ['All Wells'] + well_names
                self.timeseries_well_dropdown.options = well_options
                self.timeseries_well_dropdown.value = 'All Wells'
            else:
                self.timeseries_well_dropdown.options = ['All Wells']
                self.timeseries_well_dropdown.value = 'All Wells'
        
        # Create function to update dropdown visibility based on graph type
        def update_dropdown_visibility(change):
            if self.graph_type_dropdown.value == 'Spatial':
                self.spatial_graph_dropdown.layout.display = 'flex'
                self.spatial_layer_dropdown.layout.display = 'flex'
                self.timeseries_graph_dropdown.layout.display = 'none'
                self.timeseries_well_dropdown.layout.display = 'none'
            else:
                self.spatial_graph_dropdown.layout.display = 'none'
                self.spatial_layer_dropdown.layout.display = 'none'
                self.timeseries_graph_dropdown.layout.display = 'flex'
                self.timeseries_well_dropdown.layout.display = 'flex'
                # Update well dropdown when switching to timeseries
                update_well_dropdown_options(self.timeseries_graph_dropdown.value)
        
        # Observer for timeseries group dropdown to update well options
        def update_well_options_on_group_change(change):
            if self.graph_type_dropdown.value == 'Timeseries':
                update_well_dropdown_options(change['new'])
        
        # Set initial visibility
        self.graph_type_dropdown.observe(update_dropdown_visibility, names='value')
        self.timeseries_graph_dropdown.observe(update_well_options_on_group_change, names='value')
        update_dropdown_visibility(None)
        
        # Create tab content
        tab_content = widgets.VBox([
            widgets.HTML("<h4>📊 Overall Performance Analysis</h4>"),
            widgets.HTML("<p><i>Aggregated prediction performance across selected cases</i></p>"),
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Select Metrics to Calculate:</b>"),
            widgets.HBox([
                self.metric_r2_checkbox,
                self.metric_mse_checkbox,
                self.metric_rmse_checkbox,
                self.metric_ae_checkbox
            ]),
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Select Case Types:</b>"),
            widgets.HBox([
                self.calc_train_checkbox,
                self.calc_test_checkbox
            ]),
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Metrics Calculation Mode:</b>"),
            self.metrics_mode_dropdown,
            widgets.HTML("<p><i>Aggregated: Calculate metrics from all aggregated data points<br>Averaged: Average metrics from individual case/layer/timestep combinations</i></p>"),
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Select Graph to Display:</b>"),
            self.graph_type_dropdown,
            self.spatial_graph_dropdown,
            self.spatial_layer_dropdown,
            self.timeseries_graph_dropdown,
            self.timeseries_well_dropdown,
            widgets.HTML("<hr>"),
            widgets.HTML("<ul><li><b>Spatial Metrics:</b> Performance for each field averaged across selected cases, layers, and timesteps</li><li><b>Timeseries Metrics:</b> Performance for each observation type averaged across selected cases and timesteps</li></ul>"),
            calculate_button,
            widgets.HTML("<hr>"),
            # Save button
            widgets.HTML("<b>Save Metrics:</b>"),
            self.overall_metrics_output
        ])
        
        # Create save button separately and add handler
        save_button = widgets.Button(
            description='💾 Save Metrics to JSON',
            button_style='success',
            layout=widgets.Layout(width='200px', height='35px')
        )
        save_button.on_click(self._save_overall_metrics_to_json)
        
        # Insert save button and separator before the output widget
        children_list = list(tab_content.children)
        children_list.insert(-1, save_button)
        children_list.insert(-1, widgets.HTML("<hr>"))
        tab_content.children = children_list
        
        return tab_content
    
    def _create_animation_tab(self):
        """Create time evolution animation tab"""
        # Animation controls
        self.anim_case_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.num_case-1,
            step=1,
            description='Test Case:',
            style={'description_width': 'initial'}
        )
        
        self.anim_layer_slider = widgets.IntSlider(
            value=self.Nz//2,
            min=0,
            max=self.Nz-1,
            step=1,
            description='Layer:',
            style={'description_width': 'initial'}
        )
        
        self.anim_field_dropdown = widgets.Dropdown(
            options=[(name, idx) for idx, name in enumerate(self.field_names)],
            value=0,
            description='Spatial Property:',
            style={'description_width': 'initial'}
        )
        
        self.play_button = widgets.Button(
            description='▶️ Play Animation',
            button_style='success',
            icon='play'
        )
        
        self.stop_button = widgets.Button(
            description='⏹️ Stop',
            button_style='danger',
            icon='stop',
            disabled=True
        )
        
        self.animation_speed_slider = widgets.FloatSlider(
            value=0.5,
            min=0.1,
            max=2.0,
            step=0.1,
            description='Speed (sec/frame):',
            style={'description_width': 'initial'}
        )
        
        self.animation_status = widgets.Label(value='Animation Status: Ready')
        
        # Animation output areas
        self.animation_output = widgets.Output()
        
        # Set up event handlers for animation
        self.play_button.on_click(self._start_animation)
        self.stop_button.on_click(self._stop_animation)
        
        # Add event handlers to stop animation when controls change to prevent dashboard reset
        self.anim_case_slider.observe(self._stop_animation_on_change, names='value')
        self.anim_layer_slider.observe(self._stop_animation_on_change, names='value')
        self.anim_field_dropdown.observe(self._stop_animation_on_change, names='value')
        
        # Animation control variables
        self.animation_running = False
        self.animation_thread = None
        
        # Control layout
        animation_controls = widgets.VBox([
            widgets.HTML("<h4>🎬 Animation Controls</h4>"),
            self.anim_case_slider,
            self.anim_layer_slider,
            self.anim_field_dropdown,
            widgets.HBox([self.play_button, self.stop_button]),
            self.animation_speed_slider,
            self.animation_status
        ])
        
        return widgets.VBox([
            widgets.HTML("<h3>🎬 Time Evolution Animation</h3>"),
            widgets.HTML("<p><i>Visualize how spatial properties evolve over time. Compare predicted vs. actual field evolution.</i></p>"),
            animation_controls,
            widgets.HTML("<hr>"),
            self.animation_output
        ])
    
    def _calculate_train_test_indices_from_total(self, total_cases):
        """
        Calculate training and test case indices based on per-100-case split pattern.
        
        Args:
            total_cases: Total number of cases in the combined dataset
        
        Returns:
            train_indices: Array of training case indices
            test_indices: Array of test case indices
        """
        num_run_per_case = 75  # 75% for training
        num_run_eval = 25      # 25% for evaluation
        
        if total_cases < 100:
            # Simple split for smaller datasets
            actual_train = int(total_cases * 0.75)
            train_indices = np.arange(actual_train)
            test_indices = np.arange(actual_train, total_cases)
        else:
            # Per-100-case split pattern
            split_ratio = int(total_cases / 100)
            actual_train = num_run_per_case * split_ratio
            actual_eval = num_run_eval * split_ratio
            
    def _calculate_train_test_indices_from_total(self, total_cases):
        """
        Calculate training and test case indices based on per-100-case split pattern.
        
        Args:
            total_cases: Total number of cases in the combined dataset
        
        Returns:
            train_indices: Array of training case indices
            test_indices: Array of test case indices
        """
        num_run_per_case = 75  # 75% for training
        num_run_eval = 25      # 25% for evaluation
        
        if total_cases < 100:
            # Simple split for smaller datasets
            actual_train = int(total_cases * 0.75)
            train_indices = np.arange(actual_train)
            test_indices = np.arange(actual_train, total_cases)
        else:
            # Per-100-case split pattern
            split_ratio = int(total_cases / 100)
            actual_train = num_run_per_case * split_ratio
            actual_eval = num_run_eval * split_ratio
            
            train_indices_list = []
            test_indices_list = []
            
            for k in range(split_ratio):
                # Training: first 75 cases in each block of 100
                train_start = k * 100
                train_end = k * 100 + num_run_per_case
                train_indices_list.extend(range(train_start, train_end))
                
                # Test: last 25 cases in each block of 100
                test_start = k * 100 + num_run_per_case
                test_end = k * 100 + 100
                test_indices_list.extend(range(test_start, test_end))
            
            train_indices = np.array(train_indices_list)
            test_indices = np.array(test_indices_list)
        
        return train_indices, test_indices
    
    def _get_train_test_indices(self):
        """
        Identify training vs testing case indices.
        Returns indices for training and test cases based on available data.
        Uses per-100-case split pattern: first 75 cases in each block of 100 are training, last 25 are test.
        """
        # Get total number of cases from state_pred (combined array)
        if self.state_pred is not None:
            total_cases = self.state_pred.shape[0]
            train_indices, test_indices = self._calculate_train_test_indices_from_total(total_cases)
        else:
            # Fallback: use separate arrays if state_pred not available
            if self.train_state_pred is not None and self.train_case_indices is not None:
                train_indices = np.arange(self.num_train_case)
            else:
                train_indices = np.array([])
            
            if hasattr(self, 'test_case_indices') and self.test_case_indices is not None:
                test_indices = np.array(self.test_case_indices)
            else:
                test_indices = np.array([])
        
        return train_indices, test_indices
    
    def _plot_overall_performance_metrics(self):
        """Plot overall performance metrics for selected graph type and case types"""
        # Get selected metrics
        selected_metrics = []
        if self.metric_r2_checkbox.value:
            selected_metrics.append('r2')
        if self.metric_mse_checkbox.value:
            selected_metrics.append('mse')
        if self.metric_rmse_checkbox.value:
            selected_metrics.append('rmse')
        if self.metric_ae_checkbox.value:
            selected_metrics.append('mae')
        
        # Get train/test indices (initial check)
        train_indices, test_indices = self._get_train_test_indices()
        
        # Get selected graph type and specific graph
        graph_type = self.graph_type_dropdown.value  # 'Spatial' or 'Timeseries'
        
        # Check checkbox states
        train_checkbox_checked = self.calc_train_checkbox.value
        test_checkbox_checked = self.calc_test_checkbox.value
        
        # Training predictions should be pre-generated upfront (like test predictions)
        # If they don't exist, training metrics will be skipped gracefully (no on-demand generation)
        
        # Determine if we should show training/test plots
        show_train = train_checkbox_checked and len(train_indices) > 0
        show_test = test_checkbox_checked
        
        # Determine number of columns needed
        num_cols = 0
        
        if show_train and show_test:
            num_cols = 2
        elif show_train or show_test:
            num_cols = 1
        else:
            if not train_checkbox_checked and not test_checkbox_checked:
                print("❌ Please select at least one case type (Training or Testing)!")
            return
        
        # Check if latent-based predictions are available for a second row
        has_latent = (graph_type == 'Timeseries'
                      and self.predictions_generated
                      and self.yobs_pred_latent_based is not None)
        num_rows = 2 if has_latent else 1
        
        # Create figure
        fig, axes_2d = plt.subplots(num_rows, num_cols,
                                     figsize=(8 * num_cols, 6 * num_rows),
                                     squeeze=False)
        axes = axes_2d[0]  # first row for state-based (or spatial)
        
        plot_idx = 0
        
        # Initialize variables for storing metrics
        spatial_metrics_train = None
        spatial_metrics_test = None
        timeseries_metrics_train = None
        timeseries_metrics_test = None
        
        # Process training data if selected
        if show_train:
            if graph_type == 'Spatial':
                # Get selected spatial field
                selected_field_name = self.spatial_graph_dropdown.value
                field_idx = self.field_names.index(selected_field_name) if selected_field_name in self.field_names else 0
                
                # Get selected layer (None if "All Layers")
                selected_layer = self.spatial_layer_dropdown.value
                layer_idx = None if selected_layer == 'All Layers' else int(selected_layer.split()[-1])
                
                # Get metrics calculation mode
                use_averaged_metrics = (self.metrics_mode_dropdown.value == 'Averaged')
                
                # Calculate metrics for selected field and layer
                spatial_metrics_train, spatial_data_train = self._calculate_overall_spatial_metrics_optimized(
                    case_indices=train_indices, selected_metrics=selected_metrics,
                    field_idx=field_idx, use_training_data=True, layer_idx=layer_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                
                if spatial_metrics_train is not None and spatial_data_train is not None:
                    ax = axes[plot_idx]
                    # Handle grouped data (when layer_idx is None - All Layers)
                    if isinstance(spatial_data_train, dict) and 'layers' in spatial_data_train:
                        # Layer-grouped data for color-coding
                        layer_data_dict = spatial_data_train['layers']
                        all_true_combined = spatial_data_train.get('all_true', np.array([]))
                        all_pred_combined = spatial_data_train.get('all_pred', np.array([]))
                        title_field_name = selected_field_name  # "All Layers" already implied
                        self._plot_overall_spatial_metric(ax, field_idx, title_field_name, 
                                                         spatial_metrics_train, all_true_combined, all_pred_combined,
                                                         selected_metrics=selected_metrics,
                                                         layer_data_dict=layer_data_dict, fig=fig)
                    else:
                        # Single layer data
                        all_true, all_pred = spatial_data_train
                        # Build title with layer info if specific layer selected
                        title_field_name = selected_field_name
                        if layer_idx is not None:
                            title_field_name = f"{selected_field_name} - Layer {layer_idx}"
                        self._plot_overall_spatial_metric(ax, field_idx, title_field_name, 
                                                         spatial_metrics_train, all_true, all_pred,
                                                         selected_metrics=selected_metrics, fig=fig)
                    # Update title to include Training prefix and metrics
                    current_title = ax.get_title()
                    ax.set_title(f'Training - {current_title}', fontsize=10, fontweight='bold')
                    plot_idx += 1
            else:  # Timeseries
                # Get selected timeseries group
                selected_group_name = self.timeseries_graph_dropdown.value
                obs_groups_map = {
                    'BHP (All Injectors)': (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'),
                    'Gas Production (All Producers)': (list(range(3, 6)), ['Gas Prod1', 'Gas Prod2', 'Gas Prod3'], 'ft3/day'),
                    'Water Production (All Producers)': (list(range(6, 9)), ['Water Prod1', 'Water Prod2', 'Water Prod3'], 'ft3/day')
                }
                obs_indices, well_names, unit = obs_groups_map.get(selected_group_name, (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'))
                
                # Get selected well (None if "All Wells")
                selected_well = self.timeseries_well_dropdown.value
                obs_idx = None
                well_name_for_title = selected_group_name
                if selected_well != 'All Wells' and selected_well in well_names:
                    well_idx = well_names.index(selected_well)
                    obs_idx = obs_indices[well_idx]
                    # Remove parenthetical part (e.g., "(All Injectors)") when specific well is selected
                    # "BHP (All Injectors)" -> "BHP", "Water Production (All Producers)" -> "Water Production"
                    group_base = selected_group_name.split(' (')[0]
                    well_name_for_title = f"{group_base} - {selected_well}"
                
                # Get metrics calculation mode
                use_averaged_metrics = (self.metrics_mode_dropdown.value == 'Averaged')
                
                # Calculate metrics for selected group/well
                timeseries_metrics_train, timeseries_data_train = self._calculate_overall_timeseries_metrics_optimized(
                    case_indices=train_indices, selected_metrics=selected_metrics,
                    obs_group_indices=obs_indices if obs_idx is None else None, 
                    use_training_data=True, obs_idx=obs_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                
                if timeseries_metrics_train is not None and timeseries_data_train is not None:
                    ax = axes[plot_idx]
                    # Handle grouped data (when obs_idx is None - All Wells)
                    if isinstance(timeseries_data_train, dict) and 'observations' in timeseries_data_train:
                        # Observation-grouped data for color-coding
                        obs_data_dict = timeseries_data_train['observations']
                        all_true_combined = timeseries_data_train.get('all_true', np.array([]))
                        all_pred_combined = timeseries_data_train.get('all_pred', np.array([]))
                        # Create obs_names_map for legend
                        obs_names_map = {}
                        for obs_idx in obs_indices:
                            if obs_idx < len(self.obs_names):
                                obs_names_map[obs_idx] = self.obs_names[obs_idx]
                        self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                            timeseries_metrics_train, all_true_combined, all_pred_combined,
                                                            selected_metrics=selected_metrics,
                                                            obs_data_dict=obs_data_dict, obs_names_map=obs_names_map, fig=fig)
                    elif isinstance(timeseries_data_train, tuple) and len(timeseries_data_train) == 2:
                        # Single observation data (tuple format)
                        all_true, all_pred = timeseries_data_train
                        self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                            timeseries_metrics_train, all_true, all_pred,
                                                            selected_metrics=selected_metrics, fig=fig)
                    elif isinstance(timeseries_data_train, tuple) and len(timeseries_data_train) == 2:
                        # Single observation data (tuple format)
                        all_true, all_pred = timeseries_data_train
                        self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                            timeseries_metrics_train, all_true, all_pred,
                                                            selected_metrics=selected_metrics, fig=fig)
                    else:
                        # Handle unexpected format - try to extract data
                        print(f"⚠️ Warning: Unexpected timeseries_data_train format: {type(timeseries_data_train)}, value: {timeseries_data_train}")
                        if isinstance(timeseries_data_train, list) and len(timeseries_data_train) > 0:
                            # If it's a list, try to get the first element
                            data_item = timeseries_data_train[0]
                            if isinstance(data_item, tuple) and len(data_item) == 2:
                                all_true, all_pred = data_item
                                self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                                    timeseries_metrics_train, all_true, all_pred,
                                                                    selected_metrics=selected_metrics, fig=fig)
                            elif isinstance(data_item, dict) and 'observations' in data_item:
                                # Handle dict format
                                obs_data_dict = data_item['observations']
                                all_true_combined = data_item.get('all_true', np.array([]))
                                all_pred_combined = data_item.get('all_pred', np.array([]))
                                obs_names_map = {}
                                for obs_idx in obs_indices:
                                    if obs_idx < len(self.obs_names):
                                        obs_names_map[obs_idx] = self.obs_names[obs_idx]
                                self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                                    timeseries_metrics_train, all_true_combined, all_pred_combined,
                                                                    selected_metrics=selected_metrics,
                                                                    obs_data_dict=obs_data_dict, obs_names_map=obs_names_map, fig=fig)
                            else:
                                print(f"❌ Error: Cannot extract data from timeseries_data_train[0]: {type(data_item)}")
                        else:
                            print(f"❌ Error: Cannot extract data from timeseries_data_train (empty or invalid)")
                    # Update title to include Training prefix and metrics
                    current_title = ax.get_title()
                    ax.set_title(f'Training - {current_title}', fontsize=10, fontweight='bold')
                    plot_idx += 1
                else:
                    print(f"   ⚠️ Warning: Training timeseries metrics/data are None!")
        
        # Process test data if selected
        if show_test:
            if graph_type == 'Spatial':
                # Get selected spatial field
                selected_field_name = self.spatial_graph_dropdown.value
                field_idx = self.field_names.index(selected_field_name) if selected_field_name in self.field_names else 0
                
                # Get selected layer (None if "All Layers")
                selected_layer = self.spatial_layer_dropdown.value
                layer_idx = None if selected_layer == 'All Layers' else int(selected_layer.split()[-1])
                
                # Get metrics calculation mode
                use_averaged_metrics = (self.metrics_mode_dropdown.value == 'Averaged')
                
                # Calculate metrics for selected field and layer
                spatial_metrics_test, spatial_data_test = self._calculate_overall_spatial_metrics_optimized(
                    case_indices=test_indices, selected_metrics=selected_metrics,
                    field_idx=field_idx, use_training_data=False, layer_idx=layer_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                
                if spatial_metrics_test is not None and spatial_data_test is not None:
                    ax = axes[plot_idx]
                    # Handle grouped data (when layer_idx is None - All Layers)
                    if isinstance(spatial_data_test, dict) and 'layers' in spatial_data_test:
                        # Layer-grouped data for color-coding
                        layer_data_dict = spatial_data_test['layers']
                        all_true_combined = spatial_data_test.get('all_true', np.array([]))
                        all_pred_combined = spatial_data_test.get('all_pred', np.array([]))
                        title_field_name = selected_field_name  # "All Layers" already implied
                        self._plot_overall_spatial_metric(ax, field_idx, title_field_name,
                                                         spatial_metrics_test, all_true_combined, all_pred_combined,
                                                         selected_metrics=selected_metrics,
                                                         layer_data_dict=layer_data_dict, fig=fig)
                    else:
                        # Single layer data
                        all_true, all_pred = spatial_data_test
                        # Build title with layer info if specific layer selected
                        title_field_name = selected_field_name
                        if layer_idx is not None:
                            title_field_name = f"{selected_field_name} - Layer {layer_idx}"
                        self._plot_overall_spatial_metric(ax, field_idx, title_field_name,
                                                         spatial_metrics_test, all_true, all_pred,
                                                         selected_metrics=selected_metrics, fig=fig)
                    # Update title to include Testing prefix and metrics
                    current_title = ax.get_title()
                    ax.set_title(f'Testing - {current_title}', fontsize=10, fontweight='bold')
                    plot_idx += 1
            else:  # Timeseries
                # Get selected timeseries group
                selected_group_name = self.timeseries_graph_dropdown.value
                obs_groups_map = {
                    'BHP (All Injectors)': (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'),
                    'Gas Production (All Producers)': (list(range(3, 6)), ['Gas Prod1', 'Gas Prod2', 'Gas Prod3'], 'ft3/day'),
                    'Water Production (All Producers)': (list(range(6, 9)), ['Water Prod1', 'Water Prod2', 'Water Prod3'], 'ft3/day')
                }
                obs_indices, well_names, unit = obs_groups_map.get(selected_group_name, (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'))
                
                # Get selected well (None if "All Wells")
                selected_well = self.timeseries_well_dropdown.value
                obs_idx = None
                well_name_for_title = selected_group_name
                if selected_well != 'All Wells' and selected_well in well_names:
                    well_idx = well_names.index(selected_well)
                    obs_idx = obs_indices[well_idx]
                    # Remove parenthetical part (e.g., "(All Injectors)") when specific well is selected
                    # "BHP (All Injectors)" -> "BHP", "Water Production (All Producers)" -> "Water Production"
                    group_base = selected_group_name.split(' (')[0]
                    well_name_for_title = f"{group_base} - {selected_well}"
                
                # Get metrics calculation mode
                use_averaged_metrics = (self.metrics_mode_dropdown.value == 'Averaged')
                
                # Calculate metrics for selected group/well
                timeseries_metrics_test, timeseries_data_test = self._calculate_overall_timeseries_metrics_optimized(
                    case_indices=test_indices, selected_metrics=selected_metrics,
                    obs_group_indices=obs_indices if obs_idx is None else None,
                    use_training_data=False, obs_idx=obs_idx,
                    use_averaged_metrics=use_averaged_metrics
                )
                
                if timeseries_metrics_test is not None and timeseries_data_test is not None:
                    ax = axes[plot_idx]
                    # Handle grouped data (when obs_idx is None - All Wells)
                    if isinstance(timeseries_data_test, dict) and 'observations' in timeseries_data_test:
                        # Observation-grouped data for color-coding
                        obs_data_dict = timeseries_data_test['observations']
                        all_true_combined = timeseries_data_test.get('all_true', np.array([]))
                        all_pred_combined = timeseries_data_test.get('all_pred', np.array([]))
                        # Create obs_names_map for legend
                        obs_names_map = {}
                        for obs_idx in obs_indices:
                            if obs_idx < len(self.obs_names):
                                obs_names_map[obs_idx] = self.obs_names[obs_idx]
                        self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                            timeseries_metrics_test, all_true_combined, all_pred_combined,
                                                            selected_metrics=selected_metrics,
                                                            obs_data_dict=obs_data_dict, obs_names_map=obs_names_map, fig=fig)
                    else:
                        # Single observation data
                        all_true, all_pred = timeseries_data_test
                        self._plot_overall_timeseries_metric(ax, well_name_for_title, unit,
                                                            timeseries_metrics_test, all_true, all_pred,
                                                            selected_metrics=selected_metrics, fig=fig)
                    # Update title to include Testing prefix and metrics
                    current_title = ax.get_title()
                    ax.set_title(f'Testing - {current_title}', fontsize=10, fontweight='bold')
                    plot_idx += 1
        
        # -----------------------------------------------------------
        # Row 2: Latent-based predictions (Timeseries only)
        # -----------------------------------------------------------
        if has_latent:
            latent_axes = axes_2d[1]
            latent_plot_idx = 0

            # Re-use the same obs selection variables computed above
            selected_group_name_l = self.timeseries_graph_dropdown.value
            obs_groups_map_l = {
                'BHP (All Injectors)': (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'),
                'Gas Production (All Producers)': (list(range(3, 6)), ['Gas Prod1', 'Gas Prod2', 'Gas Prod3'], 'ft3/day'),
                'Water Production (All Producers)': (list(range(6, 9)), ['Water Prod1', 'Water Prod2', 'Water Prod3'], 'ft3/day')
            }
            obs_indices_l, well_names_l, unit_l = obs_groups_map_l.get(
                selected_group_name_l, (list(range(3)), ['BHP1', 'BHP2', 'BHP3'], 'psi'))
            selected_well_l = self.timeseries_well_dropdown.value
            obs_idx_l = None
            well_name_for_title_l = selected_group_name_l
            if selected_well_l != 'All Wells' and selected_well_l in well_names_l:
                well_idx_l = well_names_l.index(selected_well_l)
                obs_idx_l = obs_indices_l[well_idx_l]
                group_base_l = selected_group_name_l.split(' (')[0]
                well_name_for_title_l = f"{group_base_l} - {selected_well_l}"
            use_averaged_metrics_l = (self.metrics_mode_dropdown.value == 'Averaged')

            # --- Training latent ---
            if show_train:
                metrics_lt, data_lt = self._calculate_overall_timeseries_metrics_optimized(
                    case_indices=train_indices, selected_metrics=selected_metrics,
                    obs_group_indices=obs_indices_l if obs_idx_l is None else None,
                    use_training_data=True, obs_idx=obs_idx_l,
                    use_averaged_metrics=use_averaged_metrics_l,
                    yobs_pred_override=self.yobs_pred_latent_based
                )
                if metrics_lt is not None and data_lt is not None:
                    ax = latent_axes[latent_plot_idx]
                    if isinstance(data_lt, dict) and 'observations' in data_lt:
                        obs_data_dict = data_lt['observations']
                        all_true_c = data_lt.get('all_true', np.array([]))
                        all_pred_c = data_lt.get('all_pred', np.array([]))
                        obs_names_map = {oi: self.obs_names[oi] for oi in obs_indices_l if oi < len(self.obs_names)}
                        self._plot_overall_timeseries_metric(ax, well_name_for_title_l, unit_l,
                                                            metrics_lt, all_true_c, all_pred_c,
                                                            selected_metrics=selected_metrics,
                                                            obs_data_dict=obs_data_dict,
                                                            obs_names_map=obs_names_map, fig=fig)
                    elif isinstance(data_lt, tuple) and len(data_lt) == 2:
                        self._plot_overall_timeseries_metric(ax, well_name_for_title_l, unit_l,
                                                            metrics_lt, data_lt[0], data_lt[1],
                                                            selected_metrics=selected_metrics, fig=fig)
                    current_title = ax.get_title()
                    ax.set_title(f'Training (Latent-based) - {current_title}', fontsize=10, fontweight='bold')
                    latent_plot_idx += 1

            # --- Testing latent ---
            if show_test:
                metrics_lt, data_lt = self._calculate_overall_timeseries_metrics_optimized(
                    case_indices=test_indices, selected_metrics=selected_metrics,
                    obs_group_indices=obs_indices_l if obs_idx_l is None else None,
                    use_training_data=False, obs_idx=obs_idx_l,
                    use_averaged_metrics=use_averaged_metrics_l,
                    yobs_pred_override=self.yobs_pred_latent_based
                )
                if metrics_lt is not None and data_lt is not None:
                    ax = latent_axes[latent_plot_idx]
                    if isinstance(data_lt, dict) and 'observations' in data_lt:
                        obs_data_dict = data_lt['observations']
                        all_true_c = data_lt.get('all_true', np.array([]))
                        all_pred_c = data_lt.get('all_pred', np.array([]))
                        obs_names_map = {oi: self.obs_names[oi] for oi in obs_indices_l if oi < len(self.obs_names)}
                        self._plot_overall_timeseries_metric(ax, well_name_for_title_l, unit_l,
                                                            metrics_lt, all_true_c, all_pred_c,
                                                            selected_metrics=selected_metrics,
                                                            obs_data_dict=obs_data_dict,
                                                            obs_names_map=obs_names_map, fig=fig)
                    elif isinstance(data_lt, tuple) and len(data_lt) == 2:
                        self._plot_overall_timeseries_metric(ax, well_name_for_title_l, unit_l,
                                                            metrics_lt, data_lt[0], data_lt[1],
                                                            selected_metrics=selected_metrics, fig=fig)
                    current_title = ax.get_title()
                    ax.set_title(f'Testing (Latent-based) - {current_title}', fontsize=10, fontweight='bold')
                    latent_plot_idx += 1

            # Hide unused latent subplots
            for i in range(latent_plot_idx, num_cols):
                latent_axes[i].set_visible(False)

            # Format latent axes
            for ax in latent_axes[:latent_plot_idx]:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # Store metrics for saving (store as dict for compatibility)
        if graph_type == 'Spatial':
            if show_train and spatial_metrics_train is not None:
                self.latest_spatial_metrics = [spatial_metrics_train]
            elif show_test and spatial_metrics_test is not None:
                self.latest_spatial_metrics = [spatial_metrics_test]
            else:
                self.latest_spatial_metrics = None
            self.latest_timeseries_metrics = None
        else:
            self.latest_spatial_metrics = None
            if show_train and timeseries_metrics_train is not None:
                self.latest_timeseries_metrics = [timeseries_metrics_train]
            elif show_test and timeseries_metrics_test is not None:
                self.latest_timeseries_metrics = [timeseries_metrics_test]
            else:
                self.latest_timeseries_metrics = None
        
        self.latest_selected_metrics = selected_metrics
        case_types_list = []
        if show_test:
            case_types_list.append('Testing')
        if show_train:
            case_types_list.append('Training')
        self.latest_case_types = case_types_list
        
        # Hide unused state-based subplots (row 0)
        for i in range(plot_idx, num_cols):
            axes[i].set_visible(False)
        
        # Apply scientific notation formatting to state-based row
        for ax in axes[:plot_idx]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Synchronize axis limits across all visible plots so they are comparable
        all_visible_axes = list(axes[:plot_idx])
        if has_latent:
            all_visible_axes += list(axes_2d[1][:latent_plot_idx])
        if len(all_visible_axes) >= 2:
            x_min = min(a.get_xlim()[0] for a in all_visible_axes)
            x_max = max(a.get_xlim()[1] for a in all_visible_axes)
            y_min = min(a.get_ylim()[0] for a in all_visible_axes)
            y_max = max(a.get_ylim()[1] for a in all_visible_axes)
            for a in all_visible_axes:
                a.set_xlim(x_min, x_max)
                a.set_ylim(y_min, y_max)
        
        # Update state-based titles to clarify prediction mode when latent row is shown
        if has_latent:
            for ax in axes[:plot_idx]:
                t = ax.get_title()
                if t.startswith('Training - ') and '(State-based)' not in t:
                    ax.set_title(t.replace('Training - ', 'Training (State-based) - '), fontsize=10, fontweight='bold')
                elif t.startswith('Testing - ') and '(State-based)' not in t:
                    ax.set_title(t.replace('Testing - ', 'Testing (State-based) - '), fontsize=10, fontweight='bold')
        
        # Main title
        title_parts = []
        if show_test:
            title_parts.append('Testing')
        if show_train:
            title_parts.append('Training')
        title_suffix = ' & '.join(title_parts) if title_parts else 'Selected Cases'
        
        if graph_type == 'Spatial':
            selected_field_name = self.spatial_graph_dropdown.value
            selected_layer = self.spatial_layer_dropdown.value
            if selected_layer != 'All Layers':
                selected_graph = f"{selected_field_name} - {selected_layer}"
            else:
                selected_graph = selected_field_name
        else:
            selected_group_name = self.timeseries_graph_dropdown.value
            selected_well = self.timeseries_well_dropdown.value
            if selected_well != 'All Wells':
                group_base = selected_group_name.split(' (')[0]
                selected_graph = f"{group_base} - {selected_well}"
            else:
                selected_graph = selected_group_name
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88 if has_latent else 0.82,
                            hspace=0.55 if has_latent else 0.3,
                            wspace=0.15, right=0.85)
        
        fig.suptitle(f'Overall Model Performance - {selected_graph} ({title_suffix})', 
                     fontsize=16, fontweight='bold', y=0.98 if has_latent else 0.96)
        
        display(fig)
        try:
            plt.close(fig)
        except (AttributeError, RuntimeError):
            pass
    
    def _generate_cache_key(self, case_indices, selected_metrics, field_idx, obs_group_indices_or_obs_idx, use_training_data, metric_type, layer_idx=None):
        """
        Generate a cache key for metrics computation.
        
        Args:
            case_indices: Array of case indices
            selected_metrics: List of selected metrics
            field_idx: Field index (for spatial) or None
            obs_group_indices_or_obs_idx: Observation group indices (for timeseries) or single obs_idx
            use_training_data: Whether using training data
            metric_type: 'spatial' or 'timeseries'
            layer_idx: Layer index (for spatial) or None
            
        Returns:
            Cache key string
        """
        import hashlib
        # Create a hashable representation
        case_str = str(sorted(case_indices.tolist() if hasattr(case_indices, 'tolist') else case_indices))
        metrics_str = str(sorted(selected_metrics)) if selected_metrics else 'all'
        field_str = str(field_idx) if field_idx is not None else 'all'
        layer_str = str(layer_idx) if layer_idx is not None else 'all'
        # Handle both obs_group_indices (list) and obs_idx (single int)
        if isinstance(obs_group_indices_or_obs_idx, (list, tuple)):
            obs_str = str(sorted(obs_group_indices_or_obs_idx))
        elif obs_group_indices_or_obs_idx is not None:
            obs_str = str(obs_group_indices_or_obs_idx)  # Single obs_idx
        else:
            obs_str = 'all'
        train_str = str(use_training_data)
        
        key_str = f"{metric_type}_{case_str}_{metrics_str}_{field_str}_{layer_str}_{obs_str}_{train_str}"
        # Use hash for shorter key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear_metrics_cache(self):
        """Clear the metrics cache. Useful when data or parameters change."""
        self._overall_metrics_cache.clear()
        if hasattr(self, 'metrics_evaluator'):
            self.metrics_evaluator.spatial_metrics_cache.clear()
            self.metrics_evaluator.timeseries_metrics_cache.clear()
        
        # Also clear numpy conversion caches
        self._train_state_pred_np_cache = None
        self._train_state_true_np_cache = None
        self._train_yobs_pred_np_cache = None
        self._train_yobs_true_np_cache = None
        self._test_state_pred_np_cache = None
        self._test_state_true_np_cache = None
        self._test_yobs_pred_np_cache = None
        self._test_yobs_true_np_cache = None
    
    def enable_metrics_cache(self, enable=True):
        """Enable or disable metrics caching."""
        self._cache_enabled = enable
        if not enable:
            self.clear_metrics_cache()
    
    def _batch_denormalize_spatial(self, data_flat, field_idx, norm_params):
        """
        Batch denormalize spatial field data using vectorized operations.
        
        Args:
            data_flat: Flattened array of normalized data
            field_idx: Field/channel index
            norm_params: Normalization parameters dictionary
            
        Returns:
            Denormalized data array
        """
        if not norm_params or field_idx >= len(self.field_names):
            return data_flat
        
        field_key = self.field_keys[field_idx] if hasattr(self, 'field_keys') and field_idx < len(self.field_keys) else self.field_names[field_idx].upper()
        
        if field_key not in norm_params:
            return data_flat
        
        params = norm_params[field_key]
        
        if params.get('type') == 'none':
            return data_flat
        elif params.get('type') == 'log':
            log_min = params['log_min']
            log_max = params['log_max']
            epsilon = params.get('epsilon', 1e-8)
            data_shift = params.get('data_shift', 0)
            
            # Vectorized log denormalization
            data_log = data_flat * (log_max - log_min) + log_min
            return np.exp(data_log) - epsilon + data_shift
        else:
            # Standard min-max denormalization
            field_min = params['min']
            field_max = params['max']
            return data_flat * (field_max - field_min) + field_min
    
    def _batch_denormalize_timeseries(self, data_flat, obs_idx, norm_params):
        """
        Batch denormalize timeseries observation data using vectorized operations.
        
        Args:
            data_flat: Flattened array of normalized data
            obs_idx: Observation variable index
            norm_params: Normalization parameters dictionary
            
        Returns:
            Denormalized data array
        """
        if not norm_params:
            return data_flat
        
        params = None
        if obs_idx < 3:  # BHP data
            if 'BHP' in norm_params:
                params = norm_params['BHP']
        elif obs_idx < 6:  # Gas production
            if 'GASRATSC' in norm_params:
                params = norm_params['GASRATSC']
        else:  # Water production
            if 'WATRATSC' in norm_params:
                params = norm_params['WATRATSC']
        
        if params is None or params.get('type') == 'none':
            return data_flat
        
        if params.get('type') == 'log':
            log_min = params['log_min']
            log_max = params['log_max']
            epsilon = params.get('epsilon', 1e-8)
            data_shift = params.get('data_shift', 0)
            
            # Vectorized log denormalization
            data_log = data_flat * (log_max - log_min) + log_min
            return np.exp(data_log) - epsilon + data_shift
        else:
            # Standard min-max denormalization
            obs_min = params['min']
            obs_max = params['max']
            return data_flat * (obs_max - obs_min) + obs_min
    
    def _calculate_overall_spatial_metrics_optimized(self, case_indices=None, selected_metrics=None, 
                                                      field_idx=None, use_training_data=False, layer_idx=None,
                                                      use_averaged_metrics=False):
        """
        Optimized version: Calculate overall spatial metrics using vectorization with caching.
        
        Args:
            case_indices: Array of case indices to include (None = all cases)
            selected_metrics: List of metric names to calculate (None = all)
            field_idx: Optional specific field index to calculate (None = all fields)
            use_training_data: If True, use training data instead of test data
            layer_idx: Optional specific layer index to calculate (None = all layers)
            use_averaged_metrics: If True, calculate metrics for each (case, layer, timestep) combination and average them.
                                 If False, aggregate all data points and calculate metrics (default).
        
        Returns:
            overall_metrics: List of metric dictionaries (or single dict if field_idx specified)
            all_data: List of (true, pred) arrays for plotting (or single tuple if field_idx specified)
        """
        if selected_metrics is None:
            selected_metrics = ['r2', 'mse', 'rmse', 'mae']
        
        # Select data source: always use combined state_pred array, split by indices
        # Get training/test indices based on per-100-case split pattern
        selected_indices = None
        if self.state_pred is not None:
            total_cases = self.state_pred.shape[0]
            train_indices_all, test_indices_all = self._calculate_train_test_indices_from_total(total_cases)
            
            # Use cached numpy arrays if available, otherwise convert and cache
            if self._test_state_pred_np_cache is None:
                self._test_state_pred_np_cache = self.state_pred.cpu().detach().numpy() if torch.is_tensor(self.state_pred) else self.state_pred
            if self._test_state_true_np_cache is None:
                self._test_state_true_np_cache = self.state_seq_true_aligned.cpu().detach().numpy() if torch.is_tensor(self.state_seq_true_aligned) else self.state_seq_true_aligned
            
            # Select indices based on use_training_data flag
            if use_training_data:
                selected_indices = train_indices_all
            else:
                selected_indices = test_indices_all
            
            # Use the combined array with selected indices
            state_pred_np = self._test_state_pred_np_cache
            state_true_np = self._test_state_true_np_cache
        else:
            # Fallback: use separate arrays if state_pred not available
            if use_training_data:
                if self.train_state_pred is None:
                    return None, None
                if self._train_state_pred_np_cache is None:
                    self._train_state_pred_np_cache = self.train_state_pred.cpu().detach().numpy()
                if self._train_state_true_np_cache is None:
                    self._train_state_true_np_cache = self.train_state_seq_true_aligned.cpu().detach().numpy()
                state_pred_np = self._train_state_pred_np_cache
                state_true_np = self._train_state_true_np_cache
                selected_indices = None  # Use all cases from separate array
            else:
                if self._test_state_pred_np_cache is None:
                    self._test_state_pred_np_cache = self.state_pred.cpu().detach().numpy() if torch.is_tensor(self.state_pred) else self.state_pred
                if self._test_state_true_np_cache is None:
                    self._test_state_true_np_cache = self.state_seq_true_aligned.cpu().detach().numpy() if torch.is_tensor(self.state_seq_true_aligned) else self.state_seq_true_aligned
                state_pred_np = self._test_state_pred_np_cache
                state_true_np = self._test_state_true_np_cache
                selected_indices = None  # Use all cases from separate array
        
        # Determine case indices: use selected_indices from combined array if available
        if selected_indices is not None:
            # Using combined array - case_indices should be relative to selected_indices
            if case_indices is None:
                # Use all cases from the selected group (training or test)
                case_indices = selected_indices
            else:
                # case_indices are provided - map them to actual indices in combined array
                case_indices = np.array(case_indices)
                # If indices are relative to the selected group, map them
                if len(case_indices) > 0 and np.max(case_indices) < len(selected_indices):
                    case_indices = selected_indices[case_indices]
                else:
                    # Assume indices are already absolute indices in combined array
                    # Filter to only include indices that are in the selected group
                    case_indices = case_indices[np.isin(case_indices, selected_indices)]
        else:
            # Fallback: using separate arrays
            if case_indices is None:
                if use_training_data:
                    case_indices = np.arange(self.num_train_case)
                else:
                    case_indices = np.arange(self.num_case)
            else:
                case_indices = np.array(case_indices)
                # If training data is requested but case_indices are provided, ensure they're valid
                if use_training_data and len(case_indices) > 0:
                    max_idx = np.max(case_indices)
                    if max_idx >= self.num_train_case:
                        print(f"⚠️ Warning: Some case indices ({max_idx}) exceed training data size ({self.num_train_case})")
                        case_indices = case_indices[case_indices < self.num_train_case]
        
        # Check cache if enabled
        if self._cache_enabled:
            cache_key = self._generate_cache_key(case_indices, selected_metrics, field_idx, None, use_training_data, 'spatial', layer_idx=layer_idx)
            if cache_key in self._overall_metrics_cache:
                return self._overall_metrics_cache[cache_key]
        
        overall_metrics = []
        all_data = []
        
        # Determine which fields to process
        if field_idx is not None:
            field_indices = [field_idx]
        else:
            field_indices = range(len(self.field_names))
        
        # Optimize: Use basic slicing when all cases from selected group are selected (much faster than advanced indexing)
        # Check if case_indices covers all cases in the selected group (training or test)
        if selected_indices is not None:
            # Using combined array - check if all selected indices are used
            use_all_cases_spatial = (len(case_indices) == len(selected_indices) and 
                                     np.array_equal(np.sort(case_indices), np.sort(selected_indices)))
        else:
            # Using separate arrays - check if all cases are used
            use_all_cases_spatial = (len(case_indices) == state_pred_np.shape[0] and 
                                     np.array_equal(case_indices, np.arange(len(case_indices))))
        
        for field_idx_iter in field_indices:
            # Extract all data for this field at once: (cases, timesteps, Nx, Ny, layers)
            # state_pred: (num_case, num_tstep, n_channels, Nx, Ny, Nz)
            # state_true: (num_case, n_channels, num_tstep, Nx, Ny, Nz)
            
            # Get field data for selected cases (use basic slicing when all cases from selected group are selected)
            if use_all_cases_spatial and selected_indices is None:
                # Use basic slicing for all cases (much faster) - only when using separate arrays
                pred_field = state_pred_np[:, :, field_idx_iter, :, :, :]  # (n_cases, n_tstep, Nx, Ny, Nz)
                true_field = state_true_np[:, field_idx_iter, :, :, :, :]  # (n_cases, n_tstep, Nx, Ny, Nz)
            else:
                # Use advanced indexing for subset of cases (or all cases from selected group in combined array)
                pred_field = state_pred_np[case_indices, :, field_idx_iter, :, :, :]  # (n_cases, n_tstep, Nx, Ny, Nz)
                true_field = state_true_np[case_indices, field_idx_iter, :, :, :, :]  # (n_cases, n_tstep, Nx, Ny, Nz)
            
            # Apply masking if available (fully vectorized) - do this before flattening for efficiency
            if hasattr(self, 'masks_loaded_successfully') and self.masks_loaded_successfully:
                # Get all masks for all cases and layers at once using vectorized operation
                # Returns: (n_cases, Nx, Ny, Nz)
                mask_4d_cases = self._get_all_layer_masks_vectorized(case_indices, use_training_data)
                
                # Reshape to match pred_field shape: (n_cases, n_tstep, Nx, Ny, Nz)
                # Broadcast mask across timesteps dimension
                mask_4d = np.broadcast_to(mask_4d_cases[:, np.newaxis, :, :, :], pred_field.shape)
                
                # Apply mask vectorized: set inactive cells to NaN
                pred_field = np.where(mask_4d, pred_field, np.nan)
                true_field = np.where(mask_4d, true_field, np.nan)
            
            # Determine which layers to process
            layers_to_process = [layer_idx] if layer_idx is not None else range(self.Nz)
            
            # If averaged metrics mode, calculate metrics for each (case, layer, timestep) combination and average
            if use_averaged_metrics:
                individual_metrics_list = []
                n_combinations = 0
                
                for case_idx in case_indices:
                    for layer_idx_iter in layers_to_process:
                        for timestep_idx in range(self.num_tstep):
                            try:
                                # Get metrics for this specific combination using existing method
                                metrics_dict = self.metrics_evaluator.get_spatial_metrics(
                                    case_idx, field_idx_iter, layer_idx_iter, timestep_idx,
                                    norm_params=self.norm_params, dashboard=self
                                )
                                
                                # Extract only selected metrics
                                filtered_metrics = {}
                                for metric_name in selected_metrics:
                                    metric_key = metric_name.lower()
                                    if metric_key == 'r²' or metric_key == 'r2':
                                        metric_key = 'r2'
                                    elif metric_key == 'ae':
                                        metric_key = 'mae'
                                    if metric_key in metrics_dict:
                                        filtered_metrics[metric_name.lower()] = metrics_dict[metric_key]
                                
                                if filtered_metrics:
                                    individual_metrics_list.append(filtered_metrics)
                                    n_combinations += 1
                            except Exception as e:
                                # Skip this combination if metrics calculation fails
                                continue
                
                # Average all individual metrics
                if n_combinations > 0 and len(individual_metrics_list) > 0:
                    averaged_metrics = {}
                    for metric_name in selected_metrics:
                        metric_key = metric_name.lower()
                        if metric_key == 'r²' or metric_key == 'r2':
                            metric_key = 'r2'
                        elif metric_key == 'ae':
                            metric_key = 'mae'
                        
                        # Collect all values for this metric
                        metric_values = [m.get(metric_key, 0.0) for m in individual_metrics_list if metric_key in m]
                        if metric_values:
                            averaged_metrics[metric_name.lower()] = np.mean(metric_values)
                        else:
                            averaged_metrics[metric_name.lower()] = 0.0
                    
                    # For plotting, we still need aggregated data (use aggregated approach for data)
                    # First, denormalize the field data (same as aggregated mode)
                    pred_flat_all = pred_field.flatten()
                    true_flat_all = true_field.flatten()
                    
                    # Batch denormalize using helper method (vectorized) - do this once for all layers
                    if self.norm_params and field_idx_iter < len(self.field_names):
                        pred_flat_all = self._batch_denormalize_spatial(pred_flat_all, field_idx_iter, self.norm_params)
                        true_flat_all = self._batch_denormalize_spatial(true_flat_all, field_idx_iter, self.norm_params)
                    
                    # Reshape back to original shape after denormalization
                    pred_field_denorm = pred_flat_all.reshape(pred_field.shape)
                    true_field_denorm = true_flat_all.reshape(true_field.shape)
                    
                    # Extract aggregated data for plotting (same as aggregated mode)
                    pred_layer = pred_field_denorm[:, :, :, :, layers_to_process[0] if len(layers_to_process) == 1 else layers_to_process[0]]
                    true_layer = true_field_denorm[:, :, :, :, layers_to_process[0] if len(layers_to_process) == 1 else layers_to_process[0]]
                    pred_flat = pred_layer.flatten()
                    true_flat = true_layer.flatten()
                    valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
                    all_true = true_flat[valid_idx]
                    all_pred = pred_flat[valid_idx]
                    
                    # If multiple layers, aggregate all layers
                    if len(layers_to_process) > 1:
                        all_layers_true = []
                        all_layers_pred = []
                        layer_data_dict = {}
                        for layer_idx_iter in layers_to_process:
                            pred_layer = pred_field_denorm[:, :, :, :, layer_idx_iter]
                            true_layer = true_field_denorm[:, :, :, :, layer_idx_iter]
                            pred_layer_flat = pred_layer.flatten()
                            true_layer_flat = true_layer.flatten()
                            valid_idx = ~np.isnan(true_layer_flat) & ~np.isnan(pred_layer_flat)
                            layer_true = true_layer_flat[valid_idx]
                            layer_pred = pred_layer_flat[valid_idx]
                            layer_data_dict[layer_idx_iter] = (layer_true, layer_pred)
                            all_layers_true.append(layer_true)
                            all_layers_pred.append(layer_pred)
                        
                        if len(all_layers_true) > 0:
                            all_true = np.concatenate(all_layers_true)
                            all_pred = np.concatenate(all_layers_pred)
                            all_data.append({'layers': layer_data_dict, 'all_true': all_true, 'all_pred': all_pred})
                        else:
                            all_data.append({'layers': {}, 'all_true': np.array([]), 'all_pred': np.array([])})
                    else:
                        all_data.append((all_true, all_pred))
                    
                    overall_metrics.append(averaged_metrics)
                    continue  # Skip the aggregated calculation below
            
            # Batch denormalize the entire field once (before layer processing for efficiency)
            # Reshape to (n_cases * n_tstep * Nx * Ny * Nz,)
            pred_flat_all = pred_field.flatten()
            true_flat_all = true_field.flatten()
            
            # Batch denormalize using helper method (vectorized) - do this once for all layers
            if self.norm_params and field_idx_iter < len(self.field_names):
                pred_flat_all = self._batch_denormalize_spatial(pred_flat_all, field_idx_iter, self.norm_params)
                true_flat_all = self._batch_denormalize_spatial(true_flat_all, field_idx_iter, self.norm_params)
            
            # Reshape back to original shape after denormalization
            pred_field_denorm = pred_flat_all.reshape(pred_field.shape)
            true_field_denorm = true_flat_all.reshape(true_field.shape)
            
            # If layer_idx is None (All Layers), we need to return layer-grouped data for color-coding
            if layer_idx is None:
                # Process each layer separately for color-coding
                layer_data_dict = {}
                all_layers_true = []
                all_layers_pred = []
                
                for layer_idx_iter in layers_to_process:
                    # Extract data for this specific layer from already-denormalized data: (n_cases, n_tstep, Nx, Ny)
                    pred_layer = pred_field_denorm[:, :, :, :, layer_idx_iter]  # (n_cases, n_tstep, Nx, Ny)
                    true_layer = true_field_denorm[:, :, :, :, layer_idx_iter]  # (n_cases, n_tstep, Nx, Ny)
                    
                    # Flatten for this layer (already denormalized)
                    pred_layer_flat = pred_layer.flatten()
                    true_layer_flat = true_layer.flatten()
                    
                    # Remove NaN values (vectorized)
                    valid_idx = ~np.isnan(true_layer_flat) & ~np.isnan(pred_layer_flat)
                    layer_true = true_layer_flat[valid_idx]
                    layer_pred = pred_layer_flat[valid_idx]
                    
                    # Store layer data for color-coding
                    layer_data_dict[layer_idx_iter] = (layer_true, layer_pred)
                    
                    # Collect for overall metrics calculation
                    all_layers_true.append(layer_true)
                    all_layers_pred.append(layer_pred)
                
                # Aggregate all layers for overall metrics calculation
                if len(all_layers_true) > 0:
                    all_true_combined = np.concatenate(all_layers_true)
                    all_pred_combined = np.concatenate(all_layers_pred)
                    
                    # Calculate metrics from all aggregated data points (same as single plots)
                    if len(all_true_combined) > 0:
                        # Filter negative predictions for fraction fields (same as single plots)
                        field_name = self.field_names[field_idx_iter] if field_idx_iter < len(self.field_names) else f"Field {field_idx_iter}"
                        field_unit = self.metrics_evaluator._get_field_unit(field_name)
                        if field_unit == 'fraction':
                            valid_idx = all_pred_combined >= 0
                            all_true_combined = all_true_combined[valid_idx]
                            all_pred_combined = all_pred_combined[valid_idx]
                        
                        metrics = self.metrics_evaluator._compute_metrics(
                            all_true_combined, all_pred_combined, 
                            selected_metrics=selected_metrics,
                            filter_negative_predictions=(field_unit == 'fraction')
                        )
                        # Ensure ape is included for compatibility
                        if 'ape' not in metrics and 'ape' not in selected_metrics:
                            metrics['ape'] = self.metrics_evaluator._compute_metrics(
                                all_true_combined, all_pred_combined, 
                                selected_metrics=['ape'],
                                filter_negative_predictions=(field_unit == 'fraction')
                            )['ape']
                        filtered_metrics = {k: v for k, v in metrics.items() if k in selected_metrics or k == 'ape'}
                    else:
                        filtered_metrics = {k: 0.0 for k in selected_metrics}
                        filtered_metrics['ape'] = 0.0
                else:
                    filtered_metrics = {k: 0.0 for k in selected_metrics}
                    filtered_metrics['ape'] = 0.0
                
                # Return layer-grouped data for color-coding
                # Structure: {'layers': layer_data_dict, 'all_true': all_true_combined, 'all_pred': all_pred_combined}
                overall_metrics.append(filtered_metrics)
                if len(all_layers_true) > 0:
                    all_data.append({'layers': layer_data_dict, 'all_true': all_true_combined, 'all_pred': all_pred_combined})
                else:
                    all_data.append({'layers': {}, 'all_true': np.array([]), 'all_pred': np.array([])})
            else:
                # Single layer specified - process normally
                # Extract data for the specific layer from already-denormalized data: (n_cases, n_tstep, Nx, Ny)
                pred_layer = pred_field_denorm[:, :, :, :, layer_idx]  # (n_cases, n_tstep, Nx, Ny)
                true_layer = true_field_denorm[:, :, :, :, layer_idx]  # (n_cases, n_tstep, Nx, Ny)
                
                # Flatten (already denormalized)
                pred_flat = pred_layer.flatten()
                true_flat = true_layer.flatten()
                
                # Remove NaN values (vectorized)
                valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
                all_true = true_flat[valid_idx]
                all_pred = pred_flat[valid_idx]
                
                # Calculate metrics from all aggregated data points (same as single plots)
                if len(all_true) > 0:
                    # Filter negative predictions for fraction fields (same as single plots)
                    field_name = self.field_names[field_idx_iter] if field_idx_iter < len(self.field_names) else f"Field {field_idx_iter}"
                    field_unit = self.metrics_evaluator._get_field_unit(field_name)
                    if field_unit == 'fraction':
                        valid_idx = all_pred >= 0
                        all_true = all_true[valid_idx]
                        all_pred = all_pred[valid_idx]
                    
                    metrics = self.metrics_evaluator._compute_metrics(
                        all_true, all_pred, 
                        selected_metrics=selected_metrics,
                        filter_negative_predictions=(field_unit == 'fraction')
                    )
                    # Ensure ape is included for compatibility
                    if 'ape' not in metrics and 'ape' not in selected_metrics:
                        metrics['ape'] = self.metrics_evaluator._compute_metrics(
                            all_true, all_pred, 
                            selected_metrics=['ape'],
                            filter_negative_predictions=(field_unit == 'fraction')
                        )['ape']
                    filtered_metrics = {k: v for k, v in metrics.items() if k in selected_metrics or k == 'ape'}
                else:
                    filtered_metrics = {k: 0.0 for k in selected_metrics}
                    filtered_metrics['ape'] = 0.0
                
                overall_metrics.append(filtered_metrics)
                all_data.append((all_true, all_pred))
        
        # Return single item if field_idx specified, otherwise return list
        # Note: When layer_idx is None, all_data contains (layer_data_dict, combined_data) tuple
        # When layer_idx is specified, all_data contains (all_true, all_pred) tuple
        result = (overall_metrics[0] if overall_metrics else None, all_data[0] if all_data else None) if field_idx is not None else (overall_metrics, all_data)
        
        # Cache result if enabled
        if self._cache_enabled:
            cache_key = self._generate_cache_key(case_indices, selected_metrics, field_idx, None, use_training_data, 'spatial', layer_idx=layer_idx)
            self._overall_metrics_cache[cache_key] = result
        
        return result
    
    def _calculate_overall_spatial_metrics(self):
        """Legacy method - calls optimized version"""
        return self._calculate_overall_spatial_metrics_optimized()
    
    def _calculate_overall_timeseries_metrics_optimized(self, case_indices=None, selected_metrics=None,
                                                         obs_group_indices=None, use_training_data=False, obs_idx=None,
                                                         use_averaged_metrics=False,
                                                         yobs_pred_override=None):
        """
        Optimized version: Calculate overall timeseries metrics using vectorization.
        
        Args:
            case_indices: Array of case indices to include (None = all cases)
            selected_metrics: List of metric names to calculate (None = all)
            obs_group_indices: Optional list of observation indices for a specific group (None = all observations)
            use_training_data: If True, use training data instead of test data
            obs_idx: Optional specific observation index to calculate (None = all observations in group)
            use_averaged_metrics: If True, calculate metrics for each case and average them.
                                 If False, aggregate all data points and calculate metrics (default).
            yobs_pred_override: Optional tensor/array to use instead of self.yobs_pred
                                (e.g. latent-based predictions for comparison).
        
        Returns:
            overall_metrics: List of metric dictionaries (or single dict if obs_group_indices specified)
            all_data: List of (true, pred) arrays for plotting (or single tuple if obs_group_indices specified)
        """
        if selected_metrics is None:
            selected_metrics = ['r2', 'mse', 'rmse', 'mae']
        
        # Determine prediction source (override or default)
        pred_source = yobs_pred_override if yobs_pred_override is not None else self.yobs_pred
        
        # Select data source: always use combined yobs_pred array, split by indices
        # Get training/test indices based on per-100-case split pattern
        selected_indices = None
        if pred_source is not None:
            total_cases = pred_source.shape[0]
            train_indices_all, test_indices_all = self._calculate_train_test_indices_from_total(total_cases)
            
            # Convert to numpy (skip cache when using override)
            if yobs_pred_override is not None:
                yobs_pred_np = pred_source.cpu().detach().numpy() if torch.is_tensor(pred_source) else pred_source
            else:
                if self._test_yobs_pred_np_cache is None:
                    self._test_yobs_pred_np_cache = pred_source.cpu().detach().numpy() if torch.is_tensor(pred_source) else pred_source
                yobs_pred_np = self._test_yobs_pred_np_cache
            if self._test_yobs_true_np_cache is None:
                self._test_yobs_true_np_cache = self.yobs_seq_true.cpu().detach().numpy() if torch.is_tensor(self.yobs_seq_true) else self.yobs_seq_true
            
            # Select indices based on use_training_data flag
            if use_training_data:
                selected_indices = train_indices_all
            else:
                selected_indices = test_indices_all
            
            # Use the combined array with selected indices
            yobs_true_np = self._test_yobs_true_np_cache
        else:
            # Fallback: use separate arrays if yobs_pred not available
            if use_training_data:
                if self.train_yobs_pred is None:
                    return None, None
                if self._train_yobs_pred_np_cache is None:
                    self._train_yobs_pred_np_cache = self.train_yobs_pred.cpu().detach().numpy()
                if self._train_yobs_true_np_cache is None:
                    self._train_yobs_true_np_cache = self.train_yobs_seq_true.cpu().detach().numpy()
                yobs_pred_np = self._train_yobs_pred_np_cache
                yobs_true_np = self._train_yobs_true_np_cache
                selected_indices = None  # Use all cases from separate array
            else:
                if self._test_yobs_pred_np_cache is None:
                    self._test_yobs_pred_np_cache = self.yobs_pred.cpu().detach().numpy() if torch.is_tensor(self.yobs_pred) else self.yobs_pred
                if self._test_yobs_true_np_cache is None:
                    self._test_yobs_true_np_cache = self.yobs_seq_true.cpu().detach().numpy() if torch.is_tensor(self.yobs_seq_true) else self.yobs_seq_true
                yobs_pred_np = self._test_yobs_pred_np_cache
                yobs_true_np = self._test_yobs_true_np_cache
                selected_indices = None  # Use all cases from separate array
        
        # Determine case indices: use selected_indices from combined array if available
        if selected_indices is not None:
            # Using combined array - case_indices should be relative to selected_indices
            if case_indices is None:
                # Use all cases from the selected group (training or test)
                case_indices = selected_indices
            else:
                # case_indices are provided - map them to actual indices in combined array
                case_indices = np.array(case_indices)
                # If indices are relative to the selected group, map them
                if len(case_indices) > 0 and np.max(case_indices) < len(selected_indices):
                    case_indices = selected_indices[case_indices]
                else:
                    # Assume indices are already absolute indices in combined array
                    # Filter to only include indices that are in the selected group
                    case_indices = case_indices[np.isin(case_indices, selected_indices)]
        else:
            # Fallback: using separate arrays
            if case_indices is None:
                if use_training_data:
                    case_indices = np.arange(self.num_train_case)
                else:
                    case_indices = np.arange(self.num_case)
            else:
                case_indices = np.array(case_indices)
                # If training data is requested but case_indices are provided, ensure they're valid
                if use_training_data and len(case_indices) > 0:
                    max_idx = np.max(case_indices)
                    if max_idx >= self.num_train_case:
                        print(f"⚠️ Warning: Some case indices ({max_idx}) exceed training data size ({self.num_train_case})")
                        case_indices = case_indices[case_indices < self.num_train_case]
        
        # Check cache if enabled (skip cache when override is active — different data source)
        use_cache = self._cache_enabled and yobs_pred_override is None
        if use_cache:
            cache_key = self._generate_cache_key(case_indices, selected_metrics, None, obs_group_indices, use_training_data, 'timeseries', layer_idx=None)
            if cache_key in self._overall_metrics_cache:
                return self._overall_metrics_cache[cache_key]
        
        overall_metrics = []
        all_data = []
        
        # Determine which observations to process
        if obs_idx is not None:
            obs_indices_to_process = [obs_idx]
        elif obs_group_indices is not None:
            obs_indices_to_process = obs_group_indices
        else:
            obs_indices_to_process = range(len(self.obs_names))
        
        # Optimize: Use basic slicing when all cases from selected group are selected
        if selected_indices is not None:
            use_all_cases = (len(case_indices) == len(selected_indices) and 
                            np.array_equal(np.sort(case_indices), np.sort(selected_indices)))
        else:
            use_all_cases = (len(case_indices) == yobs_pred_np.shape[0] and 
                            np.array_equal(case_indices, np.arange(len(case_indices))))
        
        # If averaged metrics mode, calculate metrics for each case and average
        if use_averaged_metrics:
            individual_metrics_list = []
            n_combinations = 0
            
            for case_idx in case_indices:
                for obs_idx_iter in obs_indices_to_process:
                    try:
                        # Compute metrics directly from the numpy arrays so that
                        # yobs_pred_override is respected (the evaluator's internal
                        # method only knows about state-based predictions).
                        pred_vals = yobs_pred_np[case_idx, :, obs_idx_iter]
                        true_vals = yobs_true_np[case_idx, obs_idx_iter, :]
                        if self.norm_params:
                            pred_vals = self._batch_denormalize_timeseries(pred_vals, obs_idx_iter, self.norm_params)
                            true_vals = self._batch_denormalize_timeseries(true_vals, obs_idx_iter, self.norm_params)
                        pred_vals = np.maximum(pred_vals, 0.0)
                        valid = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
                        if np.sum(valid) == 0:
                            continue
                        metrics_dict = self.metrics_evaluator._compute_metrics(
                            true_vals[valid], pred_vals[valid], selected_metrics=selected_metrics
                        )
                        
                        # Extract only selected metrics
                        filtered_metrics = {}
                        for metric_name in selected_metrics:
                            metric_key = metric_name.lower()
                            if metric_key == 'r²' or metric_key == 'r2':
                                metric_key = 'r2'
                            elif metric_key == 'ae':
                                metric_key = 'mae'
                            if metric_key in metrics_dict:
                                filtered_metrics[metric_name.lower()] = metrics_dict[metric_key]
                        
                        if filtered_metrics:
                            individual_metrics_list.append(filtered_metrics)
                            n_combinations += 1
                    except Exception as e:
                        # Skip this combination if metrics calculation fails
                        continue
            
            # Average all individual metrics
            if n_combinations > 0 and len(individual_metrics_list) > 0:
                averaged_metrics = {}
                for metric_name in selected_metrics:
                    metric_key = metric_name.lower()
                    if metric_key == 'r²' or metric_key == 'r2':
                        metric_key = 'r2'
                    elif metric_key == 'ae':
                        metric_key = 'mae'
                    
                    # Collect all values for this metric
                    metric_values = [m.get(metric_key, 0.0) for m in individual_metrics_list if metric_key in m]
                    if metric_values:
                        averaged_metrics[metric_name.lower()] = np.mean(metric_values)
                    else:
                        averaged_metrics[metric_name.lower()] = 0.0
                
                # For plotting, we still need aggregated data (use aggregated approach for data)
                # Extract aggregated data for plotting (same as aggregated mode)
                if len(obs_indices_to_process) == 1:
                    obs_idx_for_data = obs_indices_to_process[0]
                    if use_all_cases and selected_indices is None:
                        pred_obs = yobs_pred_np[:, :, obs_idx_for_data]
                        true_obs = yobs_true_np[:, obs_idx_for_data, :]
                    else:
                        pred_obs = yobs_pred_np[case_indices, :, obs_idx_for_data]
                        true_obs = yobs_true_np[case_indices, obs_idx_for_data, :]
                    pred_flat = pred_obs.flatten()
                    true_flat = true_obs.flatten()
                    
                    # Denormalize
                    if self.norm_params:
                        pred_flat = self._batch_denormalize_timeseries(pred_flat, obs_idx_for_data, self.norm_params)
                        true_flat = self._batch_denormalize_timeseries(true_flat, obs_idx_for_data, self.norm_params)
                    
                    pred_flat = np.maximum(pred_flat, 0.0)
                    valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
                    all_true = true_flat[valid_idx]
                    all_pred = pred_flat[valid_idx]
                    all_data.append((all_true, all_pred))
                else:
                    # Multiple observations - create grouped data structure
                    obs_data_dict = {}
                    all_obs_true = []
                    all_obs_pred = []
                    for obs_idx_iter in obs_indices_to_process:
                        if use_all_cases and selected_indices is None:
                            pred_obs = yobs_pred_np[:, :, obs_idx_iter]
                            true_obs = yobs_true_np[:, obs_idx_iter, :]
                        else:
                            pred_obs = yobs_pred_np[case_indices, :, obs_idx_iter]
                            true_obs = yobs_true_np[case_indices, obs_idx_iter, :]
                        pred_flat = pred_obs.flatten()
                        true_flat = true_obs.flatten()
                        
                        # Denormalize
                        if self.norm_params:
                            pred_flat = self._batch_denormalize_timeseries(pred_flat, obs_idx_iter, self.norm_params)
                            true_flat = self._batch_denormalize_timeseries(true_flat, obs_idx_iter, self.norm_params)
                        
                        pred_flat = np.maximum(pred_flat, 0.0)
                        valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
                        obs_true = true_flat[valid_idx]
                        obs_pred = pred_flat[valid_idx]
                        obs_data_dict[obs_idx_iter] = (obs_true, obs_pred)
                        all_obs_true.append(obs_true)
                        all_obs_pred.append(obs_pred)
                    
                    if len(all_obs_true) > 0:
                        all_true = np.concatenate(all_obs_true)
                        all_pred = np.concatenate(all_obs_pred)
                        all_data.append({'observations': obs_data_dict, 'all_true': all_true, 'all_pred': all_pred})
                    else:
                        all_data.append({'observations': {}, 'all_true': np.array([]), 'all_pred': np.array([])})
                
                overall_metrics.append(averaged_metrics)
                # Return structure should match aggregated mode: single dict and single data item
                if len(overall_metrics) > 0 and len(all_data) > 0:
                    return overall_metrics[0], all_data[0]  # Return single dict and single data item
                else:
                    # Return empty structure matching expected format
                    if len(overall_metrics) > 0:
                        # Return metrics but empty data structure
                        if len(obs_indices_to_process) == 1:
                            return overall_metrics[0], (np.array([]), np.array([]))
                        else:
                            return overall_metrics[0], {'observations': {}, 'all_true': np.array([]), 'all_pred': np.array([])}
                    else:
                        return None, None
        
        # Optimize: Extract all observations at once if processing a group (faster for large datasets)
        # If obs_idx is specified, we're processing a single observation
        if obs_idx is not None:
            # Process single observation
            obs_indices_to_process = [obs_idx]
            # Extract single observation
            if use_all_cases and selected_indices is None:
                pred_obs = yobs_pred_np[:, :, obs_idx]  # (n_cases, n_tstep)
                true_obs = yobs_true_np[:, obs_idx, :]  # (n_cases, n_tstep)
            else:
                pred_obs = yobs_pred_np[case_indices, :, obs_idx]  # (n_cases, n_tstep)
                true_obs = yobs_true_np[case_indices, obs_idx, :]  # (n_cases, n_tstep)
            
            # Flatten for plotting
            pred_flat = pred_obs.flatten()
            true_flat = true_obs.flatten()
            
            # Batch denormalize using helper method (vectorized)
            if self.norm_params:
                pred_flat = self._batch_denormalize_timeseries(pred_flat, obs_idx, self.norm_params)
                true_flat = self._batch_denormalize_timeseries(true_flat, obs_idx, self.norm_params)
            
            # Ensure non-negative observations
            pred_flat = np.maximum(pred_flat, 0.0)
            
            # Remove NaN values (vectorized)
            valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
            all_true = true_flat[valid_idx]
            all_pred = pred_flat[valid_idx]
            
            # Calculate metrics from all aggregated data points (same as single plots)
            if len(all_true) > 0:
                metrics = self.metrics_evaluator._compute_metrics(all_true, all_pred, selected_metrics=selected_metrics)
                if 'ape' not in metrics and 'ape' not in selected_metrics:
                    metrics['ape'] = self.metrics_evaluator._compute_metrics(all_true, all_pred, selected_metrics=['ape'])['ape']
                averaged_metrics = {k: v for k, v in metrics.items() if k in selected_metrics or k == 'ape'}
            else:
                averaged_metrics = {k: 0.0 for k in selected_metrics}
                averaged_metrics['ape'] = 0.0
            
            overall_metrics.append(averaged_metrics)
            all_data.append((all_true, all_pred))
            
            # Return single observation result
            result = (overall_metrics[0] if overall_metrics else None, all_data[0] if all_data else None)
            
            if use_cache:
                cache_key = self._generate_cache_key(case_indices, selected_metrics, None, obs_idx, use_training_data, 'timeseries', layer_idx=None)
                self._overall_metrics_cache[cache_key] = result
            
            return result
        
        elif obs_group_indices is not None and len(obs_group_indices) > 1:
            if use_all_cases and selected_indices is None:
                # Use basic slicing for all cases (much faster) - only when using separate arrays
                pred_group = yobs_pred_np[:, :, obs_group_indices]  # (n_cases, n_tstep, n_obs_in_group)
                true_group = yobs_true_np[:, obs_group_indices, :]  # (n_cases, n_obs_in_group, n_tstep)
            else:
                # Extract all observations in group at once: (n_cases, n_tstep, n_obs_in_group)
                pred_group = yobs_pred_np[np.ix_(case_indices, np.arange(yobs_pred_np.shape[1]), obs_group_indices)]
                true_group = yobs_true_np[np.ix_(case_indices, obs_group_indices, np.arange(yobs_true_np.shape[2]))]
            
            # Process each observation in the group separately for color-coding
            obs_data_dict = {}
            all_obs_true = []
            all_obs_pred = []
            
            for i, obs_idx in enumerate(obs_indices_to_process):
                pred_obs = pred_group[:, :, i]  # (n_cases, n_tstep)
                true_obs = true_group[:, i, :]  # (n_cases, n_tstep)
                
                # Flatten for plotting
                pred_flat = pred_obs.flatten()
                true_flat = true_obs.flatten()
                
                # Batch denormalize using helper method (vectorized)
                if self.norm_params:
                    pred_flat = self._batch_denormalize_timeseries(pred_flat, obs_idx, self.norm_params)
                    true_flat = self._batch_denormalize_timeseries(true_flat, obs_idx, self.norm_params)
                
                # Ensure non-negative observations
                pred_flat = np.maximum(pred_flat, 0.0)
                
                # Remove NaN values (vectorized)
                valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
                obs_true = true_flat[valid_idx]
                obs_pred = pred_flat[valid_idx]
                
                # Store observation data for color-coding (use obs_idx as key)
                obs_data_dict[obs_idx] = (obs_true, obs_pred)
                
                # Collect for overall metrics calculation
                all_obs_true.append(obs_true)
                all_obs_pred.append(obs_pred)
            
            # Aggregate all observations for overall metrics calculation
            if len(all_obs_true) > 0:
                all_true_combined = np.concatenate(all_obs_true)
                all_pred_combined = np.concatenate(all_obs_pred)
                
                # Calculate metrics from all aggregated data points (same as single plots)
                if len(all_true_combined) > 0:
                    metrics = self.metrics_evaluator._compute_metrics(all_true_combined, all_pred_combined, selected_metrics=selected_metrics)
                    if 'ape' not in metrics and 'ape' not in selected_metrics:
                        metrics['ape'] = self.metrics_evaluator._compute_metrics(all_true_combined, all_pred_combined, selected_metrics=['ape'])['ape']
                    final_group_metrics = {k: v for k, v in metrics.items() if k in selected_metrics or k == 'ape'}
                else:
                    final_group_metrics = {k: 0.0 for k in selected_metrics}
                    final_group_metrics['ape'] = 0.0
            else:
                final_group_metrics = {k: 0.0 for k in selected_metrics}
                final_group_metrics['ape'] = 0.0
            
            # Return observation-grouped data for color-coding
            # Structure: {'observations': obs_data_dict, 'all_true': all_true_combined, 'all_pred': all_pred_combined}
            overall_metrics.append(final_group_metrics)
            if len(all_obs_true) > 0:
                all_data.append({'observations': obs_data_dict, 'all_true': all_true_combined, 'all_pred': all_pred_combined})
            else:
                all_data.append({'observations': {}, 'all_true': np.array([]), 'all_pred': np.array([])})
        else:
            # Process observations individually (original approach for single obs or all obs)
            for obs_idx in obs_indices_to_process:
                # Extract all data for this observation: (cases, timesteps)
                # yobs_pred: (num_case, num_tstep, n_obs)
                # yobs_true: (num_case, n_obs, num_tstep)
                
                # Use basic slicing when all cases from selected group are selected (much faster)
                if use_all_cases and selected_indices is None:
                    # Use basic slicing for all cases - only when using separate arrays
                    pred_obs = yobs_pred_np[:, :, obs_idx]  # (n_cases, n_tstep)
                    true_obs = yobs_true_np[:, obs_idx, :]  # (n_cases, n_tstep)
                else:
                    # Use advanced indexing for subset of cases (or all cases from selected group in combined array)
                    pred_obs = yobs_pred_np[case_indices, :, obs_idx]  # (n_cases, n_tstep)
                    true_obs = yobs_true_np[case_indices, obs_idx, :]  # (n_cases, n_tstep)
                
                # Flatten for plotting (keep aggregation for scatter plot)
                pred_flat = pred_obs.flatten()
                true_flat = true_obs.flatten()
                
                # Batch denormalize using helper method (vectorized)
                if self.norm_params:
                    pred_flat = self._batch_denormalize_timeseries(pred_flat, obs_idx, self.norm_params)
                    true_flat = self._batch_denormalize_timeseries(true_flat, obs_idx, self.norm_params)
                
                # Ensure non-negative observations
                pred_flat = np.maximum(pred_flat, 0.0)
                
                # Remove NaN values (vectorized)
                valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
                all_true = true_flat[valid_idx]
                all_pred = pred_flat[valid_idx]
                
                # Calculate metrics from all aggregated data points (same as single plots)
                if len(all_true) > 0:
                    metrics = self.metrics_evaluator._compute_metrics(all_true, all_pred, selected_metrics=selected_metrics)
                    # Ensure ape is included for compatibility
                    if 'ape' not in metrics and 'ape' not in selected_metrics:
                        metrics['ape'] = self.metrics_evaluator._compute_metrics(all_true, all_pred, selected_metrics=['ape'])['ape']
                    filtered_metrics = {k: v for k, v in metrics.items() if k in selected_metrics or k == 'ape'}
                else:
                    filtered_metrics = {k: 0.0 for k in selected_metrics}
                    filtered_metrics['ape'] = 0.0
                
                overall_metrics.append(filtered_metrics)
                all_data.append((all_true, all_pred))
        
        # Return single item if obs_group_indices specified, otherwise return list
        # Note: When obs_group_indices has multiple observations, all_data contains dict with 'observations' key
        # When single observation or all observations individually, all_data contains (all_true, all_pred) tuples
        if obs_group_indices is not None and len(obs_group_indices) > 1:
            # Return the grouped data structure (already set above)
            result = (overall_metrics[0] if overall_metrics else None, all_data[0] if all_data else None)
        elif obs_group_indices is not None and len(obs_group_indices) == 1:
            # Single observation in group - return normally
            result = (overall_metrics[0] if overall_metrics else None, all_data[0] if all_data else None)
        else:
            result = (overall_metrics, all_data)
        
        if use_cache:
            cache_key = self._generate_cache_key(case_indices, selected_metrics, None, obs_group_indices, use_training_data, 'timeseries', layer_idx=None)
            self._overall_metrics_cache[cache_key] = result
        
        return result
    
    def _calculate_overall_timeseries_metrics(self):
        """Legacy method - calls optimized version"""
        return self._calculate_overall_timeseries_metrics_optimized()
    
    def _average_metrics_group(self, timeseries_metrics, obs_indices):
        """Average metrics across a group of observations"""
        if not obs_indices:
            return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'ape': 0.0}
        
        avg_metrics = {}
        for metric_key in ['r2', 'rmse', 'mae', 'ape']:
            values = [timeseries_metrics[idx][metric_key] for idx in obs_indices if idx < len(timeseries_metrics)]
            avg_metrics[metric_key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def _format_metric_value(self, value):
        """
        Format metric value with scientific notation if it has more than 2 digits before decimal point.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted string (e.g., "0.93" or "1.73e+08")
        """
        abs_value = abs(value)
        if abs_value == 0.0:
            return "0.00"
        elif abs_value >= 100.0:
            # Use scientific notation for values >= 100
            return f"{value:.2e}"
        else:
            # Use regular decimal format for values < 100
            return f"{value:.2f}"
    
    def _plot_overall_spatial_metric(self, ax, field_idx, field_name, metrics, all_true, all_pred, selected_metrics=None, 
                                     layer_data_dict=None, fig=None):
        """
        Plot overall spatial metric as scatter plot with the same style as individual metrics.
        
        Args:
            ax: Matplotlib axis
            field_idx: Field index
            field_name: Field name
            metrics: Metrics dictionary
            all_true: True values array (or dict with 'layers' key when layer-grouped)
            all_pred: Predicted values array (or dict with 'all_true'/'all_pred' when layer-grouped)
            selected_metrics: List of selected metrics
            layer_data_dict: Dictionary mapping layer_idx to (all_true, all_pred) for color-coding (optional)
            fig: Figure reference for legend placement outside plot (optional)
        """
        if selected_metrics is None:
            selected_metrics = ['r2', 'mse', 'rmse', 'mae']
        
        # Check if we have layer-grouped data
        if layer_data_dict is not None and len(layer_data_dict) > 0:
            # Plot each layer with different color
            colors = plt.cm.tab10(np.linspace(0, 1, len(layer_data_dict)))
            
            legend_elements = []
            for i, (layer_idx, (layer_true, layer_pred)) in enumerate(sorted(layer_data_dict.items())):
                color = colors[i]
                ax.scatter(layer_true, layer_pred, alpha=0.5, s=5, color=color, label=f'Layer {layer_idx}')
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=color, markersize=8, 
                                                 label=f'Layer {layer_idx}'))
            
            # Add legend outside plot if figure is provided
            if fig is not None:
                fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                          fontsize=10, frameon=True, fancybox=True, shadow=True)
            else:
                # Fallback to axis legend
                ax.legend(loc='upper left', fontsize=10)
            
            # Use combined data for metrics calculation (already calculated and passed in metrics)
            # Extract combined data from all_true/all_pred if they're dicts
            if isinstance(all_true, dict):
                all_true_combined = all_true.get('all_true', np.array([]))
                all_pred_combined = all_pred.get('all_pred', np.array([]))
            else:
                all_true_combined = all_true
                all_pred_combined = all_pred
        else:
            # Single layer or no layer grouping - plot normally
            ax.scatter(all_true, all_pred, alpha=0.5, s=5, color='blue')
            all_true_combined = all_true
            all_pred_combined = all_pred
        
        # Add reference lines (parallel to perfect prediction line)
        if len(all_true_combined) > 0:
            # Get min and max for line plotting
            min_val = min(np.min(all_true_combined), np.min(all_pred_combined))
            max_val = max(np.max(all_true_combined), np.max(all_pred_combined))
            
            # Plot y=x line (perfect prediction) - bold and visible
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=1.0, linewidth=3, label='Perfect Prediction')
            
            # Add ±10% reference lines (parallel to perfect prediction line) - bold and visible
            x_line = np.linspace(min_val, max_val, 100)
            data_range = max_val - min_val
            offset_10_percent = 0.10 * data_range  # 10% of the data range as constant offset
            y_plus_10 = x_line + offset_10_percent   # +10% line (parallel)
            y_minus_10 = x_line - offset_10_percent  # -10% line (parallel)
            ax.plot(x_line, y_plus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='+10% Variance')
            ax.plot(x_line, y_minus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='-10% Variance')
            
            # Create legend for reference lines (only if no layer legend already added)
            if layer_data_dict is None or len(layer_data_dict) == 0:
                legend = ax.legend(loc='upper left')
                for text in legend.get_texts():
                    text.set_fontweight('bold')
        
        # Determine field unit based on field name
        field_unit = self.metrics_evaluator._get_field_unit(field_name)
        
        # Set axis labels with bold font and units
        ax.set_xlabel(f'True Values ({field_unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted Values ({field_unit})', fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid with bold appearance
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
        
        # Make tick labels bold (matching error plots format)
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Apply scientific notation formatting (matching timeseries error plots format)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        
        # Use metrics passed in (already calculated based on selected mode: aggregated or averaged)
        # Only recalculate if metrics are not provided or incomplete
        if metrics is not None and len(metrics) > 0:
            # Use passed metrics (these are already calculated based on aggregated or averaged mode)
            plot_metrics = metrics
        else:
            # Fallback: Calculate metrics directly from plotted points if not provided
            if len(all_true_combined) > 0 and len(all_pred_combined) > 0:
                # Remove NaN values if any
                valid_idx = ~np.isnan(all_true_combined) & ~np.isnan(all_pred_combined)
                all_true_valid = all_true_combined[valid_idx]
                all_pred_valid = all_pred_combined[valid_idx]
                
                if len(all_true_valid) > 0:
                    # Calculate metrics from plotted data points
                    plot_metrics = self.metrics_evaluator._compute_metrics(
                        all_true_valid, all_pred_valid, 
                        selected_metrics=selected_metrics,
                        filter_negative_predictions=(field_unit == 'fraction')
                    )
                else:
                    plot_metrics = {k: 0.0 for k in selected_metrics}
            else:
                plot_metrics = {k: 0.0 for k in selected_metrics}
        
        title = f"{field_name}\n"
        metric_parts = []
        if 'r2' in selected_metrics and 'r2' in plot_metrics:
            metric_parts.append(f"R² = {self._format_metric_value(plot_metrics['r2'])}")
        if 'mse' in selected_metrics and 'mse' in plot_metrics:
            metric_parts.append(f"MSE = {self._format_metric_value(plot_metrics['mse'])}")
        if 'rmse' in selected_metrics and 'rmse' in plot_metrics:
            metric_parts.append(f"RMSE = {self._format_metric_value(plot_metrics['rmse'])}")
        if 'mae' in selected_metrics and 'mae' in plot_metrics:
            metric_parts.append(f"MAE = {self._format_metric_value(plot_metrics['mae'])}")
        
        if metric_parts:
            title += ", ".join(metric_parts)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    def _plot_overall_timeseries_metric(self, ax, group_name, unit, metrics, all_true, all_pred, selected_metrics=None,
                                        obs_data_dict=None, obs_names_map=None, fig=None):
        """
        Plot overall timeseries metric as scatter plot with the same style as individual metrics.
        
        Args:
            ax: Matplotlib axis
            group_name: Group name
            unit: Unit string
            metrics: Metrics dictionary
            all_true: True values array (or dict with 'observations' key when observation-grouped)
            all_pred: Predicted values array (or dict with 'all_true'/'all_pred' when observation-grouped)
            selected_metrics: List of selected metrics
            obs_data_dict: Dictionary mapping obs_idx to (all_true, all_pred) for color-coding (optional)
            obs_names_map: Dictionary mapping obs_idx to observation name for legend (optional)
            fig: Figure reference for legend placement outside plot (optional)
        """
        if selected_metrics is None:
            selected_metrics = ['r2', 'mse', 'rmse', 'mae']
        
        # Check if we have observation-grouped data
        if obs_data_dict is not None and len(obs_data_dict) > 0:
            # Plot each observation with different color
            colors = plt.cm.Set3(np.linspace(0, 1, len(obs_data_dict)))
            
            legend_elements = []
            for i, (obs_idx, (obs_true, obs_pred)) in enumerate(sorted(obs_data_dict.items())):
                color = colors[i]
                obs_name = obs_names_map.get(obs_idx, f'Observation {obs_idx}') if obs_names_map else f'Observation {obs_idx}'
                ax.scatter(obs_true, obs_pred, alpha=0.7, s=30, color=color, label=obs_name)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=color, markersize=10, 
                                                 label=obs_name))
            
            # Add legend outside plot if figure is provided
            if fig is not None:
                fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                          fontsize=10, frameon=True, fancybox=True, shadow=True)
            else:
                # Fallback to axis legend
                ax.legend(loc='upper left', fontsize=10)
            
            # Use combined data for metrics calculation (already calculated and passed in metrics)
            # Extract combined data from all_true/all_pred if they're dicts
            if isinstance(all_true, dict):
                all_true_combined = all_true.get('all_true', np.array([]))
                all_pred_combined = all_pred.get('all_pred', np.array([]))
            else:
                all_true_combined = all_true
                all_pred_combined = all_pred
        else:
            # Single observation or no observation grouping - plot normally
            ax.scatter(all_true, all_pred, alpha=0.7, s=30, color='blue')
            all_true_combined = all_true
            all_pred_combined = all_pred
        
        # Add reference lines (parallel to perfect prediction line)
        if len(all_true_combined) > 0:
            # Get min and max for line plotting
            min_val = min(np.min(all_true_combined), np.min(all_pred_combined))
            max_val = max(np.max(all_true_combined), np.max(all_pred_combined))
            
            # Plot y=x line (perfect prediction) - bold and visible
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=1.0, linewidth=3, label='Perfect Prediction')
            
            # Add ±10% reference lines (parallel to perfect prediction line) - bold and visible
            x_line = np.linspace(min_val, max_val, 100)
            data_range = max_val - min_val
            offset_10_percent = 0.10 * data_range  # 10% of the data range as constant offset
            y_plus_10 = x_line + offset_10_percent   # +10% line (parallel)
            y_minus_10 = x_line - offset_10_percent  # -10% line (parallel)
            ax.plot(x_line, y_plus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='+10% Variance')
            ax.plot(x_line, y_minus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='-10% Variance')
            
            # Create legend for reference lines (only if no observation legend already added)
            if obs_data_dict is None or len(obs_data_dict) == 0:
                legend = ax.legend(loc='upper left')
                for text in legend.get_texts():
                    text.set_fontweight('bold')
        
        # Set axis labels with bold font and units
        ax.set_xlabel(f'True Values ({unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted Values ({unit})', fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid with bold appearance
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
        
        # Make tick labels bold (matching error plots format)
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Apply scientific notation formatting (matching timeseries error plots format)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        
        # Use metrics passed in (already calculated based on selected mode: aggregated or averaged)
        # Only recalculate if metrics are not provided or incomplete
        if metrics is not None and len(metrics) > 0:
            # Use passed metrics (these are already calculated based on aggregated or averaged mode)
            plot_metrics = metrics
        else:
            # Fallback: Calculate metrics directly from plotted points if not provided
            if len(all_true_combined) > 0 and len(all_pred_combined) > 0:
                # Remove NaN values if any
                valid_idx = ~np.isnan(all_true_combined) & ~np.isnan(all_pred_combined)
                all_true_valid = all_true_combined[valid_idx]
                all_pred_valid = all_pred_combined[valid_idx]
                
                if len(all_true_valid) > 0:
                    # Calculate metrics from plotted data points
                    plot_metrics = self.metrics_evaluator._compute_metrics(
                        all_true_valid, all_pred_valid, 
                        selected_metrics=selected_metrics
                    )
                else:
                    plot_metrics = {k: 0.0 for k in selected_metrics}
            else:
                plot_metrics = {k: 0.0 for k in selected_metrics}
        
        title = f"{group_name}\n"
        metric_parts = []
        if 'r2' in selected_metrics and 'r2' in plot_metrics:
            metric_parts.append(f"R² = {self._format_metric_value(plot_metrics['r2'])}")
        if 'mse' in selected_metrics and 'mse' in plot_metrics:
            metric_parts.append(f"MSE = {self._format_metric_value(plot_metrics['mse'])}")
        if 'rmse' in selected_metrics and 'rmse' in plot_metrics:
            metric_parts.append(f"RMSE = {self._format_metric_value(plot_metrics['rmse'])}")
        if 'mae' in selected_metrics and 'mae' in plot_metrics:
            metric_parts.append(f"MAE = {self._format_metric_value(plot_metrics['mae'])}")
        
        if metric_parts:
            title += ", ".join(metric_parts)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    def _save_overall_metrics_to_json(self, button):
        """Save overall metrics to JSON file"""
        with self.overall_metrics_output:
            clear_output(wait=True)
            
            if self.latest_spatial_metrics is None and self.latest_timeseries_metrics is None:
                print("❌ No metrics available to save. Please calculate metrics first!")
                return
            
            try:
                # Get selected metrics (already in lowercase format)
                selected_metric_keys = self.latest_selected_metrics if self.latest_selected_metrics else []
                
                # Get selected graph information
                graph_type = self.graph_type_dropdown.value if hasattr(self, 'graph_type_dropdown') else 'Unknown'
                selected_graph = None
                if graph_type == 'Spatial' and hasattr(self, 'spatial_graph_dropdown'):
                    selected_graph = self.spatial_graph_dropdown.value
                elif graph_type == 'Timeseries' and hasattr(self, 'timeseries_graph_dropdown'):
                    selected_graph = self.timeseries_graph_dropdown.value
                
                # Create metrics dictionary
                metrics_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'selected_metrics': selected_metric_keys,
                    'case_types': self.latest_case_types if self.latest_case_types else [],
                    'graph_type': graph_type,
                    'selected_graph': selected_graph,
                    'spatial_metrics': {},
                    'timeseries_metrics': {}
                }
                
                # Add spatial metrics (if available)
                if self.latest_spatial_metrics is not None:
                    # Check if it's a list (old format) or single dict (new format)
                    if isinstance(self.latest_spatial_metrics, list) and len(self.latest_spatial_metrics) > 0:
                        # New format: list with selected field metrics
                        metrics_dict = self.latest_spatial_metrics[0]
                        if metrics_dict is not None:
                            field_metrics = {}
                            for metric_key in selected_metric_keys:
                                if metric_key in metrics_dict:
                                    field_metrics[metric_key] = float(metrics_dict[metric_key])
                            if field_metrics and selected_graph:
                                metrics_data['spatial_metrics'][selected_graph] = field_metrics
                
                # Add timeseries metrics (if available)
                if self.latest_timeseries_metrics is not None:
                    # Check if it's a list (old format) or single dict (new format)
                    if isinstance(self.latest_timeseries_metrics, list) and len(self.latest_timeseries_metrics) > 0:
                        # New format: list with selected group metrics
                        metrics_dict = self.latest_timeseries_metrics[0]
                        if metrics_dict is not None:
                            group_metrics = {}
                            for metric_key in selected_metric_keys:
                                if metric_key in metrics_dict:
                                    group_metrics[metric_key] = float(metrics_dict[metric_key])
                            if group_metrics and selected_graph:
                                metrics_data['timeseries_metrics'][selected_graph] = group_metrics
                
                # Determine save directory - use absolute path relative to ROM_Refactored
                save_dir = str(_ROM_DIR / 'timing_logs')
                # Ensure directory exists
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # Create filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'overall_metrics_{timestamp}.json'
                filepath = os.path.join(save_dir, filename)
                
                # Save to JSON
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(metrics_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Metrics saved successfully!")
                print(f"   File: {filepath}")
                print(f"   Spatial fields: {len(metrics_data['spatial_metrics'])}")
                print(f"   Timeseries groups: {len(metrics_data['timeseries_metrics'])}")
                
            except Exception as e:
                print(f"❌ Error saving metrics: {e}")
                import traceback
                traceback.print_exc()
    
    # ================================================================
    # MODEL COMPARISON TAB (Tab 5)
    # ================================================================

    def _create_model_comparison_tab(self):
        """Create the model comparison tab with 7 plot types and CSV export."""
        plot_options = [
            ('1. Spatial Error vs. Timestep (Boxplot)', 1),
            ('2. Observation Error vs. Timestep (Boxplot)', 2),
            ('3. Latent Norm vs. Timestep', 3),
            ('4. R-squared vs. Timestep', 4),
            ('5. Error Growth Rate', 5),
            ('6. Per-Case Error Heatmap', 6),
            ('7. Observation Pred vs. Truth (Boxplot)', 7),
        ]
        self.comp_plot_dropdown = widgets.Dropdown(
            options=plot_options, value=1,
            description='Plot Type:', style={'description_width': 'initial'},
            layout=widgets.Layout(width='450px')
        )

        self.comp_field_dropdown = widgets.Dropdown(
            options=[(name, idx) for idx, name in enumerate(self.field_names)],
            value=0, description='Spatial Field:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='350px')
        )

        obs_options = [(f"{name} ({unit})", idx) for idx, (name, unit) in
                       enumerate(zip(self.obs_names, self.obs_units))]
        self.comp_obs_dropdown = widgets.Dropdown(
            options=obs_options, value=0, description='Observation:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='400px')
        )

        case_options = [('All Cases', -1)]
        if hasattr(self, 'test_case_indices') and self.test_case_indices is not None:
            for i, ci in enumerate(self.test_case_indices):
                case_options.append((f'Case {int(ci)}', i))
        self.comp_case_dropdown = widgets.Dropdown(
            options=case_options, value=-1, description='Case:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px')
        )

        self.comp_generate_btn = widgets.Button(
            description='Generate Latent Predictions',
            button_style='warning', layout=widgets.Layout(width='250px')
        )
        self.comp_refresh_btn = widgets.Button(
            description='Refresh Plot', button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.comp_csv_btn = widgets.Button(
            description='Export CSV', button_style='success',
            layout=widgets.Layout(width='150px')
        )
        default_csv_dir = os.path.join(str(_ROM_DIR), 'comparison_csv')
        self.comp_csv_dir = widgets.Text(
            value=default_csv_dir, description='CSV Dir:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='550px')
        )
        self.comp_status = widgets.Label(value='Select a plot type and click Refresh Plot.')
        self.comp_output = widgets.Output()

        self._comp_last_data = None

        self.comp_field_box = widgets.HBox([
            widgets.HTML("<b>Select Spatial Field:</b>&nbsp;&nbsp;"),
            self.comp_field_dropdown,
        ])
        self.comp_obs_box = widgets.HBox([
            widgets.HTML("<b>Select Observation:</b>&nbsp;&nbsp;"),
            self.comp_obs_dropdown,
        ])
        self.comp_case_box = widgets.HBox([
            widgets.HTML("<b>Select Case:</b>&nbsp;&nbsp;"),
            self.comp_case_dropdown,
        ])

        self.comp_refresh_btn.on_click(lambda _: self._update_comparison_plot())
        self.comp_csv_btn.on_click(lambda _: self._export_comparison_csv())
        self.comp_generate_btn.on_click(lambda _: self._comp_generate_latent())
        self.comp_plot_dropdown.observe(self._comp_toggle_dropdowns, names='value')

        self.comp_plot_dropdown.observe(lambda _: self._update_comparison_plot(), names='value')
        self.comp_field_dropdown.observe(lambda _: self._update_comparison_plot(), names='value')
        self.comp_obs_dropdown.observe(lambda _: self._update_comparison_plot(), names='value')
        self.comp_case_dropdown.observe(lambda _: self._update_comparison_plot(), names='value')

        self._comp_toggle_dropdowns({'new': 1})

        info_html = widgets.HTML(
            "<p style='color:#555; font-size:12px; margin:2px 0;'>"
            "Boxplots show the <b>full distribution</b> (median, quartiles, whiskers) "
            "across all test cases at each timestep &mdash; not averages.</p>"
        )

        controls = widgets.VBox([
            widgets.HTML("<h3>Model Comparison &mdash; Systematic Evaluation</h3>"),
            info_html,
            widgets.HBox([self.comp_plot_dropdown]),
            self.comp_field_box,
            self.comp_obs_box,
            self.comp_case_box,
            widgets.HBox([self.comp_generate_btn, self.comp_refresh_btn, self.comp_csv_btn]),
            self.comp_csv_dir,
            self.comp_status,
        ])
        return widgets.VBox([controls, self.comp_output])

    def _comp_toggle_dropdowns(self, change):
        """Show/hide field vs observation vs case selection rows based on selected plot type."""
        pt = change.get('new', 1)
        show_field = pt in (1, 4, 5, 6)
        show_obs = pt in (2, 7)
        show_case = pt == 6
        self.comp_field_box.layout.display = 'flex' if show_field else 'none'
        self.comp_obs_box.layout.display = 'flex' if show_obs else 'none'
        self.comp_case_box.layout.display = 'flex' if show_case else 'none'

    def _comp_generate_latent(self):
        """Generate both latent observation and spatial predictions if not already done."""
        with self.comp_output:
            clear_output(wait=True)
            if self.my_rom is None:
                print("ROM model not available. Cannot generate latent predictions.")
                return
            if not self.predictions_generated:
                print("Generating latent observation predictions...")
                self.yobs_pred_latent_based = self._generate_latent_predictions()
                self.yobs_pred_state_based = self.yobs_pred.clone() if hasattr(self.yobs_pred, 'clone') else self.yobs_pred.copy()
                self.predictions_generated = True
            if not self.spatial_latent_generated:
                print("Generating latent spatial predictions...")
                self._generate_latent_spatial_predictions()
            self.comp_status.value = 'Latent predictions generated. Click Refresh Plot.'

    # ---- numpy helpers ------------------------------------------------

    def _to_np(self, t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _get_state_np(self, mode='state'):
        """Return (pred, true) spatial arrays, both as (num_case, num_tstep, C, Nx, Ny, Nz)."""
        if mode == 'latent':
            pred = self._to_np(self.state_pred_latent_based)
        else:
            pred = self._to_np(self.state_pred)
        true_raw = self._to_np(self.state_seq_true_aligned)
        if true_raw is not None and true_raw.ndim == 6:
            true = np.transpose(true_raw, (0, 2, 1, 3, 4, 5))
        else:
            true = true_raw
        return pred, true

    def _get_obs_np(self, mode='state'):
        """Return (pred, true) obs arrays, both as (num_case, num_tstep, n_obs)."""
        if mode == 'latent':
            pred = self._to_np(self.yobs_pred_latent_based)
        else:
            pred = self._to_np(self.yobs_pred)
        true_raw = self._to_np(self.yobs_seq_true)
        if true_raw is not None and true_raw.ndim == 3:
            true = np.transpose(true_raw, (0, 2, 1))
        else:
            true = true_raw
        return pred, true

    def _clamp_negative(self, arr):
        """Clamp negative values to zero for physical realism (saturations, pressures, rates)."""
        return np.clip(arr, 0.0, None)

    # ---- main dispatcher -----------------------------------------------

    def _update_comparison_plot(self):
        pt = self.comp_plot_dropdown.value
        with self.comp_output:
            clear_output(wait=True)
            try:
                if pt == 1:
                    self._plot_spatial_error_boxplot()
                elif pt == 2:
                    self._plot_obs_error_boxplot()
                elif pt == 3:
                    self._plot_latent_norm()
                elif pt == 4:
                    self._plot_r2_vs_timestep()
                elif pt == 5:
                    self._plot_error_growth_rate()
                elif pt == 6:
                    self._plot_error_heatmap()
                elif pt == 7:
                    self._plot_obs_values_boxplot()
            except Exception as e:
                print(f"Error generating plot: {e}")
                import traceback; traceback.print_exc()

    # ---- Plot 1: Spatial Error vs. Timestep (Boxplot) ------------------

    def _plot_spatial_error_boxplot(self):
        fi = self.comp_field_dropdown.value
        fname = self.field_names[fi]
        has_latent = self.state_pred_latent_based is not None and self.spatial_latent_generated

        def _rmse_per_case_step(pred, true, fi):
            p = np.clip(pred[:, :, fi, :, :, :], 0.0, None)
            t = true[:, :, fi, :, :, :]
            diff2 = (p - t) ** 2
            return np.sqrt(diff2.reshape(diff2.shape[0], diff2.shape[1], -1).mean(axis=2))

        pred_s, true_s = self._get_state_np('state')
        rmse_s = _rmse_per_case_step(pred_s, true_s, fi)

        rmse_l = None
        if has_latent:
            pred_l, true_l = self._get_state_np('latent')
            rmse_l = _rmse_per_case_step(pred_l, true_l, fi)

        ncols = 2 if has_latent else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
        timesteps = np.arange(1, rmse_s.shape[1] + 1)

        bp = axes[0, 0].boxplot([rmse_s[:, t] for t in range(rmse_s.shape[1])],
                                positions=timesteps, widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.7)
        mean_s = np.mean(rmse_s, axis=0)
        axes[0, 0].plot(timesteps, mean_s, '-o', color='red', markersize=4, linewidth=1.8,
                        zorder=5, label='Mean')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title(f'State-Based  |  {fname}')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)

        if has_latent and rmse_l is not None:
            bp2 = axes[0, 1].boxplot([rmse_l[:, t] for t in range(rmse_l.shape[1])],
                                     positions=timesteps, widths=0.6, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('#DD8452')
                patch.set_alpha(0.7)
            mean_l = np.mean(rmse_l, axis=0)
            axes[0, 1].plot(timesteps, mean_l, '-o', color='red', markersize=4, linewidth=1.8,
                            zorder=5, label='Mean')
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].set_title(f'Latent-Based  |  {fname}')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].grid(True, alpha=0.3)
            y_max = max(rmse_s.max(), rmse_l.max()) * 1.1
            axes[0, 0].set_ylim(bottom=0, top=y_max)
            axes[0, 1].set_ylim(bottom=0, top=y_max)

        fig.suptitle(f'Spatial RMSE vs. Timestep  —  {fname}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(rmse_s.shape[0]):
            for t in range(rmse_s.shape[1]):
                row = {'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                       'rmse_state_based': float(rmse_s[c, t])}
                if rmse_l is not None:
                    row['rmse_latent_based'] = float(rmse_l[c, t])
                rows.append(row)
        self._comp_last_data = ('spatial_error_boxplot', fname, rows)
        self.comp_status.value = f'Plot 1 rendered. {len(rows)} data rows ready for CSV export.'

    # ---- Plot 2: Observation Error vs. Timestep (Boxplot) --------------

    def _plot_obs_error_boxplot(self):
        oi = self.comp_obs_dropdown.value
        oname = self.obs_names[oi] if oi < len(self.obs_names) else f'Obs {oi}'
        has_latent = self.yobs_pred_latent_based is not None and self.predictions_generated

        pred_s, true_s = self._get_obs_np('state')
        raw_pred_s = pred_s[:, :, oi]
        raw_true_s = true_s[:, :, oi]
        valid_s = raw_pred_s >= 0
        ae_s_full = np.abs(raw_pred_s - raw_true_s)

        ae_l_full = None
        valid_l = None
        if has_latent:
            pred_l, true_l = self._get_obs_np('latent')
            raw_pred_l = pred_l[:, :, oi]
            raw_true_l = true_l[:, :, oi]
            valid_l = raw_pred_l >= 0
            ae_l_full = np.abs(raw_pred_l - raw_true_l)

        T = raw_pred_s.shape[1]
        ncols = 2 if has_latent else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
        timesteps = np.arange(1, T + 1)

        data_s = [ae_s_full[valid_s[:, t], t] for t in range(T)]
        bp = axes[0, 0].boxplot(data_s, positions=timesteps, widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.7)
        mean_s = np.array([d.mean() if len(d) > 0 else 0.0 for d in data_s])
        axes[0, 0].plot(timesteps, mean_s, '-o', color='red', markersize=4, linewidth=1.8,
                        zorder=5, label='Mean')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title(f'State-Based  |  {oname}')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].grid(True, alpha=0.3)

        if has_latent and ae_l_full is not None and valid_l is not None:
            data_l = [ae_l_full[valid_l[:, t], t] for t in range(T)]
            bp2 = axes[0, 1].boxplot(data_l, positions=timesteps, widths=0.6, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('#DD8452')
                patch.set_alpha(0.7)
            mean_l = np.array([d.mean() if len(d) > 0 else 0.0 for d in data_l])
            axes[0, 1].plot(timesteps, mean_l, '-o', color='red', markersize=4, linewidth=1.8,
                            zorder=5, label='Mean')
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].set_title(f'Latent-Based  |  {oname}')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Absolute Error')
            axes[0, 1].grid(True, alpha=0.3)
            y_max_s = max(d.max() for d in data_s if len(d) > 0) if any(len(d) > 0 for d in data_s) else 1.0
            y_max_l = max(d.max() for d in data_l if len(d) > 0) if any(len(d) > 0 for d in data_l) else 1.0
            y_max = max(y_max_s, y_max_l) * 1.1
            axes[0, 0].set_ylim(bottom=0, top=y_max)
            axes[0, 1].set_ylim(bottom=0, top=y_max)
        else:
            axes[0, 0].set_ylim(bottom=0)

        fig.suptitle(f'Observation Absolute Error vs. Timestep  —  {oname}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(raw_pred_s.shape[0]):
            for t in range(T):
                if not valid_s[c, t]:
                    continue
                row = {'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                       'abs_error_state_based': float(ae_s_full[c, t]),
                       'ground_truth': float(raw_true_s[c, t])}
                if ae_l_full is not None and valid_l is not None and valid_l[c, t]:
                    row['abs_error_latent_based'] = float(ae_l_full[c, t])
                rows.append(row)
        self._comp_last_data = ('obs_error_boxplot', oname, rows)
        n_excl_s = int((~valid_s).sum())
        self.comp_status.value = (f'Plot 2 rendered. {len(rows)} data rows ready for CSV export. '
                                  f'{n_excl_s} negative-prediction points excluded (state-based).')

    # ---- Plot 3: Latent Norm vs. Timestep ------------------------------

    def _compute_latent_norms(self):
        """Run rollouts and collect ||z_t|| for state-based (re-encode) and latent-based (transition) modes."""
        if self.my_rom is None:
            return None, None, None

        model = self.my_rom.model
        model.eval()
        nc = self.num_case
        T = self.num_tstep

        t_steps = np.arange(0, 200, 200 // T)
        dt_val = 10
        t_steps1 = (t_steps + dt_val).astype(int)
        indt_del = t_steps1 - t_steps
        indt_del = indt_del / max(indt_del)

        norms_state = np.zeros((nc, T))
        norms_latent = np.zeros((nc, T))
        norms_truth = np.zeros((nc, T))

        has_encode_initial = hasattr(model, 'encode_initial')
        is_gnn = hasattr(model, 'graph_manager')
        is_mm = hasattr(model, 'static_encoder') and not is_gnn

        with torch.no_grad():
            initial_spatial = self.state_pred[:, 0, :, :, :, :].to(self.device)

            if has_encode_initial:
                z_latent = model.encode_initial(initial_spatial)
            else:
                enc_out = model.encoder(initial_spatial)
                z_latent = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out

            state_t = initial_spatial.clone()

            for t_idx in range(T):
                dt_seq = torch.tensor(
                    np.ones((nc, 1)) * indt_del[t_idx], dtype=torch.float32
                ).to(self.device)
                controls = self.test_controls[:, :, t_idx].to(self.device)

                z_latent, _ = model.predict_latent(z_latent, dt_seq, controls)
                norms_latent[:, t_idx] = z_latent.cpu().numpy().__pow__(2).sum(axis=-1).__pow__(0.5)

                obs_ph = torch.zeros(nc, self.yobs_pred.shape[-1]).to(self.device)
                state_t, _ = self.my_rom.predict((state_t, controls, obs_ph, dt_seq))
                if has_encode_initial:
                    z_state = model.encode_initial(state_t)
                else:
                    enc_out = model.encoder(state_t)
                    z_state = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
                norms_state[:, t_idx] = z_state.cpu().numpy().__pow__(2).sum(axis=-1).__pow__(0.5)

                true_np = self._to_np(self.state_seq_true_aligned)
                if true_np.ndim == 6:
                    true_t = torch.tensor(true_np[:, :, t_idx, :, :, :], dtype=torch.float32).to(self.device)
                else:
                    true_t = torch.tensor(true_np[:, t_idx, :, :, :, :], dtype=torch.float32).to(self.device)

                if has_encode_initial:
                    z_true = model.encode_initial(true_t)
                else:
                    enc_out = model.encoder(true_t)
                    z_true = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
                norms_truth[:, t_idx] = z_true.cpu().numpy().__pow__(2).sum(axis=-1).__pow__(0.5)

        return norms_state, norms_latent, norms_truth

    def _plot_latent_norm(self):
        self.comp_status.value = 'Computing latent norms (encoding all cases at every timestep)...'
        norms_state, norms_latent, norms_truth = self._compute_latent_norms()
        if norms_state is None:
            print("ROM model not available. Cannot compute latent norms.")
            return

        T = norms_state.shape[1]
        timesteps = np.arange(1, T + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        truth_med = np.median(norms_truth, axis=0)
        truth_q25 = np.percentile(norms_truth, 25, axis=0)
        truth_q75 = np.percentile(norms_truth, 75, axis=0)

        for ax_idx, (norms, title, color) in enumerate([
            (norms_state, 'State-Based (re-encode)', '#4C72B0'),
            (norms_latent, 'Latent-Based (transition)', '#DD8452')
        ]):
            ax = axes[ax_idx]
            ax.fill_between(timesteps, truth_q25, truth_q75, alpha=0.2, color='green', label='Truth IQR')
            ax.plot(timesteps, truth_med, '--', color='green', linewidth=1.5, label='Truth median')
            bp = ax.boxplot([norms[:, t] for t in range(T)],
                           positions=timesteps, widths=0.6, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            mean_norms = np.mean(norms, axis=0)
            ax.plot(timesteps, mean_norms, '-o', color='red', markersize=4, linewidth=1.8,
                    zorder=5, label='Mean')
            ax.set_title(title)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('||z||_2')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        y_all = np.concatenate([norms_state.flatten(), norms_latent.flatten(), norms_truth.flatten()])
        y_max = np.percentile(y_all, 99) * 1.15
        axes[0].set_ylim(bottom=0, top=y_max)
        axes[1].set_ylim(bottom=0, top=y_max)

        fig.suptitle('Latent Norm ||z_t|| vs. Timestep', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(norms_state.shape[0]):
            for t in range(T):
                rows.append({
                    'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                    'norm_state_based': float(norms_state[c, t]),
                    'norm_latent_based': float(norms_latent[c, t]),
                    'norm_ground_truth': float(norms_truth[c, t]),
                })
        self._comp_last_data = ('latent_norm', 'all', rows)
        self.comp_status.value = f'Plot 3 rendered. {len(rows)} data rows ready for CSV export.'

    # ---- Plot 4: R-squared vs. Timestep --------------------------------

    def _plot_r2_vs_timestep(self):
        fi = self.comp_field_dropdown.value
        fname = self.field_names[fi]
        has_latent = self.state_pred_latent_based is not None and self.spatial_latent_generated

        def _r2_per_case_step(pred, true, fi):
            p = np.clip(pred[:, :, fi, :, :, :], 0.0, None).reshape(pred.shape[0], pred.shape[1], -1)
            t = true[:, :, fi, :, :, :].reshape(true.shape[0], true.shape[1], -1)
            ss_res = ((t - p) ** 2).sum(axis=2)
            ss_tot = ((t - t.mean(axis=2, keepdims=True)) ** 2).sum(axis=2)
            r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
            return r2

        pred_s, true_s = self._get_state_np('state')
        r2_s = np.clip(_r2_per_case_step(pred_s, true_s, fi), 0.0, 1.0)

        r2_l = None
        if has_latent:
            pred_l, true_l = self._get_state_np('latent')
            r2_l = np.clip(_r2_per_case_step(pred_l, true_l, fi), 0.0, 1.0)

        T = r2_s.shape[1]
        timesteps = np.arange(1, T + 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        mean_s = np.mean(r2_s, axis=0)
        q25_s = np.percentile(r2_s, 25, axis=0)
        q75_s = np.percentile(r2_s, 75, axis=0)
        ax.plot(timesteps, mean_s, '-o', color='#4C72B0', label='State-Based (mean)', markersize=4)
        ax.fill_between(timesteps, q25_s, q75_s, alpha=0.2, color='#4C72B0')

        if has_latent and r2_l is not None:
            mean_l = np.mean(r2_l, axis=0)
            q25_l = np.percentile(r2_l, 25, axis=0)
            q75_l = np.percentile(r2_l, 75, axis=0)
            ax.plot(timesteps, mean_l, '--s', color='#DD8452', label='Latent-Based (mean)', markersize=4)
            ax.fill_between(timesteps, q25_l, q75_l, alpha=0.2, color='#DD8452')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('R-squared')
        ax.set_title(f'R-squared vs. Timestep  —  {fname}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=1.05)
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(r2_s.shape[0]):
            for t in range(T):
                row = {'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                       'r2_state_based': float(r2_s[c, t])}
                if r2_l is not None:
                    row['r2_latent_based'] = float(r2_l[c, t])
                rows.append(row)
        self._comp_last_data = ('r2_vs_timestep', fname, rows)
        self.comp_status.value = f'Plot 4 rendered. {len(rows)} data rows ready for CSV export.'

    # ---- Plot 5: Error Growth Rate -------------------------------------

    def _plot_error_growth_rate(self):
        fi = self.comp_field_dropdown.value
        fname = self.field_names[fi]
        has_latent = self.state_pred_latent_based is not None and self.spatial_latent_generated

        def _rmse_per_case_step(pred, true, fi):
            p = np.clip(pred[:, :, fi, :, :, :], 0.0, None)
            t = true[:, :, fi, :, :, :]
            diff2 = (p - t) ** 2
            return np.sqrt(diff2.reshape(diff2.shape[0], diff2.shape[1], -1).mean(axis=2))

        pred_s, true_s = self._get_state_np('state')
        rmse_s = _rmse_per_case_step(pred_s, true_s, fi)

        true_field_s = true_s[:, :, fi, :, :, :].reshape(true_s.shape[0], true_s.shape[1], -1)
        data_range_s = (true_field_s.max(axis=(1, 2)) - true_field_s.min(axis=(1, 2))).clip(min=1e-10)
        nrmse_s = rmse_s / data_range_s[:, None]

        nrmse_l = None
        if has_latent:
            pred_l, true_l = self._get_state_np('latent')
            rmse_l = _rmse_per_case_step(pred_l, true_l, fi)
            true_field_l = true_l[:, :, fi, :, :, :].reshape(true_l.shape[0], true_l.shape[1], -1)
            data_range_l = (true_field_l.max(axis=(1, 2)) - true_field_l.min(axis=(1, 2))).clip(min=1e-10)
            nrmse_l = rmse_l / data_range_l[:, None]

        T = nrmse_s.shape[1]
        timesteps = np.arange(1, T + 1)
        fig, ax = plt.subplots(figsize=(10, 5))

        mean_s = np.mean(nrmse_s, axis=0)
        q25_s = np.percentile(nrmse_s, 25, axis=0)
        q75_s = np.percentile(nrmse_s, 75, axis=0)
        ax.plot(timesteps, mean_s, '-o', color='#4C72B0', label='State-Based (mean)', markersize=4)
        ax.fill_between(timesteps, q25_s, q75_s, alpha=0.2, color='#4C72B0')

        if has_latent and nrmse_l is not None:
            mean_l = np.mean(nrmse_l, axis=0)
            q25_l = np.percentile(nrmse_l, 25, axis=0)
            q75_l = np.percentile(nrmse_l, 75, axis=0)
            ax.plot(timesteps, mean_l, '--s', color='#DD8452', label='Latent-Based (mean)', markersize=4)
            ax.fill_between(timesteps, q25_l, q75_l, alpha=0.2, color='#DD8452')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('NRMSE  (RMSE / data range)')
        ax.set_title(f'Normalized Error vs. Timestep  —  {fname}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(nrmse_s.shape[0]):
            for t in range(T):
                row = {'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                       'nrmse_state_based': float(nrmse_s[c, t])}
                if nrmse_l is not None:
                    row['nrmse_latent_based'] = float(nrmse_l[c, t])
                rows.append(row)
        self._comp_last_data = ('error_growth_rate', fname, rows)
        self.comp_status.value = f'Plot 5 rendered. {len(rows)} data rows ready for CSV export.'

    # ---- Plot 6: Per-Case Error Heatmap --------------------------------

    def _plot_error_heatmap(self):
        fi = self.comp_field_dropdown.value
        fname = self.field_names[fi]
        has_latent = self.state_pred_latent_based is not None and self.spatial_latent_generated
        selected_case = self.comp_case_dropdown.value  # -1 = all cases, else index

        def _rmse_per_case_step(pred, true, fi):
            p = np.clip(pred[:, :, fi, :, :, :], 0.0, None)
            t = true[:, :, fi, :, :, :]
            diff2 = (p - t) ** 2
            return np.sqrt(diff2.reshape(diff2.shape[0], diff2.shape[1], -1).mean(axis=2))

        pred_s, true_s = self._get_state_np('state')
        rmse_s = _rmse_per_case_step(pred_s, true_s, fi)

        rmse_l = None
        if has_latent:
            pred_l, true_l = self._get_state_np('latent')
            rmse_l = _rmse_per_case_step(pred_l, true_l, fi)

        if selected_case == -1:
            self._plot_error_heatmap_all(rmse_s, rmse_l, has_latent, fname)
        else:
            self._plot_error_heatmap_single(rmse_s, rmse_l, has_latent, fname, selected_case)

    def _plot_error_heatmap_all(self, rmse_s, rmse_l, has_latent, fname):
        """Full heatmap for all cases."""
        ncols = 2 if has_latent else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, max(4, 0.35 * rmse_s.shape[0])), squeeze=False)

        vmin = 0
        vmax = rmse_s.max() if rmse_l is None else max(rmse_s.max(), rmse_l.max())

        im1 = axes[0, 0].imshow(rmse_s, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='nearest')
        axes[0, 0].set_title(f'State-Based  |  {fname}')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Case Index')
        axes[0, 0].set_yticks(np.arange(rmse_s.shape[0]))
        axes[0, 0].set_yticklabels([str(int(ci)) for ci in self.test_case_indices], fontsize=7)
        fig.colorbar(im1, ax=axes[0, 0], label='RMSE')

        if has_latent and rmse_l is not None:
            im2 = axes[0, 1].imshow(rmse_l, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[0, 1].set_title(f'Latent-Based  |  {fname}')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Case Index')
            axes[0, 1].set_yticks(np.arange(rmse_l.shape[0]))
            axes[0, 1].set_yticklabels([str(int(ci)) for ci in self.test_case_indices], fontsize=7)
            fig.colorbar(im2, ax=axes[0, 1], label='RMSE')

        fig.suptitle(f'Per-Case Error Heatmap  —  {fname}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(rmse_s.shape[0]):
            for t in range(rmse_s.shape[1]):
                row = {'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                       'rmse_state_based': float(rmse_s[c, t])}
                if rmse_l is not None:
                    row['rmse_latent_based'] = float(rmse_l[c, t])
                rows.append(row)
        self._comp_last_data = ('error_heatmap', fname, rows)
        self.comp_status.value = f'Plot 6 rendered (all cases). {len(rows)} data rows ready for CSV export.'

    def _plot_error_heatmap_single(self, rmse_s, rmse_l, has_latent, fname, case_idx):
        """Line chart of RMSE vs. timestep for a single selected case."""
        case_label = int(self.test_case_indices[case_idx])
        T = rmse_s.shape[1]
        timesteps = np.arange(1, T + 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(timesteps, rmse_s[case_idx], '-o', color='#4C72B0', markersize=5,
                linewidth=1.8, label='State-Based')

        if has_latent and rmse_l is not None:
            ax.plot(timesteps, rmse_l[case_idx], '--s', color='#DD8452', markersize=5,
                    linewidth=1.8, label='Latent-Based')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('RMSE')
        ax.set_title(f'RMSE vs. Timestep  —  Case {case_label}  —  {fname}',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

        rows = []
        for t in range(T):
            row = {'case_index': case_label, 'timestep': t + 1,
                   'rmse_state_based': float(rmse_s[case_idx, t])}
            if rmse_l is not None:
                row['rmse_latent_based'] = float(rmse_l[case_idx, t])
            rows.append(row)
        self._comp_last_data = ('error_single_case', f'{fname}_case{case_label}', rows)
        self.comp_status.value = f'Plot 6 rendered (Case {case_label}). {len(rows)} data rows ready for CSV export.'

    # ---- Plot 7: Observation Pred vs. Truth (Overlaid Boxplots) --------

    def _plot_obs_values_boxplot(self):
        oi = self.comp_obs_dropdown.value
        oname = self.obs_names[oi] if oi < len(self.obs_names) else f'Obs {oi}'
        has_latent = self.yobs_pred_latent_based is not None and self.predictions_generated

        pred_s, true_s = self._get_obs_np('state')
        raw_pred_s = pred_s[:, :, oi]
        val_true = true_s[:, :, oi]
        valid_s = raw_pred_s >= 0

        raw_pred_l = None
        valid_l = None
        if has_latent:
            pred_l, _ = self._get_obs_np('latent')
            raw_pred_l = pred_l[:, :, oi]
            valid_l = raw_pred_l >= 0

        T = raw_pred_s.shape[1]
        ncols = 2 if has_latent else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)

        positions_truth = np.arange(1, T + 1) - 0.2
        positions_pred = np.arange(1, T + 1) + 0.2
        width = 0.35

        bp_t = axes[0, 0].boxplot([val_true[:, t] for t in range(T)],
                                   positions=positions_truth, widths=width, patch_artist=True)
        data_pred_s = [raw_pred_s[valid_s[:, t], t] for t in range(T)]
        bp_p = axes[0, 0].boxplot(data_pred_s,
                                   positions=positions_pred, widths=width, patch_artist=True)
        for patch in bp_t['boxes']:
            patch.set_facecolor('#55A868')
            patch.set_alpha(0.7)
        for patch in bp_p['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.7)
        mean_true = np.mean(val_true, axis=0)
        mean_pred_s = np.array([d.mean() if len(d) > 0 else 0.0 for d in data_pred_s])
        axes[0, 0].plot(np.arange(1, T + 1), mean_true, '-o', color='darkgreen', markersize=4,
                        linewidth=1.8, zorder=5, label='Mean (Truth)')
        axes[0, 0].plot(np.arange(1, T + 1), mean_pred_s, '-s', color='red', markersize=4,
                        linewidth=1.8, zorder=5, label='Mean (Pred)')
        axes[0, 0].set_title(f'State-Based  |  {oname}')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_xticks(np.arange(1, T + 1))
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        if has_latent and raw_pred_l is not None and valid_l is not None:
            bp_t2 = axes[0, 1].boxplot([val_true[:, t] for t in range(T)],
                                        positions=positions_truth, widths=width, patch_artist=True)
            data_pred_l = [raw_pred_l[valid_l[:, t], t] for t in range(T)]
            bp_p2 = axes[0, 1].boxplot(data_pred_l,
                                        positions=positions_pred, widths=width, patch_artist=True)
            for patch in bp_t2['boxes']:
                patch.set_facecolor('#55A868')
                patch.set_alpha(0.7)
            for patch in bp_p2['boxes']:
                patch.set_facecolor('#DD8452')
                patch.set_alpha(0.7)
            mean_pred_l = np.array([d.mean() if len(d) > 0 else 0.0 for d in data_pred_l])
            axes[0, 1].plot(np.arange(1, T + 1), mean_true, '-o', color='darkgreen', markersize=4,
                            linewidth=1.8, zorder=5, label='Mean (Truth)')
            axes[0, 1].plot(np.arange(1, T + 1), mean_pred_l, '-s', color='red', markersize=4,
                            linewidth=1.8, zorder=5, label='Mean (Pred)')
            axes[0, 1].set_title(f'Latent-Based  |  {oname}')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].set_xticks(np.arange(1, T + 1))
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)

        fig.suptitle(f'Observation Values vs. Timestep  —  {oname}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        rows = []
        for c in range(raw_pred_s.shape[0]):
            for t in range(T):
                if not valid_s[c, t]:
                    continue
                row = {'case_index': int(self.test_case_indices[c]), 'timestep': t + 1,
                       'ground_truth': float(val_true[c, t]),
                       'predicted_state_based': float(raw_pred_s[c, t])}
                if raw_pred_l is not None and valid_l is not None and valid_l[c, t]:
                    row['predicted_latent_based'] = float(raw_pred_l[c, t])
                rows.append(row)
        self._comp_last_data = ('obs_values_boxplot', oname, rows)
        self.comp_status.value = f'Plot 7 rendered. {len(rows)} data rows ready for CSV export.'

    # ---- CSV Export ----------------------------------------------------

    def _export_comparison_csv(self):
        """Export the last rendered comparison plot data to CSV."""
        if self._comp_last_data is None:
            self.comp_status.value = 'No plot data to export. Render a plot first.'
            return

        plot_name, label, rows = self._comp_last_data
        out_dir = self.comp_csv_dir.value.strip()
        if not out_dir:
            self.comp_status.value = 'Please specify a CSV output directory.'
            return

        try:
            os.makedirs(out_dir, exist_ok=True)
            safe_label = label.replace(' ', '_').replace('/', '_').replace('\\', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{plot_name}_{safe_label}_{timestamp}.csv'
            filepath = os.path.join(out_dir, filename)

            import csv
            if len(rows) == 0:
                self.comp_status.value = 'No data rows to export.'
                return
            fieldnames = list(rows[0].keys())
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.comp_status.value = f'CSV exported: {filepath} ({len(rows)} rows)'
        except Exception as e:
            self.comp_status.value = f'CSV export error: {e}'

    # ================================================================
    # END MODEL COMPARISON TAB
    # ================================================================

    def display_dashboard(self):
        """Display the complete interactive dashboard"""
        if not WIDGETS_AVAILABLE:
            print("❌ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
            return None
            
        print("🎨 Interactive E2C Visualization Dashboard")
        print("=" * 50)
        
        # Masking controls section
        masking_section = widgets.VBox([
            widgets.HTML("<h3>🔧 Inactive Cell Masking</h3>"),
            widgets.HTML("<p><i>Load mask file to hide inactive reservoir cells (white). Supports both global 3D masks and case-specific 4D masks.</i></p>"),
            self.use_masking_checkbox,
            self.mask_file_text,
            widgets.HBox([self.load_mask_button, self.mask_status_label])
        ])
        
        # Save predictions section
        save_section = widgets.VBox([
            widgets.HTML("<h3>💾 Save Predictions</h3>"),
            widgets.HTML("<p><i>Save all denormalized predictions to H5 files matching the structure of sr3_batch_output.</i></p>"),
            self.data_dir_text,
            self.output_dir_text,
            widgets.HBox([self.save_predictions_button, self.save_status_label]),
            self.save_output
        ])
        
        # CSV export section - single case export for visualization
        csv_export_section = widgets.VBox([
            widgets.HTML("<h3>📄 Export Case to CSV</h3>"),
            widgets.HTML("<p><i>Export a single simulation case to CSV files (timeseries + spatial) with denormalized predicted values for visualization.</i></p>"),
            self.csv_case_dropdown,
            self.csv_output_dir_text,
            widgets.HBox([self.save_csv_button, self.csv_status_label]),
            self.csv_output
        ])
        
        # Spatial visualization tab
        spatial_controls = widgets.VBox([
            widgets.HTML("<h4>🎯 Spatial Field Controls</h4>"),
            self.spatial_case_slider,
            self.spatial_layer_slider, 
            self.spatial_timestep_slider,
            self.spatial_field_dropdown,
            widgets.HTML("<br>"),
            self.spatial_latent_checkbox,
            widgets.HTML("<p><i>When enabled, generates latent-based spatial predictions (decode z at each step) for side-by-side comparison.</i></p>")
        ])
        
        spatial_tab = widgets.VBox([
            spatial_controls,
            widgets.HTML("<hr>"),
            self.spatial_output,
            widgets.HTML("<h4>📊 Performance Metrics</h4>"),
            self.spatial_metrics_output
        ])
        
        # Time series visualization tab
        timeseries_controls = widgets.VBox([
            widgets.HTML("<h4>📈 Time Series Controls</h4>"),
            self.timeseries_case_slider,
            self.timeseries_obs_dropdown,
            widgets.HTML("<br>"),
            self.comparison_mode_checkbox,
            self.show_state_based_checkbox,
            widgets.HTML("<p><i><b>Comparison Mode:</b> When enabled, plots will show both state-based and latent-based predictions against ground truth for direct comparison.</i></p>"),
            widgets.HTML("<p><i><b>Show State-based:</b> Toggle visibility of state-based predictions (can be hidden if showing poor results).</i></p>")
        ])
        
        timeseries_tab = widgets.VBox([
            timeseries_controls,
            widgets.HTML("<hr>"),
            self.timeseries_output,
            widgets.HTML("<h4>📊 Performance Metrics</h4>"),
            self.timeseries_metrics_output
        ])
        
        # Overall performance metrics tab
        overall_metrics_tab = self._create_overall_metrics_tab()
        
        # Time evolution animation tab
        animation_tab = self._create_animation_tab()
        
        # Model comparison tab
        comparison_tab = self._create_model_comparison_tab()
        
        # Set up tabs
        self.tab_widget.children = [spatial_tab, timeseries_tab, overall_metrics_tab, animation_tab, comparison_tab]
        self.tab_widget.set_title(0, '🎯 Spatial Predictions')
        self.tab_widget.set_title(1, '📈 Time Series Observations')
        self.tab_widget.set_title(2, '📊 Overall Performance')
        self.tab_widget.set_title(3, '🎬 Time Evolution')
        self.tab_widget.set_title(4, '🔍 Model Comparison')
        
        # Display complete dashboard
        complete_dashboard = widgets.VBox([
            masking_section,
            widgets.HTML("<hr>"),
            save_section,
            widgets.HTML("<hr>"),
            csv_export_section,
            widgets.HTML("<hr>"),
            self.tab_widget
        ])
        
        display(complete_dashboard)
        
        # Generate initial plots
        self._update_spatial_plot()
        self._update_timeseries_plot()

    def _start_animation(self, button):
        """Start time evolution animation and create GIF"""
        import threading
        import time
        from IPython.display import clear_output, display
        
        if self.animation_running:
            return
            
        self.animation_running = True
        self.play_button.disabled = True
        self.stop_button.disabled = False
        self.animation_status.value = 'Animation Status: Running...'
        
        def animate():
            """Animation loop with GIF creation"""
            gif_frames = []  # Store frames for GIF creation
            
            try:
                case_idx = self.anim_case_slider.value
                layer_idx = self.anim_layer_slider.value
                field_idx = self.anim_field_dropdown.value
                speed = self.animation_speed_slider.value
                
                # Create output directory for GIFs
                gif_dir = _ROM_DIR / "animation_gifs"
                gif_dir.mkdir(exist_ok=True)
                
                # Generate filename
                actual_case_idx = self.test_case_indices[case_idx]
                field_name = self.field_names[field_idx]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gif_filename = gif_dir / f"animation_case{actual_case_idx}_layer{layer_idx}_{field_name}_{timestamp}.gif"
                
                for timestep_idx in range(self.num_tstep):
                    if not self.animation_running:
                        break
                        
                    # Update status
                    current_year = self.start_year + timestep_idx
                    self.animation_status.value = f'Animation Status: Playing (Year {current_year}, Step {timestep_idx+1}/{self.num_tstep}) - Creating GIF...'
                    
                    # Create frame and capture for GIF
                    with self.animation_output:
                        clear_output(wait=True)
                        
                        # Create the plot
                        fig = self._create_animation_frame_with_capture(case_idx, layer_idx, field_idx, timestep_idx)
                        
                        # Display the plot
                        display(fig)
                        
                        # Capture frame for GIF
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        gif_frames.append(Image.open(buf))
                        
                        # Close the figure to prevent memory issues
                        try:
                            plt.close(fig)
                        except (AttributeError, RuntimeError):
                            # If closing fails (e.g., manager is None), just continue
                            pass
                    
                    # Wait for next frame
                    time.sleep(speed)
                
                # Create and save GIF if animation completed
                if self.animation_running and len(gif_frames) > 0:
                    self.animation_status.value = 'Animation Status: Saving GIF...'
                    
                    # Create GIF with proper duration
                    gif_frames[0].save(
                        gif_filename,
                        save_all=True,
                        append_images=gif_frames[1:],
                        duration=int(speed * 1000),  # Convert to milliseconds
                        loop=0  # Infinite loop
                    )
                    
                    self.animation_status.value = f'Animation Status: Completed - GIF saved: {gif_filename.name}'
                    print(f"🎬 Animation GIF saved: {gif_filename}")
                else:
                    if len(gif_frames) == 0:
                        self.animation_status.value = 'Animation Status: Stopped - No frames captured'
                    else:
                        self.animation_status.value = 'Animation Status: Stopped'
                        
            except Exception as e:
                self.animation_status.value = f'Animation Status: Error - {str(e)}'
                print(f"Animation error: {e}")
            finally:
                # Clean up
                for frame in gif_frames:
                    frame.close()
                self.animation_running = False
                self.play_button.disabled = False
                self.stop_button.disabled = True
        
        # Start animation in separate thread
        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()
    
    def _stop_animation(self, button):
        """Stop time evolution animation"""
        self.animation_running = False
        self.play_button.disabled = False
        self.stop_button.disabled = True
        self.animation_status.value = 'Animation Status: Stopped'
    
    def _stop_animation_on_change(self, change):
        """Stop animation when controls change to prevent dashboard issues"""
        if hasattr(self, 'animation_running') and self.animation_running:
            self.animation_running = False
            self.play_button.disabled = False
            self.stop_button.disabled = True
            self.animation_status.value = 'Animation Status: Stopped (Control Changed)'
    
    def _create_animation_frame_with_capture(self, case_idx, layer_idx, field_idx, timestep_idx):
        """Create animation frame and return figure for GIF capture - matches spatial tab styling exactly"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get current selections
        actual_case_idx = self.test_case_indices[case_idx]
        field_key = self.field_keys[field_idx]
        field_name = self.field_names[field_idx]
        current_year = self.start_year + timestep_idx
        
        # Get layer mask for inactive cell masking
        layer_mask = self._get_layer_mask(case_idx, layer_idx)
        
        # Get data
        pred_data = self.state_pred[case_idx, timestep_idx, field_idx, :, :, layer_idx].cpu().detach().numpy()
        true_data = self.state_seq_true_aligned[case_idx, field_idx, timestep_idx, :, :, layer_idx].cpu().numpy()
        
        # Denormalize
        pred_data_denorm = self._denormalize_field_data(pred_data, field_key)
        true_data_denorm = self._denormalize_field_data(true_data, field_key)
        
        # Apply masking
        pred_data_masked = np.where(layer_mask, pred_data_denorm, np.nan)
        true_data_masked = np.where(layer_mask, true_data_denorm, np.nan)
        
        # Enhanced color scaling with percentile-based approach (matching spatial tab exactly)
        def get_optimal_color_range(data):
            """Get optimal color range using percentile-based scaling - same as spatial tab"""
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                # Use active data for color scaling, handle positive values preferentially
                active_data = valid_data[valid_data > 0] if len(valid_data[valid_data > 0]) > 0 else valid_data
                if len(active_data) > 0:
                    vmin, vmax = np.percentile(active_data, [2, 98])  # Percentile-based scaling
                else:
                    vmin, vmax = valid_data.min(), valid_data.max()
            else:
                vmin, vmax = 0, 1
            return vmin, vmax
        
        # UNIFIED color scale for predicted and actual panels (matching spatial tab)
        combined_valid_pred = pred_data_masked[~np.isnan(pred_data_masked)]
        combined_valid_true = true_data_masked[~np.isnan(true_data_masked)]
        if len(combined_valid_pred) > 0 and len(combined_valid_true) > 0:
            combined_data = np.concatenate([combined_valid_pred, combined_valid_true])
            unified_vmin, unified_vmax = get_optimal_color_range(combined_data.reshape(-1, 1).flatten())
        elif len(combined_valid_pred) > 0:
            unified_vmin, unified_vmax = get_optimal_color_range(combined_valid_pred)
        elif len(combined_valid_true) > 0:
            unified_vmin, unified_vmax = get_optimal_color_range(combined_valid_true)
        else:
            unified_vmin, unified_vmax = 0, 1
            
        # Create plot with exact same styling as spatial tab
        plt.style.use('default')  # Clean style
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), dpi=100)  # Match spatial tab proportions
        
        # Enhanced title with minimal spacing
        fig.suptitle(f'{field_name} - Case {actual_case_idx} - K-Layer {layer_idx+1} - Year {current_year}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # High contrast colormap with NaN handling (same as spatial tab)
        cmap = plt.cm.jet.copy()  # Blue-Red (high contrast) colormap
        cmap.set_bad('white', alpha=0.3)  # Semi-transparent for masked cells
        
        # PREDICTED PANEL (matching spatial tab exactly)
        im1 = axes[0].imshow(pred_data_masked.T,  # Transpose for proper orientation
                           origin='lower',  # Proper grid orientation
                           cmap=cmap, 
                           vmin=unified_vmin, 
                           vmax=unified_vmax,
                           aspect='equal',  # Equal aspect ratio
                           interpolation='bilinear')  # Smooth interpolation
        
        axes[0].set_title('Predicted', fontsize=16, fontweight='bold', pad=15)
        axes[0].set_xlabel('I Index', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('J Index', fontsize=14, fontweight='bold')
        axes[0].tick_params(labelsize=12, width=1.5)
        # Make tick labels bold
        for label in axes[0].get_xticklabels():
            label.set_fontweight('bold')
        for label in axes[0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Enhanced colorbar - exact same configuration as spatial tab
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.7, aspect=30, pad=0.02)
        cbar1.set_label(f'{field_name}', 
                       rotation=90, labelpad=15, fontsize=14, fontweight='bold')
        cbar1.ax.tick_params(labelsize=12, width=1.5)
        # Make colorbar tick labels bold
        for label in cbar1.ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # TRUE PANEL (matching spatial tab exactly)
        im2 = axes[1].imshow(true_data_masked.T,  # Transpose for proper orientation
                           origin='lower',  # Proper grid orientation
                           cmap=cmap, 
                           vmin=unified_vmin, 
                           vmax=unified_vmax,
                           aspect='equal',  # Equal aspect ratio
                           interpolation='bilinear')  # Smooth interpolation
        
        axes[1].set_title('True', fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel('I Index', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('J Index', fontsize=14, fontweight='bold')
        axes[1].tick_params(labelsize=12, width=1.5)
        # Make tick labels bold
        for label in axes[1].get_xticklabels():
            label.set_fontweight('bold')
        for label in axes[1].get_yticklabels():
            label.set_fontweight('bold')
        
        # Enhanced colorbar - exact same configuration as spatial tab
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, aspect=30, pad=0.02)
        cbar2.set_label(f'{field_name}', 
                       rotation=90, labelpad=15, fontsize=14, fontweight='bold')
        cbar2.ax.tick_params(labelsize=12, width=1.5)
        # Make colorbar tick labels bold
        for label in cbar2.ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add well overlays if available (same as spatial tab)
        if hasattr(self, 'well_locations') and self.well_locations:
            self._add_well_overlays([axes[0], axes[1]], layer_idx)
        
        # Final layout optimization (matching spatial tab)
        # Apply 3-digit formatting to all axes
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(format_3digits))
            ax.yaxis.set_major_formatter(FuncFormatter(format_3digits))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.08)  # Minimal vertical spacing for compact layout
        
        # Return figure for capture instead of showing
        return fig

    def _get_field_unit_for_animation(self, field_name):
        """Get field unit for animation plots"""
        return self.metrics_evaluator._get_field_unit(field_name)

    def verify_masking_status(self, case_idx, field_idx, layer_idx, timestep_idx):
        """
        Verify that inactive cell masking is properly applied
        
        Args:
            case_idx: Case index
            field_idx: Field/channel index  
            layer_idx: Layer index
            timestep_idx: Time step index
            
        Returns:
            Dictionary with masking verification details
        """
        dashboard_ref = getattr(self, 'dashboard_ref', None)
        
        if dashboard_ref is None:
            return {
                'masking_enabled': False,
                'reason': 'No dashboard reference available'
            }
        
        if not hasattr(dashboard_ref, '_get_layer_mask'):
            return {
                'masking_enabled': False,
                'reason': 'Dashboard has no _get_layer_mask method'
            }
        
        if not hasattr(dashboard_ref, 'use_masking_checkbox'):
            return {
                'masking_enabled': False,
                'reason': 'Dashboard has no masking checkbox'
            }
        
        if not dashboard_ref.use_masking_checkbox.value:
            return {
                'masking_enabled': False,
                'reason': 'Masking checkbox is disabled'
            }
        
        if not dashboard_ref.masks_loaded_successfully:
            return {
                'masking_enabled': False,
                'reason': 'No mask files loaded successfully'
            }
        
        # Get the actual mask
        try:
            layer_mask = dashboard_ref._get_layer_mask(case_idx, layer_idx)
            total_cells = layer_mask.size
            active_cells = np.sum(layer_mask)
            inactive_cells = total_cells - active_cells
            
            return {
                'masking_enabled': True,
                'total_cells': total_cells,
                'active_cells': active_cells,
                'inactive_cells': inactive_cells,
                'mask_type': dashboard_ref.mask_type,
                'masking_percentage': (inactive_cells / total_cells) * 100
            }
        except Exception as e:
            return {
                'masking_enabled': False,
                'reason': f'Error getting mask: {str(e)}'
            }

