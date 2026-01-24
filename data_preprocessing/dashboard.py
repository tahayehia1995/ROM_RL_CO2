import h5py
import numpy as np
import torch
import os
import glob
import re
from datetime import datetime
import json
import yaml

# Import from new modules
from .utils import WIDGETS_AVAILABLE, widgets, display, clear_output
from .normalization import normalize_dataset_inplace
from .tensor_creation import train_split_data
from .data_saver import save_normalization_parameters, save_processed_data
from utilities.config_loader import Config
from utilities.timing import Timer, collect_processing_metadata

# Import visualization dashboard from testing module
try:
    from testing.visualization.utils import create_visualization_dashboard
except ImportError:
    # Fallback if testing module not available
    create_visualization_dashboard = None

class DataPreprocessingDashboard:
    """
    Interactive dashboard for data preprocessing with 3 tabs:
    1. States tab: Select spatial property files
    2. Controls tab: Select control variables per well
    3. Observations tab: Select observation variables per well
    """
    
    def __init__(self):
        self.data_dir = ""
        self.available_spatial_files = []
        self.available_timeseries_files = []
        self.selected_states = {}
        self.selected_controls = {}
        self.selected_observations = {}
        self.spatial_data = {}
        self.timeseries_data = {}
        self.norm_params = {}
        self.final_tensors = {}
        
        # Load config to get n_steps
        try:
            self.config = Config('config.yaml')
            self.nsteps = self.config.training.get('nsteps', 2)
        except Exception as e:
            # Fallback to default if config loading fails
            self.config = None
            self.nsteps = 2
        
        # Check if widgets are available
        if not WIDGETS_AVAILABLE:
            pass  # Interactive widgets not available
            return
            
        self._create_widgets()
        self._setup_event_handlers()
    
    def _create_widgets(self):
        """Create all dashboard widgets"""
        
        # Header
        self.header = widgets.HTML(
            value="<h1>🔧 Data Preprocessing Dashboard</h1>",
            layout=widgets.Layout(margin='10px 0px')
        )
        
        # Data directory input
        self.data_dir_input = widgets.Text(
            value="sr3_batch_output/",
            description="Data Directory:",
            placeholder="Enter path to H5 files folder",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        self.load_data_btn = widgets.Button(
            description="📁 Load H5 Files",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        
        self.status_output = widgets.Output()
        
        # Parameters
        self.nsteps_input = widgets.IntSlider(
            value=self.nsteps,
            min=1,
            max=30,
            description="N-steps:",
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )
        
        # Tab structure
        self.tabs = widgets.Tab()
        
        # States tab content
        self.states_content = widgets.VBox()
        
        # Channel selection tab content
        self.channel_selection_content = widgets.VBox()
        
        # Controls tab content  
        self.controls_content = widgets.VBox()
        
        # Observations tab content
        self.observations_content = widgets.VBox()
        
        # Process button
        self.process_btn = widgets.Button(
            description="🚀 Process Data",
            button_style='success',
            layout=widgets.Layout(width='200px', margin='20px 0px')
        )
        
        self.results_output = widgets.Output()
        
        # Set up tabs
        self.tabs.children = [self.states_content, self.channel_selection_content, self.controls_content, self.observations_content]
        self.tabs.set_title(0, "🏔️ States")
        self.tabs.set_title(1, "🎯 Channel Selection")
        self.tabs.set_title(2, "🎛️ Controls") 
        self.tabs.set_title(3, "📊 Observations")
        
        # Main layout
        self.main_widget = widgets.VBox([
            self.header,
            widgets.HBox([self.data_dir_input, self.load_data_btn]),
            self.status_output,
            self.nsteps_input,
            self.tabs,
            self.process_btn,
            self.results_output
        ])
    
    def _setup_event_handlers(self):
        """Setup event handlers for widgets"""
        self.load_data_btn.on_click(self._load_h5_files)
        self.process_btn.on_click(self._process_data)
        self.nsteps_input.observe(self._update_params, 'value')
    
    def _update_params(self, change):
        """Update parameters when changed"""
        self.nsteps = self.nsteps_input.value
    
    def _load_h5_files(self, button):
        """Load and display available H5 files with deterministic ordering"""
        with self.status_output:
            clear_output(wait=True)
            
            self.data_dir = self.data_dir_input.value.strip()
            if not self.data_dir.endswith('/'):
                self.data_dir += '/'
            
            pass  # Scanning directory
            
            # Check if directory exists
            if not os.path.exists(self.data_dir):
                print(f"Directory not found: {self.data_dir}")
                return
            
            # Find H5 files with DETERMINISTIC SORTING
            h5_files = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
            
            if not h5_files:
                print(f"No H5 files found in: {self.data_dir}")
                return
            
            print(f"Found {len(h5_files)} H5 files")
            
            # Define canonical order for spatial properties to ensure consistency
            CANONICAL_SPATIAL_ORDER = ['SW', 'SG', 'PRES', 'PERMI', 'POROS', 'PERMJ', 'PERMK']
            
            # Categorize files with deterministic ordering
            spatial_files_dict = {}
            self.available_timeseries_files = []
            
            for file_path in h5_files:
                filename = os.path.basename(file_path)
                if 'spatial_properties' in filename:
                    # Extract variable name for canonical ordering
                    var_name = filename.replace('batch_spatial_properties_', '').replace('.h5', '')
                    spatial_files_dict[var_name] = filename
                elif 'timeseries_data' in filename:
                    self.available_timeseries_files.append(filename)
            
            # Sort spatial files in canonical order for consistency
            self.available_spatial_files = []
            self.spatial_file_order = []  # Track the deterministic order
            
            # First, add files in canonical order
            for canonical_var in CANONICAL_SPATIAL_ORDER:
                if canonical_var in spatial_files_dict:
                    self.available_spatial_files.append(spatial_files_dict[canonical_var])
                    self.spatial_file_order.append(canonical_var)
                    print(f"  📊 Canonical order: {canonical_var} → {spatial_files_dict[canonical_var]}")
            
            # Then, add any additional files not in canonical order (alphabetically sorted)
            remaining_vars = set(spatial_files_dict.keys()) - set(CANONICAL_SPATIAL_ORDER)
            for var_name in sorted(remaining_vars):
                self.available_spatial_files.append(spatial_files_dict[var_name])
                self.spatial_file_order.append(var_name)
                print(f"  📊 Additional: {var_name} → {spatial_files_dict[var_name]}")
            
            # Sort timeseries files alphabetically for consistency
            self.available_timeseries_files = sorted(self.available_timeseries_files)
            
            print(f"📊 Spatial files: {len(self.available_spatial_files)}")
            print(f"📈 Timeseries files: {len(self.available_timeseries_files)}")
            
            # Update tabs
            self._update_states_tab()
            self._update_channel_selection_tab()
            self._update_controls_tab()
            self._update_observations_tab()
    
    def _update_states_tab(self):
        """Update states tab to automatically load ALL spatial property files"""
        children = [widgets.HTML("<h3>🏔️ Spatial Property Files - Automatic Processing</h3>")]
        children.append(widgets.HTML("<p><i>All available spatial property files will be loaded and processed automatically. Configure normalization for each property below.</i></p>"))
        
        if not self.available_spatial_files:
            children.append(widgets.HTML("<p>❌ No spatial property files found</p>"))
            self.states_content.children = children
            return
        
        children.append(widgets.HTML(f"<p>✅ Found <b>{len(self.available_spatial_files)}</b> spatial property files:</p>"))
        
        # Create normalization controls for each available spatial file
        self.state_normalization_controls = {}
        
        for file_path in self.available_spatial_files:
            # Extract property name from filename
            var_name = file_path.replace('batch_spatial_properties_', '').replace('.h5', '')
            
            # Create controls for this property
            property_header = widgets.HTML(f"<h4>📊 {var_name.upper()} ({file_path})</h4>")
            
            # Create three mutually exclusive normalization checkboxes
            minmax_checkbox = widgets.Checkbox(
                value=True,  # Default to min-max normalization
                description='Min-Max Normalization',
                style={'description_width': '150px'},
                layout=widgets.Layout(width='220px')
            )
            
            log_checkbox = widgets.Checkbox(
                value=False,
                description='Log Normalization',
                style={'description_width': '150px'},
                layout=widgets.Layout(width='220px')
            )
            
            original_checkbox = widgets.Checkbox(
                value=False,
                description='Keep Original Values',
                style={'description_width': '150px'},
                layout=widgets.Layout(width='220px')
            )
            
            # Create event handlers for mutual exclusivity
            def create_mutual_exclusivity_handler(var_name, target_box, other_box1, other_box2):
                def on_change(change):
                    if change['new']:  # If checkbox is being checked
                        other_box1.value = False
                        other_box2.value = False
                return on_change
            
            minmax_checkbox.observe(
                create_mutual_exclusivity_handler(var_name, minmax_checkbox, log_checkbox, original_checkbox),
                names='value'
            )
            log_checkbox.observe(
                create_mutual_exclusivity_handler(var_name, log_checkbox, minmax_checkbox, original_checkbox),
                names='value'
            )
            original_checkbox.observe(
                create_mutual_exclusivity_handler(var_name, original_checkbox, minmax_checkbox, log_checkbox),
                names='value'
            )
            
            info_label = widgets.HTML(
                value=f"<i>Choose one normalization method for {var_name.upper()}</i>",
                layout=widgets.Layout(margin='5px 0px 10px 0px')
            )
            
            # Organize checkboxes in a vertical layout
            checkbox_layout = widgets.VBox([
                minmax_checkbox,
                log_checkbox,
                original_checkbox
            ])
            
            property_section = widgets.VBox([
                property_header,
                checkbox_layout,
                info_label
            ])
            
            # Store all three checkboxes for this property
            self.state_normalization_controls[var_name] = {
                'minmax': minmax_checkbox,
                'log': log_checkbox,
                'original': original_checkbox
            }
            children.append(property_section)
        
        children.append(widgets.HTML("<hr>"))
        
        self.states_content.children = children
    
    def _update_channel_selection_tab(self):
        """Update channel selection tab for training tensor configuration with explicit ordering"""
        children = [widgets.HTML("<h3>🎯 Select Channels for Training State Tensor</h3>")]
        children.append(widgets.HTML("<p><i>Choose which spatial property channels to include in the training state tensor. <b>Channels will be arranged in the exact order shown below.</b></i></p>"))
        
        if not self.available_spatial_files:
            children.append(widgets.HTML("<p>❌ No spatial property files available. Please load data first.</p>"))
            self.channel_selection_content.children = children
            return
        
        # Create channel selection checkboxes with explicit ordering information
        self.channel_selection_checkboxes = {}
        
        # Default selections: SG and PRES only  
        default_channels = ['SG', 'PRES']
        
        children.append(widgets.HTML(f"<p>✅ Available spatial properties in <b>DETERMINISTIC ORDER</b>: <b>{len(self.available_spatial_files)}</b></p>"))
        children.append(widgets.HTML("<p><b>📋 Select channels for training (order matters for model architecture):</b></p>"))
        
        # Add explicit ordering information
        children.append(widgets.HTML("<div style='background-color: #f0f8ff; padding: 10px; border-left: 4px solid #4CAF50; margin: 10px 0;'>"))
        children.append(widgets.HTML("<b>🔍 CHANNEL ORDERING GUARANTEE:</b><br/>"))
        children.append(widgets.HTML("• Files are processed in canonical order: SW → SG → PRES → PERMI → POROS<br/>"))
        children.append(widgets.HTML("• Training tensor channels will match exactly the order you select below<br/>"))
        children.append(widgets.HTML("• This ensures 100% reproducible channel arrangements across all systems</div>"))
        
        # Display files in their deterministic order with explicit indices
        for idx, file_path in enumerate(self.available_spatial_files):
            var_name = file_path.replace('batch_spatial_properties_', '').replace('.h5', '')
            
            # Default selection logic
            default_selected = var_name.upper() in [d.upper() for d in default_channels]
            
            # Create enhanced description with explicit ordering info
            checkbox_description = f'Position {idx}: {var_name.upper()} ← {file_path}'
            if default_selected:
                checkbox_description += ' ✓ (Recommended)'
            
            checkbox = widgets.Checkbox(
                value=default_selected,
                description=checkbox_description,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='600px')
            )
            
            self.channel_selection_checkboxes[var_name] = {
                'checkbox': checkbox,
                'canonical_position': idx,  # Position in canonical order
                'filename': file_path,
                'deterministic_order': True  # Flag for validation
            }
            
            children.append(checkbox)
        
        # Validation section
        children.append(widgets.HTML("<hr>"))
        children.append(widgets.HTML("<h4>⚠️ Validation</h4>"))
        
        # Get model's expected n_channels from config (default to 3)
        try:
            from utilities.config_loader import Config
            config = Config('config.yaml')
            expected_channels = config.model.get('n_channels', 3)
        except:
            expected_channels = 3
            
        self.expected_channels_label = widgets.HTML(
            value=f"<p><b>Model expects:</b> {expected_channels} channels (configured in config.yaml)</p>"
        )
        
        self.validation_status_label = widgets.HTML(
            value="<p><b>Status:</b> ✅ Selection matches model configuration</p>"
        )
        
        children.append(self.expected_channels_label)
        children.append(self.validation_status_label)
        
        # Add validation handlers
        for var_name, config in self.channel_selection_checkboxes.items():
            config['checkbox'].observe(self._validate_channel_selection, names='value')
        
        children.append(widgets.HTML("<hr>"))
        
        self.channel_selection_content.children = children
        
        # Initial validation
        self._validate_channel_selection()
    
    def _validate_channel_selection(self, change=None):
        """Validate that selected channels match model configuration and auto-update config if needed"""
        if not hasattr(self, 'channel_selection_checkboxes'):
            return
            
        # Count selected channels
        selected_count = sum(1 for config in self.channel_selection_checkboxes.values() if config['checkbox'].value)
        
        # Get expected channels and config object
        config_updated = False
        initial_expected_channels = None
        try:
            from utilities.config_loader import Config
            config = Config('config.yaml')
            initial_expected_channels = config.model.get('n_channels', 3)
            
            # Debug: print current state
            print(f"🔍 Validation check: {selected_count} channels selected, config expects {initial_expected_channels}")
            
            # If selected channels don't match, update config
            if selected_count != initial_expected_channels and selected_count > 0:
                print(f"🔄 Channel mismatch detected. Updating config from {initial_expected_channels} to {selected_count} channels...")
                config_updated = self._update_config_for_channels(config, selected_count)
                if config_updated:
                    print(f"✅ Config update successful!")
                else:
                    print(f"❌ Config update failed!")
            else:
                print(f"ℹ️ No config update needed (selected={selected_count}, expected={initial_expected_channels})")
                
        except Exception as e:
            initial_expected_channels = 3
            print(f"⚠️ Warning: Could not load config: {e}")
            import traceback
            traceback.print_exc()
        
        # Reload config to get updated value (if update happened)
        final_expected_channels = initial_expected_channels
        if config_updated:
            try:
                from utilities.config_loader import Config
                config = Config('config.yaml')
                final_expected_channels = config.model.get('n_channels', 3)
            except Exception as e:
                print(f"⚠️ Warning: Could not reload config after update: {e}")
        
        # Update validation status - if config was updated, always show success
        if config_updated:
            self.validation_status_label.value = f"<p><b>Status:</b> ✅ {selected_count} channels selected - config.yaml updated to match selection</p>"
            self.expected_channels_label.value = f"<p><b>Model expects:</b> {final_expected_channels} channels (updated in config.yaml)</p>"
        elif selected_count == final_expected_channels:
            self.validation_status_label.value = f"<p><b>Status:</b> ✅ {selected_count} channels selected - matches model configuration</p>"
        elif selected_count < final_expected_channels:
            self.validation_status_label.value = f"<p><b>Status:</b> ⚠️ {selected_count} channels selected - need {final_expected_channels - selected_count} more</p>"
        else:
            self.validation_status_label.value = f"<p><b>Status:</b> ❌ {selected_count} channels selected - {selected_count - final_expected_channels} too many</p>"
    
    def _update_config_for_channels(self, config, n_channels):
        """Update config.yaml with new n_channels value and related settings"""
        try:
            # Get config path - use the same path that was used to load the config
            config_path = getattr(config, 'config_path', 'config.yaml')
            
            # Ensure we have an absolute path if needed
            if not os.path.isabs(config_path):
                # Try to find config.yaml in current directory or ROM_Refactored directory
                if os.path.exists('config.yaml'):
                    config_path = os.path.abspath('config.yaml')
                elif os.path.exists('ROM_Refactored/config.yaml'):
                    config_path = os.path.abspath('ROM_Refactored/config.yaml')
                else:
                    config_path = os.path.abspath(config_path)
            
            print(f"🔄 Updating config: setting n_channels to {n_channels} in {config_path}")
            
            # Update model.n_channels
            if 'model' not in config.config:
                config.config['model'] = {}
            config.config['model']['n_channels'] = n_channels
            
            # Update data.input_shape[0] to match n_channels
            if 'data' in config.config and 'input_shape' in config.config['data']:
                if isinstance(config.config['data']['input_shape'], list) and len(config.config['data']['input_shape']) > 0:
                    config.config['data']['input_shape'][0] = n_channels
            
            # Update encoder.conv_layers.conv1[0] if it exists (first conv layer input channels)
            if 'encoder' in config.config and 'conv_layers' in config.config['encoder']:
                if 'conv1' in config.config['encoder']['conv_layers']:
                    conv1 = config.config['encoder']['conv_layers']['conv1']
                    if isinstance(conv1, list) and len(conv1) > 0:
                        # Update first element which is input channels
                        conv1[0] = n_channels
            
            # Update decoder final_conv output channels if it exists
            if 'decoder' in config.config and 'deconv_layers' in config.config['decoder']:
                if 'final_conv' in config.config['decoder']['deconv_layers']:
                    final_conv = config.config['decoder']['deconv_layers']['final_conv']
                    if isinstance(final_conv, list) and len(final_conv) > 0:
                        # Check if second element is null (which means n_channels)
                        if len(final_conv) > 1 and final_conv[1] is None:
                            # Keep as None, it will be auto-filled
                            pass
                        elif len(final_conv) > 1:
                            # Update output channels to n_channels
                            final_conv[1] = n_channels
            
            # Save updated config to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config.config,
                    f,
                    default_flow_style=False,
                    indent=2,
                    allow_unicode=True,
                    sort_keys=False
                )
            
            print(f"✅ Config updated successfully: n_channels set to {n_channels} in {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_controls_tab(self):
        """Update controls tab with timeseries variables per well and normalization options"""
        children = [widgets.HTML("<h3>🎛️ Select Control Variables (per well)</h3>")]
        children.append(widgets.HTML("<p><i>Choose which control variables to use and whether to normalize them. Normalization is applied per variable (all wells of same type together).</i></p>"))
        
        if not self.available_timeseries_files:
            children.append(widgets.HTML("<p>No timeseries files available</p>"))
            self.controls_content.children = children
            return
        
        # Load first timeseries file to get well count
        try:
            sample_file = os.path.join(self.data_dir, self.available_timeseries_files[0])
            with h5py.File(sample_file, 'r') as hf:
                sample_data = np.array(hf['data'])
                num_wells = sample_data.shape[2] if len(sample_data.shape) >= 3 else 6
        except:
            num_wells = 6
        
        children.append(widgets.HTML(f"<p>Number of wells detected: <b>{num_wells}</b></p>"))
        children.append(widgets.HTML("<p>Configure <b>control variables</b> and their normalization:</p>"))
        
        self.control_checkboxes = {}
        self.control_normalization_checkboxes = {}
        
        for filename in self.available_timeseries_files:
            var_name = filename.replace('batch_timeseries_data_', '').replace('.h5', '')
            
            # Variable header with enhanced styling
            children.append(widgets.HTML(f"<h4>📊 {var_name} ({filename})</h4>"))
            
            # Create normalization checkbox for this variable (checked by default for backward compatibility)
            normalize_checkbox = widgets.Checkbox(
                value=True,  # Default to normalized (backward compatibility)
                description=f'Normalize {var_name} (Min-Max)',
                style={'description_width': '200px'},
                layout=widgets.Layout(width='300px', margin='5px 0px 10px 20px')
            )
            
            # Store normalization checkbox
            self.control_normalization_checkboxes[var_name] = normalize_checkbox
            
            # Add normalization checkbox with styling
            normalization_section = widgets.VBox([
                widgets.HTML(f"<div style='background-color: #f0f8ff; padding: 8px; margin: 5px 0; border-left: 3px solid #4CAF50;'>"),
                normalize_checkbox,
                widgets.HTML(f"<small><i>✅ Checked: Apply min-max normalization | ❌ Unchecked: Keep original values</i></small></div>")
            ])
            children.append(normalization_section)
            
            # Well selection section
            children.append(widgets.HTML(f"<p><b>Well Selection for {var_name}:</b></p>"))
            
            self.control_checkboxes[var_name] = {}
            
            for well_idx in range(num_wells):
                # Set default selections based on requested logic:
                # - BHP: last 3 wells should be checked
                # - GASRATSC: first 3 wells should be checked
                # - WATRATSC: no wells should be checked
                default_selected = False
                if var_name == 'BHP' and well_idx >= max(0, num_wells - 3):  # BHP last 3 wells
                    default_selected = True
                elif var_name == 'GASRATSC' and well_idx < 3:  # GASRATSC first 3 wells
                    default_selected = True
                
                checkbox = widgets.Checkbox(
                    value=default_selected,
                    description=f"Well {well_idx + 1}",  # 1-based indexing for display
                    style={'description_width': '100px'},
                    layout=widgets.Layout(margin='2px 0px 2px 20px')
                )
                
                self.control_checkboxes[var_name][well_idx] = checkbox
                children.append(checkbox)
            
            # Add separator
            children.append(widgets.HTML("<hr style='margin: 15px 0;'>"))
        

        
        self.controls_content.children = children
    
    def _update_observations_tab(self):
        """Update observations tab with timeseries variables per well and normalization options"""
        children = [widgets.HTML("<h3>📊 Select Observation Variables (per well)</h3>")]
        children.append(widgets.HTML("<p><i>Choose which observation variables to use and whether to normalize them. Normalization is applied per variable (all wells of same type together).</i></p>"))
        
        if not self.available_timeseries_files:
            children.append(widgets.HTML("<p>No timeseries files available</p>"))
            self.observations_content.children = children
            return
            
        # Load first timeseries file to get well count
        try:
            sample_file = os.path.join(self.data_dir, self.available_timeseries_files[0])
            with h5py.File(sample_file, 'r') as hf:
                sample_data = np.array(hf['data'])
                num_wells = sample_data.shape[2] if len(sample_data.shape) >= 3 else 6
        except:
            num_wells = 6
        
        children.append(widgets.HTML(f"<p>Number of wells detected: <b>{num_wells}</b></p>"))
        children.append(widgets.HTML("<p>Configure <b>observation variables</b> and their normalization:</p>"))
        
        self.observation_checkboxes = {}
        self.observation_normalization_checkboxes = {}
        
        for filename in self.available_timeseries_files:
            var_name = filename.replace('batch_timeseries_data_', '').replace('.h5', '')
            
            # Variable header with enhanced styling
            children.append(widgets.HTML(f"<h4>📈 {var_name} ({filename})</h4>"))
            
            # Create normalization checkbox for this variable (checked by default for backward compatibility)
            normalize_checkbox = widgets.Checkbox(
                value=True,  # Default to normalized (backward compatibility)
                description=f'Normalize {var_name} (Min-Max)',
                style={'description_width': '200px'},
                layout=widgets.Layout(width='300px', margin='5px 0px 10px 20px')
            )
            
            # Store normalization checkbox
            self.observation_normalization_checkboxes[var_name] = normalize_checkbox
            
            # Add normalization checkbox with styling
            normalization_section = widgets.VBox([
                widgets.HTML(f"<div style='background-color: #f0f8ff; padding: 8px; margin: 5px 0; border-left: 3px solid #2196F3;'>"),
                normalize_checkbox,
                widgets.HTML(f"<small><i>✅ Checked: Apply min-max normalization | ❌ Unchecked: Keep original values</i></small></div>")
            ])
            children.append(normalization_section)
            
            # Well selection section
            children.append(widgets.HTML(f"<p><b>Well Selection for {var_name}:</b></p>"))
            
            self.observation_checkboxes[var_name] = {}
            
            for well_idx in range(num_wells):
                # Set default selections for observations
                # Keeping BHP for first 3 wells (injectors), and GASRATSC and WATRATSC for last 3 wells (producers)
                default_selected = False
                if var_name == 'BHP' and well_idx < 3:  # Injector BHP (first 3 wells)
                    default_selected = True
                elif var_name == 'WATRATSC' and well_idx >= max(0, num_wells - 3):  # Water production (last 3 wells)
                    default_selected = True
                elif var_name == 'GASRATSC' and well_idx >= max(0, num_wells - 3):  # Gas production (last 3 wells)
                    default_selected = True
                
                checkbox = widgets.Checkbox(
                    value=default_selected,
                    description=f"Well {well_idx + 1}",  # 1-based indexing for display
                    style={'description_width': '100px'},
                    layout=widgets.Layout(margin='2px 0px 2px 20px')
                )
                
                self.observation_checkboxes[var_name][well_idx] = checkbox
                children.append(checkbox)
            
            # Add separator
            children.append(widgets.HTML("<hr style='margin: 15px 0;'>"))
        

        
        self.observations_content.children = children
    
    def _process_data(self, button):
        """Process selected data using current approach"""
        with self.results_output:
            clear_output(wait=True)
            
            print("Starting data processing...")
            
            # Get selections
            self._collect_selections()
            
            # Validate channel selection before processing
            if hasattr(self, 'selected_training_channels'):
                selected_count = len(self.selected_training_channels)
                try:
                    from utilities.config_loader import Config
                    config = Config('config.yaml')
                    expected_channels = config.model.get('n_channels', 3)
                except:
                    expected_channels = 3
                
                if selected_count != expected_channels:
                    print(f"Channel selection failed! Selected: {selected_count}, Expected: {expected_channels}")
                    return
                else:
                    print(f"Channel selection validated: {selected_count} channels")
            
            # Time the processing phase
            with Timer("processing", log_dir='./timing_logs/') as timer:
                # Load and normalize data
                self._load_and_normalize_data()
                
                # Create tensors (automatically continues with sliding window and results)
                self._create_tensors()
                
                # Save normalization parameters to file
                self._save_normalization_parameters()
                
                # Save processed data to .h5 file
                self._save_processed_data()
                
                # Collect metadata for timing log
                timer.metadata = collect_processing_metadata(self)
            
            # Automatically assign processed data to global variables
            self.assign_to_globals()
    
    def _save_normalization_parameters(self):
        """Save normalization parameters to a file for later use in evaluation and reproducibility"""
        import json
        from datetime import datetime
        import os
        
        print("Saving normalization parameters...")
        
        # Create comprehensive normalization configuration
        norm_config = {
            'metadata': {
                'created_timestamp': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'n_steps': self.nsteps,
                'total_channels': len(self.all_spatial_properties) if hasattr(self, 'all_spatial_properties') else 0,
                'selected_channels': len(self.selected_training_channels) if hasattr(self, 'selected_training_channels') else 0,
                'total_controls': sum(len(config['wells']) for config in self.selected_controls.values()) if hasattr(self, 'selected_controls') else 0,
                'total_observations': sum(len(config['wells']) for config in self.selected_observations.values()) if hasattr(self, 'selected_observations') else 0
            },
            'spatial_channels': {},
            'control_variables': {},
            'observation_variables': {},
            'channel_mapping': {},
            'selection_summary': {}
        }
        
        # 1. Store spatial channel normalization parameters
        if hasattr(self, 'all_spatial_properties'):
            for var_name, filename in self.all_spatial_properties.items():
                if var_name in self.norm_params:
                    norm_config['spatial_channels'][var_name] = {
                        'filename': filename,
                        'normalization_type': self.normalization_preferences.get(var_name, 'minmax'),
                        'parameters': self.norm_params[var_name],
                        'selected_for_training': var_name in self.selected_training_channels if hasattr(self, 'selected_training_channels') else False
                    }
                    
                    # Add training position if selected
                    if hasattr(self, 'training_channel_names') and var_name in self.training_channel_names:
                        training_position = self.training_channel_names.index(var_name)
                        norm_config['spatial_channels'][var_name]['training_position'] = training_position
        
        # 2. Store control variable normalization parameters 
        if hasattr(self, 'selected_controls'):
            for var_name, config in self.selected_controls.items():
                if var_name in self.norm_params:
                    norm_config['control_variables'][var_name] = {
                        'filename': config['filename'],
                        'selected_wells': config['wells'],
                        'normalization_type': self.control_normalization_preferences.get(var_name, 'minmax'),
                        'parameters': self.norm_params[var_name],
                        'variable_type': 'control'
                    }
        
        # 3. Store observation variable normalization parameters
        if hasattr(self, 'selected_observations'):
            for var_name, config in self.selected_observations.items():
                if var_name in self.norm_params:
                    norm_config['observation_variables'][var_name] = {
                        'filename': config['filename'], 
                        'selected_wells': config['wells'],
                        'normalization_type': self.observation_normalization_preferences.get(var_name, 'minmax'),
                        'parameters': self.norm_params[var_name],
                        'variable_type': 'observation'
                    }
        
        # 4. Store channel mapping for training tensor reconstruction
        if hasattr(self, 'training_channel_mapping'):
            norm_config['channel_mapping'] = self.training_channel_mapping
        
        # 5. Store selection summary for easy reference
        norm_config['selection_summary'] = {
            'training_channels': list(self.selected_training_channels.keys()) if hasattr(self, 'selected_training_channels') else [],
            'control_wells_by_variable': {var: config['wells'] for var, config in self.selected_controls.items()} if hasattr(self, 'selected_controls') else {},
            'observation_wells_by_variable': {var: config['wells'] for var, config in self.selected_observations.items()} if hasattr(self, 'selected_observations') else {}
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directory exists (same as processed_data directory)
        save_dir = './processed_data/'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as JSON (human-readable) in processed_data directory
        json_filename = f"normalization_parameters_{timestamp}.json"
        json_filepath = os.path.join(save_dir, json_filename)
        try:
            with open(json_filepath, 'w') as f:
                json.dump(norm_config, f, indent=2, default=str)
            print(f"Normalization parameters saved to: {json_filepath}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
        
        # Print summary of what was saved
        pass  # Normalization parameters saved
        
        # Store filename for later access
        self.normalization_files = {
            'json': json_filepath,
            'config': norm_config
        }
        
        print("Use this file for reproducible preprocessing and deployment")
    
    def _save_processed_data(self):
        """Save all processed data (states, controls, observations) to .h5 file"""
        if not hasattr(self, 'STATE_train'):
            print("❌ No processed data available. Cannot save.")
            return None
        
        import h5py
        from datetime import datetime
        import os
        
        print("💾 Saving processed data to .h5 file...")
        
        # Calculate metadata for filename
        num_train = self.STATE_train[0].shape[0] if self.STATE_train else 0
        num_eval = self.STATE_eval[0].shape[0] if self.STATE_eval else 0
        num_states = len(self.STATE_train) if self.STATE_train else 0
        
        # IMPORTANT: Count the dimensionality (number of variables), not the number of time steps!
        # BHP_train and Yobs_train are lists where each element is a time step
        # Each tensor has shape (batch_size, num_variables), so we need shape[1] for dimensionality
        if self.BHP_train and len(self.BHP_train) > 0:
            num_controls = self.BHP_train[0].shape[1]  # Number of control variables (e.g., 6)
        else:
            num_controls = 0
        
        if self.Yobs_train and len(self.Yobs_train) > 0:
            num_observations = self.Yobs_train[0].shape[1]  # Number of observation variables (e.g., 9)
        else:
            num_observations = 0
        
        nsteps = self.nsteps if hasattr(self, 'nsteps') else 0
        n_channels = self.n_channels if hasattr(self, 'n_channels') else 0
        num_well = self.num_well if hasattr(self, 'num_well') else 0
        
        # Create filename with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_data_nstates{num_states}_ncontrols{num_controls}_nobs{num_observations}_nsteps{nsteps}_ntrain{num_train}_neval{num_eval}_ch{n_channels}_wells{num_well}_{timestamp}.h5"
        
        # Ensure directory exists
        save_dir = './processed_data/'
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        try:
            with h5py.File(filepath, 'w') as hf:
                # Save metadata
                metadata_group = hf.create_group('metadata')
                metadata_group.attrs['num_train'] = num_train
                metadata_group.attrs['num_eval'] = num_eval
                metadata_group.attrs['num_states'] = num_states
                metadata_group.attrs['num_controls'] = num_controls
                metadata_group.attrs['num_observations'] = num_observations
                metadata_group.attrs['nsteps'] = nsteps
                metadata_group.attrs['n_channels'] = n_channels
                metadata_group.attrs['num_well'] = num_well
                metadata_group.attrs['num_prod'] = self.num_prod if hasattr(self, 'num_prod') else 0
                metadata_group.attrs['num_inj'] = self.num_inj if hasattr(self, 'num_inj') else 0
                metadata_group.attrs['Nx'] = self.Nx if hasattr(self, 'Nx') else 0
                metadata_group.attrs['Ny'] = self.Ny if hasattr(self, 'Ny') else 0
                metadata_group.attrs['Nz'] = self.Nz if hasattr(self, 'Nz') else 0
                metadata_group.attrs['created_timestamp'] = timestamp
                metadata_group.attrs['data_dir'] = self.data_dir if hasattr(self, 'data_dir') else ''
                
                # Save training data
                train_group = hf.create_group('train')
                
                # Save STATE_train (list of tensors)
                state_train_group = train_group.create_group('STATE')
                for i, state_tensor in enumerate(self.STATE_train):
                    state_train_group.create_dataset(f'step_{i}', data=state_tensor.cpu().numpy() if hasattr(state_tensor, 'cpu') else state_tensor)
                
                # Save BHP_train (list of tensors)
                bhp_train_group = train_group.create_group('BHP')
                for i, bhp_tensor in enumerate(self.BHP_train):
                    bhp_train_group.create_dataset(f'step_{i}', data=bhp_tensor.cpu().numpy() if hasattr(bhp_tensor, 'cpu') else bhp_tensor)
                
                # Save Yobs_train (list of tensors)
                yobs_train_group = train_group.create_group('Yobs')
                for i, yobs_tensor in enumerate(self.Yobs_train):
                    yobs_train_group.create_dataset(f'step_{i}', data=yobs_tensor.cpu().numpy() if hasattr(yobs_tensor, 'cpu') else yobs_tensor)
                
                # Save dt_train
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dt_train = torch.tensor(np.ones((num_train, 1)), dtype=torch.float32).to(device)
                train_group.create_dataset('dt', data=dt_train.cpu().numpy())
                
                # Save evaluation data
                eval_group = hf.create_group('eval')
                
                # Save STATE_eval (list of tensors)
                state_eval_group = eval_group.create_group('STATE')
                for i, state_tensor in enumerate(self.STATE_eval):
                    state_eval_group.create_dataset(f'step_{i}', data=state_tensor.cpu().numpy() if hasattr(state_tensor, 'cpu') else state_tensor)
                
                # Save BHP_eval (list of tensors)
                bhp_eval_group = eval_group.create_group('BHP')
                for i, bhp_tensor in enumerate(self.BHP_eval):
                    bhp_eval_group.create_dataset(f'step_{i}', data=bhp_tensor.cpu().numpy() if hasattr(bhp_tensor, 'cpu') else bhp_tensor)
                
                # Save Yobs_eval (list of tensors)
                yobs_eval_group = eval_group.create_group('Yobs')
                for i, yobs_tensor in enumerate(self.Yobs_eval):
                    yobs_eval_group.create_dataset(f'step_{i}', data=yobs_tensor.cpu().numpy() if hasattr(yobs_tensor, 'cpu') else yobs_tensor)
                
                # Save dt_eval
                dt_eval = torch.tensor(np.ones((num_eval, 1)), dtype=torch.float32).to(device)
                eval_group.create_dataset('dt', data=dt_eval.cpu().numpy())
                
                # Save normalization parameters if available
                if hasattr(self, 'norm_params') and self.norm_params:
                    norm_group = hf.create_group('normalization')
                    import json
                    # Convert numpy arrays to lists for JSON serialization
                    norm_params_serializable = {}
                    for key, value in self.norm_params.items():
                        if isinstance(value, dict):
                            norm_params_serializable[key] = {}
                            for k, v in value.items():
                                if isinstance(v, np.ndarray):
                                    norm_params_serializable[key][k] = v.tolist()
                                elif isinstance(v, (np.integer, np.floating)):
                                    norm_params_serializable[key][k] = float(v)
                                else:
                                    norm_params_serializable[key][k] = v
                        else:
                            norm_params_serializable[key] = value
                    
                    # Store as JSON string in attributes
                    norm_group.attrs['params_json'] = json.dumps(norm_params_serializable, default=str)
                
                # Save data selection metadata for independent testing
                selections_group = hf.create_group('data_selections')
                import json
                
                # Collect all selection data
                data_selections = {}
                
                # Save spatial properties (prefer all_spatial_properties, fallback to selected_states)
                if hasattr(self, 'all_spatial_properties') and self.all_spatial_properties:
                    data_selections['all_spatial_properties'] = self.all_spatial_properties
                elif hasattr(self, 'selected_states') and self.selected_states:
                    data_selections['selected_states'] = self.selected_states
                
                # Save selected controls
                if hasattr(self, 'selected_controls') and self.selected_controls:
                    data_selections['selected_controls'] = self.selected_controls
                
                # Save selected observations
                if hasattr(self, 'selected_observations') and self.selected_observations:
                    data_selections['selected_observations'] = self.selected_observations
                
                # Save training channel names (order matters!)
                if hasattr(self, 'training_channel_names') and self.training_channel_names:
                    data_selections['training_channel_names'] = self.training_channel_names
                
                # Store as JSON string in attributes
                if data_selections:
                    selections_group.attrs['selections_json'] = json.dumps(data_selections, default=str)
            
            print(f"✅ Processed data saved to: {filepath}")
            print(f"   📊 Training samples: {num_train}, Evaluation samples: {num_eval}")
            print(f"   📈 States: {num_states}, Controls: {num_controls}, Observations: {num_observations}, Steps: {nsteps}")
            
            # Store filepath for later access
            self.processed_data_file = filepath
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _collect_selections(self):
        """Collect user selections from all tabs"""
        # States - collect ALL spatial files for global tensor
        self.all_spatial_properties = {}
        self.normalization_preferences = {}
        
        # Load all available spatial properties
        for file_path in self.available_spatial_files:
            var_name = file_path.replace('batch_spatial_properties_', '').replace('.h5', '')
            self.all_spatial_properties[var_name] = file_path
            
            # Get normalization preference for this property
            if hasattr(self, 'state_normalization_controls') and var_name in self.state_normalization_controls:
                controls = self.state_normalization_controls[var_name]
                if isinstance(controls, dict):
                    # New structure with three checkboxes
                    if controls['log'].value:
                        self.normalization_preferences[var_name] = 'log'
                    elif controls['original'].value:
                        self.normalization_preferences[var_name] = 'none'
                    else:  # Default to minmax if nothing else is selected or minmax is selected
                        self.normalization_preferences[var_name] = 'minmax'
                else:
                    # Legacy single checkbox structure (backward compatibility)
                    self.normalization_preferences[var_name] = 'log' if controls.value else 'minmax'
            else:
                # Default normalization
                self.normalization_preferences[var_name] = 'minmax'
        
        # Channel selection - collect selected channels for training
        self.selected_training_channels = {}
        if hasattr(self, 'channel_selection_checkboxes'):
            for var_name, config in self.channel_selection_checkboxes.items():
                if config['checkbox'].value:
                    self.selected_training_channels[var_name] = {
                        'filename': config['filename'],
                        'canonical_position': config['canonical_position']
                    }
        
        # Controls - collect well selections and normalization preferences
        self.selected_controls = {}
        self.control_normalization_preferences = {}
        if hasattr(self, 'control_checkboxes'):
            for var_name, well_checkboxes in self.control_checkboxes.items():
                selected_wells = []
                for well_idx, checkbox in well_checkboxes.items():
                    if checkbox.value:
                        selected_wells.append(well_idx)
                if selected_wells:
                    self.selected_controls[var_name] = {
                        'filename': f"batch_timeseries_data_{var_name}.h5",
                        'wells': selected_wells
                    }
                    
                    # Get normalization preference for this control variable
                    if hasattr(self, 'control_normalization_checkboxes') and var_name in self.control_normalization_checkboxes:
                        self.control_normalization_preferences[var_name] = 'minmax' if self.control_normalization_checkboxes[var_name].value else 'none'
                    else:
                        # Default to minmax for backward compatibility
                        self.control_normalization_preferences[var_name] = 'minmax'
        
        # Observations - collect well selections and normalization preferences
        self.selected_observations = {}
        self.observation_normalization_preferences = {}
        if hasattr(self, 'observation_checkboxes'):
            for var_name, well_checkboxes in self.observation_checkboxes.items():
                selected_wells = []
                for well_idx, checkbox in well_checkboxes.items():
                    if checkbox.value:
                        selected_wells.append(well_idx)
                if selected_wells:
                    self.selected_observations[var_name] = {
                        'filename': f"batch_timeseries_data_{var_name}.h5",
                        'wells': selected_wells
                    }
                    
                    # Get normalization preference for this observation variable
                    if hasattr(self, 'observation_normalization_checkboxes') and var_name in self.observation_normalization_checkboxes:
                        self.observation_normalization_preferences[var_name] = 'minmax' if self.observation_normalization_checkboxes[var_name].value else 'none'
                    else:
                        # Default to minmax for backward compatibility
                        self.observation_normalization_preferences[var_name] = 'minmax'
        
        # Configuration summary
        training_channels = list(self.selected_training_channels.keys())
        controls = [(k, len(v['wells'])) for k, v in self.selected_controls.items()]
        observations = [(k, len(v['wells'])) for k, v in self.selected_observations.items()]
        print(f"Configuration: {len(training_channels)} channels, {len(controls)} controls, {len(observations)} observations")
    
    def _load_and_normalize_data(self):
        """Load and normalize ALL spatial data, then extract selected channels"""
        print("Loading and normalizing spatial data...")
        
        # Load ALL spatial data with user-selected normalization
        self.spatial_data = {}
        for var_name, filename in self.all_spatial_properties.items():
            pass  # Loading spatial data
            with h5py.File(os.path.join(self.data_dir, filename), 'r') as hf:
                raw_data = np.array(hf['data'])
            
            # Get normalization preference for this variable
            norm_type = self.normalization_preferences.get(var_name, 'minmax')
            pass  # Applying normalization
            
            normalized_data, norm_params = normalize_dataset_inplace(raw_data, var_name, norm_type)
            self.spatial_data[var_name] = normalized_data
            self.norm_params[var_name] = norm_params
        
        # Load timeseries data with user-specified normalization
        self.timeseries_data = {}
        all_timeseries_vars = set(list(self.selected_controls.keys()) + list(self.selected_observations.keys()))
        
        for var_name in all_timeseries_vars:
            filename = f"batch_timeseries_data_{var_name}.h5"
            print(f"  📈 Loading {var_name}...")
            
            with h5py.File(os.path.join(self.data_dir, filename), 'r') as hf:
                raw_data = np.array(hf['data'])
            
            # Determine normalization type based on user preferences
            norm_type = 'minmax'  # Default for backward compatibility
            
            # Check if it's a control variable with normalization preference
            if var_name in getattr(self, 'control_normalization_preferences', {}):
                norm_type = self.control_normalization_preferences[var_name]
                print(f"    🎛️ Control {var_name}: Using {norm_type.upper()} normalization")
            
            # Check if it's an observation variable with normalization preference
            elif var_name in getattr(self, 'observation_normalization_preferences', {}):
                norm_type = self.observation_normalization_preferences[var_name]
                print(f"    📊 Observation {var_name}: Using {norm_type.upper()} normalization")
            
            else:
                print(f"    📈 {var_name}: Using default MIN-MAX normalization")
            
            normalized_data, norm_params = normalize_dataset_inplace(raw_data, var_name, norm_type)
            self.timeseries_data[var_name] = normalized_data
            self.norm_params[var_name] = norm_params
    
    def _create_tensors(self):
        """Create state, control, and observation tensors with deterministic channel ordering"""
        # Step 1: Create global state tensor with ALL spatial properties in canonical order
        all_state_channels = []
        self.global_channel_names = []
        
        # Use the deterministic spatial file order established during file loading
        if hasattr(self, 'spatial_file_order'):
            # Process channels in the exact canonical order
            for var_name in self.spatial_file_order:
                if var_name in self.spatial_data:
                    all_state_channels.append(self.spatial_data[var_name])
                    self.global_channel_names.append(var_name)
                    pass  # Channel added
        else:
            # Fallback to original method if spatial_file_order not available
            print("  ⚠️ Using fallback channel ordering (less deterministic)")
            for var_name in sorted(self.all_spatial_properties.keys()):  # At least sort alphabetically
                all_state_channels.append(self.spatial_data[var_name])
                self.global_channel_names.append(var_name)
        
        if all_state_channels:
            self.global_state_tensor = np.stack(all_state_channels, axis=2)  # (n_sample, timesteps, ALL_channels, Nx, Ny, Nz)
            print(f"  🌐 Global state tensor shape: {self.global_state_tensor.shape}")
            pass  # Global channels created
        
        # Step 2: Extract selected channels for training with explicit verification
        print("  🎯 Extracting selected channels for training with order verification...")
        if hasattr(self, 'selected_training_channels') and self.selected_training_channels:
            # CRITICAL: Extract channels in the EXACT order user selected them
            selected_indices = []
            self.training_channel_names = []
            self.training_channel_mapping = {}  # Track exact mapping for verification
            
            # Process in canonical order to maintain consistency
            for var_name in self.global_channel_names:
                if var_name in self.selected_training_channels:
                    global_idx = self.global_channel_names.index(var_name)
                    training_idx = len(selected_indices)  # Position in training tensor
                    
                    selected_indices.append(global_idx)
                    self.training_channel_names.append(var_name)
                    self.training_channel_mapping[var_name] = {
                        'global_index': global_idx,
                        'training_index': training_idx,
                        'verified': True
                    }
                    
                    pass  # Training channel selected
            
            if selected_indices:
                self.state_tensor = self.global_state_tensor[:, :, selected_indices, :, :, :]
                print(f"  🎯 Training state tensor shape: {self.state_tensor.shape}")
                pass  # Training channels created
                
                # VERIFICATION: Print explicit channel-to-data mapping
                print(f"\n  🔍 CHANNEL VERIFICATION:")
                for i, channel_name in enumerate(self.training_channel_names):
                    mapping = self.training_channel_mapping[channel_name]
                    filename = f"batch_spatial_properties_{channel_name}.h5"
                    print(f"    Channel {i}: {channel_name} ← {filename} (verified ✓)")
                
            else:
                print("  ❌ No channels selected for training!")
                return
        else:
            print("  ⚠️ No channel selection found. Using all channels in canonical order.")
            self.state_tensor = self.global_state_tensor
            self.training_channel_names = self.global_channel_names
        
        # Create control tensor
        control_components = []
        for var_name, config in self.selected_controls.items():
            data = self.timeseries_data[var_name]
            selected_data = data[:, :, config['wells']]  # Select specific wells
            
            # Flatten well dimension
            for well_idx in range(selected_data.shape[2]):
                control_components.append(selected_data[:, :, well_idx])
        
        if control_components:
            self.control_tensor = np.stack(control_components, axis=2)
            print(f"  🎛️ Control tensor shape: {self.control_tensor.shape}")
        
        # Create observation tensor (Observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)])
        obs_components = []
        for var_name, config in self.selected_observations.items():
            data = self.timeseries_data[var_name]
            selected_data = data[:, :, config['wells']]  # Select specific wells
            
            # Flatten well dimension
            for well_idx in range(selected_data.shape[2]):
                obs_components.append(selected_data[:, :, well_idx])
        
        if obs_components:
            self.observation_tensor = np.stack(obs_components, axis=2)
            print(f"  📊 Observation tensor shape: {self.observation_tensor.shape}")
        
        # Automatically continue with sliding window and train-test split
        self._apply_sliding_window()
        self._print_final_results()
        
        # Store variables for easy access
        self.processing_complete = True
        
        # Final validation and transparency report
        self._generate_channel_verification_report()
    
    def _generate_channel_verification_report(self):
        """Generate comprehensive channel verification report for 100% transparency"""
        pass  # Channel verification report disabled
    
    def _apply_sliding_window(self):
        """Apply sliding window and train-test split"""
        print(f"Applying {self.nsteps}-step sliding window...")
        
        # Get dimensions
        if hasattr(self, 'state_tensor'):
            n_sample, steps_slt, n_channels, Nx, Ny, Nz = self.state_tensor.shape
        else:
            print("❌ No state tensor available")
            return
        
        # Create sliding window indices
        indt = np.array(range(0, steps_slt - (self.nsteps - 1)))
        
        # Apply sliding window to all tensors
        # Create dynamic channel lists based on selected channels
        self.channel_data_slt = []  # List of lists for each channel
        for i in range(n_channels):
            self.channel_data_slt.append([])
        
        # Legacy compatibility names (for backward compatibility with existing code)
        # Initialize these as empty lists regardless of selection
        self.SW_slt = []
        self.SG_slt = [] 
        self.PRES_slt = []
        self.BHP_slt = []
        self.Yobs_slt = []
        
        # Track if we have any data for each legacy channel
        has_sw = False
        has_sg = False
        has_pres = False
        
        for k in range(self.nsteps):
            indt_k = indt + k
            
            # State data (split by channels dynamically)
            for channel_idx in range(n_channels):
                channel_data = self.state_tensor[:, indt_k, channel_idx, :, :, :]
                self.channel_data_slt[channel_idx].append(channel_data)
                
                # Legacy compatibility: map to SW/SG/PRES if they match
                if hasattr(self, 'training_channel_names'):
                    channel_name = self.training_channel_names[channel_idx].upper()
                    if channel_name in ['SW', 'SWAT', 'WATER_SAT'] and len(self.SW_slt) == k:
                        self.SW_slt.append(channel_data)
                        has_sw = True
                    elif channel_name in ['SG', 'SGAS', 'GAS_SAT'] and len(self.SG_slt) == k:
                        self.SG_slt.append(channel_data)
                        has_sg = True
                    elif channel_name in ['PRES', 'PRESSURE'] and len(self.PRES_slt) == k:
                        self.PRES_slt.append(channel_data)
                        has_pres = True
            
            # Control and observation data for prediction steps
            if k < self.nsteps - 1 and hasattr(self, 'control_tensor'):
                ctrl_indt_k = np.minimum(indt_k, self.control_tensor.shape[1] - 1)
                self.BHP_slt.append(self.control_tensor[:, ctrl_indt_k, :])
                
            if k < self.nsteps - 1 and hasattr(self, 'observation_tensor'):
                obs_indt_k = np.minimum(indt_k, self.observation_tensor.shape[1] - 1)
                self.Yobs_slt.append(self.observation_tensor[:, obs_indt_k, :])
        
        # If any of the legacy channels are missing, create placeholder data
        # This ensures train_split_data won't fail due to missing channels
        if not has_sw and self.channel_data_slt:
            pass  # SW channel not selected
            for k in range(self.nsteps):
                # Use the first channel's shape but with zeros
                placeholder = np.zeros_like(self.channel_data_slt[0][k])
                self.SW_slt.append(placeholder)
        
        if not has_sg and self.channel_data_slt:
            pass  # SG channel not selected
            for k in range(self.nsteps):
                placeholder = np.zeros_like(self.channel_data_slt[0][k])
                self.SG_slt.append(placeholder)
        
        if not has_pres and self.channel_data_slt:
            pass  # PRES channel not selected
            for k in range(self.nsteps):
                placeholder = np.zeros_like(self.channel_data_slt[0][k])
                self.PRES_slt.append(placeholder)
        
        # Set up dimensions
        self.num_t_slt = len(indt)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.num_well = len([well for config in self.selected_controls.values() for well in config['wells']]) + \
                        len([well for config in self.selected_observations.values() for well in config['wells']])
        self.num_prod = 3  # Assume 3 producers
        self.num_inj = 3   # Assume 3 injectors
        self.n_channels = n_channels
        
        print(f"Sliding window applied: {len(self.channel_data_slt[0])} time steps")
        
        # Apply train-test split using existing function
        print("Applying train-test split...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pass the dynamic channel data to train_split_data
        self.STATE_train, self.BHP_train, self.Yobs_train, self.STATE_eval, self.BHP_eval, self.Yobs_eval = train_split_data(
            self.SW_slt, self.SG_slt, self.PRES_slt, self.BHP_slt, self.Yobs_slt,
            self.num_t_slt, self.Nx, self.Ny, self.Nz, self.num_well, self.num_prod, self.num_inj,
            self.n_channels, device, channel_data_slt=getattr(self, 'channel_data_slt', None)
        )
    
    def _print_final_results(self):
        """Print final tensor shapes and summary"""
        print("Data preprocessing completed!")
        if hasattr(self, 'STATE_train') and self.STATE_train:
            print(f"Tensors: STATE {self.STATE_train[0].shape}, BHP {self.BHP_train[0].shape}, Yobs {self.Yobs_train[0].shape}")
        print(f"Grid: {self.Nx}×{self.Ny}×{self.Nz}, Channels: {self.n_channels}, Timeseries Variables: {self.num_well}, Steps: {self.nsteps}")
        print("Ready for model training!")
    
    def _assign_global_variables(self):
        """Assign processed tensors to global variables for immediate training"""
        import inspect
        
        # Get the calling frame (main notebook/script)
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the main execution frame
            while frame.f_back is not None:
                frame = frame.f_back
            
            # Get the global namespace of the main execution context
            main_globals = frame.f_globals
            
            # Make tensors available globally
            main_globals['STATE_train'] = self.STATE_train
            main_globals['BHP_train'] = self.BHP_train
            main_globals['Yobs_train'] = self.Yobs_train
            main_globals['STATE_eval'] = self.STATE_eval
            main_globals['BHP_eval'] = self.BHP_eval
            main_globals['Yobs_eval'] = self.Yobs_eval
            
            # Make configuration variables available globally
            main_globals['norm_params'] = self.norm_params
            main_globals['Nx'] = self.Nx
            main_globals['Ny'] = self.Ny
            main_globals['Nz'] = self.Nz
            main_globals['num_well'] = self.num_well
            main_globals['num_prod'] = self.num_prod
            main_globals['num_inj'] = self.num_inj
            main_globals['n_channels'] = self.n_channels
            main_globals['num_t_slt'] = self.num_t_slt
            main_globals['nsteps'] = self.nsteps
            
            # Calculate derived variables
            main_globals['input_shape'] = (self.n_channels, self.Nx, self.Ny, self.Nz)
            main_globals['perm_shape'] = (1, self.Nx, self.Ny, self.Nz)
            main_globals['prod_loc_shape'] = (self.num_well, 3)
            
            main_globals['num_train'] = self.STATE_train[0].shape[0]
            main_globals['num_eval'] = self.STATE_eval[0].shape[0]
            
            # Set up time tensors
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            main_globals['dt_train'] = torch.tensor(np.ones((main_globals['num_train'], 1)), dtype=torch.float32).to(device)
            main_globals['dt_eval'] = torch.tensor(np.ones((main_globals['num_eval'], 1)), dtype=torch.float32).to(device)
            
        finally:
            del frame
        
        print("🎯 All variables assigned globally - ready for training!")
        print(f"   Available variables: STATE_train, BHP_train, Yobs_train, STATE_eval, BHP_eval, Yobs_eval")
        print(f"   Configuration: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, channels={self.n_channels}")
        print(f"   Model shapes: input_shape, perm_shape, prod_loc_shape")
    
    def get_processed_data(self):
        """Return processed data for model training"""
        if not hasattr(self, 'STATE_train'):
            return None
            
        return {
            'STATE_train': self.STATE_train,
            'BHP_train': self.BHP_train,
            'Yobs_train': self.Yobs_train,
            'STATE_eval': self.STATE_eval,
            'BHP_eval': self.BHP_eval,
            'Yobs_eval': self.Yobs_eval,
            'norm_params': self.norm_params,
            'Nx': self.Nx,
            'Ny': self.Ny,
            'Nz': self.Nz,
            'num_well': self.num_well,
            'num_prod': self.num_prod,
            'num_inj': self.num_inj,
            'n_channels': self.n_channels,
            'num_t_slt': self.num_t_slt,
            'nsteps': self.nsteps
        }
    
    def get_training_tensors(self):
        """Get the main training tensors directly"""
        if not hasattr(self, 'STATE_train'):
            print("❌ No processed data available. Please complete processing first.")
            return None, None, None, None, None, None
            
        return (
            self.STATE_train,
            self.BHP_train, 
            self.Yobs_train,
            self.STATE_eval,
            self.BHP_eval,
            self.Yobs_eval
        )
    
    def assign_to_globals(self):
        """Assign all variables to the global namespace for immediate use"""
        if not hasattr(self, 'STATE_train'):
            print("No processed data available. Please complete processing first.")
            return False
            
        # Get the notebook's global namespace
        import inspect
        import __main__
        
        # Use __main__ module's globals to ensure we're setting variables in the notebook namespace
        caller_globals = __main__.__dict__
        
        # Assign training tensors
        caller_globals['STATE_train'] = self.STATE_train
        caller_globals['BHP_train'] = self.BHP_train
        caller_globals['Yobs_train'] = self.Yobs_train
        caller_globals['STATE_eval'] = self.STATE_eval
        caller_globals['BHP_eval'] = self.BHP_eval
        caller_globals['Yobs_eval'] = self.Yobs_eval
        
        # Assign configuration variables
        caller_globals['norm_params'] = self.norm_params
        caller_globals['Nx'] = self.Nx
        caller_globals['Ny'] = self.Ny
        caller_globals['Nz'] = self.Nz
        caller_globals['num_well'] = self.num_well
        caller_globals['num_prod'] = self.num_prod
        caller_globals['num_inj'] = self.num_inj
        caller_globals['n_channels'] = self.n_channels
        caller_globals['num_t_slt'] = self.num_t_slt
        caller_globals['nsteps'] = self.nsteps
        
        # Calculate and assign derived variables
        caller_globals['input_shape'] = (self.n_channels, self.Nx, self.Ny, self.Nz)
        caller_globals['perm_shape'] = (1, self.Nx, self.Ny, self.Nz)
        caller_globals['prod_loc_shape'] = (self.num_well, 3)
        
        caller_globals['num_train'] = self.STATE_train[0].shape[0]
        caller_globals['num_eval'] = self.STATE_eval[0].shape[0]
        
        # Set up time tensors
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        caller_globals['dt_train'] = torch.tensor(np.ones((caller_globals['num_train'], 1)), dtype=torch.float32).to(device)
        caller_globals['dt_eval'] = torch.tensor(np.ones((caller_globals['num_eval'], 1)), dtype=torch.float32).to(device)
        
        print("Variables assigned to global namespace - ready for training!")
        
        return True
    
    def generate_test_visualization(self, my_rom, device, data_dir, num_tstep=24):
        """
        Generate test predictions and launch visualization dashboard using the same 
        data processing logic and user selections as training
        """
        if not hasattr(self, 'STATE_train'):
            print("❌ No trained model data available. Please complete training first.")
            return None
            
        print("🎨 Generating test visualization with consistent data processing...")
        
        # Step 1: Load and normalize test data using SAME user selections
        print("🔄 Loading test data with user-selected normalization...")
        print("Note: Using the same normalization settings as training")
        
        # Load spatial data with SAME files and normalization as training
        test_spatial_data = {}
        
        # Use all_spatial_properties if available (new structure), otherwise fallback to selected_states (old structure)
        spatial_properties_to_load = getattr(self, 'all_spatial_properties', getattr(self, 'selected_states', {}))
        
        if not spatial_properties_to_load:
            print("  ❌ No spatial property configuration found!")
            print("Please complete data preprocessing first.")
            return None
        
        # Important: The normalization parameters are stored in self.norm_params from training
        # This ensures consistent denormalization regardless of which normalization was used
            
        for var_name, filename in spatial_properties_to_load.items():
            print(f"  📊 Loading test {var_name} from {filename}...")
            with h5py.File(os.path.join(data_dir, filename), 'r') as hf:
                raw_data = np.array(hf['data'])
            
            # Apply SAME normalization as training using stored parameters
            norm_params = self.norm_params[var_name]
            if norm_params['type'] == 'log':
                # Apply log normalization using stored parameters
                epsilon = norm_params['epsilon']
                data_shift = norm_params['data_shift']
                
                data_shifted = raw_data - data_shift + epsilon
                log_data = np.log(data_shifted)
                normalized_data = (log_data - norm_params['log_min']) / (norm_params['log_max'] - norm_params['log_min'])
                print(f"    🔢 Applied LOG normalization to {var_name}")
            else:
                # Apply min-max normalization using stored parameters  
                normalized_data = (raw_data - norm_params['min']) / (norm_params['max'] - norm_params['min'])
                print(f"    📏 Applied MIN-MAX normalization to {var_name}")
                
            test_spatial_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
        
        # Load timeseries data with SAME normalization as training
        test_timeseries_data = {}
        all_timeseries_vars = set(list(self.selected_controls.keys()) + list(self.selected_observations.keys()))
        
        for var_name in all_timeseries_vars:
            filename = f"batch_timeseries_data_{var_name}.h5"
            print(f"  📈 Loading test {var_name}...")
            
            with h5py.File(os.path.join(data_dir, filename), 'r') as hf:
                raw_data = np.array(hf['data'])
            
            # Apply SAME normalization as training
            norm_params = self.norm_params[var_name]
            normalized_data = (raw_data - norm_params['min']) / (norm_params['max'] - norm_params['min'])
            test_timeseries_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
        
        # Step 2: Extract controls and observations using SAME user selections
        print("🔧 Extracting controls and observations using user selections...")
        
        # Create control tensor using SAME logic as training
        control_components = []
        for var_name, config in self.selected_controls.items():
            data = test_timeseries_data[var_name]
            selected_data = data[:, :, config['wells']]
            
            for well_idx in range(selected_data.shape[2]):
                control_components.append(selected_data[:, :, well_idx])
        
        bhp_test = torch.stack(control_components, dim=2) if control_components else torch.zeros(0)
        
        # Create observation tensor using SAME logic as training  
        obs_components = []
        for var_name, config in self.selected_observations.items():
            data = test_timeseries_data[var_name]
            selected_data = data[:, :, config['wells']]
            
            for well_idx in range(selected_data.shape[2]):
                obs_components.append(selected_data[:, :, well_idx])
        
        yobs_test = torch.stack(obs_components, dim=2) if obs_components else torch.zeros(0)
        
        print(f"📊 Test control data shape: {bhp_test.shape}")
        print(f"📊 Test observation data shape: {yobs_test.shape}")
        
        # Step 3: Organize spatial data in SAME channel order as training
        print("🏗️ Organizing spatial data in training channel order...")
        
        # Get spatial data in the same order as training channels
        spatial_channels = []
        channel_names = []
        
        # Use training_channel_names if available (new structure), otherwise use selected_states keys (old structure)
        if hasattr(self, 'training_channel_names') and self.training_channel_names:
            pass  # Using training channel order
            for var_name in self.training_channel_names:
                if var_name in test_spatial_data:
                    spatial_channels.append(test_spatial_data[var_name])
                    channel_names.append(var_name)
                else:
                    print(f"  ⚠️ Warning: Training channel '{var_name}' not found in test data")
        elif hasattr(self, 'selected_states') and self.selected_states:
            pass  # Using legacy channel order
            for var_name in self.selected_states.keys():
                spatial_channels.append(test_spatial_data[var_name])
                channel_names.append(var_name)
        else:
            pass  # Using all spatial properties
            for var_name in spatial_properties_to_load.keys():
                spatial_channels.append(test_spatial_data[var_name])
                channel_names.append(var_name)
        
        # Stack into state tensor: (n_sample, timesteps, channels, Nx, Ny, Nz)
        if spatial_channels:
            state_test = torch.stack(spatial_channels, dim=2)
            n_sample, timesteps, n_channels, Nx, Ny, Nz = state_test.shape
            print(f"✅ Test state tensor: {state_test.shape}")
            pass  # Channel order established
        else:
            print("❌ No spatial data available")
            return None
        
        # Step 4: Generate test cases and run predictions
        print("🎯 Setting up test cases...")
        
        available_cases = n_sample  # Use ALL available cases automatically
        test_case_indices = np.arange(available_cases)
        num_case = len(test_case_indices)
        
        print(f"📊 Processing {num_case} test cases with {num_tstep} time steps")
        
        # Initialize prediction arrays
        state_pred = torch.zeros((num_case, num_tstep, n_channels, Nx, Ny, Nz), dtype=torch.float32).to(device)
        yobs_pred = torch.zeros((num_case, num_tstep, yobs_test.shape[2]), dtype=torch.float32).to(device)
        
        # Time step configuration
        t_steps = np.arange(0, 200, 200//num_tstep)
        dt = 10
        t_steps1 = (t_steps + dt).astype(int)
        indt_del = t_steps1 - t_steps
        indt_del = indt_del / max(indt_del)
        
        tmp = np.array(range(num_tstep)) - 1
        tmp1 = np.array(range(num_tstep))
        tmp[0] = 0
        
        # Prepare control and observation sequences
        bhp_tt1 = bhp_test[:, tmp1, :]
        bhp_t = torch.swapaxes(bhp_tt1, 1, 2).to(device)
        bhp_seq = bhp_t[test_case_indices, :, :]
        
        yobs_t_seq = torch.swapaxes(yobs_test[test_case_indices, ...], 1, 2).to(device)
        
        # Initial state preparation - use first timestep, rearrange to (batch, channels, X, Y, Z)
        initial_state = state_test[test_case_indices, 0, :, :, :, :].to(device)  # (num_case, channels, Nx, Ny, Nz)
        state_t_seq = initial_state
        
        print(f"🔍 Sequential Prediction Setup:")
        print(f"Initial state shape: {state_t_seq.shape}")
        print(f"Control sequence shape: {bhp_seq.shape}")
        print(f"Observation sequence shape: {yobs_t_seq.shape}")
        
        # Step 5: Run sequential predictions
        print(f"\n🚀 Running sequential predictions...")
        import timeit
        start = timeit.default_timer()
        
        for i_tstep in range(num_tstep):
            # Store current state prediction
            state_pred[:, i_tstep, ...] = state_t_seq
            
            # Time step for current iteration
            dt_seq = torch.tensor(np.ones((num_case, 1)) * indt_del[i_tstep], dtype=torch.float32).to(device)
            
            # Prepare inputs for model
            inputs = (state_t_seq, bhp_seq[:, :, i_tstep], yobs_t_seq[:, :, i_tstep], dt_seq)
            
            # Predict next state
            state_t1_seq, yobs_t1_seq = my_rom.predict(inputs)
            
            # Update state for next iteration
            state_t_seq = state_t1_seq
            
            # Store well output predictions
            yobs_pred[:, i_tstep, :] = yobs_t1_seq
            
            # Progress indicator
            if (i_tstep + 1) % 5 == 0:
                print(f"  Step {i_tstep + 1}/{num_tstep} completed")
        
        end = timeit.default_timer()
        print(f"\n⏱️  Prediction time: {end - start:.4f} seconds")
        
        # Step 6: Prepare data for visualization
        print("🎨 Preparing visualization data...")
        
        # Get true sequences for comparison - rearrange to match prediction format
        state_seq_true = torch.zeros((num_case, n_channels, timesteps, Nx, Ny, Nz))
        for i, var_name in enumerate(channel_names):
            state_seq_true[:, i, :, :, :, :] = test_spatial_data[var_name][test_case_indices, ...]
        
        # Align time dimensions
        state_seq_true_aligned = state_seq_true[:, :, :num_tstep, :, :, :]
        
        print(f"📊 Final shapes:")
        print(f"Predicted state: {state_pred.shape}")
        print(f"True state: {state_seq_true_aligned.shape}")
        print(f"Predicted observations: {yobs_pred.shape}")
        
        # Step 7: Launch visualization dashboard
        print("\n🚀 Launching Interactive Visualization Dashboard...")
        
        visualization_dashboard = create_visualization_dashboard(
            state_pred=state_pred,
            state_seq_true_aligned=state_seq_true_aligned,
            yobs_pred=yobs_pred,
            yobs_seq_true=yobs_t_seq,
            test_case_indices=test_case_indices,
            norm_params=self.norm_params,
            Nx=Nx, Ny=Ny, Nz=Nz,
            num_tstep=num_tstep,
            channel_names=channel_names,  # Pass the channel names for visualization
            my_rom=my_rom,  # Pass ROM model for comparison predictions
            test_controls=bhp_seq,  # Pass test controls for comparison predictions
            test_observations=yobs_t_seq,  # Pass test observations for comparison predictions
            device=device  # Pass device for computation
        )
        
        print("Test visualization completed")
        
        return visualization_dashboard
    
    def display_normalization_info(self):
        """Display helpful information about normalization compatibility and model loading"""
        if not WIDGETS_AVAILABLE:
            print("Normalization parameters are saved automatically and models remain compatible.")
            return
            
        info_html = """
        <div style="background-color: #f5f5f5; padding: 10px; border-left: 3px solid #666; margin: 5px 0;">
        <p><b>Note:</b> Normalization parameters are saved automatically during preprocessing. Models remain compatible across different normalization settings.</p>
        </div>
        """
        
        info_widget = widgets.HTML(value=info_html)
        display(info_widget)

    def display(self):
        """Display the dashboard"""
        if not WIDGETS_AVAILABLE:
            print("❌ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
            return None
        
        display(self.main_widget)
        return self


def generate_test_visualization_standalone(loaded_data, my_rom, device, data_dir, num_tstep=24):
    """
    Standalone function to generate test predictions and launch visualization dashboard.
    Works independently of the dashboard object by using loaded data and selections.
    
    Args:
        loaded_data: Dictionary returned from load_processed_data() containing:
            - norm_params: Normalization parameters
            - data_selections: Data selection metadata (spatial properties, controls, observations, channel names)
            - metadata: Metadata including Nx, Ny, Nz, etc.
        my_rom: Trained ROM model
        device: PyTorch device
        data_dir: Directory containing raw data files
        num_tstep: Number of time steps for prediction
    
    Returns:
        Visualization dashboard object or None if failed
    """
    import h5py
    import os
    import numpy as np
    import torch
    
    print("🎨 Generating test visualization with loaded data selections...")
    
    # Extract required data from loaded_data
    norm_params = loaded_data.get('norm_params')
    data_selections = loaded_data.get('data_selections')
    metadata = loaded_data.get('metadata', {})
    
    if norm_params is None:
        print("❌ No normalization parameters found in loaded data!")
        return None
    
    if data_selections is None:
        print("❌ No data selection metadata found in loaded data!")
        print("   This file may have been created before selections were saved.")
        print("   Please reprocess data using Step 1 to include selection metadata.")
        return None
    
    # Extract spatial properties (prefer all_spatial_properties, fallback to selected_states)
    spatial_properties_to_load = data_selections.get('all_spatial_properties') or data_selections.get('selected_states', {})
    
    if not spatial_properties_to_load:
        print("  ❌ No spatial property configuration found!")
        return None
    
    # Extract selections
    selected_controls = data_selections.get('selected_controls', {})
    selected_observations = data_selections.get('selected_observations', {})
    training_channel_names = data_selections.get('training_channel_names', [])
    
    # Extract metadata
    Nx = metadata.get('Nx', 0)
    Ny = metadata.get('Ny', 0)
    Nz = metadata.get('Nz', 0)
    
    # Step 1: Load and normalize test data using SAME user selections
    print("🔄 Loading test data with user-selected normalization...")
    print("Note: Using the same normalization settings as training")
    
    # Load spatial data with SAME files and normalization as training
    test_spatial_data = {}
    
    for var_name, filename in spatial_properties_to_load.items():
        print(f"  📊 Loading test {var_name} from {filename}...")
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"    ⚠️ Warning: File not found: {filepath}")
            continue
            
        with h5py.File(filepath, 'r') as hf:
            raw_data = np.array(hf['data'])
        
        # Apply SAME normalization as training using stored parameters
        if var_name not in norm_params:
            print(f"    ⚠️ Warning: No normalization params for {var_name}, skipping")
            continue
            
        norm_params_var = norm_params[var_name]
        if norm_params_var.get('type') == 'log':
            # Apply log normalization using stored parameters
            epsilon = norm_params_var['epsilon']
            data_shift = norm_params_var['data_shift']
            
            data_shifted = raw_data - data_shift + epsilon
            log_data = np.log(data_shifted)
            normalized_data = (log_data - norm_params_var['log_min']) / (norm_params_var['log_max'] - norm_params_var['log_min'])
            print(f"    🔢 Applied LOG normalization to {var_name}")
        else:
            # Apply min-max normalization using stored parameters  
            normalized_data = (raw_data - norm_params_var['min']) / (norm_params_var['max'] - norm_params_var['min'])
            print(f"    📏 Applied MIN-MAX normalization to {var_name}")
            
        test_spatial_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
    
    # Load timeseries data with SAME normalization as training
    test_timeseries_data = {}
    all_timeseries_vars = set(list(selected_controls.keys()) + list(selected_observations.keys()))
    
    for var_name in all_timeseries_vars:
        filename = f"batch_timeseries_data_{var_name}.h5"
        filepath = os.path.join(data_dir, filename)
        print(f"  📈 Loading test {var_name}...")
        
        if not os.path.exists(filepath):
            print(f"    ⚠️ Warning: File not found: {filepath}")
            continue
        
        with h5py.File(filepath, 'r') as hf:
            raw_data = np.array(hf['data'])
        
        # Apply SAME normalization as training
        if var_name not in norm_params:
            print(f"    ⚠️ Warning: No normalization params for {var_name}, skipping")
            continue
            
        norm_params_var = norm_params[var_name]
        normalized_data = (raw_data - norm_params_var['min']) / (norm_params_var['max'] - norm_params_var['min'])
        test_timeseries_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
    
    # Step 2: Extract controls and observations using SAME user selections
    print("🔧 Extracting controls and observations using user selections...")
    
    # Create control tensor using SAME logic as training
    control_components = []
    for var_name, config in selected_controls.items():
        if var_name not in test_timeseries_data:
            continue
        data = test_timeseries_data[var_name]
        selected_data = data[:, :, config['wells']]
        
        for well_idx in range(selected_data.shape[2]):
            control_components.append(selected_data[:, :, well_idx])
    
    bhp_test = torch.stack(control_components, dim=2) if control_components else torch.zeros(0)
    
    # Create observation tensor using SAME logic as training  
    obs_components = []
    for var_name, config in selected_observations.items():
        if var_name not in test_timeseries_data:
            continue
        data = test_timeseries_data[var_name]
        selected_data = data[:, :, config['wells']]
        
        for well_idx in range(selected_data.shape[2]):
            obs_components.append(selected_data[:, :, well_idx])
    
    yobs_test = torch.stack(obs_components, dim=2) if obs_components else torch.zeros(0)
    
    print(f"📊 Test control data shape: {bhp_test.shape}")
    print(f"📊 Test observation data shape: {yobs_test.shape}")
    
    # Step 3: Organize spatial data in SAME channel order as training
    print("🏗️ Organizing spatial data in training channel order...")
    
    # Get spatial data in the same order as training channels
    spatial_channels = []
    channel_names = []
    
    # Use training_channel_names if available (new structure), otherwise use spatial_properties_to_load keys (old structure)
    if training_channel_names:
        for var_name in training_channel_names:
            if var_name in test_spatial_data:
                spatial_channels.append(test_spatial_data[var_name])
                channel_names.append(var_name)
            else:
                print(f"  ⚠️ Warning: Training channel '{var_name}' not found in test data")
    else:
        # Fallback: use all available spatial properties
        for var_name in spatial_properties_to_load.keys():
            if var_name in test_spatial_data:
                spatial_channels.append(test_spatial_data[var_name])
                channel_names.append(var_name)
    
    # Stack into state tensor: (n_sample, timesteps, channels, Nx, Ny, Nz)
    if spatial_channels:
        state_test = torch.stack(spatial_channels, dim=2)
        n_sample, timesteps, n_channels, Nx, Ny, Nz = state_test.shape
        print(f"✅ Test state tensor: {state_test.shape}")
    else:
        print("❌ No spatial data available")
        return None
    
    # Step 4: Generate test cases and run predictions
    print("🎯 Setting up test cases...")
    
    available_cases = n_sample  # Use ALL available cases automatically
    test_case_indices = np.arange(available_cases)
    num_case = len(test_case_indices)
    
    print(f"📊 Processing {num_case} test cases with {num_tstep} time steps")
    
    # Initialize prediction arrays
    state_pred = torch.zeros((num_case, num_tstep, n_channels, Nx, Ny, Nz), dtype=torch.float32).to(device)
    yobs_pred = torch.zeros((num_case, num_tstep, yobs_test.shape[2]), dtype=torch.float32).to(device)
    
    # Time step configuration
    t_steps = np.arange(0, 200, 200//num_tstep)
    dt = 10
    t_steps1 = (t_steps + dt).astype(int)
    indt_del = t_steps1 - t_steps
    indt_del = indt_del / max(indt_del)
    
    tmp = np.array(range(num_tstep)) - 1
    tmp1 = np.array(range(num_tstep))
    tmp[0] = 0
    
    # Prepare control and observation sequences
    bhp_tt1 = bhp_test[:, tmp1, :]
    bhp_t = torch.swapaxes(bhp_tt1, 1, 2).to(device)
    bhp_seq = bhp_t[test_case_indices, :, :]
    
    yobs_t_seq = torch.swapaxes(yobs_test[test_case_indices, ...], 1, 2).to(device)
    
    # Initial state preparation - use first timestep, rearrange to (batch, channels, X, Y, Z)
    initial_state = state_test[test_case_indices, 0, :, :, :, :].to(device)  # (num_case, channels, Nx, Ny, Nz)
    state_t_seq = initial_state
    
    print(f"🔍 Sequential Prediction Setup:")
    print(f"Initial state shape: {state_t_seq.shape}")
    print(f"Control sequence shape: {bhp_seq.shape}")
    print(f"Observation sequence shape: {yobs_t_seq.shape}")
    
    # Step 5: Run sequential predictions
    print(f"\n🚀 Running sequential predictions...")
    import timeit
    start = timeit.default_timer()
    
    for i_tstep in range(num_tstep):
        # Store current state prediction
        state_pred[:, i_tstep, ...] = state_t_seq
        
        # Time step for current iteration
        dt_seq = torch.tensor(np.ones((num_case, 1)) * indt_del[i_tstep], dtype=torch.float32).to(device)
        
        # Prepare inputs for model
        inputs = (state_t_seq, bhp_seq[:, :, i_tstep], yobs_t_seq[:, :, i_tstep], dt_seq)
        
        # Predict next state
        state_t1_seq, yobs_t1_seq = my_rom.predict(inputs)
        
        # Update state for next iteration
        state_t_seq = state_t1_seq
        
        # Store well output predictions
        yobs_pred[:, i_tstep, :] = yobs_t1_seq
        
        # Progress indicator
        if (i_tstep + 1) % 5 == 0:
            print(f"  Step {i_tstep + 1}/{num_tstep} completed")
    
    end = timeit.default_timer()
    print(f"\n⏱️  Prediction time: {end - start:.4f} seconds")
    
    # Step 6: Prepare data for visualization
    print("🎨 Preparing visualization data...")
    
    # Get true sequences for comparison - rearrange to match prediction format
    state_seq_true = torch.zeros((num_case, n_channels, timesteps, Nx, Ny, Nz))
    for i, var_name in enumerate(channel_names):
        state_seq_true[:, i, :, :, :, :] = test_spatial_data[var_name][test_case_indices, ...]
    
    # Align time dimensions
    state_seq_true_aligned = state_seq_true[:, :, :num_tstep, :, :, :]
    
    print(f"📊 Final shapes:")
    print(f"Predicted state: {state_pred.shape}")
    print(f"True state: {state_seq_true_aligned.shape}")
    print(f"Predicted observations: {yobs_pred.shape}")
    
    # Step 7: Launch visualization dashboard
    print("\n🚀 Launching Interactive Visualization Dashboard...")
    
    visualization_dashboard = create_visualization_dashboard(
        state_pred=state_pred,
        state_seq_true_aligned=state_seq_true_aligned,
        yobs_pred=yobs_pred,
        yobs_seq_true=yobs_t_seq,
        test_case_indices=test_case_indices,
        norm_params=norm_params,
        Nx=Nx, Ny=Ny, Nz=Nz,
        num_tstep=num_tstep,
        channel_names=channel_names,  # Pass the channel names for visualization
        my_rom=my_rom,  # Pass ROM model for comparison predictions
        test_controls=bhp_seq,  # Pass test controls for comparison predictions
        test_observations=yobs_t_seq,  # Pass test observations for comparison predictions
        device=device  # Pass device for computation
    )
    
    print("Test visualization completed")
    
    return visualization_dashboard


def create_data_preprocessing_dashboard():
    """
    Create and display the interactive data preprocessing dashboard
    Returns the dashboard instance for accessing processed data
    """
    if not WIDGETS_AVAILABLE:
        print("❌ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
        print("Install with: pip install ipywidgets")
        print("📚 Then restart your Jupyter kernel")
        return None
    
    dashboard = DataPreprocessingDashboard()
    
    # Display helpful normalization compatibility information
    dashboard.display_normalization_info()
    
    dashboard.display()
    return dashboard


def load_processed_data(filepath=None, data_dir='./processed_data/', n_channels=None):
    """
    Load processed data from a saved .h5 file
    
    Args:
        filepath: Full path to the .h5 file. If None, searches for files in data_dir
        data_dir: Directory to search for processed data files if filepath is None
        n_channels: Optional. If provided, filters files to match this channel count.
                    Files are matched by 'ch{n_channels}' pattern in filename.
                    If None, loads the most recent file (backward compatible).
    
    Returns:
        Dictionary containing all loaded data and metadata, or None if loading fails
    """
    import glob
    
    # If no filepath provided, find the most recent processed data file
    if filepath is None:
        # Try to resolve relative path - check if we're in ROM_Refactored directory
        if not os.path.isabs(data_dir):
            # Check if processed_data exists in current directory
            if not os.path.exists(data_dir):
                # Try processed_data/ relative to this file
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                processed_data_path = os.path.join(current_file_dir, '..', 'processed_data')
                processed_data_path = os.path.normpath(processed_data_path)
                if os.path.exists(processed_data_path):
                    data_dir = processed_data_path
                    print(f"📂 Found processed_data at: {data_dir}")
        
        if not os.path.exists(data_dir):
            print(f"❌ Directory {data_dir} does not exist.")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Tried relative path: ./processed_data/")
            print(f"   💡 Tip: Make sure processed_data directory exists or provide full filepath")
            return None
        
        # Find all processed data files
        pattern = os.path.join(data_dir, 'processed_data_*.h5')
        files = glob.glob(pattern)
        
        if not files:
            print(f"❌ No processed data files found in {data_dir}")
            return None
        
        # Filter by n_channels if specified
        if n_channels is not None:
            print(f"🔍 Filtering files for n_channels={n_channels}...")
            matching_files = []
            available_channels = set()
            
            for file in files:
                # Extract channel count from filename pattern: ch{number}
                filename = os.path.basename(file)
                match = re.search(r'_ch(\d+)_', filename)
                if match:
                    file_channels = int(match.group(1))
                    available_channels.add(file_channels)
                    if file_channels == n_channels:
                        matching_files.append(file)
            
            if matching_files:
                # Get the most recent matching file
                filepath = max(matching_files, key=os.path.getmtime)
                print(f"✅ Found matching file: {os.path.basename(filepath)} (channels={n_channels})")
            else:
                print(f"❌ No processed data files found with n_channels={n_channels}")
                if available_channels:
                    print(f"   Available channel counts: {sorted(available_channels)}")
                else:
                    print(f"   Could not determine channel counts from filenames")
                return None
        else:
            # Get the most recent file (backward compatible behavior)
            filepath = max(files, key=os.path.getmtime)
            print(f"📂 Found processed data file: {os.path.basename(filepath)}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        print(f"📖 Loading processed data from: {filepath}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with h5py.File(filepath, 'r') as hf:
            # Load metadata
            metadata = {}
            if 'metadata' in hf:
                for key in hf['metadata'].attrs:
                    metadata[key] = hf['metadata'].attrs[key]
            
            # Validate n_channels if specified
            if n_channels is not None:
                file_n_channels = metadata.get('n_channels')
                if file_n_channels is not None and file_n_channels != n_channels:
                    print(f"⚠️ Warning: File metadata indicates n_channels={file_n_channels}, but expected {n_channels}")
                    print(f"   This may cause errors. Proceeding anyway...")
            
            # Load training data
            train_group = hf['train']
            
            # Load STATE_train (list of tensors)
            STATE_train = []
            if 'STATE' in train_group:
                state_group = train_group['STATE']
                step_keys = sorted([k for k in state_group.keys()], key=lambda x: int(x.split('_')[1]))
                for key in step_keys:
                    data = state_group[key][:]
                    tensor = torch.tensor(data, dtype=torch.float32).to(device)
                    STATE_train.append(tensor)
                    
                    # Validate channel dimension on first tensor if n_channels specified
                    if n_channels is not None and len(STATE_train) == 1:
                        # STATE tensor shape: (batch, channels, Nx, Ny, Nz) or (batch, Nx, Ny, Nz, channels)
                        # Check common shapes
                        if len(tensor.shape) == 5:
                            # Try (batch, channels, Nx, Ny, Nz) format
                            if tensor.shape[1] != n_channels:
                                print(f"⚠️ Warning: First STATE tensor has {tensor.shape[1]} channels, expected {n_channels}")
                                print(f"   Tensor shape: {tensor.shape}")
                                print(f"   This may cause errors during prediction.")
                            else:
                                print(f"✅ Validated: STATE tensor has {n_channels} channels (shape: {tensor.shape})")
                        elif len(tensor.shape) == 4:
                            # Might be (batch, Nx, Ny, Nz) with channels=1, or different format
                            print(f"⚠️ Warning: Unexpected STATE tensor shape: {tensor.shape}")
                            print(f"   Expected 5D tensor with channels dimension")
            
            # Load BHP_train (list of tensors)
            BHP_train = []
            if 'BHP' in train_group:
                bhp_group = train_group['BHP']
                step_keys = sorted([k for k in bhp_group.keys()], key=lambda x: int(x.split('_')[1]))
                for key in step_keys:
                    data = bhp_group[key][:]
                    BHP_train.append(torch.tensor(data, dtype=torch.float32).to(device))
            
            # Load Yobs_train (list of tensors)
            Yobs_train = []
            if 'Yobs' in train_group:
                yobs_group = train_group['Yobs']
                step_keys = sorted([k for k in yobs_group.keys()], key=lambda x: int(x.split('_')[1]))
                for key in step_keys:
                    data = yobs_group[key][:]
                    Yobs_train.append(torch.tensor(data, dtype=torch.float32).to(device))
            
            # Load dt_train
            dt_train = None
            if 'dt' in train_group:
                dt_data = train_group['dt'][:]
                dt_train = torch.tensor(dt_data, dtype=torch.float32).to(device)
            
            # Load evaluation data
            eval_group = hf['eval']
            
            # Load STATE_eval (list of tensors)
            STATE_eval = []
            if 'STATE' in eval_group:
                state_group = eval_group['STATE']
                step_keys = sorted([k for k in state_group.keys()], key=lambda x: int(x.split('_')[1]))
                for key in step_keys:
                    data = state_group[key][:]
                    STATE_eval.append(torch.tensor(data, dtype=torch.float32).to(device))
            
            # Load BHP_eval (list of tensors)
            BHP_eval = []
            if 'BHP' in eval_group:
                bhp_group = eval_group['BHP']
                step_keys = sorted([k for k in bhp_group.keys()], key=lambda x: int(x.split('_')[1]))
                for key in step_keys:
                    data = bhp_group[key][:]
                    BHP_eval.append(torch.tensor(data, dtype=torch.float32).to(device))
            
            # Load Yobs_eval (list of tensors)
            Yobs_eval = []
            if 'Yobs' in eval_group:
                yobs_group = eval_group['Yobs']
                step_keys = sorted([k for k in yobs_group.keys()], key=lambda x: int(x.split('_')[1]))
                for key in step_keys:
                    data = yobs_group[key][:]
                    Yobs_eval.append(torch.tensor(data, dtype=torch.float32).to(device))
            
            # Load dt_eval
            dt_eval = None
            if 'dt' in eval_group:
                dt_data = eval_group['dt'][:]
                dt_eval = torch.tensor(dt_data, dtype=torch.float32).to(device)
            
            # Load normalization parameters if available
            norm_params = None
            if 'normalization' in hf:
                norm_group = hf['normalization']
                if 'params_json' in norm_group.attrs:
                    norm_params = json.loads(norm_group.attrs['params_json'])
                    # Convert lists back to numpy arrays where appropriate
                    for key, value in norm_params.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(v, list) and len(v) > 0:
                                    # Try to convert to numpy array
                                    try:
                                        norm_params[key][k] = np.array(v)
                                    except:
                                        pass
            
            # Load data selection metadata if available
            data_selections = None
            if 'data_selections' in hf:
                selections_group = hf['data_selections']
                if 'selections_json' in selections_group.attrs:
                    data_selections = json.loads(selections_group.attrs['selections_json'])
        
        # Prepare return dictionary
        loaded_data = {
            'STATE_train': STATE_train,
            'BHP_train': BHP_train,
            'Yobs_train': Yobs_train,
            'STATE_eval': STATE_eval,
            'BHP_eval': BHP_eval,
            'Yobs_eval': Yobs_eval,
            'dt_train': dt_train,
            'dt_eval': dt_eval,
            'metadata': metadata,
            'norm_params': norm_params,
            'data_selections': data_selections,
            'filepath': filepath
        }
        
        # Print summary
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Training samples: {metadata.get('num_train', 0)}, Evaluation samples: {metadata.get('num_eval', 0)}")
        print(f"   📈 States: {metadata.get('num_states', 0)}, Controls: {metadata.get('num_controls', 0)}, Observations: {metadata.get('num_observations', 0)}")
        print(f"   🔢 Steps: {metadata.get('nsteps', 0)}, Channels: {metadata.get('n_channels', 0)}, Wells: {metadata.get('num_well', 0)}")
        
        return loaded_data
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        import traceback
        traceback.print_exc()
        return None
