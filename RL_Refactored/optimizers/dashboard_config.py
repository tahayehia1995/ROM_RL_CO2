"""
Optimizer Configuration Dashboard
==================================

Interactive dashboard for configuring classical optimization methods.
Allows selection of ROM model, initial states, optimizer type, and parameters.

Similar structure to RL configuration dashboard but focused on optimization settings.
"""

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets not available. Dashboard functionality disabled.")

# Import ROM and config utilities
try:
    import sys
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from ROM_Refactored.model.training.rom_wrapper import ROMWithE2C
    from ROM_Refactored.utilities.config_loader import Config
    ROM_AVAILABLE = True
except ImportError:
    ROM_AVAILABLE = False
    ROMWithE2C = None
    Config = None


def auto_detect_action_ranges_from_h5(data_dir=None):
    """
    Automatically detect Gas Injection and Producer BHP ranges from H5 files.
    Same logic as RL configuration dashboard for consistency.
    
    Args:
        data_dir: Directory containing H5 files (sr3_batch_output)
        
    Returns:
        dict with detected ranges and detection status
    """
    import h5py
    
    # Default data directory
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'sr3_batch_output')
    
    # Default fallback values (same as RL dashboard)
    detected_ranges = {
        'gas_inj_min': 24720290.0,
        'gas_inj_max': 100646896.0,
        'bhp_min': 1087.78,
        'bhp_max': 1305.34,
        'detection_successful': False,
        'detection_details': {}
    }
    
    if not os.path.exists(data_dir):
        return detected_ranges
    
    try:
        # Detect Gas Injection ranges from timeseries data
        gas_file = os.path.join(data_dir, 'batch_timeseries_data_GIWELL.h5')
        if os.path.exists(gas_file):
            with h5py.File(gas_file, 'r') as f:
                gas_data = f['data'][:]
                # GIWELL data: first 3 wells are injectors
                if gas_data.shape[2] >= 3:
                    injector_gas = gas_data[:, :, :3]
                    gas_min = np.min(injector_gas)
                    gas_max = np.max(injector_gas)
                    
                    detected_ranges['gas_inj_min'] = float(gas_min)
                    detected_ranges['gas_inj_max'] = float(gas_max)
                    detected_ranges['detection_details']['gas'] = {
                        'min': float(gas_min),
                        'max': float(gas_max),
                        'source': 'batch_timeseries_data_GIWELL.h5'
                    }
        
        # Detect BHP ranges from timeseries data
        bhp_file = os.path.join(data_dir, 'batch_timeseries_data_BHP.h5')
        if os.path.exists(bhp_file):
            with h5py.File(bhp_file, 'r') as f:
                bhp_data = f['data'][:]
                # BHP data: last 3 wells (indices 3,4,5) are producers
                if bhp_data.shape[2] >= 6:
                    producer_bhp = bhp_data[:, :, 3:6]
                    bhp_min = np.min(producer_bhp)
                    bhp_max = np.max(producer_bhp)
                    
                    detected_ranges['bhp_min'] = float(bhp_min)
                    detected_ranges['bhp_max'] = float(bhp_max)
                    detected_ranges['detection_details']['bhp'] = {
                        'min': float(bhp_min),
                        'max': float(bhp_max),
                        'source': 'batch_timeseries_data_BHP.h5'
                    }
        
        # Check if detection was successful
        if 'gas' in detected_ranges['detection_details'] or 'bhp' in detected_ranges['detection_details']:
            detected_ranges['detection_successful'] = True
            
    except Exception as e:
        print(f"⚠️ Auto-detection warning: {e}")
    
    return detected_ranges


class OptimizerConfigDashboard:
    """
    Interactive dashboard for optimizer configuration.
    
    Provides tabs for:
    1. ROM Model Selection
    2. Initial States Selection
    3. Optimizer Selection & Parameters
    4. Control Bounds Configuration
    5. Economic Parameters
    """
    
    # Available optimizer types
    OPTIMIZER_TYPES = {
        'LS-SQP-StoSAG': {
            'name': 'LS-SQP with StoSAG',
            'description': 'Line-Search SQP with Stochastic Simplex Approximate Gradients (single realization)',
            'params': {
                'gradient_type': {'type': 'dropdown', 'default': 'spsa', 
                                  'options': [
                                      ('SPSA (Fast - 2 evals/sample)', 'spsa'),
                                      ('StoSAG (Accurate - N evals)', 'stosag'),
                                      ('Forward FD (N evals)', 'fd_forward'),
                                      ('Central FD (2N evals)', 'fd_central')
                                  ],
                                  'description': 'Gradient estimation method'},
                'perturbation_size': {'type': 'float', 'default': 0.01, 'min': 0.001, 'max': 0.1,
                                      'description': 'Relative perturbation for gradient estimation'},
                'spsa_num_samples': {'type': 'int', 'default': 5, 'min': 1, 'max': 20,
                                     'description': 'SPSA samples to average (more = accurate, slower)'},
                'max_iterations': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000,
                                   'description': 'Maximum optimization iterations'},
                'tolerance': {'type': 'float', 'default': 1e-6, 'min': 1e-10, 'max': 1e-2,
                             'description': 'Convergence tolerance'}
            }
        }
        # Future optimizers can be added here:
        # 'GA': {...},
        # 'PSO': {...},
        # 'EnOpt': {...},
    }
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize optimizer configuration dashboard.
        
        Args:
            config_path: Path to RL config file for shared parameters
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets required for dashboard. Install with: pip install ipywidgets")
        
        # Load configuration
        self.config_path = Path(__file__).parent / config_path
        if not self.config_path.exists():
            self.config_path = Path(__file__).parent.parent / config_path
        
        if self.config_path.exists():
            self.config = Config(str(self.config_path))
        else:
            print(f"Warning: Config file not found at {self.config_path}")
            self.config = None
        
        # Initialize paths
        self.rom_folder = str(Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'saved_models')
        self.state_folder = str(Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'sr3_batch_output')
        
        # Storage for selections
        self.optimizer_config = {
            'rom_models': [],
            'selected_rom': None,
            'available_states': [],
            'selected_states': ['SW', 'SG', 'PRES'],
            'optimizer_type': 'LS-SQP-StoSAG',
            'optimizer_params': {},
            'action_ranges': {},
            'economic_params': {},
            'num_steps': 30,
            'config': self.config,
            'norm_params': {},
            'device': None,
            'rom_model': None,
            'z0_options': None
        }
        
        # Loaded model storage
        self.loaded_rom_model = None
        self.generated_z0_options = None
        self.device = None
        
        # Create dashboard
        self._create_dashboard()
    
    def _create_dashboard(self):
        """Create the main dashboard interface."""
        # Header
        header = widgets.HTML("""
        <div style="background: linear-gradient(135deg, #1a5f7a 0%, #159895 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">Classical Optimizer Configuration</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                Configure ROM model, initial states, and optimization parameters
            </p>
        </div>
        """)
        
        # Create tabs
        self.rom_tab = widgets.VBox([])
        self.states_tab = widgets.VBox([])
        self.optimizer_tab = widgets.VBox([])
        self.bounds_tab = widgets.VBox([])
        self.economics_tab = widgets.VBox([])
        
        tabs = widgets.Tab(children=[
            self.rom_tab,
            self.states_tab, 
            self.optimizer_tab,
            self.bounds_tab,
            self.economics_tab
        ])
        tabs.set_title(0, '🧠 ROM Model')
        tabs.set_title(1, '🏔️ Initial States')
        tabs.set_title(2, '⚙️ Optimizer')
        tabs.set_title(3, '📊 Control Bounds')
        tabs.set_title(4, '💰 Economics')
        
        # Apply button
        self.apply_button = widgets.Button(
            description='🚀 Apply Configuration & Load Model',
            button_style='success',
            layout=widgets.Layout(width='300px', height='40px')
        )
        self.apply_button.on_click(self._on_apply)
        
        # Output area
        self.output = widgets.Output()
        
        # Assemble dashboard
        self.dashboard = widgets.VBox([
            header,
            tabs,
            widgets.HBox([self.apply_button], layout=widgets.Layout(justify_content='center')),
            self.output
        ])
        
        # Populate tabs
        self._populate_rom_tab()
        self._populate_states_tab()
        self._populate_optimizer_tab()
        self._populate_bounds_tab()
        self._populate_economics_tab()
        
        # Initial scan
        self._scan_rom_models()
        self._scan_available_states()
    
    def _populate_rom_tab(self):
        """Populate ROM model selection tab."""
        # ROM folder input
        self.rom_folder_input = widgets.Text(
            value=self.rom_folder,
            description='ROM Folder:',
            layout=widgets.Layout(width='80%'),
            style={'description_width': '100px'}
        )
        
        # Scan button
        scan_button = widgets.Button(description='🔍 Scan', button_style='info')
        scan_button.on_click(lambda b: self._scan_rom_models())
        
        # ROM model dropdown
        self.rom_dropdown = widgets.Dropdown(
            options=[],
            description='Select ROM:',
            layout=widgets.Layout(width='80%'),
            style={'description_width': '100px'}
        )
        
        # Model info display
        self.model_info_html = widgets.HTML(value="<p>No model selected</p>")
        
        self.rom_tab.children = [
            widgets.HTML("<h3>🧠 ROM Model Selection</h3>"),
            widgets.HBox([self.rom_folder_input, scan_button]),
            self.rom_dropdown,
            widgets.HTML("<h4>Model Information:</h4>"),
            self.model_info_html
        ]
        
        # Connect dropdown change
        self.rom_dropdown.observe(self._on_rom_selected, names='value')
    
    def _populate_states_tab(self):
        """Populate initial states selection tab."""
        self.state_checkboxes = {}
        self.known_states = ['SW', 'SG', 'PRES', 'PERMI', 'PERMJ', 'PERMK', 'POROS']
        
        state_widgets = [widgets.HTML("<h3>🏔️ Select States for Optimization</h3>")]
        state_widgets.append(widgets.HTML(
            "<p><i>Channels are auto-selected when you choose a ROM model.</i></p>"
            "<p><i><b>ch2:</b> SG, PRES | <b>ch4:</b> SG, PRES, PERMI, POROS</i></p>"
        ))
        
        # Default to ch2 (SG, PRES) - will be auto-updated when model is selected
        for state in self.known_states:
            cb = widgets.Checkbox(
                value=state in ['SG', 'PRES'],  # Default for ch2 models
                description=state,
                disabled=False
            )
            self.state_checkboxes[state] = cb
            state_widgets.append(cb)
        
        # Z0 Case Index Selection
        state_widgets.append(widgets.HTML("<h4>Initial State (Z0) Selection:</h4>"))
        state_widgets.append(widgets.HTML(
            "<p><i>Select which case (0-999) to use as initial reservoir state.</i></p>"
        ))
        
        self.z0_case_index_input = widgets.IntText(
            value=0,
            description='Case Index:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        self.z0_case_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=999,
            step=1,
            description='',
            layout=widgets.Layout(width='400px')
        )
        # Link slider and text input
        widgets.link((self.z0_case_index_input, 'value'), (self.z0_case_slider, 'value'))
        state_widgets.append(widgets.HBox([self.z0_case_index_input, self.z0_case_slider]))
        
        # Number of timesteps
        state_widgets.append(widgets.HTML("<h4>Simulation Length:</h4>"))
        self.num_steps_input = widgets.IntSlider(
            value=30,
            min=10,
            max=100,
            step=5,
            description='Timesteps:',
            style={'description_width': '100px'}
        )
        state_widgets.append(self.num_steps_input)
        
        self.states_tab.children = state_widgets
    
    def _populate_optimizer_tab(self):
        """Populate optimizer selection and parameters tab."""
        # Optimizer type dropdown
        optimizer_options = [(v['name'], k) for k, v in self.OPTIMIZER_TYPES.items()]
        self.optimizer_type_dropdown = widgets.Dropdown(
            options=optimizer_options,
            value='LS-SQP-StoSAG',
            description='Optimizer:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='50%')
        )
        
        # Parameter container (dynamically updated)
        self.optimizer_params_box = widgets.VBox([])
        
        # Optimizer description
        self.optimizer_desc_html = widgets.HTML()
        
        self.optimizer_tab.children = [
            widgets.HTML("<h3>⚙️ Optimizer Configuration</h3>"),
            self.optimizer_type_dropdown,
            self.optimizer_desc_html,
            widgets.HTML("<h4>Parameters:</h4>"),
            self.optimizer_params_box
        ]
        
        # Connect dropdown change
        self.optimizer_type_dropdown.observe(self._on_optimizer_type_changed, names='value')
        self._update_optimizer_params()
    
    def _populate_bounds_tab(self):
        """Populate control bounds configuration tab."""
        self.bounds_widgets = {}
        
        # Auto-detect ranges from H5 files (same as RL dashboard)
        detected_ranges = auto_detect_action_ranges_from_h5(data_dir=self.state_folder)
        
        bhp_min = detected_ranges['bhp_min']
        bhp_max = detected_ranges['bhp_max']
        gas_min = detected_ranges['gas_inj_min']
        gas_max = detected_ranges['gas_inj_max']
        
        if detected_ranges['detection_successful']:
            source_text = "<p style='color: green;'>✅ Ranges auto-detected from H5 files (same as RL Dashboard)</p>"
            details = detected_ranges['detection_details']
            if 'bhp' in details:
                source_text += f"<p>&nbsp;&nbsp;&nbsp;BHP: [{details['bhp']['min']:.2f}, {details['bhp']['max']:.2f}] psi from {details['bhp']['source']}</p>"
            if 'gas' in details:
                source_text += f"<p>&nbsp;&nbsp;&nbsp;Gas: [{details['gas']['min']:.0f}, {details['gas']['max']:.0f}] ft³/day from {details['gas']['source']}</p>"
        else:
            source_text = "<p style='color: orange;'>⚠️ Using fallback ranges (H5 files not found)</p>"
        
        bounds_content = [
            widgets.HTML("<h3>📊 Control Bounds Configuration</h3>"),
            widgets.HTML("<p><i>Control ranges matching RL action space.</i></p>"),
            widgets.HTML(source_text),
            widgets.HTML("<h4>Producer BHP (psi):</h4>")
        ]
        
        # Producer BHP bounds
        self.bhp_min_input = widgets.FloatText(
            value=bhp_min,
            description='Min BHP:',
            style={'description_width': '80px'}
        )
        self.bhp_max_input = widgets.FloatText(
            value=bhp_max,
            description='Max BHP:',
            style={'description_width': '80px'}
        )
        bounds_content.append(widgets.HBox([self.bhp_min_input, self.bhp_max_input]))
        
        bounds_content.append(widgets.HTML("<h4>Gas Injection Rate (ft³/day):</h4>"))
        
        # Gas injection bounds
        self.gas_min_input = widgets.FloatText(
            value=gas_min,
            description='Min Gas:',
            style={'description_width': '80px'}
        )
        self.gas_max_input = widgets.FloatText(
            value=gas_max,
            description='Max Gas:',
            style={'description_width': '80px'}
        )
        bounds_content.append(widgets.HBox([self.gas_min_input, self.gas_max_input]))
        
        # Refresh button to re-detect from H5 files
        refresh_btn = widgets.Button(description='🔄 Re-detect from H5 Files', button_style='info')
        refresh_btn.on_click(lambda b: self._refresh_bounds_from_h5())
        bounds_content.append(refresh_btn)
        
        self.bounds_tab.children = bounds_content
    
    def _refresh_bounds_from_h5(self):
        """Refresh control bounds from H5 files."""
        detected_ranges = auto_detect_action_ranges_from_h5(data_dir=self.state_folder)
        
        if detected_ranges['detection_successful']:
            self.bhp_min_input.value = detected_ranges['bhp_min']
            self.bhp_max_input.value = detected_ranges['bhp_max']
            self.gas_min_input.value = detected_ranges['gas_inj_min']
            self.gas_max_input.value = detected_ranges['gas_inj_max']
            print("✅ Control bounds refreshed from H5 files")
            print(f"   BHP: [{detected_ranges['bhp_min']:.2f}, {detected_ranges['bhp_max']:.2f}] psi")
            print(f"   Gas: [{detected_ranges['gas_inj_min']:.0f}, {detected_ranges['gas_inj_max']:.0f}] ft³/day")
        else:
            print("⚠️ H5 files not found. Using current values.")
    
    def _populate_economics_tab(self):
        """Populate economic parameters tab."""
        econ_content = [
            widgets.HTML("<h3>💰 Economic Parameters</h3>"),
            widgets.HTML("<p><i>Configure economic values for NPV calculation.</i></p>"),
            widgets.HTML("<h4>Gas Economics ($/ton):</h4>")
        ]
        
        # Get defaults from config if available
        if self.config and hasattr(self.config, 'rl_model'):
            econ_config = self.config.rl_model.get('economics', {})
            prices = econ_config.get('prices', {})
            default_gas_rev = prices.get('gas_injection_revenue', 40.0)
            default_gas_cost = prices.get('gas_injection_cost', 10.0)
            default_water_pen = prices.get('water_production_penalty', 5.0)
            default_gas_pen = prices.get('gas_production_penalty', 50.0)
            default_scale = econ_config.get('scale_factor', 1000000.0)
        else:
            default_gas_rev = 40.0
            default_gas_cost = 10.0
            default_water_pen = 5.0
            default_gas_pen = 50.0
            default_scale = 1000000.0
        
        self.gas_revenue_input = widgets.FloatText(
            value=default_gas_rev,
            description='Gas Inj Revenue:',
            style={'description_width': '120px'}
        )
        self.gas_cost_input = widgets.FloatText(
            value=default_gas_cost,
            description='Gas Inj Cost:',
            style={'description_width': '120px'}
        )
        econ_content.append(widgets.HBox([self.gas_revenue_input, self.gas_cost_input]))
        
        econ_content.append(widgets.HTML("<h4>Penalties:</h4>"))
        
        self.water_penalty_input = widgets.FloatText(
            value=default_water_pen,
            description='Water Penalty ($/bbl):',
            style={'description_width': '150px'}
        )
        self.gas_penalty_input = widgets.FloatText(
            value=default_gas_pen,
            description='Gas Prod Penalty ($/ton):',
            style={'description_width': '150px'}
        )
        econ_content.append(widgets.HBox([self.water_penalty_input, self.gas_penalty_input]))
        
        econ_content.append(widgets.HTML("<h4>Scaling:</h4>"))
        self.scale_factor_input = widgets.FloatText(
            value=default_scale,
            description='Scale Factor:',
            style={'description_width': '100px'}
        )
        econ_content.append(self.scale_factor_input)
        
        self.economics_tab.children = econ_content
    
    def _scan_rom_models(self):
        """Scan for available ROM models."""
        self.rom_folder = self.rom_folder_input.value
        self.optimizer_config['rom_models'] = []
        
        if not os.path.exists(self.rom_folder):
            print(f"❌ ROM folder not found: {self.rom_folder}")
            self.rom_dropdown.options = []
            return
        
        # Look for model triplets (encoder, decoder, transition)
        models = {}
        
        for filename in os.listdir(self.rom_folder):
            if not filename.endswith('.h5'):
                continue
            
            if 'encoder' in filename:
                base_name = filename.replace('encoder', 'MODEL')
                encoder_file = os.path.join(self.rom_folder, filename)
                decoder_file = os.path.join(self.rom_folder, filename.replace('encoder', 'decoder'))
                transition_file = os.path.join(self.rom_folder, filename.replace('encoder', 'transition'))
                
                if os.path.exists(decoder_file) and os.path.exists(transition_file):
                    model_info = self._parse_model_filename(filename)
                    display_name = self._create_model_display_name(filename, model_info)
                    
                    models[base_name] = {
                        'name': display_name,
                        'encoder': encoder_file,
                        'decoder': decoder_file,
                        'transition': transition_file,
                        'info': model_info
                    }
        
        self.optimizer_config['rom_models'] = list(models.values())
        
        # Update dropdown
        options = [(m['name'], i) for i, m in enumerate(self.optimizer_config['rom_models'])]
        self.rom_dropdown.options = options if options else [('No models found', -1)]
        
        print(f"✅ Found {len(models)} ROM model(s)")
    
    def _parse_model_filename(self, filename: str) -> Dict:
        """Parse model filename to extract configuration info."""
        info = {}
        
        patterns = {
            'latent_dim': r'ld(\d+)',
            'batch_size': r'bs(\d+)',
            'nsteps': r'ns(\d+)',
            'channels': r'ch(\d+)',
            'run': r'run(\d+)',
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                info[param] = int(match.group(1))
        
        # Extract encoder_hidden_dims
        ehd_match = re.search(r'_ehd([\d-]+)', filename)
        if ehd_match:
            try:
                info['encoder_hidden_dims'] = [int(d) for d in ehd_match.group(1).split('-')]
            except:
                pass
        
        return info
    
    def _create_model_display_name(self, filename: str, info: Dict) -> str:
        """Create display name for model."""
        parts = []
        if 'batch_size' in info:
            parts.append(f"bs={info['batch_size']}")
        if 'latent_dim' in info:
            parts.append(f"ld={info['latent_dim']}")
        if 'run' in info:
            parts.append(f"run={info['run']}")
        
        if parts:
            return f"Model ({', '.join(parts)})"
        return filename
    
    def _scan_available_states(self):
        """Scan for available state data files."""
        if not os.path.exists(self.state_folder):
            return
        
        available = []
        for state in self.known_states:
            state_file = os.path.join(self.state_folder, f'batch_spatial_properties_{state}.h5')
            if os.path.exists(state_file):
                available.append(state)
                if state in self.state_checkboxes:
                    self.state_checkboxes[state].disabled = False
            else:
                if state in self.state_checkboxes:
                    self.state_checkboxes[state].disabled = True
                    self.state_checkboxes[state].value = False
        
        self.optimizer_config['available_states'] = available
    
    def _on_rom_selected(self, change):
        """Handle ROM model selection."""
        idx = change['new']
        if idx >= 0 and idx < len(self.optimizer_config['rom_models']):
            model = self.optimizer_config['rom_models'][idx]
            self.optimizer_config['selected_rom'] = model
            
            # Auto-select channels based on model's number of channels
            num_channels = model['info'].get('channels', 2)
            self._auto_select_channels(num_channels)
            
            # Update info display
            info_html = f"""
            <table style="width:100%">
                <tr><td><b>Encoder:</b></td><td>{os.path.basename(model['encoder'])}</td></tr>
                <tr><td><b>Decoder:</b></td><td>{os.path.basename(model['decoder'])}</td></tr>
                <tr><td><b>Transition:</b></td><td>{os.path.basename(model['transition'])}</td></tr>
            """
            for key, val in model['info'].items():
                info_html += f"<tr><td><b>{key}:</b></td><td>{val}</td></tr>"
            info_html += "</table>"
            
            # Add channel selection info
            info_html += f"<p style='color: green;'>✅ Auto-selected {num_channels} channels for this model</p>"
            
            self.model_info_html.value = info_html
    
    def _auto_select_channels(self, num_channels: int):
        """
        Auto-select channels based on model's number of channels.
        
        Channel mappings:
        - ch2: SG, PRES
        - ch4: SG, PRES, PERMI, POROS
        - ch7: All channels
        """
        # Define channel sets for different model configurations
        channel_configs = {
            2: ['SG', 'PRES'],
            3: ['SG', 'PRES', 'SW'],
            4: ['SG', 'PRES', 'PERMI', 'POROS'],
            5: ['SG', 'PRES', 'PERMI', 'POROS', 'SW'],
            6: ['SG', 'PRES', 'PERMI', 'PERMJ', 'PERMK', 'POROS'],
            7: ['SW', 'SG', 'PRES', 'PERMI', 'PERMJ', 'PERMK', 'POROS'],
        }
        
        # Get the appropriate channels for this model
        selected_channels = channel_configs.get(num_channels, ['SG', 'PRES'])
        
        # Update checkboxes
        for state, checkbox in self.state_checkboxes.items():
            if state in selected_channels:
                checkbox.value = True
            else:
                checkbox.value = False
        
        print(f"   📊 Auto-selected channels for {num_channels}-channel model: {selected_channels}")
    
    def _on_optimizer_type_changed(self, change):
        """Handle optimizer type change."""
        self._update_optimizer_params()
    
    def _update_optimizer_params(self):
        """Update optimizer parameters widget based on selected type."""
        opt_type = self.optimizer_type_dropdown.value
        opt_info = self.OPTIMIZER_TYPES.get(opt_type, {})
        
        # Update description
        self.optimizer_desc_html.value = f"<p><i>{opt_info.get('description', '')}</i></p>"
        
        # Create parameter widgets
        param_widgets = []
        self.param_inputs = {}
        
        for param_name, param_info in opt_info.get('params', {}).items():
            if param_info['type'] == 'int':
                widget = widgets.IntSlider(
                    value=param_info['default'],
                    min=param_info['min'],
                    max=param_info['max'],
                    description=param_name + ':',
                    style={'description_width': '150px'},
                    layout=widgets.Layout(width='80%')
                )
            elif param_info['type'] == 'float':
                widget = widgets.FloatLogSlider(
                    value=param_info['default'],
                    min=np.log10(param_info['min']),
                    max=np.log10(param_info['max']),
                    description=param_name + ':',
                    style={'description_width': '150px'},
                    layout=widgets.Layout(width='80%')
                )
            elif param_info['type'] == 'dropdown':
                widget = widgets.Dropdown(
                    options=param_info['options'],
                    value=param_info['default'],
                    description=param_name + ':',
                    style={'description_width': '150px'},
                    layout=widgets.Layout(width='80%')
                )
            else:
                continue
            
            param_widgets.append(widget)
            param_widgets.append(widgets.HTML(f"<p style='margin-left:150px; color:gray;'><small>{param_info['description']}</small></p>"))
            self.param_inputs[param_name] = widget
        
        self.optimizer_params_box.children = param_widgets
    
    def _on_apply(self, button):
        """Apply configuration and load ROM model."""
        with self.output:
            clear_output()
            print("🔄 Applying optimizer configuration...")
            
            try:
                # Collect configuration
                self._collect_configuration()
                
                # Load ROM model
                success = self._load_rom_model()
                
                if success:
                    # Generate Z0 options
                    success = self._generate_z0_options()
                
                if success:
                    print("\n✅ Configuration applied successfully!")
                    print("\n📋 Configuration Summary:")
                    self._print_summary()
                    
                    # Store globally for access from run file
                    import builtins
                    builtins.optimizer_dashboard_config = self.get_config()
                    print("\n💾 Configuration stored as 'optimizer_dashboard_config'")
                else:
                    print("\n❌ Configuration failed. Please check errors above.")
                    
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def _collect_configuration(self):
        """Collect all configuration from widgets."""
        # Selected states
        self.optimizer_config['selected_states'] = [
            state for state, cb in self.state_checkboxes.items() if cb.value
        ]
        
        # Number of steps
        self.optimizer_config['num_steps'] = self.num_steps_input.value
        
        # Optimizer type and params
        self.optimizer_config['optimizer_type'] = self.optimizer_type_dropdown.value
        self.optimizer_config['optimizer_params'] = {
            name: widget.value for name, widget in self.param_inputs.items()
        }
        
        # Action ranges
        self.optimizer_config['action_ranges'] = {
            'producer_bhp': {
                'min': self.bhp_min_input.value,
                'max': self.bhp_max_input.value
            },
            'gas_injection': {
                'min': self.gas_min_input.value,
                'max': self.gas_max_input.value
            }
        }
        
        # Economic params (update config)
        if self.config and hasattr(self.config, 'rl_model'):
            self.config.rl_model['economics']['prices']['gas_injection_revenue'] = self.gas_revenue_input.value
            self.config.rl_model['economics']['prices']['gas_injection_cost'] = self.gas_cost_input.value
            self.config.rl_model['economics']['prices']['water_production_penalty'] = self.water_penalty_input.value
            self.config.rl_model['economics']['prices']['gas_production_penalty'] = self.gas_penalty_input.value
            self.config.rl_model['economics']['scale_factor'] = self.scale_factor_input.value
    
    def _load_rom_model(self) -> bool:
        """Load selected ROM model."""
        if not self.optimizer_config['selected_rom']:
            print("❌ No ROM model selected!")
            return False
        
        if not ROM_AVAILABLE:
            print("❌ ROM modules not available!")
            return False
        
        print("🧠 Loading ROM model...")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_config['device'] = self.device
        print(f"   Device: {self.device}")
        
        # Load ROM config
        rom_config_path = Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'config.yaml'
        rom_config = Config(str(rom_config_path))
        
        # Update config with model architecture info
        model_info = self.optimizer_config['selected_rom'].get('info', {})
        
        # CRITICAL: Set number of channels from model info
        if 'channels' in model_info:
            num_channels = model_info['channels']
            rom_config.model['n_channels'] = num_channels
            
            # Update encoder conv1 input channels
            if 'encoder' in rom_config.config and 'conv_layers' in rom_config.config['encoder']:
                if 'conv1' in rom_config.config['encoder']['conv_layers']:
                    rom_config.config['encoder']['conv_layers']['conv1'][0] = num_channels
            
            # Update decoder final_conv output channels
            if 'decoder' in rom_config.config and 'deconv_layers' in rom_config.config['decoder']:
                if 'final_conv' in rom_config.config['decoder']['deconv_layers']:
                    rom_config.config['decoder']['deconv_layers']['final_conv'][1] = num_channels
            
            print(f"   📊 Model channels: {num_channels}")
        
        if 'latent_dim' in model_info:
            rom_config.model['latent_dim'] = model_info['latent_dim']
            print(f"   📊 Latent dim: {model_info['latent_dim']}")
            
        if 'encoder_hidden_dims' in model_info:
            rom_config.config['transition']['encoder_hidden_dims'] = model_info['encoder_hidden_dims']
        else:
            # Try to infer from checkpoint
            transition_file = self.optimizer_config['selected_rom']['transition']
            encoder_hidden_dims = self._infer_encoder_hidden_dims(transition_file)
            if encoder_hidden_dims:
                rom_config.config['transition']['encoder_hidden_dims'] = encoder_hidden_dims
        
        # Initialize and load model
        self.loaded_rom_model = ROMWithE2C(rom_config).to(self.device)
        
        selected_rom = self.optimizer_config['selected_rom']
        self.loaded_rom_model.model.load_weights_from_file(
            selected_rom['encoder'],
            selected_rom['decoder'],
            selected_rom['transition']
        )
        self.loaded_rom_model.eval()
        
        self.optimizer_config['rom_model'] = self.loaded_rom_model
        print("   ✅ ROM model loaded successfully!")
        
        # Load normalization parameters
        self._load_normalization_params()
        
        return True
    
    def _infer_encoder_hidden_dims(self, transition_file: str) -> Optional[List[int]]:
        """Infer encoder hidden dims from checkpoint."""
        try:
            state_dict = torch.load(transition_file, map_location='cpu')
            hidden_dims = []
            layer_idx = 0
            
            while True:
                weight_key = f'trans_encoder.{layer_idx}.0.weight'
                if weight_key not in state_dict:
                    break
                
                out_features = state_dict[weight_key].shape[0]
                next_key = f'trans_encoder.{layer_idx + 1}.0.weight'
                
                if next_key in state_dict:
                    hidden_dims.append(out_features)
                layer_idx += 1
            
            return hidden_dims if hidden_dims else [200, 200]
        except:
            return [200, 200]
    
    def _load_normalization_params(self):
        """Load normalization parameters."""
        norm_dir = Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'processed_data'
        
        # Find most recent normalization file
        norm_files = list(norm_dir.glob('normalization_parameters_*.json'))
        if not norm_files:
            print("   ⚠️ No normalization parameters found, using defaults")
            return
        
        latest_file = max(norm_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file) as f:
            norm_data = json.load(f)
        
        # Extract relevant parameters
        self.optimizer_config['norm_params'] = {}
        
        # Handle spatial_channels (can be dict with channel names as keys)
        if 'spatial_channels' in norm_data:
            spatial = norm_data['spatial_channels']
            if isinstance(spatial, dict):
                # Dictionary format: {"SW": {...}, "SG": {...}, ...}
                for name, channel_data in spatial.items():
                    params = channel_data.get('parameters', channel_data)
                    self.optimizer_config['norm_params'][name] = {
                        'min': float(params.get('min', 0)),
                        'max': float(params.get('max', 1)),
                        'type': channel_data.get('normalization_type', 'minmax')
                    }
            elif isinstance(spatial, list):
                # List format: [{"name": "SW", ...}, ...]
                for channel in spatial:
                    name = channel.get('name', '')
                    self.optimizer_config['norm_params'][name] = {
                        'min': float(channel.get('min', 0)),
                        'max': float(channel.get('max', 1)),
                        'type': channel.get('normalization', 'minmax')
                    }
        
        # Handle control_variables (can be dict or list)
        if 'control_variables' in norm_data:
            controls = norm_data['control_variables']
            if isinstance(controls, dict):
                for name, var_data in controls.items():
                    # Parameters may be nested
                    params = var_data.get('parameters', var_data)
                    self.optimizer_config['norm_params'][name] = {
                        'min': float(params.get('min', 0)),
                        'max': float(params.get('max', 1))
                    }
            elif isinstance(controls, list):
                for var in controls:
                    name = var.get('variable', var.get('name', ''))
                    params = var.get('parameters', var)
                    self.optimizer_config['norm_params'][name] = {
                        'min': float(params.get('min', 0)),
                        'max': float(params.get('max', 1))
                    }
        
        # Handle observation_variables (can be dict or list)
        if 'observation_variables' in norm_data:
            obs = norm_data['observation_variables']
            if isinstance(obs, dict):
                for name, var_data in obs.items():
                    if name not in self.optimizer_config['norm_params']:
                        # Parameters may be nested
                        params = var_data.get('parameters', var_data)
                        self.optimizer_config['norm_params'][name] = {
                            'min': float(params.get('min', 0)),
                            'max': float(params.get('max', 1))
                        }
            elif isinstance(obs, list):
                for var in obs:
                    name = var.get('variable', var.get('name', ''))
                    if name not in self.optimizer_config['norm_params']:
                        params = var.get('parameters', var)
                        self.optimizer_config['norm_params'][name] = {
                            'min': float(params.get('min', 0)),
                            'max': float(params.get('max', 1))
                        }
        
        print(f"   ✅ Loaded normalization params from {latest_file.name}")
    
    def _generate_z0_options(self) -> bool:
        """Generate initial state (Z0) from selected case index."""
        # Get selected case index
        case_index = self.z0_case_index_input.value
        self.optimizer_config['z0_case_index'] = case_index
        
        print(f"🏔️ Loading initial state from Case #{case_index}...")
        
        try:
            import h5py
            
            # Load state data for selected case only
            selected_states = self.optimizer_config['selected_states']
            state_data = {}
            num_cases = 0
            
            for state in selected_states:
                state_file = os.path.join(self.state_folder, f'batch_spatial_properties_{state}.h5')
                if os.path.exists(state_file):
                    with h5py.File(state_file, 'r') as f:
                        data = f['data'][:]
                        num_cases = data.shape[0]
                        
                        # Validate case index
                        if case_index >= num_cases:
                            print(f"   ❌ Case index {case_index} out of range (0-{num_cases-1})")
                            return False
                        
                        # Take first timestep of selected case
                        state_data[state] = data[case_index, 0, :, :, :]  # (X, Y, Z)
            
            if not state_data:
                print("   ❌ No state data found!")
                return False
            
            # Normalize and stack
            channels = []
            for state in selected_states:
                if state in state_data:
                    data = state_data[state]
                    # Normalize
                    if state in self.optimizer_config['norm_params']:
                        params = self.optimizer_config['norm_params'][state]
                        data = (data - params['min']) / (params['max'] - params['min'] + 1e-8)
                    # Add batch dimension: (1, X, Y, Z)
                    channels.append(torch.tensor(data, dtype=torch.float32).unsqueeze(0))
            
            # Stack channels: (1, num_channels, X, Y, Z)
            spatial_state = torch.cat([ch.unsqueeze(1) for ch in channels], dim=1).to(self.device)
            
            # Encode to latent space
            with torch.no_grad():
                z0 = self.loaded_rom_model.model.encoder(spatial_state)
            
            self.generated_z0_options = z0  # Single Z0 (batch size 1)
            self.optimizer_config['z0_options'] = z0
            
            print(f"   ✅ Loaded initial state from Case #{case_index} (of {num_cases} available)")
            print(f"   Z0 shape: {z0.shape}")
            print(f"   Z0 stats: mean={z0.mean().item():.4f}, std={z0.std().item():.4f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error generating Z0: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_summary(self):
        """Print configuration summary."""
        print(f"\n{'='*50}")
        print(f"Optimizer: {self.optimizer_config['optimizer_type']}")
        print(f"ROM Model: {self.optimizer_config['selected_rom']['name']}")
        print(f"States: {self.optimizer_config['selected_states']}")
        print(f"Timesteps: {self.optimizer_config['num_steps']}")
        print(f"Z0 Case Index: {self.optimizer_config.get('z0_case_index', 0)}")
        print(f"\nOptimizer Parameters:")
        for k, v in self.optimizer_config['optimizer_params'].items():
            print(f"  {k}: {v}")
        print(f"\nControl Bounds (matching RL):")
        print(f"  BHP: [{self.optimizer_config['action_ranges']['producer_bhp']['min']:.2f}, "
              f"{self.optimizer_config['action_ranges']['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas: [{self.optimizer_config['action_ranges']['gas_injection']['min']:.0f}, "
              f"{self.optimizer_config['action_ranges']['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'='*50}")
    
    def get_config(self) -> Dict:
        """Get complete configuration dictionary."""
        opt_params = self.optimizer_config.get('optimizer_params', {})
        
        return {
            'optimizer_type': self.optimizer_config['optimizer_type'],
            'rom_model': self.loaded_rom_model,
            'config': self.config,
            'norm_params': self.optimizer_config['norm_params'],
            'device': self.device,
            'z0_options': self.generated_z0_options,
            'z0_case_index': self.optimizer_config.get('z0_case_index', 0),
            'num_steps': self.optimizer_config['num_steps'],
            'action_ranges': self.optimizer_config['action_ranges'],
            'stosag_params': {
                'num_realizations': 1,  # Single realization mode
                'perturbation_size': opt_params.get('perturbation_size', 0.01),
                'gradient_type': opt_params.get('gradient_type', 'spsa'),  # Use SPSA by default (fast!)
                'spsa_num_samples': opt_params.get('spsa_num_samples', 5)
            },
            'sqp_params': {
                'max_iterations': opt_params.get('max_iterations', 100),
                'tolerance': opt_params.get('tolerance', 1e-6)
            }
        }
    
    def display(self):
        """Display the dashboard."""
        display(self.dashboard)
        return self
    
    def _ipython_display_(self):
        """IPython display hook."""
        self.display()


def launch_optimizer_config_dashboard(config_path: str = 'config.yaml') -> OptimizerConfigDashboard:
    """
    Launch the optimizer configuration dashboard.
    
    Single-line launcher for use in Jupyter notebooks.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OptimizerConfigDashboard instance
    """
    dashboard = OptimizerConfigDashboard(config_path)
    dashboard.display()
    return dashboard
