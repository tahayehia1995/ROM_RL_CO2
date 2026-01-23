"""
Testing Dashboard for E2C Model
Interactive dashboard for loading models and generating test visualizations
"""

import os
import torch
import glob
import re
import yaml

# Import widget utilities
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None

from utilities.config_loader import Config
from data_preprocessing import load_processed_data
from testing.prediction import generate_test_visualization_standalone


class TestingDashboard:
    """
    Interactive dashboard for testing and visualization
    """
    
    def __init__(self):
        self.loaded_data = None
        self.config = None
        self.my_rom = None
        self.device = None
        self.available_models = []
        self.selected_model_info = None
        
        if not WIDGETS_AVAILABLE:
            print("⚠️ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
            return
            
        self._create_widgets()
        self._setup_event_handlers()
        # Scan for models on initialization
        self._refresh_models_handler(None)
    
    def _create_widgets(self):
        """Create all dashboard widgets"""
        
        # Header
        self.header = widgets.HTML(
            value="<h1>🧪 Testing & Visualization Dashboard</h1>",
            layout=widgets.Layout(margin='10px 0px')
        )
        
        # Processed data path
        self.processed_data_input = widgets.Text(
            value="./processed_data/",
            description="Processed Data:",
            placeholder="Path to processed data directory",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        self.load_data_btn = widgets.Button(
            description="📁 Load Data",
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        # Config file path
        self.config_path_input = widgets.Text(
            value="config.yaml",
            description="Config File:",
            placeholder="Path to config.yaml",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        self.load_config_btn = widgets.Button(
            description="📄 Load Config",
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        # Model selection dropdown
        self.model_selection = widgets.Dropdown(
            options=[("Scanning for models...", None)],
            description="Select Model:",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        self.refresh_models_btn = widgets.Button(
            description="🔄 Refresh",
            button_style='info',
            layout=widgets.Layout(width='100px')
        )
        
        self.load_model_btn = widgets.Button(
            description="🤖 Load Model",
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        # Raw data directory
        self.data_dir_input = widgets.Text(
            value="sr3_batch_output/",
            description="Raw Data Directory:",
            placeholder="Path to raw H5 files",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # Testing parameters
        self.num_tsteps_input = widgets.IntSlider(
            value=30,
            min=1,
            max=100,
            step=1,
            description="Time Steps:",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        self.status_output = widgets.Output()
        
        # Generate visualization button
        self.generate_viz_btn = widgets.Button(
            description="🎨 Generate Visualization",
            button_style='success',
            layout=widgets.Layout(width='250px', margin='20px 0px')
        )
        
        self.viz_output = widgets.Output()
        
        # Main layout
        self.main_widget = widgets.VBox([
            self.header,
            widgets.HBox([self.processed_data_input, self.load_data_btn]),
            widgets.HBox([self.config_path_input, self.load_config_btn]),
            widgets.HBox([self.model_selection, self.refresh_models_btn, self.load_model_btn]),
            widgets.HBox([self.data_dir_input]),
            self.num_tsteps_input,
            self.status_output,
            self.generate_viz_btn,
            self.viz_output
        ])
    
    def _setup_event_handlers(self):
        """Setup event handlers for widgets"""
        self.load_data_btn.on_click(self._load_data_handler)
        self.load_config_btn.on_click(self._load_config_handler)
        self.refresh_models_btn.on_click(self._refresh_models_handler)
        self.load_model_btn.on_click(self._load_model_handler)
        self.generate_viz_btn.on_click(self._generate_viz_handler)
    
    def _load_data_handler(self, button):
        """Handle processed data loading"""
        with self.status_output:
            clear_output(wait=True)
            try:
                data_dir = self.processed_data_input.value.strip()
                
                # Get n_channels from config if available
                n_channels = None
                if self.config:
                    n_channels = self.config.model.get('n_channels')
                    if n_channels is not None:
                        print(f"🔍 Model expects n_channels={n_channels}, filtering data files...")
                elif self.selected_model_info:
                    # Try to extract from model weights if config not loaded yet
                    encoder_file = self.selected_model_info.get('encoder')
                    if encoder_file:
                        n_channels = self._extract_n_channels_from_weights(encoder_file)
                        if n_channels is not None:
                            print(f"🔍 Model expects n_channels={n_channels} (from weights), filtering data files...")
                
                self.loaded_data = load_processed_data(data_dir=data_dir, n_channels=n_channels)
                if self.loaded_data:
                    print(f"✅ Processed data loaded from: {data_dir}")
                    print(f"   Training samples: {self.loaded_data['metadata'].get('num_train', 0)}")
                    print(f"   Evaluation samples: {self.loaded_data['metadata'].get('num_eval', 0)}")
                    loaded_n_channels = self.loaded_data['metadata'].get('n_channels')
                    if loaded_n_channels is not None:
                        print(f"   Channels: {loaded_n_channels}")
                        if n_channels is not None and loaded_n_channels != n_channels:
                            print(f"   ⚠️ Warning: Loaded data has {loaded_n_channels} channels, but model expects {n_channels}")
                else:
                    print(f"❌ No processed data found in: {data_dir}")
                    if n_channels is not None:
                        print(f"   Looking for files with n_channels={n_channels}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                import traceback
                traceback.print_exc()
    
    def _parse_model_filename(self, filename):
        """
        Parse model filename to extract hyperparameters.
        
        Args:
            filename: Model filename (e.g., 'e2co_encoder_grid_bs32_ld64_ns2_run0001_bs32_ld64_ns2.h5')
            
        Returns:
            Dict with batch_size, latent_dim, n_steps, run_id, component, or None if parsing fails
        """
        try:
            # Pattern: e2co_{component}_grid_bs{batch_size}_ld{latent_dim}_ns{n_steps}_run{run_id}...
            pattern = r'e2co_(encoder|decoder|transition)_grid_bs(\d+)_ld(\d+)_ns(\d+)_run(\d+)_bs\d+_ld\d+_ns\d+\.h5'
            match = re.match(pattern, filename)
            
            if match:
                component = match.group(1)
                batch_size = int(match.group(2))
                latent_dim = int(match.group(3))
                n_steps = int(match.group(4))
                run_id = match.group(5)
                
                return {
                    'component': component,
                    'batch_size': batch_size,
                    'latent_dim': latent_dim,
                    'n_steps': n_steps,
                    'run_id': run_id
                }
            return None
        except Exception as e:
            return None
    
    def _scan_available_models(self, model_dir='./saved_models/'):
        """
        Scan saved_models directory for available model files.
        
        Args:
            model_dir: Directory to scan for model files
            
        Returns:
            List of model sets, each containing encoder, decoder, transition files with matching hyperparameters
        """
        if not os.path.exists(model_dir):
            return []
        
        # Find all model files matching pattern
        encoder_files = glob.glob(os.path.join(model_dir, 'e2co_encoder_grid_*.h5'))
        decoder_files = glob.glob(os.path.join(model_dir, 'e2co_decoder_grid_*.h5'))
        transition_files = glob.glob(os.path.join(model_dir, 'e2co_transition_grid_*.h5'))
        
        # Also check grid_search subdirectory
        grid_search_dir = os.path.join(model_dir, 'grid_search')
        if os.path.exists(grid_search_dir):
            encoder_files.extend(glob.glob(os.path.join(grid_search_dir, 'e2co_encoder_grid_*.h5')))
            decoder_files.extend(glob.glob(os.path.join(grid_search_dir, 'e2co_decoder_grid_*.h5')))
            transition_files.extend(glob.glob(os.path.join(grid_search_dir, 'e2co_transition_grid_*.h5')))
        
        # Group models by composite key: (run_id, batch_size, latent_dim, n_steps)
        # This ensures models with same run_id but different hyperparameters are treated separately
        model_sets = {}
        
        for encoder_file in encoder_files:
            filename = os.path.basename(encoder_file)
            parsed = self._parse_model_filename(filename)
            if parsed:
                # Use composite key to uniquely identify each model set
                model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'])
                if model_key not in model_sets:
                    model_sets[model_key] = {
                        'run_id': parsed['run_id'],
                        'batch_size': parsed['batch_size'],
                        'latent_dim': parsed['latent_dim'],
                        'n_steps': parsed['n_steps'],
                        'encoder': None,
                        'decoder': None,
                        'transition': None
                    }
                model_sets[model_key]['encoder'] = encoder_file
        
        for decoder_file in decoder_files:
            filename = os.path.basename(decoder_file)
            parsed = self._parse_model_filename(filename)
            if parsed:
                model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'])
                if model_key in model_sets:
                    model_sets[model_key]['decoder'] = decoder_file
        
        for transition_file in transition_files:
            filename = os.path.basename(transition_file)
            parsed = self._parse_model_filename(filename)
            if parsed:
                model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'])
                if model_key in model_sets:
                    model_sets[model_key]['transition'] = transition_file
        
        # Filter to only complete sets (all three components)
        complete_sets = []
        for model_key, model_set in model_sets.items():
            if model_set['encoder'] and model_set['decoder'] and model_set['transition']:
                complete_sets.append(model_set)
        
        # Sort by run_id, then batch_size, latent_dim, n_steps
        complete_sets.sort(key=lambda x: (x['run_id'], x['batch_size'], x['latent_dim'], x['n_steps']))
        
        return complete_sets
    
    def _refresh_models_handler(self, button):
        """Handle model refresh button click"""
        with self.status_output:
            clear_output(wait=True)
            try:
                model_dir = "./saved_models/"
                print(f"🔍 Scanning for models in: {model_dir}")
                
                self.available_models = self._scan_available_models(model_dir)
                
                if self.available_models:
                    # Create dropdown options
                    options = []
                    for model_set in self.available_models:
                        label = (f"Run {model_set['run_id']} | "
                                f"bs={model_set['batch_size']}, "
                                f"ld={model_set['latent_dim']}, "
                                f"ns={model_set['n_steps']}")
                        options.append((label, model_set))
                    
                    self.model_selection.options = options
                    if options:
                        self.model_selection.value = options[0][1]  # Select first model
                    
                    print(f"✅ Found {len(self.available_models)} complete model set(s)")
                    for model_set in self.available_models:
                        print(f"   Run {model_set['run_id']}: bs={model_set['batch_size']}, "
                              f"ld={model_set['latent_dim']}, ns={model_set['n_steps']}")
                else:
                    self.model_selection.options = [("No models found", None)]
                    print(f"⚠️ No complete model sets found in {model_dir}")
                    print(f"   Looking for: e2co_encoder_grid_*.h5, e2co_decoder_grid_*.h5, e2co_transition_grid_*.h5")
                    
            except Exception as e:
                print(f"❌ Error scanning models: {e}")
                import traceback
                traceback.print_exc()
    
    def _extract_n_channels_from_weights(self, encoder_file):
        """
        Extract n_channels from encoder checkpoint weights.
        
        Args:
            encoder_file: Path to encoder checkpoint file
            
        Returns:
            n_channels (int) if successful, None otherwise
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            state_dict = torch.load(encoder_file, map_location=device, weights_only=False)
            if 'conv1.0.weight' in state_dict:
                n_channels = state_dict['conv1.0.weight'].shape[1]
                return n_channels
            return None
        except Exception as e:
            print(f"⚠️ Warning: Could not extract n_channels from weights: {e}")
            return None
    
    def _extract_latent_dim_from_weights(self, encoder_file):
        """
        Extract latent_dim from encoder checkpoint weights.
        
        Args:
            encoder_file: Path to encoder checkpoint file
            
        Returns:
            latent_dim (int) if successful, None otherwise
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            state_dict = torch.load(encoder_file, map_location=device, weights_only=False)
            if 'fc_mean.weight' in state_dict:
                latent_dim = state_dict['fc_mean.weight'].shape[0]
                return latent_dim
            return None
        except Exception as e:
            print(f"⚠️ Warning: Could not extract latent_dim from weights: {e}")
            return None
    
    def _update_config_from_model(self, model_info, latent_dim_from_weights=None):
        """
        Update config.yaml with parameters from selected model.
        Extracts n_channels and latent_dim from model weights and updates all related config parameters.
        
        Args:
            model_info: Dict containing model parameters (batch_size, latent_dim, n_steps, encoder path)
            latent_dim_from_weights: Optional latent_dim extracted from weights (if None, will extract here)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = self.config_path_input.value.strip()
            
            # Load current config
            config = Config(config_path)
            
            # Extract parameters from encoder weights
            encoder_file = model_info.get('encoder')
            n_channels = None
            latent_dim_actual = None
            
            if encoder_file:
                print("🔍 Extracting parameters from model weights...")
                
                # Extract n_channels from weights
                n_channels = self._extract_n_channels_from_weights(encoder_file)
                if n_channels is not None:
                    print(f"   Found n_channels: {n_channels}")
                else:
                    print("⚠️ Warning: Could not extract n_channels from weights. Using config value.")
                
                # Extract latent_dim from weights (use provided value or extract)
                if latent_dim_from_weights is not None:
                    latent_dim_actual = latent_dim_from_weights
                    print(f"   Found latent_dim: {latent_dim_actual} (from weights)")
                else:
                    latent_dim_actual = self._extract_latent_dim_from_weights(encoder_file)
                    if latent_dim_actual is not None:
                        print(f"   Found latent_dim: {latent_dim_actual} (from weights)")
                    else:
                        print("⚠️ Warning: Could not extract latent_dim from weights. Using filename value.")
                        latent_dim_actual = model_info['latent_dim']
            else:
                # Fallback to filename values if encoder file not available
                latent_dim_actual = model_info['latent_dim']
            
            # Update basic parameters - use actual values from weights
            config.set('model.latent_dim', latent_dim_actual)
            config.set('training.nsteps', model_info['n_steps'])
            config.set('training.batch_size', model_info['batch_size'])
            
            # Update n_channels related config if extracted
            if n_channels is not None:
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
            
            # Save config back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config.config, f, default_flow_style=False, indent=2, allow_unicode=True, sort_keys=False)
            
            # Reload config
            self.config = Config(config_path)
            
            print(f"✅ Config updated:")
            print(f"   model.latent_dim: {latent_dim_actual}")
            print(f"   training.nsteps: {model_info['n_steps']}")
            print(f"   training.batch_size: {model_info['batch_size']}")
            if encoder_file and n_channels is not None:
                print(f"   model.n_channels: {n_channels}")
                print(f"   data.input_shape[0]: {n_channels}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_config_handler(self, button):
        """Handle config file loading"""
        with self.status_output:
            clear_output(wait=True)
            try:
                config_path = self.config_path_input.value.strip()
                self.config = Config(config_path)
                print(f"✅ Config loaded from: {config_path}")
                
                # Set device
                device_config = self.config.runtime.get('device', 'auto')
                if device_config == 'auto':
                    self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
                else:
                    self.device = torch.device(device_config)
                print(f"   Device: {self.device}")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                import traceback
                traceback.print_exc()
    
    def _validate_model_config_match(self, model_info, config):
        """
        Validate that config parameters match model weights before loading.
        
        Args:
            model_info: Dict containing model parameters and file paths
            config: Config object to validate against
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            encoder_file = model_info.get('encoder')
            if not encoder_file or not os.path.exists(encoder_file):
                print("⚠️ Warning: Encoder file not found, skipping validation")
                return True
            
            # Extract n_channels from weights
            n_channels_from_weights = self._extract_n_channels_from_weights(encoder_file)
            if n_channels_from_weights is None:
                print("⚠️ Warning: Could not extract n_channels from weights, skipping validation")
                return True
            
            # Check n_channels match
            config_n_channels = config.model.get('n_channels')
            if config_n_channels is not None and config_n_channels != n_channels_from_weights:
                print(f"⚠️ Warning: n_channels mismatch detected!")
                print(f"   Config: {config_n_channels}, Model weights: {n_channels_from_weights}")
                return False
            
            # Check latent_dim match (extract from fc_mean.weight if possible)
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
                state_dict = torch.load(encoder_file, map_location=device, weights_only=False)
                if 'fc_mean.weight' in state_dict:
                    latent_dim_from_weights = state_dict['fc_mean.weight'].shape[0]
                    config_latent_dim = config.model.get('latent_dim')
                    if config_latent_dim is not None and config_latent_dim != latent_dim_from_weights:
                        print(f"⚠️ Warning: latent_dim mismatch detected!")
                        print(f"   Config: {config_latent_dim}, Model weights: {latent_dim_from_weights}")
                        return False
            except Exception as e:
                # If we can't extract latent_dim, that's okay, just skip that check
                pass
            
            print("✅ Config validation passed: parameters match model weights")
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Error during validation: {e}")
            # Don't fail on validation errors, just warn
            return True
    
    def _load_model_handler(self, button):
        """Handle model loading"""
        with self.status_output:
            clear_output(wait=True)
            try:
                # Get selected model
                selected_model = self.model_selection.value
                if not selected_model:
                    print("❌ Please select a model from the dropdown!")
                    return
                
                # Load config if not already loaded
                if not self.config:
                    config_path = self.config_path_input.value.strip()
                    self.config = Config(config_path)
                    
                    # Set device
                    device_config = self.config.runtime.get('device', 'auto')
                    if device_config == 'auto':
                        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
                    else:
                        self.device = torch.device(device_config)
                
                # Extract actual parameters from model weights BEFORE updating config
                encoder_file = selected_model.get('encoder')
                latent_dim_from_weights = None
                
                if encoder_file and os.path.exists(encoder_file):
                    print("🔍 Extracting parameters from model weights...")
                    latent_dim_from_weights = self._extract_latent_dim_from_weights(encoder_file)
                    if latent_dim_from_weights is not None:
                        print(f"   Found latent_dim in weights: {latent_dim_from_weights}")
                        # Check if it matches filename value
                        if latent_dim_from_weights != selected_model['latent_dim']:
                            print(f"   ⚠️ Note: Filename indicates latent_dim={selected_model['latent_dim']}, but weights have {latent_dim_from_weights}")
                            print(f"   Using value from weights: {latent_dim_from_weights}")
                    else:
                        print("⚠️ Warning: Could not extract latent_dim from weights. Using filename value.")
                        latent_dim_from_weights = selected_model['latent_dim']
                else:
                    print("⚠️ Warning: Encoder file not found. Using filename values.")
                    latent_dim_from_weights = selected_model['latent_dim']
                
                # Update config with model parameters (using actual values from weights)
                print("⚙️ Updating config with model parameters...")
                if not self._update_config_from_model(selected_model, latent_dim_from_weights=latent_dim_from_weights):
                    print("❌ Failed to update config!")
                    return
                
                # Store selected model info
                self.selected_model_info = selected_model
                
                from model.training.rom_wrapper import ROMWithE2C
                
                # Initialize model with updated config (which now matches weights)
                print("🔧 Initializing model...")
                self.my_rom = ROMWithE2C(self.config).to(self.device)
                
                # Load weights from specific files
                print(f"📦 Loading model weights...")
                print(f"   Encoder: {os.path.basename(selected_model['encoder'])}")
                print(f"   Decoder: {os.path.basename(selected_model['decoder'])}")
                print(f"   Transition: {os.path.basename(selected_model['transition'])}")
                
                # Load weights using the model's load_weights_from_file method
                try:
                    self.my_rom.model.load_weights_from_file(
                        selected_model['encoder'],
                        selected_model['decoder'],
                        selected_model['transition']
                    )
                    print("✅ Model loaded successfully!")
                    print(f"   Run ID: {selected_model['run_id']}")
                    print(f"   Batch size: {selected_model['batch_size']}, "
                          f"Latent dim: {latent_dim_from_weights}, "
                          f"N-steps: {selected_model['n_steps']}")
                except Exception as load_error:
                    print(f"❌ Failed to load model weights: {load_error}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
    
    def _generate_viz_handler(self, button):
        """Handle visualization generation"""
        with self.viz_output:
            clear_output(wait=True)
            
            if not self.loaded_data:
                print("❌ Please load processed data first!")
                return
            
            if not self.config:
                print("❌ Please load config first!")
                return
            
            if not self.my_rom:
                print("❌ Please load model first!")
                return
            
            try:
                # Validate that loaded data matches model's n_channels
                model_n_channels = self.config.model.get('n_channels')
                loaded_n_channels = self.loaded_data.get('metadata', {}).get('n_channels')
                
                if model_n_channels is not None and loaded_n_channels is not None:
                    if model_n_channels != loaded_n_channels:
                        print(f"⚠️ Channel mismatch detected!")
                        print(f"   Model expects: {model_n_channels} channels")
                        print(f"   Loaded data has: {loaded_n_channels} channels")
                        print(f"   Attempting to reload data with correct channel count...")
                        
                        # Try to reload data with correct n_channels
                        data_dir = self.processed_data_input.value.strip()
                        reloaded_data = load_processed_data(data_dir=data_dir, n_channels=model_n_channels)
                        
                        if reloaded_data:
                            self.loaded_data = reloaded_data
                            print(f"✅ Successfully reloaded data with {model_n_channels} channels")
                        else:
                            print(f"❌ Failed to find matching data file with {model_n_channels} channels")
                            print(f"   Please ensure processed data with n_channels={model_n_channels} exists")
                            return
                    else:
                        print(f"✅ Channel validation passed: {model_n_channels} channels")
                
                print("=" * 70)
                print("🎨 Generating Test Visualization")
                print("=" * 70)
                
                data_dir = self.data_dir_input.value.strip()
                num_tstep = self.num_tsteps_input.value
                
                visualization_dashboard = generate_test_visualization_standalone(
                    loaded_data=self.loaded_data,
                    my_rom=self.my_rom,
                    device=self.device,
                    data_dir=data_dir,
                    num_tstep=num_tstep
                )
                
                if visualization_dashboard:
                    print("\n✅ Visualization dashboard created successfully!")
                else:
                    print("\n❌ Failed to create visualization dashboard")
                    
            except Exception as e:
                print(f"❌ Error generating visualization: {e}")
                import traceback
                traceback.print_exc()
    
    def display(self):
        """Display the dashboard"""
        if not WIDGETS_AVAILABLE:
            print("⚠️ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
            return None
        display(self.main_widget)
        return self.main_widget


def create_testing_dashboard():
    """
    Create and display the testing dashboard
    
    Returns:
        TestingDashboard instance
    """
    dashboard = TestingDashboard()
    dashboard.display()
    return dashboard

