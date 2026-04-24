"""
Testing Dashboard for E2C Model
Interactive dashboard for loading models and generating test visualizations
"""

import os
import torch
import glob
import re
import yaml
from pathlib import Path

_ROM_DIR = str(Path(__file__).resolve().parent.parent)

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


def _win_long_path(path: str) -> str:
    """Prefix absolute paths with ``\\\\?\\`` on Windows to bypass the 260-char MAX_PATH limit."""
    if os.name == 'nt' and path and len(path) >= 260 and not path.startswith('\\\\?\\'):
        return '\\\\?\\' + os.path.abspath(path)
    return path


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
            value=os.path.join(_ROM_DIR, "processed_data"),
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
            value=os.path.join(_ROM_DIR, "config.yaml"),
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
            value=os.path.join(_ROM_DIR, "sr3_batch_output"),
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
        
        Supports two formats:
          New: e2co_{component}_grid_run0001_bs16_ld128_ns2_ch4_schfix_rb3_...h5
          Old: e2co_{component}_grid_bs16_ld128_ns2_ch4_run0001_..._bs16_ld128_ns2_ch4.h5
            
        Returns:
            Dict with component, batch_size, latent_dim, n_steps, n_channels, run_id, 
            residual_blocks (if present), encoder_hidden_dims (if present), or None if parsing fails
        """
        try:
            # New compact format: e2co_{component}_grid_{run_id}.h5
            # where run_id = run{N}_bs{}_ld{}_ns{}_ch{}_sch{}_rb{}_...
            new_pattern = r'e2co_(encoder|decoder|transition)_grid_(run\d+_.+)\.h5'
            new_match = re.match(new_pattern, filename)
            
            if new_match:
                component = new_match.group(1)
                full_run_id = new_match.group(2)
                
                # Extract params from within the run_id
                bs_m = re.search(r'_bs(\d+)', full_run_id)
                ld_m = re.search(r'_ld(\d+)', full_run_id)
                ns_m = re.search(r'_ns(\d+)', full_run_id)
                ch_m = re.search(r'_ch(\d+)', full_run_id)
                run_m = re.match(r'run(\d+)', full_run_id)
                
                if bs_m and ld_m and ns_m and run_m:
                    result = {
                        'component': component,
                        'batch_size': int(bs_m.group(1)),
                        'latent_dim': int(ld_m.group(1)),
                        'n_steps': int(ns_m.group(1)),
                        'n_channels': int(ch_m.group(1)) if ch_m else None,
                        'run_id': full_run_id,
                        'run_number': run_m.group(1),
                    }
                    
                    rb_m = re.search(r'_rb(\d+)', full_run_id)
                    if rb_m:
                        result['residual_blocks'] = int(rb_m.group(1))
                    
                    ehd_m = re.search(r'_ehd([\d-]+)', full_run_id)
                    if ehd_m:
                        try:
                            result['encoder_hidden_dims'] = [int(d) for d in ehd_m.group(1).split('-')]
                        except ValueError:
                            pass
                    
                    return result
            
            # Old format with triple-redundant params:
            # e2co_{component}_grid_bs{}_ld{}_ns{}_ch{}_run{}_..._bs{}_ld{}_ns{}_ch{}.h5
            old_pattern = r'e2co_(encoder|decoder|transition)_grid_bs(\d+)_ld(\d+)_ns(\d+)(?:_ch(\d+))?_run(\d+)((?:_[^_]+)*)_bs(\d+)_ld(\d+)_ns(\d+)(?:_ch(\d+))?\.h5'
            match = re.match(old_pattern, filename)
            
            if match:
                component = match.group(1)
                batch_size_final = int(match.group(8))
                latent_dim_final = int(match.group(9))
                n_steps_final = int(match.group(10))
                
                n_channels_first = match.group(5)
                n_channels_final = match.group(11)
                n_channels = int(n_channels_final) if n_channels_final else (int(n_channels_first) if n_channels_first else None)
                
                run_number = match.group(6)
                run_id_content = match.group(7)
                full_run_id = f"run{run_number}{run_id_content}"
                
                result = {
                    'component': component,
                    'batch_size': batch_size_final,
                    'latent_dim': latent_dim_final,
                    'n_steps': n_steps_final,
                    'n_channels': n_channels,
                    'run_id': full_run_id,
                    'run_number': run_number,
                }
                
                rb_m = re.search(r'_rb(\d+)', run_id_content)
                if rb_m:
                    result['residual_blocks'] = int(rb_m.group(1))
                
                ehd_m = re.search(r'_ehd([\d-]+)', run_id_content)
                if ehd_m:
                    try:
                        result['encoder_hidden_dims'] = [int(d) for d in ehd_m.group(1).split('-')]
                    except ValueError:
                        pass
                
                return result
            
            # Fallback: simple old pattern without channels
            pattern_simple = r'e2co_(encoder|decoder|transition)_grid_bs(\d+)_ld(\d+)_ns(\d+)_run(\d+)(?:_[^_]+)*\.h5'
            match_simple = re.match(pattern_simple, filename)
            
            if match_simple:
                return {
                    'component': match_simple.group(1),
                    'batch_size': int(match_simple.group(2)),
                    'latent_dim': int(match_simple.group(3)),
                    'n_steps': int(match_simple.group(4)),
                    'n_channels': None,
                    'run_id': f"run{match_simple.group(5)}",
                    'run_number': match_simple.group(5),
                }
            
            return None
        except Exception as e:
            return None
    
    def _scan_available_models(self, model_dir=None):
        """
        Scan saved_models directory for available model files.
        
        Args:
            model_dir: Directory to scan for model files
            
        Returns:
            List of model sets, each containing encoder, decoder, and optionally transition files with matching hyperparameters
        """
        if model_dir is None:
            model_dir = os.path.join(_ROM_DIR, 'saved_models')
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
        
        # Group models by composite key: (run_id, batch_size, latent_dim, n_steps, n_channels)
        # This ensures models with same run_id but different hyperparameters are treated separately
        # Include n_channels in key to distinguish models with different channel counts
        model_sets = {}
        
        for encoder_file in encoder_files:
            filename = os.path.basename(encoder_file)
            parsed = self._parse_model_filename(filename)
            if parsed:
                # Use composite key to uniquely identify each model set
                # Include n_channels (use None as placeholder if not present) to distinguish models
                n_channels = parsed.get('n_channels')
                model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'], n_channels)
                if model_key not in model_sets:
                    model_sets[model_key] = {
                        'run_id': parsed['run_id'],
                        'batch_size': parsed['batch_size'],
                        'latent_dim': parsed['latent_dim'],
                        'n_steps': parsed['n_steps'],
                        'n_channels': n_channels,
                        'residual_blocks': parsed.get('residual_blocks'),
                        'encoder_hidden_dims': parsed.get('encoder_hidden_dims'),
                        'encoder': None,
                        'decoder': None,
                        'transition': None
                    }
                model_sets[model_key]['encoder'] = _win_long_path(encoder_file)
        
        for decoder_file in decoder_files:
            filename = os.path.basename(decoder_file)
            parsed = self._parse_model_filename(filename)
            if parsed:
                n_channels = parsed.get('n_channels')
                model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'], n_channels)
                if model_key in model_sets:
                    model_sets[model_key]['decoder'] = _win_long_path(decoder_file)
        
        for transition_file in transition_files:
            filename = os.path.basename(transition_file)
            parsed = self._parse_model_filename(filename)
            if parsed:
                n_channels = parsed.get('n_channels')
                model_key = (parsed['run_id'], parsed['batch_size'], parsed['latent_dim'], parsed['n_steps'], n_channels)
                if model_key in model_sets:
                    model_sets[model_key]['transition'] = _win_long_path(transition_file)
        
        # Filter to sets with at least encoder and decoder (transition is optional)
        complete_sets = []
        for model_key, model_set in model_sets.items():
            if model_set['encoder'] and model_set['decoder']:
                complete_sets.append(model_set)
        
        # Sort by run_id, then batch_size, latent_dim, n_steps, n_channels
        complete_sets.sort(key=lambda x: (
            x.get('run_id', ''),
            x.get('batch_size', 0),
            x.get('latent_dim', 0),
            x.get('n_steps', 0),
            x.get('n_channels') if x.get('n_channels') is not None else 0
        ))
        
        return complete_sets
    
    def _refresh_models_handler(self, button):
        """Handle model refresh button click"""
        with self.status_output:
            clear_output(wait=True)
            try:
                model_dir = os.path.join(_ROM_DIR, 'saved_models')
                print(f"🔍 Scanning for models in: {model_dir}")
                
                self.available_models = self._scan_available_models(model_dir)
                
                if self.available_models:
                    # Create dropdown options with rich structured labels
                    options = []
                    for model_set in self.available_models:
                        run_id = model_set['run_id']
                        n_channels_str = f", ch={model_set['n_channels']}" if model_set.get('n_channels') is not None else ""

                        trn_match = re.search(r'_trn([A-Z0-9_]+)', run_id)
                        transition = trn_match.group(1) if trn_match else 'LINEAR'

                        if '_memT' in run_id or '_mem-' in run_id:
                            encoding = 'Multi-Embed'
                        elif '_gnn' in run_id:
                            encoding = 'GNN'
                        elif '_fno' in run_id:
                            encoding = 'FNO'
                        elif '_mm' in run_id:
                            encoding = 'Multimodal'
                        else:
                            encoding = 'Standard'

                        label = (f"{transition} | {encoding} | "
                                f"ld={model_set['latent_dim']}, ns={model_set['n_steps']}, "
                                f"bs={model_set['batch_size']}{n_channels_str}")
                        options.append((label, model_set))
                    
                    self.model_selection.options = options
                    if options:
                        self.model_selection.value = options[0][1]  # Select first model
                    
                    print(f"✅ Found {len(self.available_models)} model set(s)")
                    for model_set in self.available_models:
                        run_id = model_set['run_id']
                        transition_status = "✓" if model_set.get('transition') else "⚠️ (missing)"
                        n_channels_str = f", ch={model_set['n_channels']}" if model_set.get('n_channels') is not None else ""
                        trn_match = re.search(r'_trn([A-Z0-9_]+)', run_id)
                        transition = trn_match.group(1) if trn_match else 'LINEAR'
                        if '_memT' in run_id or '_mem-' in run_id:
                            enc = 'MEM'
                        elif '_gnn' in run_id:
                            enc = 'GNN'
                        elif '_fno' in run_id:
                            enc = 'FNO'
                        elif '_mm' in run_id:
                            enc = 'MM'
                        else:
                            enc = 'Std'
                        print(f"   {transition} | {enc} | ld={model_set['latent_dim']}, "
                              f"ns={model_set['n_steps']}, bs={model_set['batch_size']}{n_channels_str}, "
                              f"transition={transition_status}")
                else:
                    self.model_selection.options = [("No models found", None)]
                    print(f"⚠️ No model sets found in {model_dir}")
                    print(f"   Looking for: e2co_encoder_grid_*.h5, e2co_decoder_grid_*.h5")
                    print(f"   (e2co_transition_grid_*.h5 is optional)")
                    
            except Exception as e:
                print(f"❌ Error scanning models: {e}")
                import traceback
                traceback.print_exc()
    
    @staticmethod
    def _resolve_encoder_state_dict(payload):
        """
        Given a raw encoder checkpoint payload, return the state_dict for
        a single encoder branch (preferring the dynamic encoder for
        multimodal / GNN payloads) and whether the payload is multimodal/GNN.
        
        Returns:
            (state_dict, is_multimodal_or_gnn)
        """
        if isinstance(payload, dict):
            # Multi-embedding: encoders dict keyed by branch name
            if payload.get('_multi_embedding', False):
                encoders = payload.get('encoders', {})
                if encoders:
                    # Prefer a dynamic-branch encoder if branch metadata is available
                    branches = payload.get('branches', []) or []
                    dynamic_names = [b.get('name') for b in branches if b.get('role') == 'dynamic']
                    for name in dynamic_names:
                        if name in encoders:
                            return encoders[name], True
                    # Otherwise return the first encoder we can find
                    first_key = next(iter(encoders))
                    return encoders[first_key], True
            if payload.get('_multimodal', False) or payload.get('_gnn', False) or payload.get('_fno', False):
                if 'dynamic_encoder' in payload:
                    return payload['dynamic_encoder'], True
                if 'static_encoder' in payload:
                    return payload['static_encoder'], True
            if any(k.startswith('conv1.') or k.startswith('fc_mean.') for k in payload):
                return payload, False
        return payload, False

    def _extract_n_channels_from_weights(self, encoder_file):
        """
        Extract n_channels from encoder checkpoint weights.
        Handles standard, multimodal, and GNN encoder payloads.
        
        Returns:
            n_channels (int) if successful, None otherwise
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            payload = torch.load(_win_long_path(encoder_file), map_location=device, weights_only=False)
            # Multi-embedding payload: 4 channels by construction
            if isinstance(payload, dict) and payload.get('_multi_embedding'):
                branches = payload.get('branches', []) or []
                if branches:
                    seen = []
                    for b in branches:
                        seen.extend(b.get('channels', []) or [])
                    if seen:
                        return max(seen) + 1
                return 4
            state_dict, is_mm = self._resolve_encoder_state_dict(payload)
            if is_mm:
                # For multimodal the total n_channels = static_channels + dynamic_channels.
                # Each branch only sees its own channels, so we read from the filename
                # or config instead. Return None to signal "use config value".
                return None
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
        Handles standard, multimodal, and GNN encoder payloads.
        
        For multimodal models the total latent_dim = static_latent_dim + dynamic_latent_dim.
        
        Returns:
            latent_dim (int) if successful, None otherwise
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            payload = torch.load(_win_long_path(encoder_file), map_location=device, weights_only=False)

            # Standard encoder
            if isinstance(payload, dict) and 'fc_mean.weight' in payload:
                return payload['fc_mean.weight'].shape[0]

            # Multi-embedding encoder — sum every branch's latent dim
            if isinstance(payload, dict) and payload.get('_multi_embedding'):
                # Trust the embedded branch metadata when present
                branches = payload.get('branches', []) or []
                if branches:
                    return sum(int(b.get('latent_dim', 0)) for b in branches)
                # Otherwise inspect each encoder state dict
                total = 0
                for name, sd in (payload.get('encoders') or {}).items():
                    if 'fc_mean.weight' in sd:
                        total += sd['fc_mean.weight'].shape[0]
                    elif 'fc.weight' in sd:
                        total += sd['fc.weight'].shape[0]
                if total > 0:
                    return total

            # Multimodal / GNN encoder — sum branch latent dims
            if isinstance(payload, dict) and (payload.get('_multimodal') or payload.get('_gnn') or payload.get('_fno')):
                total = 0
                for branch in ('static_encoder', 'dynamic_encoder'):
                    branch_sd = payload.get(branch, {})
                    if 'fc_mean.weight' in branch_sd:
                        total += branch_sd['fc_mean.weight'].shape[0]
                    elif 'fc.weight' in branch_sd:
                        total += branch_sd['fc.weight'].shape[0]
                    elif 'grid_pool.fc.weight' in branch_sd:
                        total += branch_sd['grid_pool.fc.weight'].shape[0]
                    elif 'pool.proj.weight' in branch_sd:
                        total += branch_sd['pool.proj.weight'].shape[0]
                if total > 0:
                    return total

            return None
        except Exception as e:
            print(f"⚠️ Warning: Could not extract latent_dim from weights: {e}")
            return None
    
    def _extract_vae_enabled_from_weights(self, encoder_file):
        """
        Detect if encoder was trained with VAE mode (has fc_logvar layer).
        Handles standard, multimodal, and GNN encoder payloads.
        
        Returns:
            True if VAE is enabled, False otherwise
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            payload = torch.load(_win_long_path(encoder_file), map_location=device, weights_only=False)
            state_dict, _ = self._resolve_encoder_state_dict(payload)
            return 'fc_logvar.weight' in state_dict
        except Exception as e:
            print(f"⚠️ Warning: Could not detect VAE mode from weights: {e}")
            return False
    
    def _extract_encoder_hidden_dims_from_weights(self, transition_file):
        """
        Extract encoder_hidden_dims from transition model checkpoint weights.
        Handles both linear transitions (prefix 'trans_encoder') and CLRU
        transitions (prefix 'selector').
        
        Returns:
            List of hidden dimensions (e.g., [200, 200]) if successful, None otherwise
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            state_dict = torch.load(_win_long_path(transition_file), map_location=device, weights_only=False)
            
            # Try both key prefixes: 'trans_encoder' for linear, 'selector' for CLRU
            for prefix in ('trans_encoder', 'selector'):
                hidden_dims = []
                layer_idx = 0
                
                while True:
                    weight_key = f'{prefix}.{layer_idx}.0.weight'
                    if weight_key not in state_dict:
                        break
                    
                    weight_shape = state_dict[weight_key].shape
                    out_features = weight_shape[0]
                    
                    next_weight_key = f'{prefix}.{layer_idx + 1}.0.weight'
                    if next_weight_key in state_dict:
                        hidden_dims.append(out_features)
                    
                    layer_idx += 1
                
                if hidden_dims:
                    return hidden_dims
            
            return None
                
        except Exception as e:
            print(f"⚠️ Warning: Could not extract encoder_hidden_dims from weights: {e}")
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
            
            # Track if VAE mode is detected
            vae_enabled = False
            
            if encoder_file:
                print("🔍 Extracting parameters from model weights...")
                
                # Extract n_channels from weights
                n_channels = self._extract_n_channels_from_weights(encoder_file)
                if n_channels is not None:
                    print(f"   Found n_channels: {n_channels}")
                else:
                    print("⚠️ Warning: Could not extract n_channels from weights. Using config value.")
                
                # Detect if VAE mode is enabled in the weights
                vae_enabled = self._extract_vae_enabled_from_weights(encoder_file)
                if vae_enabled:
                    print(f"   Found VAE mode: enabled (fc_logvar layer detected)")
                
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
            
            # Update residual_blocks if present in model_info and not None
            if 'residual_blocks' in model_info and model_info['residual_blocks'] is not None:
                if 'encoder' not in config.config:
                    config.config['encoder'] = {}
                config.config['encoder']['residual_blocks'] = model_info['residual_blocks']
                print(f"   encoder.residual_blocks: {model_info['residual_blocks']}")
            
            # Update encoder_hidden_dims - extract from transition weights if not in model_info
            encoder_hidden_dims = None
            if 'encoder_hidden_dims' in model_info and model_info['encoder_hidden_dims'] is not None:
                encoder_hidden_dims = model_info['encoder_hidden_dims']
            else:
                # Try to extract from transition checkpoint weights
                transition_file = model_info.get('transition')
                if transition_file and os.path.exists(transition_file):
                    encoder_hidden_dims = self._extract_encoder_hidden_dims_from_weights(transition_file)
                    if encoder_hidden_dims:
                        print(f"   Found encoder_hidden_dims from weights: {encoder_hidden_dims}")
            
            if encoder_hidden_dims is not None:
                if 'transition' not in config.config:
                    config.config['transition'] = {}
                config.config['transition']['encoder_hidden_dims'] = encoder_hidden_dims
                print(f"   transition.encoder_hidden_dims: {encoder_hidden_dims}")
            
            # Detect model type from encoder weights or filename
            mem_detected = False
            mem_branches_payload = None
            gnn_detected = False
            fno_detected = False
            multimodal_detected = False
            enc_payload = None
            if encoder_file:
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    enc_payload = torch.load(_win_long_path(encoder_file), map_location=device, weights_only=False)
                    if isinstance(enc_payload, dict):
                        if enc_payload.get('_multi_embedding', False):
                            mem_detected = True
                            mem_branches_payload = enc_payload.get('branches', None)
                        elif enc_payload.get('_gnn', False):
                            gnn_detected = True
                        elif enc_payload.get('_fno', False):
                            fno_detected = True
                        elif enc_payload.get('_multimodal', False):
                            multimodal_detected = True
                except Exception:
                    pass
                fname = os.path.basename(encoder_file)
                if not mem_detected and ('_memT' in fname or '_mem-' in fname):
                    mem_detected = True
                if not mem_detected and not gnn_detected and '_gnnT' in fname:
                    gnn_detected = True
                if not mem_detected and not fno_detected and not gnn_detected and '_fnoT' in fname:
                    fno_detected = True
                if (not mem_detected and not multimodal_detected and not gnn_detected
                        and not fno_detected and '_mmT' in fname):
                    multimodal_detected = True
            # Multi-embedding takes priority and forces all other flags off.
            if mem_detected:
                gnn_detected = False
                fno_detected = False
                multimodal_detected = False

            # Detect transition type from actual transition weights (not filename)
            transition_type_detected = None
            transition_file_for_detection = model_info.get('transition')
            if transition_file_for_detection and os.path.exists(transition_file_for_detection):
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    trans_sd = torch.load(_win_long_path(transition_file_for_detection), map_location=device, weights_only=False)
                    has_selector = any(k.startswith('selector.') for k in trans_sd)
                    has_nu_layer = 'nu_layer.weight' in trans_sd
                    has_alpha_layer = 'alpha_layer.weight' in trans_sd
                    has_trans_encoder = any(k.startswith('trans_encoder.') for k in trans_sd)
                    has_At_layer = 'At_layer.weight' in trans_sd
                    has_ode_func = any(k.startswith('ode_func.net.') for k in trans_sd)
                    has_C_D_params = 'C' in trans_sd and 'D' in trans_sd
                    has_V_real = 'V_real' in trans_sd
                    has_U_real = 'U_real_layer.weight' in trans_sd
                    has_K_param = 'K' in trans_sd and not has_At_layer
                    has_A_ct = 'A_skew_params' in trans_sd
                    has_Kt_layer = 'Kt_layer.weight' in trans_sd
                    has_Lt_layer = 'Lt_layer.weight' in trans_sd
                    has_eig_raw = 'eig_raw' in trans_sd or 'eig_mag_raw' in trans_sd
                    has_A_log = 'A_log' in trans_sd
                    has_delta_proj = 'delta_proj.weight' in trans_sd
                    has_out_proj = 'out_proj.weight' in trans_sd
                    has_gru_cells = any(k.startswith('gru_cells.') for k in trans_sd)
                    has_lstm_cells = any(k.startswith('lstm_cells.') for k in trans_sd)
                    has_H_net = any(k.startswith('H_net.') for k in trans_sd)
                    has_B_ctrl = 'B_ctrl' in trans_sd
                    has_spectral_gates = any(k.startswith('spectral_gates.') for k in trans_sd)
                    has_rnn_lambda = any(k.startswith('rnn_lambda_real.') for k in trans_sd)
                    has_ren_H = 'ren_H' in trans_sd
                    has_aft_W_q = any(k.startswith('aft.aft_W_q') for k in trans_sd)
                    has_U_skew = 'U_skew_params' in trans_sd
                    has_sigma_raw = 'sigma_raw' in trans_sd
                    has_A_bilinear = 'A_bilinear' in trans_sd
                    has_N_bilinear = 'N_bilinear' in trans_sd or any(k.startswith('N_P') for k in trans_sd)
                    has_lift_net = any(k.startswith('lift_net.') for k in trans_sd)
                    has_exp_r = 'exp_r_real' in trans_sd
                    has_sindy_coeff = 'sindy_coefficients' in trans_sd
                    has_cde_func = any(k.startswith('cde_func.net.') for k in trans_sd)
                    has_drift_net = any(k.startswith('drift_net.') for k in trans_sd)
                    has_diffusion_net = any(k.startswith('diffusion_net.') for k in trans_sd)
                    has_temporal_transformer = any(k.startswith('temporal_transformer.') for k in trans_sd)
                    has_branch_net = any(k.startswith('branch_net.') for k in trans_sd)
                    has_trunk_net = any(k.startswith('trunk_net.') for k in trans_sd)
                    if has_sindy_coeff:
                        transition_type_detected = 'sindy'
                    elif has_cde_func:
                        transition_type_detected = 'neural_cde'
                    elif has_drift_net and has_diffusion_net:
                        transition_type_detected = 'latent_sde'
                    elif has_temporal_transformer:
                        transition_type_detected = 'transformer'
                    elif has_branch_net and has_trunk_net:
                        transition_type_detected = 'deeponet'
                    elif has_spectral_gates and has_rnn_lambda:
                        transition_type_detected = 'skolr'
                    elif has_ren_H:
                        transition_type_detected = 'ren'
                    elif has_K_param and has_aft_W_q:
                        transition_type_detected = 'koopman_aft'
                    elif has_U_skew and has_sigma_raw:
                        transition_type_detected = 'dissipative_koopman'
                    elif has_A_bilinear and has_N_bilinear:
                        transition_type_detected = 'bilinear_koopman'
                    elif has_lift_net and has_exp_r:
                        transition_type_detected = 'isfno'
                    elif has_A_ct:
                        transition_type_detected = 'ct_koopman'
                    elif has_H_net and has_B_ctrl:
                        transition_type_detected = 'hamiltonian'
                    elif has_lstm_cells:
                        transition_type_detected = 'lstm'
                    elif has_gru_cells:
                        transition_type_detected = 'gru'
                    elif has_A_log and has_delta_proj and has_out_proj:
                        transition_type_detected = 'mamba2'
                    elif has_A_log and has_delta_proj:
                        transition_type_detected = 'mamba'
                    elif has_selector and has_Kt_layer and has_Lt_layer:
                        transition_type_detected = 'deep_koopman'
                    elif has_eig_raw and not has_selector:
                        transition_type_detected = 'stable_koopman'
                    elif has_K_param and not has_selector:
                        transition_type_detected = 'koopman'
                    elif has_selector and has_alpha_layer and has_V_real:
                        transition_type_detected = 's5'
                    elif has_selector and has_alpha_layer and has_U_real:
                        transition_type_detected = 's4d_dplr'
                    elif has_selector and has_alpha_layer:
                        transition_type_detected = 's4d'
                    elif has_selector and has_nu_layer:
                        transition_type_detected = 'clru'
                    elif has_ode_func:
                        transition_type_detected = 'nonlinear'
                    elif has_trans_encoder and has_At_layer:
                        transition_type_detected = 'linear'
                except Exception:
                    pass
            # Fallback to model_info.transition_type (e.g. from grid search results)
            if transition_type_detected is None and model_info.get('transition_type'):
                transition_type_detected = model_info['transition_type']
            # Fallback to filename if weights inspection failed
            if transition_type_detected is None and encoder_file:
                fname = os.path.basename(encoder_file)
                if '_trnSKOLR' in fname:
                    transition_type_detected = 'skolr'
                elif '_trnREN' in fname:
                    transition_type_detected = 'ren'
                elif '_trnKOOPMAN_AFT' in fname or '_trnKOOPMAN-AFT' in fname:
                    transition_type_detected = 'koopman_aft'
                elif '_trnDISSIPATIVE_KOOPMAN' in fname or '_trnDISSIPATIVE-KOOPMAN' in fname:
                    transition_type_detected = 'dissipative_koopman'
                elif '_trnBILINEAR_KOOPMAN' in fname or '_trnBILINEAR-KOOPMAN' in fname:
                    transition_type_detected = 'bilinear_koopman'
                elif '_trnISFNO' in fname:
                    transition_type_detected = 'isfno'
                elif '_trnSINDY' in fname:
                    transition_type_detected = 'sindy'
                elif '_trnNEURAL_CDE' in fname or '_trnNEURAL-CDE' in fname:
                    transition_type_detected = 'neural_cde'
                elif '_trnLATENT_SDE' in fname or '_trnLATENT-SDE' in fname:
                    transition_type_detected = 'latent_sde'
                elif '_trnTRANSFORMER' in fname:
                    transition_type_detected = 'transformer'
                elif '_trnDEEPONET' in fname:
                    transition_type_detected = 'deeponet'
                elif '_trnCT_KOOPMAN' in fname or '_trnCT-KOOPMAN' in fname:
                    transition_type_detected = 'ct_koopman'
                elif '_trnDEEP_KOOPMAN' in fname or '_trnDEEP-KOOPMAN' in fname:
                    transition_type_detected = 'deep_koopman'
                elif '_trnSTABLE_KOOPMAN' in fname or '_trnSTABLE-KOOPMAN' in fname:
                    transition_type_detected = 'stable_koopman'
                elif '_trnKOOPMAN' in fname:
                    transition_type_detected = 'koopman'
                elif '_trnMAMBA2' in fname:
                    transition_type_detected = 'mamba2'
                elif '_trnMAMBA' in fname:
                    transition_type_detected = 'mamba'
                elif '_trnS5' in fname:
                    transition_type_detected = 's5'
                elif '_trnS4D_DPLR' in fname or '_trnS4D-DPLR' in fname:
                    transition_type_detected = 's4d_dplr'
                elif '_trnS4D' in fname:
                    transition_type_detected = 's4d'
                elif '_trnCLRU' in fname:
                    transition_type_detected = 'clru'
                elif '_trnGRU' in fname:
                    transition_type_detected = 'gru'
                elif '_trnLSTM' in fname:
                    transition_type_detected = 'lstm'
                elif '_trnHAMILTONIAN' in fname:
                    transition_type_detected = 'hamiltonian'
                elif '_trnNONLINEAR' in fname or '_trnNonlinear' in fname:
                    transition_type_detected = 'nonlinear'
                elif '_trnlinear' in fname or '_trnLinear' in fname:
                    transition_type_detected = 'linear'

            if mem_detected:
                # Multi-Embedding Multimodal: highest priority. Force every
                # other model flag off and copy the branch metadata from the
                # encoder payload into the config so MultiEmbeddingMultimodal
                # can build the right per-branch encoders/decoders.
                config.config.setdefault('multi_embedding', {})['enable'] = True
                if mem_branches_payload:
                    config.config['multi_embedding']['branches'] = mem_branches_payload
                config.config.setdefault('gnn', {})['enable'] = False
                config.config.setdefault('fno', {})['enable'] = False
                config.config.setdefault('multimodal', {})['enable'] = False
                # Auto-enable invertibility loss when any branch uses FNO
                branches = config.config['multi_embedding'].get('branches', []) or []
                if any(b.get('encoder', {}).get('type') == 'fno'
                       or (b.get('decoder') and b['decoder'].get('type') == 'fno')
                       for b in branches):
                    config.config.setdefault('loss', {})['enable_invertibility_loss'] = True
                print(f"   Multi-Embedding model detected: multi_embedding.enable=True "
                      f"(branches: {[b.get('name') for b in branches]})")
            elif gnn_detected:
                if 'gnn' not in config.config:
                    config.config['gnn'] = {}
                config.config['gnn']['enable'] = True
                if 'multimodal' not in config.config:
                    config.config['multimodal'] = {}
                config.config['multimodal']['enable'] = False
                config.config.setdefault('fno', {})['enable'] = False
                config.config.setdefault('multi_embedding', {})['enable'] = False
                if 'loss' not in config.config:
                    config.config['loss'] = {}
                config.config['loss']['enable_inactive_masking'] = True
                print(f"   GNN model detected: gnn.enable=True, multimodal.enable=False")
            elif fno_detected:
                config.config.setdefault('fno', {})['enable'] = True
                config.config.setdefault('gnn', {})['enable'] = False
                config.config.setdefault('multimodal', {})['enable'] = False
                config.config.setdefault('multi_embedding', {})['enable'] = False
                config.config.setdefault('loss', {})['enable_invertibility_loss'] = True
                print(f"   FNO model detected: fno.enable=True")
            elif multimodal_detected:
                config.config.setdefault('gnn', {})['enable'] = False
                config.config.setdefault('fno', {})['enable'] = False
                config.config.setdefault('multimodal', {})['enable'] = True
                config.config.setdefault('multi_embedding', {})['enable'] = False
                print(f"   Multimodal model detected: multimodal.enable=True")
            else:
                config.config.setdefault('gnn', {})['enable'] = False
                config.config.setdefault('fno', {})['enable'] = False
                config.config.setdefault('multimodal', {})['enable'] = False
                config.config.setdefault('multi_embedding', {})['enable'] = False

            # Update transition type if detected from filename
            if transition_type_detected:
                if 'transition' not in config.config:
                    config.config['transition'] = {}
                config.config['transition']['type'] = transition_type_detected
                print(f"   Transition type detected: {transition_type_detected}")

            # Update VAE configuration based on detection from weights
            if 'model' not in config.config:
                config.config['model'] = {}
            if 'loss' not in config.config:
                config.config['loss'] = {}
                
            if vae_enabled:
                config.config['model']['enable_vae'] = True
                config.config['loss']['enable_kl_loss'] = True
                print(f"   model.enable_vae: True")
                print(f"   loss.enable_kl_loss: True")
            else:
                config.config['model']['enable_vae'] = False
                config.config['loss']['enable_kl_loss'] = False
            
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
                state_dict = torch.load(_win_long_path(encoder_file), map_location=device, weights_only=False)
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
                
                transition_file = selected_model.get('transition')
                if transition_file:
                    print(f"   Transition: {os.path.basename(transition_file)}")
                else:
                    print(f"   Transition: ⚠️ Not found (will use randomly initialized transition model)")
                
                # Load weights using the model's load_weights_from_file method (handles VAE detection)
                try:
                    # Use the model's load_weights_from_file which handles VAE layer detection
                    if transition_file and os.path.exists(transition_file):
                        self.my_rom.model.load_weights_from_file(
                            selected_model['encoder'],
                            selected_model['decoder'],
                            transition_file
                        )
                    else:
                        # Load encoder and decoder only
                        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
                        encoder_state = torch.load(_win_long_path(selected_model['encoder']), map_location=device, weights_only=False)
                        decoder_state = torch.load(_win_long_path(selected_model['decoder']), map_location=device, weights_only=False)
                        self.my_rom.model.encoder.load_state_dict(encoder_state)
                        self.my_rom.model.decoder.load_state_dict(decoder_state)
                        print("   ⚠️ Transition model weights not found - using randomly initialized transition model")
                    
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

