"""
Training Dashboard for E2C Model
Interactive dashboard for configuring and starting model training
"""

import os
import yaml
import torch
from datetime import datetime

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
from utilities.timing import Timer, collect_training_metadata
from data_preprocessing import load_processed_data


class TrainingDashboard:
    """
    Interactive dashboard for training configuration and execution
    """
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = None
        self.loaded_data = None
        
        # Training state
        self.training_in_progress = False
        self.my_rom = None
        self.wandb_logger = None
        
        # Flag to prevent recursive updates during sync
        self._syncing_widgets = False
        
        # Check if widgets are available
        if not WIDGETS_AVAILABLE:
            print("⚠️ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
            return
            
        self._load_config()
        self._create_widgets()
        self._setup_event_handlers()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            self.config = Config(self.config_path)
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            print(f"   Using default config from: {self.config_path}")
            # Try to load as dict if Config class fails
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self.config = type('Config', (), config_dict)()
                self.config.config = config_dict
            except Exception as e2:
                print(f"❌ Failed to load config: {e2}")
                self.config = None
    
    def _create_widgets(self):
        """Create all dashboard widgets"""
        
        # Header
        self.header = widgets.HTML(
            value="<h1>🚀 Model Training Dashboard</h1>",
            layout=widgets.Layout(margin='10px 0px')
        )
        
        # Config file path
        self.config_path_input = widgets.Text(
            value=self.config_path,
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
        
        self.status_output = widgets.Output()
        
        # Create tabs for different configuration sections
        self.tabs = widgets.Tab()
        
        # Tab 1: Basic Training Parameters
        self.basic_training_content = self._create_basic_training_tab()
        
        # Tab 2: Model Architecture
        self.model_arch_content = self._create_model_arch_tab()
        
        # Tab 3: Loss Configuration
        self.loss_config_content = self._create_loss_config_tab()
        
        # Tab 4: Learning Rate Scheduler
        self.scheduler_content = self._create_scheduler_tab()
        
        # Tab 5: Runtime & WandB
        self.runtime_content = self._create_runtime_tab()
        
        # Set up tabs
        self.tabs.children = [
            self.basic_training_content,
            self.model_arch_content,
            self.loss_config_content,
            self.scheduler_content,
            self.runtime_content
        ]
        self.tabs.set_title(0, "⚙️ Training")
        self.tabs.set_title(1, "🏗️ Architecture")
        self.tabs.set_title(2, "📊 Loss")
        self.tabs.set_title(3, "📈 Scheduler")
        self.tabs.set_title(4, "🔧 Runtime")
        
        # Save config button
        self.save_config_btn = widgets.Button(
            description="💾 Save Config to File",
            button_style='info',
            layout=widgets.Layout(width='200px', margin='10px 0px')
        )
        
        # Verification button
        self.verify_btn = widgets.Button(
            description="✅ Verify Training Readiness",
            button_style='warning',
            layout=widgets.Layout(width='200px', margin='10px 0px')
        )
        
        # Start training button
        self.start_training_btn = widgets.Button(
            description="🚀 Start Training",
            button_style='success',
            layout=widgets.Layout(width='200px', margin='20px 0px')
        )
        
        self.training_output = widgets.Output()
        
        # Main layout
        self.main_widget = widgets.VBox([
            self.header,
            widgets.HBox([self.config_path_input, self.load_config_btn]),
            widgets.HBox([self.processed_data_input, self.load_data_btn]),
            self.status_output,
            self.tabs,
            widgets.HBox([self.save_config_btn, self.verify_btn, self.start_training_btn]),
            self.training_output
        ])
    
    def _create_basic_training_tab(self):
        """Create basic training parameters tab"""
        widgets_list = []
        
        # Epochs
        self.epoch_input = widgets.IntSlider(
            value=self.config.training['epoch'] if self.config else 2,
            min=1,
            max=1000,
            step=1,
            description="Epochs:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.epoch_input)
        
        # Batch size
        self.batch_size_input = widgets.IntSlider(
            value=self.config.training['batch_size'] if self.config else 16,
            min=1,
            max=128,
            step=1,
            description="Batch Size:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.batch_size_input)
        
        # Learning rate
        self.learning_rate_input = widgets.FloatLogSlider(
            value=self.config.training['learning_rate'] if self.config else 0.0001,
            base=10,
            min=-5,
            max=-2,
            step=0.1,
            description="Learning Rate:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.learning_rate_input)
        
        # N-steps
        self.nsteps_input = widgets.IntSlider(
            value=self.config.training['nsteps'] if self.config else 2,
            min=1,
            max=30,
            step=1,
            description="N-steps:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.nsteps_input)
        
        # Evaluation time steps
        self.num_tsteps_input = widgets.IntSlider(
            value=self.config.training['num_tsteps'] if self.config else 30,
            min=1,
            max=100,
            step=1,
            description="Eval Time Steps:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.num_tsteps_input)
        
        return widgets.VBox(widgets_list)
    
    def _create_model_arch_tab(self):
        """Create model architecture configuration tab"""
        widgets_list = []
        
        # Section header: Core Model Parameters
        widgets_list.append(widgets.HTML("<h3>Core Model Parameters</h3>"))
        
        # n_channels
        self.n_channels_input = widgets.IntSlider(
            value=self.config.model.get('n_channels', 2) if self.config else 2,
            min=1,
            max=10,
            step=1,
            description="N Channels:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.n_channels_input)
        
        # input_shape - 4 editable fields
        input_shape_default = self.config.data.get('input_shape', [2, 34, 16, 25]) if self.config else [2, 34, 16, 25]
        self.input_shape_ch_input = widgets.IntText(
            value=input_shape_default[0],
            description="Input Shape [C]:",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='140px')
        )
        self.input_shape_x_input = widgets.IntText(
            value=input_shape_default[1],
            description="[X]:",
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px')
        )
        self.input_shape_y_input = widgets.IntText(
            value=input_shape_default[2],
            description="[Y]:",
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px')
        )
        self.input_shape_z_input = widgets.IntText(
            value=input_shape_default[3],
            description="[Z]:",
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px')
        )
        input_shape_box = widgets.HBox([
            widgets.HTML("<div style='width:200px;'>Input Shape:</div>"),
            self.input_shape_ch_input,
            self.input_shape_x_input,
            self.input_shape_y_input,
            self.input_shape_z_input
        ])
        widgets_list.append(input_shape_box)
        
        # Latent dimension
        self.latent_dim_input = widgets.IntSlider(
            value=self.config.model.get('latent_dim', 128) if self.config else 128,
            min=32,
            max=512,
            step=32,
            description="Latent Dimension:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.latent_dim_input)
        
        # Latent dimension validation message output
        self.latent_dim_message = widgets.Output(
            layout=widgets.Layout(height='80px', border='1px solid #ffa500')
        )
        widgets_list.append(self.latent_dim_message)
        
        # u_dim
        self.u_dim_input = widgets.IntSlider(
            value=self.config.model.get('u_dim', 6) if self.config else 6,
            min=1,
            max=20,
            step=1,
            description="U Dimension:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.u_dim_input)
        
        # Transition type
        transition_options = ['linear', 'fno', 'hybrid_fno']
        self.transition_type_input = widgets.Dropdown(
            options=transition_options,
            value=self.config.transition.get('type', 'linear') if self.config else 'linear',
            description="Transition Type:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_type_input)
        
        # Model method (E2C only)
        self.method_input = widgets.Dropdown(
            options=['E2C'],
            value='E2C',
            description="Model Method:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px'),
            disabled=True  # Only E2C is supported
        )
        widgets_list.append(self.method_input)
        
        # Section header: Wells Configuration
        widgets_list.append(widgets.HTML("<hr><h3>Wells Configuration</h3>"))
        
        # num_prod
        self.num_prod_input = widgets.IntSlider(
            value=self.config.data.get('num_prod', 3) if self.config else 3,
            min=1,
            max=10,
            step=1,
            description="Num Producers:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.num_prod_input)
        
        # num_inj
        self.num_inj_input = widgets.IntSlider(
            value=self.config.data.get('num_inj', 3) if self.config else 3,
            min=1,
            max=10,
            step=1,
            description="Num Injectors:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.num_inj_input)
        
        # Well locations - editable widgets
        widgets_list.append(widgets.HTML("<b>Well Locations:</b>"))
        self.well_locations_widgets = self._create_well_locations_widgets()
        widgets_list.extend(self.well_locations_widgets)
        
        # Section header: Encoder Architecture
        widgets_list.append(widgets.HTML("<hr><h3>Encoder Architecture</h3>"))
        
        # Residual blocks
        self.encoder_residual_blocks_input = widgets.IntSlider(
            value=self.config.encoder.get('residual_blocks', 3) if self.config else 3,
            min=2,
            max=5,
            step=1,
            description="Residual Blocks:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.encoder_residual_blocks_input)
        
        # Residual channels
        self.encoder_residual_channels_input = widgets.IntSlider(
            value=self.config.encoder.get('residual_channels', 128) if self.config else 128,
            min=32,
            max=256,
            step=32,
            description="Residual Channels:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.encoder_residual_channels_input)
        
        # Output dims - 4 editable fields
        output_dims_default = self.config.encoder.get('output_dims', [128, 9, 4, 7]) if self.config else [128, 9, 4, 7]
        self.encoder_output_dims_ch_input = widgets.IntText(
            value=output_dims_default[0],
            description="Output Dims [C]:",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='140px')
        )
        self.encoder_output_dims_x_input = widgets.IntText(
            value=output_dims_default[1],
            description="[X]:",
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px')
        )
        self.encoder_output_dims_y_input = widgets.IntText(
            value=output_dims_default[2],
            description="[Y]:",
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px')
        )
        self.encoder_output_dims_z_input = widgets.IntText(
            value=output_dims_default[3],
            description="[Z]:",
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px')
        )
        output_dims_box = widgets.HBox([
            widgets.HTML("<div style='width:200px;'>Output Dims:</div>"),
            self.encoder_output_dims_ch_input,
            self.encoder_output_dims_x_input,
            self.encoder_output_dims_y_input,
            self.encoder_output_dims_z_input
        ])
        widgets_list.append(output_dims_box)
        
        # Flattened size (auto-calculated or editable)
        flattened_size_default = self.config.encoder.get('flattened_size', 32256) if self.config else 32256
        self.encoder_flattened_size_input = widgets.IntText(
            value=flattened_size_default,
            description="Flattened Size:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.encoder_flattened_size_input)
        
        # Conv layers - editable widgets
        widgets_list.append(widgets.HTML("<b>Conv Layers:</b> (Use 0 for In Ch to represent null/n_channels)"))
        self.conv_layers_widgets = self._create_conv_layers_widgets()
        widgets_list.extend(self.conv_layers_widgets)
        
        # Section header: Decoder Architecture
        widgets_list.append(widgets.HTML("<hr><h3>Decoder Architecture</h3>"))
        
        # Use exact dimensions
        self.decoder_use_exact_dimensions_input = widgets.Checkbox(
            value=self.config.decoder.get('use_exact_dimensions', True) if self.config else True,
            description="Use Exact Dimensions:",
            style={'description_width': '200px'}
        )
        widgets_list.append(self.decoder_use_exact_dimensions_input)
        
        # Crop Z to
        crop_z_default = self.config.decoder.get('crop_z_to')
        self.decoder_crop_z_input = widgets.IntText(
            value=crop_z_default if crop_z_default is not None else 0,
            description="Crop Z To:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.decoder_crop_z_input)
        
        # Deconv layers - editable widgets
        widgets_list.append(widgets.HTML("<b>Deconv Layers:</b> (Use 0 for Out Ch to represent null/n_channels)"))
        self.deconv_layers_widgets = self._create_deconv_layers_widgets()
        widgets_list.extend(self.deconv_layers_widgets)
        
        # Section header: Transition Architecture
        widgets_list.append(widgets.HTML("<hr><h3>Transition Architecture</h3>"))
        
        # Encoder hidden dims (text input for list)
        encoder_hidden_dims_default = self.config.transition.get('encoder_hidden_dims', [200, 200]) if self.config else [200, 200]
        self.transition_encoder_hidden_dims_input = widgets.Text(
            value=str(encoder_hidden_dims_default),
            description="Encoder Hidden Dims:",
            placeholder="[200, 200]",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_encoder_hidden_dims_input)
        
        # FNO settings (conditional on transition type)
        fno_config = self.config.transition.get('fno', {}) if self.config else {}
        self.transition_fno_width_input = widgets.IntSlider(
            value=fno_config.get('width', 64),
            min=32,
            max=128,
            step=16,
            description="FNO Width:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_fno_width_input)
        
        self.transition_fno_modes_x_input = widgets.IntSlider(
            value=fno_config.get('modes_x', 8),
            min=1,
            max=20,
            step=1,
            description="FNO Modes X:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_fno_modes_x_input)
        
        self.transition_fno_modes_y_input = widgets.IntSlider(
            value=fno_config.get('modes_y', 8),
            min=1,
            max=20,
            step=1,
            description="FNO Modes Y:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_fno_modes_y_input)
        
        self.transition_fno_modes_z_input = widgets.IntSlider(
            value=fno_config.get('modes_z', 4),
            min=1,
            max=20,
            step=1,
            description="FNO Modes Z:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_fno_modes_z_input)
        
        self.transition_fno_n_layers_input = widgets.IntSlider(
            value=fno_config.get('n_layers', 4),
            min=2,
            max=6,
            step=1,
            description="FNO N Layers:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.transition_fno_n_layers_input)
        
        return widgets.VBox(widgets_list)
    
    def _create_loss_config_tab(self):
        """Create loss configuration tab"""
        widgets_list = []
        
        # Reconstruction loss weight
        self.lambda_reconstruction_input = widgets.FloatSlider(
            value=self.config.loss.get('lambda_reconstruction_loss', 1.0) if self.config else 1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            description="Reconstruction Weight:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.lambda_reconstruction_input)
        
        # Transition loss weight
        self.lambda_trans_input = widgets.FloatSlider(
            value=self.config.loss.get('lambda_trans_loss', 1.0) if self.config else 1.0,
            min=0.0,
            max=20.0,
            step=0.5,
            description="Transition Weight:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.lambda_trans_input)
        
        # Observation loss weight
        self.lambda_yobs_input = widgets.FloatSlider(
            value=self.config.loss.get('lambda_yobs_loss', 1.0) if self.config else 1.0,
            min=0.0,
            max=100.0,
            step=1.0,
            description="Observation Weight:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.lambda_yobs_input)
        
        # Flux loss enable
        self.enable_flux_loss_input = widgets.Checkbox(
            value=self.config.loss.get('enable_flux_loss', False) if self.config else False,
            description="Enable Flux Loss:",
            style={'description_width': '200px'}
        )
        widgets_list.append(self.enable_flux_loss_input)
        
        # Flux loss weight
        self.lambda_flux_input = widgets.FloatLogSlider(
            value=self.config.loss.get('lambda_flux_loss', 0.001) if self.config else 0.001,
            base=10,
            min=-5,
            max=-2,
            step=0.1,
            description="Flux Weight:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.lambda_flux_input)
        
        # BHP loss enable
        self.enable_bhp_loss_input = widgets.Checkbox(
            value=self.config.loss.get('enable_bhp_loss', False) if self.config else False,
            description="Enable BHP Loss:",
            style={'description_width': '200px'}
        )
        widgets_list.append(self.enable_bhp_loss_input)
        
        # BHP loss weight
        self.lambda_bhp_input = widgets.FloatSlider(
            value=self.config.loss.get('lambda_bhp_loss', 20.0) if self.config else 20.0,
            min=0.0,
            max=100.0,
            step=1.0,
            description="BHP Weight:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.lambda_bhp_input)
        
        return widgets.VBox(widgets_list)
    
    def _create_scheduler_tab(self):
        """Create learning rate scheduler configuration tab"""
        widgets_list = []
        
        # Enable scheduler
        self.enable_scheduler_input = widgets.Checkbox(
            value=self.config.learning_rate_scheduler.get('enable', True) if self.config else True,
            description="Enable Scheduler:",
            style={'description_width': '200px'}
        )
        widgets_list.append(self.enable_scheduler_input)
        
        # Scheduler type
        scheduler_options = ['fixed', 'reduce_on_plateau', 'exponential_decay', 
                           'step_decay', 'cosine_annealing', 'cyclic', 'one_cycle']
        # Get scheduler type from config, mapping 'constant' to 'fixed' for backward compatibility
        scheduler_type = self.config.learning_rate_scheduler.get('type', 'step_decay') if self.config else 'step_decay'
        if scheduler_type == 'constant':
            scheduler_type = 'fixed'  # Map 'constant' to 'fixed' (they're equivalent)
        # Ensure the value is in the options list
        if scheduler_type not in scheduler_options:
            scheduler_type = 'step_decay'  # Fallback to default if invalid
        self.scheduler_type_input = widgets.Dropdown(
            options=scheduler_options,
            value=scheduler_type,
            description="Scheduler Type:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.scheduler_type_input)
        
        # Step decay parameters
        self.step_size_input = widgets.IntSlider(
            value=self.config.learning_rate_scheduler.get('step_decay', {}).get('step_size', 100) if self.config else 100,
            min=10,
            max=500,
            step=10,
            description="Step Size (epochs):",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.step_size_input)
        
        self.gamma_input = widgets.FloatSlider(
            value=self.config.learning_rate_scheduler.get('step_decay', {}).get('gamma', 0.5) if self.config else 0.5,
            min=0.1,
            max=1.0,
            step=0.05,
            description="Gamma (decay factor):",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.gamma_input)
        
        return widgets.VBox(widgets_list)
    
    def _create_runtime_tab(self):
        """Create runtime and WandB configuration tab"""
        widgets_list = []
        
        # Device selection
        device_options = ['auto', 'cuda', 'cpu', 'mps']
        self.device_input = widgets.Dropdown(
            options=device_options,
            value=self.config.runtime.get('device', 'auto') if self.config else 'auto',
            description="Device:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.device_input)
        
        # Print interval
        self.print_interval_input = widgets.IntSlider(
            value=self.config.runtime.get('print_interval', 5000) if self.config else 5000,
            min=50,
            max=10000,
            step=50,
            description="Print Interval:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.print_interval_input)
        
        # Save best model
        self.save_best_model_input = widgets.Checkbox(
            value=self.config.runtime.get('save_best_model', True) if self.config else True,
            description="Save Best Model:",
            style={'description_width': '200px'}
        )
        widgets_list.append(self.save_best_model_input)
        
        # Best model criterion
        criterion_options = ['total_loss', 'observation_loss', 'reconstruction_loss']
        self.best_model_criterion_input = widgets.Dropdown(
            options=criterion_options,
            value=self.config.runtime.get('best_model_criterion', 'total_loss') if self.config else 'total_loss',
            description="Best Model Criterion:",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.best_model_criterion_input)
        
        # WandB enable
        self.wandb_enable_input = widgets.Checkbox(
            value=self.config.runtime.get('wandb', {}).get('enable', True) if self.config else True,
            description="Enable WandB:",
            style={'description_width': '200px'}
        )
        widgets_list.append(self.wandb_enable_input)
        
        # WandB project
        self.wandb_project_input = widgets.Text(
            value=self.config.runtime.get('wandb', {}).get('project', 'ROM-E2C') if self.config else 'ROM-E2C',
            description="WandB Project:",
            placeholder="Project name",
            style={'description_width': '200px'},
            layout=widgets.Layout(width='600px')
        )
        widgets_list.append(self.wandb_project_input)
        
        return widgets.VBox(widgets_list)
    
    def _create_conv_layers_widgets(self):
        """Create editable widgets for conv_layers"""
        widgets_list = []
        self.conv_layers_inputs = {}
        
        if not self.config or 'encoder' not in self.config.config or 'conv_layers' not in self.config.config['encoder']:
            conv_layers = {
                'conv1': [None, 16, [3, 3, 3], [2, 2, 2], [1, 1, 1]],
                'conv2': [16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]],
                'conv3': [32, 64, [3, 3, 3], [2, 2, 2], [1, 1, 1]],
                'conv4': [64, 128, [3, 3, 3], [1, 1, 1], [1, 1, 1]]
            }
        else:
            conv_layers = self.config.config['encoder']['conv_layers']
        
        for layer_name, layer_config in conv_layers.items():
            widgets_list.append(widgets.HTML(f"<b>{layer_name}:</b>"))
            layer_widgets = {}
            
            # Input channels (can be null/None - use 0 to represent null/n_channels)
            in_ch = layer_config[0] if layer_config[0] is not None else 0
            in_ch_widget = widgets.IntText(
                value=in_ch,
                description="In Ch (0=null):",
                style={'description_width': '100px'},
                layout=widgets.Layout(width='130px')
            )
            layer_widgets['in_channels'] = in_ch_widget
            
            # Output channels
            out_ch_widget = widgets.IntText(
                value=layer_config[1],
                description="Out Ch:",
                style={'description_width': '80px'},
                layout=widgets.Layout(width='120px')
            )
            layer_widgets['out_channels'] = out_ch_widget
            
            # Kernel size [x, y, z]
            kernel_widgets = []
            for i, dim_name in enumerate(['Kx', 'Ky', 'Kz']):
                kw = widgets.IntText(
                    value=layer_config[2][i],
                    description=dim_name + ":",
                    style={'description_width': '50px'},
                    layout=widgets.Layout(width='100px')
                )
                kernel_widgets.append(kw)
            layer_widgets['kernel'] = kernel_widgets
            
            # Stride [x, y, z]
            stride_widgets = []
            for i, dim_name in enumerate(['Sx', 'Sy', 'Sz']):
                sw = widgets.IntText(
                    value=layer_config[3][i],
                    description=dim_name + ":",
                    style={'description_width': '50px'},
                    layout=widgets.Layout(width='100px')
                )
                stride_widgets.append(sw)
            layer_widgets['stride'] = stride_widgets
            
            # Padding [x, y, z]
            padding_widgets = []
            for i, dim_name in enumerate(['Px', 'Py', 'Pz']):
                pw = widgets.IntText(
                    value=layer_config[4][i],
                    description=dim_name + ":",
                    style={'description_width': '50px'},
                    layout=widgets.Layout(width='100px')
                )
                padding_widgets.append(pw)
            layer_widgets['padding'] = padding_widgets
            
            # Layout widgets horizontally
            widgets_list.append(widgets.HBox([
                in_ch_widget, out_ch_widget,
                widgets.HTML("<b>Kernel:</b>"), *kernel_widgets,
                widgets.HTML("<b>Stride:</b>"), *stride_widgets,
                widgets.HTML("<b>Padding:</b>"), *padding_widgets
            ]))
            
            self.conv_layers_inputs[layer_name] = layer_widgets
        
        return widgets_list
    
    def _create_deconv_layers_widgets(self):
        """Create editable widgets for deconv_layers"""
        widgets_list = []
        self.deconv_layers_inputs = {}
        
        if not self.config or 'decoder' not in self.config.config or 'deconv_layers' not in self.config.config['decoder']:
            deconv_layers = {
                'deconv1': [128, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]],
                'deconv2': [64, 32, [4, 4, 4], [2, 2, 2], [1, 1, 1]],
                'deconv3': [32, 16, [3, 3, 3], [1, 1, 1], [1, 1, 1]],
                'deconv4': [16, 16, [4, 4, 4], [2, 2, 2], [2, 1, 3], [0, 0, 1]],
                'final_conv': [16, None, [1, 1, 1], [1, 1, 1], [0, 0, 0]]
            }
        else:
            deconv_layers = self.config.config['decoder']['deconv_layers']
        
        for layer_name, layer_config in deconv_layers.items():
            widgets_list.append(widgets.HTML(f"<b>{layer_name}:</b>"))
            layer_widgets = {}
            
            # Input channels
            in_ch_widget = widgets.IntText(
                value=layer_config[0] if layer_config[0] is not None else 0,
                description="In Ch:",
                style={'description_width': '80px'},
                layout=widgets.Layout(width='120px')
            )
            layer_widgets['in_channels'] = in_ch_widget
            
            # Output channels (can be null/None - use 0 to represent null/n_channels)
            out_ch = layer_config[1] if layer_config[1] is not None else 0
            out_ch_widget = widgets.IntText(
                value=out_ch,
                description="Out Ch (0=null):",
                style={'description_width': '100px'},
                layout=widgets.Layout(width='130px')
            )
            layer_widgets['out_channels'] = out_ch_widget
            
            # Kernel size [x, y, z]
            kernel_widgets = []
            for i, dim_name in enumerate(['Kx', 'Ky', 'Kz']):
                kw = widgets.IntText(
                    value=layer_config[2][i],
                    description=dim_name + ":",
                    style={'description_width': '50px'},
                    layout=widgets.Layout(width='100px')
                )
                kernel_widgets.append(kw)
            layer_widgets['kernel'] = kernel_widgets
            
            # Stride [x, y, z]
            stride_widgets = []
            for i, dim_name in enumerate(['Sx', 'Sy', 'Sz']):
                sw = widgets.IntText(
                    value=layer_config[3][i],
                    description=dim_name + ":",
                    style={'description_width': '50px'},
                    layout=widgets.Layout(width='100px')
                )
                stride_widgets.append(sw)
            layer_widgets['stride'] = stride_widgets
            
            # Padding [x, y, z]
            padding_widgets = []
            for i, dim_name in enumerate(['Px', 'Py', 'Pz']):
                pw = widgets.IntText(
                    value=layer_config[4][i],
                    description=dim_name + ":",
                    style={'description_width': '50px'},
                    layout=widgets.Layout(width='100px')
                )
                padding_widgets.append(pw)
            layer_widgets['padding'] = padding_widgets
            
            # Output padding (optional, for some layers)
            if len(layer_config) > 5:
                output_padding_widgets = []
                for i, dim_name in enumerate(['OPx', 'OPy', 'OPz']):
                    opw = widgets.IntText(
                        value=layer_config[5][i],
                        description=dim_name + ":",
                        style={'description_width': '50px'},
                        layout=widgets.Layout(width='100px')
                    )
                    output_padding_widgets.append(opw)
                layer_widgets['output_padding'] = output_padding_widgets
            else:
                layer_widgets['output_padding'] = None
            
            # Layout widgets horizontally
            row_widgets = [
                in_ch_widget, out_ch_widget,
                widgets.HTML("<b>Kernel:</b>"), *kernel_widgets,
                widgets.HTML("<b>Stride:</b>"), *stride_widgets,
                widgets.HTML("<b>Padding:</b>"), *padding_widgets
            ]
            if layer_widgets['output_padding']:
                row_widgets.extend([widgets.HTML("<b>OutPad:</b>"), *layer_widgets['output_padding']])
            
            widgets_list.append(widgets.HBox(row_widgets))
            
            self.deconv_layers_inputs[layer_name] = layer_widgets
        
        return widgets_list
    
    def _create_well_locations_widgets(self):
        """Create editable widgets for well_locations"""
        widgets_list = []
        self.well_locations_inputs = {'injectors': {}, 'producers': {}}
        
        if not self.config or 'data' not in self.config.config or 'well_locations' not in self.config.config['data']:
            well_locs = {
                'injectors': {'I1': [10, 12, 24], 'I2': [31, 13, 24], 'I4': [8, 1, 24]},
                'producers': {'P1': [1, 11, 24], 'P2': [16, 4, 24], 'P3': [18, 11, 24]}
            }
        else:
            well_locs = self.config.config['data']['well_locations']
        
        # Injectors
        widgets_list.append(widgets.HTML("<b>Injectors:</b>"))
        if 'injectors' in well_locs:
            for well_name, coords in well_locs['injectors'].items():
                well_widgets = []
                for i, coord_name in enumerate(['X', 'Y', 'Z']):
                    cw = widgets.IntText(
                        value=coords[i],
                        description=f"{well_name} {coord_name}:",
                        style={'description_width': '100px'},
                        layout=widgets.Layout(width='120px')
                    )
                    well_widgets.append(cw)
                widgets_list.append(widgets.HBox(well_widgets))
                self.well_locations_inputs['injectors'][well_name] = well_widgets
        
        # Producers
        widgets_list.append(widgets.HTML("<b>Producers:</b>"))
        if 'producers' in well_locs:
            for well_name, coords in well_locs['producers'].items():
                well_widgets = []
                for i, coord_name in enumerate(['X', 'Y', 'Z']):
                    cw = widgets.IntText(
                        value=coords[i],
                        description=f"{well_name} {coord_name}:",
                        style={'description_width': '100px'},
                        layout=widgets.Layout(width='120px')
                    )
                    well_widgets.append(cw)
                widgets_list.append(widgets.HBox(well_widgets))
                self.well_locations_inputs['producers'][well_name] = well_widgets
        
        return widgets_list
    
    def _sync_n_channels_to_input_shape(self, change=None):
        """Sync n_channels to input_shape[0]"""
        if self._syncing_widgets:
            return
        self._syncing_widgets = True
        try:
            self.input_shape_ch_input.value = self.n_channels_input.value
        finally:
            self._syncing_widgets = False
    
    def _sync_input_shape_to_n_channels(self, change=None):
        """Sync input_shape[0] to n_channels"""
        if self._syncing_widgets:
            return
        self._syncing_widgets = True
        try:
            self.n_channels_input.value = self.input_shape_ch_input.value
        finally:
            self._syncing_widgets = False
    
    def _on_latent_dim_change(self, change=None):
        """Handle latent_dim changes and show validation message"""
        if self._syncing_widgets:
            return
        
        with self.latent_dim_message:
            clear_output(wait=True)
            latent_dim = self.latent_dim_input.value
            output_dims_ch = self.encoder_output_dims_ch_input.value
            
            messages = []
            if latent_dim != output_dims_ch:
                messages.append(f"⚠️ Warning: latent_dim ({latent_dim}) does not match output_dims[0] ({output_dims_ch})")
                messages.append(f"   Consider updating output_dims[0] to {latent_dim}")
            
            # Calculate expected flattened_size
            output_dims = [
                self.encoder_output_dims_ch_input.value,
                self.encoder_output_dims_x_input.value,
                self.encoder_output_dims_y_input.value,
                self.encoder_output_dims_z_input.value
            ]
            expected_flattened = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3]
            current_flattened = self.encoder_flattened_size_input.value
            
            if expected_flattened != current_flattened:
                messages.append(f"⚠️ Warning: flattened_size ({current_flattened}) does not match calculated value")
                messages.append(f"   Expected: {output_dims[0]} * {output_dims[1]} * {output_dims[2]} * {output_dims[3]} = {expected_flattened}")
                messages.append(f"   Consider updating flattened_size to {expected_flattened}")
            
            if messages:
                for msg in messages:
                    print(msg)
            else:
                print("✅ All dimensions are consistent")
    
    def _on_output_dims_change(self, change=None):
        """Recalculate flattened_size when output_dims change"""
        if self._syncing_widgets:
            return
        
        output_dims = [
            self.encoder_output_dims_ch_input.value,
            self.encoder_output_dims_x_input.value,
            self.encoder_output_dims_y_input.value,
            self.encoder_output_dims_z_input.value
        ]
        expected_flattened = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3]
        self._syncing_widgets = True
        try:
            self.encoder_flattened_size_input.value = expected_flattened
        finally:
            self._syncing_widgets = False
        
        # Also trigger latent_dim check
        self._on_latent_dim_change()
    
    def _verify_training_readiness(self):
        """Verify training readiness and display results"""
        with self.status_output:
            clear_output(wait=True)
            print("=" * 70)
            print("🔍 Verifying Training Readiness")
            print("=" * 70)
            
            issues = []
            warnings = []
            
            # Check n_channels matches input_shape[0]
            n_channels = self.n_channels_input.value
            input_shape_ch = self.input_shape_ch_input.value
            if n_channels != input_shape_ch:
                issues.append(f"❌ n_channels ({n_channels}) does not match input_shape[0] ({input_shape_ch})")
            else:
                print(f"✅ n_channels ({n_channels}) matches input_shape[0]")
            
            # Check latent_dim matches output_dims[0]
            latent_dim = self.latent_dim_input.value
            output_dims_ch = self.encoder_output_dims_ch_input.value
            if latent_dim != output_dims_ch:
                warnings.append(f"⚠️ latent_dim ({latent_dim}) does not match output_dims[0] ({output_dims_ch})")
                warnings.append(f"   Consider updating output_dims[0] to {latent_dim}")
            else:
                print(f"✅ latent_dim ({latent_dim}) matches output_dims[0]")
            
            # Check flattened_size matches product of output_dims
            output_dims = [
                self.encoder_output_dims_ch_input.value,
                self.encoder_output_dims_x_input.value,
                self.encoder_output_dims_y_input.value,
                self.encoder_output_dims_z_input.value
            ]
            expected_flattened = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3]
            current_flattened = self.encoder_flattened_size_input.value
            if expected_flattened != current_flattened:
                issues.append(f"❌ flattened_size ({current_flattened}) does not match calculated value ({expected_flattened})")
                issues.append(f"   Expected: {output_dims[0]} * {output_dims[1]} * {output_dims[2]} * {output_dims[3]} = {expected_flattened}")
            else:
                print(f"✅ flattened_size ({current_flattened}) matches calculated value")
            
            # Check config is loaded
            if not self.config:
                issues.append("❌ Config file not loaded")
            else:
                print("✅ Config file loaded")
            
            # Check data is loaded
            if not self.loaded_data:
                warnings.append("⚠️ Processed data not loaded (required for training)")
            else:
                print("✅ Processed data loaded")
            
            # Display results
            print()
            if issues:
                print("❌ ISSUES FOUND (must be fixed before training):")
                for issue in issues:
                    print(f"   {issue}")
                print()
            
            if warnings:
                print("⚠️ WARNINGS (should be reviewed):")
                for warning in warnings:
                    print(f"   {warning}")
                print()
            
            if not issues and not warnings:
                print("✅ All checks passed! Training is ready to start.")
            elif not issues:
                print("✅ No critical issues found. Training can proceed, but review warnings above.")
            else:
                print("❌ Please fix the issues above before starting training.")
    
    def _setup_event_handlers(self):
        """Setup event handlers for widgets"""
        self.load_config_btn.on_click(self._load_config_handler)
        self.load_data_btn.on_click(self._load_data_handler)
        self.save_config_btn.on_click(self._save_config_handler)
        self.start_training_btn.on_click(self._start_training_handler)
        self.verify_btn.on_click(lambda b: self._verify_training_readiness())
        
        # Sync handlers for n_channels and input_shape
        self.n_channels_input.observe(self._sync_n_channels_to_input_shape, names='value')
        self.input_shape_ch_input.observe(self._sync_input_shape_to_n_channels, names='value')
        
        # Latent dim change handler
        self.latent_dim_input.observe(self._on_latent_dim_change, names='value')
        
        # Output dims change handlers (to recalculate flattened_size)
        self.encoder_output_dims_ch_input.observe(self._on_output_dims_change, names='value')
        self.encoder_output_dims_x_input.observe(self._on_output_dims_change, names='value')
        self.encoder_output_dims_y_input.observe(self._on_output_dims_change, names='value')
        self.encoder_output_dims_z_input.observe(self._on_output_dims_change, names='value')
    
    def _load_config_handler(self, button):
        """Handle config file loading"""
        with self.status_output:
            clear_output(wait=True)
            try:
                self.config_path = self.config_path_input.value.strip()
                self.config = Config(self.config_path)
                print(f"✅ Config loaded from: {self.config_path}")
                self._update_widgets_from_config()
            except Exception as e:
                print(f"❌ Error loading config: {e}")
    
    def _load_data_handler(self, button):
        """Handle processed data loading"""
        with self.status_output:
            clear_output(wait=True)
            try:
                data_dir = self.processed_data_input.value.strip()
                self.loaded_data = load_processed_data(data_dir=data_dir)
                if self.loaded_data:
                    print(f"✅ Processed data loaded from: {data_dir}")
                    print(f"   Training samples: {self.loaded_data['metadata'].get('num_train', 0)}")
                    print(f"   Evaluation samples: {self.loaded_data['metadata'].get('num_eval', 0)}")
                else:
                    print(f"❌ No processed data found in: {data_dir}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
    
    def _save_config_handler(self, button):
        """Handle saving config to file"""
        if not self.config:
            with self.status_output:
                clear_output(wait=True)
                print("❌ No config loaded! Please load a config file first.")
            return
        
        # Update config from widgets first
        self._update_config_from_widgets()
        
        # Save to file
        config_path = self.config_path_input.value.strip() or self.config_path
        try:
            with self.status_output:
                clear_output(wait=True)
                self._save_config_to_file(config_path)
                print(f"✅ Configuration saved to: {config_path}")
        except Exception as e:
            with self.status_output:
                print(f"❌ Error saving config: {e}")
                import traceback
                traceback.print_exc()
    
    def _save_config_to_file(self, file_path):
        """Save current config dictionary to YAML file"""
        # Update config from widgets to ensure latest values
        self._update_config_from_widgets()
        
        # Use yaml.dump to write the config
        # Note: This will overwrite the file and may lose comments
        # For preserving comments, would need ruamel.yaml, but using standard yaml for now
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.config.config,
                f,
                default_flow_style=False,
                indent=2,
                allow_unicode=True,
                sort_keys=False
            )
    
    def _update_widgets_from_config(self):
        """Update widget values from loaded config"""
        if not self.config:
            return
        
        self._syncing_widgets = True
        try:
            # Basic training
            self.epoch_input.value = self.config.training['epoch']
            self.batch_size_input.value = self.config.training['batch_size']
            self.learning_rate_input.value = self.config.training['learning_rate']
            self.nsteps_input.value = self.config.training['nsteps']
            self.num_tsteps_input.value = self.config.training['num_tsteps']
            
            # Model architecture - core parameters
            self.n_channels_input.value = self.config.model.get('n_channels', 2)
            input_shape = self.config.data.get('input_shape', [2, 34, 16, 25])
            self.input_shape_ch_input.value = input_shape[0]
            self.input_shape_x_input.value = input_shape[1]
            self.input_shape_y_input.value = input_shape[2]
            self.input_shape_z_input.value = input_shape[3]
            
            self.latent_dim_input.value = self.config.model.get('latent_dim', 128)
            self.u_dim_input.value = self.config.model.get('u_dim', 6)
            self.transition_type_input.value = self.config.transition.get('type', 'linear')
            self.method_input.value = self.config.model.get('method', 'E2C')
            
            # Wells configuration
            self.num_prod_input.value = self.config.data.get('num_prod', 3)
            self.num_inj_input.value = self.config.data.get('num_inj', 3)
            
            # Update well_locations widgets
            if hasattr(self, 'well_locations_inputs') and 'data' in self.config.config and 'well_locations' in self.config.config['data']:
                well_locs = self.config.config['data']['well_locations']
                if 'injectors' in well_locs:
                    for well_name, coords in well_locs['injectors'].items():
                        if well_name in self.well_locations_inputs['injectors']:
                            for i, cw in enumerate(self.well_locations_inputs['injectors'][well_name]):
                                cw.value = coords[i]
                if 'producers' in well_locs:
                    for well_name, coords in well_locs['producers'].items():
                        if well_name in self.well_locations_inputs['producers']:
                            for i, cw in enumerate(self.well_locations_inputs['producers'][well_name]):
                                cw.value = coords[i]
            
            # Encoder architecture
            self.encoder_residual_blocks_input.value = self.config.encoder.get('residual_blocks', 3)
            self.encoder_residual_channels_input.value = self.config.encoder.get('residual_channels', 128)
            output_dims = self.config.encoder.get('output_dims', [128, 9, 4, 7])
            self.encoder_output_dims_ch_input.value = output_dims[0]
            self.encoder_output_dims_x_input.value = output_dims[1]
            self.encoder_output_dims_y_input.value = output_dims[2]
            self.encoder_output_dims_z_input.value = output_dims[3]
            self.encoder_flattened_size_input.value = self.config.encoder.get('flattened_size', 32256)
            
            # Update conv_layers widgets
            if hasattr(self, 'conv_layers_inputs') and 'encoder' in self.config.config and 'conv_layers' in self.config.config['encoder']:
                conv_layers = self.config.config['encoder']['conv_layers']
                for layer_name, layer_config in conv_layers.items():
                    if layer_name in self.conv_layers_inputs:
                        widgets = self.conv_layers_inputs[layer_name]
                        widgets['in_channels'].value = layer_config[0] if layer_config[0] is not None else 0
                        widgets['out_channels'].value = layer_config[1]
                        for i, kw in enumerate(widgets['kernel']):
                            kw.value = layer_config[2][i]
                        for i, sw in enumerate(widgets['stride']):
                            sw.value = layer_config[3][i]
                        for i, pw in enumerate(widgets['padding']):
                            pw.value = layer_config[4][i]
            
            # Decoder architecture
            self.decoder_use_exact_dimensions_input.value = self.config.decoder.get('use_exact_dimensions', True)
            crop_z = self.config.decoder.get('crop_z_to')
            self.decoder_crop_z_input.value = crop_z if crop_z is not None else 0
            
            # Update deconv_layers widgets
            if hasattr(self, 'deconv_layers_inputs') and 'decoder' in self.config.config and 'deconv_layers' in self.config.config['decoder']:
                deconv_layers = self.config.config['decoder']['deconv_layers']
                for layer_name, layer_config in deconv_layers.items():
                    if layer_name in self.deconv_layers_inputs:
                        widgets = self.deconv_layers_inputs[layer_name]
                        widgets['in_channels'].value = layer_config[0] if layer_config[0] is not None else 0
                        widgets['out_channels'].value = layer_config[1] if layer_config[1] is not None else 0
                        for i, kw in enumerate(widgets['kernel']):
                            kw.value = layer_config[2][i]
                        for i, sw in enumerate(widgets['stride']):
                            sw.value = layer_config[3][i]
                        for i, pw in enumerate(widgets['padding']):
                            pw.value = layer_config[4][i]
                        if len(layer_config) > 5 and widgets['output_padding']:
                            for i, opw in enumerate(widgets['output_padding']):
                                opw.value = layer_config[5][i]
            
            # Transition architecture
            encoder_hidden_dims = self.config.transition.get('encoder_hidden_dims', [200, 200])
            self.transition_encoder_hidden_dims_input.value = str(encoder_hidden_dims)
            fno_config = self.config.transition.get('fno', {})
            self.transition_fno_width_input.value = fno_config.get('width', 64)
            self.transition_fno_modes_x_input.value = fno_config.get('modes_x', 8)
            self.transition_fno_modes_y_input.value = fno_config.get('modes_y', 8)
            self.transition_fno_modes_z_input.value = fno_config.get('modes_z', 4)
            self.transition_fno_n_layers_input.value = fno_config.get('n_layers', 4)
            
            # Loss configuration
            self.lambda_reconstruction_input.value = self.config.loss.get('lambda_reconstruction_loss', 1.0)
            self.lambda_trans_input.value = self.config.loss.get('lambda_trans_loss', 1.0)
            self.lambda_yobs_input.value = self.config.loss.get('lambda_yobs_loss', 1.0)
            self.enable_flux_loss_input.value = self.config.loss.get('enable_flux_loss', False)
            self.lambda_flux_input.value = self.config.loss.get('lambda_flux_loss', 0.001)
            self.enable_bhp_loss_input.value = self.config.loss.get('enable_bhp_loss', False)
            self.lambda_bhp_input.value = self.config.loss.get('lambda_bhp_loss', 20.0)
            
            # Scheduler
            self.enable_scheduler_input.value = self.config.learning_rate_scheduler.get('enable', True)
            scheduler_type = self.config.learning_rate_scheduler.get('type', 'step_decay')
            if scheduler_type == 'constant':
                scheduler_type = 'fixed'  # Map 'constant' to 'fixed' (they're equivalent)
            if scheduler_type not in self.scheduler_type_input.options:
                scheduler_type = 'step_decay'  # Fallback to default if invalid
            self.scheduler_type_input.value = scheduler_type
            step_decay_config = self.config.learning_rate_scheduler.get('step_decay', {})
            self.step_size_input.value = step_decay_config.get('step_size', 100)
            self.gamma_input.value = step_decay_config.get('gamma', 0.5)
            
            # Runtime
            self.device_input.value = self.config.runtime.get('device', 'auto')
            self.print_interval_input.value = self.config.runtime.get('print_interval', 5000)
            self.save_best_model_input.value = self.config.runtime.get('save_best_model', True)
            self.best_model_criterion_input.value = self.config.runtime.get('best_model_criterion', 'total_loss')
            wandb_config = self.config.runtime.get('wandb', {})
            self.wandb_enable_input.value = wandb_config.get('enable', True)
            self.wandb_project_input.value = wandb_config.get('project', 'ROM-E2C')
            
        except Exception as e:
            print(f"⚠️ Error updating widgets: {e}")
        finally:
            self._syncing_widgets = False
            # Trigger initial validation check
            self._on_latent_dim_change()
    
    def _start_training_handler(self, button):
        """Handle training start"""
        if self.training_in_progress:
            with self.training_output:
                print("⚠️ Training already in progress!")
            return
        
        if not self.loaded_data:
            with self.status_output:
                print("❌ Please load processed data first!")
            return
        
        if not self.config:
            with self.status_output:
                print("❌ Please load config first!")
            return
        
        # Update config with widget values
        self._update_config_from_widgets()
        
        # Auto-save config to file before training (so config file reflects what's being used)
        config_path = self.config_path_input.value.strip() or self.config_path
        try:
            self._save_config_to_file(config_path)
            with self.status_output:
                clear_output(wait=True)
                print(f"💾 Configuration auto-saved to: {config_path}")
                print("🚀 Starting training with current configuration...")
        except Exception as e:
            with self.status_output:
                print(f"⚠️ Warning: Could not save config file: {e}")
                print("   Continuing with in-memory configuration...")
        
        # Start training in a separate thread or asynchronously
        self.training_in_progress = True
        self.start_training_btn.disabled = True
        self.start_training_btn.description = "⏳ Training..."
        
        try:
            self._run_training()
        except Exception as e:
            with self.training_output:
                print(f"❌ Training error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self.training_in_progress = False
            self.start_training_btn.disabled = False
            self.start_training_btn.description = "🚀 Start Training"
    
    def _update_config_from_widgets(self):
        """Update config object with widget values"""
        if not self.config:
            return
        
        # Update training parameters (access via config.config dict)
        self.config.config['training']['epoch'] = self.epoch_input.value
        self.config.config['training']['batch_size'] = self.batch_size_input.value
        self.config.config['training']['learning_rate'] = self.learning_rate_input.value
        self.config.config['training']['nsteps'] = self.nsteps_input.value
        self.config.config['training']['num_tsteps'] = self.num_tsteps_input.value
        
        # Update model architecture - core parameters
        self.config.config['model']['n_channels'] = self.n_channels_input.value
        self.config.config['data']['input_shape'] = [
            self.input_shape_ch_input.value,
            self.input_shape_x_input.value,
            self.input_shape_y_input.value,
            self.input_shape_z_input.value
        ]
        self.config.config['model']['latent_dim'] = self.latent_dim_input.value
        self.config.config['model']['u_dim'] = self.u_dim_input.value
        self.config.config['transition']['type'] = self.transition_type_input.value
        self.config.config['model']['method'] = self.method_input.value
        
        # Update wells configuration
        self.config.config['data']['num_prod'] = self.num_prod_input.value
        self.config.config['data']['num_inj'] = self.num_inj_input.value
        
        # Update well_locations
        if 'well_locations' not in self.config.config['data']:
            self.config.config['data']['well_locations'] = {'injectors': {}, 'producers': {}}
        if hasattr(self, 'well_locations_inputs'):
            # Update injectors
            if 'injectors' not in self.config.config['data']['well_locations']:
                self.config.config['data']['well_locations']['injectors'] = {}
            for well_name, coord_widgets in self.well_locations_inputs['injectors'].items():
                coords = [cw.value for cw in coord_widgets]
                self.config.config['data']['well_locations']['injectors'][well_name] = coords
            
            # Update producers
            if 'producers' not in self.config.config['data']['well_locations']:
                self.config.config['data']['well_locations']['producers'] = {}
            for well_name, coord_widgets in self.well_locations_inputs['producers'].items():
                coords = [cw.value for cw in coord_widgets]
                self.config.config['data']['well_locations']['producers'][well_name] = coords
        
        # Update encoder architecture
        if 'encoder' not in self.config.config:
            self.config.config['encoder'] = {}
        self.config.config['encoder']['residual_blocks'] = self.encoder_residual_blocks_input.value
        self.config.config['encoder']['residual_channels'] = self.encoder_residual_channels_input.value
        self.config.config['encoder']['output_dims'] = [
            self.encoder_output_dims_ch_input.value,
            self.encoder_output_dims_x_input.value,
            self.encoder_output_dims_y_input.value,
            self.encoder_output_dims_z_input.value
        ]
        self.config.config['encoder']['flattened_size'] = self.encoder_flattened_size_input.value
        
        # Update conv_layers
        if 'conv_layers' not in self.config.config['encoder']:
            self.config.config['encoder']['conv_layers'] = {}
        if hasattr(self, 'conv_layers_inputs'):
            for layer_name, widgets in self.conv_layers_inputs.items():
                in_ch = widgets['in_channels'].value
                out_ch = widgets['out_channels'].value
                kernel = [kw.value for kw in widgets['kernel']]
                stride = [sw.value for sw in widgets['stride']]
                padding = [pw.value for pw in widgets['padding']]
                # Use None for input_channels if it's 0 (meaning null/n_channels)
                in_ch_val = None if in_ch == 0 else in_ch
                self.config.config['encoder']['conv_layers'][layer_name] = [
                    in_ch_val, out_ch, kernel, stride, padding
                ]
        
        # Update decoder architecture
        if 'decoder' not in self.config.config:
            self.config.config['decoder'] = {}
        self.config.config['decoder']['use_exact_dimensions'] = self.decoder_use_exact_dimensions_input.value
        crop_z_value = self.decoder_crop_z_input.value
        self.config.config['decoder']['crop_z_to'] = crop_z_value if crop_z_value > 0 else None
        
        # Update deconv_layers
        if 'deconv_layers' not in self.config.config['decoder']:
            self.config.config['decoder']['deconv_layers'] = {}
        if hasattr(self, 'deconv_layers_inputs'):
            for layer_name, widgets in self.deconv_layers_inputs.items():
                in_ch = widgets['in_channels'].value
                out_ch = widgets['out_channels'].value
                kernel = [kw.value for kw in widgets['kernel']]
                stride = [sw.value for sw in widgets['stride']]
                padding = [pw.value for pw in widgets['padding']]
                # Use None for output_channels if it's 0 (meaning null/n_channels)
                out_ch_val = None if out_ch == 0 else out_ch
                layer_config = [in_ch, out_ch_val, kernel, stride, padding]
                # Add output_padding if it exists
                if widgets['output_padding']:
                    output_padding = [opw.value for opw in widgets['output_padding']]
                    layer_config.append(output_padding)
                self.config.config['decoder']['deconv_layers'][layer_name] = layer_config
        
        # Update transition architecture
        if 'transition' not in self.config.config:
            self.config.config['transition'] = {}
        # Parse encoder_hidden_dims from text input
        try:
            encoder_hidden_dims_str = self.transition_encoder_hidden_dims_input.value.strip()
            if encoder_hidden_dims_str.startswith('[') and encoder_hidden_dims_str.endswith(']'):
                encoder_hidden_dims_str = encoder_hidden_dims_str[1:-1]
            encoder_hidden_dims = [int(x.strip()) for x in encoder_hidden_dims_str.split(',')]
            self.config.config['transition']['encoder_hidden_dims'] = encoder_hidden_dims
        except Exception as e:
            print(f"⚠️ Warning: Could not parse encoder_hidden_dims, using default: {e}")
            self.config.config['transition']['encoder_hidden_dims'] = [200, 200]
        
        # Update FNO settings
        if 'fno' not in self.config.config['transition']:
            self.config.config['transition']['fno'] = {}
        self.config.config['transition']['fno']['width'] = self.transition_fno_width_input.value
        self.config.config['transition']['fno']['modes_x'] = self.transition_fno_modes_x_input.value
        self.config.config['transition']['fno']['modes_y'] = self.transition_fno_modes_y_input.value
        self.config.config['transition']['fno']['modes_z'] = self.transition_fno_modes_z_input.value
        self.config.config['transition']['fno']['n_layers'] = self.transition_fno_n_layers_input.value
        
        # Update loss configuration
        self.config.config['loss']['lambda_reconstruction_loss'] = self.lambda_reconstruction_input.value
        self.config.config['loss']['lambda_trans_loss'] = self.lambda_trans_input.value
        self.config.config['loss']['lambda_yobs_loss'] = self.lambda_yobs_input.value
        self.config.config['loss']['enable_flux_loss'] = self.enable_flux_loss_input.value
        self.config.config['loss']['lambda_flux_loss'] = self.lambda_flux_input.value
        self.config.config['loss']['enable_bhp_loss'] = self.enable_bhp_loss_input.value
        self.config.config['loss']['lambda_bhp_loss'] = self.lambda_bhp_input.value
        
        # Update scheduler
        self.config.config['learning_rate_scheduler']['enable'] = self.enable_scheduler_input.value
        self.config.config['learning_rate_scheduler']['type'] = self.scheduler_type_input.value
        if 'step_decay' not in self.config.config['learning_rate_scheduler']:
            self.config.config['learning_rate_scheduler']['step_decay'] = {}
        self.config.config['learning_rate_scheduler']['step_decay']['step_size'] = self.step_size_input.value
        self.config.config['learning_rate_scheduler']['step_decay']['gamma'] = self.gamma_input.value
        
        # Update runtime
        self.config.config['runtime']['device'] = self.device_input.value
        self.config.config['runtime']['print_interval'] = self.print_interval_input.value
        self.config.config['runtime']['save_best_model'] = self.save_best_model_input.value
        self.config.config['runtime']['best_model_criterion'] = self.best_model_criterion_input.value
        if 'wandb' not in self.config.config['runtime']:
            self.config.config['runtime']['wandb'] = {}
        self.config.config['runtime']['wandb']['enable'] = self.wandb_enable_input.value
        self.config.config['runtime']['wandb']['project'] = self.wandb_project_input.value
    
    def _run_training(self):
        """Execute the training loop"""
        with self.training_output:
            clear_output(wait=True)
            print("=" * 70)
            print("🚀 Starting Training")
            print("=" * 70)
            
            # Import training components
            from model.training.rom_wrapper import ROMWithE2C
            from utilities.wandb_integration import create_wandb_logger
            
            # Get device
            device = self.config.runtime.get('device', 'auto')
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            self.config.device = torch.device(device)
            
            # Initialize WandB logger
            if self.config.runtime.get('wandb', {}).get('enable', True):
                self.wandb_logger = create_wandb_logger(self.config)
            else:
                self.wandb_logger = type('Logger', (), {'enabled': False, 'log_training_step': lambda *args: None, 
                                                       'log_evaluation_step': lambda *args: None, 'watch_model': lambda *args: None})()
            
            # Extract data from loaded_data
            STATE_train = self.loaded_data['STATE_train']
            BHP_train = self.loaded_data['BHP_train']
            Yobs_train = self.loaded_data['Yobs_train']
            STATE_eval = self.loaded_data['STATE_eval']
            BHP_eval = self.loaded_data['BHP_eval']
            Yobs_eval = self.loaded_data['Yobs_eval']
            dt_train = self.loaded_data['dt_train']
            dt_eval = self.loaded_data['dt_eval']
            
            metadata = self.loaded_data['metadata']
            num_train = metadata.get('num_train', 0)
            num_well = metadata.get('num_well', 0)
            
            # Validate loaded data n_steps matches config n_steps
            loaded_nsteps = metadata.get('nsteps', None)
            config_nsteps = self.config.training['nsteps']
            if loaded_nsteps is not None and loaded_nsteps != config_nsteps:
                raise ValueError(
                    f"Data preprocessing used n_steps={loaded_nsteps}, but training config has n_steps={config_nsteps}. "
                    f"Please reprocess data with n_steps={config_nsteps} or update config to match."
                )
            
            # Generate model filenames
            output_dir = './saved_models/'
            os.makedirs(output_dir, exist_ok=True)
            encoder_file = output_dir + self.config.create_model_filename('encoder', num_train, num_well)
            decoder_file = output_dir + self.config.create_model_filename('decoder', num_train, num_well)
            transition_file = output_dir + self.config.create_model_filename('transition', num_train, num_well)
            
            print(f"📁 Model files will be saved to:")
            print(f"   Encoder: {encoder_file}")
            print(f"   Decoder: {decoder_file}")
            print(f"   Transition: {transition_file}")
            
            # Initialize model
            print("\n🔧 Initializing model...")
            self.my_rom = ROMWithE2C(self.config).to(self.config.device)
            self.wandb_logger.watch_model(self.my_rom)
            print(f"✅ {self.config.model['method']} model initialized on {device}")
            
            # Setup schedulers
            num_batch = int(num_train / self.config.training['batch_size'])
            total_training_steps = num_batch * self.config.training['epoch']
            self.my_rom.setup_schedulers_with_steps(total_training_steps)
            
            # Training loop
            best_loss = 1.0e9
            best_observation_loss = 1.0e9
            best_reconstruction_loss = 1.0e9
            best_model_criterion = self.config.runtime.get('best_model_criterion', 'total_loss')
            global_step = 0
            
            print(f"\n📊 Training Configuration:")
            print(f"   Epochs: {self.config.training['epoch']}")
            print(f"   Batch size: {self.config.training['batch_size']}")
            print(f"   Batches per epoch: {num_batch}")
            print(f"   Learning rate: {self.config.training['learning_rate']}")
            print(f"   Best model criterion: {best_model_criterion}")
            print(f"\n🎯 Starting training loop...\n")
            
            # Time the training phase
            with Timer("training", log_dir='./timing_logs/') as timer:
                for e in range(self.config.training['epoch']):
                    for ib in range(num_batch):
                        ind0 = ib * self.config.training['batch_size']
                        
                        X_batch = [state[ind0:ind0+self.config.training['batch_size'], ...] for state in STATE_train]
                        U_batch = [bhp[ind0:ind0+self.config.training['batch_size'], ...] for bhp in BHP_train]
                        Y_batch = [yobs[ind0:ind0+self.config.training['batch_size'], ...] for yobs in Yobs_train]
                        dt_batch = dt_train[ind0:ind0+self.config.training['batch_size'], ...]
                        
                        inputs = (X_batch, U_batch, Y_batch, dt_batch)
                        self.my_rom.update(inputs)
                        
                        global_step += 1
                        self.wandb_logger.log_training_step(self.my_rom, e+1, ib+1, global_step)
                        
                        if ib % self.config.runtime['print_interval'] == 0:
                            print(f'Epoch {e+1}/{self.config.training["epoch"]}, Batch {ib+1}/{num_batch}, Loss {self.my_rom.train_loss:.6f}')
                            
                            # Evaluate
                            X_batch_eval = [state for state in STATE_eval]
                            U_batch_eval = [bhp for bhp in BHP_eval]
                            Y_batch_eval = [yobs for yobs in Yobs_eval]
                            test_inputs = (X_batch_eval, U_batch_eval, Y_batch_eval, dt_eval)
                            self.my_rom.evaluate(test_inputs)
                            
                            self.wandb_logger.log_evaluation_step(self.my_rom, e+1, global_step)
                            
                            print(f'       • Reconstruction: {self.my_rom.get_test_reconstruction_loss():.6f}')
                            print(f'       • Transition: {self.my_rom.get_test_transition_loss():.6f}')
                            print(f'       • Observation: {self.my_rom.get_test_observation_loss():.6f}')
                    
                    # Step scheduler
                    current_eval_loss = self.my_rom.test_loss.item() if hasattr(self.my_rom.test_loss, 'item') else float(self.my_rom.test_loss)
                    self.my_rom.step_scheduler_on_epoch(validation_loss=current_eval_loss)
                    
                    print(f'Epoch {e + 1}/{self.config.training["epoch"]}, Train: {self.my_rom.train_loss:.6f}, Eval: {self.my_rom.test_loss:.6f}')
                    
                    # Save best model
                    if self.config.runtime['save_best_model']:
                        should_save = False
                        if best_model_criterion == 'observation_loss':
                            current_obs_loss = self.my_rom.get_test_observation_loss()
                            if current_obs_loss < best_observation_loss:
                                best_observation_loss = current_obs_loss
                                should_save = True
                        elif best_model_criterion == 'reconstruction_loss':
                            current_recon_loss = self.my_rom.get_test_reconstruction_loss()
                            if current_recon_loss < best_reconstruction_loss:
                                best_reconstruction_loss = current_recon_loss
                                should_save = True
                        else:  # total_loss
                            if self.my_rom.test_loss < best_loss:
                                best_loss = self.my_rom.test_loss
                                should_save = True
                        
                        if should_save:
                            self.my_rom.model.save_weights_to_file(encoder_file, decoder_file, transition_file)
                            print(f"💾 Saved best model (criterion: {best_model_criterion})")
                
                # Collect metadata for timing log (including final loss values)
                metadata = collect_training_metadata(self.config, self.loaded_data)
                metadata['final_loss'] = float(self.my_rom.test_loss.item() if hasattr(self.my_rom.test_loss, 'item') else float(self.my_rom.test_loss))
                metadata['final_reconstruction_loss'] = float(self.my_rom.get_test_reconstruction_loss())
                metadata['final_transition_loss'] = float(self.my_rom.get_test_transition_loss())
                metadata['final_observation_loss'] = float(self.my_rom.get_test_observation_loss())
                metadata['best_loss'] = float(best_loss.item() if hasattr(best_loss, 'item') else float(best_loss))
                metadata['best_observation_loss'] = float(best_observation_loss)
                metadata['best_reconstruction_loss'] = float(best_reconstruction_loss)
                timer.metadata = metadata
            
            print(f"\n✅ Training completed!")
            print(f"   Final loss: {self.my_rom.test_loss:.6f}")
            print(f"   Best {best_model_criterion}: {best_observation_loss if best_model_criterion == 'observation_loss' else (best_reconstruction_loss if best_model_criterion == 'reconstruction_loss' else best_loss):.6f}")
    
    def display(self):
        """Display the dashboard"""
        if not WIDGETS_AVAILABLE:
            print("⚠️ Interactive widgets not available. Please install ipywidgets: pip install ipywidgets")
            return None
        display(self.main_widget)
        return self.main_widget


def create_training_dashboard(config_path='config.yaml'):
    """
    Create and display the training dashboard
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        TrainingDashboard instance
    """
    dashboard = TrainingDashboard(config_path=config_path)
    dashboard.display()
    return dashboard

