"""
RL Visualization Dashboard
Full implementation for RL training results visualization
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import time
import io
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Ensure ROM_Refactored is in path
rom_refactored_path = Path(__file__).parent.parent.parent / 'ROM_Refactored'
if str(rom_refactored_path) not in sys.path:
    sys.path.insert(0, str(rom_refactored_path))

# Import RL_Refactored modules
from RL_Refactored.training import EnhancedTrainingOrchestrator
from RL_Refactored.utilities import Config

# Try to import widgets
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None
    HTML = None


class ScientificVisualization:
    """
    Enhanced scientific visualization dashboard for RL training results
    Supports episode selection and comprehensive analysis
    """
    
    def __init__(self, training_orchestrator, config):
        """
        Initialize with training orchestrator
        
        Args:
            training_orchestrator: EnhancedTrainingOrchestrator with episode data
            config: Configuration object
        """
        self.training_orchestrator = training_orchestrator
        self.config = config
        
        # Get episode statistics
        self.episode_stats = training_orchestrator.get_episode_summary_stats()
        
        # Current selected episode (defaults to best episode)
        self.current_episode = training_orchestrator.get_best_episode_number()
        
        # Ensure the current episode exists
        available_episodes = training_orchestrator.get_available_episodes()
        if self.current_episode not in available_episodes and available_episodes:
            print(f"⚠️ Warning: Best episode {self.current_episode} not in available episodes")
            self.current_episode = max(available_episodes) if available_episodes else -1
        
        # Load current episode data
        self.data = training_orchestrator.get_episode_data(self.current_episode)
        if self.data is None:
            self.data = training_orchestrator.get_best_episode_data()
        
        # Identify wells
        self.wells = self._identify_wells()
        
        # Check spatial data availability
        self.spatial_data_available = self._check_spatial_data_availability()
        
        # Load normalization parameters for denormalization
        self.norm_params = self.training_orchestrator._load_latest_preprocessing_parameters() if hasattr(self.training_orchestrator, '_load_latest_preprocessing_parameters') else None
        if self.norm_params is None:
            print("⚠️ Could not load preprocessing normalization parameters for visualization.")
        
        # Publication-quality plot settings
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'grid.alpha': 0.3,
            'axes.formatter.limits': [-3, 3]
        })
        
        print(f"✅ Visualization dashboard initialized")
        print(f"   📊 Episodes available: {len(available_episodes)}")
        print(f"   🏆 Best episode: {self.current_episode}")
        print(f"   🗺️ Spatial data: {'Available' if self.spatial_data_available else 'Not available'}")
    
    def _identify_wells(self):
        """Identify available wells from episode data by analyzing action structure"""
        wells = {}
        
        if not self.data or not self.data.get('actions'):
            return wells
        
        # Analyze first action to identify well structure
        first_action = self.data['actions'][0]
        
        if isinstance(first_action, dict):
            for key in first_action.keys():
                if 'BHP_psi' in key:
                    # Extract well name (e.g., "P1_BHP_psi" -> "P1")
                    well_name = key.split('_')[0]
                    wells[well_name] = {'type': 'Producer', 'actions': ['BHP'], 'observations': ['Water', 'Gas']}
                elif 'Gas_ft3day' in key:
                    # Extract well name (e.g., "I1_Gas_ft3day" -> "I1")
                    well_name = key.split('_')[0]
                    if well_name not in wells:  # Don't overwrite if already added
                        wells[well_name] = {'type': 'Injector', 'actions': ['Gas'], 'observations': ['BHP']}
        
        return wells
    
    def _check_spatial_data_availability(self):
        """Check if spatial reservoir state data is available for visualization"""
        if self.data is None:
            return False
        spatial_states = self.data.get('spatial_states', [])
        return bool(spatial_states and len(spatial_states) > 0)
    
    def set_current_episode(self, episode_number):
        """Set the current episode for visualization"""
        self.current_episode = episode_number
        self.data = self.training_orchestrator.get_episode_data(episode_number)
        if self.data:
            self.wells = self._identify_wells()
        self.spatial_data_available = self._check_spatial_data_availability()
        
        print(f"✅ Switched to episode {episode_number}")
        print(f"   💰 Reward: {self.data.get('total_reward', 'Unknown'):.3f}" if self.data else "   ⚠️ No data")
        print(f"   📈 Spatial data: {'✅ Available' if self.spatial_data_available else '❌ Not available'}")
        
        return self.data is not None
    
    def _get_episode_info_html(self):
        """Generate HTML summary for current episode display"""
        if not self.data:
            return "<b>No episode data available</b>"
            
        reward = self.data.get('total_reward', 0)
        is_best = self.current_episode == self.episode_stats.get('best_episode')
        spatial_available = "✅" if self.spatial_data_available else "❌"
        
        html = f"""
        <b>Episode {self.current_episode}</b> {'🏆 BEST' if is_best else ''}<br/>
        <b>Reward:</b> {reward:.3f}<br/>
        <b>Spatial Data:</b> {spatial_available}<br/>
        <b>Steps:</b> {len(self.data.get('timesteps', []))}
        """
        return html
    
    def get_available_episodes(self):
        """Get list of available episodes for selection"""
        return self.training_orchestrator.get_available_episodes()
    
    def create_interactive_episode_selector(self):
        """Create interactive episode selector interface with tabbed visualization panels"""
        if not WIDGETS_AVAILABLE:
            print("❌ Interactive widgets not available. Please install ipywidgets.")
            return None
        
        available_episodes = self.get_available_episodes()
        if not available_episodes:
            print("❌ No episodes available for selection")
            return None
        
        # Create episode selector dropdown
        episode_options = [(f"Episode {ep}", ep) for ep in available_episodes]
        
        episode_selector = widgets.Dropdown(
            options=episode_options,
            value=self.current_episode,
            description='Episode:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Episode info display
        episode_info = widgets.HTML(
            value=self._get_episode_info_html(),
            layout=widgets.Layout(width='400px')
        )
        
        # Update function for episode selection
        def update_episode(change):
            new_episode = change['new']
            if self.set_current_episode(new_episode):
                episode_info.value = self._get_episode_info_html()
                # Refresh all tabs with new episode data
                if hasattr(self, '_refresh_well_tab'):
                    self._refresh_well_tab()
                if hasattr(self, '_refresh_spatial_tab'):
                    self._refresh_spatial_tab()
                if hasattr(self, '_refresh_economic_tab'):
                    self._refresh_economic_tab()
                if hasattr(self, '_refresh_training_tab'):
                    self._refresh_training_tab()
                if hasattr(self, '_refresh_rl_animation_tab'):
                    self._refresh_rl_animation_tab()
        
        episode_selector.observe(update_episode, names='value')
        
        # Create tabbed interface with analysis panels
        # Well Analysis Tab Content
        well_analysis_content = self._create_well_analysis_tab()
        
        # Spatial Analysis Tab Content  
        spatial_content = self._create_spatial_analysis_tab()
        
        # Economic Analysis Tab Content
        economic_content = self._create_economic_analysis_tab()
        
        # RL Training Performance Tab Content
        training_content = self._create_training_performance_tab()
        
        # RL Training Animation content
        rl_animation_content = self._create_rl_training_animation_tab()
        
        # Episode Progression Animation content
        episode_progression_content = self._create_episode_progression_animation_tab()
        
        # Main tabs with analysis panels
        main_tabs = widgets.Tab()
        main_tabs.children = [well_analysis_content, spatial_content, economic_content, training_content, rl_animation_content, episode_progression_content]
        main_tabs.set_title(0, '🏭 Well Analysis')
        main_tabs.set_title(1, '🗺️ Spatial Analysis')
        main_tabs.set_title(2, '💰 Economic Analysis')
        main_tabs.set_title(3, 'RL Training Performance')
        main_tabs.set_title(4, '🎬 RL Training Animation')
        main_tabs.set_title(5, '📈 Episode Progression')
        
        # Layout with episode selector and info header
        header = widgets.HBox([episode_selector, episode_info])
        
        return widgets.VBox([
            widgets.HTML("<h3>🔬 Enhanced RL Results Dashboard</h3>"),
            header,
            main_tabs
        ])
    
    def _create_training_performance_tab(self):
        """Create RL training performance analysis tab with metrics visualization"""
        if not WIDGETS_AVAILABLE:
            return widgets.HTML("❌ Interactive widgets not available")
        
        # Create containers for dynamic content
        content_container = widgets.VBox()
        
        def refresh_training_tab():
            """Refresh training performance tab content"""
            training_output = widgets.Output()
            
            with training_output:
                clear_output(wait=True)
                # Get training metrics from orchestrator
                training_metrics = self.training_orchestrator.training_metrics if hasattr(self.training_orchestrator, 'training_metrics') else None
                if training_metrics:
                    self._plot_training_performance(training_metrics)
                else:
                    self._plot_training_performance()
            
            content_container.children = [
                widgets.HTML("<h4>📊 RL Training Performance</h4>"),
                widgets.HTML("<p><i>Comprehensive training metrics and statistics</i></p>"),
                training_output
            ]
        
        # Store refresh function
        self._refresh_training_tab = refresh_training_tab
        
        # Initial refresh
        refresh_training_tab()
        
        return content_container
    
    def _create_well_analysis_tab(self):
        """Create well analysis tab content with interactive well selection"""
        if not WIDGETS_AVAILABLE:
            return widgets.HTML("❌ Interactive widgets not available")
        
        # Create containers for dynamic content
        well_content_container = widgets.VBox()
        
        def refresh_well_tab():
            """Refresh well analysis tab content for current episode"""
            # Well selection dropdown
            well_options = [(f"{well} ({info['type']})", well) for well, info in self.wells.items()]
            
            if not well_options:
                well_content_container.children = [widgets.HTML("<p>❌ No well data available</p>")]
                return
            
            well_selector = widgets.Dropdown(
                options=well_options,
                value=well_options[0][1] if well_options else None,
                description='Well:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='300px')
            )
            
            well_output = widgets.Output()
            
            def update_well_plots(change):
                with well_output:
                    clear_output(wait=True)
                    well_name = change['new']
                    self.create_well_analysis_plots(well_name, ['Actions', 'Observations'], 'years')
            
            well_selector.observe(update_well_plots, names='value')
            
            # Initial plot
            with well_output:
                if well_options:
                    self.create_well_analysis_plots(well_options[0][1], ['Actions', 'Observations'], 'years')
            
            well_content_container.children = [
                widgets.HTML("<h3>🔬 Scientific Well Analysis</h3>"),
                widgets.HTML("<p><i>Interactive analysis of best-performing episode</i></p>"),
                well_selector,
                well_output
            ]
        
        # Store refresh function for episode change callback
        self._refresh_well_tab = refresh_well_tab
        
        # Initial refresh
        refresh_well_tab()
        
        return well_content_container
    
    def _create_economic_analysis_tab(self):
        """Create economic analysis tab content with cashflow breakdown"""
        if not WIDGETS_AVAILABLE:
            return widgets.HTML("❌ Interactive widgets not available")
        
        # Create containers for dynamic content
        content_container = widgets.VBox()
        
        def refresh_economic_tab():
            """Refresh economic analysis tab content"""
            economic_output = widgets.Output()
            
            with economic_output:
                clear_output(wait=True)
                self._plot_economic_analysis()
            
            content_container.children = [
                widgets.HTML("<h4>💰 Economic Analysis</h4>"),
                widgets.HTML("<p><i>Economic breakdown and cashflow analysis</i></p>"),
                economic_output
            ]
        
        # Store refresh function
        self._refresh_economic_tab = refresh_economic_tab
        
        # Initial refresh
        refresh_economic_tab()
        
        return content_container
    
    def _create_spatial_analysis_tab(self):
        """Create spatial analysis tab content with interactive spatial visualization"""
        if not WIDGETS_AVAILABLE:
            return widgets.HTML("❌ Interactive widgets not available")
        
        # Create containers for dynamic content
        content_container = widgets.VBox()
        
        def refresh_spatial_tab():
            """Refresh spatial tab content for current episode"""
            if not self.spatial_data_available:
                content_container.children = [widgets.HTML(
                    "<h4>❌ Spatial Data Not Available</h4>"
                    "<p>Spatial visualization requires:</p>"
                    "<ul>"
                    "<li>✅ Spatial state capture enabled in config</li>"
                    "<li>❌ Spatial data captured during training</li>"
                    "</ul>"
                    "<p>To enable spatial capture, set <code>capture_spatial_states: true</code> in config.yaml</p>"
                )]
                return
            
            # Get FRESH spatial data for CURRENT episode every time
            spatial_data = self.training_orchestrator.get_spatial_data_for_episode(self.current_episode)
            
            if not spatial_data:
                content_container.children = [widgets.HTML(f"❌ No spatial data available for episode {self.current_episode}")]
                return
            
            # Timestep selection
            available_timesteps = spatial_data['available_timesteps']
            timestep_options = [(f"Step {ts}", ts) for ts in available_timesteps]
            
            timestep_selector = widgets.Dropdown(
                options=timestep_options,
                value=available_timesteps[0] if available_timesteps else None,
                description='Timestep:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            
            # Layer selection (Z-axis)
            max_layers = spatial_data.get('spatial_shape', [0,0,0])[2] if spatial_data.get('spatial_shape') else 0
            layer_options = [(f"Layer {i}", i) for i in range(max_layers)]
            
            layer_selector = widgets.Dropdown(
                options=layer_options,
                value=max_layers//2 if max_layers > 0 else 0,  # Default to middle layer
                description='Layer:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            
            # Field/Channel selection - Use actual training channels
            try:
                field_names, field_units, training_channels = self._get_rl_training_channel_info()
                num_channels = spatial_data.get('num_channels', len(field_names))
                channel_options = [(field_names[i], i) for i in range(min(num_channels, len(field_names)))]
            except Exception as e:
                # Fallback if method fails
                available_channels = ['Pressure', 'Gas Saturation', 'Water Saturation']
                num_channels = spatial_data.get('num_channels', len(available_channels))
                channel_options = [(field, i) for i, field in enumerate(available_channels[:num_channels])]
            
            channel_selector = widgets.Dropdown(
                options=channel_options,
                value=0,  # Default to first channel
                description='Field:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            
            # Color scale controls for user adjustment
            vmin_input = widgets.FloatText(
                value=0.0,
                description='Min:',
                style={'description_width': '40px'},
                layout=widgets.Layout(width='120px')
            )
            
            vmax_input = widgets.FloatText(
                value=1.0,
                description='Max:',
                style={'description_width': '40px'},
                layout=widgets.Layout(width='120px')
            )
            
            auto_scale_button = widgets.Button(
                description="Auto",
                button_style='info',
                layout=widgets.Layout(width='60px'),
                tooltip="Auto-adjust color scale to data range"
            )
            
            color_scale_label = widgets.HTML(
                value="<b>Color Scale:</b>",
                layout=widgets.Layout(width='80px')
            )
            
            # Output area for spatial plots
            spatial_output = widgets.Output()
            
            # Update function that gets FRESH spatial data every time
            def update_spatial_plot(change=None, auto_scale=True):
                timestep = timestep_selector.value
                layer = layer_selector.value
                channel = channel_selector.value
                
                # Get CURRENT episode's spatial data, not cached data
                current_spatial_data = self.training_orchestrator.get_spatial_data_for_episode(self.current_episode)
                
                if auto_scale and current_spatial_data:
                    # AUTO-CALCULATE color scale based on actual data
                    vmin_calc, vmax_calc = self._calculate_data_color_range(
                        current_spatial_data, timestep, layer, channel
                    )
                    vmin_input.value = vmin_calc
                    vmax_input.value = vmax_calc
                
                # Get user-specified color scale
                vmin = vmin_input.value
                vmax = vmax_input.value
                
                with spatial_output:
                    spatial_output.clear_output()
                    if current_spatial_data:
                        self.plot_spatial_visualization(timestep, layer, channel, current_spatial_data, 
                                                      user_vmin=vmin, user_vmax=vmax)
                    else:
                        print(f"❌ No spatial data available for episode {self.current_episode}")
            
            # Auto-scale button callback
            def on_auto_scale_click(b):
                update_spatial_plot(auto_scale=True)
            
            auto_scale_button.on_click(on_auto_scale_click)
            
            # Manual color scale change callbacks
            def on_color_scale_change(change):
                update_spatial_plot(auto_scale=False)  # Don't auto-scale when user manually changes
            
            vmin_input.observe(on_color_scale_change, names='value')
            vmax_input.observe(on_color_scale_change, names='value')
            
            # Observe selectors for auto-scaling
            def on_selector_change(change):
                update_spatial_plot(auto_scale=True)  # Auto-scale when field/layer/timestep changes
                
            timestep_selector.observe(on_selector_change, names='value')
            layer_selector.observe(on_selector_change, names='value')
            channel_selector.observe(on_selector_change, names='value')
            
            # Initialize with default values
            if available_timesteps and max_layers > 0:
                update_spatial_plot()
            
            # Layout
            controls_row1 = widgets.HBox([timestep_selector, layer_selector, channel_selector])
            controls_row2 = widgets.HBox([color_scale_label, vmin_input, vmax_input, auto_scale_button])
            
            # Update container content
            content_container.children = [
                widgets.HTML("<h4>🗺️ Spatial Reservoir State Visualization</h4>"),
                controls_row1,
                controls_row2,
                spatial_output
            ]
        
        # Store refresh function for episode change callback
        self._refresh_spatial_tab = refresh_spatial_tab
        
        # Initial refresh
        refresh_spatial_tab()
        
        return content_container
    
    def _plot_training_performance(self, training_metrics=None):
        """Plot comprehensive RL training performance analysis with metrics visualization"""
        try:
            # Get training metrics
            if training_metrics is None:
                if hasattr(self.training_orchestrator, 'training_metrics') and self.training_orchestrator.training_metrics:
                    training_metrics = self.training_orchestrator.training_metrics
                else:
                    # Fallback: generate from stored episodes
                    if not self.training_orchestrator.stored_episodes:
                        print("No training data available")
                        return
                    episodes = sorted(self.training_orchestrator.stored_episodes.keys())
                    episode_rewards = [self.training_orchestrator.stored_episodes[ep]['total_reward'] for ep in episodes]
                    window = min(10, len(episode_rewards))
                    avg_rewards = [np.mean(episode_rewards[max(0, i-window+1):i+1]) for i in range(len(episode_rewards))]
                    training_metrics = {
                        'episodes': episodes,
                        'episode_rewards': episode_rewards,
                        'avg_rewards': avg_rewards
                    }
            
            episodes = training_metrics['episodes']
            episode_rewards = training_metrics['episode_rewards']
            avg_rewards = training_metrics['avg_rewards']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
            
            # 1. Training Progress Over Episodes
            ax1.plot(episodes, episode_rewards, label='Episode Reward', alpha=0.7, color='blue', linewidth=1.5)
            ax1.plot(episodes, avg_rewards, label='Avg Reward (10 episodes)', linewidth=3, color='red')
            ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Reward', fontsize=12, fontweight='bold')
            ax1.set_title('RL Training Progress', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add best reward annotation
            if episode_rewards:
                best_reward = max(episode_rewards)
                best_episode_idx = episode_rewards.index(best_reward)
                best_episode = episodes[best_episode_idx]
                ax1.annotate(f'Best: {best_reward:.3f}\n(Episode {best_episode})', 
                           xy=(best_episode, best_reward), xytext=(10, 10),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # 2. Reward Distribution
            ax2.hist(episode_rewards, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Reward', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Policy Loss (if available)
            policy_losses = training_metrics.get('policy_losses', [])
            if policy_losses:
                ax3.plot(episodes[:len(policy_losses)], policy_losses, label='Policy Loss', color='purple', alpha=0.7)
                ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
                ax3.set_title('Policy Loss Over Episodes', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Policy Loss Data Not Available', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12, color='gray')
            
            # 4. Q-Value Loss (if available)
            q_losses = training_metrics.get('q_losses', [])
            if q_losses:
                ax4.plot(episodes[:len(q_losses)], q_losses, label='Q-Value Loss', color='orange', alpha=0.7)
                ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
                ax4.set_title('Q-Value Loss Over Episodes', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'Q-Value Loss Data Not Available', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, fontsize=12, color='gray')
            
            plt.suptitle(f"RL Training Performance (Episode {self.current_episode})", fontsize=16, fontweight='bold')
            display(fig)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error plotting training performance: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_denormalized_actions(self):
        """Get actions for all timesteps - data is ALREADY in physical units from RL environment"""
        # CRITICAL: The training orchestrator stores actions that are ALREADY denormalized 
        # to physical units by the Environment's step() method using exact preprocessing parameters.
        # NO FURTHER DENORMALIZATION IS NEEDED - this would cause double denormalization!
        
        if not self.data or not self.data.get('actions'):
            return []
        
        processed_actions = []
        
        for action_step in self.data['actions']:
            if isinstance(action_step, dict):
                # Data is already in correct format with physical units
                # Keys like "P1_BHP_psi", "I1_Gas_ft3day"
                processed_actions.append(action_step)
            else:
                # Unexpected format - this shouldn't happen with current training orchestrator
                print(f"⚠️ Warning: Unexpected action format: {type(action_step)}")
                print(f"   Expected dict with well keys, got: {action_step}")
                processed_actions.append({})
        
        return processed_actions
    
    def _get_denormalized_observations(self):
        """Get observations for all timesteps - data is ALREADY in physical units from RL environment"""
        # CRITICAL: The training orchestrator stores observations that are ALREADY denormalized 
        # to physical units by the Environment's step() method using exact preprocessing parameters.
        # NO FURTHER DENORMALIZATION IS NEEDED - this would cause double denormalization!
        
        if not self.data or not self.data.get('observations'):
            return []
        
        processed_observations = []
        
        for obs_step in self.data['observations']:
            if isinstance(obs_step, dict):
                # Data is already in correct format with physical units
                # Keys like "P1_Water_ft3day", "P1_Gas_ft3day", "I1_BHP_psi"
                processed_observations.append(obs_step)
            else:
                # Unexpected format - this shouldn't happen with current training orchestrator
                print(f"⚠️ Warning: Unexpected observation format: {type(obs_step)}")
                print(f"   Expected dict with well keys, got: {obs_step}")
                processed_observations.append({})
        
        return processed_observations
    
    def create_well_analysis_plots(self, well_name, plot_types=['Actions', 'Observations'], time_unit='years'):
        """Create comprehensive well analysis plots with actions and observations"""
        
        if well_name not in self.wells:
            print(f"❌ Well {well_name} not found in data")
            return
        
        well_info = self.wells[well_name]
        
        # Calculate time values
        num_steps = len(self.data.get('actions', []))
        if time_unit == 'years':
            time_values = list(range(num_steps))  # Each step = 1 year
            time_label = 'Time (years)'
        else:
            time_values = list(range(num_steps))
            time_label = 'Time (steps)'
        
        # Create subplots based on plot types
        num_plots = len(plot_types)
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
        if num_plots == 1:
            axes = [axes]
        
        fig.suptitle(f'{well_name} ({well_info["type"]}) - Episode {self.current_episode}', 
                     fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Actions plot
        if 'Actions' in plot_types:
            ax = axes[plot_idx]
            self._plot_well_actions(ax, well_name, well_info, time_values, time_label)
            plot_idx += 1
        
        # Observations plot  
        if 'Observations' in plot_types:
            ax = axes[plot_idx]
            self._plot_well_observations(ax, well_name, well_info, time_values, time_label)
            plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self._print_well_summary(well_name, well_info)
    
    def _plot_well_actions(self, ax, well_name, well_info, time_values, time_label):
        """Plot well actions over time - data is already in physical units from RL training"""
        
        # Get actions - data is already in physical units (no denormalization needed)
        physical_actions = self._get_denormalized_actions()
        
        if well_info['type'] == 'Producer':
            # Plot BHP control in psi (already physical units)
            bhp_key = f"{well_name}_BHP_psi"
            bhp_values = [action.get(bhp_key, 0) for action in physical_actions]
            
            if any(val > 0 for val in bhp_values):
                ax.plot(time_values, bhp_values, 'b-', marker='s', markersize=4, 
                       label='BHP Control', linewidth=2)
                ax.set_ylabel('BHP (psi)', fontsize=14, color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                
                # Add reasonable range shading based on actual data
                bhp_min = min(bhp_values) * 0.95 if bhp_values else 0
                bhp_max = max(bhp_values) * 1.05 if bhp_values else 1000
                ax.axhspan(bhp_min, bhp_max, alpha=0.1, color='blue', label='BHP Range')
                ax.set_title(f'{well_name} - BHP Control Profile (Physical Units)', fontsize=16, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No BHP action data available', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title(f'{well_name} - BHP Control Profile (No Data)', fontsize=16, fontweight='bold')
            
        else:  # Injector
            # Plot gas injection rate in ft³/day (already physical units)
            gas_key = f"{well_name}_Gas_ft3day"
            gas_values = [action.get(gas_key, 0) for action in physical_actions]
            
            if any(val > 0 for val in gas_values):
                ax.plot(time_values, gas_values, 'g-', marker='o', markersize=4,
                       label='Gas Injection', linewidth=2)
                ax.set_ylabel('Gas Injection (ft³/day)', fontsize=14, color='green')
                ax.tick_params(axis='y', labelcolor='green')
                
                # Add reasonable range shading
                gas_min = min(gas_values) * 0.95 if gas_values else 0
                gas_max = max(gas_values) * 1.05 if gas_values else 10000000
                ax.axhspan(gas_min, gas_max, alpha=0.1, color='green', label='Gas Injection Range')
                ax.set_title(f'{well_name} - Gas Injection Profile (Physical Units)', fontsize=16, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No gas injection data available',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title(f'{well_name} - Gas Injection Profile (No Data)', fontsize=16, fontweight='bold')
        
        ax.set_xlabel(time_label, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    def _plot_well_observations(self, ax, well_name, well_info, time_values, time_label):
        """Plot well observations - data is already in physical units from RL training"""
        
        # Get observations - data is already in physical units (no denormalization needed)
        physical_observations = self._get_denormalized_observations()
        
        if well_info['type'] == 'Producer':
            # Plot water and gas production with dual y-axes
            water_key = f"{well_name}_Water_bblday"
            gas_key = f"{well_name}_Gas_ft3day"
            
            water_values = [obs.get(water_key, 0) for obs in physical_observations]
            gas_values = [obs.get(gas_key, 0) for obs in physical_observations]
            
            if any(val > 0 for val in water_values) or any(val > 0 for val in gas_values):
                # Water production on left axis
                ax1 = ax
                line1 = ax1.plot(time_values, water_values, 'b-', marker='^', markersize=4,
                                label='Water Production', linewidth=2)
                ax1.set_xlabel(time_label, fontsize=14)
                ax1.set_ylabel('Water Production (bbl/day)', fontsize=14, color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Gas production on right axis
                ax2 = ax1.twinx()
                line2 = ax2.plot(time_values, gas_values, 'r-', marker='o', markersize=4,
                                label='Gas Production', linewidth=2)
                ax2.set_ylabel('Gas Production (ft³/day)', fontsize=14, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left', fontsize=11)
                
                ax1.set_title(f'{well_name} - Production Profile (Physical Units)', fontsize=16, fontweight='bold')
                ax1.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No production data available',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title(f'{well_name} - Production Profile (No Data)', fontsize=16, fontweight='bold')
        
        else:  # Injector
            # Plot BHP observation
            bhp_key = f"{well_name}_BHP_psi"
            bhp_values = [obs.get(bhp_key, 0) for obs in physical_observations]
            
            if any(val > 0 for val in bhp_values):
                ax.plot(time_values, bhp_values, 'purple', marker='s', markersize=4,
                       label='BHP Observation', linewidth=2)
                ax.set_ylabel('BHP (psi)', fontsize=14, color='purple')
                ax.tick_params(axis='y', labelcolor='purple')
                ax.set_title(f'{well_name} - BHP Observation Profile (Physical Units)', fontsize=16, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No BHP observation data available',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title(f'{well_name} - BHP Observation Profile (No Data)', fontsize=16, fontweight='bold')
            
            ax.set_xlabel(time_label, fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
    
    def _print_well_summary(self, well_name, well_info):
        """Print summary statistics for a well"""
        physical_actions = self._get_denormalized_actions()
        physical_observations = self._get_denormalized_observations()
        
        print(f"\n📊 {well_name} ({well_info['type']}) Summary:")
        print("=" * 60)
        
        if well_info['type'] == 'Producer':
            bhp_key = f"{well_name}_BHP_psi"
            water_key = f"{well_name}_Water_bblday"
            gas_key = f"{well_name}_Gas_ft3day"
            
            bhp_values = [action.get(bhp_key, 0) for action in physical_actions]
            water_values = [obs.get(water_key, 0) for obs in physical_observations]
            gas_values = [obs.get(gas_key, 0) for obs in physical_observations]
            
            if bhp_values:
                print(f"BHP Control: Min={min(bhp_values):.1f}, Max={max(bhp_values):.1f}, Mean={np.mean(bhp_values):.1f} psi")
            if water_values:
                print(f"Water Production: Min={min(water_values):.1f}, Max={max(water_values):.1f}, Mean={np.mean(water_values):.1f} bbl/day")
            if gas_values:
                print(f"Gas Production: Min={min(gas_values):.1f}, Max={max(gas_values):.1f}, Mean={np.mean(gas_values):.1f} ft³/day")
        
        else:  # Injector
            gas_key = f"{well_name}_Gas_ft3day"
            bhp_key = f"{well_name}_BHP_psi"
            
            gas_values = [action.get(gas_key, 0) for action in physical_actions]
            bhp_values = [obs.get(bhp_key, 0) for obs in physical_observations]
            
            if gas_values:
                print(f"Gas Injection: Min={min(gas_values):.1f}, Max={max(gas_values):.1f}, Mean={np.mean(gas_values):.1f} ft³/day")
            if bhp_values:
                print(f"BHP Observation: Min={min(bhp_values):.1f}, Max={max(bhp_values):.1f}, Mean={np.mean(bhp_values):.1f} psi")
        
        print("=" * 60)
    
    def _plot_economic_analysis(self):
        """
        Plot Complete Project Lifecycle Analysis with NPV analysis
        
        Creates 4 analysis plots:
        1. Complete Project Lifecycle Cashflow Analysis
        2. Project Phase Analysis
        3. Annual Cashflow Breakdown
        4. NPV Analysis with multiple discount rates
        """
        if self.data is None:
            print("No episode data available")
            return
        
        economic_breakdown = self.data.get('economic_breakdown', [])
        
        if not economic_breakdown:
            print("No economic breakdown data available")
            return
        
        # Get pre-project parameters from config
        try:
            years_before = self.config.rl_model.get('economics', {}).get('years_before_project_start', 5)
            cost_per_year = self.config.rl_model.get('economics', {}).get('capital_cost_per_year', 100000000.0)  # $100M
        except:
            years_before = 5
            cost_per_year = 100000000.0
        
        total_capital_cost = years_before * cost_per_year
        
        # Extract operational cashflow from economic breakdown
        if 'operational_cashflow' in economic_breakdown[0]:
            operational_cashflows = [bd.get('operational_cashflow', 0) for bd in economic_breakdown]
        else:
            # Fallback: use net_step_cashflow
            operational_cashflows = [bd.get('net_step_cashflow', 0) for bd in economic_breakdown]
        
        max_steps = len(operational_cashflows)
        total_timeline_years = years_before + max_steps
        
        # Create pre-project cashflow timeline (in Million USD)
        pre_project_years = list(range(0, years_before + 1))
        pre_project_cashflow = [0]  # Start at zero
        for year in range(1, years_before + 1):
            pre_project_cashflow.append(-cost_per_year / 1e6)  # Convert to Million USD
        
        # Create operational cashflow timeline (in Million USD)
        operational_years = list(range(years_before, total_timeline_years + 1))
        operational_cashflow_millions = [cf / 1e6 for cf in operational_cashflows]  # Convert to Million USD
        
        # Complete timeline (pre-project + operational)
        all_years = list(range(0, total_timeline_years + 1))
        all_cashflows = pre_project_cashflow + operational_cashflow_millions
        
        # Calculate cumulative cashflow
        cumulative_cashflow = np.cumsum(all_cashflows)
        
        # Find break-even point
        break_even_year = None
        for i, cum_cf in enumerate(cumulative_cashflow):
            if cum_cf >= 0:
                break_even_year = i
                break
        
        # Calculate NPV at different discount rates
        discount_rates = [0.05, 0.08, 0.10, 0.12, 0.15]
        npv_curves = {}
        for rate in discount_rates:
            npv_values = []
            npv_cumulative = 0
            for year, cf in enumerate(all_cashflows):
                npv_cumulative += cf / ((1 + rate) ** year)
                npv_values.append(npv_cumulative)
            npv_curves[rate] = npv_values
        
        # Create 4-panel figure
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get episode reward for display
        episode_reward = self.data.get('total_reward', 0)
        
        # Main title
        fig.suptitle(f'Complete Project Lifecycle Analysis - Episode {self.current_episode}\n'
                    f'Total Investment: ${total_capital_cost/1e6:.0f}M | Episode Reward: {episode_reward:.3f}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Complete Project Lifecycle Cashflow Analysis (Top-Left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(all_years, cumulative_cashflow, 'r-', linewidth=3, label='Cumulative Net Cashflow')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Break-even Line')
        
        if break_even_year is not None:
            ax1.axvline(x=break_even_year, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax1.plot(break_even_year, 0, 'go', markersize=10)
            ax1.text(break_even_year, max(cumulative_cashflow) * 0.1, f'Break-even: Year {break_even_year}',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax1.axvline(x=years_before, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(years_before, max(cumulative_cashflow) * 0.9, 'Operations Start',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax1.set_xlabel('Project Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Cashflow (Million USD)', fontsize=12, fontweight='bold')
        ax1.set_title('Complete Project Lifecycle Cashflow Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 2. Project Phase Analysis (Top-Right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Pre-project phase cumulative
        pre_project_cumulative = np.cumsum(pre_project_cashflow)
        ax2.plot(pre_project_years, pre_project_cumulative, 'b-', linewidth=2, label='Pre-Project Phase')
        
        # Operational phase cumulative (starting from pre-project end)
        operational_cumulative_start = pre_project_cumulative[-1]
        operational_cumulative = [operational_cumulative_start]
        for cf in operational_cashflow_millions:
            operational_cumulative.append(operational_cumulative[-1] + cf)
        
        ax2.plot(operational_years, operational_cumulative, 'g-', linewidth=2, label='Operational Phase')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Break-even Line')
        ax2.axvline(x=years_before, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax2.text(years_before, max(operational_cumulative) * 0.9, 'Operations Start',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax2.set_xlabel('Project Year', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Cashflow (Million USD)', fontsize=12, fontweight='bold')
        ax2.set_title('Project Phase Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 3. Annual Cashflow Breakdown (Bottom-Left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Pre-project annual cashflow
        pre_project_annual = [-cost_per_year / 1e6] * years_before
        pre_project_year_labels = list(range(years_before))
        
        # Operational annual cashflow
        operational_year_labels = list(range(years_before, total_timeline_years))
        
        # Combine for bar chart
        all_annual_labels = list(range(total_timeline_years))
        all_annual_cashflows = pre_project_annual + operational_cashflow_millions
        
        # Color bars: red for pre-project (negative), green for operational
        colors = ['red'] * years_before + ['green'] * max_steps
        
        ax3.bar(pre_project_year_labels, pre_project_annual, color='red', alpha=0.7, 
               label=f'Pre-Project Spending (${cost_per_year/1e6:.1f}M/year)')
        ax3.bar(operational_year_labels, operational_cashflow_millions, color='green', alpha=0.7,
               label='Operational Cashflow')
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax3.axvline(x=years_before - 0.5, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax3.text(years_before, max(operational_cashflow_millions) * 0.9, 'Operations Start',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax3.set_xlabel('Project Year', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Annual Cashflow (Million USD)', fontsize=12, fontweight='bold')
        ax3.set_title('Annual Cashflow Breakdown', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(fontsize=10)
        
        # 4. NPV Analysis (Bottom-Right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        colors_npv = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (rate, npv_values) in enumerate(npv_curves.items()):
            ax4.plot(all_years, npv_values, linewidth=2, color=colors_npv[i], 
                    label=f'NPV @ {int(rate*100)}%', alpha=0.8)
        
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Break-even Line')
        ax4.axvline(x=years_before, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax4.text(years_before, max([max(v) for v in npv_curves.values()]) * 0.9, 'Operations Start',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax4.set_xlabel('Project Year', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Net Present Value (Million USD)', fontsize=12, fontweight='bold')
        ax4.set_title(f'NPV Analysis - Episode {self.current_episode}', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def _create_rl_training_animation_tab(self):
        """Create RL Training Animation tab with spatial state animation"""
        content_container = widgets.VBox()
        
        def refresh_animation_tab():
            # Check if spatial data is available
            available_episodes = list(self.episode_stats.get('available_episodes', []))
            episodes_with_spatial = self.episode_stats.get('episodes_with_spatial_data', 0)
            
            if not available_episodes or episodes_with_spatial == 0:
                content_container.children = [
                    widgets.HTML("<h4>🎬 RL Training Animation</h4>"),
                    widgets.HTML("<p>❌ No episodes with spatial data available for animation</p>"),
                    widgets.HTML("<p><i>Spatial state capture must be enabled during RL training to use this feature.</i></p>"),
                    widgets.HTML("<p><b>To enable:</b> Set 'capture_spatial_states: true' in config.yaml RL dashboard settings</p>")
                ]
                return
            
            # Create animation controls
            self.anim_episode_selector = widgets.Dropdown(
                options=[(f"Episode {ep}", ep) for ep in available_episodes],
                value=self.current_episode,
                description='Episode:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            
            # Layer selector for spatial visualization
            self.anim_layer_selector = widgets.IntSlider(
                value=12,
                min=0,
                max=24,  # All 25 layers (0-24)
                step=1,
                description='Layer:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='300px')
            )
            
            # Animation speed control
            self.anim_speed_slider = widgets.FloatSlider(
                value=1.0,
                min=0.1,
                max=3.0,
                step=0.1,
                description='Speed (s):',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='300px')
            )
            
            # Animation control buttons
            self.anim_play_button = widgets.Button(
                description="▶ Play Animation",
                button_style='success',
                layout=widgets.Layout(width='150px')
            )
            
            self.anim_stop_button = widgets.Button(
                description="⏹ Stop",
                button_style='danger',
                disabled=True,
                layout=widgets.Layout(width='150px')
            )
            
            # Animation status
            self.anim_status_label = widgets.Label(
                value='Animation Status: Ready',
                layout=widgets.Layout(width='300px')
            )
            
            # Output area for animation
            self.rl_animation_output = widgets.Output()
            
            # Animation control variables
            self.rl_animation_running = False
            self.rl_animation_thread = None
            
            # Set up event handlers
            self.anim_play_button.on_click(self._start_rl_animation)
            self.anim_stop_button.on_click(self._stop_rl_animation)
            
            # Episode change handler
            def on_episode_change(change):
                if self.set_current_episode(change['new']):
                    self._stop_rl_animation(None)  # Stop any running animation
                    self.anim_status_label.value = 'Animation Status: Episode Changed - Ready'
            
            self.anim_episode_selector.observe(on_episode_change, names='value')
            
            # Stop animation when controls change
            def stop_on_change(change):
                if hasattr(self, 'rl_animation_running') and self.rl_animation_running:
                    self._stop_rl_animation(None)
                    self.anim_status_label.value = 'Animation Status: Stopped (Control Changed)'
            
            self.anim_layer_selector.observe(stop_on_change, names='value')
            
            # Control layout
            controls_row1 = widgets.HBox([
                self.anim_episode_selector,
                self.anim_layer_selector,
                self.anim_speed_slider
            ])
            
            controls_row2 = widgets.HBox([
                self.anim_play_button,
                self.anim_stop_button,
                self.anim_status_label
            ])
            
            # Update container content
            content_container.children = [
                widgets.HTML("<h4>🎬 RL Training Animation</h4>"),
                widgets.HTML("<p><i>Multi-panel synchronized visualization of RL agent actions and reservoir response</i></p>"),
                widgets.HTML("<hr>"),
                widgets.HTML("<h5>🎮 Animation Controls</h5>"),
                controls_row1,
                controls_row2,
                widgets.HTML("<hr>"),
                widgets.HTML("<h5>📊 Real-time Visualization</h5>"),
                self.rl_animation_output
            ]
        
        # Store refresh function for episode change callback
        self._refresh_rl_animation_tab = refresh_animation_tab
        
        # Initial refresh
        refresh_animation_tab()
        
        return content_container
    
    def _create_episode_progression_animation_tab(self):
        """Create Episode Progression Animation tab showing episode evolution"""
        content_container = widgets.VBox()
        
        def refresh_episode_progression_tab():
            # Check available episodes
            available_episodes = list(self.episode_stats.get('available_episodes', []))
            
            if len(available_episodes) < 2:
                content_container.children = [
                    widgets.HTML("<h4>📈 Episode Progression Animation</h4>"),
                    widgets.HTML("<p>❌ Need at least 2 episodes for progression animation</p>"),
                    widgets.HTML(f"<p>Available episodes: {len(available_episodes)}</p>")
                ]
                return
            
            # Controls for episode progression
            self.ep_speed_slider = widgets.FloatSlider(
                value=1.0,
                min=0.2,
                max=3.0,
                step=0.1,
                description='Speed (s):',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='300px')
            )
            
            self.ep_play_button = widgets.Button(
                description="▶ Play Episodes",
                button_style='success',
                layout=widgets.Layout(width='150px')
            )
            
            self.ep_stop_button = widgets.Button(
                description="⏹ Stop",
                button_style='danger',
                disabled=True,
                layout=widgets.Layout(width='150px')
            )
            
            self.ep_status_label = widgets.Label(
                value='Animation Status: Ready',
                layout=widgets.Layout(width='400px')
            )
            
            self.ep_animation_output = widgets.Output()
            self.ep_animation_running = False
            
            # Event handlers
            self.ep_play_button.on_click(self._start_episode_progression)
            self.ep_stop_button.on_click(self._stop_episode_progression)
            
            # Control layout
            controls_row1 = widgets.HBox([self.ep_speed_slider])
            controls_row2 = widgets.HBox([self.ep_play_button, self.ep_stop_button, self.ep_status_label])
            
            content_container.children = [
                widgets.HTML("<h4>📈 Episode Progression Animation</h4>"),
                widgets.HTML("<p><i>Watch complete episode time series evolve across training episodes (same as RL animation but no spatial plots)</i></p>"),
                widgets.HTML("<hr>"),
                widgets.HTML("<h5>🎮 Animation Controls</h5>"),
                controls_row1,
                controls_row2,
                widgets.HTML("<hr>"),
                widgets.HTML("<h5>📊 Training Evolution</h5>"),
                self.ep_animation_output
            ]
        
        refresh_episode_progression_tab()
        return content_container
    
    def _get_rl_training_channel_info(self):
        """Get actual channel names and units used during RL training"""
        try:
            # Try to load the actual channel information from normalization files
            import glob
            import json
            from pathlib import Path
            
            # Look for normalization parameters file
            json_files = glob.glob("normalization_parameters_*.json")
            
            if json_files:
                # Use most recent file
                latest_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
                with open(latest_file, 'r') as f:
                    norm_params = json.load(f)
                
                # Extract channel information
                spatial_channels = norm_params.get('spatial_channels', {})
                field_names = []
                field_units = []
                training_channels = []
                
                # Standard order: SW, SG, PRES
                channel_order = ['SW', 'SG', 'PRES']
                for ch_name in channel_order:
                    if ch_name in spatial_channels:
                        field_names.append(spatial_channels[ch_name].get('display_name', ch_name))
                        field_units.append(spatial_channels[ch_name].get('unit', ''))
                        training_channels.append(ch_name)
                
                return field_names, field_units, training_channels
            else:
                # Fallback to defaults
                return ['Water Saturation', 'Gas Saturation', 'Pressure'], ['fraction', 'fraction', 'psi'], ['SW', 'SG', 'PRES']
        except Exception as e:
            print(f"⚠️ Warning: Could not load channel info: {e}")
            return ['Water Saturation', 'Gas Saturation', 'Pressure'], ['fraction', 'fraction', 'psi'], ['SW', 'SG', 'PRES']
    
    def _calculate_data_color_range(self, spatial_data, timestep, layer, channel):
        """Calculate optimal color range based on actual data values"""
        try:
            # Get spatial state for selected timestep
            if timestep not in spatial_data['spatial_states']:
                return (0.0, 1.0)  # Fallback
            
            spatial_state = spatial_data['spatial_states'][timestep]
            
            # Handle shape: expect (batch, channels, X, Y, Z) or (channels, X, Y, Z)
            if len(spatial_state.shape) == 5:
                field_data = spatial_state[0, channel, :, :, layer]
            elif len(spatial_state.shape) == 4:
                field_data = spatial_state[channel, :, :, layer]
            else:
                return (0.0, 1.0)  # Fallback
            
            # Convert to numpy if tensor
            if hasattr(field_data, 'detach'):
                field_data = field_data.detach().cpu().numpy()
            
            # Get the actual channel name for denormalization
            field_names, field_units, training_channels = self._get_rl_training_channel_info()
            channel_name = training_channels[channel] if channel < len(training_channels) else f'Channel_{channel}'
            
            # Denormalize to physical units
            field_data = self._denormalize_spatial_data(field_data, channel_name)
            
            # Apply physical constraints (non-negative values)
            field_data = np.maximum(field_data, 0.0)
            
            # Calculate range from valid (non-NaN) data
            valid_data = field_data[~np.isnan(field_data)]
            
            if len(valid_data) > 0:
                # Use percentile-based scaling to handle outliers
                vmin = np.percentile(valid_data, 2)   # 2nd percentile
                vmax = np.percentile(valid_data, 98)  # 98th percentile
                
                # Ensure minimum range for visualization
                if vmax - vmin < 1e-6:
                    center = (vmax + vmin) / 2
                    vmin = center - 0.1
                    vmax = center + 0.1
                
                return (float(vmin), float(vmax))
            else:
                return (0.0, 1.0)  # Fallback
                
        except Exception as e:
            print(f"⚠️ Error calculating data color range: {e}")
            return (0.0, 1.0)  # Fallback
    
    def _denormalize_spatial_data(self, field_data, channel_name):
        """
        Denormalize spatial field data back to physical units using normalization parameters
        
        Args:
            field_data: Normalized field data (0-1 range)
            channel_name: Name of the channel (e.g., 'SG', 'PRES', 'SW')
            
        Returns:
            Denormalized field data in physical units
        """
        try:
            # Load the normalization parameters
            import glob
            import json
            from pathlib import Path
            
            json_files = glob.glob("normalization_parameters_*.json")
            if not json_files:
                print(f"⚠️ No normalization files found - keeping normalized data")
                return field_data
            
            # Get the latest normalization file
            json_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            latest_json = json_files[0]
            
            with open(latest_json, 'r') as f:
                norm_config = json.load(f)
            
            # Look for the channel in spatial_channels
            spatial_channels = norm_config.get('spatial_channels', {})
            if channel_name in spatial_channels:
                params = spatial_channels[channel_name]['parameters']
                
                # Convert string parameters to float if needed
                param_min = float(params['min']) if isinstance(params['min'], str) else params['min']
                param_max = float(params['max']) if isinstance(params['max'], str) else params['max']
                
                # Apply denormalization: denormalized = normalized * (max - min) + min
                denormalized_data = field_data * (param_max - param_min) + param_min
                
                print(f"✅ Denormalized {channel_name}: [{param_min:.2f}, {param_max:.2f}]")
                return denormalized_data
            else:
                print(f"⚠️ Channel {channel_name} not found in normalization parameters")
                return field_data
                
        except Exception as e:
            print(f"⚠️ Error denormalizing {channel_name}: {e}")
            return field_data
    
    def _apply_inactive_cell_masking(self, field_data, episode_number, layer, channel_name):
        """Apply inactive cell masking if available - returns None if no mask available"""
        # This is a placeholder - full implementation would load mask files
        # For now, return None to skip masking
        return None
    
    def plot_spatial_visualization(self, timestep, layer, channel, spatial_data, user_vmin=None, user_vmax=None):
        """Plot spatial reservoir state visualization"""
        try:
            # Get spatial state for selected timestep
            if timestep not in spatial_data['spatial_states']:
                print(f"❌ Timestep {timestep} not available in spatial data")
                return
            
            spatial_state = spatial_data['spatial_states'][timestep]
            
            # Handle shape: expect (batch, channels, X, Y, Z) or (channels, X, Y, Z)
            if len(spatial_state.shape) == 5:
                field_data = spatial_state[0, channel, :, :, layer]
            elif len(spatial_state.shape) == 4:
                field_data = spatial_state[channel, :, :, layer]
            else:
                print(f"❌ Unexpected spatial state shape: {spatial_state.shape}")
                return
            
            # Convert to numpy if tensor
            if hasattr(field_data, 'detach'):
                field_data = field_data.detach().cpu().numpy()
            
            # Get actual channel names from RL training
            field_names, field_units, training_channels = self._get_rl_training_channel_info()
            
            # Get the actual channel name for denormalization
            channel_name = training_channels[channel] if channel < len(training_channels) else f'Channel_{channel}'
            field_name = field_names[channel] if channel < len(field_names) else f'Channel_{channel}'
            field_unit = field_units[channel] if channel < len(field_units) else 'units'
            
            # Denormalize to physical units
            field_data = self._denormalize_spatial_data(field_data, channel_name)
            
            # Apply physical constraints
            field_data = np.maximum(field_data, 0.0)
            
            # Apply inactive cell masking if available
            field_data_masked = self._apply_inactive_cell_masking(field_data, spatial_data.get('episode_number', 0), layer, channel_name)
            if field_data_masked is not None:
                field_data = field_data_masked
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Colormap with NaN handling for inactive cells
            cmap = plt.cm.jet.copy()
            cmap.set_bad('white', alpha=0.3)
            
            # User-controllable color scales
            if user_vmin is not None and user_vmax is not None:
                vmin, vmax = user_vmin, user_vmax
            else:
                vmin, vmax = self._calculate_data_color_range(spatial_data, timestep, layer, channel)
            
            # Plot spatial field
            im = ax.imshow(field_data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect='equal', interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{field_name} ({field_unit})', fontsize=12, fontweight='bold')
            
            # Customize
            ax.set_title(f'Episode {self.current_episode} | Step {timestep} | Layer {layer} | {field_name}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('I Index', fontsize=12)
            ax.set_ylabel('J Index', fontsize=12)
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ Error plotting spatial visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_rl_animation(self, button):
        """Start RL training animation with multi-panel synchronized visualization"""
        if self.rl_animation_running:
            return
            
        self.rl_animation_running = True
        self.anim_play_button.disabled = True
        self.anim_stop_button.disabled = False
        self.anim_status_label.value = 'Animation Status: Starting...'
        
        def animate():
            """Animation loop with GIF creation"""
            gif_frames = []  # Store frames for GIF creation
            
            try:
                # Get current episode data
                episode_data = self.training_orchestrator.get_episode_data(self.current_episode)
                if not episode_data:
                    self.anim_status_label.value = 'Animation Status: Error - No episode data'
                    return
                
                # Get spatial data
                spatial_data = episode_data.get('spatial_states', [])
                if not spatial_data:
                    self.anim_status_label.value = 'Animation Status: Error - No spatial data'
                    return
                
                actions = episode_data.get('actions', [])
                observations = episode_data.get('observations', [])
                timesteps = episode_data.get('timesteps', [])
                
                layer_idx = self.anim_layer_selector.value
                speed = self.anim_speed_slider.value
                
                # Create output directory for GIFs
                gif_dir = Path("rl_animation_gifs")
                gif_dir.mkdir(exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gif_filename = gif_dir / f"rl_training_episode{self.current_episode}_layer{layer_idx}_{timestamp}.gif"
                
                # Animation loop through timesteps
                num_steps = min(len(actions), len(spatial_data))
                
                for step_idx in range(num_steps):
                    if not self.rl_animation_running:
                        break
                    
                    # Update status
                    self.anim_status_label.value = f'Animation Status: Playing (Step {step_idx+1}/{num_steps}) - Creating GIF...'
                    
                    # Create synchronized visualization frame
                    with self.rl_animation_output:
                        clear_output(wait=True)
                        
                        # Create the multi-panel plot
                        fig = self._create_rl_animation_frame(
                            step_idx=step_idx,
                            layer_idx=layer_idx,
                            actions=actions,
                            observations=observations,
                            spatial_data=spatial_data,
                            timesteps=timesteps
                        )
                        
                        if fig:
                            # Display the plot
                            display(fig)
                            
                            # Capture frame for GIF
                            if PIL_AVAILABLE:
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                                buf.seek(0)
                                gif_frames.append(Image.open(buf))
                            
                            # Close the figure to prevent memory issues
                            plt.close(fig)
                    
                    # Wait for next frame
                    time.sleep(speed)
                
                # Create and save GIF if animation completed
                if self.rl_animation_running and len(gif_frames) > 0 and PIL_AVAILABLE:
                    self.anim_status_label.value = 'Animation Status: Saving GIF...'
                    
                    # Create GIF with proper duration
                    gif_frames[0].save(
                        gif_filename,
                        save_all=True,
                        append_images=gif_frames[1:],
                        duration=int(speed * 1000),  # Convert to milliseconds
                        loop=0  # Infinite loop
                    )
                    
                    self.anim_status_label.value = f'Animation Status: Completed - GIF saved: {gif_filename.name}'
                    print(f"🎬 RL Animation GIF saved: {gif_filename}")
                else:
                    if len(gif_frames) == 0:
                        self.anim_status_label.value = 'Animation Status: Stopped - No frames captured'
                    else:
                        self.anim_status_label.value = 'Animation Status: Stopped'
                        
            except Exception as e:
                self.anim_status_label.value = f'Animation Status: Error - {str(e)}'
                print(f"RL Animation error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up
                for frame in gif_frames:
                    if hasattr(frame, 'close'):
                        frame.close()
                self.rl_animation_running = False
                self.anim_play_button.disabled = False
                self.anim_stop_button.disabled = True
        
        # Start animation in separate thread
        self.rl_animation_thread = threading.Thread(target=animate)
        self.rl_animation_thread.daemon = True
        self.rl_animation_thread.start()
    
    def _stop_rl_animation(self, button):
        """Stop RL training animation"""
        self.rl_animation_running = False
        self.anim_play_button.disabled = False
        self.anim_stop_button.disabled = True
        self.anim_status_label.value = 'Animation Status: Stopped'
    
    def _create_rl_animation_frame(self, step_idx, layer_idx, actions, observations, spatial_data, timesteps):
        """Create synchronized multi-panel RL training animation frame"""
        try:
            # Get current timestep data
            if step_idx >= len(actions) or step_idx >= len(spatial_data):
                return None
                
            current_action = actions[step_idx]
            current_observation = observations[step_idx] if step_idx < len(observations) else {}
            current_spatial = spatial_data[step_idx]['spatial_state']
            current_timestep = timesteps[step_idx] if step_idx < len(timesteps) else step_idx
            
            # Calculate time in years for display (each step = 1 year for animation)
            base_year = 2024  # Starting year for display
            current_year = base_year + step_idx
            
            # Create figure with expanded grid layout for economic analysis
            fig = plt.figure(figsize=(22, 16), dpi=100)
            gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[1, 1, 1, 1.2], width_ratios=[1, 1, 1, 1, 1])
            
            # Add main title
            fig.suptitle(f'RL Training Animation - Episode {self.current_episode} | Step {step_idx+1} | Year {current_year:.2f}', 
                        fontsize=16, fontweight='bold', y=0.97)
            
            # Row 1: Injector Gas Actions (I1, I2, I3)
            for i in range(3):
                ax = fig.add_subplot(gs[0, i])
                self._plot_individual_injector_timeseries(ax, f'I{i+1}', step_idx, actions)
            
            # Row 2: Producer BHP Actions (P1, P2, P3)  
            for i in range(3):
                ax = fig.add_subplot(gs[1, i])
                self._plot_individual_producer_bhp_timeseries(ax, f'P{i+1}', step_idx, actions)
            
            # Row 3: Producer Response (P1, P2, P3)
            for i in range(3):
                ax = fig.add_subplot(gs[2, i])
                self._plot_individual_producer_response_timeseries_dual_axis(ax, f'P{i+1}', step_idx, observations)
            
            # Row 4: Enhanced with Economic Analysis
            # Spatial Fields (positions 0-1)
            ax_pressure = fig.add_subplot(gs[3, 0])
            self._plot_spatial_field_simple(ax_pressure, current_spatial, layer_idx, 'PRES', step_idx, 'Pressure')
            
            ax_saturation = fig.add_subplot(gs[3, 1])
            self._plot_spatial_field_simple(ax_saturation, current_spatial, layer_idx, 'SG', step_idx, 'Gas Saturation')
            
            # Performance & Economic Analysis (positions 2-4)
            ax_reward = fig.add_subplot(gs[3, 2])
            self._plot_reward_evolution_for_rl_animation(ax_reward, step_idx)
            
            # Economic Cashflow Analysis
            ax_net_cashflow = fig.add_subplot(gs[3, 3])
            self._plot_net_cashflow_evolution_for_rl_animation(ax_net_cashflow, step_idx)
            
            ax_cumulative_cashflow = fig.add_subplot(gs[3, 4])
            self._plot_cumulative_cashflow_evolution_for_rl_animation(ax_cumulative_cashflow, step_idx)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.94, hspace=0.4, wspace=0.25)
            
            return fig
            
        except Exception as e:
            print(f"Error creating RL animation frame: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_individual_injector_timeseries(self, ax, well_name, step_idx, actions):
        """Plot individual injector gas injection time series up to current step"""
        try:
            if step_idx == 0 or len(actions) < 1:
                ax.text(0.5, 0.5, f'{well_name}\nGas Injection\n(Starting...)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # Extract gas injection data up to current step
            steps = list(range(step_idx + 1))
            gas_values = []
            
            gas_key = f"{well_name}_Gas_ft3day"
            for i in range(step_idx + 1):
                gas_value = actions[i].get(gas_key, 0) / 1e6  # Convert to millions
                gas_values.append(gas_value)
            
            # Plot time series
            ax.plot(steps, gas_values, 'g-o', linewidth=2, markersize=3, alpha=0.8, color='green')
            
            # Highlight current step
            if step_idx > 0:
                ax.plot(step_idx, gas_values[step_idx], 'go', markersize=6, 
                       markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=2)
            
            # Customize
            ax.set_title(f'{well_name} Gas Injection', fontsize=11, fontweight='bold')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('MMft³/day', fontsize=10, color='green')
            ax.tick_params(axis='y', labelcolor='green', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add current value annotation
            if len(gas_values) > 0:
                current_val = gas_values[step_idx]
                ax.text(0.95, 0.95, f'{current_val:.1f}', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_individual_producer_bhp_timeseries(self, ax, well_name, step_idx, actions):
        """Plot individual producer BHP time series up to current step"""
        try:
            if step_idx == 0 or len(actions) < 1:
                ax.text(0.5, 0.5, f'{well_name}\nBHP Control\n(Starting...)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # Extract BHP data up to current step
            steps = list(range(step_idx + 1))
            bhp_values = []
            
            bhp_key = f"{well_name}_BHP_psi"
            for i in range(step_idx + 1):
                bhp_value = actions[i].get(bhp_key, 0)
                bhp_values.append(bhp_value)
            
            # Plot time series
            ax.plot(steps, bhp_values, 'r-o', linewidth=2, markersize=3, alpha=0.8, color='red')
            
            # Highlight current step
            if step_idx > 0:
                ax.plot(step_idx, bhp_values[step_idx], 'ro', markersize=6, 
                       markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
            
            # Customize
            ax.set_title(f'{well_name} BHP Control', fontsize=11, fontweight='bold')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('psi', fontsize=10, color='red')
            ax.tick_params(axis='y', labelcolor='red', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add current value annotation
            if len(bhp_values) > 0:
                current_val = bhp_values[step_idx]
                ax.text(0.95, 0.95, f'{current_val:.0f}', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_individual_producer_response_timeseries_dual_axis(self, ax, well_name, step_idx, observations):
        """Plot individual producer response (water + gas) time series with dual y-axes"""
        try:
            if step_idx == 0 or len(observations) < 1:
                ax.text(0.5, 0.5, f'{well_name}\nProduction\n(Starting...)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # Extract production data up to current step
            steps = list(range(step_idx + 1))
            water_values = []
            gas_values = []
            
            water_key = f"{well_name}_Water_ft3day"
            gas_key = f"{well_name}_Gas_ft3day"
            
            for i in range(step_idx + 1):
                if i < len(observations):
                    water_val = observations[i].get(water_key, 0) / 1e6  # Convert to millions
                    gas_val = observations[i].get(gas_key, 0) / 1e6
                else:
                    water_val = 0
                    gas_val = 0
                water_values.append(water_val)
                gas_values.append(gas_val)
            
            # Create dual-axis plot for better scaling
            ax2 = ax.twinx()
            
            # Plot water on left axis (ax)
            line1 = ax.plot(steps, water_values, 'b-o', linewidth=2, markersize=3, label='Water', alpha=0.8)
            
            # Plot gas on right axis (ax2)
            line2 = ax2.plot(steps, gas_values, 'orange', linewidth=2, markersize=3, label='Gas', alpha=0.8, linestyle='-', marker='s')
            
            # Highlight current step
            if step_idx > 0:
                ax.plot(step_idx, water_values[step_idx], 'bo', markersize=6, markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=2)
                ax2.plot(step_idx, gas_values[step_idx], 'o', color='orange', markersize=6, markerfacecolor='orange', markeredgecolor='darkorange', markeredgewidth=2)
            
            # Customize left axis (water)
            ax.set_title(f'{well_name} Production', fontsize=11, fontweight='bold')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('Water (MMft³/day)', fontsize=10, color='blue')
            ax.tick_params(axis='y', labelcolor='blue', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(True, alpha=0.3)
            
            # Customize right axis (gas)
            ax2.set_ylabel('Gas (MMft³/day)', fontsize=10, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange', labelsize=9)
            
            # Create combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
            
            # Add current values annotation
            if len(water_values) > 0 and len(gas_values) > 0:
                water_val = water_values[step_idx]
                gas_val = gas_values[step_idx]
                ax.text(0.95, 0.95, f'W:{water_val:.1f}\nG:{gas_val:.1f}', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=9, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_spatial_field_simple(self, ax, spatial_state, layer_idx, field_type, step_idx, field_name):
        """Plot spatial field (pressure or saturation) without masking for RL animation"""
        try:
            # Handle spatial data extraction based on field type
            # Training Channel 0 = SG (Gas Saturation), Training Channel 1 = PRES (Pressure)
            if hasattr(spatial_state, 'shape') and len(spatial_state.shape) >= 4:
                if len(spatial_state.shape) == 5:  # (batch, channels, X, Y, Z)
                    channel_idx = 1 if field_type == 'PRES' else 0 if field_type == 'SG' else 0
                    field_data = spatial_state[0, channel_idx, :, :, layer_idx]
                elif len(spatial_state.shape) == 4:  # (channels, X, Y, Z)
                    channel_idx = 1 if field_type == 'PRES' else 0 if field_type == 'SG' else 0
                    field_data = spatial_state[channel_idx, :, :, layer_idx]
                else:
                    ax.text(0.5, 0.5, f'Unexpected shape: {spatial_state.shape}', 
                           transform=ax.transAxes, ha='center', va='center')
                    return
            else:
                ax.text(0.5, 0.5, 'No spatial data', transform=ax.transAxes, ha='center', va='center')
                return
            
            # Convert to numpy
            if hasattr(field_data, 'detach'):
                field_data = field_data.detach().cpu().numpy()
            
            # Simple color scaling (no masking)
            valid_data = field_data[~np.isnan(field_data)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
                # Ensure reasonable range
                if vmax - vmin < 1e-6:
                    vmin = valid_data.min()
                    vmax = valid_data.max()
            else:
                vmin, vmax = 0, 1
            
            # Plot with jet colormap (no masking)
            im = ax.imshow(field_data.T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax, 
                          aspect='equal', interpolation='bilinear')
            
            # Customize
            ax.set_title(f'{field_name}\nLayer {layer_idx}', fontsize=11, fontweight='bold')
            ax.set_xlabel('I Index', fontsize=10)
            ax.set_ylabel('J Index', fontsize=10)
            ax.tick_params(labelsize=9)
            
            # Add compact colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=15)
            unit = 'Pressure (psi)' if field_type == 'PRES' else 'Gas Saturation' if field_type == 'SG' else field_name
            cbar.set_label(unit, fontsize=9, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_reward_evolution_for_rl_animation(self, ax, step_idx):
        """Plot reward evolution for current episode up to current step"""
        try:
            episode_data = self.training_orchestrator.get_episode_data(self.current_episode)
            if not episode_data or 'rewards' not in episode_data:
                ax.text(0.5, 0.5, 'No reward data available', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                return
            
            rewards = episode_data['rewards']
            if step_idx >= len(rewards):
                step_idx = len(rewards) - 1
            
            steps = list(range(step_idx + 1))
            current_rewards = rewards[:step_idx + 1]
            
            if len(current_rewards) == 0:
                ax.text(0.5, 0.5, 'Reward Evolution\n(Starting...)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                return
            
            # Plot reward evolution
            ax.plot(steps, current_rewards, 'b-o', linewidth=2, markersize=4, alpha=0.8)
            
            # Highlight current step
            if step_idx > 0:
                ax.plot(step_idx, current_rewards[step_idx], 'bo', markersize=8, markerfacecolor='blue')
            
            # Calculate cumulative reward
            cumulative_reward = sum(current_rewards)
            
            ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
            ax.set_ylabel('Step Reward', fontsize=12, fontweight='bold')
            ax.set_title(f'Reward Evolution (Cumulative: {cumulative_reward:.2f})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add average line if enough data
            if len(current_rewards) > 1:
                avg_reward = np.mean(current_rewards)
                ax.axhline(y=avg_reward, color='red', linestyle='--', alpha=0.7, 
                          label=f'Average: {avg_reward:.3f}')
                ax.legend(fontsize=10)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_net_cashflow_evolution_for_rl_animation(self, ax, step_idx):
        """Plot complete project net cashflow including pre-project phase"""
        try:
            episode_data = self.training_orchestrator.get_episode_data(self.current_episode)
            if not episode_data or 'economic_breakdown' not in episode_data:
                ax.text(0.5, 0.5, 'No economic data\navailable', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            economic_breakdown = episode_data['economic_breakdown']
            if not economic_breakdown:
                ax.text(0.5, 0.5, 'Net Cashflow\n(Starting...)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # PRE-PROJECT PARAMETERS: Get from existing infrastructure
            years_before = self.data.get('years_before_project', 5) if self.data else 5
            cost_per_year = self.data.get('capital_cost_per_year', 100000000.0) if self.data else 100000000.0
            scale_factor = self.config.rl_model['economics']['scale_factor']
            
            # CREATE COMPLETE PROJECT TIMELINE
            # Pre-project years: Fixed capital expenditure
            pre_project_cashflows = []
            for year in range(1, years_before + 1):
                annual_capex = -cost_per_year / scale_factor  # Convert to millions, negative (expense)
                pre_project_cashflows.append(annual_capex)
            
            # OPERATIONAL PHASE: RL-driven cashflows up to current step
            operational_cashflows = []
            for i in range(step_idx + 1):
                if i < len(economic_breakdown):
                    # Get operational cashflow (already in physical units)
                    if 'operational_cashflow' in economic_breakdown[i]:
                        cashflow = economic_breakdown[i]['operational_cashflow']
                    else:
                        # Backward compatibility
                        cashflow = economic_breakdown[i].get('net_step_cashflow', 0)
                    
                    # Convert to millions for display
                    cashflow_millions = cashflow / scale_factor
                    operational_cashflows.append(cashflow_millions)
                else:
                    operational_cashflows.append(0)
            
            # COMBINE COMPLETE PROJECT TIMELINE
            complete_cashflows = pre_project_cashflows + operational_cashflows
            
            # Create timeline labels (negative years for pre-project, positive for operations)
            pre_project_years = list(range(-years_before, 0))
            operational_years = list(range(step_idx + 1))
            complete_years = pre_project_years + operational_years
            
            if len(complete_cashflows) == 0:
                ax.text(0.5, 0.5, 'Net Cashflow\n(No Data)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # CREATE ENHANCED BAR PLOT WITH PHASE DISTINCTION
            # Pre-project bars (red for capital expenditure)
            if len(pre_project_cashflows) > 0:
                bars_pre = ax.bar(pre_project_years, pre_project_cashflows, 
                                color='darkred', alpha=0.8, width=0.8, 
                                label='Capital Expenditure')
            
            # Operational bars (green/red based on performance)
            if len(operational_cashflows) > 0:
                colors = ['green' if cf >= 0 else 'red' for cf in operational_cashflows]
                bars_ops = ax.bar(operational_years, operational_cashflows, 
                                color=colors, alpha=0.7, width=0.8, 
                                label='RL Operations')
                
                # Highlight current operational step
                if step_idx >= 0 and step_idx < len(bars_ops):
                    current_color = 'darkgreen' if operational_cashflows[step_idx] >= 0 else 'darkred'
                    bars_ops[step_idx].set_color(current_color)
                    bars_ops[step_idx].set_edgecolor('black')
                    bars_ops[step_idx].set_linewidth(2)
            
            # ADD REFERENCE LINES AND PHASES
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=-0.5, color='orange', linestyle=':', alpha=0.7, linewidth=2, 
                      label='Operations Start')
            
            # CUSTOMIZE PLOT
            ax.set_title('Complete Project Net Cashflow', fontsize=11, fontweight='bold')
            ax.set_xlabel('Project Year', fontsize=10)
            ax.set_ylabel('Annual Cashflow\n(Million USD)', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=8, loc='upper left')
            
            # ADD CURRENT VALUE ANNOTATION
            if len(operational_cashflows) > 0 and step_idx >= 0:
                current_val = operational_cashflows[step_idx]
                color = 'green' if current_val >= 0 else 'red'
                ax.text(0.95, 0.95, f'Current: ${current_val:.1f}M', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            
            # ADD PHASE INFORMATION
            total_capex = years_before * cost_per_year / scale_factor
            ax.text(0.05, 0.95, f'Total CapEx: ${total_capex:.0f}M', transform=ax.transAxes, 
                   ha='left', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center', fontsize=8)
    
    def _plot_cumulative_cashflow_evolution_for_rl_animation(self, ax, step_idx):
        """Plot complete project cumulative cashflow including pre-project debt"""
        try:
            episode_data = self.training_orchestrator.get_episode_data(self.current_episode)
            if not episode_data or 'economic_breakdown' not in episode_data:
                ax.text(0.5, 0.5, 'No economic data\navailable', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            economic_breakdown = episode_data['economic_breakdown']
            if not economic_breakdown:
                ax.text(0.5, 0.5, 'Cumulative Cashflow\n(Starting...)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # PRE-PROJECT PARAMETERS: Get from existing infrastructure
            years_before = self.data.get('years_before_project', 5) if self.data else 5
            cost_per_year = self.data.get('capital_cost_per_year', 100000000.0) if self.data else 100000000.0
            scale_factor = self.config.rl_model['economics']['scale_factor']
            
            # CREATE PRE-PROJECT CUMULATIVE DEBT
            pre_project_cumulative = []
            running_debt = 0
            for year in range(1, years_before + 1):
                annual_capex = cost_per_year / scale_factor  # Convert to millions (positive for debt)
                running_debt -= annual_capex  # Accumulate debt (negative values)
                pre_project_cumulative.append(running_debt)
            
            # OPERATIONAL PHASE: RL-driven recovery from debt
            operational_cumulative = []
            operational_total = running_debt  # Start from final pre-project debt
            
            for i in range(step_idx + 1):
                if i < len(economic_breakdown):
                    # Get operational cashflow (already in physical units)
                    if 'operational_cashflow' in economic_breakdown[i]:
                        cashflow = economic_breakdown[i]['operational_cashflow']
                    else:
                        # Backward compatibility
                        cashflow = economic_breakdown[i].get('net_step_cashflow', 0)
                    
                    # Convert to millions and add to running total
                    cashflow_millions = cashflow / scale_factor
                    operational_total += cashflow_millions
                    operational_cumulative.append(operational_total)
                else:
                    operational_cumulative.append(operational_total)
            
            # COMBINE COMPLETE PROJECT TIMELINE
            complete_cumulative = pre_project_cumulative + operational_cumulative
            
            # Create timeline labels (negative years for pre-project, positive for operations)
            pre_project_years = list(range(-years_before, 0))
            operational_years = list(range(step_idx + 1))
            complete_years = pre_project_years + operational_years
            
            if len(complete_cumulative) == 0:
                ax.text(0.5, 0.5, 'Cumulative Cashflow\n(No Data)', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                return
            
            # ENHANCED VISUALIZATION WITH PHASE DISTINCTION
            # Plot complete cumulative line with color transition
            final_value = complete_cumulative[-1] if complete_cumulative else 0
            overall_color = 'green' if final_value >= 0 else 'red'
            
            ax.plot(complete_years, complete_cumulative, color=overall_color, linewidth=3, 
                   marker='o', markersize=3, alpha=0.8, markerfacecolor=overall_color)
            
            # HIGHLIGHT KEY POINTS
            # Highlight end of pre-project phase (maximum debt)
            if len(pre_project_cumulative) > 0:
                max_debt = pre_project_cumulative[-1]
                ax.plot(-1, max_debt, 's', markersize=8, color='darkred', 
                       markeredgecolor='black', markeredgewidth=2, 
                       label=f'Max Debt: ${abs(max_debt):.0f}M')
            
            # Highlight current operational step
            if step_idx >= 0 and len(operational_cumulative) > 0:
                current_val = operational_cumulative[step_idx]
                current_color = 'darkgreen' if current_val >= 0 else 'darkred'
                ax.plot(step_idx, current_val, 'o', markersize=8, 
                       markerfacecolor=current_color, markeredgecolor='black', markeredgewidth=2)
            
            # ADD REFERENCE LINES AND PHASES
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Break-even')
            ax.axvline(x=-0.5, color='orange', linestyle=':', alpha=0.7, linewidth=2, 
                      label='Operations Start')
            
            # ENHANCED FILL AREAS FOR VISUAL IMPACT
            if len(complete_cumulative) > 1:
                # Fill area below zero (debt/loss) in red
                ax.fill_between(complete_years, complete_cumulative, 0, 
                               where=[cf <= 0 for cf in complete_cumulative], 
                               color='red', alpha=0.3, interpolate=True, label='Debt/Loss')
                
                # Fill area above zero (profit) in green
                ax.fill_between(complete_years, complete_cumulative, 0, 
                               where=[cf > 0 for cf in complete_cumulative], 
                               color='green', alpha=0.3, interpolate=True, label='Profit')
            
            # BREAK-EVEN ANALYSIS
            break_even_year = None
            for i, (year, cf) in enumerate(zip(complete_years, complete_cumulative)):
                if cf >= 0 and year >= 0:  # First positive value in operational phase
                    break_even_year = year
                    ax.axvline(x=year, color='gold', linestyle='--', alpha=0.8, linewidth=2,
                              label=f'Break-even: Year {year}')
                    ax.plot(year, cf, '*', markersize=12, color='gold', 
                           markeredgecolor='black', markeredgewidth=1)
                    break
            
            # CUSTOMIZE PLOT
            ax.set_title('Complete Project Cumulative Cashflow', fontsize=11, fontweight='bold')
            ax.set_xlabel('Project Year', fontsize=10)
            ax.set_ylabel('Cumulative Cashflow\n(Million USD)', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='lower right')
            
            # COMPREHENSIVE STATUS ANNOTATION
            if len(operational_cumulative) > 0 and step_idx >= 0:
                current_val = operational_cumulative[step_idx]
                total_debt = abs(pre_project_cumulative[-1]) if pre_project_cumulative else 0
                recovery_pct = (total_debt + current_val) / total_debt * 100 if total_debt > 0 else 0
                
                status = 'PROFITABLE' if current_val >= 0 else 'RECOVERING'
                color = 'green' if current_val >= 0 else 'orange'
                
                status_text = f'{status}\n${current_val:.1f}M\n{recovery_pct:.1f}% recovered'
                ax.text(0.95, 0.05, status_text, transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
            
            # PROJECT SUMMARY
            total_capex = years_before * cost_per_year / scale_factor
            ax.text(0.05, 0.95, f'Project Investment: ${total_capex:.0f}M\nYears to Break-even: {break_even_year if break_even_year else "TBD"}', 
                   transform=ax.transAxes, ha='left', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center', fontsize=8)
    
    def _start_episode_progression(self, button):
        """Start episode progression animation with complete episode time series"""
        if self.ep_animation_running:
            return
            
        self.ep_animation_running = True
        self.ep_play_button.disabled = True
        self.ep_stop_button.disabled = False
        self.ep_status_label.value = 'Animation Status: Starting...'
        
        def animate():
            gif_frames = []
            try:
                available_episodes = sorted(list(self.episode_stats.get('available_episodes', [])))
                speed = self.ep_speed_slider.value
                
                # Create GIF directory
                gif_dir = Path("episode_progression_gifs")
                gif_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gif_filename = gif_dir / f"episode_progression_full_episodes_{timestamp}.gif"
                
                for ep_idx, episode_num in enumerate(available_episodes):
                    if not self.ep_animation_running:
                        break
                    
                    self.ep_status_label.value = f'Playing Episode {episode_num} ({ep_idx+1}/{len(available_episodes)}) - Full Episode Data'
                    
                    with self.ep_animation_output:
                        clear_output(wait=True)
                        
                        fig = self._create_episode_progression_frame(episode_num, ep_idx, len(available_episodes))
                        if fig:
                            display(fig)
                            
                            # Capture for GIF
                            if PIL_AVAILABLE:
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                                buf.seek(0)
                                gif_frames.append(Image.open(buf))
                            plt.close(fig)
                    
                    time.sleep(speed)
                
                # Save GIF
                if self.ep_animation_running and len(gif_frames) > 0 and PIL_AVAILABLE:
                    self.ep_status_label.value = 'Saving GIF...'
                    gif_frames[0].save(gif_filename, save_all=True, append_images=gif_frames[1:],
                                     duration=int(speed * 1000), loop=0)
                    self.ep_status_label.value = f'Completed - GIF saved: {gif_filename.name}'
                else:
                    self.ep_status_label.value = 'Animation Status: Stopped'
                    
            except Exception as e:
                self.ep_status_label.value = f'Error: {str(e)}'
            finally:
                self.ep_animation_running = False
                self.ep_play_button.disabled = False
                self.ep_stop_button.disabled = True
        
        threading.Thread(target=animate, daemon=True).start()
    
    def _stop_episode_progression(self, button):
        """Stop episode progression animation"""
        self.ep_animation_running = False
        self.ep_play_button.disabled = False
        self.ep_stop_button.disabled = True
        self.ep_status_label.value = 'Animation Status: Stopped'
    
    def _create_episode_progression_frame(self, episode_num, ep_idx, total_episodes):
        """Create frame showing complete episode time series for episode progression"""
        try:
            # Get complete episode data
            episode_data = self.training_orchestrator.get_episode_data(episode_num)
            if not episode_data:
                return None
            
            actions = episode_data.get('actions', [])
            observations = episode_data.get('observations', [])
            timesteps = episode_data.get('timesteps', [])
            
            if not actions:
                return None
            
            # 4×3 layout (same as RL animation but no spatial plots)
            fig = plt.figure(figsize=(18, 12), dpi=100)
            gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
            
            fig.suptitle(f'Episode Progression - Episode {episode_num} | Complete Episode Data | Progress: {ep_idx+1}/{total_episodes}', 
                        fontsize=16, fontweight='bold', y=0.97)
            
            # Row 1: Injector Gas Actions - FULL TIME SERIES
            for i in range(3):
                ax = fig.add_subplot(gs[0, i])
                self._plot_episode_progression_injector_timeseries(ax, f'I{i+1}', episode_num, actions)
            
            # Row 2: Producer BHP Actions - FULL TIME SERIES
            for i in range(3):
                ax = fig.add_subplot(gs[1, i])
                self._plot_episode_progression_producer_bhp_timeseries(ax, f'P{i+1}', episode_num, actions)
            
            # Row 3: Producer Response - FULL TIME SERIES
            for i in range(3):
                ax = fig.add_subplot(gs[2, i])
                self._plot_episode_progression_producer_response_timeseries(ax, f'P{i+1}', episode_num, observations)
            
            # Row 4: Performance Metrics - FULL TIME SERIES
            ax_reward = fig.add_subplot(gs[3, 0])
            self._plot_episode_progression_reward_evolution(ax_reward, episode_num)
            
            ax_net_cf = fig.add_subplot(gs[3, 1])
            self._plot_episode_progression_net_cashflow_evolution(ax_net_cf, episode_num)
            
            ax_cum_cf = fig.add_subplot(gs[3, 2])
            self._plot_episode_progression_cumulative_cashflow_evolution(ax_cum_cf, episode_num)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.94, hspace=0.4, wspace=0.3)
            return fig
            
        except Exception as e:
            print(f"Error creating episode progression frame: {e}")
            return None
    
    def _plot_episode_progression_injector_timeseries(self, ax, well_name, episode_num, actions):
        """Plot complete injector gas injection time series for episode"""
        try:
            if not actions:
                ax.text(0.5, 0.5, f'{well_name}\nNo Data', transform=ax.transAxes, ha='center', va='center')
                return
            
            # Extract full episode time series
            steps = list(range(len(actions)))
            gas_values = []
            gas_key = f"{well_name}_Gas_ft3day"
            
            for action in actions:
                gas_value = action.get(gas_key, 0) / 1e6  # Convert to millions
                gas_values.append(gas_value)
            
            # Plot complete time series
            ax.plot(steps, gas_values, 'g-o', linewidth=2, markersize=3, alpha=0.8, color='green')
            ax.set_title(f'{well_name} Gas Injection\nEpisode {episode_num}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('MMft³/day', fontsize=10, color='green')
            ax.tick_params(axis='y', labelcolor='green', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_episode_progression_producer_bhp_timeseries(self, ax, well_name, episode_num, actions):
        """Plot complete producer BHP time series for episode"""
        try:
            if not actions:
                ax.text(0.5, 0.5, f'{well_name}\nNo Data', transform=ax.transAxes, ha='center', va='center')
                return
            
            # Extract full episode time series
            steps = list(range(len(actions)))
            bhp_values = []
            bhp_key = f"{well_name}_BHP_psi"
            
            for action in actions:
                bhp_value = action.get(bhp_key, 0)
                bhp_values.append(bhp_value)
            
            # Plot complete time series
            ax.plot(steps, bhp_values, 'r-o', linewidth=2, markersize=3, alpha=0.8, color='red')
            ax.set_title(f'{well_name} BHP Control\nEpisode {episode_num}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('psi', fontsize=10, color='red')
            ax.tick_params(axis='y', labelcolor='red', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_episode_progression_producer_response_timeseries(self, ax, well_name, episode_num, observations):
        """Plot complete producer response time series for episode"""
        try:
            if not observations:
                ax.text(0.5, 0.5, f'{well_name}\nNo Data', transform=ax.transAxes, ha='center', va='center')
                return
            
            # Extract full episode time series
            steps = list(range(len(observations)))
            water_values = []
            gas_values = []
            
            water_key = f"{well_name}_Water_ft3day"
            gas_key = f"{well_name}_Gas_ft3day"
            
            for obs in observations:
                water_val = obs.get(water_key, 0) / 1e6  # Convert to millions
                gas_val = obs.get(gas_key, 0) / 1e6
                water_values.append(water_val)
                gas_values.append(gas_val)
            
            # Create dual-axis plot for better scaling
            ax2 = ax.twinx()
            
            # Plot time series
            line1 = ax.plot(steps, water_values, 'b-o', linewidth=2, markersize=3, label='Water', alpha=0.8)
            line2 = ax2.plot(steps, gas_values, 'orange', linewidth=2, markersize=3, label='Gas', alpha=0.8, linestyle='-', marker='s')
            
            # Customize axes
            ax.set_title(f'{well_name} Production\nEpisode {episode_num}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Water (MMft³/day)', fontsize=10, color='blue')
            ax.tick_params(axis='y', labelcolor='blue', labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(True, alpha=0.3)
            
            ax2.set_ylabel('Gas (MMft³/day)', fontsize=10, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange', labelsize=9)
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_episode_progression_reward_evolution(self, ax, episode_num):
        """Plot complete reward evolution for episode"""
        try:
            episode_data = self.training_orchestrator.get_episode_data(episode_num)
            rewards = episode_data.get('rewards', [])
            
            if not rewards:
                ax.text(0.5, 0.5, 'No reward data', transform=ax.transAxes, ha='center', va='center')
                return
            
            steps = list(range(len(rewards)))
            cumulative_reward = sum(rewards)
            
            # Plot complete reward evolution
            ax.plot(steps, rewards, 'b-o', linewidth=2, markersize=4, alpha=0.8)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'Reward Evolution\nEpisode {episode_num} (Total: {cumulative_reward:.2f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Step Reward', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_episode_progression_net_cashflow_evolution(self, ax, episode_num):
        """Plot complete net cashflow evolution for episode including pre-project"""
        try:
            episode_data = self.training_orchestrator.get_episode_data(episode_num)
            economic_breakdown = episode_data.get('economic_breakdown', [])
            
            if not economic_breakdown:
                ax.text(0.5, 0.5, 'No economic data', transform=ax.transAxes, ha='center', va='center')
                return
            
            # Get pre-project and scale parameters
            years_before = self.data.get('years_before_project', 5) if self.data else 5
            cost_per_year = self.data.get('capital_cost_per_year', 100000000.0) if self.data else 100000000.0
            scale_factor = self.config.rl_model['economics']['scale_factor']
            
            # Create complete timeline (pre-project + operational)
            pre_project_cashflows = []
            for year in range(1, years_before + 1):
                annual_capex = -cost_per_year / scale_factor
                pre_project_cashflows.append(annual_capex)
            
            operational_cashflows = []
            for breakdown in economic_breakdown:
                cashflow = breakdown.get('operational_cashflow', breakdown.get('net_step_cashflow', 0)) / scale_factor
                operational_cashflows.append(cashflow)
            
            # Combined timeline
            complete_cashflows = pre_project_cashflows + operational_cashflows
            pre_project_years = list(range(-years_before, 0))
            operational_years = list(range(len(operational_cashflows)))
            complete_years = pre_project_years + operational_years
            
            # Plot with phase distinction
            if len(pre_project_cashflows) > 0:
                ax.bar(pre_project_years, pre_project_cashflows, color='darkred', alpha=0.8, width=0.8, label='CapEx')
            if len(operational_cashflows) > 0:
                colors = ['green' if cf >= 0 else 'red' for cf in operational_cashflows]
                ax.bar(operational_years, operational_cashflows, color=colors, alpha=0.7, width=0.8, label='Operations')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=-0.5, color='orange', linestyle=':', alpha=0.7, linewidth=2)
            ax.set_title(f'Net Cashflow Evolution\nEpisode {episode_num}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Project Year', fontsize=10)
            ax.set_ylabel('Annual Cashflow (Million USD)', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def _plot_episode_progression_cumulative_cashflow_evolution(self, ax, episode_num):
        """Plot complete cumulative cashflow evolution for episode including pre-project debt"""
        try:
            episode_data = self.training_orchestrator.get_episode_data(episode_num)
            economic_breakdown = episode_data.get('economic_breakdown', [])
            
            if not economic_breakdown:
                ax.text(0.5, 0.5, 'No economic data', transform=ax.transAxes, ha='center', va='center')
                return
            
            # Get pre-project and scale parameters
            years_before = self.data.get('years_before_project', 5) if self.data else 5
            cost_per_year = self.data.get('capital_cost_per_year', 100000000.0) if self.data else 100000000.0
            scale_factor = self.config.rl_model['economics']['scale_factor']
            
            # Create pre-project cumulative debt
            pre_project_cumulative = []
            running_debt = 0
            for year in range(1, years_before + 1):
                annual_capex = cost_per_year / scale_factor
                running_debt -= annual_capex
                pre_project_cumulative.append(running_debt)
            
            # Create operational cumulative recovery
            operational_cumulative = []
            operational_total = running_debt  # Start from final debt
            
            for breakdown in economic_breakdown:
                cashflow = breakdown.get('operational_cashflow', breakdown.get('net_step_cashflow', 0)) / scale_factor
                operational_total += cashflow
                operational_cumulative.append(operational_total)
            
            # Combined timeline
            complete_cumulative = pre_project_cumulative + operational_cumulative
            pre_project_years = list(range(-years_before, 0))
            operational_years = list(range(len(operational_cumulative)))
            complete_years = pre_project_years + operational_years
            
            # Plot cumulative line
            final_value = complete_cumulative[-1] if complete_cumulative else 0
            overall_color = 'green' if final_value >= 0 else 'red'
            
            ax.plot(complete_years, complete_cumulative, color=overall_color, linewidth=3, marker='o', markersize=3, alpha=0.8)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Break-even')
            ax.axvline(x=-0.5, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Operations Start')
            
            # Fill areas
            if len(complete_cumulative) > 1:
                ax.fill_between(complete_years, complete_cumulative, 0, 
                               where=[cf <= 0 for cf in complete_cumulative], 
                               color='red', alpha=0.3, interpolate=True, label='Debt/Loss')
                ax.fill_between(complete_years, complete_cumulative, 0, 
                               where=[cf > 0 for cf in complete_cumulative], 
                               color='green', alpha=0.3, interpolate=True, label='Profit')
            
            ax.set_title(f'Cumulative Cashflow Evolution\nEpisode {episode_num} (Final: ${final_value:.1f}M)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Project Year', fontsize=10)
            ax.set_ylabel('Cumulative Cashflow (Million USD)', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='lower right')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
    
    def display(self):
        """Display the visualization dashboard"""
        if not WIDGETS_AVAILABLE:
            print("❌ Widgets not available")
            return
        
        interface = self.create_interactive_episode_selector()
        if interface:
            display(interface)


def launch_interactive_scientific_analysis(training_orchestrator=None, config=None, training_dashboard=None):
    """
    Launch interactive scientific analysis dashboard
    
    Args:
        training_orchestrator: EnhancedTrainingOrchestrator with episode data (optional, will auto-detect)
        config: Configuration object (optional, will load from config.yaml)
        training_dashboard: Training dashboard instance (optional, used to get orchestrator)
    
    Returns:
        ScientificVisualization: Dashboard instance
    """
    if not WIDGETS_AVAILABLE:
        print("❌ Interactive widgets not available")
        return None
    
    # Auto-load config if not provided
    if config is None:
        from RL_Refactored.utilities import Config
        config = Config('config.yaml')
    
    # Auto-detect training orchestrator if not provided
    if training_orchestrator is None:
        from RL_Refactored.utilities import get_training_orchestrator
        training_orchestrator = get_training_orchestrator(training_dashboard)
    
    if not training_orchestrator or not training_orchestrator.has_best_episode_data():
        print("⚠️ No training data available for visualization")
        print("   Please complete training first to generate visualization data")
        print("   💡 Make sure you've run the training dashboard (Step 2) before visualization")
        return None
    
    print(f"\n🔬 Launching Enhanced RL Results Dashboard...")
    viz = ScientificVisualization(training_orchestrator, config)
    viz.display()
    
    return viz
