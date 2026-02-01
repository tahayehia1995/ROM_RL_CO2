"""
Optimizer Results Dashboard
============================

Interactive dashboard for visualizing classical optimization results.
Similar to launch_interactive_scientific_analysis but focused on optimizer outputs.

Provides visualizations for:
1. Optimization Performance (convergence, gradients)
2. Optimal Controls (per-well trajectories)
3. Observations/Production Profiles
4. 3D Spatial Visualization
5. Economic Summary
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets or matplotlib not available.")

from .base_optimizer import OptimizationResult


class OptimizerResultsDashboard:
    """
    Interactive dashboard for visualizing optimization results.
    
    Displays:
    - Convergence plots
    - Optimal control trajectories
    - Production profiles
    - 3D spatial fields
    - Economic analysis
    """
    
    def __init__(self, result: OptimizationResult, config: Optional[Any] = None):
        """
        Initialize results dashboard.
        
        Args:
            result: OptimizationResult from optimizer
            config: Optional configuration object for additional info
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets required for dashboard")
        
        self.result = result
        self.config = config
        
        # Extract key data
        self.num_steps = result.optimal_controls.shape[0]
        self.num_controls = result.optimal_controls.shape[1]
        self.num_prod = 3  # Default, can be updated from config
        self.num_inj = 3
        
        if config:
            self.num_prod = config.data.get('num_prod', 3)
            self.num_inj = config.data.get('num_inj', 3)
        
        # Well names
        self.producer_names = [f'P{i+1}' for i in range(self.num_prod)]
        self.injector_names = [f'I{i+1}' for i in range(self.num_inj)]
        
        # Create dashboard
        self._create_dashboard()
    
    def _create_dashboard(self):
        """Create the main dashboard interface."""
        # Header
        header = widgets.HTML(f"""
        <div style="background: linear-gradient(135deg, #2d3436 0%, #636e72 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">Optimization Results: {self.result.optimizer_type}</h2>
            <p style="color: #dfe6e9; margin: 5px 0 0 0;">
                Optimal NPV: {self.result.optimal_objective:.6f} | 
                Improvement: {self.result.improvement_ratio()*100:.2f}% |
                Iterations: {self.result.num_iterations}
            </p>
        </div>
        """)
        
        # Create tabs
        self.performance_tab = widgets.VBox([])
        self.controls_tab = widgets.VBox([])
        self.observations_tab = widgets.VBox([])
        self.spatial_tab = widgets.VBox([])
        self.economics_tab = widgets.VBox([])
        
        tabs = widgets.Tab(children=[
            self.performance_tab,
            self.controls_tab,
            self.observations_tab,
            self.spatial_tab,
            self.economics_tab
        ])
        tabs.set_title(0, '📈 Performance')
        tabs.set_title(1, '🎛️ Controls')
        tabs.set_title(2, '📊 Observations')
        tabs.set_title(3, '🗺️ Spatial')
        tabs.set_title(4, '💰 Economics')
        
        # Assemble dashboard
        self.dashboard = widgets.VBox([header, tabs])
        
        # Populate tabs
        self._populate_performance_tab()
        self._populate_controls_tab()
        self._populate_observations_tab()
        self._populate_spatial_tab()
        self._populate_economics_tab()
    
    def _populate_performance_tab(self):
        """Populate optimization performance tab."""
        output = widgets.Output()
        
        with output:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Convergence plot
            ax1 = axes[0, 0]
            if self.result.objective_history:
                ax1.plot(self.result.objective_history, 'b-', linewidth=2)
                ax1.axhline(y=self.result.optimal_objective, color='g', linestyle='--', 
                           label=f'Optimal: {self.result.optimal_objective:.6f}')
                ax1.axhline(y=self.result.initial_objective, color='r', linestyle=':', 
                           label=f'Initial: {self.result.initial_objective:.6f}')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective (NPV)')
            ax1.set_title('Convergence History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gradient norm history
            ax2 = axes[0, 1]
            if self.result.gradient_norm_history:
                ax2.semilogy(self.result.gradient_norm_history, 'r-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Gradient Norm (log scale)')
            ax2.set_title('Gradient Norm History')
            ax2.grid(True, alpha=0.3)
            
            # Summary text
            ax3 = axes[1, 0]
            ax3.axis('off')
            summary_text = f"""
            Optimization Summary
            {'='*40}
            Optimizer: {self.result.optimizer_type}
            Termination: {self.result.termination_reason}
            Converged: {self.result.convergence_achieved}
            
            Iterations: {self.result.num_iterations}
            Function Evaluations: {self.result.num_function_evaluations}
            Gradient Evaluations: {self.result.num_gradient_evaluations}
            
            Total Time: {self.result.total_time_seconds:.2f} seconds
            Time per Iteration: {self.result.total_time_seconds/max(1,self.result.num_iterations):.3f} s
            
            Realizations Used: {self.result.num_realizations}
            Control Variables: {np.prod(self.result.optimal_controls.shape)}
            """
            ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes, 
                    fontsize=11, verticalalignment='center', fontfamily='monospace')
            ax3.set_title('Optimization Statistics')
            
            # Improvement bar chart
            ax4 = axes[1, 1]
            categories = ['Initial', 'Optimal']
            values = [self.result.initial_objective, self.result.optimal_objective]
            colors = ['#e74c3c', '#27ae60']
            bars = ax4.bar(categories, values, color=colors)
            ax4.set_ylabel('Objective (NPV)')
            ax4.set_title(f'Improvement: {self.result.improvement_ratio()*100:.2f}%')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.show()
        
        self.performance_tab.children = [
            widgets.HTML("<h3>📈 Optimization Performance</h3>"),
            output
        ]
    
    def _populate_controls_tab(self):
        """Populate optimal controls visualization tab with well selection."""
        # Create well selection dropdowns
        all_producers = ['All Producers'] + self.producer_names
        all_injectors = ['All Injectors'] + self.injector_names
        
        producer_selector = widgets.Dropdown(
            options=all_producers,
            value='All Producers',
            description='Producer:',
            style={'description_width': '80px'}
        )
        
        injector_selector = widgets.Dropdown(
            options=all_injectors,
            value='All Injectors',
            description='Injector:',
            style={'description_width': '80px'}
        )
        
        show_initial = widgets.Checkbox(
            value=True,
            description='Show Initial Controls',
            style={'description_width': '150px'}
        )
        
        output = widgets.Output()
        
        def update_controls_plot(*args):
            with output:
                clear_output(wait=True)
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                timesteps = np.arange(self.num_steps)
                colors = plt.cm.tab10.colors
                
                # Producer BHP
                ax1 = axes[0]
                selected_prod = producer_selector.value
                
                if selected_prod == 'All Producers':
                    producers_to_plot = list(range(self.num_prod))
                else:
                    producers_to_plot = [self.producer_names.index(selected_prod)]
                
                for idx, p in enumerate(producers_to_plot):
                    bhp_values = self.result.optimal_controls[:, p]
                    color = colors[p % len(colors)]
                    ax1.plot(timesteps, bhp_values, 'o-', linewidth=2, color=color,
                            label=f'{self.producer_names[p]} (optimal)', markersize=4)
                    
                    if show_initial.value and self.result.initial_controls is not None:
                        init_bhp = self.result.initial_controls[:, p]
                        ax1.plot(timesteps, init_bhp, '--', alpha=0.5, color=color,
                                label=f'{self.producer_names[p]} (initial)')
                
                ax1.set_xlabel('Timestep')
                ax1.set_ylabel('BHP (psi)')
                ax1.set_title('Producer Bottom Hole Pressure')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                
                # Gas injection
                ax2 = axes[1]
                selected_inj = injector_selector.value
                
                if selected_inj == 'All Injectors':
                    injectors_to_plot = list(range(self.num_inj))
                else:
                    injectors_to_plot = [self.injector_names.index(selected_inj)]
                
                for idx, i in enumerate(injectors_to_plot):
                    gas_values = self.result.optimal_controls[:, self.num_prod + i]
                    color = colors[(i + 3) % len(colors)]  # Offset colors for injectors
                    ax2.plot(timesteps, gas_values / 1e6, 'o-', linewidth=2, color=color,
                            label=f'{self.injector_names[i]} (optimal)', markersize=4)
                    
                    if show_initial.value and self.result.initial_controls is not None:
                        init_gas = self.result.initial_controls[:, self.num_prod + i]
                        ax2.plot(timesteps, init_gas / 1e6, '--', alpha=0.5, color=color,
                                label=f'{self.injector_names[i]} (initial)')
                
                ax2.set_xlabel('Timestep')
                ax2.set_ylabel('Gas Injection (MMscf/day)')
                ax2.set_title('Injector Gas Injection Rate')
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Connect observers
        producer_selector.observe(update_controls_plot, names='value')
        injector_selector.observe(update_controls_plot, names='value')
        show_initial.observe(update_controls_plot, names='value')
        
        # Initial plot
        update_controls_plot()
        
        # Control statistics table
        stats_html = self._create_control_stats_table()
        
        controls_box = widgets.HBox([producer_selector, injector_selector, show_initial])
        
        self.controls_tab.children = [
            widgets.HTML("<h3>🎛️ Optimal Well Controls</h3>"),
            controls_box,
            output,
            widgets.HTML("<h4>Control Statistics:</h4>"),
            widgets.HTML(stats_html)
        ]
    
    def _create_control_stats_table(self) -> str:
        """Create HTML table with control statistics."""
        rows = []
        
        # Producer BHP stats
        for p in range(self.num_prod):
            bhp = self.result.optimal_controls[:, p]
            rows.append(f"""
            <tr>
                <td>{self.producer_names[p]} BHP</td>
                <td>{np.mean(bhp):.2f}</td>
                <td>{np.min(bhp):.2f}</td>
                <td>{np.max(bhp):.2f}</td>
                <td>{np.std(bhp):.2f}</td>
            </tr>
            """)
        
        # Gas injection stats
        for i in range(self.num_inj):
            gas = self.result.optimal_controls[:, self.num_prod + i]
            rows.append(f"""
            <tr>
                <td>{self.injector_names[i]} Gas</td>
                <td>{np.mean(gas)/1e6:.2f}</td>
                <td>{np.min(gas)/1e6:.2f}</td>
                <td>{np.max(gas)/1e6:.2f}</td>
                <td>{np.std(gas)/1e6:.2f}</td>
            </tr>
            """)
        
        return f"""
        <table style="width:80%; border-collapse: collapse;">
            <tr style="background-color: #2d3436; color: white;">
                <th style="padding: 10px;">Well Control</th>
                <th style="padding: 10px;">Mean</th>
                <th style="padding: 10px;">Min</th>
                <th style="padding: 10px;">Max</th>
                <th style="padding: 10px;">Std Dev</th>
            </tr>
            {''.join(rows)}
        </table>
        <p style="color: gray;"><i>BHP in psi, Gas in MMscf/day</i></p>
        """
    
    def _populate_observations_tab(self):
        """Populate observations/production tab with well selection."""
        if self.result.optimal_observations is None:
            self.observations_tab.children = [
                widgets.HTML("<h3>📊 Production Profiles</h3>"),
                widgets.HTML("<p>No observation data available.</p>")
            ]
            return
        
        # Create well selection dropdowns
        all_producers = ['All Producers'] + self.producer_names
        all_injectors = ['All Injectors'] + self.injector_names
        
        producer_selector = widgets.Dropdown(
            options=all_producers,
            value='All Producers',
            description='Producer:',
            style={'description_width': '80px'}
        )
        
        injector_selector = widgets.Dropdown(
            options=all_injectors,
            value='All Injectors',
            description='Injector:',
            style={'description_width': '80px'}
        )
        
        obs_type_selector = widgets.Dropdown(
            options=['All Observations', 'Injector BHP Only', 'Gas Production Only', 'Water Production Only'],
            value='All Observations',
            description='Show:',
            style={'description_width': '80px'}
        )
        
        output = widgets.Output()
        
        def update_observations_plot(*args):
            with output:
                clear_output(wait=True)
                
                obs = self.result.optimal_observations
                if obs.ndim == 3:
                    obs = obs.squeeze(1)
                
                timesteps = np.arange(len(obs))
                colors = plt.cm.tab10.colors
                
                obs_type = obs_type_selector.value
                
                # Determine which plots to show
                show_inj_bhp = obs_type in ['All Observations', 'Injector BHP Only']
                show_gas_prod = obs_type in ['All Observations', 'Gas Production Only']
                show_water_prod = obs_type in ['All Observations', 'Water Production Only']
                
                num_plots = sum([show_inj_bhp, show_gas_prod, show_water_prod])
                if num_plots == 0:
                    num_plots = 1
                
                fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots))
                if num_plots == 1:
                    axes = [axes]
                
                ax_idx = 0
                
                # Injector BHP
                if show_inj_bhp:
                    ax = axes[ax_idx]
                    ax_idx += 1
                    
                    selected_inj = injector_selector.value
                    if selected_inj == 'All Injectors':
                        injectors_to_plot = list(range(self.num_inj))
                    else:
                        injectors_to_plot = [self.injector_names.index(selected_inj)]
                    
                    for i in injectors_to_plot:
                        color = colors[(i + 3) % len(colors)]
                        ax.plot(timesteps, obs[:, i], 'o-', linewidth=2, color=color,
                                label=self.injector_names[i], markersize=4)
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('BHP (psi)')
                    ax.set_title('Injector Bottom Hole Pressure')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
                
                # Gas production
                if show_gas_prod:
                    ax = axes[ax_idx]
                    ax_idx += 1
                    
                    selected_prod = producer_selector.value
                    if selected_prod == 'All Producers':
                        producers_to_plot = list(range(self.num_prod))
                    else:
                        producers_to_plot = [self.producer_names.index(selected_prod)]
                    
                    for p in producers_to_plot:
                        gas_prod = obs[:, self.num_inj + p]
                        color = colors[p % len(colors)]
                        ax.plot(timesteps, gas_prod / 1e6, 'o-', linewidth=2, color=color,
                                label=self.producer_names[p], markersize=4)
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('Gas Production (MMscf/day)')
                    ax.set_title('Gas Production Rate')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
                
                # Water production
                if show_water_prod:
                    ax = axes[ax_idx]
                    ax_idx += 1
                    
                    selected_prod = producer_selector.value
                    if selected_prod == 'All Producers':
                        producers_to_plot = list(range(self.num_prod))
                    else:
                        producers_to_plot = [self.producer_names.index(selected_prod)]
                    
                    for p in producers_to_plot:
                        water_prod = obs[:, self.num_inj + self.num_prod + p]
                        color = colors[p % len(colors)]
                        ax.plot(timesteps, water_prod / 1e3, 'o-', linewidth=2, color=color,
                                label=self.producer_names[p], markersize=4)
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('Water Production (Mscf/day)')
                    ax.set_title('Water Production Rate')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Connect observers
        producer_selector.observe(update_observations_plot, names='value')
        injector_selector.observe(update_observations_plot, names='value')
        obs_type_selector.observe(update_observations_plot, names='value')
        
        # Initial plot
        update_observations_plot()
        
        # Cumulative production
        cum_output = widgets.Output()
        obs = self.result.optimal_observations
        if obs.ndim == 3:
            obs = obs.squeeze(1)
        timesteps = np.arange(len(obs))
        
        with cum_output:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Cumulative gas
            ax1 = axes[0]
            total_gas_prod = np.sum(obs[:, self.num_inj:self.num_inj+self.num_prod], axis=1)
            total_gas_inj = np.sum(self.result.optimal_controls[:, self.num_prod:], axis=1)
            cum_gas_prod = np.cumsum(total_gas_prod)
            cum_gas_inj = np.cumsum(total_gas_inj)
            
            ax1.plot(timesteps, cum_gas_prod / 1e9, 'r-', linewidth=2, label='Cumulative Production')
            ax1.plot(timesteps, cum_gas_inj / 1e9, 'b-', linewidth=2, label='Cumulative Injection')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Cumulative Gas (Bscf)')
            ax1.set_title('Cumulative Gas Balance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Cumulative water
            ax2 = axes[1]
            total_water = np.sum(obs[:, self.num_inj+self.num_prod:], axis=1)
            cum_water = np.cumsum(total_water)
            ax2.plot(timesteps, cum_water / 1e6, 'c-', linewidth=2)
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Cumulative Water (MMscf)')
            ax2.set_title('Cumulative Water Production')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        obs_controls_box = widgets.HBox([producer_selector, injector_selector, obs_type_selector])
        
        self.observations_tab.children = [
            widgets.HTML("<h3>📊 Production Profiles</h3>"),
            obs_controls_box,
            output,
            widgets.HTML("<h4>Cumulative Production:</h4>"),
            cum_output
        ]
    
    def _populate_spatial_tab(self):
        """Populate 3D spatial visualization tab (matching RL dashboard style)."""
        if self.result.optimal_spatial_states is None:
            self.spatial_tab.children = [
                widgets.HTML("<h3>🗺️ Spatial Visualization</h3>"),
                widgets.HTML("<p>No spatial data available. Enable spatial state capture in optimizer.</p>")
            ]
            return
        
        # Create controls
        num_spatial_steps = len(self.result.optimal_spatial_states)
        spatial_shape = self.result.optimal_spatial_states[0].shape
        
        # Get dimensions
        if len(spatial_shape) == 5:  # (batch, channels, X, Y, Z)
            num_channels = spatial_shape[1]
            num_layers = spatial_shape[4]
        elif len(spatial_shape) == 4:  # (channels, X, Y, Z)
            num_channels = spatial_shape[0]
            num_layers = spatial_shape[3]
        else:
            num_channels = 2
            num_layers = 25
        
        # Channel names based on actual training channels (ch2: SG, PRES)
        # These match the ROM training - NOT SW which isn't trained
        channel_config = {
            2: [('SG (Gas Saturation)', 'fraction'), ('PRES (Pressure)', 'psi')],
            3: [('SG (Gas Saturation)', 'fraction'), ('PRES (Pressure)', 'psi'), ('SW (Water Sat.)', 'fraction')],
            4: [('SG (Gas Saturation)', 'fraction'), ('PRES (Pressure)', 'psi'), 
                ('PERMI (Permeability I)', 'mD'), ('POROS (Porosity)', 'fraction')],
        }
        
        channel_info = channel_config.get(num_channels, 
            [(f'Channel {i}', 'units') for i in range(num_channels)])
        channel_names = [c[0] for c in channel_info]
        channel_units = [c[1] for c in channel_info]
        
        timestep_slider = widgets.IntSlider(
            value=0, min=0, max=num_spatial_steps-1,
            description='Timestep:', style={'description_width': '80px'}
        )
        channel_dropdown = widgets.Dropdown(
            options=[(name, i) for i, name in enumerate(channel_names)],
            description='Channel:', style={'description_width': '80px'}
        )
        layer_slider = widgets.IntSlider(
            value=num_layers//2, min=0, max=num_layers-1,
            description='Z Layer:', style={'description_width': '80px'}
        )
        
        output = widgets.Output()
        
        def update_plot(*args):
            with output:
                clear_output(wait=True)
                
                t = timestep_slider.value
                ch = channel_dropdown.value
                layer = layer_slider.value
                
                spatial_data = self.result.optimal_spatial_states[t]
                if isinstance(spatial_data, torch.Tensor):
                    spatial_data = spatial_data.cpu().numpy()
                
                # Extract slice
                if len(spatial_data.shape) == 5:
                    slice_data = spatial_data[0, ch, :, :, layer]
                elif len(spatial_data.shape) == 4:
                    slice_data = spatial_data[ch, :, :, layer]
                else:
                    slice_data = spatial_data[:, :, layer] if len(spatial_data.shape) == 3 else spatial_data
                
                # Denormalize if possible (matching RL dashboard)
                slice_data_denorm = self._denormalize_spatial_field(slice_data, ch, channel_names)
                
                # Create visualization (matching RL dashboard style)
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Use jet colormap with bilinear interpolation (matching RL)
                cmap = plt.cm.jet.copy()
                cmap.set_bad('white', alpha=0.3)
                
                # Calculate color range from data
                vmin = np.nanmin(slice_data_denorm)
                vmax = np.nanmax(slice_data_denorm)
                
                # Plot with RL dashboard style
                im = ax.imshow(slice_data_denorm.T, origin='lower', cmap=cmap, 
                              vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear')
                
                # Add colorbar (matching RL style)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f'{channel_names[ch]} ({channel_units[ch]})', fontsize=12, fontweight='bold')
                
                # Customize (matching RL style)
                ax.set_title(f'Optimal Solution | Step {t} | Layer {layer} | {channel_names[ch]}', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('I Index', fontsize=12)
                ax.set_ylabel('J Index', fontsize=12)
                
                # Remove grid lines from spatial plot
                ax.grid(False)
                
                plt.tight_layout()
                plt.show()
        
        # Connect observers
        timestep_slider.observe(update_plot, names='value')
        channel_dropdown.observe(update_plot, names='value')
        layer_slider.observe(update_plot, names='value')
        
        # Initial plot
        update_plot()
        
        controls = widgets.HBox([timestep_slider, channel_dropdown, layer_slider])
        
        self.spatial_tab.children = [
            widgets.HTML("<h3>🗺️ Spatial Field Visualization</h3>"),
            widgets.HTML("<p><i>Showing predicted fields from ROM (trained channels only)</i></p>"),
            controls,
            output
        ]
    
    def _denormalize_spatial_field(self, field_data, channel_idx, channel_names):
        """Denormalize spatial field data back to physical units."""
        try:
            # Load normalization parameters
            import glob
            import json
            from pathlib import Path
            
            norm_dir = Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'processed_data'
            json_files = list(norm_dir.glob('normalization_parameters_*.json'))
            
            if not json_files:
                return field_data
            
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                norm_params = json.load(f)
            
            # Map channel index to variable name
            # For ch2 models: 0=SG, 1=PRES
            channel_map = {0: 'SG', 1: 'PRES', 2: 'PERMI', 3: 'POROS'}
            var_name = channel_map.get(channel_idx, None)
            
            if var_name and var_name in norm_params.get('spatial_channels', {}):
                params = norm_params['spatial_channels'][var_name].get('parameters', {})
                data_min = float(params.get('min', 0))
                data_max = float(params.get('max', 1))
                
                # Denormalize: data * (max - min) + min
                field_data = field_data * (data_max - data_min) + data_min
            
            return field_data
            
        except Exception as e:
            print(f"⚠️ Warning: Could not denormalize field: {e}")
            return field_data
    
    def _populate_economics_tab(self):
        """Populate economic analysis tab (matching RL dashboard style with 5-year capital cost)."""
        output = widgets.Output()
        
        # Get economic parameters
        years_before = 5  # Pre-project years
        capital_cost_per_year = 100000000.0  # $100M per year
        scale_factor = 1000000.0  # Scale for display
        
        if self.config and hasattr(self.config, 'rl_model'):
            econ = self.config.rl_model.get('economics', {})
            years_before = econ.get('years_before_project_start', 5)
            capital_cost_per_year = econ.get('capital_cost_per_year', 100000000.0)
            scale_factor = econ.get('scale_factor', 1000000.0)
        
        with output:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            breakdown = self.result.economic_breakdown or {}
            
            # NPV components pie chart
            ax1 = axes[0, 0]
            if breakdown:
                labels = ['Gas Inj Revenue', 'Water Penalty', 'Gas Prod Penalty']
                sizes = [
                    abs(breakdown.get('gas_injection_revenue', 0)),
                    abs(breakdown.get('water_production_penalty', 0)),
                    abs(breakdown.get('gas_production_penalty', 0))
                ]
                colors = ['#27ae60', '#e74c3c', '#f39c12']
                
                if sum(sizes) > 0:
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('NPV Component Breakdown')
                else:
                    ax1.text(0.5, 0.5, 'No breakdown data', ha='center', va='center')
                    ax1.set_title('NPV Components')
            else:
                ax1.text(0.5, 0.5, 'Economic breakdown not available', 
                        ha='center', va='center', fontsize=12)
                ax1.set_title('NPV Components')
            
            # Step-wise NPV (matching RL)
            ax2 = axes[0, 1]
            if breakdown and 'step_npvs' in breakdown:
                step_npvs = breakdown['step_npvs']
                colors_bar = ['#27ae60' if v >= 0 else '#e74c3c' for v in step_npvs]
                ax2.bar(range(len(step_npvs)), step_npvs, color=colors_bar, alpha=0.7)
                ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax2.set_xlabel('Timestep (Operational Year)')
                ax2.set_ylabel('Step NPV ($M)')
                ax2.set_title('NPV per Timestep')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Step NPV data not available', 
                        ha='center', va='center', fontsize=12)
                ax2.set_title('NPV per Timestep')
            
            # Cumulative NPV with 5-year Capital Cost (matching RL dashboard)
            ax3 = axes[1, 0]
            if breakdown and 'step_npvs' in breakdown:
                # Create complete project timeline (pre-project + operational)
                step_npvs = breakdown['step_npvs']
                
                # Pre-project years: Fixed capital expenditure
                pre_project_cashflows = []
                for year in range(1, years_before + 1):
                    annual_capex = -capital_cost_per_year / scale_factor
                    pre_project_cashflows.append(annual_capex)
                
                # Operational cashflows (from optimization)
                operational_cashflows = step_npvs
                
                # Create year labels
                pre_project_years = list(range(-years_before, 0))
                operational_years = list(range(0, len(operational_cashflows)))
                all_years = pre_project_years + operational_years
                all_cashflows = pre_project_cashflows + list(operational_cashflows)
                
                # Calculate cumulative NPV including capital cost
                cum_npv = np.cumsum(all_cashflows)
                
                # Plot with phase distinction
                ax3.plot(all_years, cum_npv, 'g-', linewidth=2.5, label='Cumulative NPV')
                
                # Color fill based on positive/negative
                ax3.fill_between(all_years, cum_npv, 0, 
                                where=np.array(cum_npv) >= 0, 
                                color='green', alpha=0.3, label='Profit')
                ax3.fill_between(all_years, cum_npv, 0, 
                                where=np.array(cum_npv) < 0, 
                                color='red', alpha=0.3, label='Loss')
                
                # Add vertical line at year 0 (project start)
                ax3.axvline(x=0, color='blue', linestyle='--', linewidth=1.5, 
                           label='Production Start')
                ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                
                # Find break-even point
                break_even_year = None
                for i, (year, npv) in enumerate(zip(all_years, cum_npv)):
                    if i > 0 and cum_npv[i-1] < 0 and npv >= 0:
                        break_even_year = year
                        ax3.axvline(x=year, color='orange', linestyle=':', linewidth=2)
                        ax3.annotate(f'Break-even\nYear {year}', 
                                    xy=(year, 0), xytext=(year+1, cum_npv[i]/2),
                                    arrowprops=dict(arrowstyle='->', color='orange'),
                                    fontsize=9, fontweight='bold')
                        break
                
                ax3.set_xlabel('Year (0 = Production Start)')
                ax3.set_ylabel('Cumulative NPV ($M)')
                ax3.set_title(f'Complete Project Lifecycle\n(Capital: ${years_before * capital_cost_per_year / 1e6:.0f}M over {years_before} years)')
                ax3.legend(loc='lower right', fontsize=8)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Cumulative NPV not available', 
                        ha='center', va='center', fontsize=12)
                ax3.set_title('Cumulative NPV')
            
            # Summary text with project-level economics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calculate project-level metrics
            total_capex = years_before * capital_cost_per_year / scale_factor
            operational_npv = self.result.optimal_objective
            project_npv = operational_npv - total_capex
            
            summary = f"""
Economic Summary
{'='*40}

PROJECT-LEVEL ANALYSIS:
  Capital Investment: ${total_capex:.2f}M
  ({years_before} years × ${capital_cost_per_year/1e6:.0f}M/year)
  
  Operational NPV: {operational_npv:.6f}
  Project NPV: {project_npv:.6f}
  
OPTIMIZATION RESULTS:
  Initial NPV: {self.result.initial_objective:.6f}
  Optimal NPV: {self.result.optimal_objective:.6f}
  Improvement: {self.result.improvement_ratio()*100:.2f}%

"""
            
            if breakdown:
                summary += f"""COMPONENT BREAKDOWN:
  Gas Inj Revenue: ${breakdown.get('gas_injection_revenue', 0):.2f}M
  Water Penalty: ${breakdown.get('water_production_penalty', 0):.2f}M
  Gas Prod Penalty: ${breakdown.get('gas_production_penalty', 0):.2f}M
  
  Timesteps: {breakdown.get('num_steps', self.num_steps)}
"""
            
            ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
            ax4.set_title('Economic Summary')
            
            plt.tight_layout()
            plt.show()
        
        self.economics_tab.children = [
            widgets.HTML("<h3>💰 Economic Analysis</h3>"),
            output
        ]
    
    def display(self):
        """Display the dashboard."""
        display(self.dashboard)
        return self
    
    def _ipython_display_(self):
        """IPython display hook."""
        self.display()
    
    def export_results(self, filepath: str):
        """
        Export optimization results to file.
        
        Args:
            filepath: Path to save results (JSON or NPZ)
        """
        import json
        
        if filepath.endswith('.json'):
            # Export to JSON (without large arrays)
            export_data = {
                'optimizer_type': self.result.optimizer_type,
                'optimal_objective': self.result.optimal_objective,
                'initial_objective': self.result.initial_objective,
                'improvement_ratio': self.result.improvement_ratio(),
                'num_iterations': self.result.num_iterations,
                'total_time_seconds': self.result.total_time_seconds,
                'convergence_achieved': self.result.convergence_achieved,
                'termination_reason': self.result.termination_reason,
                'num_realizations': self.result.num_realizations,
                'optimizer_params': self.result.optimizer_params,
                'economic_breakdown': self.result.economic_breakdown,
                'optimal_controls': self.result.optimal_controls.tolist(),
                'objective_history': self.result.objective_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif filepath.endswith('.npz'):
            # Export to NPZ (with full arrays)
            np.savez(
                filepath,
                optimal_controls=self.result.optimal_controls,
                optimal_objective=self.result.optimal_objective,
                objective_history=self.result.objective_history,
                gradient_norm_history=self.result.gradient_norm_history,
                optimal_observations=self.result.optimal_observations
            )
        
        print(f"Results exported to {filepath}")


def launch_optimizer_results_dashboard(
    result: OptimizationResult,
    config: Optional[Any] = None
) -> OptimizerResultsDashboard:
    """
    Launch the optimizer results dashboard.
    
    Single-line launcher for use in Jupyter notebooks.
    
    Args:
        result: OptimizationResult from optimizer
        config: Optional configuration object
        
    Returns:
        OptimizerResultsDashboard instance
    """
    dashboard = OptimizerResultsDashboard(result, config)
    dashboard.display()
    return dashboard
