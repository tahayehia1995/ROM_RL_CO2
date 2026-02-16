"""
Evaluation Dashboard
=====================

Interactive dashboard for evaluating trained RL policies.
Provides checkpoint selection, evaluation controls, and results visualization.
"""

import os
import csv
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Check for widget availability
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None

# Check for plotting availability
try:
    import matplotlib.pyplot as plt
    from IPython.display import display
    # Don't set backend - let Jupyter/IPython handle it automatically
    # Setting 'Agg' prevents display in notebooks
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    display = print  # Fallback

from .evaluator import PolicyEvaluator
from .results import EvaluationResults, ComparisonResults
from .baselines import get_all_baselines


class EvaluationDashboard:
    """
    Interactive dashboard for RL policy evaluation.
    
    Features:
    - Checkpoint file browser
    - Number of evaluation episodes selector
    - Deterministic vs stochastic toggle
    - Baseline selection checkboxes
    - Results visualization (box plots, trajectories, comparisons)
    
    Usage:
        from RL_Refactored.evaluation import EvaluationDashboard
        
        eval_dashboard = EvaluationDashboard()
        eval_dashboard.display()
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the evaluation dashboard.
        
        Args:
            config_path: Path to configuration file
        """
        if not WIDGETS_AVAILABLE:
            print("Warning: ipywidgets not available. Dashboard will have limited functionality.")
        
        self.config_path = config_path
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State
        self.agent = None
        self.environment = None
        self.evaluator = None
        self.z0_options = None
        self.rl_config = None
        self.rom_model = None
        
        # Results storage
        self.evaluation_results: Optional[EvaluationResults] = None
        self.comparison_results: Optional[ComparisonResults] = None
        
        # Checkpoint info
        self.checkpoints = []
        self.selected_checkpoint = None
        
        # UI Components
        self.dashboard = None
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup the dashboard UI."""
        if not WIDGETS_AVAILABLE:
            return
        
        # Create tabs
        self.setup_tab = widgets.VBox()
        self.evaluation_tab = widgets.VBox()
        self.results_tab = widgets.VBox()
        self.comparison_tab = widgets.VBox()
        self.details_tab = widgets.VBox()
        
        # Main tabs container
        self.tabs = widgets.Tab()
        self.tabs.children = [
            self.setup_tab,
            self.evaluation_tab,
            self.results_tab,
            self.comparison_tab,
            self.details_tab
        ]
        self.tabs.set_title(0, '1. Setup')
        self.tabs.set_title(1, '2. Evaluation')
        self.tabs.set_title(2, '3. Results')
        self.tabs.set_title(3, '4. Comparison')
        self.tabs.set_title(4, '5. Episode Details')
        
        # Populate tabs
        self._populate_setup_tab()
        self._populate_evaluation_tab()
        self._populate_results_tab()
        self._populate_comparison_tab()
        self._populate_details_tab()
        
        # Header
        header = widgets.HTML("""
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <h2 style="margin: 0;">RL Policy Evaluation Dashboard</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Evaluate trained policies and compare against baselines</p>
        </div>
        """)
        
        # Assemble dashboard
        self.dashboard = widgets.VBox([header, self.tabs])
    
    def _populate_setup_tab(self):
        """Populate the setup tab."""
        # Status display
        self.status_output = widgets.Output()
        
        # Configuration source selector
        self.config_source = widgets.RadioButtons(
            options=[
                ('Use existing RL Dashboard configuration', 'dashboard'),
                ('Load fresh configuration', 'fresh')
            ],
            value='dashboard',
            description='Config Source:',
            style={'description_width': '120px'}
        )
        
        # Checkpoint directory
        self.checkpoint_dir = widgets.Text(
            value='checkpoints',
            description='Checkpoint Dir:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='50%')
        )
        
        # Scan checkpoints button
        self.scan_button = widgets.Button(
            description='Scan Checkpoints',
            button_style='info',
            icon='search'
        )
        self.scan_button.on_click(self._on_scan_checkpoints)
        
        # Checkpoint selector
        self.checkpoint_dropdown = widgets.Dropdown(
            options=[],
            description='Checkpoint:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='70%')
        )
        
        # Checkpoint info display
        self.checkpoint_info = widgets.HTML()
        
        # Load configuration button
        self.load_config_button = widgets.Button(
            description='Load Configuration',
            button_style='primary',
            icon='cog'
        )
        self.load_config_button.on_click(self._on_load_config)
        
        # Load checkpoint button
        self.load_checkpoint_button = widgets.Button(
            description='Load Checkpoint',
            button_style='success',
            icon='download',
            disabled=True
        )
        self.load_checkpoint_button.on_click(self._on_load_checkpoint)
        
        self.setup_tab.children = [
            widgets.HTML("<h3>Step 1: Load Configuration</h3>"),
            widgets.HTML("<p><i>First, load the RL configuration (uses same settings as RL training dashboard)</i></p>"),
            self.config_source,
            self.load_config_button,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Step 2: Select Checkpoint</h3>"),
            widgets.HBox([self.checkpoint_dir, self.scan_button]),
            self.checkpoint_dropdown,
            self.checkpoint_info,
            self.load_checkpoint_button,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Status</h3>"),
            self.status_output
        ]
    
    def _populate_evaluation_tab(self):
        """Populate the evaluation tab."""
        # Number of cases
        self.num_cases_slider = widgets.IntSlider(
            value=20,
            min=5,
            max=100,
            step=5,
            description='Num Cases:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='50%')
        )
        
        # Deterministic toggle
        self.deterministic_toggle = widgets.Checkbox(
            value=True,
            description='Use Deterministic Actions (recommended for evaluation)',
            style={'description_width': 'initial'}
        )
        
        # Baseline selection
        self.baseline_checkboxes = {
            'Random': widgets.Checkbox(value=True, description='Random'),
            'Midpoint': widgets.Checkbox(value=True, description='Midpoint'),
            'NaiveMaxGas': widgets.Checkbox(value=True, description='Naive Max Gas'),
            'NaiveLowGas': widgets.Checkbox(value=True, description='Naive Low Gas')
        }
        
        # Run evaluation button
        self.run_eval_button = widgets.Button(
            description='Run Evaluation',
            button_style='success',
            icon='play',
            disabled=True
        )
        self.run_eval_button.on_click(self._on_run_evaluation)
        
        # Run comparison button
        self.run_compare_button = widgets.Button(
            description='Run Baseline Comparison',
            button_style='primary',
            icon='balance-scale',
            disabled=True
        )
        self.run_compare_button.on_click(self._on_run_comparison)
        
        # Progress output
        self.eval_progress = widgets.Output()
        
        self.evaluation_tab.children = [
            widgets.HTML("<h3>Evaluation Settings</h3>"),
            widgets.HTML("<p><i>Configure evaluation parameters</i></p>"),
            self.num_cases_slider,
            self.deterministic_toggle,
            widgets.HTML("<h4>Baselines to Compare:</h4>"),
            widgets.VBox(list(self.baseline_checkboxes.values())),
            widgets.HTML("<hr>"),
            widgets.HBox([self.run_eval_button, self.run_compare_button]),
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Progress</h3>"),
            self.eval_progress
        ]
    
    def _populate_results_tab(self):
        """Populate the results tab."""
        self.results_output = widgets.Output()
        self.results_plots = widgets.Output()
        
        # Refresh button
        self.refresh_results_button = widgets.Button(
            description='Refresh Results',
            button_style='info',
            icon='refresh'
        )
        self.refresh_results_button.on_click(self._on_refresh_results)
        
        # CSV export for best episode
        self.results_csv_dir = widgets.Text(
            value='csv_exports/',
            placeholder='Output directory',
            description='CSV Output Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        results_save_best_btn = widgets.Button(
            description='📄 Export Best Episode (Actions+Obs+Economics)',
            button_style='info', icon='file-text',
            layout=widgets.Layout(width='380px')
        )
        self.results_csv_status = widgets.Label(value='Run evaluation first')
        self.results_csv_output = widgets.Output()
        results_save_best_btn.on_click(self._on_save_best_episode_csv)
        
        results_csv_section = widgets.VBox([
            widgets.HTML("<h4>📄 Export Best Episode to CSV</h4>"),
            self.results_csv_dir,
            widgets.HBox([results_save_best_btn, self.results_csv_status]),
            self.results_csv_output
        ])
        
        self.results_tab.children = [
            widgets.HTML("<h3>Evaluation Results</h3>"),
            self.refresh_results_button,
            self.results_output,
            widgets.HTML("<h3>Visualizations</h3>"),
            self.results_plots,
            widgets.HTML("<hr>"),
            results_csv_section
        ]
    
    def _populate_comparison_tab(self):
        """Populate the comparison tab."""
        self.comparison_output = widgets.Output()
        self.comparison_plots = widgets.Output()
        
        # CSV export for comparison summary
        self.comp_csv_dir = widgets.Text(
            value='csv_exports/',
            placeholder='Output directory',
            description='CSV Output Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        comp_save_btn = widgets.Button(
            description='📄 Export All Policies Best Episodes',
            button_style='info', icon='file-text',
            layout=widgets.Layout(width='380px')
        )
        self.comp_csv_status = widgets.Label(value='Run comparison first')
        self.comp_csv_output = widgets.Output()
        comp_save_btn.on_click(self._on_save_comparison_csv)
        
        comp_csv_section = widgets.VBox([
            widgets.HTML("<h4>📄 Export Comparison to CSV</h4>"),
            widgets.HTML("<p><i>Exports the best episode from each policy (Trained RL + baselines) to CSV.</i></p>"),
            self.comp_csv_dir,
            widgets.HBox([comp_save_btn, self.comp_csv_status]),
            self.comp_csv_output
        ])
        
        self.comparison_tab.children = [
            widgets.HTML("<h3>Baseline Comparison Results</h3>"),
            self.comparison_output,
            widgets.HTML("<h3>Comparison Visualizations</h3>"),
            self.comparison_plots,
            widgets.HTML("<hr>"),
            comp_csv_section
        ]
    
    def _populate_details_tab(self):
        """Populate the episode details tab."""
        # Episode selector
        self.episode_selector = widgets.Dropdown(
            options=[],
            description='Episode:',
            style={'description_width': '80px'}
        )
        self.episode_selector.observe(self._on_episode_selected, names='value')
        
        self.episode_details_output = widgets.Output()
        self.episode_plots = widgets.Output()
        
        # CSV export controls
        self.eval_csv_dir = widgets.Text(
            value='csv_exports/',
            placeholder='Output directory',
            description='CSV Output Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.eval_save_actions_btn = widgets.Button(
            description='📄 Save Actions & Observations',
            button_style='info', icon='file-text',
            layout=widgets.Layout(width='250px')
        )
        self.eval_save_econ_btn = widgets.Button(
            description='💰 Save Economics / NCF',
            button_style='info', icon='file-text',
            layout=widgets.Layout(width='250px')
        )
        self.eval_csv_status = widgets.Label(value='Ready to export')
        self.eval_csv_output = widgets.Output()
        
        self.eval_save_actions_btn.on_click(self._on_save_actions_obs_csv)
        self.eval_save_econ_btn.on_click(self._on_save_economics_csv)
        
        csv_section = widgets.VBox([
            widgets.HTML("<h4>📄 Export Episode to CSV</h4>"),
            widgets.HTML("<p><i>Export the selected episode's data to CSV for visualization.</i></p>"),
            self.eval_csv_dir,
            widgets.HBox([self.eval_save_actions_btn, self.eval_save_econ_btn, self.eval_csv_status]),
            self.eval_csv_output
        ])
        
        self.details_tab.children = [
            widgets.HTML("<h3>Episode Details</h3>"),
            widgets.HTML("<p><i>Select an episode to view detailed results</i></p>"),
            self.episode_selector,
            self.episode_details_output,
            self.episode_plots,
            widgets.HTML("<hr>"),
            csv_section
        ]
    
    def _export_episode_to_csv(self, episode, output_dir, prefix):
        """Helper: export a single episode's actions+obs and economics to CSV. Returns list of saved filenames."""
        saved = []
        
        # Actions & Observations CSV
        actions = episode.actions or []
        observations = episode.observations or []
        num_steps = max(len(actions), len(observations))
        
        if num_steps > 0:
            action_keys = list(actions[0].keys()) if actions and actions[0] else []
            obs_keys = list(observations[0].keys()) if observations and observations[0] else []
            header = ['Year'] + action_keys + obs_keys
            
            filename = f"{prefix}_actions_observations.csv"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for t in range(num_steps):
                    row = [2025 + t]
                    act = actions[t] if t < len(actions) else {}
                    obs = observations[t] if t < len(observations) else {}
                    row += [act.get(k, '') for k in action_keys]
                    row += [obs.get(k, '') for k in obs_keys]
                    writer.writerow(row)
            saved.append(filename)
        
        # Economics CSV
        economic_breakdown = episode.economic_breakdown or []
        if economic_breakdown:
            gas_rev = [bd.get('gas_injection_revenue', 0) for bd in economic_breakdown]
            gas_cost = [bd.get('gas_injection_cost', 0) for bd in economic_breakdown]
            water_pen = [bd.get('water_production_penalty', 0) for bd in economic_breakdown]
            gas_pen = [bd.get('gas_production_penalty', 0) for bd in economic_breakdown]
            ncf = [bd.get('net_step_cashflow', 0) for bd in economic_breakdown]
            cum_npv = episode.cumulative_npv or []
            
            header = ['Year', 'Gas_Injection_Revenue', 'Gas_Injection_Cost',
                      'Water_Production_Penalty', 'Gas_Production_Penalty',
                      'Net_Step_Cashflow', 'Cumulative_NPV']
            
            filename = f"{prefix}_economics.csv"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for t in range(len(economic_breakdown)):
                    row = [2025 + t, gas_rev[t], gas_cost[t],
                           water_pen[t], gas_pen[t], ncf[t],
                           cum_npv[t] if t < len(cum_npv) else '']
                    writer.writerow(row)
            saved.append(filename)
        
        return saved
    
    def _on_save_best_episode_csv(self, button):
        """Export best episode from evaluation results to CSV."""
        self.results_csv_status.value = '📄 Saving...'
        with self.results_csv_output:
            clear_output(wait=True)
            try:
                if self.evaluation_results is None or not self.evaluation_results.all_episodes:
                    print("❌ No evaluation results. Run evaluation first.")
                    self.results_csv_status.value = '❌ No results'
                    return
                
                output_dir = self.results_csv_dir.value.strip()
                os.makedirs(output_dir, exist_ok=True)
                
                best = self.evaluation_results.get_best_episode()
                saved = self._export_episode_to_csv(best, output_dir,
                                                     f"eval_best_case_{best.z0_case_idx}")
                
                for f in saved:
                    size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
                    print(f"  ✅ {f} ({size_kb:.1f} KB)")
                
                self.results_csv_status.value = f'✅ Saved {len(saved)} files (best case #{best.z0_case_idx})'
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback; traceback.print_exc()
                self.results_csv_status.value = f'❌ Error: {e}'
    
    def _on_save_comparison_csv(self, button):
        """Export best episode from each policy (trained + baselines) to CSV."""
        self.comp_csv_status.value = '📄 Saving...'
        with self.comp_csv_output:
            clear_output(wait=True)
            try:
                if self.comparison_results is None:
                    print("❌ No comparison results. Run baseline comparison first.")
                    self.comp_csv_status.value = '❌ No results'
                    return
                
                output_dir = self.comp_csv_dir.value.strip()
                os.makedirs(output_dir, exist_ok=True)
                
                total_saved = []
                
                # Trained policy best episode
                trained_best = self.comparison_results.trained_policy.get_best_episode()
                if trained_best:
                    saved = self._export_episode_to_csv(
                        trained_best, output_dir,
                        f"trained_rl_best_case_{trained_best.z0_case_idx}")
                    total_saved.extend(saved)
                    print(f"📄 Trained RL (best case #{trained_best.z0_case_idx}, NPV: {trained_best.total_npv:.4f}):")
                    for f in saved:
                        size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
                        print(f"     ✅ {f} ({size_kb:.1f} KB)")
                
                # Baseline best episodes
                for baseline_name, baseline_results in self.comparison_results.baselines.items():
                    best = baseline_results.get_best_episode()
                    if best:
                        safe_name = baseline_name.replace(' ', '_').lower()
                        saved = self._export_episode_to_csv(
                            best, output_dir,
                            f"baseline_{safe_name}_best_case_{best.z0_case_idx}")
                        total_saved.extend(saved)
                        print(f"📄 {baseline_name} (best case #{best.z0_case_idx}, NPV: {best.total_npv:.4f}):")
                        for f in saved:
                            size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
                            print(f"     ✅ {f} ({size_kb:.1f} KB)")
                
                print(f"\n✅ Exported {len(total_saved)} files total to {output_dir}")
                self.comp_csv_status.value = f'✅ Saved {len(total_saved)} files'
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback; traceback.print_exc()
                self.comp_csv_status.value = f'❌ Error: {e}'
    
    def _get_selected_episode(self):
        """Get the currently selected EpisodeResult."""
        if self.evaluation_results is None:
            return None
        case_idx = self.episode_selector.value
        for ep in self.evaluation_results.all_episodes:
            if ep.z0_case_idx == case_idx:
                return ep
        return None
    
    def _on_save_actions_obs_csv(self, button):
        """Export selected episode's actions and observations to CSV."""
        self.eval_csv_status.value = '📄 Saving...'
        with self.eval_csv_output:
            clear_output(wait=True)
            try:
                episode = self._get_selected_episode()
                if episode is None:
                    print("❌ No episode selected or no evaluation results.")
                    self.eval_csv_status.value = '❌ No data'
                    return
                
                output_dir = self.eval_csv_dir.value.strip()
                os.makedirs(output_dir, exist_ok=True)
                
                actions = episode.actions or []
                observations = episode.observations or []
                num_steps = max(len(actions), len(observations))
                
                if num_steps == 0:
                    print("❌ No actions/observations data.")
                    self.eval_csv_status.value = '❌ No data'
                    return
                
                action_keys = list(actions[0].keys()) if actions and actions[0] else []
                obs_keys = list(observations[0].keys()) if observations and observations[0] else []
                
                header = ['Year'] + action_keys + obs_keys
                
                filename = f"eval_case_{episode.z0_case_idx}_actions_observations.csv"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for t in range(num_steps):
                        year = 2025 + t
                        row = [year]
                        act = actions[t] if t < len(actions) else {}
                        obs = observations[t] if t < len(observations) else {}
                        row += [act.get(k, '') for k in action_keys]
                        row += [obs.get(k, '') for k in obs_keys]
                        writer.writerow(row)
                
                size_kb = os.path.getsize(filepath) / 1024
                print(f"✅ Saved {filename} ({num_steps} rows, {size_kb:.1f} KB)")
                self.eval_csv_status.value = f'✅ Saved {filename}'
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback; traceback.print_exc()
                self.eval_csv_status.value = f'❌ Error: {e}'
    
    def _on_save_economics_csv(self, button):
        """Export selected episode's economics / NCF to CSV."""
        self.eval_csv_status.value = '💰 Saving...'
        with self.eval_csv_output:
            clear_output(wait=True)
            try:
                episode = self._get_selected_episode()
                if episode is None:
                    print("❌ No episode selected or no evaluation results.")
                    self.eval_csv_status.value = '❌ No data'
                    return
                
                output_dir = self.eval_csv_dir.value.strip()
                os.makedirs(output_dir, exist_ok=True)
                
                economic_breakdown = episode.economic_breakdown or []
                if not economic_breakdown:
                    print("❌ No economic breakdown data for this episode.")
                    self.eval_csv_status.value = '❌ No economics data'
                    return
                
                # Per-step components
                gas_rev = [bd.get('gas_injection_revenue', 0) for bd in economic_breakdown]
                gas_cost = [bd.get('gas_injection_cost', 0) for bd in economic_breakdown]
                water_pen = [bd.get('water_production_penalty', 0) for bd in economic_breakdown]
                gas_pen = [bd.get('gas_production_penalty', 0) for bd in economic_breakdown]
                ncf = [bd.get('net_step_cashflow', 0) for bd in economic_breakdown]
                cum_npv = episode.cumulative_npv or []
                
                header = ['Year', 'Gas_Injection_Revenue', 'Gas_Injection_Cost',
                          'Water_Production_Penalty', 'Gas_Production_Penalty',
                          'Net_Step_Cashflow', 'Cumulative_NPV']
                
                filename = f"eval_case_{episode.z0_case_idx}_economics.csv"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for t in range(len(economic_breakdown)):
                        year = 2025 + t
                        row = [year,
                               gas_rev[t], gas_cost[t],
                               water_pen[t], gas_pen[t],
                               ncf[t],
                               cum_npv[t] if t < len(cum_npv) else '']
                        writer.writerow(row)
                
                size_kb = os.path.getsize(filepath) / 1024
                print(f"✅ Saved {filename} ({len(economic_breakdown)} rows, {size_kb:.1f} KB)")
                self.eval_csv_status.value = f'✅ Saved {filename}'
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback; traceback.print_exc()
                self.eval_csv_status.value = f'❌ Error: {e}'
    
    def _on_scan_checkpoints(self, button):
        """Handle scan checkpoints button click."""
        with self.status_output:
            clear_output(wait=True)
            print("Scanning for checkpoints...")
            
            ckpt_dir = self.checkpoint_dir.value
            # Include architecture info for better checkpoint selection
            self.checkpoints = PolicyEvaluator.discover_checkpoints(ckpt_dir, include_architecture=True)
            
            if not self.checkpoints:
                print(f"No checkpoints found in '{ckpt_dir}'")
                self.checkpoint_dropdown.options = []
                return
            
            # Update dropdown with architecture info
            options = []
            for ckpt in self.checkpoints:
                arch = ckpt.get('architecture', {})
                if arch:
                    arch_str = f"h={arch.get('hidden_dim', '?')}, {arch.get('policy_type', '?')}"
                    label = f"{ckpt['name']} (ep {ckpt['episode']}, {ckpt['type']}) [{arch_str}]"
                else:
                    label = f"{ckpt['name']} (ep {ckpt['episode']}, {ckpt['type']}) [legacy]"
                options.append((label, ckpt['path']))
            
            self.checkpoint_dropdown.options = options
            print(f"Found {len(self.checkpoints)} checkpoint(s)")
            
            # Show current config requirements
            if hasattr(self, 'config') and self.config is not None:
                try:
                    current_hidden = self.config.rl_model['networks']['hidden_dim']
                    current_policy = self.config.rl_model['networks']['policy']['type']
                    print(f"\nCurrent config requires: hidden_dim={current_hidden}, policy_type={current_policy}")
                    print("Select a checkpoint with matching architecture.\n")
                except:
                    pass
            
            # List checkpoints with architecture info
            print("\nCheckpoint Architecture Summary:")
            print("-" * 60)
            for ckpt in self.checkpoints:
                arch = ckpt.get('architecture', {})
                if arch:
                    print(f"  {ckpt['name']}: hidden_dim={arch.get('hidden_dim', '?')}, "
                          f"policy={arch.get('policy_type', '?')}")
                else:
                    print(f"  {ckpt['name']}: [legacy format - architecture unknown]")
            print("-" * 60)
            
            # Select best checkpoint by default
            best_ckpts = [c for c in self.checkpoints if c['type'] == 'best']
            if best_ckpts:
                self.checkpoint_dropdown.value = best_ckpts[0]['path']
    
    def _on_load_config(self, button):
        """Handle load configuration button click."""
        with self.status_output:
            clear_output(wait=True)
            print("Loading configuration...")
            
            try:
                if self.config_source.value == 'dashboard':
                    # Try to get from RL dashboard globals
                    self._load_from_dashboard()
                else:
                    self._load_fresh_config()
                
                if self.config is not None:
                    print("Configuration loaded successfully!")
                    self.load_checkpoint_button.disabled = False
                else:
                    print("Failed to load configuration")
                    
            except Exception as e:
                print(f"Error loading configuration: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_from_dashboard(self):
        """Load configuration from RL dashboard globals."""
        import builtins
        
        # Try to get pre-generated Z0 options
        if hasattr(builtins, 'rl_generated_z0_options') and builtins.rl_generated_z0_options is not None:
            self.z0_options = builtins.rl_generated_z0_options
            print(f"  Z0 options loaded: {self.z0_options.shape}")
        else:
            # Fallback to function-based retrieval
            try:
                from ..configuration.dashboard import get_pre_generated_z0
                z0_result = get_pre_generated_z0()
                if z0_result[0] is not None:
                    self.z0_options, z0_metadata = z0_result
                    print(f"  Z0 options loaded: {self.z0_options.shape}")
            except Exception as e:
                print(f"  Warning: Could not load Z0 options: {e}")
        
        # Try to get ROM model from globals
        if hasattr(builtins, 'rl_loaded_rom') and builtins.rl_loaded_rom is not None:
            self.rom_model = builtins.rl_loaded_rom
            print(f"  ROM model loaded from RL dashboard")
        
        # Try to get RL config from globals
        if hasattr(builtins, 'rl_dashboard_config') and builtins.rl_dashboard_config is not None:
            self.rl_config = builtins.rl_dashboard_config
            print(f"  RL config loaded from dashboard")
        
        # Load main config
        import sys
        from pathlib import Path
        config_path = Path(__file__).parent.parent / 'config.yaml'
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from ROM_Refactored.utilities.config_loader import Config
        self.config = Config(str(config_path))
        print(f"  Config loaded from: {config_path}")
    
    def _load_fresh_config(self):
        """Load fresh configuration from file."""
        from pathlib import Path
        import sys
        
        config_path = Path(__file__).parent.parent / 'config.yaml'
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from ROM_Refactored.utilities.config_loader import Config
        self.config = Config(str(config_path))
        print(f"  Config loaded from: {config_path}")
    
    def _on_load_checkpoint(self, button):
        """Handle load checkpoint button click."""
        with self.status_output:
            clear_output(wait=True)
            
            if not self.checkpoint_dropdown.value:
                print("Please select a checkpoint first")
                return
            
            checkpoint_path = self.checkpoint_dropdown.value
            print(f"Loading checkpoint: {checkpoint_path}")
            
            try:
                # Check if ROM model and Z0 options are available
                if self.rom_model is None:
                    print("\n" + "="*60)
                    print("⚠️  ROM MODEL NOT LOADED")
                    print("="*60)
                    print("\nPlease run STEP 1 (RL Configuration Dashboard) first:")
                    print("  1. Execute the 'STEP 1: RL Configuration Dashboard' cell")
                    print("  2. Configure settings and click 'Apply Configuration'")
                    print("  3. Return here and try loading the checkpoint again")
                    print("="*60)
                    return
                
                if self.z0_options is None:
                    print("\n" + "="*60)
                    print("⚠️  Z0 OPTIONS NOT LOADED")
                    print("="*60)
                    print("\nPlease run STEP 1 (RL Configuration Dashboard) first.")
                    print("="*60)
                    return
                
                # Create agent if needed
                if self.agent is None:
                    self._create_agent()
                
                # Create environment if needed
                if self.environment is None:
                    self._create_environment()
                
                # Create evaluator
                self.evaluator = PolicyEvaluator(
                    agent=self.agent,
                    environment=self.environment,
                    config=self.config,
                    rl_config=self.rl_config
                )
                
                # Load checkpoint
                success = self.evaluator.load_checkpoint(checkpoint_path, evaluate=True)
                
                if success:
                    print("✅ Checkpoint loaded successfully!")
                    self.run_eval_button.disabled = False
                    self.run_compare_button.disabled = False
                    
                    # Update checkpoint info
                    self._update_checkpoint_info()
                else:
                    print("❌ Failed to load checkpoint")
                    
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_agent(self):
        """Create SAC agent."""
        from ..agent.sac_agent import SAC
        
        # Get latent dim from z0_options or config
        if self.z0_options is not None:
            latent_dim = self.z0_options.shape[-1]
        else:
            # Try 'model' section first (RL config), then 'rom_model' as fallback
            if hasattr(self.config, 'model') and isinstance(self.config.model, dict):
                latent_dim = self.config.model.get('latent_dim', 128)
            elif hasattr(self.config, 'rom_model') and isinstance(self.config.rom_model, dict):
                latent_dim = self.config.rom_model.get('latent_dim', 128)
            else:
                latent_dim = 128
        
        num_producers = self.config.rl_model['reservoir']['num_producers']
        num_injectors = self.config.rl_model['reservoir']['num_injectors']
        u_dim = num_producers + num_injectors
        
        print(f"  Creating SAC agent (state_dim={latent_dim}, action_dim={u_dim})")
        self.agent = SAC(num_inputs=latent_dim, u_dim=u_dim, config=self.config)
    
    def _create_environment(self):
        """Create reservoir environment."""
        from ..environment.reservoir_env import ReservoirEnvironment
        
        # Need ROM model for environment
        if self.rom_model is None:
            self._load_rom_model()
        
        # Check if ROM model was loaded
        if self.rom_model is None:
            raise RuntimeError(
                "ROM model not available. Please run STEP 1 (RL Configuration Dashboard) first "
                "and click 'Apply Configuration' to load the ROM model."
            )
        
        # Get initial Z0
        if self.z0_options is not None:
            z0 = self.z0_options[0:1]
        else:
            # Get latent dim from config
            if hasattr(self.config, 'model') and isinstance(self.config.model, dict):
                latent_dim = self.config.model.get('latent_dim', 128)
            else:
                latent_dim = 128
            z0 = torch.zeros(1, latent_dim, device=self.device)
        
        print(f"  Creating environment (z0 shape={z0.shape})")
        # ReservoirEnvironment signature: (state0, config, my_rom)
        self.environment = ReservoirEnvironment(
            state0=z0,
            config=self.config,
            my_rom=self.rom_model
        )
        
        # Set rl_config if available (used for action range conversion)
        if self.rl_config and hasattr(self.environment, 'set_rl_config'):
            self.environment.set_rl_config(self.rl_config)
    
    def _load_rom_model(self):
        """Load ROM model from RL dashboard's global storage or fresh."""
        import builtins
        
        # First, try to get ROM from RL dashboard's global storage
        if hasattr(builtins, 'rl_loaded_rom') and builtins.rl_loaded_rom is not None:
            self.rom_model = builtins.rl_loaded_rom
            print("  ROM model loaded from RL dashboard (shared)")
            return
        
        # If not available, we need to inform the user to run RL config first
        print("  Warning: ROM model not found in global storage.")
        print("  Please run STEP 1 (RL Configuration Dashboard) first and click 'Apply Configuration'.")
        print("  This will load the ROM model needed for evaluation.")
        self.rom_model = None
    
    def _update_checkpoint_info(self):
        """Update checkpoint info display."""
        if self.evaluator and self.evaluator.checkpoint_loaded:
            self.checkpoint_info.value = f"""
            <div style="background: #e8f5e9; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <p><strong>Loaded:</strong> {self.evaluator.checkpoint_path}</p>
                <p><strong>Mode:</strong> Evaluation (Deterministic)</p>
            </div>
            """
    
    def _on_run_evaluation(self, button):
        """Handle run evaluation button click."""
        with self.eval_progress:
            clear_output(wait=True)
            
            if self.evaluator is None or not self.evaluator.checkpoint_loaded:
                print("Please load a checkpoint first")
                return
            
            if self.z0_options is None:
                print("No Z0 options available. Please load configuration first.")
                return
            
            print("Starting evaluation...")
            
            try:
                self.evaluation_results = self.evaluator.evaluate_multiple_cases(
                    z0_options=self.z0_options,
                    num_cases=self.num_cases_slider.value,
                    deterministic=self.deterministic_toggle.value,
                    verbose=True
                )
                
                print("\nEvaluation complete!")
                
                # Update results tab
                self._update_results_display()
                
                # Update episode selector
                self._update_episode_selector()
                
                # Switch to results tab
                self.tabs.selected_index = 2
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_run_comparison(self, button):
        """Handle run comparison button click."""
        with self.eval_progress:
            clear_output(wait=True)
            
            if self.evaluator is None or not self.evaluator.checkpoint_loaded:
                print("Please load a checkpoint first")
                return
            
            if self.z0_options is None:
                print("No Z0 options available. Please load configuration first.")
                return
            
            # Get selected baselines
            selected_baselines = [
                name for name, checkbox in self.baseline_checkboxes.items()
                if checkbox.value
            ]
            
            if not selected_baselines:
                print("Please select at least one baseline for comparison")
                return
            
            print("Starting baseline comparison...")
            
            try:
                self.comparison_results = self.evaluator.run_baseline_comparison(
                    z0_options=self.z0_options,
                    num_cases=self.num_cases_slider.value,
                    deterministic=self.deterministic_toggle.value,
                    baselines=selected_baselines,
                    verbose=True
                )
                
                print("\nComparison complete!")
                
                # Update comparison tab
                self._update_comparison_display()
                
                # Store trained policy results
                self.evaluation_results = self.comparison_results.trained_policy
                self._update_episode_selector()
                
                # Switch to comparison tab
                self.tabs.selected_index = 3
                
            except Exception as e:
                print(f"Comparison error: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_refresh_results(self, button):
        """Handle refresh results button click."""
        self._update_results_display()
    
    def _update_results_display(self):
        """Update the results display."""
        with self.results_output:
            clear_output(wait=True)
            
            if self.evaluation_results is None:
                print("No evaluation results available. Run evaluation first.")
                return
            
            print(self.evaluation_results)
        
        with self.results_plots:
            clear_output(wait=True)
            
            if not MATPLOTLIB_AVAILABLE:
                print("Matplotlib not available for plotting")
                return
            
            self._plot_evaluation_results()
    
    def _update_comparison_display(self):
        """Update the comparison display."""
        with self.comparison_output:
            clear_output(wait=True)
            
            if self.comparison_results is None:
                print("No comparison results available. Run comparison first.")
                return
            
            print(self.comparison_results)
        
        with self.comparison_plots:
            clear_output(wait=True)
            
            if not MATPLOTLIB_AVAILABLE:
                print("Matplotlib not available for plotting")
                return
            
            self._plot_comparison_results()
    
    def _plot_evaluation_results(self):
        """Plot evaluation results."""
        if self.evaluation_results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. NPV distribution (histogram)
        ax1 = axes[0, 0]
        npvs = [ep.total_npv for ep in self.evaluation_results.all_episodes]
        ax1.hist(npvs, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(self.evaluation_results.mean_npv, color='red', linestyle='--', 
                   label=f'Mean: {self.evaluation_results.mean_npv:.4f}')
        ax1.axvline(self.evaluation_results.median_npv, color='green', linestyle=':', 
                   label=f'Median: {self.evaluation_results.median_npv:.4f}')
        ax1.set_xlabel('NPV')
        ax1.set_ylabel('Frequency')
        ax1.set_title('NPV Distribution Across Z0 Cases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. NPV box plot
        ax2 = axes[0, 1]
        box_data = ax2.boxplot([npvs], labels=['Trained Policy'])
        ax2.set_ylabel('NPV')
        ax2.set_title('NPV Statistics')
        ax2.grid(True, alpha=0.3)
        
        # Add mean marker
        ax2.scatter([1], [self.evaluation_results.mean_npv], color='red', marker='D', s=100, zorder=5, label='Mean')
        ax2.legend()
        
        # 3. Cumulative NPV trajectories (sample episodes)
        ax3 = axes[1, 0]
        num_to_plot = min(10, len(self.evaluation_results.all_episodes))
        
        # Sort by NPV and get diverse sample
        sorted_eps = sorted(self.evaluation_results.all_episodes, key=lambda x: x.total_npv)
        sample_indices = np.linspace(0, len(sorted_eps)-1, num_to_plot, dtype=int)
        
        colors = plt.cm.viridis(np.linspace(0, 1, num_to_plot))
        for i, idx in enumerate(sample_indices):
            ep = sorted_eps[idx]
            if ep.cumulative_npv:
                ax3.plot(ep.cumulative_npv, color=colors[i], alpha=0.7, 
                        label=f'Case {ep.z0_case_idx}' if i < 3 else None)
        
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Cumulative NPV')
        ax3.set_title('Cumulative NPV Trajectories (Sample Episodes)')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. NPV vs Case Index
        ax4 = axes[1, 1]
        case_indices = [ep.z0_case_idx for ep in self.evaluation_results.all_episodes]
        ax4.scatter(case_indices, npvs, alpha=0.6, c='steelblue')
        ax4.axhline(self.evaluation_results.mean_npv, color='red', linestyle='--', label='Mean')
        ax4.set_xlabel('Z0 Case Index')
        ax4.set_ylabel('NPV')
        ax4.set_title('NPV by Initial State (Z0 Case)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        display(fig)
        plt.close(fig)  # Close to prevent memory leak
    
    def _plot_comparison_results(self):
        """Plot comparison results."""
        if self.comparison_results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gather data
        policy_names = ['Trained RL']
        means = [self.comparison_results.trained_policy.mean_npv]
        stds = [self.comparison_results.trained_policy.std_npv]
        all_npvs = {'Trained RL': [ep.total_npv for ep in self.comparison_results.trained_policy.all_episodes]}
        
        for name, results in self.comparison_results.baselines.items():
            policy_names.append(name)
            means.append(results.mean_npv)
            stds.append(results.std_npv)
            all_npvs[name] = [ep.total_npv for ep in results.all_episodes]
        
        colors = ['#2ecc71'] + ['#e74c3c', '#3498db', '#f39c12', '#9b59b6'][:len(policy_names)-1]
        
        # 1. Bar chart with error bars
        ax1 = axes[0, 0]
        x = np.arange(len(policy_names))
        bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors[:len(policy_names)], 
                      edgecolor='black', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(policy_names, rotation=45, ha='right')
        ax1.set_ylabel('Mean NPV')
        ax1.set_title('Policy Comparison (Mean NPV ± Std)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Box plot comparison
        ax2 = axes[0, 1]
        box_data = [all_npvs[name] for name in policy_names]
        bp = ax2.boxplot(box_data, labels=policy_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(policy_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xticklabels(policy_names, rotation=45, ha='right')
        ax2.set_ylabel('NPV')
        ax2.set_title('NPV Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. Improvement ratios
        ax3 = axes[1, 0]
        baseline_names = list(self.comparison_results.improvement_ratios.keys())
        improvements = [self.comparison_results.improvement_ratios[name] for name in baseline_names]
        colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        
        ax3.barh(baseline_names, improvements, color=colors_imp, edgecolor='black', alpha=0.8)
        ax3.axvline(0, color='black', linewidth=1)
        ax3.set_xlabel('Improvement (%)')
        ax3.set_title('Trained Policy Improvement Over Baselines')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (name, imp) in enumerate(zip(baseline_names, improvements)):
            ax3.text(imp + (2 if imp > 0 else -2), i, f'{imp:+.1f}%', 
                    va='center', ha='left' if imp > 0 else 'right')
        
        # 4. Average cumulative NPV trajectories
        ax4 = axes[1, 1]
        
        for i, (name, npv_list) in enumerate(all_npvs.items()):
            # Get cumulative trajectories
            results = self.comparison_results.trained_policy if name == 'Trained RL' else self.comparison_results.baselines.get(name)
            if results:
                cum_trajectories = [ep.cumulative_npv for ep in results.all_episodes if ep.cumulative_npv]
                if cum_trajectories:
                    # Compute mean trajectory
                    max_len = max(len(traj) for traj in cum_trajectories)
                    padded = [traj + [traj[-1]]*(max_len - len(traj)) for traj in cum_trajectories]
                    mean_traj = np.mean(padded, axis=0)
                    ax4.plot(mean_traj, label=name, color=colors[i], linewidth=2)
        
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Cumulative NPV')
        ax4.set_title('Average Cumulative NPV Trajectories')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        display(fig)
        plt.close(fig)  # Close to prevent memory leak
    
    def _update_episode_selector(self):
        """Update episode selector dropdown."""
        if self.evaluation_results is None:
            self.episode_selector.options = []
            return
        
        options = []
        for ep in self.evaluation_results.all_episodes:
            label = f"Case {ep.z0_case_idx} (NPV: {ep.total_npv:.4f})"
            options.append((label, ep.z0_case_idx))
        
        self.episode_selector.options = options
    
    def _on_episode_selected(self, change):
        """Handle episode selection change."""
        if self.evaluation_results is None:
            return
        
        case_idx = change['new']
        
        # Find the episode
        episode = None
        for ep in self.evaluation_results.all_episodes:
            if ep.z0_case_idx == case_idx:
                episode = ep
                break
        
        if episode is None:
            return
        
        with self.episode_details_output:
            clear_output(wait=True)
            print(f"Episode Details: Z0 Case {case_idx}")
            print(f"{'='*50}")
            print(f"Total NPV: {episode.total_npv:.4f}")
            print(f"Steps: {len(episode.step_rewards)}")
            
            summary = episode.get_final_economic_summary()
            if summary:
                print(f"\nEconomic Summary:")
                for key, value in summary.items():
                    print(f"  {key}: ${value:,.2f}")
        
        with self.episode_plots:
            clear_output(wait=True)
            
            if not MATPLOTLIB_AVAILABLE:
                return
            
            self._plot_episode_details(episode)
    
    def _plot_episode_details(self, episode):
        """Plot detailed episode data."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Step rewards
        ax1 = axes[0, 0]
        ax1.bar(range(len(episode.step_rewards)), episode.step_rewards, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Step Reward')
        ax1.set_title('Step-wise Rewards')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative NPV
        ax2 = axes[0, 1]
        ax2.plot(episode.cumulative_npv, color='green', linewidth=2)
        ax2.fill_between(range(len(episode.cumulative_npv)), episode.cumulative_npv, alpha=0.3, color='green')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cumulative NPV')
        ax2.set_title('Cumulative NPV Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Actions over time
        ax3 = axes[1, 0]
        if episode.actions:
            # Extract BHP and Gas actions
            bhp_actions = []
            gas_actions = []
            for action in episode.actions:
                bhp = [v for k, v in action.items() if 'BHP' in k]
                gas = [v for k, v in action.items() if 'Gas' in k]
                bhp_actions.append(np.mean(bhp) if bhp else 0)
                gas_actions.append(np.mean(gas) if gas else 0)
            
            ax3_twin = ax3.twinx()
            l1 = ax3.plot(bhp_actions, label='Avg BHP (psi)', color='blue', linewidth=2)
            l2 = ax3_twin.plot(gas_actions, label='Avg Gas Inj (ft³/day)', color='orange', linewidth=2)
            ax3.set_xlabel('Timestep')
            ax3.set_ylabel('BHP (psi)', color='blue')
            ax3_twin.set_ylabel('Gas Injection (ft³/day)', color='orange')
            ax3.set_title('Control Actions Over Time')
            
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right')
            ax3.grid(True, alpha=0.3)
        
        # 4. Economic breakdown pie chart
        ax4 = axes[1, 1]
        summary = episode.get_final_economic_summary()
        if summary:
            labels = ['Gas Revenue', 'Water Penalty', 'Gas Penalty']
            values = [
                abs(summary.get('total_gas_injection_revenue', 0)),
                abs(summary.get('total_water_production_penalty', 0)),
                abs(summary.get('total_gas_production_penalty', 0))
            ]
            colors = ['#2ecc71', '#e74c3c', '#f39c12']
            
            # Filter out zero values
            non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
            if non_zero:
                labels, values, colors = zip(*non_zero)
                ax4.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Economic Breakdown')
            else:
                ax4.text(0.5, 0.5, 'No economic data', ha='center', va='center')
                ax4.set_title('Economic Breakdown')
        
        plt.tight_layout()
        display(fig)
        plt.close(fig)  # Close to prevent memory leak
    
    def show(self):
        """Display the dashboard."""
        if self.dashboard is None:
            print("Dashboard not available. Please ensure ipywidgets is installed.")
            return
        
        display(self.dashboard)
    
    # Alias for backward compatibility
    def display(self):
        """Display the dashboard (alias for show())."""
        self.show()
        return self
    
    def _ipython_display_(self):
        """IPython display hook."""
        self.display()


def launch_evaluation_dashboard(config_path: str = 'config.yaml') -> EvaluationDashboard:
    """
    Launch the evaluation dashboard.
    
    Single-line launcher for use in Jupyter notebooks.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        EvaluationDashboard instance
    """
    dashboard = EvaluationDashboard(config_path)
    dashboard.display()
    return dashboard
