"""
RL Training Dashboard
Full implementation for RL-SAC training with interactive controls
"""
import sys
import numpy as np
import torch
import wandb
from pathlib import Path
from datetime import datetime

# Ensure ROM_Refactored is in path
rom_refactored_path = Path(__file__).parent.parent.parent / 'ROM_Refactored'
if str(rom_refactored_path) not in sys.path:
    sys.path.insert(0, str(rom_refactored_path))

# Import RL_Refactored modules
from RL_Refactored.agent import create_rl_agent, create_sac_agent
from RL_Refactored.agent.replay_memory import ReplayMemory
from RL_Refactored.environment import create_environment
from RL_Refactored.training.orchestrator import EnhancedTrainingOrchestrator
from RL_Refactored.utilities import Config
from ROM_Refactored.utilities.wandb_integration import create_wandb_logger

# Import configuration utilities
from RL_Refactored.configuration import (
    get_rl_config, has_rl_config, get_pre_loaded_rom, get_pre_generated_z0,
    get_pre_generated_spatial_states,
    get_action_scaling_params, create_rl_reward_function, update_config_with_dashboard
)

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


class RLTrainingDashboard:
    """
    Interactive dashboard for RL training
    
    Provides controls to start/stop training and monitor progress
    """
    
    def __init__(self, config_path='config.yaml'):
        # Resolve to RL_Refactored/config.yaml so it works regardless of CWD
        if not Path(config_path).is_absolute() and not Path(config_path).exists():
            rl_cfg = Path(__file__).parent.parent / config_path
            if rl_cfg.exists():
                config_path = str(rl_cfg)
        self.config_path = config_path
        self.config = None
        self.rl_config = None
        self.training_in_progress = False
        self.training_orchestrator = None
        self.agent = None
        self.environment = None
        self.memory = None
        self.wandb_logger = None
        
        # Training state
        self.episode_rewards = []
        self.avg_rewards = []
        self.best_reward = -np.inf
        self.global_step = 0
        self.total_numsteps = 0
        
        # Metrics for WandB
        self.rl_metrics = {
            'episode': 0,
            'reward/total': 0,
            'reward/avg': 0,
            'reward/min': 0,
            'reward/max': 0,
            'policy/loss': 0,
            'q_value/loss': 0,
            'alpha/value': 0,
            'alpha/loss': 0,
            'training/step': 0
        }
        
        # Initialize widgets
        self._create_widgets()
        
        # Load configuration
        self._load_configuration()
    
    def _create_widgets(self):
        """Create interactive widgets for training control"""
        if not WIDGETS_AVAILABLE:
            return
        
        # Status output
        self.status_output = widgets.Output()
        
        # Training controls
        self.start_button = widgets.Button(
            description='🚀 Start Training',
            button_style='success',
            layout=widgets.Layout(width='200px', margin='10px')
        )
        self.start_button.on_click(self._start_training)
        
        self.stop_button = widgets.Button(
            description='⏹️ Stop Training',
            button_style='danger',
            layout=widgets.Layout(width='200px', margin='10px'),
            disabled=True
        )
        self.stop_button.on_click(self._stop_training)
        
        # Training info display
        self.training_info = widgets.HTML(
            value="<p><b>Status:</b> Ready to start training</p>",
            layout=widgets.Layout(margin='10px')
        )
        
        # Progress bar
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='100%', margin='10px')
        )
        
        # === Evaluation-Based Checkpoint Selection Options ===
        self.eval_checkpoint_checkbox = widgets.Checkbox(
            value=False,
            description='Evaluation-Based Checkpoint Selection',
            indent=False,
            layout=widgets.Layout(width='auto', margin='5px 10px'),
            style={'description_width': 'initial'}
        )
        self.eval_checkpoint_checkbox.observe(self._on_eval_checkpoint_toggle, names='value')
        
        self.eval_interval_spinner = widgets.BoundedIntText(
            value=5,
            min=1,
            max=50,
            step=1,
            description='Eval every N episodes:',
            layout=widgets.Layout(width='250px', margin='2px 10px'),
            style={'description_width': 'initial'},
            disabled=True
        )
        
        self.eval_num_cases_spinner = widgets.BoundedIntText(
            value=10,
            min=1,
            max=100,
            step=1,
            description='Eval Z0 cases:',
            layout=widgets.Layout(width='250px', margin='2px 10px'),
            style={'description_width': 'initial'},
            disabled=True
        )
        
        self.eval_info_label = widgets.HTML(
            value=(
                "<p style='font-size:11px; color:#666; margin:2px 10px;'>"
                "When enabled, checkpoints are saved based on <b>deterministic</b> policy "
                "evaluation (no exploration noise), giving a true measure of policy quality.</p>"
            )
        )
        
        eval_options_box = widgets.VBox([
            widgets.HTML("<hr style='margin:5px 0'>"),
            widgets.HTML("<b style='margin-left:10px'>Checkpoint Strategy</b>"),
            self.eval_checkpoint_checkbox,
            widgets.HBox([self.eval_interval_spinner, self.eval_num_cases_spinner]),
            self.eval_info_label
        ])
        
        # Main widget
        self.main_widget = widgets.VBox([
            widgets.HTML("<h3>RL Training Dashboard</h3>"),
            widgets.HTML("<p>Configure and start RL training</p>"),
            widgets.HBox([self.start_button, self.stop_button]),
            self.training_info,
            self.progress_bar,
            eval_options_box,
            self.status_output
        ])
    
    def _on_eval_checkpoint_toggle(self, change):
        """Enable/disable evaluation checkpoint sub-options"""
        enabled = change['new']
        self.eval_interval_spinner.disabled = not enabled
        self.eval_num_cases_spinner.disabled = not enabled
    
    def _run_deterministic_evaluation(self, z0_options, num_cases=10):
        """
        Run a lightweight deterministic evaluation of the current policy.
        
        Uses the raw policy output (no ActionVariationManager noise) to measure
        the true policy quality for checkpoint selection.
        
        Args:
            z0_options: Tensor of available Z0 initial states (num_available, latent_dim)
            num_cases: Number of Z0 cases to evaluate
            
        Returns:
            mean_eval_npv: Mean NPV across evaluated cases
        """
        max_steps = self.config.config.get('rl_model', {}).get('training', {}).get('max_steps_per_episode', 100)
        total_available = z0_options.shape[0]
        num_cases = min(num_cases, total_available)
        
        # Randomly select cases for evaluation
        eval_indices = np.random.choice(total_available, size=num_cases, replace=False)
        
        eval_npvs = []
        
        # Temporarily set policy to eval mode (disables dropout, uses running BN stats)
        self.agent.policy.eval()
        
        for case_idx in eval_indices:
            z0 = z0_options[case_idx:case_idx + 1]
            state = self.environment.reset(z0_options=z0)
            episode_reward = 0.0
            
            for step in range(max_steps):
                with torch.no_grad():
                    # Use evaluate=True for deterministic action (pure policy, no noise)
                    action = self.agent.select_action(state, evaluate=True)
                
                next_state, reward, done = self.environment.step(action)
                episode_reward += reward.item()
                state = next_state
                
                if done:
                    break
            
            eval_npvs.append(episode_reward)
        
        # Restore policy to train mode
        self.agent.policy.train()
        
        mean_eval_npv = float(np.mean(eval_npvs))
        std_eval_npv = float(np.std(eval_npvs))
        
        return mean_eval_npv, std_eval_npv
    
    def _load_configuration(self):
        """Load configuration and check prerequisites"""
        try:
            # Load main config
            self.config = Config(self.config_path)
            
            # Check if RL config is available
            if not has_rl_config():
                print("⚠️ No RL configuration found!")
                print("   Please run the Configuration Dashboard first and apply configuration.")
                return False
            
            # Get RL config
            self.rl_config = get_rl_config()
            
            # Check if models are ready
            from RL_Refactored.configuration import are_models_ready
            if not are_models_ready():
                print("⚠️ Models not ready!")
                print("   Please run the Configuration Dashboard and apply configuration.")
                return False
            
            print("✅ Configuration loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False
    
    def _setup_training_components(self):
        """Setup all training components"""
        with self.status_output:
            clear_output(wait=True)
            print("🔧 Setting up training components...")
            
            try:
                # Update config with dashboard values
                update_config_with_dashboard(self.config, self.rl_config)
                
                # Safe access to rl_model section (may not exist in ROM-only configs)
                rl_model_cfg = self.config.config.get('rl_model', {})
                
                # Set WandB run name to include ROM model info
                rom_name = 'unknown'
                prediction_mode = self.config.config.get('rl_model', {}).get('environment', {}).get('prediction_mode', 'state_based')
                if hasattr(self, 'rl_config') and 'selected_rom' in (get_rl_config() or {}):
                    rl_cfg = get_rl_config()
                    rom_name = rl_cfg['selected_rom'].get('name', 'unknown')
                elif 'runtime' in self.config.config and 'wandb' in self.config.config['runtime']:
                    pass
                algo_type = rl_model_cfg.get('algorithm', {}).get('type', 'SAC')
                wandb_cfg = self.config.config.setdefault('runtime', {}).setdefault('wandb', {})
                wandb_cfg['project'] = 'RL-SAC'
                if not wandb_cfg.get('name'):
                    wandb_cfg['name'] = f"{algo_type}_{prediction_mode}_{rom_name}"
                wandb_cfg.setdefault('tags', [])
                if isinstance(wandb_cfg['tags'], list):
                    for tag in [f'RL-{algo_type}', prediction_mode, rom_name]:
                        if tag not in wandb_cfg['tags']:
                            wandb_cfg['tags'].append(tag)
                
                # Initialize WandB Logger
                print("   📊 Initializing WandB logger...")
                self.wandb_logger = create_wandb_logger(self.config)
                
                # Get pre-loaded ROM model
                print("   🧠 Loading ROM model...")
                my_rom = get_pre_loaded_rom()
                if my_rom is None:
                    raise ValueError("ROM model not available!")
                
                # Get pre-generated Z0 options
                print("   🏔️ Loading Z0 options...")
                z0_options, z0_metadata = get_pre_generated_z0()
                if z0_options is None:
                    raise ValueError("Z0 options not available!")
                
                print(f"      ✅ Loaded {z0_options.shape[0]} initial states")
                
                # Watch ROM model in WandB
                self.wandb_logger.watch_model(my_rom)
                
                # Create RL agent (algorithm selected via config)
                algo_type = rl_model_cfg.get('algorithm', {}).get('type', 'SAC')
                print(f"   Creating {algo_type} agent...")
                self.agent = create_rl_agent(self.config, self.rl_config, rom_model=my_rom)
                self._is_ppo = algo_type.upper() == 'PPO'

                import re as _re
                self._rom_model_tag = _re.sub(r'[|<>:"/\\?*\s]+', '_', rom_name).strip('_')
                self._algo_type = algo_type
                self._prediction_mode = prediction_mode
                
                # Create environment (pass spatial states for multimodal models)
                print("   🌍 Creating environment...")
                spatial_states = get_pre_generated_spatial_states()
                self.environment = create_environment(z0_options, self.config, my_rom, self.rl_config,
                                                      spatial_states=spatial_states)
                
                # Verify dashboard action mapping
                print("   🔍 Verifying action mapping...")
                self.environment.verify_dashboard_action_mapping()
                
                # Set global seeds for reproducibility
                import random as _random
                seeds = rl_model_cfg.get('training', {}).get('seeds', {'torch': 42, 'numpy': 42, 'replay_memory': 42})
                torch.manual_seed(seeds.get('torch', 42))
                np.random.seed(seeds.get('numpy', 42))
                _random.seed(seeds.get('torch', 42))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seeds.get('torch', 42))
                print(f"   🎲 Seeds set: torch={seeds.get('torch', 42)}, numpy={seeds.get('numpy', 42)}")

                # Create replay memory
                print("   💾 Creating replay memory...")
                replay_cfg = rl_model_cfg.get('replay_memory', {})
                batch_size = replay_cfg.get('batch_size', 256)
                capacity = replay_cfg.get('capacity', 1000000)
                seed = seeds.get('replay_memory', 42)
                self.memory = ReplayMemory(capacity, seed)
                
                # Create training orchestrator
                print("   🎯 Creating training orchestrator...")
                self.training_orchestrator = EnhancedTrainingOrchestrator(self.config, self.rl_config)
                self.training_orchestrator.set_environment(self.environment)
                
                # Get action scaling for display
                action_scaling = get_action_scaling_params(self.rl_config)
                
                print("✅ All components ready!")
                print(f"   📊 BHP range: [{action_scaling['bhp']['min']:.1f}, {action_scaling['bhp']['max']:.1f}] psi")
                print(f"   💨 Gas range: [{action_scaling['gas_injection']['min']:.0f}, {action_scaling['gas_injection']['max']:.0f}] ft³/day")
                
                return True
                
            except Exception as e:
                print(f"❌ Error setting up components: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def _start_training(self, button):
        """Start RL training"""
        if self.training_in_progress:
            print("⚠️ Training already in progress!")
            return
        
        # Setup components if not already done
        if self.agent is None:
            if not self._setup_training_components():
                return
        
        # Update UI
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.training_in_progress = True
        
        # Start training in background (or synchronously for now)
        self._run_training_loop()
    
    def _stop_training(self, button):
        """Stop RL training"""
        self.training_in_progress = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        print("⏹️ Training stopped by user")
    
    def _run_training_loop(self):
        """Run the main RL training loop"""
        with self.status_output:
            clear_output(wait=True)
            print("🚀 Starting RL training...")
            
            # Training parameters
            rl_model_cfg = self.config.config.get('rl_model', {})
            training_config = rl_model_cfg.get('training', {})
            max_episodes = training_config.get('max_episodes', 500)
            max_steps = training_config.get('max_steps_per_episode', 100)
            batch_size = rl_model_cfg.get('replay_memory', {}).get('batch_size', 256)
            updates_per_step = training_config.get('updates_per_step', 1)
            save_interval = 100
            exploration_steps = training_config.get('exploration_steps', 0)
            print_interval = training_config.get('print_interval', 10)
            
            # Get Z0 options
            z0_options, z0_metadata = get_pre_generated_z0()
            
            print(f"📊 Training configuration:")
            print(f"   Episodes: {max_episodes}")
            print(f"   Steps per episode: {max_steps}")
            print(f"   Batch size: {batch_size}")
            print(f"   Initial states: {z0_options.shape[0]}")
            
            # Read evaluation-based checkpoint settings from dashboard
            use_eval_checkpoint = self.eval_checkpoint_checkbox.value
            eval_interval = self.eval_interval_spinner.value
            eval_num_cases = self.eval_num_cases_spinner.value
            
            if use_eval_checkpoint:
                print(f"\n   ✅ Evaluation-Based Checkpoint Selection: ENABLED")
                print(f"      Eval every {eval_interval} episodes | {eval_num_cases} Z0 cases per eval")
                print(f"      Checkpoints saved based on deterministic policy NPV (no exploration noise)")
            else:
                print(f"\n   ℹ️  Checkpoint Selection: Training reward (default)")
            
            # Reset tracking variables
            self.episode_rewards = []
            self.avg_rewards = []
            self.best_reward = -np.inf
            self.best_eval_reward = -np.inf  # Track best evaluation reward separately
            self.global_step = 0
            self.total_numsteps = 0
            
            # Update progress bar
            self.progress_bar.max = max_episodes
            self.progress_bar.value = 0
            
            # === MAIN TRAINING LOOP ===
            try:
                for episode in range(max_episodes):
                    if not self.training_in_progress:
                        print("⏹️ Training stopped")
                        break
                    
                    episode_reward = 0
                    step_rewards = []
                    episode_policy_losses = []  # Track policy losses for this episode
                    episode_q_losses = []        # Track Q-value losses for this episode
                    
                    # Reset environment - start from randomly selected realistic initial latent state
                    state = self.environment.reset(z0_options)
                    
                    # Start tracking this episode
                    self.training_orchestrator.start_new_episode()
                    
                    for step in range(max_steps):
                        if not self.training_in_progress:
                            break
                        
                        # Enhanced action selection (for PPO, ActionVariation is bypassed)
                        if getattr(self, '_is_ppo', False):
                            action = self.agent.select_action(state, evaluate=False)
                        else:
                            action = self.training_orchestrator.select_enhanced_action(
                                self.agent, state, episode, step, exploration_steps, self.total_numsteps
                            )
                        
                        # Step environment
                        next_state, reward, done = self.environment.step(action)
                        
                        # Record step data
                        observation = getattr(self.environment, 'last_observation', None)
                        self.training_orchestrator.record_step_data(
                            step=step,
                            action=action,
                            observation=observation,
                            reward=reward,
                            state=state
                        )
                        
                        self.total_numsteps += 1
                        step_rewards.append(reward.item())
                        
                        if getattr(self, '_is_ppo', False):
                            # PPO: collect transition in rollout buffer
                            self.agent.collect_step(state, action, reward, done)
                        else:
                            # Off-policy: store transition in replay memory
                            self.memory.push(state, action, reward, next_state)
                        
                        # Move to next state
                        state = next_state
                        episode_reward += reward.item()
                        
                        # Update agent (off-policy only; PPO update_parameters is a no-op)
                        if not getattr(self, '_is_ppo', False) and len(self.memory) > batch_size:
                            for _ in range(updates_per_step):
                                critic1_loss, critic2_loss, policy_loss, alpha_loss, alpha_val = \
                                    self.agent.update_parameters(self.memory, batch_size, self.global_step)
                                
                                episode_policy_losses.append(policy_loss)
                                episode_q_losses.append(critic1_loss + critic2_loss)
                                
                                self.rl_metrics['policy/loss'] = policy_loss
                                self.rl_metrics['q_value/loss'] = critic1_loss + critic2_loss
                                self.rl_metrics['alpha/loss'] = alpha_loss
                                self.rl_metrics['alpha/value'] = alpha_val
                                self.rl_metrics['training/step'] = self.global_step
                                
                                self.global_step += 1
                        
                        # Early termination
                        if done:
                            break
                    
                    # PPO: episode-level update
                    if getattr(self, '_is_ppo', False):
                        c1, c2, pl, al, av = self.agent.update_from_buffer(state)
                        episode_policy_losses.append(pl)
                        episode_q_losses.append(c1)
                        self.rl_metrics['policy/loss'] = pl
                        self.rl_metrics['q_value/loss'] = c1
                        self.global_step += 1
                    
                    # Finalize episode
                    operational_reward = self.training_orchestrator.finalize_episode(
                        episode, total_reward=episode_reward
                    )
                    final_reward = operational_reward if operational_reward is not None else episode_reward
                    
                    # Store rewards
                    self.episode_rewards.append(final_reward)
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    self.avg_rewards.append(avg_reward)
                    
                    # Calculate average losses for this episode
                    avg_policy_loss = np.mean(episode_policy_losses) if episode_policy_losses else None
                    avg_q_loss = np.mean(episode_q_losses) if episode_q_losses else None
                    
                    # Record metrics (including losses)
                    self.training_orchestrator.record_training_metrics(
                        episode, final_reward, avg_reward, 
                        policy_loss=avg_policy_loss, 
                        q_loss=avg_q_loss
                    )
                    
                    # Update WandB metrics
                    self.rl_metrics['episode'] = episode
                    self.rl_metrics['reward/total'] = final_reward
                    self.rl_metrics['reward/avg'] = np.mean(step_rewards)
                    self.rl_metrics['reward/min'] = np.min(step_rewards)
                    self.rl_metrics['reward/max'] = np.max(step_rewards)
                    
                    # Log to WandB
                    if self.wandb_logger:
                        self.wandb_logger.log_training_step(
                            get_pre_loaded_rom(), episode, 0, self.global_step
                        )
                        wandb.log(self.rl_metrics, step=self.global_step)
                    
                    # Update progress
                    self.progress_bar.value = episode + 1
                    
                    # Print progress
                    if episode % print_interval == 0:
                        if use_eval_checkpoint:
                            status_msg = (
                                f"Episode {episode+1}/{max_episodes} | "
                                f"Train Reward: {final_reward:.2f} | "
                                f"Avg(10): {avg_reward:.2f} | "
                                f"Best Train: {self.best_reward:.2f} | "
                                f"Best Eval: {self.best_eval_reward:.4f}"
                            )
                        else:
                            status_msg = (
                                f"Episode {episode+1}/{max_episodes} | "
                                f"Reward: {final_reward:.2f} | "
                                f"Avg(10): {avg_reward:.2f} | "
                                f"Best: {self.best_reward:.2f}"
                            )
                        print(status_msg)
                        self.training_info.value = f"<p><b>Status:</b> {status_msg}</p>"
                    
                    # === CHECKPOINT SAVING ===
                    if use_eval_checkpoint:
                        # --- Evaluation-Based Checkpoint Selection ---
                        # Track best training reward for display only
                        if final_reward > self.best_reward:
                            self.best_reward = final_reward
                        
                        # Run deterministic evaluation at the specified interval
                        if episode % eval_interval == 0:
                            mean_eval_npv, std_eval_npv = self._run_deterministic_evaluation(
                                z0_options, num_cases=eval_num_cases
                            )
                            
                            # Log evaluation result
                            eval_msg = (
                                f"   🔍 Eval (deterministic): Mean NPV = {mean_eval_npv:.4f} "
                                f"± {std_eval_npv:.4f} | Best Eval = {self.best_eval_reward:.4f}"
                            )
                            print(eval_msg)
                            
                            # Log to WandB
                            if self.wandb_logger:
                                wandb.log({
                                    'eval/mean_npv': mean_eval_npv,
                                    'eval/std_npv': std_eval_npv,
                                    'eval/best_npv': self.best_eval_reward,
                                    'eval/episode': episode,
                                }, step=self.global_step)
                            
                            # Save checkpoint only if evaluation NPV improved
                            if mean_eval_npv > self.best_eval_reward:
                                self.best_eval_reward = mean_eval_npv
                                _rom_tag = getattr(self, '_rom_model_tag', 'unknown')
                                self.agent.save_checkpoint(f"best_{_rom_tag}", suffix=f"ep{episode}")
                                print(
                                    f"   💾 New best model saved! "
                                    f"Eval NPV: {self.best_eval_reward:.4f} "
                                    f"(training reward was {final_reward:.2f})"
                                )
                    else:
                        # --- Default: Training-reward-based checkpoint ---
                        if final_reward > self.best_reward:
                            self.best_reward = final_reward
                            _rom_tag = getattr(self, '_rom_model_tag', 'unknown')
                            self.agent.save_checkpoint(f"best_{_rom_tag}", suffix=f"ep{episode}")
                            print(f"   💾 New best model saved! Reward: {self.best_reward:.2f}")
                    
                    # Periodic save (always, regardless of checkpoint strategy)
                    if (episode + 1) % save_interval == 0:
                        _rom_tag = getattr(self, '_rom_model_tag', 'unknown')
                        self.agent.save_checkpoint(f"periodic_{_rom_tag}", suffix=f"ep{episode+1}")
                        print(f"   💾 Checkpoint saved at episode {episode+1}")
                
                # Training complete
                print(f"\n✅ Training completed!")
                print(f"   Best training reward: {self.best_reward:.2f}")
                if use_eval_checkpoint:
                    print(f"   Best evaluation NPV (deterministic): {self.best_eval_reward:.4f}")
                    print(f"   Checkpoint saved based on: evaluation NPV")
                else:
                    print(f"   Checkpoint saved based on: training reward")
                print(f"   Total episodes: {len(self.episode_rewards)}")
                
                # Get training summary
                variation_summary = self.training_orchestrator.get_training_summary()
                if isinstance(variation_summary, dict):
                    print(f"\n📊 Action Variation Summary:")
                    print(f"   Mean variation: {variation_summary.get('mean_variation', 0):.4f}")
                    print(f"   Max variation: {variation_summary.get('max_variation', 0):.4f}")
                
                # Auto-save training results for later visualization
                try:
                    self.last_saved_results = self.training_orchestrator.save_results(
                        rom_name=getattr(self, '_rom_model_tag', None),
                        algorithm_type=getattr(self, '_algo_type', None),
                        prediction_mode=getattr(self, '_prediction_mode', None),
                    )
                except Exception as _save_err:
                    print(f"⚠️ Could not auto-save training results: {_save_err}")

                # Finish WandB
                if self.wandb_logger:
                    self.wandb_logger.finish()
                
                if use_eval_checkpoint:
                    self.training_info.value = (
                        f"<p><b>Status:</b> Training completed! "
                        f"Best eval NPV: {self.best_eval_reward:.4f} | "
                        f"Best train reward: {self.best_reward:.2f}</p>"
                    )
                else:
                    self.training_info.value = (
                        f"<p><b>Status:</b> Training completed! "
                        f"Best reward: {self.best_reward:.2f}</p>"
                    )
                
            except Exception as e:
                print(f"❌ Training error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.training_in_progress = False
                self.start_button.disabled = False
                self.stop_button.disabled = True
    
    def display(self):
        """Display the training dashboard"""
        if not WIDGETS_AVAILABLE:
            print("❌ Widgets not available - cannot display dashboard")
            return
        
        display(self.main_widget)
    
    def get_training_orchestrator(self):
        """Get the training orchestrator for visualization"""
        return self.training_orchestrator


def create_rl_training_dashboard(config_path='config.yaml'):
    """Create RL training dashboard instance"""
    if not WIDGETS_AVAILABLE:
        print("⚠️ Interactive widgets not available")
        print("   Training will run without interactive controls")
        return None
    
    dashboard = RLTrainingDashboard(config_path)
    
    if dashboard.config is None:
        print("⚠️ Configuration not loaded - please run Configuration Dashboard first")
        return dashboard
    
    print("✅ Training dashboard created successfully!")
    print("   Click 'Start Training' to begin RL training")
    
    dashboard.display()
    return dashboard
