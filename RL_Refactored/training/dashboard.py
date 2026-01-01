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
from RL_Refactored.agent import create_sac_agent
from RL_Refactored.agent.replay_memory import ReplayMemory
from RL_Refactored.environment import create_environment
from RL_Refactored.training.orchestrator import EnhancedTrainingOrchestrator
from RL_Refactored.utilities import Config
from ROM_Refactored.utilities.wandb_integration import create_wandb_logger

# Import configuration utilities
from RL_Refactored.configuration import (
    get_rl_config, has_rl_config, get_pre_loaded_rom, get_pre_generated_z0,
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
        
        # Main widget
        self.main_widget = widgets.VBox([
            widgets.HTML("<h3>🚀 RL Training Dashboard</h3>"),
            widgets.HTML("<p>Configure and start RL training with SAC algorithm</p>"),
            widgets.HBox([self.start_button, self.stop_button]),
            self.training_info,
            self.progress_bar,
            self.status_output
        ])
    
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
                
                # Create SAC agent (pass ROM model to extract correct latent_dim)
                print("   🤖 Creating SAC agent...")
                self.agent = create_sac_agent(self.config, self.rl_config, rom_model=my_rom)
                
                # Create environment
                print("   🌍 Creating environment...")
                self.environment = create_environment(z0_options, self.config, my_rom, self.rl_config)
                
                # Verify dashboard action mapping
                print("   🔍 Verifying action mapping...")
                self.environment.verify_dashboard_action_mapping()
                
                # Create replay memory
                print("   💾 Creating replay memory...")
                batch_size = self.config.rl_model['replay_memory']['batch_size']
                capacity = self.config.rl_model['replay_memory']['capacity']
                seed = self.config.rl_model['training']['seeds']['replay_memory']
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
            training_config = self.config.rl_model['training']
            max_episodes = training_config['max_episodes']
            max_steps = training_config['max_steps_per_episode']
            batch_size = self.config.rl_model['replay_memory']['batch_size']
            updates_per_step = training_config['updates_per_step']
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
            
            # Reset tracking variables
            self.episode_rewards = []
            self.avg_rewards = []
            self.best_reward = -np.inf
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
                        
                        # Enhanced action selection
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
                        
                        # Store transition
                        self.memory.push(state, action, reward, next_state)
                        
                        # Move to next state
                        state = next_state
                        episode_reward += reward.item()
                        
                        # Update agent
                        if len(self.memory) > batch_size:
                            for _ in range(updates_per_step):
                                critic1_loss, critic2_loss, policy_loss, alpha_loss, alpha_val = \
                                    self.agent.update_parameters(self.memory, batch_size, self.global_step)
                                
                                # Track losses for this episode
                                episode_policy_losses.append(policy_loss)
                                episode_q_losses.append(critic1_loss + critic2_loss)
                                
                                # Update metrics
                                self.rl_metrics['policy/loss'] = policy_loss
                                self.rl_metrics['q_value/loss'] = critic1_loss + critic2_loss
                                self.rl_metrics['alpha/loss'] = alpha_loss
                                self.rl_metrics['alpha/value'] = alpha_val
                                self.rl_metrics['training/step'] = self.global_step
                                
                                self.global_step += 1
                        
                        # Early termination
                        if done:
                            break
                    
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
                        status_msg = (
                            f"Episode {episode+1}/{max_episodes} | "
                            f"Reward: {final_reward:.2f} | "
                            f"Avg(10): {avg_reward:.2f} | "
                            f"Best: {self.best_reward:.2f}"
                        )
                        print(status_msg)
                        self.training_info.value = f"<p><b>Status:</b> {status_msg}</p>"
                    
                    # Save best model
                    if final_reward > self.best_reward:
                        self.best_reward = final_reward
                        self.agent.save_checkpoint("best_model", suffix=f"ep{episode}")
                        print(f"   💾 New best model saved! Reward: {self.best_reward:.2f}")
                    
                    # Periodic save
                    if (episode + 1) % save_interval == 0:
                        self.agent.save_checkpoint("periodic", suffix=f"ep{episode+1}")
                        print(f"   💾 Checkpoint saved at episode {episode+1}")
                
                # Training complete
                print(f"\n✅ Training completed!")
                print(f"   Best reward: {self.best_reward:.2f}")
                print(f"   Total episodes: {len(self.episode_rewards)}")
                
                # Get training summary
                variation_summary = self.training_orchestrator.get_training_summary()
                if isinstance(variation_summary, dict):
                    print(f"\n📊 Action Variation Summary:")
                    print(f"   Mean variation: {variation_summary.get('mean_variation', 0):.4f}")
                    print(f"   Max variation: {variation_summary.get('max_variation', 0):.4f}")
                
                # Finish WandB
                if self.wandb_logger:
                    self.wandb_logger.finish()
                
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
