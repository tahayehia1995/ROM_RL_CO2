"""
Base Optimizer Abstract Class
=============================

Defines the interface that all optimizers must implement for ROM-based reservoir optimization.
This ensures consistency across different optimization methods (LS-SQP, GA, PSO, etc.)
and enables fair comparison with RL approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import time


@dataclass
class OptimizationResult:
    """
    Container for optimization results.
    
    Stores all relevant data for analysis and comparison with RL results.
    """
    # Optimal solution
    optimal_controls: np.ndarray  # Shape: (num_steps, num_wells * num_control_types)
    optimal_objective: float  # Final objective value (total NPV)
    
    # Trajectory at optimal solution
    optimal_states: Optional[torch.Tensor] = None  # Latent states: (num_steps+1, latent_dim)
    optimal_spatial_states: Optional[List[torch.Tensor]] = None  # Decoded spatial states
    optimal_observations: Optional[np.ndarray] = None  # Well observations: (num_steps, num_obs)
    
    # Optimization history
    objective_history: List[float] = field(default_factory=list)  # Objective at each iteration
    gradient_norm_history: List[float] = field(default_factory=list)  # Gradient norm history
    control_history: List[np.ndarray] = field(default_factory=list)  # Controls at each iteration
    
    # Performance metrics
    num_iterations: int = 0  # Total iterations
    num_function_evaluations: int = 0  # Total ROM rollouts
    num_gradient_evaluations: int = 0  # Total gradient computations
    total_time_seconds: float = 0.0  # Wall-clock time
    convergence_achieved: bool = False  # Whether converged to tolerance
    termination_reason: str = ""  # Why optimization stopped
    
    # Configuration used
    optimizer_type: str = ""  # Name of optimizer used
    optimizer_params: Dict[str, Any] = field(default_factory=dict)  # Optimizer parameters
    
    # Initial condition info
    num_realizations: int = 1  # Number of Z0 realizations used
    initial_controls: Optional[np.ndarray] = None  # Starting control guess
    initial_objective: float = 0.0  # Objective at initial guess
    
    # Economic breakdown at optimal solution
    economic_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def improvement_ratio(self) -> float:
        """Calculate improvement from initial to optimal."""
        if self.initial_objective == 0:
            return float('inf') if self.optimal_objective > 0 else 0.0
        return (self.optimal_objective - self.initial_objective) / abs(self.initial_objective)
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            f"Optimization Results: {self.optimizer_type}",
            "=" * 60,
            f"Optimal Objective (NPV): {self.optimal_objective:.6f}",
            f"Initial Objective: {self.initial_objective:.6f}",
            f"Improvement: {self.improvement_ratio() * 100:.2f}%",
            "-" * 60,
            f"Iterations: {self.num_iterations}",
            f"Function Evaluations: {self.num_function_evaluations}",
            f"Gradient Evaluations: {self.num_gradient_evaluations}",
            f"Total Time: {self.total_time_seconds:.2f}s",
            f"Converged: {self.convergence_achieved}",
            f"Termination: {self.termination_reason}",
            "-" * 60,
            f"Realizations Used: {self.num_realizations}",
            f"Control Shape: {self.optimal_controls.shape}",
            "=" * 60,
        ]
        return "\n".join(lines)


class BaseOptimizer(ABC):
    """
    Abstract base class for all reservoir optimization methods.
    
    Provides common functionality for:
    - ROM model integration and rollouts
    - Objective function evaluation using RL reward function
    - Action normalization/denormalization
    - Constraint handling
    
    Subclasses must implement the `optimize` method with their specific algorithm.
    """
    
    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        action_ranges: Optional[Dict] = None
    ):
        """
        Initialize base optimizer.
        
        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device (cuda/cpu)
            action_ranges: Optional action range constraints
        """
        self.rom = rom_model
        self.config = config
        self.norm_params = norm_params
        self.device = device
        
        # Extract well configuration
        self.num_prod = config.data.get('num_prod', 3)
        self.num_inj = config.data.get('num_inj', 3)
        self.num_controls = self.num_prod + self.num_inj  # Total control variables per timestep
        
        # Default action ranges (can be overridden)
        self.action_ranges = action_ranges or {
            'producer_bhp': {
                'min': 1087.784912109375,
                'max': 5050.42871094
            },
            'gas_injection': {
                'min': 0.0,
                'max': 143972480.0
            }
        }
        
        # Get normalization bounds for ROM compatibility
        self._setup_normalization_bounds()
        
        # Print configuration for debugging
        print(f"\n📊 Optimizer Configuration:")
        print(f"   Action Ranges (Physical Bounds for Optimization):")
        print(f"      Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, {self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"      Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, {self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"   Normalization Bounds (For ROM Input):")
        print(f"      BHP: [{self.bhp_norm_min:.2f}, {self.bhp_norm_max:.2f}] psi")
        print(f"      GASRATSC: [{self.gas_norm_min:.0f}, {self.gas_norm_max:.0f}] ft³/day")
        print(f"   Norm Params Keys: {list(self.norm_params.keys())}")
        
        # Time step for ROM integration
        self.dt = torch.tensor([[1.0]], device=device)
        
        # Tracking
        self.function_eval_count = 0
        self.gradient_eval_count = 0
    
    def _setup_normalization_bounds(self):
        """Extract normalization bounds from norm_params."""
        # BHP normalization bounds (for producers)
        if 'BHP' in self.norm_params:
            bhp_params = self.norm_params['BHP']
            self.bhp_norm_min = float(bhp_params.get('min', 0))
            self.bhp_norm_max = float(bhp_params.get('max', 1))
        else:
            self.bhp_norm_min = 0.0
            self.bhp_norm_max = 5050.0
        
        # Gas injection normalization bounds
        if 'GASRATSC' in self.norm_params:
            gas_params = self.norm_params['GASRATSC']
            self.gas_norm_min = float(gas_params.get('min', 0))
            self.gas_norm_max = float(gas_params.get('max', 1))
        else:
            self.gas_norm_min = 0.0
            self.gas_norm_max = 143972480.0
    
    @abstractmethod
    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> OptimizationResult:
        """
        Run optimization to find optimal control sequence.
        
        Args:
            z0_options: Tensor of initial latent states (num_realizations, latent_dim)
                       If None, uses stored z0_options from config
            num_steps: Number of control timesteps
                      If None, uses default from config (typically 30)
        
        Returns:
            OptimizationResult with optimal controls and performance data
        """
        pass
    
    def evaluate_objective(
        self,
        controls_normalized: np.ndarray,
        z0: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[float, Optional[Dict]]:
        """
        Evaluate objective function for given controls.
        
        IMPORTANT: Controls are in NORMALIZED [0,1] space.
        They are converted to physical units before ROM evaluation.
        
        Args:
            controls_normalized: Control sequence in [0,1], shape (num_steps, num_controls) or flattened
            z0: Initial latent state, shape (1, latent_dim)
            return_trajectory: If True, also return states and observations
            
        Returns:
            objective: Scalar objective value (total NPV)
            trajectory: Optional dict with states, observations, rewards
        """
        self.function_eval_count += 1
        
        # Reshape controls if flattened
        if controls_normalized.ndim == 1:
            num_steps = len(controls_normalized) // self.num_controls
            controls_normalized = controls_normalized.reshape(num_steps, self.num_controls)
        
        # Convert normalized [0,1] controls to physical units
        controls_physical = self.controls_normalized_to_physical(controls_normalized)
        
        # Run ROM rollout with physical controls
        states, observations, rewards = self.rom_rollout(z0, controls_physical)
        
        # Total objective is sum of rewards (NPV)
        total_objective = sum(rewards)
        
        if return_trajectory:
            trajectory = {
                'states': states,
                'observations': observations,
                'rewards': rewards,
                'per_step_objective': rewards,
                'controls_physical': controls_physical  # Store physical controls for visualization
            }
            return total_objective, trajectory
        
        return total_objective, None
    
    def evaluate_robust_objective(
        self,
        controls: np.ndarray,
        z0_ensemble: torch.Tensor
    ) -> Tuple[float, List[float]]:
        """
        Evaluate expected objective over multiple realizations.
        
        Args:
            controls: Control sequence
            z0_ensemble: Multiple initial states, shape (num_realizations, latent_dim)
            
        Returns:
            mean_objective: Average objective over realizations
            individual_objectives: List of objectives per realization
        """
        individual_objectives = []
        
        for i in range(z0_ensemble.shape[0]):
            z0_single = z0_ensemble[i:i+1]  # Keep batch dimension
            obj, _ = self.evaluate_objective(controls, z0_single)
            individual_objectives.append(obj)
        
        mean_objective = np.mean(individual_objectives)
        return mean_objective, individual_objectives
    
    def rom_rollout(
        self,
        z0: torch.Tensor,
        control_sequence: np.ndarray
    ) -> Tuple[List[torch.Tensor], List[np.ndarray], List[float]]:
        """
        Run ROM simulation with given control sequence.
        
        Uses EXACT same prediction method as RL environment (state_based mode):
        1. Decode latent to spatial state
        2. Call rom.predict() with spatial state, controls, dummy obs
        3. Re-encode to latent for next step
        4. Denormalize observations for reward calculation
        
        Args:
            z0: Initial latent state, shape (1, latent_dim)
            control_sequence: Control actions in PHYSICAL units, shape (num_steps, num_controls)
                              [Producer_BHP (psi), Gas_Injection (ft³/day)]
            
        Returns:
            states: List of latent states at each step
            observations: List of well observations at each step (physical units)
            rewards: List of economic rewards at each step
        """
        from .objective import compute_step_reward
        
        states = [z0.clone()]
        observations = []
        rewards = []
        
        current_latent = z0.clone()
        num_steps = len(control_sequence)
        
        # Initialize spatial state from latent (matching RL environment)
        with torch.no_grad():
            current_spatial = self.rom.model.decoder(current_latent)
        
        for t in range(num_steps):
            # Get PHYSICAL control for this step
            control_physical = control_sequence[t]  # In physical units (psi, ft³/day)
            
            # Convert physical controls to NORMALIZED format for ROM input
            control_normalized = self._prepare_control_for_rom(control_physical)
            
            # Create dummy observation (same as RL environment)
            # Structure: [Injector_BHP(3), Gas_production(3), Water_production(3)] = 9 observations
            dummy_obs = torch.zeros(current_spatial.shape[0], 9).to(self.device)
            
            # ROM prediction using EXACT same method as RL environment (state_based)
            # Input format: (spatial_state, controls, observations, dt)
            with torch.no_grad():
                inputs = (current_spatial, control_normalized, dummy_obs, self.dt)
                next_spatial, yobs_normalized = self.rom.predict(inputs)
                
                # Encode next spatial to latent for state tracking
                # Encoder returns (z, mean, logvar) tuple - extract z (first element)
                encoder_output = self.rom.model.encoder(next_spatial)
                if isinstance(encoder_output, tuple):
                    next_latent = encoder_output[0]
                else:
                    next_latent = encoder_output
            
            # Denormalize observations to PHYSICAL units
            yobs_physical = self._denormalize_observations(yobs_normalized)
            
            # Clamp to non-negative (same as RL environment)
            yobs_physical = torch.clamp(yobs_physical, min=0.0)
            
            # Create PHYSICAL action tensor for reward calculation
            action_physical_tensor = torch.tensor(
                control_physical, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            
            # Compute reward using PHYSICAL values (same as RL)
            reward = compute_step_reward(
                yobs_physical, action_physical_tensor, 
                self.num_prod, self.num_inj, self.config
            )
            
            # Store results
            states.append(next_latent.clone())
            observations.append(yobs_physical.cpu().numpy())
            rewards.append(float(reward.sum().item()))
            
            # Update for next iteration
            current_spatial = next_spatial
            current_latent = next_latent
        
        return states, observations, rewards
    
    def _prepare_control_for_rom(self, control_physical: np.ndarray) -> torch.Tensor:
        """
        Convert physical controls to ROM-normalized format.
        
        This matches EXACTLY the RL environment's _map_dashboard_action_to_rom_input:
        1. Takes physical controls (BHP in psi, Gas in ft³/day)
        2. Normalizes using TRAINING normalization parameters
        3. Returns [0,1] normalized values for ROM
        
        Args:
            control_physical: Physical control values [BHP(psi), Gas(ft³/day)]
                              Shape: (num_controls,) = [P1_BHP, P2_BHP, P3_BHP, I1_Gas, I2_Gas, I3_Gas]
            
        Returns:
            control_normalized: Tensor normalized for ROM input, shape (1, num_controls)
        """
        control_normalized = np.zeros(self.num_controls)
        
        # Normalize producer BHP (first num_prod controls) using TRAINING params
        bhp_physical = control_physical[:self.num_prod]
        control_normalized[:self.num_prod] = (bhp_physical - self.bhp_norm_min) / (self.bhp_norm_max - self.bhp_norm_min + 1e-8)
        
        # Normalize gas injection (last num_inj controls) using TRAINING params
        gas_physical = control_physical[self.num_prod:]
        control_normalized[self.num_prod:] = (gas_physical - self.gas_norm_min) / (self.gas_norm_max - self.gas_norm_min + 1e-8)
        
        # Clip to [0, 1] for safety (values outside training range get clipped)
        control_normalized = np.clip(control_normalized, 0, 1)
        
        return torch.tensor(control_normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _denormalize_observations(self, yobs_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize ROM observations to physical units.
        
        Observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
        This matches EXACTLY the RL environment's _denormalize_observations_rom.
        
        Args:
            yobs_normalized: Normalized observations from ROM, shape (batch, 9)
            
        Returns:
            yobs_physical: Physical observations (BHP in psi, rates in ft³/day)
        """
        yobs_physical = yobs_normalized.clone()
        
        # Denormalize injector BHP (first num_inj=3 observations, indices 0-2)
        if 'BHP' in self.norm_params:
            bhp_params = self.norm_params['BHP']
            norm_type = bhp_params.get('type', 'minmax')
            
            if norm_type == 'log':
                # Log normalization (if used)
                log_min = float(bhp_params.get('log_min', 0))
                log_max = float(bhp_params.get('log_max', 1))
                epsilon = float(bhp_params.get('epsilon', 1e-8))
                data_shift = float(bhp_params.get('data_shift', 0))
                log_data = yobs_normalized[:, :self.num_inj] * (log_max - log_min) + log_min
                yobs_physical[:, :self.num_inj] = torch.exp(log_data) - epsilon + data_shift
            else:
                # Minmax normalization (standard)
                bhp_min = float(bhp_params.get('min', 0))
                bhp_max = float(bhp_params.get('max', 1))
                yobs_physical[:, :self.num_inj] = yobs_normalized[:, :self.num_inj] * (bhp_max - bhp_min) + bhp_min
        
        # Denormalize gas production (next num_prod=3 observations, indices 3-5)
        if 'GASRATSC' in self.norm_params:
            gas_params = self.norm_params['GASRATSC']
            norm_type = gas_params.get('type', 'minmax')
            
            if norm_type == 'log':
                log_min = float(gas_params.get('log_min', 0))
                log_max = float(gas_params.get('log_max', 1))
                epsilon = float(gas_params.get('epsilon', 1e-8))
                data_shift = float(gas_params.get('data_shift', 0))
                log_data = yobs_normalized[:, self.num_inj:self.num_inj+self.num_prod] * (log_max - log_min) + log_min
                yobs_physical[:, self.num_inj:self.num_inj+self.num_prod] = torch.exp(log_data) - epsilon + data_shift
            else:
                gas_min = float(gas_params.get('min', 0))
                gas_max = float(gas_params.get('max', 1))
                yobs_physical[:, self.num_inj:self.num_inj+self.num_prod] = (
                    yobs_normalized[:, self.num_inj:self.num_inj+self.num_prod] * (gas_max - gas_min) + gas_min
                )
        
        # Denormalize water production (last num_prod=3 observations, indices 6-8)
        if 'WATRATSC' in self.norm_params:
            wat_params = self.norm_params['WATRATSC']
            norm_type = wat_params.get('type', 'minmax')
            
            if norm_type == 'log':
                log_min = float(wat_params.get('log_min', 0))
                log_max = float(wat_params.get('log_max', 1))
                epsilon = float(wat_params.get('epsilon', 1e-8))
                data_shift = float(wat_params.get('data_shift', 0))
                log_data = yobs_normalized[:, self.num_inj+self.num_prod:] * (log_max - log_min) + log_min
                yobs_physical[:, self.num_inj+self.num_prod:] = torch.exp(log_data) - epsilon + data_shift
            else:
                wat_min = float(wat_params.get('min', 0))
                wat_max = float(wat_params.get('max', 1))
                yobs_physical[:, self.num_inj+self.num_prod:] = (
                    yobs_normalized[:, self.num_inj+self.num_prod:] * (wat_max - wat_min) + wat_min
                )
        
        # Clamp to non-negative (same as RL environment)
        yobs_physical = torch.clamp(yobs_physical, min=0.0)
        
        return yobs_physical
    
    def get_bounds(self, num_steps: int) -> List[Tuple[float, float]]:
        """
        Get scipy-compatible bounds for all control variables.
        
        IMPORTANT: We optimize in NORMALIZED [0,1] space to ensure
        all variables have similar scales for the optimizer.
        
        Args:
            num_steps: Number of control timesteps
            
        Returns:
            List of (min, max) tuples for each control variable (all [0,1])
        """
        bounds = []
        
        for _ in range(num_steps):
            # All controls bounded to [0, 1] in normalized space
            for _ in range(self.num_controls):
                bounds.append((0.0, 1.0))
        
        return bounds
    
    def controls_normalized_to_physical(self, controls_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized [0,1] controls to physical units.
        
        Args:
            controls_normalized: Controls in [0,1] range
            
        Returns:
            controls_physical: Controls in physical units (psi, ft³/day)
        """
        if controls_normalized.ndim == 1:
            controls_normalized = controls_normalized.reshape(-1, self.num_controls)
        
        controls_physical = np.zeros_like(controls_normalized)
        
        # BHP: [0,1] -> physical range
        bhp_min = self.action_ranges['producer_bhp']['min']
        bhp_max = self.action_ranges['producer_bhp']['max']
        controls_physical[:, :self.num_prod] = (
            controls_normalized[:, :self.num_prod] * (bhp_max - bhp_min) + bhp_min
        )
        
        # Gas: [0,1] -> physical range
        gas_min = self.action_ranges['gas_injection']['min']
        gas_max = self.action_ranges['gas_injection']['max']
        controls_physical[:, self.num_prod:] = (
            controls_normalized[:, self.num_prod:] * (gas_max - gas_min) + gas_min
        )
        
        return controls_physical
    
    def controls_physical_to_normalized(self, controls_physical: np.ndarray) -> np.ndarray:
        """
        Convert physical controls to normalized [0,1] range.
        
        Args:
            controls_physical: Controls in physical units
            
        Returns:
            controls_normalized: Controls in [0,1] range
        """
        if controls_physical.ndim == 1:
            controls_physical = controls_physical.reshape(-1, self.num_controls)
        
        controls_normalized = np.zeros_like(controls_physical)
        
        # BHP: physical -> [0,1]
        bhp_min = self.action_ranges['producer_bhp']['min']
        bhp_max = self.action_ranges['producer_bhp']['max']
        controls_normalized[:, :self.num_prod] = (
            (controls_physical[:, :self.num_prod] - bhp_min) / (bhp_max - bhp_min + 1e-8)
        )
        
        # Gas: physical -> [0,1]
        gas_min = self.action_ranges['gas_injection']['min']
        gas_max = self.action_ranges['gas_injection']['max']
        controls_normalized[:, self.num_prod:] = (
            (controls_physical[:, self.num_prod:] - gas_min) / (gas_max - gas_min + 1e-8)
        )
        
        return controls_normalized
    
    def generate_initial_guess(self, num_steps: int, strategy: str = 'midpoint') -> np.ndarray:
        """
        Generate initial control guess in NORMALIZED [0,1] space.
        
        Args:
            num_steps: Number of control timesteps
            strategy: 'midpoint' (0.5), 'random', or 'low' (0.1)
            
        Returns:
            Initial control array in [0,1], shape (num_steps * num_controls,)
        """
        n_vars = num_steps * self.num_controls
        
        if strategy == 'midpoint':
            initial = np.full(n_vars, 0.5)  # Middle of [0,1]
        elif strategy == 'random':
            initial = np.random.uniform(0, 1, n_vars)
        elif strategy == 'low':
            initial = np.full(n_vars, 0.1)  # 10% of range
        else:
            initial = np.full(n_vars, 0.5)
        
        return initial
    
    def decode_spatial_states(self, latent_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Decode latent states to spatial fields using ROM decoder.
        
        Args:
            latent_states: List of latent state tensors
            
        Returns:
            List of decoded spatial state tensors
        """
        spatial_states = []
        
        with torch.no_grad():
            for z in latent_states:
                spatial = self.rom.model.decoder(z)
                spatial_states.append(spatial)
        
        return spatial_states
    
    def reset_counters(self):
        """Reset function and gradient evaluation counters."""
        self.function_eval_count = 0
        self.gradient_eval_count = 0
    
    def test_rom_sensitivity(self, z0: torch.Tensor):
        """
        Test if ROM predictions change when controls change.
        
        This is a diagnostic function to verify the ROM responds to control inputs.
        """
        print("\n" + "="*60)
        print("ROM SENSITIVITY TEST")
        print("="*60)
        
        # Test controls at bounds and midpoint
        bhp_min = self.action_ranges['producer_bhp']['min']
        bhp_max = self.action_ranges['producer_bhp']['max']
        gas_min = self.action_ranges['gas_injection']['min']
        gas_max = self.action_ranges['gas_injection']['max']
        
        control_sets = {
            'Low BHP, Low Gas': np.array([bhp_min]*self.num_prod + [gas_min]*self.num_inj),
            'High BHP, High Gas': np.array([bhp_max]*self.num_prod + [gas_max]*self.num_inj),
            'Low BHP, High Gas': np.array([bhp_min]*self.num_prod + [gas_max]*self.num_inj),
            'High BHP, Low Gas': np.array([bhp_max]*self.num_prod + [gas_min]*self.num_inj),
        }
        
        # Initialize spatial state
        with torch.no_grad():
            current_spatial = self.rom.model.decoder(z0)
        
        results = {}
        for name, control_physical in control_sets.items():
            # Normalize controls
            control_normalized = self._prepare_control_for_rom(control_physical)
            
            # ROM prediction
            dummy_obs = torch.zeros(1, 9).to(self.device)
            with torch.no_grad():
                inputs = (current_spatial, control_normalized, dummy_obs, self.dt)
                _, yobs_normalized = self.rom.predict(inputs)
            
            # Denormalize
            yobs_physical = self._denormalize_observations(yobs_normalized)
            
            results[name] = {
                'control_norm': control_normalized.cpu().numpy(),
                'yobs_physical': yobs_physical.cpu().numpy()
            }
            
            print(f"\n{name}:")
            print(f"  Control (normalized): {control_normalized.cpu().numpy().flatten()[:6]}")
            print(f"  Observations: BHP={yobs_physical[0, :3].cpu().numpy()}, "
                  f"Gas={yobs_physical[0, 3:6].cpu().numpy()}, "
                  f"Water={yobs_physical[0, 6:].cpu().numpy()}")
        
        # Check if observations differ
        obs_arrays = [r['yobs_physical'] for r in results.values()]
        obs_range = np.max([np.max(o) for o in obs_arrays]) - np.min([np.min(o) for o in obs_arrays])
        
        print(f"\nObservation range across control sets: {obs_range:.6f}")
        if obs_range < 1e-6:
            print("WARNING: ROM outputs appear INVARIANT to control changes!")
            print("This indicates the ROM may not be using controls properly.")
        else:
            print("ROM outputs VARY with control changes.")
        
        print("="*60 + "\n")
        
        return results
