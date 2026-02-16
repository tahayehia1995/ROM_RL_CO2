"""
Genetic Algorithm (GA) Optimizer
=================================

Genetic Algorithm optimizer for reservoir production optimization.
Implements a standard GA with optional DEAP backend.

GA is particularly effective for:
- Global optimization without gradients
- Mixed discrete/continuous optimization
- Problems with complex constraints
- Highly multimodal landscapes

References:
- Holland (1975) - Adaptation in Natural and Artificial Systems
- Goldberg (1989) - Genetic Algorithms in Search, Optimization, and Machine Learning
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any, Callable

from .base_optimizer import BaseOptimizer, OptimizationResult

# Check if DEAP library is available
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


class GAOptimizer(BaseOptimizer):
    """
    Genetic Algorithm for reservoir production optimization.
    
    Uses either DEAP library (if available) or a built-in implementation.
    
    Key Features:
    - Population-based evolutionary search
    - Selection, crossover, and mutation operators
    - No gradient information required
    - RL-like random case sampling from Z0 pool
    """
    
    def __init__(
        self,
        rom_model,
        config,
        norm_params: Dict,
        device: torch.device,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        action_ranges: Optional[Dict] = None,
        # GA specific parameters
        population_size: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        mutation_eta: float = 20.0,  # Distribution index for polynomial mutation
        crossover_eta: float = 20.0,  # Distribution index for SBX crossover
        selection_method: str = 'tournament',  # 'tournament', 'roulette'
        tournament_size: int = 3,
        elitism: int = 2,  # Number of best individuals to preserve
        # Stopping criteria
        n_stagnation: int = 20,  # Stop after N generations without improvement
        seed: Optional[int] = None,
        verbose: int = 1,
        init_strategy: str = 'midpoint'
    ):
        """
        Initialize Genetic Algorithm optimizer.
        
        Args:
            rom_model: ROMWithE2C model instance
            config: Configuration object with economic parameters
            norm_params: Normalization parameters dictionary
            device: PyTorch device
            max_iterations: Maximum generations
            tolerance: Convergence tolerance
            action_ranges: Optional well control bounds
            
            GA specific:
                population_size: Number of individuals in population
                crossover_prob: Probability of crossover (0-1)
                mutation_prob: Probability of mutation per gene (0-1)
                mutation_eta: Distribution index for polynomial mutation
                crossover_eta: Distribution index for SBX crossover
                selection_method: 'tournament' or 'roulette'
                tournament_size: Size of tournament for selection
                elitism: Number of best individuals to preserve each generation
            
            Stopping:
                n_stagnation: Stop if no improvement for N generations
                
            seed: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=summary)
            init_strategy: Initialization strategy ('midpoint', 'random', 'naive_zero', etc.)
        """
        super().__init__(rom_model, config, norm_params, device, action_ranges, init_strategy)
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta
        self.crossover_eta = crossover_eta
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.n_stagnation = n_stagnation
        self.seed = seed
        self.verbose = verbose
        
        # Storage for optimization history
        self.history = {
            'objectives': [],
            'best_objectives': [],
            'mean_objectives': [],
            'sampled_indices': []
        }
        
        # For RL-like sampling
        self.z0_ensemble = None
        self.total_samples_count = {}
        self.current_iteration = 0
    
    def _get_random_realization(self) -> Tuple[torch.Tensor, int]:
        """
        Get a single random realization from the Z0 ensemble.
        
        Returns:
            z0: Single initial state tensor (1, latent_dim)
            idx: Index of selected case
        """
        num_cases = self.z0_ensemble.shape[0]
        idx = np.random.randint(0, num_cases)
        z0 = self.z0_ensemble[idx:idx+1]
        
        # Track sampling statistics
        if idx not in self.total_samples_count:
            self.total_samples_count[idx] = 0
        self.total_samples_count[idx] += 1
        
        return z0, idx
    
    def _evaluate_individual(self, individual: np.ndarray) -> float:
        """
        Evaluate fitness for a single individual.
        
        Uses RL-like sampling: picks one random case per evaluation.
        
        Args:
            individual: Control vector in normalized [0,1] space
            
        Returns:
            Fitness value (negative objective for minimization)
        """
        # Get random realization (RL-like sampling)
        z0, idx = self._get_random_realization()
        
        # Reshape controls
        x = np.array(individual)
        controls = x.reshape(self.num_steps, self.num_controls)
        
        # Evaluate objective
        obj, _ = self.evaluate_objective(controls, z0)
        
        # Track sampling
        self.history['sampled_indices'].append(idx)
        
        # Return negative (GA minimizes by default in our implementation)
        return -obj
    
    def optimize(
        self,
        z0_options: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> OptimizationResult:
        """
        Run Genetic Algorithm optimization.
        
        Args:
            z0_options: Tensor of initial states (num_cases, latent_dim)
            num_steps: Number of control timesteps (default: 30)
            
        Returns:
            OptimizationResult with optimal controls and performance data
        """
        start_time = time.time()
        self.reset_counters()
        self.history = {
            'objectives': [],
            'best_objectives': [],
            'mean_objectives': [],
            'sampled_indices': []
        }
        self.total_samples_count = {}
        self.current_iteration = 0
        
        # Setup
        num_steps = num_steps or 30
        self.num_steps = num_steps
        
        if z0_options is None:
            raise ValueError("z0_options must be provided")
        
        self.z0_ensemble = z0_options
        actual_cases = z0_options.shape[0]
        num_vars = num_steps * self.num_controls
        
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Print header
        print(f"\n{'='*60}")
        print(f"Genetic Algorithm Optimization (RL-like Sampling)")
        print(f"{'='*60}")
        print(f"Cases available: {actual_cases}")
        print(f"Timesteps: {num_steps}")
        print(f"Control variables (genes): {num_vars}")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_iterations}")
        print(f"Crossover probability: {self.crossover_prob}")
        print(f"Mutation probability: {self.mutation_prob}")
        print(f"Elitism: {self.elitism}")
        print(f"\nPhysical Ranges:")
        print(f"  Producer BHP: [{self.action_ranges['producer_bhp']['min']:.2f}, {self.action_ranges['producer_bhp']['max']:.2f}] psi")
        print(f"  Gas Injection: [{self.action_ranges['gas_injection']['min']:.0f}, {self.action_ranges['gas_injection']['max']:.0f}] ft³/day")
        print(f"{'='*60}\n")
        
        # Evaluate initial objective using configured strategy
        x0 = self.generate_initial_guess(num_steps, strategy=self.init_strategy)
        z0_first = self.z0_ensemble[0:1]
        initial_obj, _ = self.evaluate_objective(x0.reshape(num_steps, self.num_controls), z0_first)
        print(f"Initial objective ({self.init_strategy}): {initial_obj:.6f}")
        
        # Run GA
        if DEAP_AVAILABLE:
            result = self._run_deap_ga(num_vars)
        else:
            result = self._run_builtin_ga(num_vars)
        
        best_individual, best_fitness = result
        
        # Extract optimal solution
        optimal_controls_normalized = np.array(best_individual).reshape(num_steps, self.num_controls)
        optimal_controls_physical = self.controls_normalized_to_physical(optimal_controls_normalized)
        initial_controls_physical = self.controls_normalized_to_physical(x0.reshape(num_steps, self.num_controls))
        
        # Final objective (convert from negative)
        final_obj = -best_fitness
        
        # Get trajectory for visualization
        _, trajectory = self.evaluate_objective(optimal_controls_normalized, z0_first, return_trajectory=True)
        
        # Decode spatial states
        spatial_states = self.decode_spatial_states(trajectory['states'])
        
        # Compute economic breakdown
        from .objective import compute_trajectory_npv
        observations_array = np.array(trajectory['observations'])
        _, economic_breakdown = compute_trajectory_npv(
            trajectory['observations'],
            optimal_controls_physical,
            self.config,
            self.num_prod,
            self.num_inj
        )
        
        total_time = time.time() - start_time
        
        # Build result
        optimization_result = OptimizationResult(
            optimal_controls=optimal_controls_physical,
            optimal_objective=final_obj,
            optimal_states=torch.stack(trajectory['states']),
            optimal_spatial_states=spatial_states,
            optimal_observations=observations_array,
            objective_history=self.history['best_objectives'],
            gradient_norm_history=[],  # No gradients for GA
            control_history=[],
            num_iterations=self.current_iteration,
            num_function_evaluations=self.function_eval_count,
            num_gradient_evaluations=0,
            total_time_seconds=total_time,
            convergence_achieved=True,
            termination_reason='Max generations reached' if self.current_iteration >= self.max_iterations else 'Converged',
            optimizer_type='Genetic Algorithm',
            optimizer_params={
                'population_size': self.population_size,
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob,
                'elitism': self.elitism,
                'max_iterations': self.max_iterations,
                'backend': 'DEAP' if DEAP_AVAILABLE else 'builtin'
            },
            num_realizations=actual_cases,
            initial_controls=initial_controls_physical,
            initial_objective=initial_obj,
            economic_breakdown=economic_breakdown
        )
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(optimization_result.summary())
        self._print_sampling_statistics()
        
        return optimization_result
    
    def _run_deap_ga(self, num_vars: int) -> Tuple[List, float]:
        """Run optimization using DEAP library."""
        print(f"\nStarting GA (DEAP backend)...\n")
        
        # Create DEAP types (handle if already created)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Gene: random float in [0, 1]
        toolbox.register("attr_float", np.random.uniform, 0, 1)
        
        # Individual: list of genes
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_float, n=num_vars)
        
        # Population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Fitness function
        def evaluate(individual):
            return (self._evaluate_individual(individual),)
        
        toolbox.register("evaluate", evaluate)
        
        # Genetic operators
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                        eta=self.crossover_eta, low=0.0, up=1.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        eta=self.mutation_eta, low=0.0, up=1.0, indpb=self.mutation_prob)
        
        if self.selection_method == 'tournament':
            toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        else:
            toolbox.register("select", tools.selRoulette)
        
        # Create initial population
        population = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track best
        best_fitness = min(f[0] for f in fitnesses)
        best_individual = tools.selBest(population, 1)[0]
        self.history['best_objectives'].append(-best_fitness)
        
        stagnation_count = 0
        
        for gen in range(self.max_iterations):
            self.current_iteration = gen + 1
            
            # Select offspring
            offspring = toolbox.select(population, len(population) - self.elitism)
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            
            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Elitism: add best individuals from previous generation
            elite = tools.selBest(population, self.elitism)
            offspring.extend(elite)
            
            # Replace population
            population[:] = offspring
            
            # Track statistics
            fits = [ind.fitness.values[0] for ind in population]
            current_best = min(fits)
            mean_fit = np.mean(fits)
            
            self.history['mean_objectives'].append(-mean_fit)
            
            # Update best
            if current_best < best_fitness - self.tolerance:
                best_fitness = current_best
                best_individual = tools.selBest(population, 1)[0]
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            self.history['best_objectives'].append(-best_fitness)
            
            # Progress reporting
            if self.verbose > 0 and (gen < 5 or (gen + 1) % 10 == 0):
                print(f"Generation {gen + 1:4d}: Best = {-best_fitness:.6f}, Mean = {-mean_fit:.6f}")
                print(f"   Unique cases sampled: {len(self.total_samples_count)}/{self.z0_ensemble.shape[0]}")
            
            # Check stagnation
            if stagnation_count >= self.n_stagnation:
                print(f"\nStopped: No improvement for {self.n_stagnation} generations")
                break
        
        return list(best_individual), best_fitness
    
    def _run_builtin_ga(self, num_vars: int) -> Tuple[np.ndarray, float]:
        """Run optimization using built-in GA implementation."""
        print(f"\nStarting GA (built-in implementation)...\n")
        
        # Initialize population
        population = np.random.uniform(0, 1, (self.population_size, num_vars))
        
        # Evaluate initial population
        fitness = np.array([self._evaluate_individual(ind) for ind in population])
        
        # Track best
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        self.history['best_objectives'].append(-best_fitness)
        
        stagnation_count = 0
        
        for gen in range(self.max_iterations):
            self.current_iteration = gen + 1
            
            # Selection (tournament)
            selected_indices = []
            for _ in range(self.population_size - self.elitism):
                tournament = np.random.choice(self.population_size, self.tournament_size, replace=False)
                winner = tournament[np.argmin(fitness[tournament])]
                selected_indices.append(winner)
            
            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected_indices) - 1, 2):
                parent1 = population[selected_indices[i]].copy()
                parent2 = population[selected_indices[i + 1]].copy()
                
                # SBX Crossover
                if np.random.random() < self.crossover_prob:
                    child1, child2 = self._sbx_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Polynomial mutation
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Handle odd number
            if len(selected_indices) % 2 == 1:
                offspring.append(population[selected_indices[-1]].copy())
            
            offspring = np.array(offspring[:self.population_size - self.elitism])
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness)[:self.elitism]
            elite = population[elite_indices]
            
            # Combine offspring and elite
            population = np.vstack([offspring, elite])
            
            # Evaluate new population
            fitness = np.array([self._evaluate_individual(ind) for ind in population])
            
            # Track statistics
            current_best = np.min(fitness)
            mean_fit = np.mean(fitness)
            
            self.history['mean_objectives'].append(-mean_fit)
            
            # Update best
            if current_best < best_fitness - self.tolerance:
                best_idx = np.argmin(fitness)
                best_fitness = current_best
                best_individual = population[best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            self.history['best_objectives'].append(-best_fitness)
            
            # Progress reporting
            if self.verbose > 0 and (gen < 5 or (gen + 1) % 10 == 0):
                print(f"Generation {gen + 1:4d}: Best = {-best_fitness:.6f}, Mean = {-mean_fit:.6f}")
                print(f"   Unique cases sampled: {len(self.total_samples_count)}/{self.z0_ensemble.shape[0]}")
            
            # Check stagnation
            if stagnation_count >= self.n_stagnation:
                print(f"\nStopped: No improvement for {self.n_stagnation} generations")
                break
        
        return best_individual, best_fitness
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    # Calculate beta
                    rand = np.random.random()
                    beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                    alpha = 2.0 - beta ** -(self.crossover_eta + 1)
                    
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (self.crossover_eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.crossover_eta + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    
                    child1[i] = np.clip(c1, 0, 1)
                    child2[i] = np.clip(c2, 0, 1)
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        mutant = individual.copy()
        
        for i in range(len(mutant)):
            if np.random.random() < self.mutation_prob:
                y = mutant[i]
                delta1 = y - 0.0
                delta2 = 1.0 - y
                
                rand = np.random.random()
                mut_pow = 1.0 / (self.mutation_eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.mutation_eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.mutation_eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                mutant[i] = np.clip(y + deltaq, 0, 1)
        
        return mutant
    
    def _print_sampling_statistics(self):
        """Print RL-like sampling statistics."""
        print(f"\n{'='*60}")
        print(f"Sampling Statistics (RL-like)")
        print(f"{'='*60}")
        print(f"Total cases available: {self.z0_ensemble.shape[0]}")
        print(f"Unique cases sampled: {len(self.total_samples_count)}")
        print(f"Total samples drawn: {sum(self.total_samples_count.values())}")
        
        if self.total_samples_count:
            counts = list(self.total_samples_count.values())
            print(f"Coverage: {100 * len(self.total_samples_count) / self.z0_ensemble.shape[0]:.1f}%")
            print(f"Samples per case: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.2f}")
        print(f"{'='*60}\n")


def check_deap_available():
    """Check if DEAP library is available."""
    return DEAP_AVAILABLE
