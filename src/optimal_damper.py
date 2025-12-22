"""
Optimal Damper Design Module.
Optimization tools for damper placement and sizing.
"""
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MIN_DRIFT = "minimize_drift"
    MIN_ACCELERATION = "minimize_acceleration"
    MIN_ENERGY = "minimize_energy"
    MIN_COST = "minimize_cost"


@dataclass
class DamperConfig:
    """Damper configuration for optimization."""
    story: int
    damping_coefficient: float  # Ns/m
    cost: float


@dataclass
class OptimizationResult:
    """Result of damper optimization."""
    optimal_config: List[DamperConfig]
    objective_value: float
    total_cost: float
    response_reduction: float
    iterations: int
    convergence_history: List[float]


def optimize_damper_placement_simple(
    n_stories: int,
    story_stiffness: List[float],
    story_mass: List[float],
    target_damping_ratio: float = 0.10,
    max_dampers: int = 3,
    damper_cost_per_unit: float = 100000
) -> OptimizationResult:
    """
    Simple optimal damper placement using stiffness proportional distribution.
    
    Places dampers in stories with highest drift participation.
    
    Args:
        n_stories: Number of stories
        story_stiffness: Stiffness of each story
        story_mass: Mass of each story
        target_damping_ratio: Target additional damping
        max_dampers: Maximum number of damper locations
        damper_cost_per_unit: Cost per unit damping coefficient
        
    Returns:
        OptimizationResult
    """
    # Calculate mode shape (simplified first mode)
    phi = np.zeros(n_stories)
    for i in range(n_stories):
        phi[i] = (i + 1) / n_stories
        
    # Calculate story drift participation
    drift_phi = np.diff(np.insert(phi, 0, 0))
    
    # Contribution of each story to overall damping
    contributions = drift_phi**2 * np.array(story_stiffness[:n_stories])
    
    # Select top stories for damper placement
    sorted_stories = np.argsort(contributions)[::-1][:max_dampers]
    
    # Calculate required damping coefficient
    total_mass = sum(story_mass)
    omega = np.sqrt(sum(story_stiffness) / total_mass)  # Approximate first mode
    
    total_c_required = 2 * target_damping_ratio * omega * total_mass
    
    # Distribute among selected stories proportionally
    selected_contributions = contributions[sorted_stories]
    total_selected = sum(selected_contributions)
    
    configs = []
    total_cost = 0
    
    for i, story in enumerate(sorted_stories):
        ratio = selected_contributions[i] / total_selected if total_selected > 0 else 1/max_dampers
        c = total_c_required * ratio
        cost = c * damper_cost_per_unit / 1e6  # Per MNs/m
        
        configs.append(DamperConfig(
            story=int(story),
            damping_coefficient=c,
            cost=cost
        ))
        total_cost += cost
        
    return OptimizationResult(
        optimal_config=configs,
        objective_value=target_damping_ratio,
        total_cost=total_cost,
        response_reduction=target_damping_ratio * 2,  # Rough estimate
        iterations=1,
        convergence_history=[target_damping_ratio]
    )


def optimize_damper_genetic(
    evaluate_function: Callable,
    n_stories: int,
    c_range: Tuple[float, float] = (1e5, 1e7),
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    objective: OptimizationObjective = OptimizationObjective.MIN_DRIFT
) -> OptimizationResult:
    """
    Genetic algorithm optimization for damper design.
    
    Args:
        evaluate_function: Function(damper_coefficients) -> response_dict
        n_stories: Number of stories (damper locations)
        c_range: Range of damping coefficients
        population_size: GA population size
        generations: Number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        objective: Optimization objective
        
    Returns:
        OptimizationResult
    """
    c_min, c_max = c_range
    
    # Initialize population
    population = np.random.uniform(0, 1, (population_size, n_stories))
    population *= (c_max - c_min)
    population += c_min
    
    # Add zeros (no damper option)
    for i in range(population_size):
        n_zeros = np.random.randint(0, n_stories // 2)
        zero_idx = np.random.choice(n_stories, n_zeros, replace=False)
        population[i, zero_idx] = 0
        
    best_fitness = float('inf')
    best_individual = None
    convergence = []
    
    for gen in range(generations):
        # Evaluate fitness
        fitness = np.zeros(population_size)
        
        for i in range(population_size):
            try:
                response = evaluate_function(population[i])
                
                if objective == OptimizationObjective.MIN_DRIFT:
                    fitness[i] = response.get('max_drift', 1.0)
                elif objective == OptimizationObjective.MIN_ACCELERATION:
                    fitness[i] = response.get('max_accel', 1.0)
                elif objective == OptimizationObjective.MIN_COST:
                    fitness[i] = np.sum(population[i]) / c_max
                else:
                    fitness[i] = response.get('max_drift', 1.0)
            except:
                fitness[i] = float('inf')
                
        # Update best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_individual = population[min_idx].copy()
            
        convergence.append(best_fitness)
        
        # Selection (tournament)
        new_population = np.zeros_like(population)
        
        for i in range(population_size):
            # Tournament selection
            t1, t2 = np.random.randint(0, population_size, 2)
            parent1 = population[t1] if fitness[t1] < fitness[t2] else population[t2]
            
            t1, t2 = np.random.randint(0, population_size, 2)
            parent2 = population[t1] if fitness[t1] < fitness[t2] else population[t2]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                cross_point = np.random.randint(1, n_stories)
                child = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
            else:
                child = parent1.copy()
                
            # Mutation
            for j in range(n_stories):
                if np.random.rand() < mutation_rate:
                    if np.random.rand() < 0.3:
                        child[j] = 0  # Remove damper
                    else:
                        child[j] = np.random.uniform(c_min, c_max)
                        
            new_population[i] = child
            
        # Elitism
        new_population[0] = best_individual
        population = new_population
        
    # Create result
    configs = []
    for i, c in enumerate(best_individual):
        if c > 0:
            configs.append(DamperConfig(
                story=i,
                damping_coefficient=c,
                cost=c / 1e6  # Simplified cost
            ))
            
    return OptimizationResult(
        optimal_config=configs,
        objective_value=best_fitness,
        total_cost=sum(c.cost for c in configs),
        response_reduction=1 - best_fitness if best_fitness < 1 else 0,
        iterations=generations,
        convergence_history=convergence
    )


def optimize_tmd_parameters(
    structure_mass: float,
    structure_frequency: float,
    mass_ratio: float = 0.02
) -> Dict[str, float]:
    """
    Optimize TMD parameters using Den Hartog's formulas.
    
    Args:
        structure_mass: Main structure mass (kg)
        structure_frequency: Structure natural frequency (Hz)
        mass_ratio: TMD mass / structure mass
        
    Returns:
        Optimal TMD parameters
    """
    mu = mass_ratio
    
    # Optimal frequency ratio
    f_opt = 1 / (1 + mu)
    
    # Optimal damping ratio
    zeta_opt = np.sqrt(3 * mu / (8 * (1 + mu)**3))
    
    # TMD parameters
    m_tmd = mu * structure_mass
    omega_s = 2 * np.pi * structure_frequency
    omega_tmd = f_opt * omega_s
    k_tmd = m_tmd * omega_tmd**2
    c_tmd = 2 * zeta_opt * m_tmd * omega_tmd
    
    # Effectiveness
    H_max_reduction = np.sqrt((1 + mu) / (1 + mu / 2))  # Approximate
    
    return {
        'mass': m_tmd,
        'stiffness': k_tmd,
        'damping': c_tmd,
        'frequency_ratio': f_opt,
        'damping_ratio': zeta_opt,
        'amplitude_reduction': 1 / H_max_reduction,
        'mass_ratio': mu
    }


def plot_optimization_convergence(result: OptimizationResult, ax=None):
    """Plot GA convergence history."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    ax.plot(result.convergence_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Objective Value')
    ax.set_title('Optimization Convergence')
    ax.grid(True, alpha=0.3)
    
    # Mark best
    best_gen = np.argmin(result.convergence_history)
    ax.plot(best_gen, result.objective_value, 'ro', markersize=10, label='Best')
    ax.legend()
    
    return ax


def plot_damper_distribution(result: OptimizationResult, n_stories: int, ax=None):
    """Plot optimized damper distribution."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
        
    c_by_story = np.zeros(n_stories)
    for config in result.optimal_config:
        if config.story < n_stories:
            c_by_story[config.story] = config.damping_coefficient
            
    ax.barh(range(1, n_stories + 1), c_by_story / 1e6, color='steelblue')
    ax.set_ylabel('Story')
    ax.set_xlabel('Damping Coefficient (MN·s/m)')
    ax.set_title('Optimized Damper Distribution')
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax
