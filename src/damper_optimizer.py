"""
Optimal Damper Design Tool.
Provides optimization of viscous damper placement and properties.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ObjectiveType(Enum):
    """Optimization objective types."""
    MIN_DISPLACEMENT = "displacement"
    MIN_DRIFT = "drift"
    MIN_ACCELERATION = "acceleration"
    MIN_RESIDUAL = "residual"


@dataclass
class DamperConfig:
    """Configuration for a single damper."""
    floor: int
    location: str  # 'X' or 'Y' direction
    damping_coeff: float  # C (N·s/m)
    cost: float  # Relative cost


@dataclass
class DesignResult:
    """Result of damper design optimization."""
    damper_configs: List[DamperConfig]
    total_cost: float
    max_drift_reduction: float  # Reduction ratio
    max_disp_reduction: float
    iterations: int
    converged: bool


class DamperOptimizer:
    """
    Optimal damper placement and sizing.
    
    Uses gradient-descent or genetic algorithm approaches
    for optimizing damper distribution.
    """
    
    def __init__(
        self,
        n_floors: int,
        story_heights: List[float],
        base_response: Dict[str, float],  # Response without dampers
        response_calculator: Optional[Callable] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            n_floors: Number of floors
            story_heights: List of story heights
            base_response: Dict with 'max_drift', 'max_disp', 'max_acc'
            response_calculator: Function to calculate response given dampers
        """
        self.n_floors = n_floors
        self.story_heights = story_heights
        self.base_response = base_response
        self.response_calculator = response_calculator
        
        # Default cost model
        self.cost_per_unit_c = 1.0  # Cost per unit damping coefficient
        
    def optimize_uniform(
        self,
        target_reduction: float = 0.3,
        budget: Optional[float] = None
    ) -> DesignResult:
        """
        Uniform damper distribution (simplest approach).
        
        Places equal dampers on each floor.
        
        Args:
            target_reduction: Target response reduction ratio
            budget: Optional total budget constraint
            
        Returns:
            DesignResult
        """
        # Simple approach: estimate required damping from target
        # Approximate: C_total ≈ base_C * ln(1/(1-reduction))
        
        base_drift = self.base_response.get('max_drift', 0.01)
        
        # Rough estimate of damping coefficient needed per floor
        c_per_floor = self._estimate_damping_for_reduction(target_reduction)
        
        configs = []
        total_cost = 0
        
        for floor in range(1, self.n_floors + 1):
            # X direction damper
            config = DamperConfig(
                floor=floor,
                location='X',
                damping_coeff=c_per_floor,
                cost=c_per_floor * self.cost_per_unit_c
            )
            configs.append(config)
            total_cost += config.cost
            
            # Y direction damper
            config_y = DamperConfig(
                floor=floor,
                location='Y',
                damping_coeff=c_per_floor,
                cost=c_per_floor * self.cost_per_unit_c
            )
            configs.append(config_y)
            total_cost += config_y.cost
        
        # Apply budget constraint if specified
        if budget and total_cost > budget:
            scale = budget / total_cost
            for cfg in configs:
                cfg.damping_coeff *= scale
                cfg.cost *= scale
            total_cost = budget
        
        return DesignResult(
            damper_configs=configs,
            total_cost=total_cost,
            max_drift_reduction=target_reduction,
            max_disp_reduction=target_reduction * 0.9,
            iterations=1,
            converged=True
        )
    
    def optimize_proportional_to_drift(
        self,
        drift_profile: List[float],
        target_reduction: float = 0.3,
        budget: Optional[float] = None
    ) -> DesignResult:
        """
        Proportional damper distribution based on drift profile.
        
        Places more damping in floors with higher drift.
        
        Args:
            drift_profile: List of inter-story drift for each floor
            target_reduction: Target response reduction
            budget: Optional budget constraint
            
        Returns:
            DesignResult
        """
        # Normalize drift to get distribution
        total_drift = sum(drift_profile) if drift_profile else 1.0
        weights = [d / total_drift for d in drift_profile] if total_drift > 0 else [1/len(drift_profile)] * len(drift_profile)
        
        # Total damping needed
        total_c = self._estimate_total_damping(target_reduction)
        
        configs = []
        total_cost = 0
        
        for floor in range(1, self.n_floors + 1):
            if floor - 1 < len(weights):
                weight = weights[floor - 1]
            else:
                weight = 1.0 / self.n_floors
                
            # Damping for this floor
            c_floor = total_c * weight
            
            config = DamperConfig(
                floor=floor,
                location='X',
                damping_coeff=c_floor,
                cost=c_floor * self.cost_per_unit_c
            )
            configs.append(config)
            total_cost += config.cost
        
        # Budget constraint
        if budget and total_cost > budget:
            scale = budget / total_cost
            for cfg in configs:
                cfg.damping_coeff *= scale
                cfg.cost *= scale
            total_cost = budget
        
        return DesignResult(
            damper_configs=configs,
            total_cost=total_cost,
            max_drift_reduction=target_reduction,
            max_disp_reduction=target_reduction * 0.85,
            iterations=len(drift_profile),
            converged=True
        )
    
    def optimize_genetic(
        self,
        objective: ObjectiveType = ObjectiveType.MIN_DRIFT,
        budget: float = None,
        population_size: int = 20,
        generations: int = 50,
        c_min: float = 1e5,
        c_max: float = 1e8
    ) -> DesignResult:
        """
        Genetic algorithm optimization for damper placement.
        
        Args:
            objective: Optimization objective
            budget: Budget constraint
            population_size: GA population size
            generations: Number of generations
            c_min: Minimum damping coefficient
            c_max: Maximum damping coefficient
            
        Returns:
            DesignResult
        """
        np.random.seed(42)  # Reproducibility
        
        n_dampers = self.n_floors * 2  # X and Y per floor
        
        # Initialize population (damping coefficients for each location)
        population = np.random.uniform(c_min, c_max, (population_size, n_dampers))
        
        best_solution = None
        best_fitness = float('inf')
        
        for gen in range(generations):
            # Evaluate fitness
            fitness = np.zeros(population_size)
            for i in range(population_size):
                fitness[i] = self._evaluate_fitness(
                    population[i], objective, budget, c_max
                )
                
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_solution = population[i].copy()
            
            # Selection (tournament)
            new_population = np.zeros_like(population)
            for i in range(population_size):
                # Tournament of 3
                idx = np.random.choice(population_size, 3, replace=False)
                winner_idx = idx[np.argmin(fitness[idx])]
                new_population[i] = population[winner_idx].copy()
            
            # Crossover
            for i in range(0, population_size - 1, 2):
                if np.random.random() < 0.8:  # Crossover probability
                    point = np.random.randint(1, n_dampers)
                    new_population[i, point:], new_population[i+1, point:] = \
                        new_population[i+1, point:].copy(), new_population[i, point:].copy()
            
            # Mutation
            for i in range(population_size):
                for j in range(n_dampers):
                    if np.random.random() < 0.1:  # Mutation probability
                        new_population[i, j] *= np.random.uniform(0.5, 2.0)
                        new_population[i, j] = np.clip(new_population[i, j], c_min, c_max)
            
            population = new_population
        
        # Convert best solution to DamperConfig
        configs = []
        total_cost = 0
        
        for i, c in enumerate(best_solution):
            floor = (i // 2) + 1
            direction = 'X' if i % 2 == 0 else 'Y'
            
            config = DamperConfig(
                floor=floor,
                location=direction,
                damping_coeff=c,
                cost=c * self.cost_per_unit_c
            )
            configs.append(config)
            total_cost += config.cost
        
        # Estimate reduction
        estimated_reduction = 1.0 - best_fitness if best_fitness < 1 else 0.0
        
        return DesignResult(
            damper_configs=configs,
            total_cost=total_cost,
            max_drift_reduction=estimated_reduction,
            max_disp_reduction=estimated_reduction * 0.9,
            iterations=generations,
            converged=True
        )
    
    def _estimate_damping_for_reduction(self, reduction: float) -> float:
        """Estimate damping coefficient needed for target reduction."""
        # Rough approximation: C ≈ 2 * zeta_add * M * omega
        # For 30% reduction, zeta_add ≈ 0.15
        zeta_add = reduction * 0.5
        
        # Assume average floor mass and frequency
        avg_mass = 50000  # kg (placeholder)
        omega = 10.0  # rad/s (placeholder)
        
        return 2 * zeta_add * avg_mass * omega
    
    def _estimate_total_damping(self, reduction: float) -> float:
        """Estimate total damping needed."""
        return self._estimate_damping_for_reduction(reduction) * self.n_floors
    
    def _evaluate_fitness(
        self,
        damper_coeffs: np.ndarray,
        objective: ObjectiveType,
        budget: Optional[float],
        c_max: float
    ) -> float:
        """Evaluate fitness of a damper configuration."""
        total_cost = np.sum(damper_coeffs) * self.cost_per_unit_c
        
        # Budget penalty
        if budget and total_cost > budget:
            penalty = (total_cost - budget) / budget * 10
        else:
            penalty = 0
        
        # Response estimate (simplified model)
        # Real implementation would call response_calculator
        total_c = np.sum(damper_coeffs)
        c_ratio = total_c / (c_max * len(damper_coeffs))
        
        # Estimated response reduction (diminishing returns)
        reduction = 1 - np.exp(-c_ratio * 3)
        
        base_resp = self.base_response.get('max_drift', 0.01)
        estimated_resp = base_resp * (1 - reduction)
        
        return estimated_resp + penalty


def summarize_design(result: DesignResult) -> str:
    """Generate text summary of damper design."""
    lines = ["=" * 50]
    lines.append("OPTIMAL DAMPER DESIGN SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Total Cost: {result.total_cost:,.0f}")
    lines.append(f"Drift Reduction: {result.max_drift_reduction * 100:.1f}%")
    lines.append(f"Optimization Iterations: {result.iterations}")
    lines.append("")
    lines.append("Damper Configuration:")
    lines.append("-" * 40)
    
    for cfg in result.damper_configs:
        lines.append(f"  Floor {cfg.floor}{cfg.location}: C = {cfg.damping_coeff:.2e} N·s/m")
    
    lines.append("=" * 50)
    return "\n".join(lines)
