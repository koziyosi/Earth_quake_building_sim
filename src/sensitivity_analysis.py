"""
Sensitivity Analysis Module.
Parameter variation and response sensitivity studies.
"""
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import itertools


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    parameter: str
    baseline_value: float
    values: np.ndarray
    responses: np.ndarray
    sensitivity: float  # dResponse/dParameter (normalized)
    elasticity: float   # (dR/R)/(dP/P)


def run_one_at_a_time_sensitivity(
    run_function: Callable,
    baseline_params: Dict[str, float],
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 5,
    response_key: str = 'max_drift'
) -> Dict[str, SensitivityResult]:
    """
    One-at-a-time sensitivity analysis.
    
    Varies each parameter independently while holding others constant.
    
    Args:
        run_function: Function that takes params dict and returns results dict
        baseline_params: Baseline parameter values
        param_ranges: (min, max) for each parameter
        n_samples: Number of samples per parameter
        response_key: Key in results to use as response
        
    Returns:
        Dictionary of SensitivityResult for each parameter
    """
    results = {}
    
    # Get baseline response
    baseline_result = run_function(baseline_params)
    baseline_response = baseline_result.get(response_key, 0)
    
    for param, (p_min, p_max) in param_ranges.items():
        baseline_value = baseline_params.get(param, (p_min + p_max) / 2)
        values = np.linspace(p_min, p_max, n_samples)
        responses = []
        
        for val in values:
            test_params = baseline_params.copy()
            test_params[param] = val
            
            try:
                result = run_function(test_params)
                responses.append(result.get(response_key, 0))
            except Exception:
                responses.append(np.nan)
                
        responses = np.array(responses)
        
        # Calculate sensitivity (slope at baseline)
        valid = ~np.isnan(responses)
        if np.sum(valid) >= 2:
            sensitivity = np.polyfit(values[valid], responses[valid], 1)[0]
        else:
            sensitivity = 0
            
        # Calculate elasticity
        if baseline_value != 0 and baseline_response != 0:
            elasticity = (sensitivity * baseline_value) / baseline_response
        else:
            elasticity = 0
            
        results[param] = SensitivityResult(
            parameter=param,
            baseline_value=baseline_value,
            values=values,
            responses=responses,
            sensitivity=sensitivity,
            elasticity=elasticity
        )
        
    return results


def run_morris_screening(
    run_function: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    n_trajectories: int = 10,
    n_levels: int = 4,
    response_key: str = 'max_drift'
) -> Dict[str, Tuple[float, float]]:
    """
    Morris method for global sensitivity screening.
    
    Returns (mu*, sigma) for each parameter, where:
    - mu*: Mean of absolute elementary effects (importance)
    - sigma: Standard deviation (non-linearity/interaction)
    
    Args:
        run_function: Simulation function
        param_ranges: Parameter bounds
        n_trajectories: Number of random trajectories
        n_levels: Number of grid levels
        response_key: Response variable key
        
    Returns:
        Dict of (mu*, sigma) tuples
    """
    params = list(param_ranges.keys())
    k = len(params)
    delta = 1 / (n_levels - 1)
    
    effects = {p: [] for p in params}
    
    for _ in range(n_trajectories):
        # Generate random starting point
        x_base = {p: np.random.choice(np.linspace(r[0], r[1], n_levels)) 
                  for p, r in param_ranges.items()}
        
        # Random order of parameters
        order = np.random.permutation(k)
        
        x_current = x_base.copy()
        try:
            y_current = run_function(x_current).get(response_key, 0)
        except:
            continue
            
        for idx in order:
            param = params[idx]
            p_min, p_max = param_ranges[param]
            
            # Step in parameter space
            x_next = x_current.copy()
            step = delta * (p_max - p_min) * (1 if np.random.rand() > 0.5 else -1)
            x_next[param] = np.clip(x_current[param] + step, p_min, p_max)
            
            try:
                y_next = run_function(x_next).get(response_key, 0)
                
                # Elementary effect
                if step != 0:
                    ee = (y_next - y_current) / step * (p_max - p_min)
                    effects[param].append(ee)
                    
                x_current = x_next
                y_current = y_next
            except:
                pass
                
    # Calculate statistics
    results = {}
    for p in params:
        if effects[p]:
            mu_star = np.mean(np.abs(effects[p]))
            sigma = np.std(effects[p])
            results[p] = (mu_star, sigma)
        else:
            results[p] = (0, 0)
            
    return results


def run_factorial_design(
    run_function: Callable,
    factors: Dict[str, List[float]],
    response_key: str = 'max_drift',
    parallel: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Full factorial design experiment.
    
    Args:
        run_function: Simulation function
        factors: Parameter levels {param: [level1, level2, ...]}
        response_key: Response variable key
        parallel: Run in parallel
        
    Returns:
        (response_array, design_info)
    """
    param_names = list(factors.keys())
    levels = [factors[p] for p in param_names]
    
    # Generate all combinations
    combinations = list(itertools.product(*levels))
    n_runs = len(combinations)
    
    responses = np.zeros(n_runs)
    
    def run_single(idx_combo):
        idx, combo = idx_combo
        params = {param_names[j]: combo[j] for j in range(len(param_names))}
        try:
            result = run_function(params)
            return result.get(response_key, np.nan)
        except:
            return np.nan
            
    if parallel:
        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(run_single, enumerate(combinations)))
    else:
        for idx, combo in enumerate(combinations):
            responses[idx] = run_single((idx, combo))
            
    responses = np.array(responses)
    
    design_info = {
        'param_names': param_names,
        'combinations': combinations,
        'n_runs': n_runs,
        'factors': factors
    }
    
    return responses, design_info


def calculate_main_effects(
    responses: np.ndarray,
    design_info: Dict
) -> Dict[str, float]:
    """
    Calculate main effects from factorial design.
    
    Main effect = mean(high) - mean(low)
    """
    param_names = design_info['param_names']
    factors = design_info['factors']
    combinations = design_info['combinations']
    
    effects = {}
    
    for i, param in enumerate(param_names):
        levels = factors[param]
        
        if len(levels) == 2:
            low_val, high_val = levels[0], levels[1]
            
            low_responses = [responses[j] for j, c in enumerate(combinations) if c[i] == low_val]
            high_responses = [responses[j] for j, c in enumerate(combinations) if c[i] == high_val]
            
            effect = np.nanmean(high_responses) - np.nanmean(low_responses)
            effects[param] = effect
        else:
            # Multi-level: use range
            level_means = []
            for level in levels:
                level_resps = [responses[j] for j, c in enumerate(combinations) if c[i] == level]
                level_means.append(np.nanmean(level_resps))
            effects[param] = max(level_means) - min(level_means)
            
    return effects


def plot_sensitivity_tornado(
    sensitivities: Dict[str, SensitivityResult],
    ax = None
):
    """Create tornado diagram of sensitivities."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    params = list(sensitivities.keys())
    elasticities = [sensitivities[p].elasticity for p in params]
    
    # Sort by absolute elasticity
    sorted_idx = np.argsort(np.abs(elasticities))[::-1]
    params = [params[i] for i in sorted_idx]
    elasticities = [elasticities[i] for i in sorted_idx]
    
    colors = ['red' if e > 0 else 'blue' for e in elasticities]
    
    ax.barh(range(len(params)), elasticities, color=colors, alpha=0.7)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Elasticity (normalized sensitivity)')
    ax.set_title('Parameter Sensitivity - Tornado Diagram')
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def plot_spider_chart(
    sensitivities: Dict[str, SensitivityResult],
    baseline_response: float,
    ax = None
):
    """Create spider chart showing parameter variations."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    for param, result in sensitivities.items():
        # Normalize to baseline
        norm_values = result.values / result.baseline_value * 100 - 100
        norm_responses = result.responses / baseline_response * 100 - 100
        
        ax.plot(norm_values, norm_responses, '-o', label=param, markersize=4)
        
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Parameter Change (%)')
    ax.set_ylabel('Response Change (%)')
    ax.set_title('Spider Chart - Sensitivity Analysis')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax
