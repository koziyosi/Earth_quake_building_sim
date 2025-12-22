"""
Advanced Damage Assessment Module.
Implements damage indices and fragility curves.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DamageState(Enum):
    """Structural damage states."""
    NONE = 0
    SLIGHT = 1
    MODERATE = 2
    EXTENSIVE = 3
    COMPLETE = 4


@dataclass
class DamageResult:
    """Damage assessment result."""
    park_ang_index: float
    damage_state: DamageState
    residual_drift: float
    cumulative_ductility: float
    energy_ratio: float


def calculate_park_ang_damage(
    max_disp: float,
    ultimate_disp: float,
    cumulative_energy: float,
    yield_force: float,
    ultimate_disp_monotonic: float,
    beta: float = 0.15
) -> float:
    """
    Calculate Park-Ang damage index.
    
    D = δm/δu + β * ∫dE / (Fy * δu)
    
    Args:
        max_disp: Maximum displacement reached
        ultimate_disp: Ultimate displacement capacity
        cumulative_energy: Cumulative hysteretic energy
        yield_force: Yield force
        ultimate_disp_monotonic: Ultimate displacement under monotonic loading
        beta: Energy factor (typically 0.05-0.20)
        
    Returns:
        Park-Ang damage index (0=none, 1=failure)
    """
    if ultimate_disp <= 0 or yield_force <= 0:
        return 0.0
        
    disp_term = max_disp / ultimate_disp
    energy_term = beta * cumulative_energy / (yield_force * ultimate_disp_monotonic)
    
    return disp_term + energy_term


def calculate_modified_park_ang(
    max_disp: float,
    yield_disp: float,
    ultimate_ductility: float,
    cumulative_energy: float,
    yield_force: float,
    beta: float = 0.15
) -> float:
    """
    Modified Park-Ang index using ductility.
    
    D = (μ - 1) / (μu - 1) + β * Eh / (Fy * δy * μu)
    """
    if yield_disp <= 0 or ultimate_ductility <= 1:
        return 0.0
        
    mu = max_disp / yield_disp
    
    disp_term = (mu - 1) / (ultimate_ductility - 1)
    energy_term = beta * cumulative_energy / (yield_force * yield_disp * ultimate_ductility)
    
    return max(0, disp_term + energy_term)


def get_damage_state(damage_index: float) -> DamageState:
    """Convert damage index to damage state."""
    if damage_index < 0.1:
        return DamageState.NONE
    elif damage_index < 0.25:
        return DamageState.SLIGHT
    elif damage_index < 0.50:
        return DamageState.MODERATE
    elif damage_index < 0.75:
        return DamageState.EXTENSIVE
    else:
        return DamageState.COMPLETE


def calculate_residual_drift(
    disp_history: np.ndarray,
    story_height: float
) -> float:
    """
    Calculate residual (permanent) inter-story drift.
    
    Args:
        disp_history: Displacement time history
        story_height: Story height (m)
        
    Returns:
        Residual drift ratio
    """
    if len(disp_history) < 2 or story_height <= 0:
        return 0.0
        
    # Use last 10% of record to estimate residual
    n_avg = max(1, len(disp_history) // 10)
    residual_disp = np.mean(disp_history[-n_avg:])
    
    return abs(residual_disp) / story_height


def calculate_cumulative_ductility(
    disp_history: np.ndarray,
    yield_disp: float
) -> float:
    """
    Calculate cumulative plastic ductility demand.
    
    μc = Σ|Δμp| (sum of plastic excursions)
    """
    if yield_disp <= 0 or len(disp_history) < 2:
        return 0.0
        
    plastic_disp = np.maximum(np.abs(disp_history) - yield_disp, 0)
    cumulative = np.sum(np.abs(np.diff(plastic_disp)))
    
    return cumulative / yield_disp


# ===== Fragility Curves =====

@dataclass
class FragilityParameters:
    """Fragility curve parameters (lognormal)."""
    median: float  # Median intensity measure
    beta: float    # Standard deviation of ln(IM)


def lognormal_fragility(
    im: np.ndarray,
    median: float,
    beta: float
) -> np.ndarray:
    """
    Calculate fragility probability using lognormal CDF.
    
    P(DS|IM) = Φ[(ln(IM) - ln(median)) / β]
    """
    from scipy import stats
    return stats.norm.cdf(np.log(im / median) / beta)


def fit_fragility_from_ida(
    im_values: np.ndarray,
    collapse_flags: np.ndarray,
    method: str = 'mle'
) -> FragilityParameters:
    """
    Fit fragility curve from IDA results.
    
    Args:
        im_values: Intensity measure values (e.g., PGA, Sa)
        collapse_flags: 1 if collapse, 0 if survived
        method: 'mle' or 'least_squares'
        
    Returns:
        FragilityParameters
    """
    # Maximum likelihood estimation for lognormal
    collapsed = im_values[collapse_flags == 1]
    survived = im_values[collapse_flags == 0]
    
    if len(collapsed) == 0:
        return FragilityParameters(median=np.max(im_values)*10, beta=0.5)
    if len(survived) == 0:
        return FragilityParameters(median=np.min(im_values)*0.1, beta=0.5)
    
    # Simple moment estimator
    ln_im = np.log(collapsed)
    median = np.exp(np.mean(ln_im))
    beta = np.std(ln_im)
    
    beta = max(0.2, min(beta, 1.0))  # Bound beta
    
    return FragilityParameters(median=median, beta=beta)


def generate_fragility_curves(
    ida_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    damage_thresholds: Dict[DamageState, float]
) -> Dict[DamageState, FragilityParameters]:
    """
    Generate fragility curves for each damage state.
    
    Args:
        ida_results: Dict of (im_array, edp_array) for each run
        damage_thresholds: EDP threshold for each damage state
        
    Returns:
        Fragility parameters for each damage state
    """
    fragilities = {}
    
    for ds, threshold in damage_thresholds.items():
        im_all = []
        exceed_flags = []
        
        for run_id, (im_arr, edp_arr) in ida_results.items():
            for im, edp in zip(im_arr, edp_arr):
                im_all.append(im)
                exceed_flags.append(1 if edp >= threshold else 0)
                
        im_all = np.array(im_all)
        exceed_flags = np.array(exceed_flags)
        
        fragilities[ds] = fit_fragility_from_ida(im_all, exceed_flags)
        
    return fragilities


def plot_fragility_curves(
    fragilities: Dict[DamageState, FragilityParameters],
    im_range: Tuple[float, float] = (0.01, 2.0),
    ax = None
):
    """Plot fragility curves for all damage states."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    im = np.linspace(im_range[0], im_range[1], 100)
    
    colors = {
        DamageState.SLIGHT: 'green',
        DamageState.MODERATE: 'yellow',
        DamageState.EXTENSIVE: 'orange', 
        DamageState.COMPLETE: 'red'
    }
    
    for ds, params in fragilities.items():
        prob = lognormal_fragility(im, params.median, params.beta)
        ax.plot(im, prob, color=colors.get(ds, 'blue'), 
                label=ds.name, linewidth=2)
        
    ax.set_xlabel('Intensity Measure (g)')
    ax.set_ylabel('Probability of Exceedance')
    ax.set_title('Fragility Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(im_range)
    ax.set_ylim(0, 1)
    
    return ax


# ===== Japanese Seismic Indices =====

def calculate_ds_index(story_drifts: Dict[int, float]) -> float:
    """
    Calculate structural characteristic coefficient Ds.
    
    Simplified approach based on ductility expectations.
    """
    max_drift = max(story_drifts.values()) if story_drifts else 0
    
    # Ds based on expected ductility
    if max_drift < 0.005:
        return 0.55  # High ductility
    elif max_drift < 0.01:
        return 0.45
    elif max_drift < 0.015:
        return 0.35
    else:
        return 0.30  # Very high ductility


def calculate_ai_distribution(
    story_heights: List[float],
    story_weights: List[float],
    T1: float = 1.0
) -> np.ndarray:
    """
    Calculate Ai distribution (seismic coefficient distribution).
    
    Ai = 1 + (1/√αi - αi) * 2T / (1 + 3T)
    
    where αi = Σwj(above) / Σwall
    """
    n = len(story_heights)
    total_weight = sum(story_weights)
    
    if total_weight <= 0:
        return np.ones(n)
        
    Ai = np.zeros(n)
    
    for i in range(n):
        weight_above = sum(story_weights[i:])
        alpha_i = weight_above / total_weight
        
        if alpha_i > 0:
            Ai[i] = 1 + (1/np.sqrt(alpha_i) - alpha_i) * 2*T1 / (1 + 3*T1)
        else:
            Ai[i] = 1.0
            
    return Ai


def calculate_required_ultimate_strength(
    story_weights: List[float],
    Ai: np.ndarray,
    C0: float = 0.2,
    Ds: float = 0.4,
    Fes: float = 1.0,
    Z: float = 1.0
) -> np.ndarray:
    """
    Calculate required ultimate horizontal strength (Qun).
    
    Qun = Ds * Fes * Z * Σ(Ci * Wi)
    Ci = Z * Rt * Ai * C0
    """
    n = len(story_weights)
    Qun = np.zeros(n)
    
    for i in range(n):
        Ci = Z * Ai[i] * C0  # Simplified Rt=1
        weight_above = sum(story_weights[i:])
        Qi = Ci * weight_above
        Qun[i] = Ds * Fes * Qi
        
    return Qun
