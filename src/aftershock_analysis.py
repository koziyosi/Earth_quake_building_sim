"""
Aftershock Analysis Module.
Analyzes structural response to earthquake sequences.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AftershockRecord:
    """Aftershock earthquake record."""
    time: np.ndarray
    acceleration: np.ndarray
    magnitude: float
    delay: float  # Time after mainshock (hours)
    epicenter_distance: float  # km


@dataclass
class SequenceResult:
    """Result of mainshock-aftershock sequence analysis."""
    mainshock_damage: float
    cumulative_damage: float
    final_residual_drift: float
    damage_progression: List[float]
    max_drifts: List[float]
    critical_aftershock_index: int  # Which aftershock caused most damage


def generate_aftershock_sequence(
    mainshock_magnitude: float,
    duration_days: float = 7,
    target_count: int = 10,
    dt: float = 0.01
) -> List[AftershockRecord]:
    """
    Generate synthetic aftershock sequence.
    
    Uses Omori-Utsu law for temporal decay:
    n(t) = K / (t + c)^p
    
    Uses Bath's law for magnitude:
    M_largest_aftershock ≈ M_mainshock - 1.2
    
    Args:
        mainshock_magnitude: Mainshock magnitude
        duration_days: Duration to generate aftershocks
        target_count: Approximate number of aftershocks
        dt: Time step for waveforms
        
    Returns:
        List of AftershockRecord objects
    """
    aftershocks = []
    
    # Bath's law estimate
    max_aftershock_mag = mainshock_magnitude - 1.2
    
    # Generate magnitudes (Gutenberg-Richter distribution)
    b = 1.0  # b-value
    mags = max_aftershock_mag - np.random.exponential(1/b, target_count)
    mags = np.clip(mags, mainshock_magnitude - 3, max_aftershock_mag)
    mags = np.sort(mags)[::-1]  # Largest first typically
    
    # Generate times (Omori law)
    c = 0.1  # hour
    p = 1.0
    
    # Inverse sampling from Omori distribution
    u = np.random.uniform(0, 1, target_count)
    if p != 1:
        times = c * ((1 - u) ** (-1/(p-1)) - 1)
    else:
        times = c * (np.exp(u * 3) - 1)  # Approximate
        
    times = np.sort(times)
    times = np.clip(times, 0.5, duration_days * 24)  # Hours
    
    # Generate waveforms
    for i, (mag, t_delay) in enumerate(zip(mags, times)):
        # Estimate duration from magnitude
        wave_duration = 5 + 5 * (mag - 4)  # seconds
        wave_duration = np.clip(wave_duration, 5, 60)
        
        # Estimate PGA from magnitude and distance
        distance = 10 + 50 * np.random.rand()  # km
        log_pga = 0.5 * mag - 0.003 * distance - 0.5  # Simplified attenuation
        pga = 10 ** log_pga * 100  # gal
        pga = np.clip(pga, 10, 1000)
        
        # Generate waveform
        time_arr, acc_arr = _generate_simple_wave(wave_duration, dt, pga)
        
        aftershocks.append(AftershockRecord(
            time=time_arr,
            acceleration=acc_arr,
            magnitude=mag,
            delay=t_delay,
            epicenter_distance=distance
        ))
        
    return aftershocks


def _generate_simple_wave(
    duration: float,
    dt: float,
    pga: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple synthetic earthquake wave."""
    t = np.arange(0, duration, dt)
    n = len(t)
    
    # Filtered white noise
    noise = np.random.normal(0, 1, n)
    
    # Apply envelope
    t_max = duration * 0.25
    envelope = (t / t_max) * np.exp(1 - t / t_max)
    envelope = np.where(t < 2 * t_max, envelope, envelope * np.exp(-(t - 2*t_max) / (duration * 0.3)))
    
    acc = noise * envelope
    
    # Normalize to target PGA
    acc = acc / np.max(np.abs(acc)) * pga * 0.01  # Convert gal to m/s²
    
    return t, acc


def run_sequence_analysis(
    run_single_analysis,  # Function to run single earthquake analysis
    mainshock: Tuple[np.ndarray, np.ndarray],  # (time, acc)
    aftershocks: List[AftershockRecord],
    recovery_time_factor: float = 0.0  # Stiffness recovery between events (0-1)
) -> SequenceResult:
    """
    Run mainshock-aftershock sequence analysis.
    
    Args:
        run_single_analysis: Function(time, acc) -> {max_drift, damage, ...}
        mainshock: (time, acceleration) tuple
        aftershocks: List of aftershock records
        recovery_time_factor: How much damage "heals" between events
        
    Returns:
        SequenceResult with cumulative damage tracking
    """
    damage_progression = []
    max_drifts = []
    cumulative_damage = 0.0
    
    # Run mainshock
    mainshock_result = run_single_analysis(mainshock[0], mainshock[1])
    mainshock_damage = mainshock_result.get('damage_index', 0)
    cumulative_damage = mainshock_damage
    damage_progression.append(cumulative_damage)
    max_drifts.append(mainshock_result.get('max_drift', 0))
    
    critical_idx = -1
    max_increment = 0
    
    # Run aftershocks
    for i, aftershock in enumerate(aftershocks):
        # Partial recovery
        cumulative_damage *= (1 - recovery_time_factor)
        
        # Run aftershock analysis
        try:
            result = run_single_analysis(aftershock.time, aftershock.acceleration)
            aftershock_damage = result.get('damage_index', 0)
            
            # Cumulative damage
            increment = aftershock_damage * (1 + cumulative_damage)  # Increased vulnerability
            cumulative_damage += increment
            
            if increment > max_increment:
                max_increment = increment
                critical_idx = i
                
            damage_progression.append(cumulative_damage)
            max_drifts.append(result.get('max_drift', 0))
        except Exception:
            damage_progression.append(cumulative_damage)
            max_drifts.append(0)
            
    # Final residual drift (approximate)
    final_drift = sum(max_drifts) * 0.2  # Rough estimate of residual
    
    return SequenceResult(
        mainshock_damage=mainshock_damage,
        cumulative_damage=cumulative_damage,
        final_residual_drift=final_drift,
        damage_progression=damage_progression,
        max_drifts=max_drifts,
        critical_aftershock_index=critical_idx
    )


def estimate_collapse_probability(
    sequence_result: SequenceResult,
    collapse_threshold: float = 1.0
) -> float:
    """
    Estimate probability of collapse under sequence.
    
    Based on damage index progression.
    """
    # Simplified: probability = min(damage / threshold, 1)
    prob = min(sequence_result.cumulative_damage / collapse_threshold, 1.0)
    return prob


def plot_damage_progression(
    result: SequenceResult,
    aftershocks: List[AftershockRecord],
    ax = None
):
    """Plot damage progression through sequence."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    events = ['Mainshock'] + [f'AS-{i+1}' for i in range(len(aftershocks))]
    
    ax.bar(range(len(result.damage_progression)), result.damage_progression, 
           color=['red'] + ['orange'] * len(aftershocks), alpha=0.7)
    ax.plot(range(len(result.damage_progression)), result.damage_progression, 
            'ko-', markersize=4)
            
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events, rotation=45)
    ax.set_ylabel('Cumulative Damage Index')
    ax.set_title('Damage Progression Through Earthquake Sequence')
    ax.axhline(1.0, color='red', linestyle='--', label='Collapse Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def get_modified_spectrum_for_damaged_building(
    base_spectrum: Dict[str, np.ndarray],
    damage_index: float,
    period_elongation_factor: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Get modified response spectrum for damaged building.
    
    Damage causes period elongation and reduced effective damping.
    
    Args:
        base_spectrum: {'periods': array, 'Sa': array}
        damage_index: Current damage state (0-1)
        period_elongation_factor: Factor for period increase per unit damage
        
    Returns:
        Modified spectrum
    """
    periods = base_spectrum['periods']
    Sa = base_spectrum['Sa']
    
    # Period elongation
    elongation = 1 + damage_index * period_elongation_factor
    
    # Shift spectrum
    modified_periods = periods * elongation
    
    # Interpolate back to original period range
    modified_Sa = np.interp(periods, modified_periods, Sa, left=Sa[0], right=Sa[-1])
    
    return {
        'periods': periods,
        'Sa': modified_Sa,
        'elongation_factor': elongation
    }
