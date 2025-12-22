"""
Performance-Based Design Module.
Implements performance-based earthquake engineering (PBEE) concepts.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PerformanceLevel(Enum):
    """Performance levels (based on ASCE 41)."""
    OPERATIONAL = "operational"       # IO
    IMMEDIATE_OCCUPANCY = "io"        # IO
    LIFE_SAFETY = "ls"               # LS
    COLLAPSE_PREVENTION = "cp"        # CP


class HazardLevel(Enum):
    """Seismic hazard levels."""
    FREQUENT = "frequent"         # 50% in 50 years
    DESIGN_BASIS = "dbe"          # 10% in 50 years (475 year)
    MAXIMUM_CONSIDERED = "mce"    # 2% in 50 years (2475 year)


@dataclass
class PerformanceObjective:
    """Performance objective pairing hazard with performance."""
    hazard_level: HazardLevel
    performance_level: PerformanceLevel
    description: str


@dataclass
class PerformanceCheckResult:
    """Result of performance check."""
    passed: bool
    demand: float
    capacity: float
    demand_capacity_ratio: float
    performance_level: PerformanceLevel
    check_type: str
    message: str


# ===== Standard Performance Objectives =====

PERFORMANCE_OBJECTIVES = {
    'basic': [
        PerformanceObjective(
            HazardLevel.DESIGN_BASIS,
            PerformanceLevel.LIFE_SAFETY,
            "Life Safety at Design Earthquake"
        )
    ],
    'essential': [
        PerformanceObjective(
            HazardLevel.DESIGN_BASIS,
            PerformanceLevel.IMMEDIATE_OCCUPANCY,
            "Immediate Occupancy at Design Earthquake"
        ),
        PerformanceObjective(
            HazardLevel.MAXIMUM_CONSIDERED,
            PerformanceLevel.LIFE_SAFETY,
            "Life Safety at Maximum Considered Earthquake"
        )
    ],
    'critical': [
        PerformanceObjective(
            HazardLevel.FREQUENT,
            PerformanceLevel.OPERATIONAL,
            "Operational at Frequent Earthquake"
        ),
        PerformanceObjective(
            HazardLevel.DESIGN_BASIS,
            PerformanceLevel.IMMEDIATE_OCCUPANCY,
            "Immediate Occupancy at Design Earthquake"
        ),
        PerformanceObjective(
            HazardLevel.MAXIMUM_CONSIDERED,
            PerformanceLevel.COLLAPSE_PREVENTION,
            "Collapse Prevention at MCE"
        )
    ]
}


# ===== Acceptance Criteria =====

DRIFT_LIMITS = {
    PerformanceLevel.OPERATIONAL: 0.005,
    PerformanceLevel.IMMEDIATE_OCCUPANCY: 0.010,
    PerformanceLevel.LIFE_SAFETY: 0.020,
    PerformanceLevel.COLLAPSE_PREVENTION: 0.040,
}

DUCTILITY_LIMITS = {
    PerformanceLevel.OPERATIONAL: 1.0,
    PerformanceLevel.IMMEDIATE_OCCUPANCY: 2.0,
    PerformanceLevel.LIFE_SAFETY: 4.0,
    PerformanceLevel.COLLAPSE_PREVENTION: 6.0,
}


def check_drift_performance(
    max_drift: float,
    performance_level: PerformanceLevel
) -> PerformanceCheckResult:
    """Check if drift meets performance criteria."""
    limit = DRIFT_LIMITS.get(performance_level, 0.02)
    dcr = max_drift / limit
    
    return PerformanceCheckResult(
        passed=dcr <= 1.0,
        demand=max_drift,
        capacity=limit,
        demand_capacity_ratio=dcr,
        performance_level=performance_level,
        check_type="Inter-story Drift",
        message=f"Drift {max_drift:.4f} vs limit {limit:.4f} (DCR={dcr:.2f})"
    )


def check_ductility_performance(
    ductility: float,
    performance_level: PerformanceLevel
) -> PerformanceCheckResult:
    """Check if ductility meets performance criteria."""
    limit = DUCTILITY_LIMITS.get(performance_level, 4.0)
    dcr = ductility / limit
    
    return PerformanceCheckResult(
        passed=dcr <= 1.0,
        demand=ductility,
        capacity=limit,
        demand_capacity_ratio=dcr,
        performance_level=performance_level,
        check_type="Ductility Demand",
        message=f"Ductility {ductility:.2f} vs limit {limit:.2f} (DCR={dcr:.2f})"
    )


def run_performance_assessment(
    analysis_results: Dict[str, float],
    objectives: List[PerformanceObjective]
) -> Dict[str, List[PerformanceCheckResult]]:
    """
    Run full performance assessment against objectives.
    
    Args:
        analysis_results: Results from seismic analysis
        objectives: Performance objectives to check
        
    Returns:
        Dictionary of check results by hazard level
    """
    results = {}
    
    for objective in objectives:
        hazard = objective.hazard_level.value
        perf = objective.performance_level
        
        checks = []
        
        # Drift check
        drift_key = f'max_drift_{hazard}'
        if drift_key in analysis_results:
            checks.append(check_drift_performance(
                analysis_results[drift_key], perf
            ))
        elif 'max_drift' in analysis_results:
            checks.append(check_drift_performance(
                analysis_results['max_drift'], perf
            ))
            
        # Ductility check
        duct_key = f'ductility_{hazard}'
        if duct_key in analysis_results:
            checks.append(check_ductility_performance(
                analysis_results[duct_key], perf
            ))
        elif 'ductility' in analysis_results:
            checks.append(check_ductility_performance(
                analysis_results['ductility'], perf
            ))
            
        results[f'{hazard}_{perf.value}'] = checks
        
    return results


def calculate_annualized_loss(
    fragility_params: Dict[PerformanceLevel, Tuple[float, float]],
    hazard_curve: Tuple[np.ndarray, np.ndarray],
    repair_costs: Dict[PerformanceLevel, float]
) -> float:
    """
    Calculate Expected Annual Loss (EAL).
    
    EAL = Σ ∫ P(DS|IM) * |dλ/dIM| * L(DS) dIM
    
    Args:
        fragility_params: {level: (median, beta)} for lognormal fragility
        hazard_curve: (IM_values, annual_exceedance_rate)
        repair_costs: Loss ratio for each damage state
        
    Returns:
        Expected annual loss ratio
    """
    from scipy import stats
    
    im_vals, rates = hazard_curve
    
    # Calculate derivative of hazard curve (negative)
    d_rate = np.diff(rates) / np.diff(im_vals)
    im_mid = (im_vals[:-1] + im_vals[1:]) / 2
    
    eal = 0.0
    
    for level, (median, beta) in fragility_params.items():
        # Probability of exceeding this damage state
        p_exceed = stats.norm.cdf(np.log(im_mid / median) / beta)
        
        # Loss for this state
        loss = repair_costs.get(level, 0)
        
        # Integrate
        eal += np.sum(p_exceed * np.abs(d_rate) * loss * np.diff(im_vals))
        
    return eal


def design_for_target_reliability(
    target_probability: float,
    hazard_curve: Tuple[np.ndarray, np.ndarray],
    demand_uncertainty: float = 0.3,
    capacity_uncertainty: float = 0.3
) -> float:
    """
    Calculate required capacity for target reliability.
    
    Uses Load and Resistance Factor Design (LRFD) approach.
    
    Args:
        target_probability: Target annual collapse probability
        hazard_curve: (IM, rate)
        demand_uncertainty: βD
        capacity_uncertainty: βC
        
    Returns:
        Required capacity margin factor
    """
    from scipy import stats
    
    # Total uncertainty
    beta_total = np.sqrt(demand_uncertainty**2 + capacity_uncertainty**2)
    
    # Find IM at target probability
    im_vals, rates = hazard_curve
    target_im = np.interp(target_probability, rates[::-1], im_vals[::-1])
    
    # Capacity margin to achieve target reliability
    # Using first-order reliability
    reliability_index = stats.norm.ppf(1 - target_probability)
    margin = np.exp(reliability_index * beta_total)
    
    return margin


def generate_performance_report(
    check_results: Dict[str, List[PerformanceCheckResult]]
) -> str:
    """Generate text report of performance assessment."""
    lines = [
        "=" * 60,
        "PERFORMANCE-BASED SEISMIC ASSESSMENT",
        "=" * 60,
        ""
    ]
    
    all_passed = True
    
    for key, checks in check_results.items():
        lines.append(f"\n--- {key.upper()} ---")
        
        for check in checks:
            status = "✓ PASS" if check.passed else "✗ FAIL"
            lines.append(f"  {check.check_type}: {status}")
            lines.append(f"    Demand/Capacity = {check.demand:.4f} / {check.capacity:.4f}")
            lines.append(f"    DCR = {check.demand_capacity_ratio:.2f}")
            
            if not check.passed:
                all_passed = False
                
    lines.append("")
    lines.append("=" * 60)
    
    if all_passed:
        lines.append("OVERALL: ALL PERFORMANCE OBJECTIVES MET ✓")
    else:
        lines.append("OVERALL: SOME PERFORMANCE OBJECTIVES NOT MET ✗")
        
    lines.append("=" * 60)
    
    return '\n'.join(lines)
