"""
Drift Limit Checker Module.
Checks inter-story drifts against code limits.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BuildingCode(Enum):
    """Building code standards."""
    JAPAN_BCJ = "japan_bcj"          # Japanese Building Standard
    ASCE7 = "asce7"                  # US ASCE 7
    EUROCODE8 = "eurocode8"          # European Eurocode 8
    NZ_1170 = "nz_1170"              # New Zealand


class OccupancyCategory(Enum):
    """Building occupancy/importance."""
    NORMAL = "normal"
    IMPORTANT = "important"      # Schools, hospitals
    ESSENTIAL = "essential"       # Emergency facilities


@dataclass
class DriftLimit:
    """Drift limit specification."""
    code: BuildingCode
    limit: float
    condition: str
    description: str


# Drift limits by code
DRIFT_LIMITS = {
    BuildingCode.JAPAN_BCJ: {
        'serviceability': DriftLimit(BuildingCode.JAPAN_BCJ, 1/200, 'level_1', '稀に発生する地震 (Level 1)'),
        'ultimate': DriftLimit(BuildingCode.JAPAN_BCJ, 1/100, 'level_2', '極めて稀な地震 (Level 2)'),
        'collapse_prevention': DriftLimit(BuildingCode.JAPAN_BCJ, 1/50, 'level_3', '崩壊防止限界'),
    },
    BuildingCode.ASCE7: {
        'design': DriftLimit(BuildingCode.ASCE7, 0.020, 'design', 'Design Earthquake'),
        'mce': DriftLimit(BuildingCode.ASCE7, 0.030, 'mce', 'Maximum Considered Earthquake'),
    },
    BuildingCode.EUROCODE8: {
        'damage_limitation': DriftLimit(BuildingCode.EUROCODE8, 0.005, 'dl', 'Damage Limitation'),
        'no_collapse': DriftLimit(BuildingCode.EUROCODE8, 0.020, 'nc', 'No Collapse'),
    },
}


@dataclass
class DriftCheckResult:
    """Result of drift check for one story."""
    story: int
    drift: float
    limit: float
    ratio: float  # drift / limit
    passed: bool
    message: str


def calculate_story_drift(
    displacements: np.ndarray,
    story_heights: List[float]
) -> List[float]:
    """
    Calculate inter-story drift ratios.
    
    Args:
        displacements: Array of story displacements [story_0, story_1, ...]
        story_heights: Height of each story
        
    Returns:
        List of drift ratios for each story
    """
    drifts = []
    
    for i in range(len(displacements)):
        if i == 0:
            story_disp = displacements[i]
        else:
            story_disp = displacements[i] - displacements[i-1]
            
        h = story_heights[i] if i < len(story_heights) else story_heights[-1]
        drift_ratio = abs(story_disp) / h if h > 0 else 0
        drifts.append(drift_ratio)
        
    return drifts


def check_drift_limits(
    drifts: List[float],
    code: BuildingCode = BuildingCode.JAPAN_BCJ,
    level: str = 'serviceability',
    importance_factor: float = 1.0
) -> List[DriftCheckResult]:
    """
    Check drift limits against code requirements.
    
    Args:
        drifts: List of story drift ratios
        code: Building code to use
        level: Check level ('serviceability', 'ultimate', etc.)
        importance_factor: Importance factor (reduces limit)
        
    Returns:
        List of DriftCheckResult for each story
    """
    if code not in DRIFT_LIMITS:
        raise ValueError(f"Unknown code: {code}")
        
    limits = DRIFT_LIMITS[code]
    
    if level not in limits:
        level = list(limits.keys())[0]
        
    limit_spec = limits[level]
    effective_limit = limit_spec.limit / importance_factor
    
    results = []
    
    for i, drift in enumerate(drifts):
        ratio = drift / effective_limit if effective_limit > 0 else 0
        passed = ratio <= 1.0
        
        if passed:
            msg = f"✓ Drift {drift:.4f} ≤ {effective_limit:.4f}"
        else:
            msg = f"✗ Drift {drift:.4f} > {effective_limit:.4f} (Exceed {(ratio-1)*100:.1f}%)"
            
        results.append(DriftCheckResult(
            story=i + 1,
            drift=drift,
            limit=effective_limit,
            ratio=ratio,
            passed=passed,
            message=msg
        ))
        
    return results


def get_max_drift_story(drifts: List[float]) -> Tuple[int, float]:
    """Get story with maximum drift."""
    if not drifts:
        return 0, 0
    max_drift = max(drifts)
    max_story = drifts.index(max_drift) + 1
    return max_story, max_drift


def check_all_limits(
    drifts: List[float],
    code: BuildingCode = BuildingCode.JAPAN_BCJ
) -> Dict[str, List[DriftCheckResult]]:
    """Check all limit levels for a code."""
    if code not in DRIFT_LIMITS:
        return {}
        
    results = {}
    for level in DRIFT_LIMITS[code]:
        results[level] = check_drift_limits(drifts, code, level)
        
    return results


def generate_drift_report(
    drifts: List[float],
    story_heights: List[float],
    code: BuildingCode = BuildingCode.JAPAN_BCJ,
    building_name: str = "Building"
) -> str:
    """Generate drift check report."""
    lines = [
        "=" * 60,
        f"DRIFT CHECK REPORT: {building_name}",
        f"Code: {code.value}",
        "=" * 60,
        ""
    ]
    
    max_story, max_drift = get_max_drift_story(drifts)
    lines.append(f"Maximum Drift: 1/{int(1/max_drift) if max_drift > 0 else 'N/A'} at Story {max_story}")
    lines.append("")
    
    all_results = check_all_limits(drifts, code)
    
    for level, results in all_results.items():
        lines.append(f"--- {level.upper()} (Limit: 1/{int(1/results[0].limit)}) ---")
        
        all_passed = all(r.passed for r in results)
        
        for r in results:
            status = "✓" if r.passed else "✗"
            lines.append(f"  Story {r.story}: {status} 1/{int(1/r.drift) if r.drift > 0 else 'N/A'} ({r.ratio*100:.0f}%)")
            
        lines.append(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
        lines.append("")
        
    lines.append("=" * 60)
    
    return '\n'.join(lines)


def plot_drift_profile(
    drifts: List[float],
    story_heights: List[float],
    limits: Dict[str, float] = None,
    ax = None
):
    """Plot drift profile with limits."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        
    n_stories = len(drifts)
    stories = np.arange(1, n_stories + 1)
    
    # Cumulative height
    cumulative_height = np.cumsum([0] + list(story_heights[:n_stories]))
    story_centers = (cumulative_height[:-1] + cumulative_height[1:]) / 2
    
    # Plot drift profile
    ax.barh(stories, [d * 100 for d in drifts], height=0.7, color='steelblue', alpha=0.8)
    
    # Plot limits
    if limits is None:
        limits = {'1/200': 0.5, '1/100': 1.0}
        
    colors = ['green', 'orange', 'red']
    for i, (label, limit) in enumerate(limits.items()):
        ax.axvline(limit, color=colors[i % len(colors)], linestyle='--', label=label)
        
    ax.set_ylabel('Story')
    ax.set_xlabel('Drift Ratio (%)')
    ax.set_title('Inter-Story Drift Profile')
    ax.set_yticks(stories)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def calculate_drift_concentration_factor(drifts: List[float]) -> float:
    """
    Calculate drift concentration factor.
    
    High values indicate soft story behavior.
    """
    if not drifts or len(drifts) < 2:
        return 1.0
        
    max_drift = max(drifts)
    avg_drift = sum(drifts) / len(drifts)
    
    if avg_drift > 0:
        return max_drift / avg_drift
    return 1.0
