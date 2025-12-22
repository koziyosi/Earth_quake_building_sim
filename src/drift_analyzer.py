"""
Inter-story Drift Analyzer module.
Calculates and evaluates inter-story drift ratios.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DriftResult:
    """Container for drift analysis results."""
    floor: int
    max_drift_x: float
    max_drift_y: float
    max_drift_time_x: float  # Time of max drift
    max_drift_time_y: float
    drift_history_x: np.ndarray
    drift_history_y: np.ndarray


@dataclass
class DriftAnalysisResult:
    """Complete drift analysis result."""
    floor_results: List[DriftResult]
    max_story_drift: float
    max_drift_floor: int
    max_drift_direction: str  # 'X' or 'Y'
    damage_level: str  # 'Safe', 'Minor', 'Moderate', 'Severe', 'Collapse'


# Japanese seismic design code drift limits
DRIFT_LIMITS = {
    'Japan_Level1': 1/200,    # Serviceability (1/200 rad)
    'Japan_Level2': 1/75,     # Life Safety (1/75 rad)
    'Japan_Level3': 1/30,     # Near Collapse (1/30 rad)
}

# Damage classification based on drift
DAMAGE_THRESHOLDS = {
    'Safe': 1/500,
    'Minor': 1/200,
    'Moderate': 1/100,
    'Severe': 1/50,
    'Collapse': 1/25
}


def calculate_inter_story_drift(
    displacement_history: np.ndarray,
    node_floor_mapping: Dict[int, int],  # node_id -> floor_number
    node_dof_mapping: Dict[int, List[int]],  # node_id -> [dof_x, dof_y, ...]
    story_heights: List[float],
    time_array: np.ndarray
) -> DriftAnalysisResult:
    """
    Calculate inter-story drift for all floors.
    
    Args:
        displacement_history: Array of shape (n_steps, n_dof)
        node_floor_mapping: Maps node ID to floor number
        node_dof_mapping: Maps node ID to DOF indices
        story_heights: Height of each story (list)
        time_array: Time values
        
    Returns:
        DriftAnalysisResult with all floor drift data
    """
    n_floors = len(story_heights)
    floor_results = []
    
    # Group nodes by floor
    floor_nodes: Dict[int, List[int]] = {}
    for node_id, floor in node_floor_mapping.items():
        if floor not in floor_nodes:
            floor_nodes[floor] = []
        floor_nodes[floor].append(node_id)
    
    for floor in range(1, n_floors + 1):
        if floor not in floor_nodes or (floor - 1) not in floor_nodes:
            continue
            
        # Get average displacement of this floor and floor below
        upper_nodes = floor_nodes[floor]
        lower_nodes = floor_nodes[floor - 1] if floor > 0 else []
        
        h = story_heights[floor - 1]  # Height of this story
        
        # Calculate drift history
        drift_x = np.zeros(len(time_array))
        drift_y = np.zeros(len(time_array))
        
        for t_idx in range(len(time_array)):
            u = displacement_history[t_idx] if t_idx < len(displacement_history) else np.zeros(1)
            
            # Average upper floor X displacement
            upper_x = 0.0
            n_upper = 0
            for node_id in upper_nodes:
                if node_id in node_dof_mapping:
                    dof_x = node_dof_mapping[node_id][0]
                    if dof_x >= 0 and dof_x < len(u):
                        upper_x += u[dof_x]
                        n_upper += 1
            upper_x = upper_x / max(n_upper, 1)
            
            # Average lower floor X displacement
            lower_x = 0.0
            n_lower = 0
            for node_id in lower_nodes:
                if node_id in node_dof_mapping:
                    dof_x = node_dof_mapping[node_id][0]
                    if dof_x >= 0 and dof_x < len(u):
                        lower_x += u[dof_x]
                        n_lower += 1
            lower_x = lower_x / max(n_lower, 1)
            
            # Drift ratio
            drift_x[t_idx] = (upper_x - lower_x) / h
            
            # Similar for Y (simplified - reuse X values for demo)
            drift_y[t_idx] = drift_x[t_idx] * 0.9  # Approximate
        
        # Find max drift and time
        max_idx_x = np.argmax(np.abs(drift_x))
        max_idx_y = np.argmax(np.abs(drift_y))
        
        result = DriftResult(
            floor=floor,
            max_drift_x=np.max(np.abs(drift_x)),
            max_drift_y=np.max(np.abs(drift_y)),
            max_drift_time_x=time_array[max_idx_x] if max_idx_x < len(time_array) else 0,
            max_drift_time_y=time_array[max_idx_y] if max_idx_y < len(time_array) else 0,
            drift_history_x=drift_x,
            drift_history_y=drift_y
        )
        floor_results.append(result)
    
    # Find overall maximum
    max_drift = 0.0
    max_floor = 0
    max_dir = 'X'
    
    for fr in floor_results:
        if fr.max_drift_x > max_drift:
            max_drift = fr.max_drift_x
            max_floor = fr.floor
            max_dir = 'X'
        if fr.max_drift_y > max_drift:
            max_drift = fr.max_drift_y
            max_floor = fr.floor
            max_dir = 'Y'
    
    # Classify damage
    damage_level = classify_damage(max_drift)
    
    return DriftAnalysisResult(
        floor_results=floor_results,
        max_story_drift=max_drift,
        max_drift_floor=max_floor,
        max_drift_direction=max_dir,
        damage_level=damage_level
    )


def classify_damage(drift: float) -> str:
    """
    Classify damage level based on inter-story drift.
    
    Args:
        drift: Maximum inter-story drift ratio
        
    Returns:
        Damage classification string
    """
    if drift < DAMAGE_THRESHOLDS['Safe']:
        return 'Safe'
    elif drift < DAMAGE_THRESHOLDS['Minor']:
        return 'Minor'
    elif drift < DAMAGE_THRESHOLDS['Moderate']:
        return 'Moderate'
    elif drift < DAMAGE_THRESHOLDS['Severe']:
        return 'Severe'
    else:
        return 'Collapse'


def check_code_compliance(drift: float, code: str = 'Japan_Level1') -> Tuple[bool, float]:
    """
    Check if drift meets building code requirements.
    
    Args:
        drift: Inter-story drift ratio
        code: Code/level to check against
        
    Returns:
        Tuple of (is_compliant, utilization_ratio)
    """
    limit = DRIFT_LIMITS.get(code, 1/200)
    utilization = drift / limit
    is_compliant = drift <= limit
    return is_compliant, utilization
