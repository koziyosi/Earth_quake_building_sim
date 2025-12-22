"""
Residual Deformation Evaluator.
Evaluates post-earthquake residual drift and tilt for damage assessment.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ResidualResult:
    """Result of residual deformation analysis."""
    floor: int
    residual_drift_x: float  # Residual inter-story drift ratio (X)
    residual_drift_y: float  # Residual inter-story drift ratio (Y)
    residual_tilt: float     # Overall tilt (rad)
    repair_category: str     # Repair classification


# Repair classification based on residual drift (Japanese guidelines)
REPAIR_THRESHOLDS = {
    'No Repair': 0.002,       # 1/500
    'Minor Repair': 0.005,    # 1/200
    'Major Repair': 0.01,     # 1/100
    'Reconstruction': 0.015   # 1/67
}


@dataclass
class ResidualAnalysisResult:
    """Complete residual analysis result."""
    floor_results: List[ResidualResult]
    max_residual_drift: float
    max_residual_floor: int
    overall_tilt: float
    recommendation: str


def evaluate_residual_deformation(
    displacement_history: np.ndarray,
    time_array: np.ndarray,
    story_heights: List[float],
    floor_dof_mapping: Dict[int, Tuple[int, int]],  # floor -> (dof_x, dof_y)
    window_size: float = 2.0  # seconds to average at the end
) -> ResidualAnalysisResult:
    """
    Evaluate residual (permanent) deformation after earthquake.
    
    Args:
        displacement_history: Array of shape (n_steps, n_dof)
        time_array: Time array
        story_heights: Height of each story
        floor_dof_mapping: Maps floor number to (dof_x, dof_y)
        window_size: Time window at end to average for residual
        
    Returns:
        ResidualAnalysisResult
    """
    n_steps = len(time_array)
    dt = time_array[1] - time_array[0] if n_steps > 1 else 0.01
    
    # Number of steps to average at the end
    n_avg = int(window_size / dt)
    n_avg = min(n_avg, n_steps // 4)  # Max 25% of history
    n_avg = max(n_avg, 1)
    
    # Get average displacement at end (residual)
    residual_disp = np.mean(displacement_history[-n_avg:], axis=0)
    
    floor_results = []
    n_floors = len(story_heights)
    
    for floor in range(1, n_floors + 1):
        # Get floor displacements
        if floor in floor_dof_mapping:
            dof_x, dof_y = floor_dof_mapping[floor]
        else:
            continue
            
        # Lower floor
        if floor - 1 in floor_dof_mapping:
            lower_dof_x, lower_dof_y = floor_dof_mapping[floor - 1]
        else:
            lower_dof_x, lower_dof_y = -1, -1
        
        h = story_heights[floor - 1]
        
        # Upper floor displacement
        u_x = residual_disp[dof_x] if dof_x >= 0 and dof_x < len(residual_disp) else 0
        u_y = residual_disp[dof_y] if dof_y >= 0 and dof_y < len(residual_disp) else 0
        
        # Lower floor displacement
        l_x = residual_disp[lower_dof_x] if lower_dof_x >= 0 and lower_dof_x < len(residual_disp) else 0
        l_y = residual_disp[lower_dof_y] if lower_dof_y >= 0 and lower_dof_y < len(residual_disp) else 0
        
        # Residual drift
        drift_x = abs(u_x - l_x) / h if h > 0 else 0
        drift_y = abs(u_y - l_y) / h if h > 0 else 0
        
        # Tilt (combined)
        tilt = np.sqrt(drift_x**2 + drift_y**2)
        
        # Repair category
        max_drift = max(drift_x, drift_y)
        category = classify_repair(max_drift)
        
        floor_results.append(ResidualResult(
            floor=floor,
            residual_drift_x=drift_x,
            residual_drift_y=drift_y,
            residual_tilt=tilt,
            repair_category=category
        ))
    
    # Find maximum
    max_drift = 0.0
    max_floor = 0
    
    for fr in floor_results:
        drift = max(fr.residual_drift_x, fr.residual_drift_y)
        if drift > max_drift:
            max_drift = drift
            max_floor = fr.floor
    
    # Overall building tilt (top floor vs. base)
    if n_floors in floor_dof_mapping and 0 in floor_dof_mapping:
        top_dof = floor_dof_mapping[n_floors]
        base_dof = floor_dof_mapping[0]
        
        top_x = residual_disp[top_dof[0]] if top_dof[0] >= 0 and top_dof[0] < len(residual_disp) else 0
        base_x = residual_disp[base_dof[0]] if base_dof[0] >= 0 and base_dof[0] < len(residual_disp) else 0
        
        total_height = sum(story_heights)
        overall_tilt = abs(top_x - base_x) / total_height if total_height > 0 else 0
    else:
        overall_tilt = max_drift  # Approximation
    
    # Generate recommendation
    recommendation = generate_recommendation(max_drift, overall_tilt, floor_results)
    
    return ResidualAnalysisResult(
        floor_results=floor_results,
        max_residual_drift=max_drift,
        max_residual_floor=max_floor,
        overall_tilt=overall_tilt,
        recommendation=recommendation
    )


def classify_repair(drift: float) -> str:
    """Classify repair needs based on residual drift."""
    if drift < REPAIR_THRESHOLDS['No Repair']:
        return 'No Repair'
    elif drift < REPAIR_THRESHOLDS['Minor Repair']:
        return 'Minor Repair'
    elif drift < REPAIR_THRESHOLDS['Major Repair']:
        return 'Major Repair'
    elif drift < REPAIR_THRESHOLDS['Reconstruction']:
        return 'Reconstruction Required'
    else:
        return 'Unsafe - Demolition'


def generate_recommendation(
    max_drift: float,
    overall_tilt: float,
    floor_results: List[ResidualResult]
) -> str:
    """Generate recommendation text based on residual analysis."""
    lines = []
    
    # Overall assessment
    if max_drift < 0.002:
        lines.append("建物は安全です。残留変形は許容範囲内です。")
    elif max_drift < 0.005:
        lines.append("軽微な残留変形が検出されました。詳細点検を推奨します。")
    elif max_drift < 0.01:
        lines.append("中程度の残留変形があります。補修が必要です。")
    elif max_drift < 0.015:
        lines.append("重大な残留変形があります。大規模補修または建替えを検討してください。")
    else:
        lines.append("危険：建物は安全ではありません。即時避難が必要です。")
    
    # Floor-specific notes
    severe_floors = [fr.floor for fr in floor_results if fr.repair_category in ['Reconstruction Required', 'Unsafe - Demolition']]
    if severe_floors:
        lines.append(f"重大な損傷が確認された階: {', '.join(map(str, severe_floors))}F")
    
    # Tilt assessment
    if overall_tilt > 1/200:
        lines.append(f"建物全体の傾斜: 1/{int(1/overall_tilt)} (要注意)")
    
    return "\n".join(lines)


def plot_residual_profile(
    result: ResidualAnalysisResult,
    story_heights: List[float]
):
    """
    Plot residual deformation profile.
    
    Returns matplotlib figure.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Calculate cumulative heights
    heights = [0]
    for h in story_heights:
        heights.append(heights[-1] + h)
    
    floors = [fr.floor for fr in result.floor_results]
    drift_x = [fr.residual_drift_x * 100 for fr in result.floor_results]  # percent
    drift_y = [fr.residual_drift_y * 100 for fr in result.floor_results]
    
    # Height at floor center
    floor_heights = [heights[f] - story_heights[f-1]/2 for f in floors]
    
    # Drift profile
    ax1.barh(floor_heights, drift_x, height=story_heights[0]*0.8, 
             alpha=0.7, label='X方向', color='blue')
    ax1.barh([h + 0.2 for h in floor_heights], drift_y, height=story_heights[0]*0.8, 
             alpha=0.7, label='Y方向', color='red')
    ax1.axvline(0.2, color='orange', linestyle='--', label='軽微損傷限界')
    ax1.axvline(0.5, color='red', linestyle='--', label='補修要限界')
    ax1.set_xlabel('残留層間変形角 (%)')
    ax1.set_ylabel('高さ (m)')
    ax1.set_title('残留変形分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Repair category bar
    categories = ['No Repair', 'Minor Repair', 'Major Repair', 'Reconstruction Required', 'Unsafe - Demolition']
    cat_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    cat_counts = {cat: 0 for cat in categories}
    for fr in result.floor_results:
        if fr.repair_category in cat_counts:
            cat_counts[fr.repair_category] += 1
    
    ax2.barh(list(cat_counts.keys()), list(cat_counts.values()), 
             color=cat_colors[:len(cat_counts)])
    ax2.set_xlabel('階数')
    ax2.set_title('補修判定分布')
    
    plt.tight_layout()
    return fig
