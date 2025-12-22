"""
Response Analyzer Module.
Calculates structural response indices from simulation results:
- Inter-story drift ratio (#16)
- Acceleration response (#18)
- Energy absorption (#19)
- Ductility factor (#20)
- Base shear coefficient (#21)
- Center of rigidity/mass (#22)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StoryResponse:
    """Response values for a single story."""
    story: int
    max_drift: float          # Maximum inter-story drift ratio
    max_disp: float           # Maximum displacement (m)
    max_accel: float          # Maximum acceleration (m/s²)
    max_shear: float          # Maximum story shear (N)
    energy: float             # Energy absorbed (J)
    ductility: float          # Ductility factor (μ)


@dataclass
class GlobalResponse:
    """Global building response summary."""
    max_base_shear: float     # Maximum base shear (N)
    base_shear_coef: float    # Base shear coefficient (Cb = V/W)
    max_top_disp: float       # Maximum top displacement (m)
    max_top_accel: float      # Maximum top acceleration (m/s²)
    total_energy: float       # Total absorbed energy (J)
    period_T1: float          # Fundamental period (s)
    center_of_rigidity: Tuple[float, float]  # (x, y)
    center_of_mass: Tuple[float, float]      # (x, y)


def calculate_inter_story_drift(
    u_history: np.ndarray,  # Shape: (n_steps, n_dof)
    nodes: List,
    story_heights: List[float]
) -> Dict[int, np.ndarray]:
    """
    Calculate inter-story drift ratio time history for each story.
    
    Args:
        u_history: Displacement history array
        nodes: List of Node objects  
        story_heights: List of story heights [m]
        
    Returns:
        Dictionary mapping story index to drift ratio time history
    """
    n_steps = u_history.shape[0]
    n_stories = len(story_heights)
    
    # Group nodes by floor
    floor_nodes = {}
    for node in nodes:
        floor_z = node.z
        if floor_z not in floor_nodes:
            floor_nodes[floor_z] = []
        floor_nodes[floor_z].append(node)
    
    # Sort floor levels
    floor_levels = sorted(floor_nodes.keys())
    
    drift_history = {}
    
    for i, (z_bot, z_top) in enumerate(zip(floor_levels[:-1], floor_levels[1:])):
        story_height = z_top - z_bot
        
        # Get average X displacement for nodes at each level
        disp_bot = np.zeros(n_steps)
        disp_top = np.zeros(n_steps)
        
        for node in floor_nodes.get(z_bot, []):
            if node.dof_indices[0] >= 0:
                disp_bot += u_history[:, node.dof_indices[0]]
        
        for node in floor_nodes.get(z_top, []):
            if node.dof_indices[0] >= 0:
                disp_top += u_history[:, node.dof_indices[0]]
        
        # Average
        n_bot = len([n for n in floor_nodes.get(z_bot, []) if n.dof_indices[0] >= 0])
        n_top = len([n for n in floor_nodes.get(z_top, []) if n.dof_indices[0] >= 0])
        
        if n_bot > 0:
            disp_bot /= n_bot
        if n_top > 0:
            disp_top /= n_top
            
        # Drift ratio = (disp_top - disp_bot) / story_height
        drift = (disp_top - disp_bot) / story_height
        drift_history[i + 1] = drift  # 1-indexed stories
        
    return drift_history


def calculate_acceleration_response(
    a_history: np.ndarray,  # Shape: (n_steps, n_dof)
    nodes: List,
    ground_acc: np.ndarray  # Ground acceleration time history
) -> Dict[int, np.ndarray]:
    """
    Calculate absolute acceleration time history for each floor.
    
    Absolute acceleration = relative acceleration + ground acceleration
    
    Args:
        a_history: Relative acceleration history
        nodes: List of Node objects
        ground_acc: Ground acceleration
        
    Returns:
        Dictionary mapping floor index to absolute acceleration
    """
    n_steps = a_history.shape[0]
    
    # Group nodes by floor
    floor_nodes = {}
    for node in nodes:
        floor_z = node.z
        if floor_z not in floor_nodes:
            floor_nodes[floor_z] = []
        floor_nodes[floor_z].append(node)
    
    floor_levels = sorted(floor_nodes.keys())
    accel_history = {}
    
    for i, z_level in enumerate(floor_levels):
        if z_level == 0:
            # Base level = ground motion
            accel_history[0] = ground_acc[:len(a_history)]
        else:
            # Average acceleration of nodes at this level
            avg_accel = np.zeros(n_steps)
            count = 0
            
            for node in floor_nodes[z_level]:
                if node.dof_indices[0] >= 0:
                    avg_accel += a_history[:, node.dof_indices[0]]
                    count += 1
                    
            if count > 0:
                avg_accel /= count
                
            # Absolute = relative + ground
            abs_accel = avg_accel + ground_acc[:n_steps]
            accel_history[i] = abs_accel
            
    return accel_history


def calculate_energy_absorption(
    u_history: np.ndarray,
    f_history: np.ndarray,  # Internal force history
    dt: float
) -> Tuple[np.ndarray, float]:
    """
    Calculate cumulative energy absorption time history.
    
    Energy = ∫ F · du = Σ F_i * Δu_i
    
    Args:
        u_history: Displacement history
        f_history: Force history
        dt: Time step
        
    Returns:
        Tuple of (energy time history, total energy)
    """
    n_steps = u_history.shape[0]
    n_dof = u_history.shape[1] if u_history.ndim > 1 else 1
    
    energy = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        du = u_history[i] - u_history[i-1]
        f_avg = (f_history[i] + f_history[i-1]) / 2
        
        if u_history.ndim > 1:
            dE = np.sum(f_avg * du)
        else:
            dE = f_avg * du
            
        energy[i] = energy[i-1] + abs(dE)
        
    return energy, energy[-1]


def calculate_ductility(
    max_disp: float,
    yield_disp: float
) -> float:
    """
    Calculate ductility factor.
    
    μ = δ_max / δ_y
    
    Args:
        max_disp: Maximum displacement
        yield_disp: Yield displacement
        
    Returns:
        Ductility factor
    """
    if yield_disp <= 0:
        return 0.0
    return abs(max_disp) / yield_disp


def calculate_base_shear_coefficient(
    base_shear_history: np.ndarray,
    total_weight: float
) -> Tuple[np.ndarray, float]:
    """
    Calculate base shear coefficient time history.
    
    Cb = V / W
    
    Args:
        base_shear_history: Base shear force time history (N)
        total_weight: Total building weight (N)
        
    Returns:
        Tuple of (Cb time history, max Cb)
    """
    if total_weight <= 0:
        return np.zeros_like(base_shear_history), 0.0
        
    Cb = base_shear_history / total_weight
    return Cb, np.max(np.abs(Cb))


def calculate_center_of_rigidity(
    nodes: List,
    elements: List
) -> Tuple[float, float]:
    """
    Calculate center of rigidity (剛心) for the building.
    
    CR = Σ(Ki * xi) / Σ Ki
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        
    Returns:
        (x_cr, y_cr) coordinates of center of rigidity
    """
    sum_kx = 0.0
    sum_ky = 0.0
    sum_k = 0.0
    
    for elem in elements:
        # Get element center
        x_c = (elem.node_i.x + elem.node_j.x) / 2
        y_c = (elem.node_i.y + elem.node_j.y) / 2
        
        # Get approximate stiffness (use lateral stiffness)
        if hasattr(elem, 'E') and hasattr(elem, 'Iy'):
            L = elem.get_length()
            k = 12 * elem.E * elem.Iy / (L ** 3) if L > 0 else 0
        else:
            k = 1.0  # Fallback
            
        sum_kx += k * x_c
        sum_ky += k * y_c
        sum_k += k
        
    if sum_k > 0:
        return (sum_kx / sum_k, sum_ky / sum_k)
    return (0.0, 0.0)


def calculate_center_of_mass(
    nodes: List
) -> Tuple[float, float]:
    """
    Calculate center of mass (重心) for the building.
    
    CM = Σ(mi * xi) / Σ mi
    
    Args:
        nodes: List of Node objects with mass attribute
        
    Returns:
        (x_cm, y_cm) coordinates of center of mass
    """
    sum_mx = 0.0
    sum_my = 0.0
    sum_m = 0.0
    
    for node in nodes:
        m = node.mass
        sum_mx += m * node.x
        sum_my += m * node.y
        sum_m += m
        
    if sum_m > 0:
        return (sum_mx / sum_m, sum_my / sum_m)
    return (0.0, 0.0)


def analyze_response(
    nodes: List,
    elements: List,
    u_history: np.ndarray,
    a_history: np.ndarray,
    t: np.ndarray,
    story_heights: List[float],
    ground_acc: np.ndarray,
    yield_disp: float = 0.01
) -> Tuple[List[StoryResponse], GlobalResponse]:
    """
    Comprehensive response analysis.
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        u_history: Displacement history (n_steps, n_dof)
        a_history: Acceleration history (n_steps, n_dof)
        t: Time array
        story_heights: List of story heights
        ground_acc: Ground acceleration
        yield_disp: Approximate yield displacement for ductility
        
    Returns:
        Tuple of (list of StoryResponse, GlobalResponse)
    """
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    
    # Calculate drift
    drift_history = calculate_inter_story_drift(u_history, nodes, story_heights)
    
    # Calculate acceleration
    accel_history = calculate_acceleration_response(a_history, nodes, ground_acc)
    
    # Calculate centers
    cr = calculate_center_of_rigidity(nodes, elements)
    cm = calculate_center_of_mass(nodes)
    
    # Total weight
    total_weight = sum(node.mass * 9.81 for node in nodes)
    
    # Story responses
    story_responses = []
    total_energy = 0.0
    
    for story_idx in drift_history.keys():
        drift = drift_history[story_idx]
        accel = accel_history.get(story_idx, np.zeros_like(t))
        
        max_drift = np.max(np.abs(drift))
        max_accel = np.max(np.abs(accel))
        
        # Approximate max displacement for this story
        max_disp = max_drift * story_heights[story_idx - 1] if story_idx <= len(story_heights) else 0
        
        # Ductility
        ductility = calculate_ductility(max_disp, yield_disp)
        
        story_resp = StoryResponse(
            story=story_idx,
            max_drift=max_drift,
            max_disp=max_disp,
            max_accel=max_accel,
            max_shear=0.0,  # Would need force history
            energy=0.0,     # Would need force history
            ductility=ductility
        )
        story_responses.append(story_resp)
    
    # Global response
    max_top_disp = np.max(np.abs(u_history[:, -6])) if u_history.shape[1] >= 6 else 0
    max_top_accel = np.max(np.abs(a_history[:, -6])) if a_history.shape[1] >= 6 else 0
    
    global_resp = GlobalResponse(
        max_base_shear=0.0,
        base_shear_coef=0.0,
        max_top_disp=max_top_disp,
        max_top_accel=max_top_accel,
        total_energy=total_energy,
        period_T1=0.0,  # Would need modal analysis
        center_of_rigidity=cr,
        center_of_mass=cm
    )
    
    return story_responses, global_resp


def print_response_summary(
    story_responses: List[StoryResponse],
    global_response: GlobalResponse
):
    """Print formatted response summary."""
    print("\n" + "="*60)
    print(" RESPONSE ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n--- Story Responses ---")
    print(f"{'Story':<8}{'Max Drift':<15}{'Max Disp (m)':<15}{'Ductility':<12}")
    print("-"*50)
    
    for sr in story_responses:
        print(f"{sr.story:<8}{sr.max_drift:<15.4f}{sr.max_disp:<15.4f}{sr.ductility:<12.2f}")
    
    print("\n--- Global Response ---")
    print(f"Max Top Displacement: {global_response.max_top_disp:.4f} m")
    print(f"Max Top Acceleration: {global_response.max_top_accel:.3f} m/s²")
    print(f"Center of Rigidity: ({global_response.center_of_rigidity[0]:.2f}, {global_response.center_of_rigidity[1]:.2f})")
    print(f"Center of Mass: ({global_response.center_of_mass[0]:.2f}, {global_response.center_of_mass[1]:.2f})")
    print("="*60)
