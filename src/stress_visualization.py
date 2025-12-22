"""
Stress Visualization Module.
Visualizes structural stress and internal forces.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class StressResult:
    """Stress result for an element."""
    element_id: int
    axial_stress: float      # N/m²
    bending_stress_y: float  # N/m²
    bending_stress_z: float  # N/m²
    shear_stress_y: float    # N/m²
    shear_stress_z: float    # N/m²
    combined_stress: float   # Von Mises or principal
    
    @property
    def max_stress(self) -> float:
        """Maximum absolute stress value."""
        return max(abs(self.axial_stress),
                   abs(self.bending_stress_y),
                   abs(self.bending_stress_z),
                   abs(self.combined_stress))


def calculate_element_stresses(
    axial_force: float,
    moment_y: float,
    moment_z: float,
    shear_y: float,
    shear_z: float,
    area: float,
    I_y: float,
    I_z: float,
    height: float,
    width: float
) -> StressResult:
    """
    Calculate stresses in beam-column element.
    
    Args:
        axial_force: Axial force (N)
        moment_y: Moment about Y axis (Nm)
        moment_z: Moment about Z axis (Nm)
        shear_y: Shear in Y direction (N)
        shear_z: Shear in Z direction (N)
        area: Cross-section area (m²)
        I_y: Moment of inertia about Y (m⁴)
        I_z: Moment of inertia about Z (m⁴)
        height: Section height (m)
        width: Section width (m)
        
    Returns:
        StressResult object
    """
    # Axial stress
    sigma_axial = axial_force / area if area > 0 else 0
    
    # Bending stress (max at extreme fiber)
    c_y = height / 2
    c_z = width / 2
    sigma_by = moment_y * c_y / I_y if I_y > 0 else 0
    sigma_bz = moment_z * c_z / I_z if I_z > 0 else 0
    
    # Shear stress (average)
    tau_y = shear_y / area if area > 0 else 0
    tau_z = shear_z / area if area > 0 else 0
    
    # Combined stress (Von Mises approximation)
    sigma_total = sigma_axial + sigma_by + sigma_bz
    tau_total = np.sqrt(tau_y**2 + tau_z**2)
    sigma_vm = np.sqrt(sigma_total**2 + 3 * tau_total**2)
    
    return StressResult(
        element_id=0,
        axial_stress=sigma_axial,
        bending_stress_y=sigma_by,
        bending_stress_z=sigma_bz,
        shear_stress_y=tau_y,
        shear_stress_z=tau_z,
        combined_stress=sigma_vm
    )


def calculate_utilization_ratio(
    stress: StressResult,
    yield_stress: float = 235e6  # Pa
) -> float:
    """
    Calculate stress utilization ratio.
    
    Returns:
        Ratio of stress to allowable (>1 means overstressed)
    """
    return stress.combined_stress / yield_stress if yield_stress > 0 else 0


class StressColorMapper:
    """
    Maps stress values to colors for visualization.
    """
    
    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 235e6,
        colormap: str = 'RdYlGn_r'
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.cmap = plt.get_cmap(colormap)
        
    def get_color(self, value: float) -> str:
        """Get hex color for stress value."""
        if self.max_val <= self.min_val:
            norm = 0
        else:
            norm = (value - self.min_val) / (self.max_val - self.min_val)
            norm = max(0, min(1, norm))
            
        rgba = self.cmap(norm)
        return mcolors.to_hex(rgba)
        
    def get_color_rgb(self, value: float) -> Tuple[int, int, int]:
        """Get RGB tuple for stress value."""
        hex_color = self.get_color(value)
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def draw_stress_contour_2d(
    nodes: List,
    elements: List,
    stresses: Dict[int, StressResult],
    stress_type: str = 'combined',
    ax = None,
    show_legend: bool = True
):
    """
    Draw 2D stress contour plot.
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        stresses: Dict of element_id -> StressResult
        stress_type: 'axial', 'bending', 'shear', or 'combined'
        ax: Matplotlib axis
        show_legend: Show color legend
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Get stress values
    stress_values = []
    for elem_id, stress in stresses.items():
        if stress_type == 'axial':
            val = abs(stress.axial_stress)
        elif stress_type == 'bending':
            val = max(abs(stress.bending_stress_y), abs(stress.bending_stress_z))
        elif stress_type == 'shear':
            val = max(abs(stress.shear_stress_y), abs(stress.shear_stress_z))
        else:
            val = stress.combined_stress
        stress_values.append(val)
        
    if not stress_values:
        return ax
        
    # Create color mapper
    mapper = StressColorMapper(min(stress_values), max(stress_values))
    
    # Draw elements with color
    for elem in elements:
        elem_stress = stresses.get(elem.id)
        if elem_stress:
            if stress_type == 'combined':
                val = elem_stress.combined_stress
            else:
                val = getattr(elem_stress, f'{stress_type}_stress', 0)
            color = mapper.get_color(abs(val))
        else:
            color = '#808080'
            
        x = [elem.node_i.x, elem.node_j.x]
        z = [elem.node_i.z, elem.node_j.z]
        
        ax.plot(x, z, color=color, linewidth=4)
        
    # Draw nodes
    xs = [n.x for n in nodes]
    zs = [n.z for n in nodes]
    ax.scatter(xs, zs, c='white', s=20, zorder=5)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{stress_type.capitalize()} Stress Contour')
    ax.set_facecolor('#1a1a2e')
    
    # Legend
    if show_legend:
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap('RdYlGn_r'),
            norm=plt.Normalize(min(stress_values)/1e6, max(stress_values)/1e6)
        )
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Stress (MPa)')
        
    return ax


def create_stress_report(
    elements: List,
    stresses: Dict[int, StressResult],
    yield_stress: float = 235e6
) -> str:
    """Generate text report of stress results."""
    lines = [
        "=" * 60,
        "STRESS ANALYSIS REPORT",
        "=" * 60,
        f"Yield Stress: {yield_stress/1e6:.0f} MPa",
        "",
        f"{'Elem':>6} {'σ_axial':>12} {'σ_bend':>12} {'τ':>12} {'σ_VM':>12} {'Util':>8}",
        f"{'':>6} {'(MPa)':>12} {'(MPa)':>12} {'(MPa)':>12} {'(MPa)':>12} {'(%)':>8}",
        "-" * 60
    ]
    
    max_util = 0
    critical_elem = None
    
    for elem_id, stress in sorted(stresses.items()):
        util = calculate_utilization_ratio(stress, yield_stress) * 100
        
        if util > max_util:
            max_util = util
            critical_elem = elem_id
            
        lines.append(
            f"{elem_id:>6} {stress.axial_stress/1e6:>12.2f} "
            f"{stress.bending_stress_y/1e6:>12.2f} "
            f"{stress.shear_stress_y/1e6:>12.2f} "
            f"{stress.combined_stress/1e6:>12.2f} "
            f"{util:>8.1f}{'*' if util > 100 else ''}"
        )
        
    lines.append("-" * 60)
    lines.append(f"Critical Element: {critical_elem} (Utilization: {max_util:.1f}%)")
    
    if max_util > 100:
        lines.append("⚠ WARNING: Some elements exceed yield stress!")
    else:
        lines.append("✓ All elements within allowable stress")
        
    lines.append("=" * 60)
    
    return '\n'.join(lines)


def plot_stress_history(
    time: np.ndarray,
    stress_history: np.ndarray,
    element_ids: List[int] = None,
    yield_stress: float = 235e6,
    ax = None
):
    """Plot stress time history for selected elements."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    n_elements = stress_history.shape[1] if len(stress_history.shape) > 1 else 1
    
    if element_ids is None:
        element_ids = list(range(n_elements))
        
    colors = plt.cm.viridis(np.linspace(0, 1, len(element_ids)))
    
    for i, elem_id in enumerate(element_ids[:10]):  # Max 10 elements
        if elem_id < n_elements:
            stress = stress_history[:, elem_id] if n_elements > 1 else stress_history
            ax.plot(time, stress / 1e6, color=colors[i], label=f'Elem {elem_id}')
            
    ax.axhline(yield_stress / 1e6, color='red', linestyle='--', label='Yield')
    ax.axhline(-yield_stress / 1e6, color='red', linestyle='--')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Stress Time History')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax
