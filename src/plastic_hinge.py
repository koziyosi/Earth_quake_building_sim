"""
Plastic Hinge Visualization Module.
Tracks and visualizes plastic hinge formation.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt


class HingeState(Enum):
    """Plastic hinge state."""
    ELASTIC = "elastic"
    YIELD = "yield"
    STRAIN_HARDENING = "strain_hardening"
    CAPPING = "capping"
    POST_CAPPING = "post_capping"
    RESIDUAL = "residual"
    FAILED = "failed"


@dataclass
class PlasticHinge:
    """Plastic hinge data."""
    element_id: int
    location: str  # 'i', 'j', or 'mid'
    rotation: float = 0.0
    moment: float = 0.0
    state: HingeState = HingeState.ELASTIC
    max_rotation: float = 0.0
    cumulative_rotation: float = 0.0
    cycle_count: int = 0
    
    # Hinge properties
    yield_rotation: float = 0.01
    capping_rotation: float = 0.05
    ultimate_rotation: float = 0.10
    
    def update(self, rotation: float, moment: float):
        """Update hinge state based on rotation."""
        self.rotation = rotation
        self.moment = moment
        
        # Track maximum
        if abs(rotation) > self.max_rotation:
            self.max_rotation = abs(rotation)
            
        # Determine state
        rot_abs = abs(rotation)
        
        if rot_abs < self.yield_rotation:
            self.state = HingeState.ELASTIC
        elif rot_abs < self.capping_rotation:
            self.state = HingeState.STRAIN_HARDENING
        elif rot_abs < self.ultimate_rotation:
            self.state = HingeState.POST_CAPPING
        else:
            self.state = HingeState.FAILED
            
    @property
    def ductility(self) -> float:
        """Rotation ductility demand."""
        if self.yield_rotation > 0:
            return self.max_rotation / self.yield_rotation
        return 0


@dataclass
class HingeTracker:
    """Tracks plastic hinges for entire structure."""
    hinges: Dict[Tuple[int, str], PlasticHinge] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    
    def add_hinge(self, element_id: int, location: str, **kwargs):
        """Add potential hinge location."""
        key = (element_id, location)
        self.hinges[key] = PlasticHinge(element_id, location, **kwargs)
        
    def update_hinge(self, element_id: int, location: str, rotation: float, moment: float):
        """Update hinge state."""
        key = (element_id, location)
        if key in self.hinges:
            self.hinges[key].update(rotation, moment)
            
    def record_snapshot(self, time: float):
        """Record current state."""
        snapshot = {
            'time': time,
            'states': {k: h.state.value for k, h in self.hinges.items()},
            'rotations': {k: h.rotation for k, h in self.hinges.items()}
        }
        self.history.append(snapshot)
        
    def get_yielded_hinges(self) -> List[PlasticHinge]:
        """Get all yielded hinges."""
        return [h for h in self.hinges.values() if h.state != HingeState.ELASTIC]
        
    def get_failed_hinges(self) -> List[PlasticHinge]:
        """Get all failed hinges."""
        return [h for h in self.hinges.values() if h.state == HingeState.FAILED]
        
    def count_by_state(self) -> Dict[HingeState, int]:
        """Count hinges by state."""
        counts = {state: 0 for state in HingeState}
        for h in self.hinges.values():
            counts[h.state] += 1
        return counts


# Color mapping for hinge states
HINGE_COLORS = {
    HingeState.ELASTIC: '#4dabf7',       # Blue
    HingeState.YIELD: '#ffd43b',          # Yellow
    HingeState.STRAIN_HARDENING: '#ff9f40', # Orange
    HingeState.CAPPING: '#ff6b6b',        # Light red
    HingeState.POST_CAPPING: '#e03131',   # Red
    HingeState.RESIDUAL: '#862e9c',       # Purple
    HingeState.FAILED: '#212529',         # Dark
}


def draw_hinges_2d(
    nodes: List,
    elements: List,
    tracker: HingeTracker,
    ax = None,
    scale: float = 1.0
):
    """
    Draw structure with plastic hinges.
    
    Args:
        nodes: List of nodes
        elements: List of elements  
        tracker: HingeTracker object
        ax: Matplotlib axis
        scale: Deformation scale
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Draw elements
    for elem in elements:
        x = [elem.node_i.x, elem.node_j.x]
        z = [elem.node_i.z, elem.node_j.z]
        
        # Element line
        ax.plot(x, z, 'gray', linewidth=2)
        
        # Hinges at i-end
        hinge_i = tracker.hinges.get((elem.id, 'i'))
        if hinge_i:
            color = HINGE_COLORS[hinge_i.state]
            size = 10 + hinge_i.ductility * 5
            ax.scatter([elem.node_i.x], [elem.node_i.z], 
                      c=[color], s=size, zorder=5, marker='o')
                      
        # Hinges at j-end
        hinge_j = tracker.hinges.get((elem.id, 'j'))
        if hinge_j:
            color = HINGE_COLORS[hinge_j.state]
            size = 10 + hinge_j.ductility * 5
            ax.scatter([elem.node_j.x], [elem.node_j.z],
                      c=[color], s=size, zorder=5, marker='o')
                      
    # Legend
    for state, color in HINGE_COLORS.items():
        ax.scatter([], [], c=[color], s=50, label=state.value)
        
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Plastic Hinge Map')
    ax.set_facecolor('#f8f9fa')
    
    return ax


def plot_hinge_progression(
    tracker: HingeTracker,
    ax = None
):
    """Plot hinge formation over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    if not tracker.history:
        return ax
        
    times = [s['time'] for s in tracker.history]
    
    # Count by state at each time
    state_counts = {state: [] for state in HingeState}
    
    for snapshot in tracker.history:
        counts = {state: 0 for state in HingeState}
        for state_val in snapshot['states'].values():
            state = HingeState(state_val)
            counts[state] += 1
        for state, count in counts.items():
            state_counts[state].append(count)
            
    # Stack plot
    labels = []
    data = []
    colors = []
    
    for state in HingeState:
        if max(state_counts[state]) > 0:
            labels.append(state.value)
            data.append(state_counts[state])
            colors.append(HINGE_COLORS[state])
            
    ax.stackplot(times, *data, labels=labels, colors=colors, alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of Hinges')
    ax.set_title('Plastic Hinge Formation')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax


def generate_hinge_report(tracker: HingeTracker) -> str:
    """Generate plastic hinge report."""
    lines = [
        "=" * 60,
        "PLASTIC HINGE REPORT",
        "=" * 60,
        ""
    ]
    
    counts = tracker.count_by_state()
    total = sum(counts.values())
    
    lines.append("Hinge States Summary:")
    for state, count in counts.items():
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        lines.append(f"  {state.value:20} {bar} {count:3} ({pct:.1f}%)")
        
    lines.append("")
    
    # Critical hinges
    yielded = tracker.get_yielded_hinges()
    if yielded:
        lines.append("Yielded Hinges:")
        for h in sorted(yielded, key=lambda x: -x.ductility)[:10]:
            lines.append(f"  Element {h.element_id}-{h.location}: μ={h.ductility:.2f}, θ={h.max_rotation:.4f}")
            
    failed = tracker.get_failed_hinges()
    if failed:
        lines.append("")
        lines.append("⚠ FAILED HINGES:")
        for h in failed:
            lines.append(f"  Element {h.element_id}-{h.location}")
            
    lines.append("")
    lines.append("=" * 60)
    
    return '\n'.join(lines)


def create_asce41_hinges(
    element_type: str = 'beam',
    section_type: str = 'compact'
) -> Dict:
    """
    Create ASCE 41 plastic hinge properties.
    
    Returns dict of rotation limits.
    """
    # ASCE 41-17 Table 9-6 (Steel beams)
    if element_type == 'beam':
        if section_type == 'compact':
            return {
                'yield_rotation': 0.01,
                'capping_rotation': 0.05,
                'ultimate_rotation': 0.10,
                'residual_strength': 0.2
            }
        else:  # non-compact
            return {
                'yield_rotation': 0.008,
                'capping_rotation': 0.03,
                'ultimate_rotation': 0.05,
                'residual_strength': 0.2
            }
    elif element_type == 'column':
        return {
            'yield_rotation': 0.008,
            'capping_rotation': 0.025,
            'ultimate_rotation': 0.05,
            'residual_strength': 0.2
        }
    else:
        return {
            'yield_rotation': 0.01,
            'capping_rotation': 0.04,
            'ultimate_rotation': 0.08,
            'residual_strength': 0.2
        }
