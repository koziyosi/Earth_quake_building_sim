"""
Visualization Utilities Module.
Enhanced plotting and display functions.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PlotSettings:
    """Customizable plot settings."""
    line_width: float = 1.5
    marker_size: float = 6
    grid_alpha: float = 0.3
    title_size: int = 12
    label_size: int = 10
    legend_size: int = 9
    dpi: int = 100
    style: str = 'seaborn-v0_8-whitegrid'


def plot_hysteresis_loops(
    disp_history: np.ndarray,
    force_history: np.ndarray,
    element_ids: List[int] = None,
    ax = None,
    settings: PlotSettings = None
):
    """
    Plot hysteresis loops for elements.
    
    Args:
        disp_history: (n_steps, n_elements) displacement
        force_history: (n_steps, n_elements) force
        element_ids: Element IDs to label
        ax: Existing axis or None
        settings: Plot settings
    """
    import matplotlib.pyplot as plt
    
    settings = settings or PlotSettings()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    n_elements = disp_history.shape[1] if len(disp_history.shape) > 1 else 1
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_elements))
    
    for i in range(n_elements):
        d = disp_history[:, i] if n_elements > 1 else disp_history
        f = force_history[:, i] if n_elements > 1 else force_history
        
        label = f'Element {element_ids[i]}' if element_ids else f'Element {i+1}'
        ax.plot(d * 100, f / 1000, color=colors[i], 
                linewidth=settings.line_width, label=label)
        
    ax.set_xlabel('Displacement (cm)', fontsize=settings.label_size)
    ax.set_ylabel('Force (kN)', fontsize=settings.label_size)
    ax.set_title('Hysteresis Loops', fontsize=settings.title_size)
    ax.grid(True, alpha=settings.grid_alpha)
    ax.legend(fontsize=settings.legend_size)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    return ax


def plot_internal_forces(
    nodes: List,
    elements: List,
    forces: Dict[str, np.ndarray],
    force_type: str = 'moment',
    scale: float = 1.0,
    ax = None
):
    """
    Plot internal force diagrams.
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        forces: Dict with 'axial', 'shear', 'moment' arrays
        force_type: 'axial', 'shear', or 'moment'
        scale: Scale factor for drawing
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Draw structure
    for elem in elements:
        x = [elem.node_i.x, elem.node_j.x]
        z = [elem.node_i.z, elem.node_j.z]
        ax.plot(x, z, 'k-', linewidth=2)
        
    # Draw force diagram
    if force_type in forces:
        f = forces[force_type]
        
        for i, elem in enumerate(elements):
            if i >= len(f):
                continue
                
            # Simple representation - perpendicular offset
            dx = elem.node_j.x - elem.node_i.x
            dz = elem.node_j.z - elem.node_i.z
            L = np.sqrt(dx**2 + dz**2)
            
            if L > 0:
                nx, nz = -dz/L, dx/L  # Normal vector
                
                # Draw at midpoint
                mx = (elem.node_i.x + elem.node_j.x) / 2
                mz = (elem.node_i.z + elem.node_j.z) / 2
                
                val = f[i] * scale
                ax.plot([mx, mx + nx * val], [mz, mz + nz * val], 'r-')
                ax.annotate(f'{f[i]:.0f}', (mx + nx * val, mz + nz * val), fontsize=8)
                
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{force_type.capitalize()} Diagram')
    
    return ax


def plot_time_history_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ylabel: str = 'Response',
    title: str = 'Time History Comparison',
    ax = None
):
    """
    Plot multiple time histories for comparison.
    
    Args:
        results: Dict of {label: (time, values)}
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    for label, (t, v) in results.items():
        ax.plot(t, v, label=label, linewidth=1.2)
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_envelope(
    heights: np.ndarray,
    max_values: np.ndarray,
    min_values: np.ndarray = None,
    xlabel: str = 'Max Value',
    title: str = 'Envelope',
    ax = None
):
    """Plot vertical envelope diagram."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
        
    ax.plot(max_values, heights, 'r-o', label='Max', linewidth=2)
    if min_values is not None:
        ax.plot(min_values, heights, 'b-o', label='Min', linewidth=2)
        ax.fill_betweenx(heights, min_values, max_values, alpha=0.2)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Height (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_damage_colormap():
    """Create custom colormap for damage visualization."""
    import matplotlib.colors as mcolors
    
    colors = [
        (0.0, 'green'),       # No damage
        (0.25, 'yellowgreen'),
        (0.5, 'yellow'),      # Moderate
        (0.75, 'orange'),
        (1.0, 'red')          # Severe
    ]
    
    return mcolors.LinearSegmentedColormap.from_list('damage', colors)


def export_figure(fig, filepath: str, dpi: int = 150, transparent: bool = False):
    """Export figure to file."""
    fig.savefig(filepath, dpi=dpi, transparent=transparent, bbox_inches='tight')
    

def take_screenshot(widget, filepath: str):
    """Take screenshot of tkinter widget."""
    import tkinter as tk
    
    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    w = widget.winfo_width()
    h = widget.winfo_height()
    
    from PIL import ImageGrab
    img = ImageGrab.grab((x, y, x+w, y+h))
    img.save(filepath)
    return filepath


# ===== Animation Speed Control =====

class AnimationController:
    """Controls animation playback."""
    
    def __init__(self, total_frames: int, fps: float = 30):
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
        
    def play(self):
        self.playing = True
        
    def pause(self):
        self.playing = False
        
    def stop(self):
        self.playing = False
        self.current_frame = 0
        
    def step_forward(self, n: int = 1):
        self.current_frame = min(self.current_frame + n, self.total_frames - 1)
        
    def step_backward(self, n: int = 1):
        self.current_frame = max(self.current_frame - n, 0)
        
    def set_frame(self, frame: int):
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        
    def set_speed(self, speed: float):
        self.speed = max(0.1, min(speed, 10.0))
        
    @property
    def frame_delay_ms(self) -> int:
        return int(1000 / (self.fps * self.speed))
        
    @property
    def progress(self) -> float:
        return self.current_frame / max(1, self.total_frames - 1)


# ===== Color Schemes =====

class ColorScheme:
    """Predefined color schemes."""
    
    DARK = {
        'bg': '#1e1e1e',
        'fg': '#e0e0e0',
        'accent': '#0078d4',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'grid': '#404040'
    }
    
    LIGHT = {
        'bg': '#ffffff',
        'fg': '#333333',
        'accent': '#2196f3',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'grid': '#cccccc'
    }
    
    SEISMIC = {
        'bg': '#0a0a20',
        'fg': '#ffffff',
        'accent': '#ff4444',
        'grid': '#333366',
        'intensity_1': '#00ff00',
        'intensity_5': '#ffff00',
        'intensity_7': '#ff0000'
    }
