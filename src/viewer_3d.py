"""
3D Visualization Viewer module.
Provides interactive 3D visualization using matplotlib or optional PyVista.
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ViewerConfig:
    """Configuration for 3D viewer."""
    window_width: int = 800
    window_height: int = 600
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    show_grid: bool = True
    show_axes: bool = True
    deformation_scale: float = 10.0


class SimpleViewer3D:
    """Simple 3D viewer using Matplotlib for basic visualization."""
    
    def __init__(self, config: Optional[ViewerConfig] = None):
        self.config = config or ViewerConfig()
        self.fig = None
        self.ax = None
        
    def setup(self):
        """Initialize the viewer."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        self.fig = plt.figure(figsize=(
            self.config.window_width / 100,
            self.config.window_height / 100
        ))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.config.background_color)
        
    def plot_structure(
        self,
        nodes: List,
        elements: List,
        displacement: Optional[np.ndarray] = None,
        title: str = "Structure View"
    ):
        """
        Plot the structure in 3D.
        
        Args:
            nodes: List of Node objects
            elements: List of Element objects
            displacement: Optional displacement vector (n_dof,)
            title: Plot title
        """
        if self.ax is None:
            self.setup()
            
        self.ax.clear()
        
        scale = self.config.deformation_scale
        
        # Create node coordinate mapping with displacement
        node_coords = {}
        for node in nodes:
            dx, dy, dz = 0, 0, 0
            if displacement is not None:
                if node.dof_indices[0] >= 0 and node.dof_indices[0] < len(displacement):
                    dx = displacement[node.dof_indices[0]] * scale
                if node.dof_indices[1] >= 0 and node.dof_indices[1] < len(displacement):
                    dy = displacement[node.dof_indices[1]] * scale
                if node.dof_indices[2] >= 0 and node.dof_indices[2] < len(displacement):
                    dz = displacement[node.dof_indices[2]] * scale
                    
            node_coords[node.id] = (node.x + dx, node.y + dy, node.z + dz)
        
        # Plot elements
        for elem in elements:
            if hasattr(elem, 'node_i') and hasattr(elem, 'node_j'):
                ni_id = elem.node_i.id
                nj_id = elem.node_j.id
                
                if ni_id in node_coords and nj_id in node_coords:
                    c1 = node_coords[ni_id]
                    c2 = node_coords[nj_id]
                    
                    # Determine color based on element type or damage
                    color = 'blue'
                    if hasattr(elem, 'damage_index'):
                        damage = getattr(elem, 'damage_index', 0)
                        if damage > 0.8:
                            color = 'red'
                        elif damage > 0.5:
                            color = 'orange'
                        elif damage > 0.2:
                            color = 'yellow'
                    
                    # Check if it's a column (vertical) or beam (horizontal)
                    is_column = abs(c2[2] - c1[2]) > abs(c2[0] - c1[0]) + abs(c2[1] - c1[1])
                    linewidth = 2.5 if is_column else 1.5
                    
                    self.ax.plot3D(
                        [c1[0], c2[0]],
                        [c1[1], c2[1]],
                        [c1[2], c2[2]],
                        color=color,
                        linewidth=linewidth
                    )
        
        # Plot nodes
        xs = [c[0] for c in node_coords.values()]
        ys = [c[1] for c in node_coords.values()]
        zs = [c[2] for c in node_coords.values()]
        
        # Color nodes by elevation
        if zs:
            self.ax.scatter3D(xs, ys, zs, c=zs, cmap='viridis', s=30, alpha=0.8)
        
        # Formatting
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(title)
        
        # Set equal aspect ratio
        if xs and ys and zs:
            max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
            mid_x = (max(xs) + min(xs)) / 2
            mid_y = (max(ys) + min(ys)) / 2
            mid_z = (max(zs) + min(zs)) / 2
            
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(0, max(zs) * 1.1)
        
    def show(self):
        """Display the plot."""
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()
        
    def save(self, filepath: str, dpi: int = 150):
        """Save the plot to file."""
        if self.fig:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            
    def animate_response(
        self,
        nodes: List,
        elements: List,
        displacement_history: List[np.ndarray],
        time_array: np.ndarray,
        output_path: str = "animation_3d.gif",
        skip_frames: int = 5
    ) -> str:
        """
        Create animation of structural response.
        
        Args:
            nodes: List of nodes
            elements: List of elements
            displacement_history: List of displacement arrays
            time_array: Time values
            output_path: Output GIF path
            skip_frames: Frame skip for speed
            
        Returns:
            Path to created animation
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        if self.fig is None:
            self.setup()
            
        frames = range(0, len(displacement_history), skip_frames)
        
        def update(frame):
            idx = frame * skip_frames if frame * skip_frames < len(displacement_history) else -1
            u = displacement_history[idx] if idx < len(displacement_history) else displacement_history[-1]
            t = time_array[idx] if idx < len(time_array) else time_array[-1]
            
            self.plot_structure(
                nodes, elements, u,
                title=f"Response at t={t:.2f}s"
            )
            
        ani = FuncAnimation(
            self.fig, update,
            frames=len(list(frames)),
            interval=50
        )
        
        try:
            ani.save(output_path, writer='pillow', fps=20, dpi=80)
            return output_path
        except Exception as e:
            print(f"Animation save failed: {e}")
            return ""


def try_pyvista_viewer():
    """
    Attempt to use PyVista for advanced visualization.
    Returns None if PyVista is not available.
    """
    try:
        import pyvista as pv
        return PyVistaViewer()
    except ImportError:
        return None


class PyVistaViewer:
    """
    Advanced 3D viewer using PyVista (optional dependency).
    Provides interactive visualization with better performance.
    """
    
    def __init__(self):
        try:
            import pyvista as pv
            self.pv = pv
            self.plotter = None
        except ImportError:
            raise ImportError("PyVista is required for this viewer. Install with: pip install pyvista")
    
    def setup(self, window_size: Tuple[int, int] = (1024, 768)):
        """Initialize PyVista plotter."""
        self.plotter = self.pv.Plotter(window_size=window_size)
        self.plotter.set_background('#1a1a2e')
        
    def plot_structure(
        self,
        nodes: List,
        elements: List,
        displacement: Optional[np.ndarray] = None,
        scale: float = 10.0
    ):
        """Plot structure using PyVista tubes for elements."""
        if self.plotter is None:
            self.setup()
            
        # Create points array
        points = []
        for node in nodes:
            dx = dy = dz = 0
            if displacement is not None:
                if node.dof_indices[0] >= 0:
                    dx = displacement[node.dof_indices[0]] * scale
                if node.dof_indices[1] >= 0:
                    dy = displacement[node.dof_indices[1]] * scale
                if node.dof_indices[2] >= 0:
                    dz = displacement[node.dof_indices[2]] * scale
            points.append([node.x + dx, node.y + dy, node.z + dz])
        
        points = np.array(points)
        
        # Create lines for elements
        for elem in elements:
            if hasattr(elem, 'node_i') and hasattr(elem, 'node_j'):
                idx_i = next((i for i, n in enumerate(nodes) if n.id == elem.node_i.id), None)
                idx_j = next((i for i, n in enumerate(nodes) if n.id == elem.node_j.id), None)
                
                if idx_i is not None and idx_j is not None:
                    line = self.pv.Line(points[idx_i], points[idx_j])
                    tube = line.tube(radius=0.1)
                    
                    # Color based on damage or height
                    color = 'lightblue'
                    if hasattr(elem, 'damage_index'):
                        damage = elem.damage_index
                        if damage > 0.5:
                            color = 'red'
                        elif damage > 0.2:
                            color = 'orange'
                            
                    self.plotter.add_mesh(tube, color=color)
        
        # Add nodes as spheres
        cloud = self.pv.PolyData(points)
        self.plotter.add_mesh(cloud, render_points_as_spheres=True, 
                              point_size=10, color='white')
        
    def show(self):
        """Display the plot."""
        if self.plotter:
            self.plotter.show()
            
    def close(self):
        """Close the plotter."""
        if self.plotter:
            self.plotter.close()
