"""
Enhanced 3D Building Viewer.
Provides interactive 3D visualization of building models with damage display.
Uses matplotlib for embedded visualization in Tkinter.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


@dataclass
class ViewSettings:
    """3D view settings."""
    elevation: float = 25.0
    azimuth: float = -60.0
    scale: float = 1.0
    show_nodes: bool = True
    show_labels: bool = False
    show_axes: bool = True
    damage_colormap: str = 'RdYlGn_r'  # Red for damage, green for safe


class Building3DViewer(ttk.Frame):
    """
    Interactive 3D building viewer with damage visualization.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.nodes = []
        self.elements = []
        self.displacement_history = None
        self.damage_history = None
        self.current_frame = 0
        
        # Animation state
        self.is_animating = False
        self.animation_job = None
        self.realtime_interval = 50  # ms per frame
        self.animation_duration = 5.0  # default 5s
        
        self.settings = ViewSettings()
        self.create_widgets()
        
    def create_widgets(self):
        # Control Panel
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # View controls
        ttk.Label(control_frame, text="視点:").pack(side=tk.LEFT)
        
        views = [("正面 (X)", 0, 0), ("側面 (Y)", 0, 90), ("上面 (Z)", 90, 0), ("アイソメ", 25, -60)]
        for name, elev, azim in views:
            btn = ttk.Button(control_frame, text=name, 
                            command=lambda e=elev, a=azim: self.set_view(e, a))
            btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Display options
        self.show_nodes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="節点", variable=self.show_nodes_var,
                       command=self.redraw).pack(side=tk.LEFT)
        
        self.show_labels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="ラベル", variable=self.show_labels_var,
                       command=self.redraw).pack(side=tk.LEFT)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="グリッド", variable=self.show_grid_var,
                       command=self.redraw).pack(side=tk.LEFT)
        
        self.equal_aspect_var = tk.BooleanVar(value=True)  # Default ON
        ttk.Checkbutton(control_frame, text="等比率", variable=self.equal_aspect_var,
                       command=self.redraw).pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Model info display
        self.info_label = ttk.Label(control_frame, text="モデル: -- | 節点: -- | 要素: --")
        self.info_label.pack(side=tk.LEFT)
        
        # Animation slider and controls
        slider_frame = ttk.Frame(self)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Playback controls
        self.play_btn = ttk.Button(slider_frame, text="▶ 再生", width=8, command=self.toggle_animation)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(slider_frame, text="⏹ 停止", width=8, command=self.stop_animation).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(slider_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(slider_frame, text="時刻:").pack(side=tk.LEFT)
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                     variable=self.time_var, orient=tk.HORIZONTAL,
                                     command=self.on_time_change)
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.time_label = ttk.Label(slider_frame, text="0.00 / 0.00 s")
        self.time_label.pack(side=tk.LEFT)
        
        # Scale slider
        ttk.Label(slider_frame, text="変形倍率:").pack(side=tk.LEFT, padx=(20, 5))
        self.scale_var = tk.DoubleVar(value=20)
        scale_slider = ttk.Scale(slider_frame, from_=1, to=100, 
                                variable=self.scale_var, orient=tk.HORIZONTAL,
                                command=self.on_scale_change, length=100)
        scale_slider.pack(side=tk.LEFT)
        
        self.scale_label = ttk.Label(slider_frame, text="x20")
        self.scale_label.pack(side=tk.LEFT)
        
        # Matplotlib Figure
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='#1e1e1e')
        self.ax = self.figure.add_subplot(111, projection='3d', facecolor='#1e1e1e')
        
        # Style the 3D axes
        self.ax.set_xlabel('X (m)', color='white')
        self.ax.set_ylabel('Y (m)', color='white')
        self.ax.set_zlabel('Z (m)', color='white')
        self.ax.tick_params(colors='white')
        
        # Darker grid
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Canvas
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        
    def set_model(self, nodes: List, elements: List):
        """Set the building model to display."""
        self.nodes = nodes
        self.elements = elements
        
        # Update model info display
        if nodes and elements:
            max_z = max(n.z for n in nodes) if nodes else 0
            self.info_label.config(text=f"高さ: {max_z:.1f}m | 節点: {len(nodes)} | 要素: {len(elements)}")
        else:
            self.info_label.config(text="モデル: -- | 節点: -- | 要素: --")
        
        self.redraw()
        
    def set_results(self, time_array: np.ndarray, 
                    displacement_history: List[np.ndarray],
                    damage_history: List[List[float]] = None):
        """Set simulation results for animation."""
        self.time_array = time_array
        self.displacement_history = displacement_history
        self.damage_history = damage_history
        
        # Update slider range
        n_frames = len(displacement_history)
        self.time_slider.configure(to=n_frames - 1)
        
        self.redraw()
        
    def set_view(self, elevation: float, azimuth: float):
        """Set camera view angles."""
        self.settings.elevation = elevation
        self.settings.azimuth = azimuth
        self.ax.view_init(elev=elevation, azim=azimuth)
        self.canvas.draw()
        
    def on_time_change(self, value):
        """Handle time slider change."""
        self.current_frame = int(float(value))
        if hasattr(self, 'time_array') and len(self.time_array) > self.current_frame:
            t = self.time_array[self.current_frame]
            total_t = self.time_array[-1] if len(self.time_array) > 0 else self.animation_duration
            self.time_label.config(text=f"{t:.2f} / {total_t:.2f} s")
        self.redraw()
        
    def on_scale_change(self, value):
        """Handle deformation scale change."""
        scale = float(value)
        self.scale_label.config(text=f"x{int(scale)}")
        self.redraw()
        
    def redraw(self):
        """Redraw the 3D model."""
        self.ax.clear()
        
        if not self.nodes or not self.elements:
            self.canvas.draw()
            return
        
        # Get current displacement if available
        if self.displacement_history and self.current_frame < len(self.displacement_history):
            u = self.displacement_history[self.current_frame]
        else:
            u = None
            
        # Get current damage if available
        if self.damage_history and self.current_frame < len(self.damage_history):
            damage = self.damage_history[self.current_frame]
        else:
            damage = None
        
        scale = self.scale_var.get()
        
        # Calculate node positions
        node_positions = {}
        for node in self.nodes:
            if u is not None:
                dx = u[node.dof_indices[0]] if node.dof_indices[0] >= 0 else 0
                dy = u[node.dof_indices[1]] if node.dof_indices[1] >= 0 else 0
                dz = u[node.dof_indices[2]] if node.dof_indices[2] >= 0 else 0
                node_positions[node.id] = (
                    node.x + dx * scale,
                    node.y + dy * scale,
                    node.z + dz * scale
                )
            else:
                node_positions[node.id] = (node.x, node.y, node.z)
        
        # Draw elements with type-based coloring
        for i, el in enumerate(self.elements):
            p1 = node_positions.get(el.node_i.id, (0, 0, 0))
            p2 = node_positions.get(el.node_j.id, (0, 0, 0))
            
            # Determine element type and default color/width
            elem_type = type(el).__name__
            is_vertical = abs(p1[2] - p2[2]) > 0.1  # Column if vertical
            is_horizontal = abs(p1[2] - p2[2]) < 0.1  # Beam if horizontal
            
            # Default colors by element type (matches reference images)
            if 'Wall' in elem_type or (hasattr(el, 'custom_color') and el.custom_color == 'red'):
                base_color = '#FF4444'  # Red for walls
                linewidth = 1.5
            elif 'Isolator' in elem_type:
                base_color = '#FF00FF'  # Magenta for isolators
                linewidth = 4
            elif 'Damper' in elem_type:
                base_color = '#00FFFF'  # Cyan for dampers
                linewidth = 3
            elif is_vertical:
                base_color = '#4DABF7'  # Light blue for columns
                linewidth = 2.5
            else:
                # Horizontal beams - distinguish X and Y direction
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])
                if dx > dy:
                    base_color = '#69DB7C'  # Green for X-beams
                else:
                    base_color = '#FFD43B'  # Yellow for Y-beams
                linewidth = 2
            
            # Apply damage coloring if available
            if damage and i < len(damage):
                d = damage[i]
                if d >= 0.5:  # Only override color if damage is significant
                    base_color = self.get_damage_color(d)
                    linewidth = linewidth + 0.5 if d >= 1.0 else linewidth
            
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        color=base_color, linewidth=linewidth, alpha=0.9)
        
        # Draw nodes
        if self.show_nodes_var.get():
            for node in self.nodes:
                pos = node_positions.get(node.id, (node.x, node.y, node.z))
                marker = 'o' if node.z > 0 else 's'  # Square for base nodes
                color = '#FFFF00' if node.z == 0 else '#00FF00'
                self.ax.scatter(*pos, c=color, s=20, marker=marker)
                
                if self.show_labels_var.get():
                    self.ax.text(pos[0], pos[1], pos[2], f' N{node.id}', 
                               color='white', fontsize=8)
        
        # Draw ground grid
        if self.show_grid_var.get() and self.nodes:
            x_vals = [n.x for n in self.nodes]
            y_vals = [n.y for n in self.nodes]
            x_min, x_max = min(x_vals) - 2, max(x_vals) + 2
            y_min, y_max = min(y_vals) - 2, max(y_vals) + 2
            
            # Draw grid lines
            grid_step = 5  # 5m grid
            for x in range(int(x_min // grid_step) * grid_step, int(x_max) + grid_step, grid_step):
                self.ax.plot([x, x], [y_min, y_max], [0, 0], color='#444444', linewidth=0.5, alpha=0.5)
            for y in range(int(y_min // grid_step) * grid_step, int(y_max) + grid_step, grid_step):
                self.ax.plot([x_min, x_max], [y, y], [0, 0], color='#444444', linewidth=0.5, alpha=0.5)
            
            # Draw axes indicators
            self.ax.plot([0, 5], [0, 0], [0, 0], color='#FF0000', linewidth=2)  # X axis - red
            self.ax.plot([0, 0], [0, 5], [0, 0], color='#00FF00', linewidth=2)  # Y axis - green
            self.ax.plot([0, 0], [0, 0], [0, 5], color='#0000FF', linewidth=2)  # Z axis - blue
        
        # Set axis properties
        self.ax.set_xlabel('X (m)', color='white')
        self.ax.set_ylabel('Y (m)', color='white')
        self.ax.set_zlabel('Z (m)', color='white')
        self.ax.tick_params(colors='white')
        
        # Equal aspect ratio
        self.set_axes_equal()
        
        # Apply view settings
        self.ax.view_init(elev=self.settings.elevation, azim=self.settings.azimuth)
        
        self.canvas.draw()
        
    def get_damage_color(self, damage_index: float) -> str:
        """Get color based on damage index."""
        if damage_index < 0.5:
            return '#00FF00'  # Green - safe
        elif damage_index < 1.0:
            return '#88FF00'  # Yellow-green - pre-yield
        elif damage_index < 1.5:
            return '#FFFF00'  # Yellow - yielded
        elif damage_index < 2.0:
            return '#FF8800'  # Orange - moderate damage
        else:
            return '#FF0000'  # Red - severe damage
            
    def set_axes_equal(self):
        """Set 3D axes to equal scale for proper visualization."""
        if not self.nodes:
            return
            
        x_vals = [n.x for n in self.nodes]
        y_vals = [n.y for n in self.nodes]
        z_vals = [n.z for n in self.nodes]
        
        if self.equal_aspect_var.get():
            # Equal aspect ratio - use max range for all axes
            max_range = max(
                max(x_vals) - min(x_vals),
                max(y_vals) - min(y_vals),
                max(z_vals) - min(z_vals)
            ) / 2.0 + 2  # margin
            
            mid_x = (max(x_vals) + min(x_vals)) / 2
            mid_y = (max(y_vals) + min(y_vals)) / 2
            
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(0, max(z_vals) * 1.1)
        else:
            # Auto-fit to data
            margin = 2
            self.ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
            self.ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)
            self.ax.set_zlim(0, max(z_vals) * 1.2)
    
    def set_realtime_duration(self, duration: float):
        """Set animation duration for real-time sync."""
        self.animation_duration = duration
        if self.displacement_history:
            num_frames = len(self.displacement_history)
            # Calculate interval to match real time
            self.realtime_interval = int((duration * 1000) / num_frames)
            # Ensure minimum interval of 16ms (60fps max)
            self.realtime_interval = max(16, self.realtime_interval)
    
    def start_animation(self):
        """Start automatic animation playback with real-time sync."""
        if not self.displacement_history:
            return
        self.is_animating = True
        self.play_btn.config(text="⏸ 一時")
        self._animate_step()
    
    def stop_animation(self):
        """Stop animation and reset to start."""
        self.is_animating = False
        if self.animation_job:
            self.after_cancel(self.animation_job)
            self.animation_job = None
        self.current_frame = 0
        self.time_var.set(0)
        self.play_btn.config(text="▶ 再生")
        self.redraw()
    
    def toggle_animation(self):
        """Toggle between play and pause."""
        if self.is_animating:
            self.pause_animation()
        else:
            self.start_animation()
    
    def pause_animation(self):
        """Pause animation at current frame."""
        self.is_animating = False
        if self.animation_job:
            self.after_cancel(self.animation_job)
            self.animation_job = None
        self.play_btn.config(text="▶ 再生")
    
    def _animate_step(self):
        """One step of the animation loop."""
        if not self.is_animating or not self.displacement_history:
            return
        
        # Frame skip for performance - skip more frames for large datasets
        n_frames = len(self.displacement_history)
        frame_skip = max(1, n_frames // 200)  # Max ~200 frames for smooth playback
        
        # Update frame with skip
        self.current_frame = (self.current_frame + frame_skip) % n_frames
        self.time_var.set(self.current_frame)
        
        # Update time display
        if hasattr(self, 'time_array') and len(self.time_array) > self.current_frame:
            t = self.time_array[self.current_frame]
            total_t = self.time_array[-1] if len(self.time_array) > 0 else self.animation_duration
            self.time_label.config(text=f"{t:.2f} / {total_t:.2f} s")
        
        self.redraw()
        
        # Schedule next frame - minimum 50ms interval for smoother UI
        interval = max(50, self.realtime_interval)
        self.animation_job = self.after(interval, self._animate_step)
    
    def animate(self, interval: int = 50):
        """Legacy: Start/continue animation."""
        self.set_realtime_duration(5.0)
        self.start_animation()


class DamageColorLegend(ttk.Frame):
    """Legend for damage color scale."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()
        
    def create_widgets(self):
        ttk.Label(self, text="損傷度", font=('Arial', 10, 'bold')).pack()
        
        colors = [
            ('#00FF00', '安全 (< 0.5)'),
            ('#88FF00', '降伏前 (0.5-1.0)'),
            ('#FFFF00', '降伏 (1.0-1.5)'),
            ('#FF8800', '中損傷 (1.5-2.0)'),
            ('#FF0000', '大損傷 (> 2.0)'),
        ]
        
        for color, label in colors:
            row = ttk.Frame(self)
            row.pack(fill=tk.X, pady=1)
            
            color_box = tk.Label(row, bg=color, width=2, height=1)
            color_box.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(row, text=label, font=('Arial', 8)).pack(side=tk.LEFT)


def create_3d_viewer_tab(parent, layout=None) -> Building3DViewer:
    """
    Create a 3D viewer tab with all controls.
    
    Args:
        parent: Parent widget
        layout: Optional BuildingLayout to display
        
    Returns:
        Building3DViewer instance
    """
    main_frame = ttk.Frame(parent)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Viewer
    viewer = Building3DViewer(main_frame)
    viewer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Side panel
    side_panel = ttk.Frame(main_frame, width=150)
    side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
    side_panel.pack_propagate(False)
    
    # Legend
    legend = DamageColorLegend(side_panel)
    legend.pack(pady=10)
    
    # Info panel
    info_frame = ttk.LabelFrame(side_panel, text="モデル情報")
    info_frame.pack(fill=tk.X, pady=10)
    
    ttk.Label(info_frame, text="節点数: -").pack(anchor=tk.W)
    ttk.Label(info_frame, text="要素数: -").pack(anchor=tk.W)
    ttk.Label(info_frame, text="自由度: -").pack(anchor=tk.W)
    
    return viewer
