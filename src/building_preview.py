"""
3D Building Preview Module.
Visualizes building structure from layout before simulation.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class PreviewNode:
    """Node for preview rendering."""
    x: float
    y: float
    z: float
    
    
@dataclass
class PreviewElement:
    """Element for preview rendering."""
    node_i: PreviewNode
    node_j: PreviewNode
    element_type: str  # 'column', 'beam_x', 'beam_y', 'brace'
    section_name: str = ""


class Building3DPreview(tk.Canvas):
    """
    3D building preview canvas.
    
    Shows wireframe view of building structure.
    Supports rotation, zoom, and pan.
    """
    
    def __init__(self, parent, width=500, height=400, **kwargs):
        super().__init__(parent, width=width, height=height, bg='#1a1a2e', **kwargs)
        
        self.width = width
        self.height = height
        
        # Building data
        self.nodes: List[PreviewNode] = []
        self.elements: List[PreviewElement] = []
        
        # View parameters
        self.rotation_x = 25  # degrees
        self.rotation_z = 45  # degrees
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Camera parameters
        self.camera_distance = 50.0  # Distance from center
        self.use_perspective = True  # Toggle perspective/orthographic
        self.fov = 60  # Field of view for perspective
        
        # Center of building
        self.center_x = 0
        self.center_y = 0
        self.center_z = 0
        
        # Colors
        self.colors = {
            'column': '#4dabf7',
            'beam_x': '#69db7c',
            'beam_y': '#ffd43b',
            'brace': '#ff6b6b',
            'floor': '#3a3a50',
            'grid': '#333355'
        }
        
        # Mouse interaction
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.bind('<Button-1>', self._on_mouse_down)
        self.bind('<B1-Motion>', self._on_mouse_drag)
        self.bind('<MouseWheel>', self._on_mouse_wheel)
        self.bind('<Button-3>', self._on_right_click)
        self.bind('<B3-Motion>', self._on_right_drag)
        self.bind('<Button-2>', self._on_middle_click)  # Middle button for camera move
        self.bind('<B2-Motion>', self._on_middle_drag)
        
        # Keyboard bindings
        self.bind('<Key>', self._on_key)
        self.focus_set()
        
        self._draw_empty_state()
        
    def set_building_from_layout(self, layout, story_height: float = 3.5):
        """
        Set building from BuildingLayout object.
        
        Args:
            layout: BuildingLayout object with floors and grid
            story_height: Height of each story (m)
        """
        self.nodes = []
        self.elements = []
        
        if layout is None:
            self._draw_empty_state()
            return
            
        grid = layout.grid
        floors_dict = layout.floors  # Dict[int, FloorLayout]
        
        if not floors_dict:
            self._draw_empty_state()
            return
        
        # Get x and y coordinates from grid
        x_coords = grid.get_x_coords()  # [0, 6, 12, 18, ...]
        y_coords = grid.get_y_coords()
        
        # Get story heights
        story_heights = grid.story_heights if grid.story_heights else [story_height] * len(floors_dict)
        
        # Create nodes at each column position for each floor
        node_map = {}  # (floor_idx, gx, gy) -> PreviewNode
        
        # Sorted floor indices
        floor_indices = sorted(floors_dict.keys())
        
        for floor_idx in floor_indices:
            floor = floors_dict[floor_idx]
            
            # Calculate z height
            z = sum(story_heights[:floor_idx]) if floor_idx <= len(story_heights) else floor_idx * story_height
            
            # columns is Dict[(gx, gy), section_name]
            for (gx, gy), section_name in floor.columns.items():
                if gx < len(x_coords) and gy < len(y_coords):
                    x = x_coords[gx]
                    y = y_coords[gy]
                    
                    node = PreviewNode(x, y, z)
                    self.nodes.append(node)
                    node_map[(floor_idx, gx, gy)] = node
                
        # Create elements
        for floor_idx in floor_indices:
            floor = floors_dict[floor_idx]
            
            # Columns (vertical elements to floor above)
            next_floor_idx = floor_idx + 1
            if next_floor_idx in floors_dict:
                next_floor = floors_dict[next_floor_idx]
                for (gx, gy) in floor.columns.keys():
                    key_bottom = (floor_idx, gx, gy)
                    key_top = (next_floor_idx, gx, gy)
                    
                    # Only add column if there's a column at the same position above
                    if key_bottom in node_map and key_top in node_map:
                        self.elements.append(PreviewElement(
                            node_map[key_bottom],
                            node_map[key_top],
                            'column'
                        ))
                        
            # Beams (horizontal elements) - from floor.beams dict
            for ((gx1, gy1), (gx2, gy2)), section in floor.beams.items():
                key1 = (floor_idx, gx1, gy1)
                key2 = (floor_idx, gx2, gy2)
                
                if key1 in node_map and key2 in node_map:
                    # Determine beam direction
                    if gx1 != gx2:
                        elem_type = 'beam_x'
                    else:
                        elem_type = 'beam_y'
                        
                    self.elements.append(PreviewElement(
                        node_map[key1],
                        node_map[key2],
                        elem_type
                    ))
                        
        self._calculate_center()
        self._auto_zoom()
        self.render()

        
    def set_building_simple(
        self,
        n_stories: int,
        n_bays_x: int,
        n_bays_y: int,
        story_height: float = 3.5,
        bay_width_x: float = 6.0,
        bay_width_y: float = 6.0,
        with_braces: bool = False
    ):
        """
        Set up a simple rectangular building.
        
        Args:
            n_stories: Number of stories
            n_bays_x: Number of bays in X direction
            n_bays_y: Number of bays in Y direction
            story_height: Height of each story (m)
            bay_width_x: Bay width in X (m)
            bay_width_y: Bay width in Y (m)
            with_braces: Add braces
        """
        self.nodes = []
        self.elements = []
        
        # Create nodes
        node_grid = {}
        for k in range(n_stories + 1):
            z = k * story_height
            for i in range(n_bays_x + 1):
                for j in range(n_bays_y + 1):
                    x = i * bay_width_x
                    y = j * bay_width_y
                    node = PreviewNode(x, y, z)
                    self.nodes.append(node)
                    node_grid[(i, j, k)] = node
                    
        # Create columns
        for k in range(n_stories):
            for i in range(n_bays_x + 1):
                for j in range(n_bays_y + 1):
                    n1 = node_grid[(i, j, k)]
                    n2 = node_grid[(i, j, k + 1)]
                    self.elements.append(PreviewElement(n1, n2, 'column'))
                    
        # Create beams
        for k in range(1, n_stories + 1):
            for i in range(n_bays_x + 1):
                for j in range(n_bays_y):
                    n1 = node_grid[(i, j, k)]
                    n2 = node_grid[(i, j + 1, k)]
                    self.elements.append(PreviewElement(n1, n2, 'beam_y'))
                    
            for i in range(n_bays_x):
                for j in range(n_bays_y + 1):
                    n1 = node_grid[(i, j, k)]
                    n2 = node_grid[(i + 1, j, k)]
                    self.elements.append(PreviewElement(n1, n2, 'beam_x'))
                    
        # Optional braces
        if with_braces:
            for k in range(n_stories):
                # X-direction braces on edges
                for j in [0, n_bays_y]:
                    for i in range(0, n_bays_x, 2):
                        n1 = node_grid[(i, j, k)]
                        n2 = node_grid[(i + 1, j, k + 1)]
                        self.elements.append(PreviewElement(n1, n2, 'brace'))
                        
        self._calculate_center()
        self._auto_zoom()
        self.render()
        
    def _calculate_center(self):
        """Calculate building center."""
        if not self.nodes:
            self.center_x = self.center_y = self.center_z = 0
            return
            
        self.center_x = sum(n.x for n in self.nodes) / len(self.nodes)
        self.center_y = sum(n.y for n in self.nodes) / len(self.nodes)
        self.center_z = sum(n.z for n in self.nodes) / len(self.nodes)
        
    def _auto_zoom(self):
        """Auto-adjust zoom to fit building."""
        if not self.nodes:
            self.zoom = 1.0
            return
            
        max_extent = max(
            max(abs(n.x - self.center_x) for n in self.nodes),
            max(abs(n.y - self.center_y) for n in self.nodes),
            max(abs(n.z - self.center_z) for n in self.nodes)
        )
        
        if max_extent > 0:
            self.zoom = min(self.width, self.height) / (4 * max_extent)
        else:
            self.zoom = 1.0
            
    def _project_point(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Project 3D point to 2D screen coordinates with depth."""
        # Center the point
        x -= self.center_x
        y -= self.center_y
        z -= self.center_z
        
        # Rotate around Z axis
        rad_z = math.radians(self.rotation_z)
        x1 = x * math.cos(rad_z) - y * math.sin(rad_z)
        y1 = x * math.sin(rad_z) + y * math.cos(rad_z)
        z1 = z
        
        # Rotate around X axis  
        rad_x = math.radians(self.rotation_x)
        y2 = y1 * math.cos(rad_x) - z1 * math.sin(rad_x)
        z2 = y1 * math.sin(rad_x) + z1 * math.cos(rad_x)
        x2 = x1
        
        # Depth value for sorting
        depth = y2
        
        if self.use_perspective:
            # Perspective projection
            # Camera at (0, -camera_distance, 0) looking at origin
            dist = self.camera_distance - y2
            if dist < 1:
                dist = 1
            fov_factor = math.tan(math.radians(self.fov / 2))
            scale = (self.height / 2) / (dist * fov_factor) * self.zoom
            
            screen_x = self.width / 2 + x2 * scale + self.pan_x
            screen_y = self.height / 2 - z2 * scale + self.pan_y
        else:
            # Orthographic projection
            screen_x = self.width / 2 + x2 * self.zoom + self.pan_x
            screen_y = self.height / 2 - z2 * self.zoom + self.pan_y
        
        return screen_x, screen_y, depth
        
    def render(self):
        """Render the building."""
        self.delete('all')
        
        if not self.elements:
            self._draw_empty_state()
            return
            
        # Draw floor plates (semi-transparent)
        self._draw_floor_plates()
        
        # Sort elements by depth for proper occlusion
        def element_depth(elem):
            mid_y = (elem.node_i.y + elem.node_j.y) / 2
            mid_x = (elem.node_i.x + elem.node_j.x) / 2
            rad_z = math.radians(self.rotation_z)
            return mid_x * math.sin(rad_z) + mid_y * math.cos(rad_z)
            
        sorted_elements = sorted(self.elements, key=element_depth)
        
        # Draw elements
        for elem in sorted_elements:
            x1, y1, _ = self._project_point(elem.node_i.x, elem.node_i.y, elem.node_i.z)
            x2, y2, _ = self._project_point(elem.node_j.x, elem.node_j.y, elem.node_j.z)
            
            color = self.colors.get(elem.element_type, '#ffffff')
            width = 3 if elem.element_type == 'column' else 2
            
            self.create_line(x1, y1, x2, y2, fill=color, width=width)
            
        # Draw nodes
        for node in self.nodes:
            x, y, _ = self._project_point(node.x, node.y, node.z)
            r = 3
            self.create_oval(x-r, y-r, x+r, y+r, fill='#ffffff', outline='')
            
        # Draw legend
        self._draw_legend()
        
        # Draw info
        self._draw_info()
        
    def _draw_floor_plates(self):
        """Draw semi-transparent floor plates."""
        if not self.nodes:
            return
            
        # Group nodes by floor (Z level)
        floors = {}
        for node in self.nodes:
            z = round(node.z, 2)
            if z not in floors:
                floors[z] = []
            floors[z].append(node)
            
        for z, floor_nodes in floors.items():
            if len(floor_nodes) < 3:
                continue
                
            # Get convex hull (simplified: bounding box)
            xs = [n.x for n in floor_nodes]
            ys = [n.y for n in floor_nodes]
            
            corners = [
                (min(xs), min(ys), z),
                (max(xs), min(ys), z),
                (max(xs), max(ys), z),
                (min(xs), max(ys), z),
            ]
            
            screen_points = []
            for x, y, zc in corners:
                sx, sy, _ = self._project_point(x, y, zc)
                screen_points.extend([sx, sy])
                
            if len(screen_points) >= 6:
                self.create_polygon(screen_points, fill=self.colors['floor'], outline='')
                
    def _draw_legend(self):
        """Draw color legend."""
        legend_items = [
            ('Column', 'column'),
            ('Beam-X', 'beam_x'),
            ('Beam-Y', 'beam_y'),
            ('Brace', 'brace'),
        ]
        
        x, y = 10, 10
        for label, elem_type in legend_items:
            color = self.colors.get(elem_type, '#ffffff')
            self.create_line(x, y+6, x+20, y+6, fill=color, width=2)
            self.create_text(x+25, y+6, text=label, anchor='w', fill='#aaaaaa', font=('Arial', 9))
            y += 18
            
    def _draw_info(self):
        """Draw building info."""
        info = f"Nodes: {len(self.nodes)}  Elements: {len(self.elements)}"
        self.create_text(
            self.width - 10, self.height - 10,
            text=info, anchor='se', fill='#666666', font=('Arial', 9)
        )
        
        # Controls hint and projection mode
        proj_mode = "Perspective" if self.use_perspective else "Orthographic"
        hint = f"Drag: Rotate | Right: Pan | Scroll: Zoom | P: Toggle [{proj_mode}] | WASD: Move"
        self.create_text(
            self.width / 2, self.height - 10,
            text=hint, anchor='s', fill='#444444', font=('Arial', 8)
        )
        
    def _draw_empty_state(self):
        """Draw empty state message."""
        self.create_text(
            self.width / 2, self.height / 2,
            text="建物データがありません\n\nレイアウトを設定してください",
            fill='#555555', font=('Arial', 12), justify='center'
        )
        
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
    def _on_mouse_drag(self, event):
        """Handle mouse drag for rotation."""
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        self.rotation_z += dx * 0.5
        self.rotation_x += dy * 0.5
        self.rotation_x = max(-90, min(90, self.rotation_x))
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        self.render()
        
    def _on_right_click(self, event):
        """Handle right mouse button."""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
    def _on_right_drag(self, event):
        """Handle right-drag for panning."""
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        self.pan_x += dx
        self.pan_y += dy
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        self.render()
        
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom."""
        if event.delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
            
        self.zoom = max(0.1, min(10, self.zoom))
        self.render()
        
    def _on_middle_click(self, event):
        """Handle middle mouse button."""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
    def _on_middle_drag(self, event):
        """Handle middle-drag for camera distance."""
        dy = event.y - self.last_mouse_y
        self.camera_distance += dy * 0.5
        self.camera_distance = max(10, min(200, self.camera_distance))
        self.last_mouse_y = event.y
        self.render()
        
    def _on_key(self, event):
        """Handle keyboard input."""
        key = event.keysym.lower()
        
        # Camera movement with WASD
        move_speed = 2
        if key == 'w':
            self.pan_y += move_speed * 5
        elif key == 's':
            self.pan_y -= move_speed * 5
        elif key == 'a':
            self.pan_x += move_speed * 5
        elif key == 'd':
            self.pan_x -= move_speed * 5
        elif key == 'q':
            self.camera_distance -= 3
        elif key == 'e':
            self.camera_distance += 3
        # Toggle perspective
        elif key == 'p':
            self.use_perspective = not self.use_perspective
        # Reset
        elif key == 'r':
            self.reset_view()
            return
            
        self.camera_distance = max(10, min(200, self.camera_distance))
        self.render()
        
    def toggle_perspective(self):
        """Toggle between perspective and orthographic projection."""
        self.use_perspective = not self.use_perspective
        self.render()
        
    def reset_view(self):
        """Reset view to default."""
        self.rotation_x = 25
        self.rotation_z = 45
        self.pan_x = 0
        self.pan_y = 0
        self.camera_distance = 50.0
        self._auto_zoom()
        self.render()
        
    def set_view(self, preset: str):
        """Set predefined view angle."""
        presets = {
            'front': (0, 0),
            'back': (0, 180),
            'left': (0, -90),
            'right': (0, 90),
            'top': (90, 0),
            'iso': (30, 45),
        }
        if preset in presets:
            self.rotation_x, self.rotation_z = presets[preset]
            self.render()


class BuildingPreviewPanel(ttk.Frame):
    """
    Panel containing 3D preview with controls.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Preview canvas
        self.preview = Building3DPreview(self, width=500, height=400)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Reset", command=self.preview.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Front", command=lambda: self.preview.set_view('front')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Top", command=lambda: self.preview.set_view('top')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Iso", command=lambda: self.preview.set_view('iso')).pack(side=tk.LEFT, padx=2)
        
    def update_preview(self, layout=None, **kwargs):
        """Update preview with layout or simple parameters."""
        if layout:
            self.preview.set_building_from_layout(layout, **kwargs)
        else:
            self.preview.set_building_simple(**kwargs)


def show_preview_window(parent, layout=None, title="3D Building Preview"):
    """
    Show preview in a new window.
    
    Args:
        parent: Parent window
        layout: BuildingLayout object (optional)
        title: Window title
    """
    window = tk.Toplevel(parent)
    window.title(title)
    window.geometry("600x500")
    
    panel = BuildingPreviewPanel(window)
    panel.pack(fill=tk.BOTH, expand=True)
    
    if layout:
        panel.preview.set_building_from_layout(layout)
    else:
        # Demo building
        panel.preview.set_building_simple(
            n_stories=5,
            n_bays_x=3,
            n_bays_y=2,
            with_braces=True
        )
        
    return window, panel
