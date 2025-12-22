import tkinter as tk
from tkinter import ttk
import math
from .layout_model import BuildingLayout, GridSystem, SectionProperties
from .logging_config import get_logger

logger = get_logger("gui_layout")

class LayoutEditorPanel(ttk.Frame):
    def __init__(self, parent, layout, on_change_callback=None):
        super().__init__(parent)
        self.layout = layout
        self.on_change_callback = on_change_callback # New callback
        self.current_floor = 1
        self.scale = 20.0
        self.offset_x = 50.0
        self.offset_y = 50.0
        self.mode = "column" # or "beam"
        
        self.create_widgets()
        
    def create_widgets(self):
        # Layout: Left Control Panel, Right Canvas
        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        self.controls = ttk.Frame(self.paned, width=200, padding=5)
        self.paned.add(self.controls)
        
        self.canvas = tk.Canvas(self.paned, bg='white', cursor="cross")
        self.paned.add(self.canvas)
        
        # Controls
        # File Operations
        lbl_file = ttk.Label(self.controls, text="File Ops:", font=('Arial', 10, 'bold'))
        lbl_file.pack(anchor=tk.W, pady=(0, 5))
        
        btn_frame = ttk.Frame(self.controls)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Save Layout", command=self.save_layout).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_frame, text="Load Layout", command=self.load_layout).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # 3D Preview button
        ttk.Button(self.controls, text="🏢 3D Preview", command=self.show_3d_preview).pack(fill=tk.X, pady=5)
        

        
        # Floor Selector
        ttk.Label(self.controls, text="Floor Selector").pack(anchor=tk.W)
        self.floor_var = tk.StringVar(value="1F")
        self.floor_combo = ttk.Combobox(self.controls, textvariable=self.floor_var, state="readonly")
        self.floor_combo.pack(fill=tk.X, pady=2)
        self.floor_combo.bind("<<ComboboxSelected>>", self.on_floor_change)
        
        ttk.Button(self.controls, text="Copy from Below", command=self.copy_floor_below).pack(fill=tk.X, pady=5)
        
        ttk.Separator(self.controls, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Floor Management
        ttk.Label(self.controls, text="Floor Mgmt:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Button(self.controls, text="Add Floor Top", command=self.add_floor).pack(fill=tk.X, pady=2)
        ttk.Button(self.controls, text="Remove Top Floor", command=self.remove_floor).pack(fill=tk.X, pady=2)
        
        # Grid Configuration
        ttk.Separator(self.controls, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.controls, text="Grid Config (m):").pack(anchor=tk.W)
        
        ttk.Label(self.controls, text="X Grid (comma sep):").pack(anchor=tk.W)
        self.entry_x_grid = ttk.Entry(self.controls)
        self.entry_x_grid.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.controls, text="Y Grid (comma sep):").pack(anchor=tk.W)
        self.entry_y_grid = ttk.Entry(self.controls)
        self.entry_y_grid.pack(fill=tk.X, pady=2)
        
        ttk.Button(self.controls, text="Update Grid", command=self.update_grid_config).pack(fill=tk.X, pady=5)
        
        ttk.Separator(self.controls, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(self.controls, text="Instructions:").pack(anchor=tk.W)
        ttk.Label(self.controls, text="Click Node: Toggle Column").pack(anchor=tk.W)
        ttk.Label(self.controls, text="Click Line: Toggle Beam/Wall").pack(anchor=tk.W)
        
        ttk.Separator(self.controls, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.controls, text="Section Colors:").pack(anchor=tk.W)
        self.legend_frame = ttk.Frame(self.controls)
        self.legend_frame.pack(fill=tk.X)
        self.update_legend()
        
        # Events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.draw_layout)
        
        self.scale = 40.0 # Pixels per meter
        self.offset_x = 50
        self.offset_y = 50
        
        self.refresh_ui()

    def refresh_ui(self):
        # Update floor selector
        floors = list(self.layout.floors.keys())
        floors.sort()
        values = [f"{f}F" for f in floors]
        self.floor_combo['values'] = values
        if self.current_floor not in floors and floors:
             self.current_floor = floors[0]
        self.floor_combo.set(f"{self.current_floor}F")
        
        # Update Grid Config Entries
        if self.layout.grid:
             x_str = ", ".join([str(x) for x in self.layout.grid.x_spacings])
             y_str = ", ".join([str(x) for x in self.layout.grid.y_spacings])
             
             # Only update if not focused? Or always?
             # Always update to reflect current state
             self.entry_x_grid.delete(0, tk.END)
             self.entry_x_grid.insert(0, x_str)
             
             self.entry_y_grid.delete(0, tk.END)
             self.entry_y_grid.insert(0, y_str)
        
        self.draw_layout()

    def update_legend(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
            
        for name, prop in self.layout.sections.items():
            f = ttk.Frame(self.legend_frame)
            f.pack(fill=tk.X, pady=2)
            lbl = tk.Label(f, bg=prop.color, width=2)
            lbl.pack(side=tk.LEFT)
            ttk.Label(f, text=name).pack(side=tk.LEFT, padx=5)

    def on_floor_change(self, event):
        val = self.floor_var.get()
        self.current_floor = int(val.replace("F", ""))
        self.draw_layout()
        
    def copy_from_below(self):
        if self.current_floor > 1:
            src = self.layout.get_floor(self.current_floor - 1)
            dst = self.layout.get_floor(self.current_floor)
            dst.columns = src.columns.copy()
            dst.beams = src.beams.copy()
            self.draw_layout()

    def update_grid_config(self):
        try:
            x_str = self.entry_x_grid.get()
            y_str = self.entry_y_grid.get()
            
            # Parse CSV
            new_x = [float(s.strip()) for s in x_str.split(',') if s.strip()]
            new_y = [float(s.strip()) for s in y_str.split(',') if s.strip()]
            
            if not new_x or not new_y:
                 print("Grid must have at least one span.")
                 # Could show error messagebox
                 return
                 
            self.layout.grid.x_spacings = new_x
            self.layout.grid.y_spacings = new_y
            
            # Cleanup out of bounds
            self.layout.cleanup_elements()
            
            self.refresh_ui()
            
        except ValueError:
            logger.warning("Invalid Grid Input")

    def save_layout(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(defaultextension=".json", 
                                                filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                self.layout.save_to_file(filename)
                logger.info(f"Layout saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving layout: {e}")

    def load_layout(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                new_layout = BuildingLayout.load_from_file(filename)
                self.layout = new_layout
                self.refresh_ui()
                logger.info(f"Layout loaded from {filename}")
                if self.on_change_callback:
                    self.on_change_callback()
            except Exception as e:
                logger.error(f"Error loading layout: {e}")

    def show_3d_preview(self):
        """Show 3D preview window of current building layout."""
        try:
            from .building_preview import show_preview_window, Building3DPreview
            
            window = tk.Toplevel(self)
            window.title("3D Building Preview")
            window.geometry("600x500")
            
            # Create preview
            preview = Building3DPreview(window, width=600, height=450)
            preview.pack(fill=tk.BOTH, expand=True)
            
            # Set building from current layout
            preview.set_building_from_layout(self.layout, story_height=3.5)
            
            # Control buttons
            btn_frame = ttk.Frame(window)
            btn_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(btn_frame, text="Reset View", command=preview.reset_view).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="Front", command=lambda: preview.set_view('front')).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="Top", command=lambda: preview.set_view('top')).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="Isometric", command=lambda: preview.set_view('iso')).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="Close", command=window.destroy).pack(side=tk.RIGHT, padx=2)
            
            logger.info("3D Preview window opened")
        except Exception as e:
            logger.error(f"Error opening 3D preview: {e}")
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to open 3D preview:\n{e}")

    def add_floor(self):
        # Add new floor index = max + 1
        n_stories = len(self.layout.grid.story_heights)
        new_story_idx = n_stories + 1
        
        # Add story height (default 3.5m)
        self.layout.grid.story_heights.append(3.5)
        
        # Initialize new floor
        # Use previous floor's beams/cols as default? Or empty?
        # User requested "Copy from Below" button, so here just empty or simple grid.
        self.layout.get_floor(new_story_idx) # Creates it
        
        # Switch to it
        self.current_floor = new_story_idx
        
        self.refresh_ui()
        if self.on_change_callback:
            self.on_change_callback()

    def remove_floor(self):
        n_stories = len(self.layout.grid.story_heights)
        if n_stories > 1:
            self.layout.grid.story_heights.pop()
            if n_stories in self.layout.floors:
                del self.layout.floors[n_stories]
            
            # Update current floor if it was the top one
            if self.current_floor >= n_stories:
                self.current_floor = n_stories - 1
            
            self.refresh_ui()
            if self.on_change_callback:
                self.on_change_callback()

    def copy_floor_below(self):
        if self.current_floor > 1:
            src = self.layout.get_floor(self.current_floor - 1)
            dst = self.layout.get_floor(self.current_floor)
            
            dst.columns = src.columns.copy()
            dst.beams = src.beams.copy()
            self.draw_layout()
            # No callback needed if floor count doesn't change?
            # Or maybe we want to trigger something? Not strictly required by requested feature.
            
    def world_to_screen(self, x, y):
        sx = self.offset_x + x * self.scale
        # Invert Y for screen? No, keep plan view intuitive. 
        # Usually screen Y is down. Physical Y is "Up" in plan?
        # Let's map Physical Y (up) to Screen Y (down) inverted?
        # Grid (0,0) at bottom-left corner of canvas?
        # Let's verify grid coords. GridSystem 0,0 is origin.
        # Let's map (0,0) to bottom-left.
        h = self.canvas.winfo_height()
        sy = h - (self.offset_y + y * self.scale)
        return sx, sy
        
    def screen_to_grid(self, sx, sy):
        # Find nearest grid node or line segment
        grid = self.layout.grid
        x_coords = grid.get_x_coords()
        y_coords = grid.get_y_coords()
        
        # Brute force search for nearest node within simplified radius
        click_radius = 15
        
        nearest_node = None
        min_dist = float('inf')
        
        for ix, x in enumerate(x_coords):
            for iy, y in enumerate(y_coords):
                cx, cy = self.world_to_screen(x, y)
                dist = math.sqrt((cx-sx)**2 + (cy-sy)**2)
                if dist < click_radius:
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = (ix, iy)
        
        if nearest_node:
            return "node", nearest_node
            
        # Check beams (horizontal/vertical segments)
        beam_radius = 10
        # X-beams (horizontal)
        for iy, y in enumerate(y_coords):
            _, scr_y = self.world_to_screen(0, y)
             # Check Y proximity first
            if abs(scr_y - sy) < beam_radius:
                 # iterate spans
                 for ix, x in enumerate(x_coords[:-1]):
                     x1 = x
                     x2 = x_coords[ix+1]
                     scr_x1, _ = self.world_to_screen(x1, y)
                     scr_x2, _ = self.world_to_screen(x2, y)
                     # Start/End are sorted? world_to_screen x is usually increasing
                     if min(scr_x1, scr_x2) <= sx <= max(scr_x1, scr_x2):
                         return "beam", ((ix, iy), (ix+1, iy))
                         
        # Y-beams (vertical)
        for ix, x in enumerate(x_coords):
            scr_x, _ = self.world_to_screen(x, 0)
            if abs(scr_x - sx) < beam_radius:
                for iy, y in enumerate(y_coords[:-1]):
                    y1 = y
                    y2 = y_coords[iy+1]
                    _, scr_y1 = self.world_to_screen(x, y1)
                    _, scr_y2 = self.world_to_screen(x, y2)
                    # Screen Y is inverted, check min/max
                    if min(scr_y1, scr_y2) <= sy <= max(scr_y1, scr_y2):
                        return "beam", ((ix, iy), (ix, iy+1))
                        
        return None, None

    def on_canvas_click(self, event):
        type, target = self.screen_to_grid(event.x, event.y)
        floor = self.layout.get_floor(self.current_floor)
        
        if type == "node":
            gx, gy = target
            # Cycle Column
            current = floor.columns.get((gx, gy))
            # Sequence: C1 -> C2 -> None -> C1
            # Or get keys starting with C
            c_types = [k for k in self.layout.sections.keys() if k.startswith("C")]
            c_types.sort()
            
            if current is None:
                new_type = c_types[0] if c_types else None
            else:
                try:
                    idx = c_types.index(current)
                    if idx + 1 < len(c_types):
                        new_type = c_types[idx+1]
                    else:
                        new_type = None # Toggle off
                except ValueError:
                     new_type = None
            
            if new_type:
                floor.add_column(gx, gy, new_type)
            else:
                floor.remove_column(gx, gy)
                
        elif type == "beam":
            p1, p2 = target
            current = floor.beams.get((p1, p2))
            if not current:
                # Order matters for key retrieval? floor beam dict keys are sorted p1, p2?
                # My data model helper handles sorting.
                # Here we reconstruct tuple.
                if p1 > p2: p1, p2 = p2, p1
                current = floor.beams.get((p1, p2))

            b_types = [k for k in self.layout.sections.keys() if k.startswith("B") or k.startswith("W")]
            b_types.sort() # B1, B2, W1 ...
            
            if current is None:
                new_type = b_types[0] if b_types else None
            else:
                try:
                    idx = b_types.index(current)
                    if idx + 1 < len(b_types):
                        new_type = b_types[idx+1]
                    else:
                        new_type = None
                except ValueError:
                    new_type = None
            
            if new_type:
                floor.add_beam(p1, p2, new_type)
            else:
                floor.remove_beam(p1, p2)
                
        self.draw_layout()

    def draw_layout(self, event=None):
        self.canvas.delete("all")
        self.canvas.config(bg='black') # STERA 3D Style
        
        if not self.layout:
            return
            
        # Recalculate scale to fit
        self.fit_view()
        
        self.draw_grid()
        self.draw_elements()

    def draw_grid(self):
        # Draw Grid Lines
        x_coords = self.layout.grid.get_x_coords()
        y_coords = self.layout.grid.get_y_coords()
        
        h = self.canvas.winfo_height()
        w = self.canvas.winfo_width()
        
        # Draw X Lines (Vertical)
        for x in x_coords:
            sx, _ = self.world_to_screen(x, 0)
            self.canvas.create_line(sx, 0, sx, h, fill='#444444', dash=(2, 4))
            
        # Draw Y Lines (Horizontal)
        for y in y_coords:
            _, sy = self.world_to_screen(0, y)
            self.canvas.create_line(0, sy, w, sy, fill='#444444', dash=(2, 4))
            
        # Draw Dimensions
        # X Dimensions (Top)
        for i, val in enumerate(self.layout.grid.x_spacings):
            x1 = x_coords[i]
            x2 = x_coords[i+1]
            center_x = (x1 + x2) / 2
            
            sx, sy = self.world_to_screen(center_x, y_coords[-1]) 
            # Place above the top grid line
            # sy is screen Y of top line. We want higher (smaller screen Y).
            
            # Draw dimension box/text
            text = f"{int(val*1000)}" # mm usually
            self.canvas.create_text(sx, sy - 20, text=text, fill='white', font=('Arial', 10, 'bold'))
            
            # Draw arrows/lines? For now just text centered.
            # Optional: Box bg
            # self.canvas.create_rectangle(sx-20, sy-30, sx+20, sy-10, fill='white')
            # self.canvas.create_text(sx, sy-20, text=text, fill='black')
        
        # Y Dimensions (Left)
        for j, val in enumerate(self.layout.grid.y_spacings):
            y1 = y_coords[j]
            y2 = y_coords[j+1]
            center_y = (y1 + y2) / 2
            
            sx, sy = self.world_to_screen(x_coords[0], center_y)
            # Place left of the left grid line
            
            text = f"{int(val*1000)}"
            self.canvas.create_text(sx - 30, sy, text=text, fill='white', font=('Arial', 10, 'bold'))

    def draw_elements(self):
        floor = self.layout.get_floor(self.current_floor)
        
        # Beams
        for ((gx1, gy1), (gx2, gy2)), section_name in floor.beams.items():
            if section_name:
                x1 = self.layout.grid.get_x_coords()[gx1]
                y1 = self.layout.grid.get_y_coords()[gy1]
                x2 = self.layout.grid.get_x_coords()[gx2]
                y2 = self.layout.grid.get_y_coords()[gy2]
                
                sx1, sy1 = self.world_to_screen(x1, y1)
                sx2, sy2 = self.world_to_screen(x2, y2)
                
                # Check Type
                prop = self.layout.sections.get(section_name)
                color = prop.color if prop else 'blue'
                
                width = 4
                if section_name.startswith("W"):
                     color = 'red'
                     width = 6
                
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=width, tags="beam")
                
                # Draw Label
                mid_sx = (sx1 + sx2) / 2
                mid_sy = (sy1 + sy2) / 2
                self.canvas.create_text(mid_sx, mid_sy - 10, text=section_name, fill=color, font=('Arial', 8))
        
        # Columns
        col_size = 20 # px square
        for (gx, gy), section_name in floor.columns.items():
            if section_name:
                x = self.layout.grid.get_x_coords()[gx]
                y = self.layout.grid.get_y_coords()[gy]
                
                sx, sy = self.world_to_screen(x, y)
                
                prop = self.layout.sections.get(section_name)
                color = prop.color if prop else 'green'
                if section_name.startswith("C"): color = "#00AA00" # Active Green
                
                # Draw Rectangle
                self.canvas.create_rectangle(sx - col_size/2, sy - col_size/2, 
                                             sx + col_size/2, sy + col_size/2, 
                                             fill=color, outline='white')
                
                # Draw Label
                self.canvas.create_text(sx, sy, text=section_name, fill='white', font=('Arial', 9, 'bold'))
            else:
                # Draw empty node intersection?
                pass
                
        # Draw Nodes (Intersections) invisible clickable areas? 
        # Or Just Grid clicks?
        # Current logic checks proximity.
        pass

    def fit_view(self):
        # Calculate fit
        # Simple padding
        x_coords = self.layout.grid.get_x_coords()
        y_coords = self.layout.grid.get_y_coords()
        
        if not x_coords or not y_coords: return
        
        width_m = x_coords[-1] - x_coords[0]
        height_m = y_coords[-1] - y_coords[0]
        
        # Margins for Dimensions
        margin_x = 100
        margin_y = 100
        
        c_w = self.canvas.winfo_width() - margin_x * 2
        c_h = self.canvas.winfo_height() - margin_y * 2
        
        if c_w <= 0 or c_h <= 0: return

        # Scale
        scale_x = c_w / width_m if width_m > 0 else 50
        scale_y = c_h / height_m if height_m > 0 else 50
        
        self.scale = min(scale_x, scale_y)
        
        # Center with Offset
        # (0,0) world -> screen
        # Center of drawing in world:
        cx_w = width_m / 2
        cy_w = height_m / 2
        
        screen_cx = self.canvas.winfo_width() / 2
        screen_cy = self.canvas.winfo_height() / 2
        
        # We want: screen_cx = offset_x + cx_w * self.scale
        self.offset_x = screen_cx - cx_w * self.scale
        # screen_cy = h - (offset_y + cy_w * scale)
        # offset_y = h - screen_cy - cy_w * scale = screen_cy - cy_w * scale (if screen_cy is half H)
        self.offset_y = (self.canvas.winfo_height() / 2) - cy_w * self.scale
