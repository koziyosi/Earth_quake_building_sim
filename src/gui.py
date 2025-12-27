import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from PIL import Image, ImageTk
import os
import time
import numpy as np

# Matplotlib embedding
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .layout_model import BuildingLayout, GridSystem
from .gui_layout import LayoutEditorPanel
from .exporter import ResultsExporter, OBJExporter
from .config_loader import get_config
from .logging_config import setup_logging, get_logger
from .property_inspector import PropertyInspectorPanel
from .viewer_3d_enhanced import Building3DViewer, DamageColorLegend
from .section_database import SECTION_DB
from .material_library import MATERIAL_LIB
from .gui_properties import open_properties_editor

import main as sim2d
import main_3d as sim3d

# Initialize logging
logger = get_logger("gui")


def setup_dark_theme(root):
    """Configure dark theme for ttk widgets."""
    style = ttk.Style()
    
    # Try to use a modern theme as base
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    
    # Dark color palette
    bg_dark = '#1e1e1e'
    bg_medium = '#2d2d2d'
    bg_light = '#3c3c3c'
    fg_light = '#e0e0e0'
    fg_dim = '#a0a0a0'
    accent = '#0078d4'
    
    # Configure styles
    style.configure('.', 
                    background=bg_medium, 
                    foreground=fg_light,
                    fieldbackground=bg_light)
    
    style.configure('TFrame', background=bg_medium)
    style.configure('TLabel', background=bg_medium, foreground=fg_light)
    style.configure('TButton', background=bg_light, foreground=fg_light)
    style.configure('TEntry', fieldbackground=bg_light, foreground=fg_light)
    style.configure('TCheckbutton', background=bg_medium, foreground=fg_light)
    style.configure('TRadiobutton', background=bg_medium, foreground=fg_light)
    style.configure('TNotebook', background=bg_medium)
    style.configure('TNotebook.Tab', background=bg_light, foreground=fg_light, padding=[10, 5])
    style.configure('TCombobox', fieldbackground=bg_light, foreground=fg_light)
    style.configure('Horizontal.TProgressbar', background=accent)
    
    # Map for state changes
    style.map('TButton',
              background=[('active', accent), ('pressed', accent)],
              foreground=[('active', 'white')])
    style.map('TNotebook.Tab',
              background=[('selected', accent)],
              foreground=[('selected', 'white')])
    
    # Configure root window
    root.configure(bg=bg_medium)
    
    return style, bg_dark, bg_medium, fg_light

class EarthquakeSimGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EarthQuake Building Sim")
        self.root.geometry("1200x850")
        
        # Apply dark theme
        self.style, self.bg_dark, self.bg_medium, self.fg_light = setup_dark_theme(root)
        
        # Store simulation results for export
        self.last_results = {
            'time': None,
            'displacement': None,
            'nodes': None,
            'elements': None
        }
        
        # Load config
        self.config = get_config()
        
        # --- Control Panel (Left) ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(control_frame, text="Simulation Settings", font=("Helvetica", 14)).pack(pady=10)
        
        # Tabs for Basic / Advanced
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: General
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text='General')
        
        # Model Selection
        ttk.Label(tab1, text="Model Type:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="3D Frame")
        model_combo = ttk.Combobox(tab1, textvariable=self.model_var)
        model_combo['values'] = ("2D Frame", "3D Frame", "Custom Layout")
        model_combo.pack(fill=tk.X, pady=5)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Earthquake Parameters
        ttk.Label(tab1, text="Parameters:").pack(anchor=tk.W, pady=(10,0))
        
        # Input Source
        self.input_method = tk.StringVar(value="Generated")
        ttk.Radiobutton(tab1, text="Generate Synthetic", variable=self.input_method, value="Generated", command=self.toggle_input_ui).pack(anchor=tk.W)
        ttk.Radiobutton(tab1, text="Load File", variable=self.input_method, value="File", command=self.toggle_input_ui).pack(anchor=tk.W)
        
        # Generated Params Frame
        self.gen_frame = ttk.Frame(tab1)
        self.gen_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.gen_frame, text="Max Acc (gal):").pack(anchor=tk.W)
        self.acc_var = tk.DoubleVar(value=400.0)
        ttk.Entry(self.gen_frame, textvariable=self.acc_var).pack(fill=tk.X)
        
        ttk.Label(self.gen_frame, text="Duration (s):").pack(anchor=tk.W)
        self.dur_var = tk.DoubleVar(value=5.0)
        ttk.Entry(self.gen_frame, textvariable=self.dur_var).pack(fill=tk.X)
        
        ttk.Label(self.gen_frame, text="卓越周期 (s):").pack(anchor=tk.W)
        period_frame = ttk.Frame(self.gen_frame)
        period_frame.pack(fill=tk.X)
        self.period_var = tk.DoubleVar(value=0.0)
        ttk.Entry(period_frame, textvariable=self.period_var, width=8).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(period_frame, text="(0=ランダム)", foreground='gray').pack(side=tk.RIGHT)
        
        # File Params Frame
        self.file_frame = ttk.Frame(tab1)
        # self.file_frame.pack(fill=tk.X, pady=5) # Initially hidden
        
        ttk.Label(self.file_frame, text="Earthquake File:").pack(anchor=tk.W)
        self.filename_var = tk.StringVar()
        ttk.Entry(self.file_frame, textvariable=self.filename_var).pack(fill=tk.X)
        ttk.Button(self.file_frame, text="Browse...", command=self.browse_file).pack(fill=tk.X, pady=2)
        
        # Tab 2: Advanced Builder
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text='Building Design')
        
        # Template Selection
        ttk.Label(tab2, text="建物テンプレート:").pack(anchor=tk.W)
        self.template_var = tk.StringVar(value="(カスタム)")
        template_combo = ttk.Combobox(tab2, textvariable=self.template_var)
        template_names = [
            "(カスタム)",
            "低層住宅 (3F)",
            "中層オフィス (7F)",
            "高層オフィス (20F)",
            "病院 (5F, 免震)",
            "学校 (4F)",
            "倉庫 (2F)",
            "ピロティ建物 (5F)",
            "超高層タワー (40F)",
        ]
        template_combo['values'] = template_names
        template_combo.pack(fill=tk.X, pady=5)
        template_combo.bind("<<ComboboxSelected>>", self.on_template_change)
        
        # Template key mapping
        self._template_keys = {
            "低層住宅 (3F)": "low_rise_residential",
            "中層オフィス (7F)": "mid_rise_office",
            "高層オフィス (20F)": "high_rise_office",
            "病院 (5F, 免震)": "hospital",
            "学校 (4F)": "school",
            "倉庫 (2F)": "warehouse",
            "ピロティ建物 (5F)": "piloti",
            "超高層タワー (40F)": "tower",
        }
        
        ttk.Separator(tab2, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(tab2, text="Floors:").pack(anchor=tk.W)
        self.floors_var = tk.IntVar(value=3)
        ttk.Entry(tab2, textvariable=self.floors_var).pack(fill=tk.X)
        
        # Flag to prevent recursive updates
        self._syncing = False
        
        # Add trace to sync floors_var -> Layout Editor
        self.floors_var.trace_add('write', self._on_floors_var_changed)
        
        self.soft_story_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Soft 1st Story (Piloti)", variable=self.soft_story_var).pack(anchor=tk.W, pady=5)
        self.soft_story_var.trace_add('write', self._on_building_option_changed)
        
        self.iso_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Base Isolation", variable=self.iso_var).pack(anchor=tk.W, pady=5)
        self.iso_var.trace_add('write', self._on_building_option_changed)
        
        self.damper_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Oil Dampers", variable=self.damper_var).pack(anchor=tk.W, pady=5)
        self.damper_var.trace_add('write', self._on_building_option_changed)
        
        ttk.Separator(tab2, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Performance options
        ttk.Label(tab2, text="パフォーマンス:").pack(anchor=tk.W)
        self.fast_mode_var = tk.BooleanVar(value=True)  # Default ON for faster simulation
        ttk.Checkbutton(tab2, text="高速モード (4x速度UP)", variable=self.fast_mode_var).pack(anchor=tk.W, pady=5)
        
        # Run Button
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=20, fill=tk.X)
        
        self.run_btn = ttk.Button(btn_frame, text="Run Simulation", command=self.start_simulation)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_simulation, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(2, 0))
        
        self.sim_cancelled = False
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=150).pack(pady=10)
        
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=150, mode='determinate')
        self.progress.pack(pady=5)
        
        # --- Visualization Panel (Right) ---
        self.vis_frame = ttk.Frame(root, padding="10")
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Visualization Tabs
        self.vis_tabs = ttk.Notebook(self.vis_frame)
        self.vis_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Animation
        self.anim_tab = ttk.Frame(self.vis_tabs)
        self.vis_tabs.add(self.anim_tab, text='Animation')
        
        self.canvas_label = ttk.Label(self.anim_tab, text="Visualization will appear here")
        self.canvas_label.pack(expand=True)
        
        # Tab 2: Graphs
        self.graph_tab = ttk.Frame(self.vis_tabs)
        self.vis_tabs.add(self.graph_tab, text='Graphs')
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.graph_tab)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3: Layout Editor
        self.layout_tab = ttk.Frame(self.vis_tabs)
        self.vis_tabs.add(self.layout_tab, text='Layout Editor')
        
        # Data Model
        self.layout = BuildingLayout()
        self.layout.initialize_default()
        
        # Layout Editor Widget
        self.layout_editor = LayoutEditorPanel(self.layout_tab, self.layout, on_change_callback=self.on_layout_changed)
        self.layout_editor.pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: 3D Viewer (New)
        self.viewer_tab = ttk.Frame(self.vis_tabs)
        self.vis_tabs.add(self.viewer_tab, text='3D View')
        
        viewer_container = ttk.Frame(self.viewer_tab)
        viewer_container.pack(fill=tk.BOTH, expand=True)
        
        self.viewer_3d = Building3DViewer(viewer_container)
        self.viewer_3d.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Side panel with legend
        viewer_side = ttk.Frame(viewer_container, width=150)
        viewer_side.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        viewer_side.pack_propagate(False)
        
        legend = DamageColorLegend(viewer_side)
        legend.pack(pady=10)
        
        # Tab 5: Property Inspector (New)
        self.property_tab = ttk.Frame(self.vis_tabs)
        self.vis_tabs.add(self.property_tab, text='Properties')
        
        self.property_inspector = PropertyInspectorPanel(
            self.property_tab, 
            on_property_change=self.on_property_changed
        )
        self.property_inspector.pack(fill=tk.BOTH, expand=True)
        
        # Button to open detailed Properties Editor
        props_btn_frame = ttk.Frame(self.property_tab)
        props_btn_frame.pack(fill=tk.X, pady=10, padx=10)
        ttk.Button(props_btn_frame, text="詳細エディタを開く / Open Properties Editor",
                   command=self.open_properties_editor).pack(fill=tk.X)
        
        # Tab 4: Export
        self.export_tab = ttk.Frame(self.vis_tabs)
        self.vis_tabs.add(self.export_tab, text='Export')
        
        ttk.Label(self.export_tab, text="Export Results", font=("Helvetica", 14)).pack(pady=20)
        
        export_frame = ttk.Frame(self.export_tab)
        export_frame.pack(pady=10, padx=20, fill=tk.X)
        
        ttk.Label(export_frame, text="Results Export:").pack(anchor=tk.W)
        ttk.Button(export_frame, text="Export Time History (CSV)", 
                   command=self.export_csv).pack(fill=tk.X, pady=5)
        ttk.Button(export_frame, text="Export Results (JSON)", 
                   command=self.export_json).pack(fill=tk.X, pady=5)
        
        ttk.Separator(export_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(export_frame, text="3D Model Export:").pack(anchor=tk.W)
        ttk.Button(export_frame, text="Export Model (OBJ)", 
                   command=self.export_obj).pack(fill=tk.X, pady=5)
        
        ttk.Separator(export_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Export status
        self.export_status = tk.StringVar(value="Run simulation first to enable export")
        ttk.Label(export_frame, textvariable=self.export_status, 
                  foreground='gray').pack(pady=10)

    def on_layout_changed(self):
        """Callback from Layout Editor when floors/grid changes."""
        if self._syncing:
            return
        
        self._syncing = True
        try:
            # Update floor count input in sidebar
            n_stories = len(self.layout.grid.story_heights)
            self.floors_var.set(n_stories)
            
            # Sync building options from layout metadata if available
            if hasattr(self.layout, 'metadata') and self.layout.metadata:
                if 'soft_story' in self.layout.metadata:
                    self.soft_story_var.set(self.layout.metadata['soft_story'])
                if 'base_isolation' in self.layout.metadata:
                    self.iso_var.set(self.layout.metadata['base_isolation'])
                if 'dampers' in self.layout.metadata:
                    self.damper_var.set(self.layout.metadata['dampers'])
            
            # Reset template selector to indicate custom modification
            if hasattr(self, 'template_var'):
                self.template_var.set("(カスタム)")
                
            logger.info(f"Layout -> Sidebar synced: {n_stories} floors")
        finally:
            self._syncing = False
    
    def _on_floors_var_changed(self, *args):
        """Callback when floors_var is changed from sidebar - sync to Layout Editor."""
        if self._syncing:
            return
            
        self._syncing = True
        try:
            new_floors = self.floors_var.get()
            current_floors = len(self.layout.grid.story_heights)
            
            if new_floors == current_floors:
                return
                
            if new_floors < 1:
                self.floors_var.set(1)
                new_floors = 1
            
            # Adjust story_heights
            if new_floors > current_floors:
                # Add floors
                for i in range(new_floors - current_floors):
                    self.layout.grid.story_heights.append(3.5)  # Default 3.5m
                    new_story_idx = len(self.layout.grid.story_heights)
                    self.layout.get_floor(new_story_idx)  # Create floor layout
            else:
                # Remove floors from top
                while len(self.layout.grid.story_heights) > new_floors:
                    removed_idx = len(self.layout.grid.story_heights)
                    self.layout.grid.story_heights.pop()
                    if removed_idx in self.layout.floors:
                        del self.layout.floors[removed_idx]
            
            # Reset template to custom
            if hasattr(self, 'template_var'):
                self.template_var.set("(カスタム)")
            
            # Refresh Layout Editor UI
            if hasattr(self, 'layout_editor'):
                self.layout_editor.refresh_ui()
                
            logger.info(f"Sidebar -> Layout synced: {new_floors} floors")
        except (ValueError, tk.TclError):
            pass  # Ignore invalid input while typing
        finally:
            self._syncing = False
    
    def _on_building_option_changed(self, *args):
        """Callback when soft_story, iso, or damper options change - sync to Layout Editor."""
        if self._syncing:
            return
        
        self._syncing = True
        try:
            # Store building options in layout metadata
            if not hasattr(self.layout, 'metadata'):
                self.layout.metadata = {}
            
            self.layout.metadata['soft_story'] = self.soft_story_var.get()
            self.layout.metadata['base_isolation'] = self.iso_var.get()
            self.layout.metadata['dampers'] = self.damper_var.get()
            
            # Reset template selector
            if hasattr(self, 'template_var'):
                self.template_var.set("(カスタム)")
            
            logger.info(f"Building options synced: soft_story={self.soft_story_var.get()}, iso={self.iso_var.get()}, dampers={self.damper_var.get()}")
        finally:
            self._syncing = False
        
    def on_property_changed(self, property_name: str, value):
        """Callback when property inspector changes a property."""
        logger.info(f"Property changed: {property_name} = {value}")
        # Update layout model with new property
        if property_name == 'section':
            # Update section properties in layout
            pass
        elif property_name == 'material':
            # Update material in layout
            pass
    
    def open_properties_editor(self):
        """Open the detailed Properties Editor dialog."""
        def on_save(layout, analysis_settings):
            logger.info("Properties saved from editor")
            # Store analysis settings for use in simulation
            self.analysis_settings = analysis_settings
            # Refresh the layout editor if needed
            if hasattr(self, 'layout_editor'):
                self.layout_editor.refresh_ui()
        
        open_properties_editor(self.root, self.layout, on_save_callback=on_save)
        
    def on_model_change(self, event=None):
        val = self.model_var.get()
        if val == "Custom Layout":
            self.vis_tabs.select(self.layout_tab)
        else:
            self.vis_tabs.select(self.anim_tab)

    def on_template_change(self, event=None):
        """Update settings when a template is selected."""
        template_name = self.template_var.get()
        if template_name == "(カスタム)":
            return
            
        template_key = self._template_keys.get(template_name)
        if template_key:
            from src.building_templates import get_template
            from src.layout_model import GridSystem, BuildingLayout
            template = get_template(template_key)
            if template:
                # Update floor/feature settings
                self.floors_var.set(template.n_stories)
                self.soft_story_var.set(template.soft_story)
                self.iso_var.set(template.base_isolation)
                self.damper_var.set(template.dampers)
                
                # Synchronize with Layout Editor
                # Create new grid based on template
                x_spacings = [template.bay_width_x] * template.n_bays_x
                y_spacings = [template.bay_width_y] * template.n_bays_y
                story_heights = [template.story_height] * template.n_stories
                
                new_grid = GridSystem(
                    x_spacings=x_spacings,
                    y_spacings=y_spacings,
                    story_heights=story_heights
                )
                
                # Update the shared layout object
                self.layout.grid = new_grid
                self.layout.floors = {}  # Clear floors
                self.layout.initialize_default()  # Re-populate with default elements
                
                # Refresh the Layout Editor UI
                if hasattr(self, 'layout_editor'):
                    self.layout_editor.refresh_ui()
                
                logger.info(f"Template synced to Layout: {template.name} ({template.n_bays_x}x{template.n_bays_y}x{template.n_stories})")

    def toggle_input_ui(self):
        method = self.input_method.get()
        if method == "Generated":
            self.file_frame.pack_forget()
            self.gen_frame.pack(fill=tk.X, pady=5)
        else:
            self.gen_frame.pack_forget()
            self.file_frame.pack(fill=tk.X, pady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Earthquake Record",
            filetypes=(("Text/CSV Files", "*.txt *.csv"), ("All Files", "*.*"))
        )
        if filename:
            self.filename_var.set(filename)

    def start_simulation(self):
        # Validate input
        try:
            acc = self.acc_var.get()
            dur = self.dur_var.get()
            model = self.model_var.get()
            
            # File check
            earthquake_file = None
            if self.input_method.get() == "File":
                earthquake_file = self.filename_var.get()
                if not os.path.exists(earthquake_file):
                    messagebox.showerror("Error", "File does not exist")
                    return
            
            # Advanced params
            floors = self.floors_var.get()
            soft_story = self.soft_story_var.get()
            isolation = self.iso_var.get()
            dampers = self.damper_var.get()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid input parameters")
            return
            
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.sim_cancelled = False
        self.status_var.set(f"Running {model}...")
        self.progress['value'] = 0
        
        # Get template key if selected
        template_name = self.template_var.get()
        template_key = None
        if template_name != "(カスタム)":
            template_key = self._template_keys.get(template_name)
        
        params = {
            'model': model,
            'acc': acc,
            'dur': dur,
            'period': self.period_var.get(),
            'template': template_key,
            'floors': floors,
            'soft_story': soft_story,
            'isolation': isolation,
            'dampers': dampers,
            'file': earthquake_file,
            'fast_mode': self.fast_mode_var.get()
        }
        
        # Run in thread
        self.sim_thread = threading.Thread(target=self.run_thread, args=(params,))
        self.sim_thread.start()
        
    def cancel_simulation(self):
        """Cancel the running simulation."""
        self.sim_cancelled = True
        self.status_var.set("Cancelling...")
        self.cancel_btn.config(state=tk.DISABLED)
        logger.info("Simulation cancelled by user")
        
    def run_thread(self, params):
        def progress_callback(step, total):
            prog = (step / total) * 100
            self.root.after(0, lambda: self.progress.config(value=prog))
            self.root.after(0, lambda: self.status_var.set(f"Step {step}/{total}"))
            
        try:
            if params['model'] == "2D Frame":
                gif_path, t, res = sim2d.run_simulation(
                    duration=params['dur'], 
                    max_acc=params['acc'], 
                    callback=progress_callback,
                    earthquake_file=params['file']
                )
                res_type = 'drift'
                result_dict = None
            else:
                # 3D Frame or Custom Layout - always use Layout Editor's layout
                # This ensures simulation results reflect what's shown in Layout Editor
                layout_obj = self.layout
                    
                result_dict = sim3d.run_3d_simulation(
                    duration=params['dur'], 
                    max_acc=params['acc'], 
                    callback=progress_callback,
                    builder_params=params, 
                    earthquake_file=params['file'],
                    layout=layout_obj,
                    fast_mode=params.get('fast_mode', False)
                )
                
                # Handle new dictionary return format
                gif_path = result_dict['gif_path']
                t = result_dict['time']
                res = result_dict['displacement_history']
                res_type = 'disp_3d'
                
            self.root.after(0, lambda: self.on_simulation_complete(gif_path, t, res, res_type, result_dict))
        except Exception as e:
            logger.exception(f"Simulation error: {e}")
            self.root.after(0, lambda: self.on_simulation_error(str(e)))
            
    def on_simulation_complete(self, gif_path, t, result_data, res_type, result_dict=None):
        self.status_var.set("Simulation Complete. Loading 3D View...")
        self.progress['value'] = 100
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        
        # Store results for export
        self.last_results['time'] = t
        self.last_results['displacement'] = np.array(result_data) if res_type == 'disp_3d' else None
        
        # Store nodes and elements from 3D simulation for export and 3D viewer
        if result_dict:
            self.last_results['nodes'] = result_dict.get('nodes')
            self.last_results['elements'] = result_dict.get('elements')
            self.last_results['damage_history'] = result_dict.get('damage_history')
            self.last_results['duration'] = result_dict.get('duration')
            self.last_results['dt'] = result_dict.get('dt')
        
        self.export_status.set("Results ready for export")
        logger.info(f"Simulation complete. Results stored for export.")
        
        # 1. Update Graphs
        self.update_graphs(t, result_data, res_type)
        
        # 2. Update 3D Viewer with simulation results
        if res_type == 'disp_3d' and result_dict:
            try:
                nodes = result_dict.get('nodes')
                elements = result_dict.get('elements')
                damage_history = result_dict.get('damage_history')
                duration = result_dict.get('duration', 5.0)
                
                if nodes and elements:
                    self.viewer_3d.set_model(nodes, elements)
                    if isinstance(result_data, list) and len(result_data) > 0:
                        self.viewer_3d.set_results(t, result_data, damage_history)
                        # Set real-time animation interval based on duration
                        self.viewer_3d.set_realtime_duration(duration)
                    
                    # Auto-switch to 3D View tab for better visual experience
                    self.vis_tabs.select(self.viewer_tab)
                    
                    # Start animation automatically
                    self.viewer_3d.start_animation()
                    
                    logger.info(f"3D viewer updated with simulation results (real-time: {duration}s)")
            except Exception as e:
                logger.warning(f"Failed to update 3D viewer: {e}")
                # Fall back to GIF display
                self.vis_tabs.select(self.anim_tab)
        
        # 3. Display GIF (for Animation tab)
        try:
            self.load_gif(gif_path)
        except Exception as e:
            logger.warning(f"Failed to load animation GIF: {e}")

    def update_graphs(self, t, data, res_type):
        self.figure.clear()
        
        # Layout: 1x2 or 2x1
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        if res_type == 'drift':
            # 2D case, data is drift history list
            ax1.plot(t, data, label='1F Drift')
            ax1.set_title("Inter-story Drift Ratio (1F)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Drift (rad)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Use data history? We only got drift back.
            # Ideally we want base shear / displacement hysteresis.
            # Future improvement: Return full history dict.
            ax2.text(0.5, 0.5, "Additional data not returned by 2D sim", ha='center')
            
        elif res_type == 'disp_3d':
            # 3D case, data is history_u (List of arrays)
            # Extract Top Node Displacement logic check
            # Nodes are mixed. Need to find top node?
            # BuildingBuilder constructed 4 nodes per floor?
            # 3D sim has N nodes.
            # Assuming standard build from 0 to Z max.
            
            # Convert list of arrays to matrix (Steps x DOFs)
            hist = np.array(data) # shape (Steps, ndof)
            
            # Find max Z dof index?
            # Simplification: Plot dof 0 (Node 1 X) vs Time?
            # Node 1 is Base. 
            # We want Top Node. Top node ID is last?
            # Let's plot Max Abs Displacement over all DOFs per step.
            
            max_disp = np.max(np.abs(hist), axis=1)
            ax1.plot(t, max_disp, color='r')
            ax1.set_title("Max Global Displacement Envelope")
            ax1.set_ylabel("Disp (m)")
            ax1.grid(True)
            
            # Hysteresis? Need Force.
            # We have damage indices? 
            # Let's plot "Max inter-story drift" proxy?
            # Or just X-displacement of top node.
            # Let's try to get Top Node X disp.
            # Top node index is roughly last.
            top_x = hist[:, -6] # Approx?
            
            ax2.plot(t, top_x)
            ax2.set_title("Top Node X-Displacement")
            ax2.set_xlabel("Time (s)")
            ax2.grid(True)
            
        self.canvas_plot.draw()

    def on_simulation_error(self, error_msg):
        self.status_var.set("Error occurred")
        self.run_btn.config(state=tk.NORMAL)
        messagebox.showerror("Simulation Error", error_msg)

    def load_gif(self, path):
        # Stop previous animation if any
        if hasattr(self, 'anim_job'):
            self.root.after_cancel(self.anim_job)
            
        self.gif_image = Image.open(path)
        self.gif_frames = []
        try:
            while True:
                self.gif_frames.append(self.gif_image.copy())
                self.gif_image.seek(len(self.gif_frames))
        except EOFError:
            pass
            
        self.current_frame = 0
        self.animate_gif()
        
    def animate_gif(self):
        if not self.gif_frames: return
        
        # Resize frame to fit canvas
        frame = self.gif_frames[self.current_frame]
        # frame.thumbnail((600, 500)) 
        
        photo = ImageTk.PhotoImage(frame)
        self.canvas_label.config(image=photo, text="")
        self.canvas_label.image = photo # Keep reference
        
        self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
        
        # Delay based on duration
        delay = 100 # ms
        if 'duration' in self.gif_image.info:
            delay = self.gif_image.info['duration']
            
        self.anim_job = self.root.after(delay, self.animate_gif)
    
    def export_csv(self):
        """Export time history results to CSV."""
        if self.last_results['time'] is None:
            messagebox.showwarning("Export", "No results to export. Run simulation first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Export Time History"
        )
        
        if filepath:
            try:
                time_array = self.last_results['time']
                hist = self.last_results['displacement']
                
                # Create data dict with max displacement per step
                if hist is not None and len(hist) > 0:
                    data_dict = {
                        'Max_Displacement': np.max(np.abs(hist), axis=1) if len(hist[0]) > 0 else np.zeros(len(time_array))
                    }
                    
                    path = ResultsExporter.export_to_csv(
                        filepath, time_array, data_dict,
                        metadata={'simulation': 'Earthquake Building Sim'}
                    )
                    self.export_status.set(f"Exported to: {os.path.basename(path)}")
                    messagebox.showinfo("Export", f"Results exported to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
                logger.error(f"Export CSV failed: {e}")
    
    def export_json(self):
        """Export results to JSON format."""
        if self.last_results['time'] is None:
            messagebox.showwarning("Export", "No results to export. Run simulation first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Export Results"
        )
        
        if filepath:
            try:
                results = {
                    'time': self.last_results['time'],
                    'displacement_history': self.last_results['displacement']
                }
                
                path = ResultsExporter.export_to_json(
                    filepath, results,
                    metadata={'simulation': 'Earthquake Building Sim'}
                )
                self.export_status.set(f"Exported to: {os.path.basename(path)}")
                messagebox.showinfo("Export", f"Results exported to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
                logger.error(f"Export JSON failed: {e}")
    
    def export_obj(self):
        """Export 3D model to OBJ format."""
        if self.last_results['nodes'] is None:
            messagebox.showwarning("Export", "No model to export. Run simulation first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")],
            title="Export 3D Model"
        )
        
        if filepath:
            try:
                path = OBJExporter.export_frame_model(
                    filepath,
                    self.last_results['nodes'],
                    self.last_results['elements']
                )
                self.export_status.set(f"Exported to: {os.path.basename(path)}")
                messagebox.showinfo("Export", f"Model exported to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
                logger.error(f"Export OBJ failed: {e}")

def main():
    root = tk.Tk()
    app = EarthquakeSimGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
