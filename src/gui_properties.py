"""
Properties Editor Panel - GUI for editing materials, sections, joints, and analysis settings.

プロパティエディタ - 材料、断面、接合部、解析設定を編集するGUI
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Optional, Callable
from dataclasses import fields

from .material_library import (
    MATERIAL_LIB, MaterialProperties, MaterialType,
    JointProperties, JOINT_LIBRARY,
    AnalysisSettings, DEFAULT_ANALYSIS_SETTINGS
)
from .layout_model import SectionProperties


class PropertiesEditorDialog(tk.Toplevel):
    """
    Dialog for editing all properties: materials, sections, joints, analysis settings.
    """
    
    def __init__(self, parent, layout=None, on_save_callback: Callable = None):
        super().__init__(parent)
        self.title("プロパティエディタ / Properties Editor")
        self.geometry("900x650")
        self.transient(parent)
        
        self.layout = layout
        self.on_save_callback = on_save_callback
        self.analysis_settings = AnalysisSettings()
        
        self._create_widgets()
        self._load_data()
        
    def _create_widgets(self):
        """Create the tabbed interface."""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Materials (材料)
        self.materials_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.materials_frame, text="材料 / Materials")
        self._create_materials_tab()
        
        # Tab 2: Sections (断面)
        self.sections_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sections_frame, text="断面 / Sections")
        self._create_sections_tab()
        
        # Tab 3: Joints (接合部)
        self.joints_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.joints_frame, text="接合部 / Joints")
        self._create_joints_tab()
        
        # Tab 4: Analysis Settings (解析設定)
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="解析設定 / Analysis")
        self._create_analysis_tab()
        
        # Bottom buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="保存して閉じる / Save & Close", 
                   command=self._save_and_close).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="キャンセル / Cancel",
                   command=self.destroy).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="適用 / Apply",
                   command=self._apply_changes).pack(side='right', padx=5)
    
    def _create_materials_tab(self):
        """Create materials tab content."""
        # Left: Material list
        left_frame = ttk.LabelFrame(self.materials_frame, text="材料リスト")
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        # Filter by type
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(filter_frame, text="フィルタ:").pack(side='left')
        self.mat_filter = ttk.Combobox(filter_frame, values=['全て', '鉄骨', 'コンクリート', '鉄筋'], width=12)
        self.mat_filter.set('全て')
        self.mat_filter.pack(side='left', padx=5)
        self.mat_filter.bind('<<ComboboxSelected>>', self._filter_materials)
        
        # Material listbox
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.mat_listbox = tk.Listbox(list_frame, width=25, height=20)
        self.mat_listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.mat_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.mat_listbox.config(yscrollcommand=scrollbar.set)
        self.mat_listbox.bind('<<ListboxSelect>>', self._on_material_select)
        
        # Right: Material properties
        right_frame = ttk.LabelFrame(self.materials_frame, text="材料特性")
        right_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Property grid
        self.mat_props = {}
        props = [
            ('name', '名前', 'entry'),
            ('E', '弾性係数 E (Pa)', 'entry'),
            ('G', 'せん断弾性係数 G (Pa)', 'entry'),
            ('nu', 'ポアソン比 ν', 'entry'),
            ('rho', '密度 ρ (kg/m³)', 'entry'),
            ('Fy', '降伏強度 Fy (Pa)', 'entry'),
            ('Fu', '極限強度 Fu (Pa)', 'entry'),
            ('Fc', '圧縮強度 Fc (Pa)', 'entry'),
            ('description', '説明', 'entry'),
        ]
        
        for i, (key, label, widget_type) in enumerate(props):
            ttk.Label(right_frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=3)
            if widget_type == 'entry':
                var = tk.StringVar()
                entry = ttk.Entry(right_frame, textvariable=var, width=30)
                entry.grid(row=i, column=1, sticky='w', padx=5, pady=3)
                self.mat_props[key] = var
    
    def _create_sections_tab(self):
        """Create sections tab content."""
        # Left: Section list
        left_frame = ttk.LabelFrame(self.sections_frame, text="断面リスト")
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        # Section listbox
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.sec_listbox = tk.Listbox(list_frame, width=20, height=18)
        self.sec_listbox.pack(side='left', fill='both', expand=True)
        self.sec_listbox.bind('<<ListboxSelect>>', self._on_section_select)
        
        # Add/Remove buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_frame, text="追加", command=self._add_section).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="削除", command=self._remove_section).pack(side='left', padx=2)
        
        # Right: Section properties
        right_frame = ttk.LabelFrame(self.sections_frame, text="断面特性")
        right_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        self.sec_props = {}
        props = [
            ('name', '名前', 'entry'),
            ('width', '幅 B (m)', 'entry'),
            ('height', '高さ D (m)', 'entry'),
            ('area', '断面積 A (m²)', 'entry'),
            ('I_y', '断面二次モーメント Iy (m⁴)', 'entry'),
            ('I_z', '断面二次モーメント Iz (m⁴)', 'entry'),
            ('J', 'ねじり定数 J (m⁴)', 'entry'),
            ('yield_moment', '降伏モーメント My (N·m)', 'entry'),
            ('E', '弾性係数 E (Pa)', 'entry'),
            ('color', '表示色', 'entry'),
        ]
        
        for i, (key, label, widget_type) in enumerate(props):
            ttk.Label(right_frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=3)
            var = tk.StringVar()
            entry = ttk.Entry(right_frame, textvariable=var, width=25)
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=3)
            self.sec_props[key] = var
        
        # Auto-calculate button
        ttk.Button(right_frame, text="矩形断面から自動計算", 
                   command=self._auto_calc_section).grid(row=len(props), column=0, columnspan=2, pady=10)
    
    def _create_joints_tab(self):
        """Create joints tab content."""
        # Left: Joint list
        left_frame = ttk.LabelFrame(self.joints_frame, text="接合部リスト")
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        self.joint_listbox = tk.Listbox(left_frame, width=20, height=18)
        self.joint_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        self.joint_listbox.bind('<<ListboxSelect>>', self._on_joint_select)
        
        # Right: Joint properties
        right_frame = ttk.LabelFrame(self.joints_frame, text="接合部特性")
        right_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        self.joint_props = {}
        
        row = 0
        ttk.Label(right_frame, text="名前:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
        self.joint_props['name'] = tk.StringVar()
        ttk.Entry(right_frame, textvariable=self.joint_props['name'], width=25).grid(row=row, column=1, sticky='w', padx=5, pady=3)
        
        row += 1
        ttk.Label(right_frame, text="タイプ:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
        self.joint_props['joint_type'] = tk.StringVar()
        type_combo = ttk.Combobox(right_frame, textvariable=self.joint_props['joint_type'], 
                                  values=['rigid', 'pinned', 'semi_rigid'], width=22)
        type_combo.grid(row=row, column=1, sticky='w', padx=5, pady=3)
        
        row += 1
        ttk.Label(right_frame, text="回転剛性 (N·m/rad):").grid(row=row, column=0, sticky='e', padx=5, pady=3)
        self.joint_props['rotational_stiffness'] = tk.StringVar()
        ttk.Entry(right_frame, textvariable=self.joint_props['rotational_stiffness'], width=25).grid(row=row, column=1, sticky='w', padx=5, pady=3)
        
        row += 1
        ttk.Label(right_frame, text="ヒステリシス:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
        self.joint_props['hysteresis_type'] = tk.StringVar()
        hyst_combo = ttk.Combobox(right_frame, textvariable=self.joint_props['hysteresis_type'],
                                   values=['takeda', 'bilinear', 'elastic'], width=22)
        hyst_combo.grid(row=row, column=1, sticky='w', padx=5, pady=3)
        
        row += 1
        ttk.Label(right_frame, text="剛性劣化指数 α:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
        self.joint_props['degradation_alpha'] = tk.StringVar()
        ttk.Entry(right_frame, textvariable=self.joint_props['degradation_alpha'], width=25).grid(row=row, column=1, sticky='w', padx=5, pady=3)
        
        row += 1
        ttk.Label(right_frame, text="降伏後剛性比:").grid(row=row, column=0, sticky='e', padx=5, pady=3)
        self.joint_props['post_yield_ratio'] = tk.StringVar()
        ttk.Entry(right_frame, textvariable=self.joint_props['post_yield_ratio'], width=25).grid(row=row, column=1, sticky='w', padx=5, pady=3)
    
    def _create_analysis_tab(self):
        """Create analysis settings tab content."""
        # Frame for settings groups
        main_frame = ttk.Frame(self.analysis_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.analysis_vars = {}
        
        # Time Integration Settings
        time_frame = ttk.LabelFrame(main_frame, text="時間積分 / Time Integration")
        time_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        ttk.Label(time_frame, text="時間刻み dt (s):").grid(row=0, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['time_step'] = tk.StringVar(value='0.005')
        ttk.Entry(time_frame, textvariable=self.analysis_vars['time_step'], width=15).grid(row=0, column=1, sticky='w', padx=5, pady=3)
        
        ttk.Label(time_frame, text="Newmark β:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['beta'] = tk.StringVar(value='0.25')
        ttk.Entry(time_frame, textvariable=self.analysis_vars['beta'], width=15).grid(row=1, column=1, sticky='w', padx=5, pady=3)
        
        ttk.Label(time_frame, text="Newmark γ:").grid(row=2, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['gamma'] = tk.StringVar(value='0.5')
        ttk.Entry(time_frame, textvariable=self.analysis_vars['gamma'], width=15).grid(row=2, column=1, sticky='w', padx=5, pady=3)
        
        # Newton-Raphson Settings
        nr_frame = ttk.LabelFrame(main_frame, text="Newton-Raphson")
        nr_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        ttk.Label(nr_frame, text="最大反復回数:").grid(row=0, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['max_iterations'] = tk.StringVar(value='10')
        ttk.Entry(nr_frame, textvariable=self.analysis_vars['max_iterations'], width=15).grid(row=0, column=1, sticky='w', padx=5, pady=3)
        
        ttk.Label(nr_frame, text="収束判定誤差:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['tolerance'] = tk.StringVar(value='0.001')
        ttk.Entry(nr_frame, textvariable=self.analysis_vars['tolerance'], width=15).grid(row=1, column=1, sticky='w', padx=5, pady=3)
        
        # Damping Settings
        damp_frame = ttk.LabelFrame(main_frame, text="減衰 / Damping")
        damp_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        ttk.Label(damp_frame, text="減衰タイプ:").grid(row=0, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['damping_type'] = tk.StringVar(value='rayleigh')
        damp_combo = ttk.Combobox(damp_frame, textvariable=self.analysis_vars['damping_type'],
                                   values=['rayleigh', 'modal', 'caughey'], width=12)
        damp_combo.grid(row=0, column=1, sticky='w', padx=5, pady=3)
        
        ttk.Label(damp_frame, text="減衰定数 ζ:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['damping_ratio'] = tk.StringVar(value='0.05')
        ttk.Entry(damp_frame, textvariable=self.analysis_vars['damping_ratio'], width=15).grid(row=1, column=1, sticky='w', padx=5, pady=3)
        
        # Nonlinear Settings
        nonlin_frame = ttk.LabelFrame(main_frame, text="非線形 / Nonlinear")
        nonlin_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        
        self.analysis_vars['p_delta_enabled'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(nonlin_frame, text="P-Delta効果", 
                        variable=self.analysis_vars['p_delta_enabled']).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=3)
        
        self.analysis_vars['line_search'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(nonlin_frame, text="ライン探索", 
                        variable=self.analysis_vars['line_search']).grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=3)
        
        ttk.Label(nonlin_frame, text="最大変位増分 (m):").grid(row=2, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['max_displacement_step'] = tk.StringVar(value='0.5')
        ttk.Entry(nonlin_frame, textvariable=self.analysis_vars['max_displacement_step'], width=12).grid(row=2, column=1, sticky='w', padx=5, pady=3)
        
        # Mass Matrix Settings
        mass_frame = ttk.LabelFrame(main_frame, text="質量マトリックス / Mass")
        mass_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        
        ttk.Label(mass_frame, text="質量タイプ:").grid(row=0, column=0, sticky='e', padx=5, pady=3)
        self.analysis_vars['mass_type'] = tk.StringVar(value='lumped')
        mass_combo = ttk.Combobox(mass_frame, textvariable=self.analysis_vars['mass_type'],
                                   values=['lumped', 'consistent'], width=12)
        mass_combo.grid(row=0, column=1, sticky='w', padx=5, pady=3)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def _load_data(self):
        """Load data into the UI."""
        # Load materials
        self._filter_materials(None)
        
        # Load sections from layout
        if self.layout:
            for name in self.layout.sections.keys():
                self.sec_listbox.insert(tk.END, name)
        
        # Load joints
        for name, joint in JOINT_LIBRARY.items():
            self.joint_listbox.insert(tk.END, f"{name} ({joint.display_name})")
    
    def _filter_materials(self, event):
        """Filter materials by type."""
        self.mat_listbox.delete(0, tk.END)
        filter_val = self.mat_filter.get()
        
        type_map = {'鉄骨': MaterialType.STEEL, 'コンクリート': MaterialType.CONCRETE, '鉄筋': MaterialType.REBAR}
        
        for name, mat in MATERIAL_LIB.materials.items():
            if filter_val == '全て' or mat.type == type_map.get(filter_val):
                self.mat_listbox.insert(tk.END, name)
    
    def _on_material_select(self, event):
        """Handle material selection."""
        sel = self.mat_listbox.curselection()
        if not sel:
            return
        name = self.mat_listbox.get(sel[0])
        mat = MATERIAL_LIB.get_material(name)
        if mat:
            self.mat_props['name'].set(mat.name)
            self.mat_props['E'].set(f"{mat.E:.2e}")
            self.mat_props['G'].set(f"{mat.G:.2e}")
            self.mat_props['nu'].set(str(mat.nu))
            self.mat_props['rho'].set(str(mat.rho))
            self.mat_props['Fy'].set(f"{mat.Fy:.2e}")
            self.mat_props['Fu'].set(f"{mat.Fu:.2e}")
            self.mat_props['Fc'].set(f"{mat.Fc:.2e}")
            self.mat_props['description'].set(mat.description)
    
    def _on_section_select(self, event):
        """Handle section selection."""
        sel = self.sec_listbox.curselection()
        if not sel or not self.layout:
            return
        name = self.sec_listbox.get(sel[0])
        sec = self.layout.sections.get(name)
        if sec:
            self.sec_props['name'].set(sec.name)
            self.sec_props['area'].set(str(sec.area))
            self.sec_props['I_y'].set(str(sec.I_y))
            self.sec_props['I_z'].set(str(sec.I_z))
            self.sec_props['J'].set(str(sec.J))
            self.sec_props['yield_moment'].set(str(sec.yield_moment))
            self.sec_props['E'].set(f"{sec.E:.2e}")
            self.sec_props['color'].set(sec.color)
    
    def _on_joint_select(self, event):
        """Handle joint selection."""
        sel = self.joint_listbox.curselection()
        if not sel:
            return
        item = self.joint_listbox.get(sel[0])
        name = item.split(' ')[0]
        joint = JOINT_LIBRARY.get(name)
        if joint:
            self.joint_props['name'].set(joint.name)
            self.joint_props['joint_type'].set(joint.joint_type)
            self.joint_props['rotational_stiffness'].set(f"{joint.rotational_stiffness:.2e}")
            self.joint_props['hysteresis_type'].set(joint.hysteresis_type)
            self.joint_props['degradation_alpha'].set(str(joint.degradation_alpha))
            self.joint_props['post_yield_ratio'].set(str(joint.post_yield_ratio))
    
    def _add_section(self):
        """Add a new section."""
        name = f"New_Section_{len(self.layout.sections) + 1}" if self.layout else "New_Section"
        if self.layout:
            self.layout.sections[name] = SectionProperties(name=name)
            self.sec_listbox.insert(tk.END, name)
    
    def _remove_section(self):
        """Remove selected section."""
        sel = self.sec_listbox.curselection()
        if not sel or not self.layout:
            return
        name = self.sec_listbox.get(sel[0])
        if name in self.layout.sections:
            del self.layout.sections[name]
            self.sec_listbox.delete(sel[0])
    
    def _auto_calc_section(self):
        """Auto-calculate section properties from width/height."""
        try:
            b = float(self.sec_props.get('width', tk.StringVar()).get() or 0.5)
            d = float(self.sec_props.get('height', tk.StringVar()).get() or 0.5)
            
            A = b * d
            Iy = b * d**3 / 12  # About Y-axis (weak)
            Iz = d * b**3 / 12  # About Z-axis (strong)
            J = b * d * (b**2 + d**2) / 12  # Approximate for rectangular
            
            self.sec_props['area'].set(f"{A:.6f}")
            self.sec_props['I_y'].set(f"{Iy:.6e}")
            self.sec_props['I_z'].set(f"{Iz:.6e}")
            self.sec_props['J'].set(f"{J:.6e}")
            
            # Estimate yield moment (assuming RC with typical steel ratio)
            E = float(self.sec_props['E'].get() or 2.5e10)
            My = 0.9 * 345e6 * 0.02 * b * d**2  # Approx: 0.9 * fy * rho * b * d²
            self.sec_props['yield_moment'].set(f"{My:.0f}")
            
        except ValueError:
            messagebox.showerror("エラー", "幅と高さに有効な数値を入力してください")
    
    def _apply_changes(self):
        """Apply changes to the layout and analysis settings."""
        if not self.layout:
            return
        
        # Update section from UI
        sel = self.sec_listbox.curselection()
        if sel:
            name = self.sec_listbox.get(sel[0])
            if name in self.layout.sections:
                sec = self.layout.sections[name]
                try:
                    sec.area = float(self.sec_props['area'].get())
                    sec.I_y = float(self.sec_props['I_y'].get())
                    sec.I_z = float(self.sec_props['I_z'].get())
                    sec.J = float(self.sec_props['J'].get())
                    sec.yield_moment = float(self.sec_props['yield_moment'].get())
                    sec.E = float(self.sec_props['E'].get())
                    sec.color = self.sec_props['color'].get()
                except ValueError as e:
                    messagebox.showerror("エラー", f"数値変換エラー: {e}")
        
        # Update analysis settings
        try:
            self.analysis_settings.time_step = float(self.analysis_vars['time_step'].get())
            self.analysis_settings.beta = float(self.analysis_vars['beta'].get())
            self.analysis_settings.gamma = float(self.analysis_vars['gamma'].get())
            self.analysis_settings.max_iterations = int(self.analysis_vars['max_iterations'].get())
            self.analysis_settings.tolerance = float(self.analysis_vars['tolerance'].get())
            self.analysis_settings.damping_type = self.analysis_vars['damping_type'].get()
            self.analysis_settings.damping_ratio = float(self.analysis_vars['damping_ratio'].get())
            self.analysis_settings.p_delta_enabled = self.analysis_vars['p_delta_enabled'].get()
            self.analysis_settings.line_search = self.analysis_vars['line_search'].get()
            self.analysis_settings.max_displacement_step = float(self.analysis_vars['max_displacement_step'].get())
            self.analysis_settings.mass_type = self.analysis_vars['mass_type'].get()
        except ValueError as e:
            messagebox.showerror("エラー", f"解析設定の変換エラー: {e}")
        
        if self.on_save_callback:
            self.on_save_callback(self.layout, self.analysis_settings)
        
        messagebox.showinfo("完了", "設定を適用しました")
    
    def _save_and_close(self):
        """Save and close the dialog."""
        self._apply_changes()
        self.destroy()
    
    def get_analysis_settings(self) -> AnalysisSettings:
        """Return the current analysis settings."""
        return self.analysis_settings


def open_properties_editor(parent, layout=None, on_save_callback=None):
    """
    Convenience function to open the properties editor dialog.
    
    Args:
        parent: Parent tk window
        layout: BuildingLayout object to edit
        on_save_callback: Function to call when saving (layout, settings)
    
    Returns:
        PropertiesEditorDialog instance
    """
    dialog = PropertiesEditorDialog(parent, layout, on_save_callback)
    return dialog
