"""
Advanced Property Inspector GUI.
Provides detailed editing of structural element properties including
sections, materials, connections, and analysis parameters.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .section_database import SECTION_DB, SteelSection
from .material_library import MATERIAL_LIB, MaterialProperties, MaterialType
from .layout_model import SectionProperties


@dataclass
class ConnectionProperties:
    """Properties for beam-column connections."""
    type: str = 'rigid'  # 'rigid', 'pinned', 'semi-rigid'
    weld_type: str = 'full_penetration'  # 'full_penetration', 'fillet', 'partial'
    weld_size: float = 0.0  # mm (for fillet welds)
    bolt_diameter: str = 'M22'  # M16, M20, M22, M24
    bolt_count: int = 0
    rotational_stiffness: float = 1e15  # N·m/rad (high = rigid)


class SectionSelectorDialog(tk.Toplevel):
    """Dialog for selecting steel sections from database."""
    
    def __init__(self, parent, on_select: Callable[[SteelSection], None]):
        super().__init__(parent)
        self.title("断面選択 - Section Selector")
        self.geometry("600x500")
        self.on_select = on_select
        self.selected_section: Optional[SteelSection] = None
        
        self.create_widgets()
        self.transient(parent)
        self.grab_set()
        
    def create_widgets(self):
        # Search Frame
        search_frame = ttk.Frame(self, padding=10)
        search_frame.pack(fill=tk.X)
        
        ttk.Label(search_frame, text="検索:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Type filter
        ttk.Label(search_frame, text="タイプ:").pack(side=tk.LEFT, padx=(20, 5))
        self.type_var = tk.StringVar(value="ALL")
        type_combo = ttk.Combobox(search_frame, textvariable=self.type_var, width=10)
        type_combo['values'] = ['ALL', 'H', 'BOX', 'PIPE']
        type_combo.bind('<<ComboboxSelected>>', self.on_search)
        type_combo.pack(side=tk.LEFT)
        
        # Section List with Treeview
        list_frame = ttk.Frame(self, padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('name', 'type', 'H', 'B', 'weight', 'Ix', 'Iy')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        self.tree.heading('name', text='名称')
        self.tree.heading('type', text='タイプ')
        self.tree.heading('H', text='H (mm)')
        self.tree.heading('B', text='B (mm)')
        self.tree.heading('weight', text='重量 (kg/m)')
        self.tree.heading('Ix', text='Ix (cm⁴)')
        self.tree.heading('Iy', text='Iy (cm⁴)')
        
        self.tree.column('name', width=120)
        self.tree.column('type', width=60)
        self.tree.column('H', width=60)
        self.tree.column('B', width=60)
        self.tree.column('weight', width=80)
        self.tree.column('Ix', width=80)
        self.tree.column('Iy', width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # Detail Panel
        detail_frame = ttk.LabelFrame(self, text="断面詳細", padding=10)
        detail_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.detail_text = tk.Text(detail_frame, height=5, width=60)
        self.detail_text.pack(fill=tk.X)
        
        # Buttons
        btn_frame = ttk.Frame(self, padding=10)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="選択", command=self.confirm_selection).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.RIGHT)
        
        # Load initial data
        self.refresh_list()
        
    def refresh_list(self, search_query: str = "", type_filter: str = "ALL"):
        self.tree.delete(*self.tree.get_children())
        
        for name, section in SECTION_DB.sections.items():
            if type_filter != "ALL" and section.type != type_filter:
                continue
            if search_query and search_query.lower() not in name.lower():
                continue
                
            self.tree.insert('', 'end', values=(
                section.name,
                section.type,
                f"{section.H:.0f}",
                f"{section.B:.0f}",
                f"{section.weight:.1f}",
                f"{section.Ix/1e4:.0f}",  # mm⁴ to cm⁴
                f"{section.Iy/1e4:.0f}"
            ))
    
    def on_search(self, *args):
        query = self.search_var.get()
        type_filter = self.type_var.get()
        self.refresh_list(query, type_filter)
    
    def on_tree_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            name = item['values'][0]
            section = SECTION_DB.get_section(name)
            if section:
                self.selected_section = section
                self.show_detail(section)
    
    def on_double_click(self, event):
        self.confirm_selection()
    
    def show_detail(self, section: SteelSection):
        self.detail_text.delete('1.0', tk.END)
        
        si = section.to_si_units()
        detail = f"""断面: {section.name}
寸法: H={section.H}mm, B={section.B}mm, tw={section.tw}mm, tf={section.tf}mm
断面積 A = {section.A:.0f} mm² ({si['A']*1e6:.2f} m²)
断面2次モーメント Ix = {section.Ix:.2e} mm⁴, Iy = {section.Iy:.2e} mm⁴
断面係数 Zx = {section.Zx:.0f} mm³, Zy = {section.Zy:.0f} mm³
重量 = {section.weight:.1f} kg/m"""
        self.detail_text.insert('1.0', detail)
    
    def confirm_selection(self):
        if self.selected_section and self.on_select:
            self.on_select(self.selected_section)
        self.destroy()


class MaterialSelectorDialog(tk.Toplevel):
    """Dialog for selecting materials from library."""
    
    def __init__(self, parent, material_type: Optional[MaterialType] = None,
                 on_select: Callable[[MaterialProperties], None] = None):
        super().__init__(parent)
        self.title("材料選択 - Material Selector")
        self.geometry("500x400")
        self.on_select = on_select
        self.material_type = material_type
        self.selected_material: Optional[MaterialProperties] = None
        
        self.create_widgets()
        self.transient(parent)
        self.grab_set()
        
    def create_widgets(self):
        # Type filter
        filter_frame = ttk.Frame(self, padding=10)
        filter_frame.pack(fill=tk.X)
        
        ttk.Label(filter_frame, text="カテゴリ:").pack(side=tk.LEFT)
        self.type_var = tk.StringVar(value="ALL")
        type_combo = ttk.Combobox(filter_frame, textvariable=self.type_var, width=15)
        type_combo['values'] = ['ALL', 'steel', 'concrete', 'rebar']
        type_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        type_combo.pack(side=tk.LEFT, padx=5)
        
        if self.material_type:
            self.type_var.set(self.material_type.value)
        
        # Material List
        list_frame = ttk.Frame(self, padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('name', 'type', 'E', 'Fy', 'description')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        self.tree.heading('name', text='名称')
        self.tree.heading('type', text='種類')
        self.tree.heading('E', text='E (GPa)')
        self.tree.heading('Fy', text='Fy (MPa)')
        self.tree.heading('description', text='説明')
        
        self.tree.column('name', width=80)
        self.tree.column('type', width=70)
        self.tree.column('E', width=60)
        self.tree.column('Fy', width=60)
        self.tree.column('description', width=200)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<<TreeviewSelect>>', self.on_select_item)
        self.tree.bind('<Double-1>', self.confirm_selection)
        
        # Buttons
        btn_frame = ttk.Frame(self, padding=10)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="選択", command=self.confirm_selection).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.RIGHT)
        
        self.refresh_list()
        
    def refresh_list(self):
        self.tree.delete(*self.tree.get_children())
        
        type_filter = self.type_var.get()
        
        for name, mat in MATERIAL_LIB.materials.items():
            if type_filter != "ALL" and mat.type.value != type_filter:
                continue
                
            e_gpa = mat.E / 1e9
            fy_mpa = mat.Fy / 1e6 if mat.Fy > 0 else mat.Fc / 1e6
            
            self.tree.insert('', 'end', values=(
                mat.name,
                mat.type.value,
                f"{e_gpa:.0f}",
                f"{fy_mpa:.0f}",
                mat.description[:30] + "..." if len(mat.description) > 30 else mat.description
            ))
    
    def on_filter_change(self, event):
        self.refresh_list()
    
    def on_select_item(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            name = item['values'][0]
            self.selected_material = MATERIAL_LIB.get_material(name)
    
    def confirm_selection(self, event=None):
        if self.selected_material and self.on_select:
            self.on_select(self.selected_material)
        self.destroy()


class PropertyInspectorPanel(ttk.Frame):
    """
    Property inspector panel for editing structural element properties.
    Shows and allows editing of section, material, and connection properties.
    """
    
    def __init__(self, parent, on_property_change: Callable[[str, Any], None] = None):
        super().__init__(parent)
        self.on_property_change = on_property_change
        
        self.current_element = None
        self.current_section: Optional[SteelSection] = None
        self.current_material: Optional[MaterialProperties] = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(title_frame, text="プロパティ", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.element_label = ttk.Label(title_frame, text="(要素未選択)")
        self.element_label.pack(anchor=tk.W)
        
        # Notebook for categories
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: Section Properties
        self.section_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.section_tab, text='断面')
        self.create_section_tab()
        
        # Tab 2: Material Properties
        self.material_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.material_tab, text='材料')
        self.create_material_tab()
        
        # Tab 3: Connection Properties
        self.connection_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.connection_tab, text='接合部')
        self.create_connection_tab()
        
        # Tab 4: Analysis Parameters
        self.analysis_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.analysis_tab, text='解析設定')
        self.create_analysis_tab()
        
    def create_section_tab(self):
        # Section selector
        ttk.Label(self.section_tab, text="断面:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.section_var = tk.StringVar(value="(未選択)")
        section_entry = ttk.Entry(self.section_tab, textvariable=self.section_var, width=20, state='readonly')
        section_entry.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Button(self.section_tab, text="選択...", command=self.open_section_selector).grid(row=0, column=2, padx=5)
        
        # Section properties (read-only display)
        props_frame = ttk.LabelFrame(self.section_tab, text="断面諸元", padding=5)
        props_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        
        self.section_props_labels = {}
        prop_names = [
            ('area', '断面積 A:', 'mm²'),
            ('Ix', 'Ix (強軸):', 'mm⁴'),
            ('Iy', 'Iy (弱軸):', 'mm⁴'),
            ('Zx', 'Zx:', 'mm³'),
            ('weight', '単位重量:', 'kg/m'),
        ]
        
        for i, (key, label, unit) in enumerate(prop_names):
            ttk.Label(props_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            val_label = ttk.Label(props_frame, text="-")
            val_label.grid(row=i, column=1, sticky=tk.W)
            ttk.Label(props_frame, text=unit).grid(row=i, column=2, sticky=tk.W, padx=5)
            self.section_props_labels[key] = val_label
        
        # Custom section input
        custom_frame = ttk.LabelFrame(self.section_tab, text="カスタム入力", padding=5)
        custom_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        
        ttk.Label(custom_frame, text="A (m²):").grid(row=0, column=0, sticky=tk.W)
        self.custom_a_var = tk.StringVar()
        ttk.Entry(custom_frame, textvariable=self.custom_a_var, width=15).grid(row=0, column=1)
        
        ttk.Label(custom_frame, text="Iy (m⁴):").grid(row=1, column=0, sticky=tk.W)
        self.custom_iy_var = tk.StringVar()
        ttk.Entry(custom_frame, textvariable=self.custom_iy_var, width=15).grid(row=1, column=1)
        
        ttk.Label(custom_frame, text="Iz (m⁴):").grid(row=2, column=0, sticky=tk.W)
        self.custom_iz_var = tk.StringVar()
        ttk.Entry(custom_frame, textvariable=self.custom_iz_var, width=15).grid(row=2, column=1)
        
        ttk.Button(custom_frame, text="カスタム適用", command=self.apply_custom_section).grid(row=3, column=0, columnspan=2, pady=5)
        
    def create_material_tab(self):
        # Material selector
        ttk.Label(self.material_tab, text="材料:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.material_var = tk.StringVar(value="SS400")
        material_entry = ttk.Entry(self.material_tab, textvariable=self.material_var, width=20, state='readonly')
        material_entry.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Button(self.material_tab, text="選択...", command=self.open_material_selector).grid(row=0, column=2, padx=5)
        
        # Material properties
        props_frame = ttk.LabelFrame(self.material_tab, text="材料特性", padding=5)
        props_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        
        self.material_props_labels = {}
        prop_names = [
            ('E', 'ヤング率 E:', 'GPa'),
            ('G', 'せん断弾性率 G:', 'GPa'),
            ('Fy', '降伏強度 Fy:', 'MPa'),
            ('Fu', '極限強度 Fu:', 'MPa'),
            ('rho', '密度 ρ:', 'kg/m³'),
        ]
        
        for i, (key, label, unit) in enumerate(prop_names):
            ttk.Label(props_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            val_label = ttk.Label(props_frame, text="-")
            val_label.grid(row=i, column=1, sticky=tk.W)
            ttk.Label(props_frame, text=unit).grid(row=i, column=2, sticky=tk.W, padx=5)
            self.material_props_labels[key] = val_label
        
        # Set default material
        self.set_material(MATERIAL_LIB.get_material('SS400'))
        
    def create_connection_tab(self):
        # Connection type (端部i)
        ttk.Label(self.connection_tab, text="接合部タイプ:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # End i
        end_i_frame = ttk.LabelFrame(self.connection_tab, text="端部 i (始端)", padding=5)
        end_i_frame.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N, padx=5)
        
        self.conn_type_i = tk.StringVar(value='rigid')
        ttk.Radiobutton(end_i_frame, text="剛接合", variable=self.conn_type_i, value='rigid').pack(anchor=tk.W)
        ttk.Radiobutton(end_i_frame, text="ピン接合", variable=self.conn_type_i, value='pinned').pack(anchor=tk.W)
        ttk.Radiobutton(end_i_frame, text="半剛接", variable=self.conn_type_i, value='semi-rigid').pack(anchor=tk.W)
        
        # End j
        end_j_frame = ttk.LabelFrame(self.connection_tab, text="端部 j (終端)", padding=5)
        end_j_frame.grid(row=1, column=1, sticky=tk.W+tk.E+tk.N, padx=5)
        
        self.conn_type_j = tk.StringVar(value='rigid')
        ttk.Radiobutton(end_j_frame, text="剛接合", variable=self.conn_type_j, value='rigid').pack(anchor=tk.W)
        ttk.Radiobutton(end_j_frame, text="ピン接合", variable=self.conn_type_j, value='pinned').pack(anchor=tk.W)
        ttk.Radiobutton(end_j_frame, text="半剛接", variable=self.conn_type_j, value='semi-rigid').pack(anchor=tk.W)
        
        # Weld settings
        weld_frame = ttk.LabelFrame(self.connection_tab, text="溶接設定", padding=5)
        weld_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10, padx=5)
        
        ttk.Label(weld_frame, text="溶接タイプ:").grid(row=0, column=0, sticky=tk.W)
        self.weld_type_var = tk.StringVar(value='完全溶け込み')
        weld_combo = ttk.Combobox(weld_frame, textvariable=self.weld_type_var, width=20)
        weld_combo['values'] = ['完全溶け込み', '隅肉溶接', '部分溶け込み']
        weld_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(weld_frame, text="隅肉サイズ (mm):").grid(row=1, column=0, sticky=tk.W)
        self.weld_size_var = tk.StringVar(value="6")
        ttk.Entry(weld_frame, textvariable=self.weld_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Bolt settings
        bolt_frame = ttk.LabelFrame(self.connection_tab, text="ボルト設定", padding=5)
        bolt_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E, padx=5)
        
        ttk.Label(bolt_frame, text="ボルト径:").grid(row=0, column=0, sticky=tk.W)
        self.bolt_dia_var = tk.StringVar(value='M22')
        bolt_combo = ttk.Combobox(bolt_frame, textvariable=self.bolt_dia_var, width=10)
        bolt_combo['values'] = ['M16', 'M20', 'M22', 'M24', 'M27', 'M30']
        bolt_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(bolt_frame, text="本数:").grid(row=1, column=0, sticky=tk.W)
        self.bolt_count_var = tk.StringVar(value="8")
        ttk.Entry(bolt_frame, textvariable=self.bolt_count_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
    def create_analysis_tab(self):
        # Hysteresis model
        ttk.Label(self.analysis_tab, text="履歴モデル:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.hysteresis_var = tk.StringVar(value='Takeda')
        hyst_combo = ttk.Combobox(self.analysis_tab, textvariable=self.hysteresis_var, width=15)
        hyst_combo['values'] = ['Takeda', 'Bilinear', 'Trilinear', 'Clough']
        hyst_combo.grid(row=0, column=1, padx=5)
        
        # Post-yield stiffness ratio
        ttk.Label(self.analysis_tab, text="降伏後剛性比 r:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.r_var = tk.StringVar(value="0.05")
        ttk.Entry(self.analysis_tab, textvariable=self.r_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Yield moment calculation
        calc_frame = ttk.LabelFrame(self.analysis_tab, text="降伏モーメント", padding=5)
        calc_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        
        ttk.Label(calc_frame, text="My (kN·m):").grid(row=0, column=0, sticky=tk.W)
        self.my_var = tk.StringVar(value="0")
        ttk.Entry(calc_frame, textvariable=self.my_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Button(calc_frame, text="自動計算 (My=Z×Fy)", command=self.calculate_yield_moment).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Damping
        damp_frame = ttk.LabelFrame(self.analysis_tab, text="減衰", padding=5)
        damp_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        
        ttk.Label(damp_frame, text="減衰定数 h:").grid(row=0, column=0, sticky=tk.W)
        self.damping_var = tk.StringVar(value="0.05")
        ttk.Entry(damp_frame, textvariable=self.damping_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
    def open_section_selector(self):
        """Open section selector dialog."""
        SectionSelectorDialog(self, self.set_section)
    
    def open_material_selector(self):
        """Open material selector dialog."""
        MaterialSelectorDialog(self, on_select=self.set_material)
    
    def set_section(self, section: SteelSection):
        """Set selected section and update display."""
        self.current_section = section
        self.section_var.set(section.name)
        
        # Update property labels
        self.section_props_labels['area'].config(text=f"{section.A:.0f}")
        self.section_props_labels['Ix'].config(text=f"{section.Ix:.2e}")
        self.section_props_labels['Iy'].config(text=f"{section.Iy:.2e}")
        self.section_props_labels['Zx'].config(text=f"{section.Zx:.0f}")
        self.section_props_labels['weight'].config(text=f"{section.weight:.1f}")
        
        # Update custom input fields with SI values
        si = section.to_si_units()
        self.custom_a_var.set(f"{si['A']:.6e}")
        self.custom_iy_var.set(f"{si['I_y']:.6e}")
        self.custom_iz_var.set(f"{si['I_z']:.6e}")
        
        # Calculate yield moment
        self.calculate_yield_moment()
        
        if self.on_property_change:
            self.on_property_change('section', section)
    
    def set_material(self, material: MaterialProperties):
        """Set selected material and update display."""
        if material is None:
            return
            
        self.current_material = material
        self.material_var.set(material.name)
        
        self.material_props_labels['E'].config(text=f"{material.E/1e9:.0f}")
        self.material_props_labels['G'].config(text=f"{material.G/1e9:.0f}")
        self.material_props_labels['Fy'].config(text=f"{material.Fy/1e6:.0f}")
        self.material_props_labels['Fu'].config(text=f"{material.Fu/1e6:.0f}")
        self.material_props_labels['rho'].config(text=f"{material.rho:.0f}")
        
        if self.on_property_change:
            self.on_property_change('material', material)
    
    def apply_custom_section(self):
        """Apply custom section values."""
        try:
            a = float(self.custom_a_var.get())
            iy = float(self.custom_iy_var.get())
            iz = float(self.custom_iz_var.get())
            
            if self.on_property_change:
                self.on_property_change('custom_section', {'A': a, 'Iy': iy, 'Iz': iz})
        except ValueError:
            pass
    
    def calculate_yield_moment(self):
        """Calculate yield moment from section Z and material Fy."""
        if self.current_section and self.current_material:
            Zx = self.current_section.Zx * 1e-9  # mm³ to m³
            Fy = self.current_material.Fy  # Pa
            My = Zx * Fy  # N·m
            self.my_var.set(f"{My/1e3:.1f}")  # Display in kN·m
    
    def set_element(self, element_info: dict):
        """Set the current element being edited."""
        self.current_element = element_info
        name = element_info.get('name', '(不明)')
        self.element_label.config(text=f"選択中: {name}")
