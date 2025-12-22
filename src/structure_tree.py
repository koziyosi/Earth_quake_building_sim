"""
Structure Tree View Module.
Implements hierarchical tree view for building structure (#97).
"""
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Callable, Optional, Any


class StructureTreeView(ttk.Frame):
    """
    Hierarchical tree view for building structure.
    
    Structure:
    - Project
      - Floors
        - Floor 1
          - Columns
          - Beams
        - Floor 2
          ...
      - Materials
      - Sections
    """
    
    def __init__(
        self, 
        parent, 
        layout = None,
        on_select: Optional[Callable[[str, Any], None]] = None
    ):
        super().__init__(parent)
        
        self.layout = layout
        self.on_select = on_select
        
        self._setup_ui()
        
        if layout:
            self.refresh_tree()
            
    def _setup_ui(self):
        """Setup the tree view UI."""
        # Search/filter
        search_frame = ttk.Frame(self)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="🔍").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self._on_search_change)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Tree view
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, selectmode='browse')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure columns
        self.tree['columns'] = ('type', 'info')
        self.tree.column('#0', width=200, minwidth=100)
        self.tree.column('type', width=80, minwidth=50)
        self.tree.column('info', width=100, minwidth=50)
        
        self.tree.heading('#0', text='Name')
        self.tree.heading('type', text='Type')
        self.tree.heading('info', text='Info')
        
        # Bind events
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.tree.bind('<Double-1>', self._on_double_click)
        
        # Context menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Edit", command=self._edit_selected)
        self.context_menu.add_command(label="Delete", command=self._delete_selected)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Select All of Type", command=self._select_same_type)
        
        self.tree.bind('<Button-3>', self._show_context_menu)
        
    def refresh_tree(self):
        """Refresh tree from layout data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if not self.layout:
            return
            
        # Root items
        project_id = self.tree.insert('', 'end', text='📁 Project', open=True)
        
        # Floors
        floors_id = self.tree.insert(project_id, 'end', text='🏢 Floors', open=True)
        
        if hasattr(self.layout, 'floors'):
            for i, floor in enumerate(self.layout.floors):
                floor_id = self.tree.insert(
                    floors_id, 'end',
                    text=f'📐 Floor {i+1}',
                    values=('Floor', f'H={floor.height if hasattr(floor, "height") else 3.5}m'),
                    open=False
                )
                
                # Add columns
                cols_id = self.tree.insert(floor_id, 'end', text='⬜ Columns')
                if hasattr(floor, 'columns'):
                    for j, col in enumerate(floor.columns):
                        self.tree.insert(
                            cols_id, 'end',
                            text=f'Column C{j+1}',
                            values=('Column', col.section if hasattr(col, 'section') else '-'),
                            tags=('element', f'column_{i}_{j}')
                        )
                        
                # Add beams
                beams_id = self.tree.insert(floor_id, 'end', text='➖ Beams')
                if hasattr(floor, 'beams'):
                    for j, beam in enumerate(floor.beams):
                        self.tree.insert(
                            beams_id, 'end',
                            text=f'Beam B{j+1}',
                            values=('Beam', beam.section if hasattr(beam, 'section') else '-'),
                            tags=('element', f'beam_{i}_{j}')
                        )
        else:
            # Simplified structure based on grid
            if hasattr(self.layout, 'grid'):
                n_stories = len(self.layout.grid.story_heights)
                for i in range(n_stories):
                    self.tree.insert(
                        floors_id, 'end',
                        text=f'📐 Floor {i+1}',
                        values=('Floor', f'H={self.layout.grid.story_heights[i]}m')
                    )
                    
        # Materials
        materials_id = self.tree.insert(project_id, 'end', text='🧱 Materials')
        if hasattr(self.layout, 'materials'):
            for mat_id, mat in self.layout.materials.items():
                self.tree.insert(
                    materials_id, 'end',
                    text=f'{mat_id}',
                    values=('Material', mat.get('name', '-'))
                )
                
        # Sections
        sections_id = self.tree.insert(project_id, 'end', text='📏 Sections')
        if hasattr(self.layout, 'sections'):
            for sec_id, sec in self.layout.sections.items():
                self.tree.insert(
                    sections_id, 'end',
                    text=f'{sec_id}',
                    values=('Section', sec.get('name', '-'))
                )
                
    def set_layout(self, layout):
        """Set the layout object."""
        self.layout = layout
        self.refresh_tree()
        
    def _on_tree_select(self, event):
        """Handle tree selection."""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            item_text = self.tree.item(item, 'text')
            item_values = self.tree.item(item, 'values')
            item_tags = self.tree.item(item, 'tags')
            
            if self.on_select:
                self.on_select(item_text, {
                    'values': item_values,
                    'tags': item_tags,
                    'id': item
                })
                
    def _on_double_click(self, event):
        """Handle double-click (edit)."""
        self._edit_selected()
        
    def _on_search_change(self, *args):
        """Handle search text change."""
        search_text = self.search_var.get().lower()
        
        if not search_text:
            # Show all items
            self._show_all_items()
            return
            
        # Filter items
        self._filter_items(search_text)
        
    def _show_all_items(self):
        """Show all items in tree."""
        def show_children(item):
            self.tree.item(item, open=False)
            for child in self.tree.get_children(item):
                show_children(child)
                
        for item in self.tree.get_children():
            show_children(item)
            
    def _filter_items(self, search_text: str):
        """Filter and expand matching items."""
        def check_children(item) -> bool:
            item_text = self.tree.item(item, 'text').lower()
            item_values = ' '.join(str(v).lower() for v in self.tree.item(item, 'values'))
            
            # Check this item
            matches = search_text in item_text or search_text in item_values
            
            # Check children
            any_child_matches = False
            for child in self.tree.get_children(item):
                if check_children(child):
                    any_child_matches = True
                    
            # Expand if matches found
            if matches or any_child_matches:
                self.tree.item(item, open=True)
                return True
            return False
            
        for item in self.tree.get_children():
            check_children(item)
            
    def _show_context_menu(self, event):
        """Show context menu."""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
            
    def _edit_selected(self):
        """Edit selected item."""
        selection = self.tree.selection()
        if selection:
            # Would open property editor dialog
            pass
            
    def _delete_selected(self):
        """Delete selected item."""
        selection = self.tree.selection()
        if selection:
            # Would delete with confirmation
            pass
            
    def _select_same_type(self):
        """Select all items of same type (#98 batch edit)."""
        selection = self.tree.selection()
        if not selection:
            return
            
        item_values = self.tree.item(selection[0], 'values')
        if not item_values:
            return
            
        item_type = item_values[0]
        
        # Find all items of same type
        same_type_items = []
        
        def find_same_type(item):
            values = self.tree.item(item, 'values')
            if values and values[0] == item_type:
                same_type_items.append(item)
            for child in self.tree.get_children(item):
                find_same_type(child)
                
        for item in self.tree.get_children():
            find_same_type(item)
            
        # Select all
        self.tree.selection_set(same_type_items)
        
    def get_selected_elements(self) -> List[str]:
        """Get list of selected element IDs."""
        selection = self.tree.selection()
        elements = []
        
        for item in selection:
            tags = self.tree.item(item, 'tags')
            if tags and 'element' in tags:
                elements.append(tags[1] if len(tags) > 1 else item)
                
        return elements


class BatchEditDialog(tk.Toplevel):
    """
    Dialog for batch editing multiple elements (#98).
    """
    
    def __init__(self, parent, elements: List, property_name: str):
        super().__init__(parent)
        
        self.title(f"Batch Edit: {property_name}")
        self.geometry("400x300")
        
        self.elements = elements
        self.result = None
        
        self._setup_ui(property_name)
        
    def _setup_ui(self, property_name: str):
        ttk.Label(self, text=f"Editing {len(self.elements)} elements").pack(pady=10)
        ttk.Label(self, text=f"Set new value for: {property_name}").pack()
        
        self.value_var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self.value_var)
        entry.pack(pady=20, padx=20, fill=tk.X)
        entry.focus()
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
    def _apply(self):
        self.result = self.value_var.get()
        self.destroy()
