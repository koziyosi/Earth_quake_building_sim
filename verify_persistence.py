import sys
import os
import json
import tkinter as tk

sys.path.append(os.getcwd())
from src.layout_model import BuildingLayout, GridSystem, FloorLayout
from src.gui_layout import LayoutEditorPanel

def verify_persistence():
    print("Testing Persistence & Sync...")
    
    # 1. Test Serialization
    layout = BuildingLayout()
    layout.initialize_default()
    
    # Modify
    layout.grid.x_spacings = [6.0, 6.0]
    layout.cleanup_elements() # Important: Remove out-of-bounds elements!
    # Add floor 4 equivalent
    layout.grid.story_heights.append(3.5)
    layout.get_floor(4) # Initialize floor 4
    
    data = layout.to_dict()
    # Check
    assert len(data['grid']['x_spacings']) == 2
    # Floors should be 1, 2, 3, 4. initialize_default made 1,2,3. We added 4.
    # So length 4?
    # initialize_default iterates story_heights.
    # Default story_heights len=3. So 1,2,3.
    # we appended. data['floors'] depends on self.floors.values()
    # we called get_floor(4) which created it.
    # So 4 floors.
    assert len(data['floors']) == 4
    
    print("Serialization OK.")
    
    # 2. Test File I/O
    filename = "test_layout.json"
    layout.save_to_file(filename)
    
    loaded = BuildingLayout.load_from_file(filename)
    assert len(loaded.grid.x_spacings) == 2
    print("File I/O OK.")
    import os
    os.remove(filename)
    
    # 3. Test Sync Callback
    root = tk.Tk()
    
    callback_fired = False
    def my_callback():
        nonlocal callback_fired
        callback_fired = True
        
    panel = LayoutEditorPanel(root, layout, on_change_callback=my_callback)
    
    # Simulate Add Floor
    print("Testing Sync Callback (Add Floor)...")
    panel.add_floor()
    assert callback_fired == True
    print("Callback Fired OK.")
    
    # Simulate Remove Floor
    callback_fired = False
    panel.remove_floor()
    assert callback_fired == True
    print("Callback Fired (Remove) OK.")
    
    print("All Persistence & Sync Tests Passed.")
    root.destroy()

if __name__ == "__main__":
    verify_persistence()
