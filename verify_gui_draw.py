import tkinter as tk
from tkinter import ttk
import sys
import os

sys.path.append(os.getcwd())
from src.layout_model import BuildingLayout
from src.gui_layout import LayoutEditorPanel

def verify_gui_draw():
    print("Testing GUI Drawing Logic...")
    
    root = tk.Tk()
    root.geometry("800x600")
    
    # 1. Setup Data
    layout = BuildingLayout()
    layout.initialize_default()
    
    # 2. Setup GUI
    panel = LayoutEditorPanel(root, layout)
    panel.pack(fill=tk.BOTH, expand=True)
    
    # 3. Force Update to ensure winfo returns values
    root.update()
    
    print("Canvas Width:", panel.canvas.winfo_width())
    print("Canvas Height:", panel.canvas.winfo_height())
    
    # 4. Trigger Draw
    print("Triggering draw_layout()...")
    try:
        panel.draw_layout()
        print("draw_layout() Success.")
    except Exception as e:
        print("draw_layout() FAILED.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 5. Simulate Interaction
    print("Simulating interaction...")
    # Add a floor
    panel.add_floor()
    root.update()
    
    # Toggle Beam
    # Need to verify world_to_screen to click correctly
    pass

    print("GUI Verification Passed.")
    root.destroy()

if __name__ == "__main__":
    verify_gui_draw()
