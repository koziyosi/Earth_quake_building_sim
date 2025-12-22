import sys
import os
import numpy as np

# Adjust path to import src
sys.path.append(os.getcwd())

from src.layout_model import BuildingLayout
from src.builder import BuildingBuilder
import main_3d as sim3d

def verify_layout_builder():
    print("Testing BuildingLayout Logic...")
    
    # 1. Create Layout
    layout = BuildingLayout()
    layout.initialize_default()
    
    # Grid: default 3 spaces X (4 lines), 3 spaces Y.
    # Floor 0 (Base), Floor 1, Floor 2, Floor 3 (Roof).
    # Total 3 Stories.
    
    # 2. Modify Layout
    # Remove a column at (1, 1) on Floor 1
    f1 = layout.get_floor(1)
    if (1, 1) in f1.columns:
        print("Removing column at (1,1) on Floor 1")
        f1.remove_column(1, 1)
        
    # Remove beam between (0,0)-(1,0) on Floor 2
    f2 = layout.get_floor(2)
    f2.remove_beam((0,0), (1,0))
    print("Removing beam (0,0)-(1,0) on Floor 2")
    
    # 3. New Tests: Add Floor and Add Wall
    print("Testing Dynamic Floors and Walls...")
    
    # Add a floor (Floor 4)
    layout.grid.story_heights.append(3.5)
    f4 = layout.get_floor(4) # Initialize
    
    # Add a Wall on Floor 1
    # Adding Wall between (0,1) and (1,1)
    layout.get_floor(1).add_beam((0,1), (1,1), "W1")
    
    # Count expected elements
    # Originally 3 Stories. Now 4 Stories.
    # Floor 4 (Roof) has no columns on top?
    # Our logic: Floor i Layout has columns connecting i-1 to i.
    # So F1 connects 0-1, F2 connects 1-2, F3 connects 2-3, F4 connects 3-4.
    # F4 is initialized empty.
    # Let's add default columns to F4 to match standard.
    nx = len(layout.grid.x_spacings) + 1
    ny = len(layout.grid.y_spacings) + 1
    for i in range(nx):
        for j in range(ny):
            f4.add_column(i, j, "C1")
    # Add beams F4
    for j in range(ny):
         for i in range(nx - 1):
             f4.add_beam((i, j), (i+1, j), "B1")
    for i in range(nx):
         for j in range(ny - 1):
             f4.add_beam((i, j), (i, j+1), "B1")
             
    # Recalculate Expected Count
    # 4 Stories.
    # Per floor: 16 columns, 24 beams (Total 40).
    # 4 * 40 = 160 elements.
    
    # Modifications:
    # 1. Removed Col (1,1) on F1 -> -1
    # 2. Removed Beam (0,0)-(1,0) on F2 -> -1
    # 3. Added Wall "W1" at (0,1)-(1,1) on F1.
    #    "W1" replaces "B1"? or Add?
    #    If we just call `add_beam`, it overwrites existing B1 if present.
    #    Yes, (0,1)-(1,1) is a standard beam location.
    #    It overwrites the Beam element with a Wall element.
    #    Wall element generates 2 Braces independently.
    #    So: -1 Beam, +2 Braces = +1 Net Element.
    
    # Net: 160 - 1 - 1 + 1 = 159 elements.
    # Actually, Wall is colored 'red'.
    # We should verify 'custom_color' attribute on Walls.
    
    print("Building FE Model (Phase 3)...")
    nodes, elements = BuildingBuilder.build_from_layout(layout)
    print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")
    
    wall_count = 0
    for el in elements:
        if hasattr(el, 'custom_color') and el.custom_color == 'red':
            wall_count += 1
            
    print(f"Wall Elements (Red): {wall_count}")
    # We added 1 Wall "W1" which generates 2 Braces.
    # So expected wall_count = 2.
    
    if len(elements) == 159 and wall_count == 2:
        print("SUCCESS: Element count 159, Wall count 2.")
    else:
        print(f"FAILURE: Count {len(elements)}, Wall {wall_count}")

    if len(elements) == 159 and wall_count == 2:
        print("SUCCESS: Element count 159, Wall count 2.")
    else:
        print(f"FAILURE: Count {len(elements)}, Wall {wall_count}")

    # 4. New Test: Grid Resizing
    print("Testing Grid Resizing...")
    # Change X Grid from [6,6,6] (3 spans) to [5, 5] (2 spans).
    # This should remove Columns at index 3.
    # And Beams at index 2 (connecting 2-3).
    layout.grid.x_spacings = [5.0, 5.0]
    layout.cleanup_elements()
    
    print("Building FE Model (Phase 4 - Resized)...")
    nodes_r, elements_r = BuildingBuilder.build_from_layout(layout)
    print(f"Nodes: {len(nodes_r)}, Elements: {len(elements_r)}")
    
    # Expected:
    # 4 Floors (0,1,2,3... wait, we added floor 4 earlier).
    # 4 Stories High (Floors 0,1,2,3,4).
    # X Grid: 2 spans (3 lines: 0,1,2).
    # Y Grid: 3 spans (4 lines: 0,1,2,3).
    # Nodes per floor = 3 * 4 = 12.
    # Total Nodes = 12 * 5 = 60.
    
    # Elements:
    # Columns: 12 per floor * 4 stories = 48.
    # Beams X: 2 spans * 4 lines = 8 per floor.
    # Beams Y: 3 lines * 3 spans? No. 3 spans * 3 lines ?
    # Beams Y: 3 spans * 3 lines (x=0,1,2) = 9 per floor.
    # Total Beams per floor = 17.
    # Elevated floors: 4 (1,2,3,4).
    # Beams total = 17 * 4 = 68.
    
    # Total Elements = 48 + 68 = 116.
    # Corrections:
    # 1. Removed Col F1(1,1) -> -1. (Total 115).
    # 2. Removed Beam F2(0,0)-(1,0) -> Left side beam -> -1. (Total 114).
    # 3. Wall F1(0,1)-(1,1) -> Replaces Beam -> +1 Net. (Total 115).
    # Final Expectation: 115.
    
    if len(elements_r) == 115:
         print("SUCCESS: Resized Element count matches expected (115).")
    else:
         print(f"FAILURE: Resized Count {len(elements_r)} != 115")

    # 5. Run Simulation
    print("Running Simulation with Custom Layout...")
    try:
        gif, t, res = sim3d.run_3d_simulation(duration=0.5, layout=layout)
        print("Simulation completed successfully.")
    except Exception as e:
        print(f"Simulation FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_layout_builder()
