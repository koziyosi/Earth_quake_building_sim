import numpy as np
from typing import List, Tuple
from src.fem import Node
from src.fem_3d import BeamColumn3D
from src.devices import BaseIsolator, OilDamper
from src.layout_model import BuildingLayout

class BuildingBuilder:
    @staticmethod
    def build_model(floors: int, span_x: float, span_y: float, story_h: float,
                    soft_first_story: bool, base_isolation: bool, add_dampers: bool):
        """
        Builds a 3D frame model based on parameters.
        Returns: nodes, elements
        """
        nodes = []
        elements = []
        node_id = 1
        elem_id = 1
        
        # Grid: 1 span in X, 1 span in Y (4 columns)
        # Coordinates: (0,0), (Sx,0), (Sx,Sy), (0,Sy)
        plan = [(0,0), (span_x,0), (span_x,span_y), (0,span_y)]
        
        # Z levels
        levels = [0.0]
        current_z = 0.0
        
        # If Base Isolation, we insert a "Foundation" level at 0, and "Base" level at 0.5m
        if base_isolation:
            current_z += 0.5 # Isolator height
            levels.append(current_z)
            
        for i in range(floors):
            h = story_h
            if i == 0 and soft_first_story:
                h = story_h * 1.5 # 1.5x height for soft story
            current_z += h
            levels.append(current_z)
            
        # Create Nodes
        # Map: (level_idx, plan_idx) -> Node
        node_map = {}
        
        for i, z in enumerate(levels):
            mass = 20000.0 if i > 0 else 0.0 # No mass at foundation
            
            # If isolation, Level 0 is foundation (Fixed), Level 1 is Base (Massive?)
            # Usually Base Slab is heavy.
            if base_isolation and i == 1:
                mass = 50000.0 # Heavy base slab
            elif base_isolation and i == 0:
                mass = 0.0
                
            for j, (x, y) in enumerate(plan):
                n = Node(node_id, x, y, z, mass=mass)
                nodes.append(n)
                node_map[(i, j)] = n
                node_id += 1
                
        # Assign DOFs
        dof_counter = 0
        for n in nodes:
            # Fixed Base
            if n.z == 0.0:
                n.set_dof_indices([-1]*6)
                continue
            
            indices = list(range(dof_counter, dof_counter+6))
            n.set_dof_indices(indices)
            dof_counter += 6
            
        # Create Elements
        
        # Materials
        E = 2.5e10
        G = 1.0e10
        A = 0.25; Iy = 0.005; Iz = 0.005; J = 0.01
        
        My_col = 200000.0
        My_beam = 150000.0
        
        start_level = 0
        if base_isolation:
            # Isolators between Level 0 and 1
            Kv = 1.0e9 # High vertical
            Kh = 1.0e6 # Low horizontal (1 MN/m) -> T ~ 2pi * sqrt(50t / 4MN/m) ~ 1.4s
            Fy = 50000.0 # Yield force
            
            for j in range(4):
                n_bot = node_map[(0, j)]
                n_top = node_map[(1, j)]
                
                iso = BaseIsolator(elem_id, n_bot, n_top, Kv, Kh, Fy, 0.1)
                elements.append(iso)
                elem_id += 1
                
            start_level = 1
            
        # Columns
        for i in range(start_level, len(levels)-1):
            for j in range(4):
                n_bot = node_map[(i, j)]
                n_top = node_map[(i+1, j)]
                
                # Soft story logic: Reduce A/I? Or just Length does it?
                # Length is already handled by Z coordinates.
                # If soft first story, maybe reduce stiffness too?
                # Let's just rely on Length (Stiffness ~ 1/L^3). 1.5L -> 0.3x Stiffness.
                
                col = BeamColumn3D(elem_id, n_bot, n_top, E, G, A, Iy, Iz, J)
                col.set_yield_properties(My_col, My_col)
                elements.append(col)
                elem_id += 1
                
                # Dampers (Diagonal braces)
                if add_dampers:
                    # Add damper in X direction frame?
                    # Between (i, 0)-(i+1, 1)?
                    # Let's add X-braced dampers in X-frames.
                    # Frame 1: Nodes 0-1. Frame 2: Nodes 3-2.
                    if j == 0: # (x=0, y=0) -> (x=Sx, y=0) is neighbor 1
                        # Diagonal 0->1_top
                        n_next_top = node_map[(i+1, 1)]
                        damp = OilDamper(elem_id, n_bot, n_next_top, 500000.0) # 500 kNs/m
                        elements.append(damp)
                        elem_id += 1
                    elif j == 3: # (0, Sy) -> (Sx, Sy) is neighbor 2
                        n_next_top = node_map[(i+1, 2)]
                        damp = OilDamper(elem_id, n_bot, n_next_top, 500000.0)
                        elements.append(damp)
                        elem_id += 1
                        
        # Beams
        for i in range(start_level+1, len(levels)):
            pairs = [(0,1), (1,2), (2,3), (3,0)]
            for (j1, j2) in pairs:
                n1 = node_map[(i, j1)]
                n2 = node_map[(i, j2)]
                
                bm = BeamColumn3D(elem_id, n1, n2, E, G, A, Iy, Iz, J)
                bm.set_yield_properties(My_beam, My_beam)
                elements.append(bm)
                elem_id += 1
                
        return nodes, elements

    @staticmethod
    def build_from_layout(layout: BuildingLayout) -> Tuple[List[Node], List[BeamColumn3D]]:
        """
        Constructs the FE Model from a BuildingLayout object.
        """
        nodes = []
        elements = []
        node_map = {} # (floor_idx, gx, gy) -> Node
        
        grid = layout.grid
        x_coords = grid.get_x_coords()
        y_coords = grid.get_y_coords()
        z_coords = [0.0]
        current_z = 0.0
        for h in grid.story_heights:
            current_z += h
            z_coords.append(current_z)
            
        n_floors = len(z_coords)
        node_id_counter = 0
        
        # Calculate building height for adaptive section scaling
        building_height = sum(grid.story_heights)
        
        # Adaptive scaling factor based on building height
        # Default properties are designed for ~3-story (10m) buildings
        # For taller buildings, scale section properties to maintain stability
        # Super-tall buildings need exponential scaling to prevent numerical instability
        if building_height <= 15:
            section_scale = 1.0
        elif building_height <= 40:
            section_scale = 3.0
        elif building_height <= 100:
            section_scale = 10.0
        elif building_height <= 200:
            section_scale = 40.0
        else:
            # Super-tall: exponential scaling to handle extreme heights
            # 270m -> ~80, 300m -> ~100, 400m -> ~150
            section_scale = 40.0 * (1 + (building_height - 200) / 100)
        
        print(f"Building height: {building_height:.1f}m, section scale factor: {section_scale:.1f}")
        
        # 1. Create Nodes
        for f_idx in range(n_floors):
            for gx, x in enumerate(x_coords):
                for gy, y in enumerate(y_coords):
                    z = z_coords[f_idx]
                    
                    # Logic to determine mass
                    # Only add mass if there are elements connected to this node?
                    # For now simplistically add mass to all nodes above ground.
                    # Or check layout usage.
                    # Simple: Mass for all.
                    mass = 0.0
                    if f_idx > 0:
                        # Scale mass proportionally with section scaling
                        # to maintain realistic natural frequencies
                        # Mass should scale roughly with section_scale^0.5 to ^1.0
                        # (mass ~ volume ~ A * L, and if A scales, mass scales similarly)
                        mass_scale = min(section_scale ** 0.7, 20.0)  # Power-law scaling with cap
                        mass = 15000.0 * mass_scale
                        
                    n = Node(node_id_counter, x, y, z, mass=mass)
                    nodes.append(n)
                    node_map[(f_idx, gx, gy)] = n
                    node_id_counter += 1
                    
        # 2. Assign DOFs
        dof_counter = 0
        for n in nodes:
            if n.z == 0.0:
                 n.set_dof_indices([-1]*6)
            else:
                 indices = list(range(dof_counter, dof_counter+6))
                 n.set_dof_indices(indices)
                 dof_counter += 6
                 
        # 3. Create Elements based on Layout
        elem_id_counter = 0
        
        # Iterate "Stories" (intervals between floors)
        # Story i connects Floor i-1 to Floor i.
        # FloorLayout i defines columns in this Story i.
        
        for f_idx in range(1, n_floors):
            floor_layout = layout.get_floor(f_idx)
            
            # Columns (Vertical elements below this floor)
            for (gx, gy), section_name in floor_layout.columns.items():
                if section_name:
                    n_bot = node_map.get((f_idx-1, gx, gy))
                    n_top = node_map.get((f_idx, gx, gy))
                    
                    props = layout.sections[section_name]
                    
                    # Apply section scaling for tall buildings
                    scaled_I_y = props.I_y * section_scale
                    scaled_I_z = props.I_z * section_scale
                    scaled_J = props.J * section_scale
                    scaled_A = props.area * min(section_scale, 10.0)  # Cap area scaling
                    scaled_My = props.yield_moment * section_scale
                    
                    # Create Element
                    # G assumption: E/2.5 for concrete
                    G_mat = props.E / 2.5
                    
                    el = BeamColumn3D(elem_id_counter, n_bot, n_top, 
                                      props.E, G_mat, scaled_A, scaled_I_y, scaled_I_z, scaled_J)
                    el.set_yield_properties(scaled_My, scaled_My)
                    
                    elements.append(el)
                    elem_id_counter += 1

            # Beams (Horizontal elements AT this floor)
            for ((gx1, gy1), (gx2, gy2)), section_name in floor_layout.beams.items():
                if section_name:
                    n1 = node_map.get((f_idx, gx1, gy1))
                    n2 = node_map.get((f_idx, gx2, gy2))
                    
                    props = layout.sections[section_name]
                    G_mat = props.E / 2.5
                    
                    # Apply section scaling for tall buildings
                    scaled_I_y = props.I_y * section_scale
                    scaled_I_z = props.I_z * section_scale
                    scaled_J = props.J * section_scale
                    scaled_A = props.area * min(section_scale, 10.0)
                    scaled_My = props.yield_moment * section_scale
                    
                    # Logic: If name starts with 'W', it's a Wall -> X-Brace
                    # X-brace needs Top Node and Bottom Node?
                    # The "Beam" entry is defined at floor f_idx.
                    # Does it represent a beam at floor f_idx level?
                    # OR a Wall *below* floor f_idx?
                    # "1F Layout" defines Columns below 1F.
                    # It makes sense that "1F Walls" are also below 1F (connecting 0F to 1F).
                    # But the GUI draws them as lines.
                    # If they are Walls, they should be vertical panels?
                    # No, the request says "Wall ... and represent as lines in 3D".
                    # Usually walls are vertical.
                    # If "Beam" tool places horizontal lines, effectively "Lintels" or "Girders".
                    # If user clicks "Beam" on 1F plan, it draws a line between (x1,y1) and (x2,y2) at Z=height(1F).
                    # A Wall is usually a vertical panel in that span.
                    # So a "Wall" placed at "1F" should correspond to the vertical panel between (x1,y1) and (x2,y2) spanning from Floor 0 to Floor 1?
                    # YES.
                    # So if section_name starts with 'W':
                    # We need 4 nodes: (f-1, x1, y1), (f-1, x2, y2), (f, x1, y1), (f, x2, y2).
                    # Create X-braces: (f-1,1)-(f,2) and (f-1,2)-(f,1).
                    
                    if section_name.startswith("W"):
                        # Wall Logic
                        n_bot1 = node_map.get((f_idx-1, gx1, gy1))
                        n_bot2 = node_map.get((f_idx-1, gx2, gy2))
                        n_top1 = node_map.get((f_idx, gx1, gy1)) # This is n1
                        n_top2 = node_map.get((f_idx, gx2, gy2)) # This is n2
                        
                        # Brace 1: Bot1 -> Top2
                        el1 = BeamColumn3D(elem_id_counter, n_bot1, n_top2, 
                                           props.E, G_mat, scaled_A, scaled_I_y, scaled_I_z, scaled_J)
                        # Walls are usually strong. High yield?
                        el1.set_yield_properties(scaled_My*10, scaled_My*10)
                        el1.custom_color = props.color # Pass color from SectionProperties
                        elements.append(el1)
                        elem_id_counter += 1
                        
                        # Brace 2: Bot2 -> Top1
                        el2 = BeamColumn3D(elem_id_counter, n_bot2, n_top1,
                                           props.E, G_mat, scaled_A, scaled_I_y, scaled_I_z, scaled_J)
                        el2.set_yield_properties(scaled_My*10, scaled_My*10)
                        el2.custom_color = props.color
                        elements.append(el2)
                        elem_id_counter += 1
                        
                    else:
                        # Normal Beam Logic
                        el = BeamColumn3D(elem_id_counter, n1, n2,
                                          props.E, G_mat, scaled_A, scaled_I_y, scaled_I_z, scaled_J)
                        el.set_yield_properties(scaled_My, scaled_My)
                        
                        elements.append(el)
                        elem_id_counter += 1
                    
        # Also clean up nodes that have no elements connected? 
        # Solver handles disconnected nodes (singular matrix)?
        # Ideally remove them. But for now keep simple.
        # Solver might crash if K is singular. 
        # With mass, M term stabilizes dynamics? No, K=0 -> rigid body mode. 
        # If node has Mass but no K, singular.
        # We should only return nodes that are connected.
        
        # Filter Connected Nodes
        connected_node_ids = set()
        for el in elements:
            connected_node_ids.add(el.node_i.id)
            connected_node_ids.add(el.node_j.id)
            
        # Also include fixed base nodes that support columns
        
        final_nodes = []
        # Re-index DOFs?
        # Easier: Just return all nodes, but fix DOFs of disconnected nodes?
        # Or let Solver handle it?
        # Solver uses numpy.linalg.solve. Singular matrix error likely.
        
        # Let's filter nodes.
        # Re-assign DOFs for connected nodes only.
        
        filtered_nodes = [n for n in nodes if n.id in connected_node_ids]
        
        # Re-DOF
        dof_counter = 0
        for n in filtered_nodes:
            if n.z == 0.0:
                n.set_dof_indices([-1]*6)
            else:
                indices = list(range(dof_counter, dof_counter+6))
                n.set_dof_indices(indices)
                dof_counter += 6
                
        return filtered_nodes, elements

    @staticmethod
    def build_from_template(template) -> Tuple[List[Node], List[BeamColumn3D]]:
        """
        Build a model from a BuildingTemplate object.
        
        Supports multi-bay buildings for realistic high-rise structures.
        """
        from src.building_templates import BuildingTemplate
        
        nodes = []
        elements = []
        node_id = 1
        elem_id = 1
        
        n_stories = template.n_stories
        n_bays_x = template.n_bays_x
        n_bays_y = template.n_bays_y
        story_h = template.story_height
        bay_x = template.bay_width_x
        bay_y = template.bay_width_y
        
        # Create grid coordinates
        x_coords = [i * bay_x for i in range(n_bays_x + 1)]
        y_coords = [i * bay_y for i in range(n_bays_y + 1)]
        z_coords = [0.0] + [story_h * (i + 1) for i in range(n_stories)]
        
        # Adjust for base isolation
        if template.base_isolation:
            z_coords = [0.0, 0.5] + [0.5 + story_h * (i + 1) for i in range(n_stories)]
        
        # Node map: (floor_idx, x_idx, y_idx) -> Node
        node_map = {}
        
        # Calculate mass per node based on floor area
        floor_area = (n_bays_x * bay_x) * (n_bays_y * bay_y)
        nodes_per_floor = (n_bays_x + 1) * (n_bays_y + 1)
        mass_per_node = (floor_area * 1000.0) / nodes_per_floor  # ~1t/m²
        
        # 1. Create Nodes
        for f_idx, z in enumerate(z_coords):
            for x_idx, x in enumerate(x_coords):
                for y_idx, y in enumerate(y_coords):
                    mass = 0.0 if f_idx == 0 else mass_per_node
                    
                    n = Node(node_id, x, y, z, mass=mass)
                    nodes.append(n)
                    node_map[(f_idx, x_idx, y_idx)] = n
                    node_id += 1
        
        # 2. Assign DOFs
        dof_counter = 0
        for n in nodes:
            if n.z == 0.0:
                n.set_dof_indices([-1] * 6)
            else:
                indices = list(range(dof_counter, dof_counter + 6))
                n.set_dof_indices(indices)
                dof_counter += 6
        
        # 3. Material properties (scaled for building size)
        E = 2.05e11  # Steel
        G = 7.9e10
        
        # Scale section properties based on building height
        height = n_stories * story_h
        if height < 15:
            A = 0.03; Iy = 2e-4; Iz = 2e-4; J = 3e-4
            My = 300000.0
        elif height < 40:
            A = 0.05; Iy = 5e-4; Iz = 5e-4; J = 7e-4
            My = 500000.0
        elif height < 80:
            A = 0.08; Iy = 1e-3; Iz = 1e-3; J = 1.5e-3
            My = 800000.0
        else:
            A = 0.12; Iy = 2e-3; Iz = 2e-3; J = 3e-3
            My = 1200000.0
        
        # 4. Create Columns (vertical elements)
        start_level = 1 if template.base_isolation else 0
        
        for f_idx in range(start_level, len(z_coords) - 1):
            for x_idx in range(len(x_coords)):
                for y_idx in range(len(y_coords)):
                    n_bot = node_map.get((f_idx, x_idx, y_idx))
                    n_top = node_map.get((f_idx + 1, x_idx, y_idx))
                    
                    if n_bot and n_top:
                        col = BeamColumn3D(elem_id, n_bot, n_top, E, G, A, Iy, Iz, J)
                        col.set_yield_properties(My, My)
                        elements.append(col)
                        elem_id += 1
        
        # 5. Create Beams (horizontal elements)
        for f_idx in range(1, len(z_coords)):
            # X-direction beams
            for y_idx in range(len(y_coords)):
                for x_idx in range(len(x_coords) - 1):
                    n1 = node_map.get((f_idx, x_idx, y_idx))
                    n2 = node_map.get((f_idx, x_idx + 1, y_idx))
                    
                    if n1 and n2:
                        bm = BeamColumn3D(elem_id, n1, n2, E, G, A * 0.8, Iy * 0.7, Iz * 0.7, J * 0.7)
                        bm.set_yield_properties(My * 0.7, My * 0.7)
                        elements.append(bm)
                        elem_id += 1
            
            # Y-direction beams
            for x_idx in range(len(x_coords)):
                for y_idx in range(len(y_coords) - 1):
                    n1 = node_map.get((f_idx, x_idx, y_idx))
                    n2 = node_map.get((f_idx, x_idx, y_idx + 1))
                    
                    if n1 and n2:
                        bm = BeamColumn3D(elem_id, n1, n2, E, G, A * 0.8, Iy * 0.7, Iz * 0.7, J * 0.7)
                        bm.set_yield_properties(My * 0.7, My * 0.7)
                        elements.append(bm)
                        elem_id += 1
        
        # 6. Add base isolators if specified
        if template.base_isolation:
            from src.devices import BaseIsolator
            for x_idx in range(len(x_coords)):
                for y_idx in range(len(y_coords)):
                    n_bot = node_map.get((0, x_idx, y_idx))
                    n_top = node_map.get((1, x_idx, y_idx))
                    
                    if n_bot and n_top:
                        iso = BaseIsolator(elem_id, n_bot, n_top, 1e9, 1e6, 50000, 0.1)
                        elements.append(iso)
                        elem_id += 1
        
        # 7. Add dampers if specified
        if template.dampers:
            from src.devices import OilDamper
            # Add diagonal dampers in corner bays
            for f_idx in range(start_level, len(z_coords) - 1):
                # Corner positions
                corners = [(0, 0), (0, n_bays_y), (n_bays_x, 0), (n_bays_x, n_bays_y)]
                for (x_idx, y_idx) in corners:
                    n_bot = node_map.get((f_idx, x_idx, y_idx))
                    # Diagonal to adjacent corner at next level
                    if x_idx < n_bays_x and y_idx < n_bays_y:
                        n_top = node_map.get((f_idx + 1, x_idx + 1, y_idx + 1))
                        if n_bot and n_top:
                            damp = OilDamper(elem_id, n_bot, n_top, 500000.0)
                            elements.append(damp)
                            elem_id += 1
        
        print(f"Built from template: {template.name}")
        print(f"  Nodes: {len(nodes)}, Elements: {len(elements)}")
        print(f"  Grid: {n_bays_x+1}x{n_bays_y+1}x{n_stories+1} = {len(nodes)} nodes")
        
        return nodes, elements
