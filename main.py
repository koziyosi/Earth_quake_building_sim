import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.fem import Node, BeamColumn2D
from src.hysteresis import Takeda, Bilinear
from src.solver import NewmarkBetaSolver
from src.earthquake import generate_synthetic_wave, load_wave_from_file

def run_simulation(duration=10.0, max_acc=500.0, dt=0.005, callback=None, earthquake_file=None):
    """
    Run the 2D simulation.
    Args:
        duration: Seconds
        max_acc: gal
        dt: Time step
        callback: Function(step, total_steps) for progress update
        earthquake_file: Path to file (optional)
    """
    # --- 1. Model Definition ---
    # Simple 3-Story Frame (1 Bay)
    # Units: N, m, kg
    
    # Geometry
    H = 3.5 # Story Height
    W = 6.0 # Bay Width
    
    # Nodes
    # Base
    n1 = Node(1, 0, 0); n2 = Node(2, W, 0)
    # Floor 1
    n3 = Node(3, 0, H, mass=20000); n4 = Node(4, W, H, mass=20000)
    # Floor 2
    n5 = Node(5, 0, 2*H, mass=20000); n6 = Node(6, W, 2*H, mass=20000)
    # Floor 3 (Roof)
    n7 = Node(7, 0, 3*H, mass=15000); n8 = Node(8, W, 3*H, mass=15000)
    
    nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
    
    # Assign DOFs
    # Base nodes fixed (indices -1)
    # Others have 3 DOFs in 2D mode (x, y, theta_z)
    # Node class is 3D (6 DOFs).
    # We map indices 0, 1, 5. Others are -1.
    dof_counter = 0
    for n in nodes:
        if n.y == 0: 
            n.set_dof_indices([-1]*6)
            continue
            
        # [u_x, u_y, u_z, theta_x, theta_y, theta_z]
        # Active: u_x, u_y, theta_z
        idx_list = [-1] * 6
        idx_list[0] = dof_counter
        idx_list[1] = dof_counter + 1
        idx_list[5] = dof_counter + 2
        n.set_dof_indices(idx_list)
        dof_counter += 3
        
    # Materials
    # Concrete Columns
    Ec = 2.5e10 # N/m^2
    Ic = 0.005 # m^4 (approx 50x50cm)
    Ac = 0.25 # m^2
    
    # Steel Beams (or RC Beams)
    Ib = 0.005
    Ab = 0.25
    
    # Hysteresis Properties
    # Yield Moment My
    # assume My approx Z * Fy
    # concrete: My ~ 200 kN.m
    # Material Properties (Realistic)
    My_col = 300000.0 # Nm
    My_beam = 250000.0 # Nm
    
    K0_rot = 6 * Ec * Ic / H # Reference stiffness
    
    # Elements
    elements = []
    
    def make_col(n_bot, n_top):
        # Takeda for RC Columns
        # k_spring = 100x beam stiffness to approximate rigid node
        k_spring = K0_rot * 100
        h_i = Takeda(k_spring, My_col, 0.05) 
        h_j = Takeda(k_spring, My_col, 0.05)
        
        return BeamColumn2D(len(elements), n_bot, n_top, Ec, Ac, Ic, h_i, h_j)
        
    def make_beam(n_left, n_right):
        # Bilinear for Steel/RC Beams
        k_spring = K0_rot * 100
        h_i = Bilinear(k_spring, My_beam, 0.05)
        h_j = Bilinear(k_spring, My_beam, 0.05)
        return BeamColumn2D(len(elements), n_left, n_right, Ec, Ab, Ib, h_i, h_j)

    # Columns
    elements.append(make_col(n1, n3))
    elements.append(make_col(n2, n4))
    elements.append(make_col(n3, n5))
    elements.append(make_col(n4, n6))
    elements.append(make_col(n5, n7))
    elements.append(make_col(n6, n8))
    
    # Beams
    elements.append(make_beam(n3, n4))
    elements.append(make_beam(n5, n6))
    elements.append(make_beam(n7, n8))
    
    # --- 2. Input ---
    if earthquake_file:
        time_vals, acc_vals = load_wave_from_file(earthquake_file)
        # Intepolate
        t = np.arange(0, duration, dt)
        acc = np.interp(t, time_vals, acc_vals)
    else:
        # Parameters passed from args
        t, acc = generate_synthetic_wave(duration, dt, max_acc=max_acc)
    
    # --- 3. Solver ---
    solver = NewmarkBetaSolver(nodes, elements, dt)
    solver.set_rayleigh_damping(omega1=10.0, omega2=50.0, zeta=0.03)
    
    # --- 4. Run ---
    history_u = []
    history_base_shear = []
    
    print("Starting simulation...")
    for k in range(len(t)):
        acc_g = acc[k]
        u, v, a = solver.solve_step(acc_g)
        history_u.append(u.copy())
        
        # Calculate Base Shear
        # Sum of shear forces in bottom columns (Element 0 and 1)
        # Force is in global coordinates? No, element returns global.
        # forces_0 = elements[0].get_current_global_forces() -> Need to implement getter
        # Element update_state returns restoring force vector.
        # We need to store it or recalculate.
        # Solver calculates F_int implicitly.
        
        # Simple output
        if k % 100 == 0:
            # print(f"Step {k}/{len(t)}: Max Disp = {np.max(np.abs(u)):.4f} m")
            if callback:
                callback(k, len(t))

    # --- 5. Visualization ---
    # Animation
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(-2, W+2)
    ax.set_ylim(-1, 3*H+1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    lines = []
    for el in elements:
        ln, = ax.plot([], [], 'o-', lw=2, color='blue')
        lines.append(ln)
        
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    def init():
        for ln in lines:
            ln.set_data([], [])
        time_text.set_text('')
        return lines + [time_text]
        
    def update(frame):
        # frame is index
        u_curr = history_u[frame]
        
        # Scale deformation for visibility
        scale = 20.0 
        
        current_nodes_x = {}
        current_nodes_y = {}
        
        # Map displacements to nodes
        for node in nodes:
            dx, dy = 0, 0
            if node.dof_indices[0] != -1:
                dx = u_curr[node.dof_indices[0]]
            if node.dof_indices[1] != -1:
                dy = u_curr[node.dof_indices[1]]
                
            current_nodes_x[node.id] = node.x + dx * scale
            current_nodes_y[node.id] = node.y + dy * scale
            
        # Update lines
        for i, el in enumerate(elements):
            x = [current_nodes_x[el.node_i.id], current_nodes_x[el.node_j.id]]
            y = [current_nodes_y[el.node_i.id], current_nodes_y[el.node_j.id]]
            lines[i].set_data(x, y)
            
            # Color change based on ductility/damage?
            # We need to access element state.
            # Ideally store history of damage.
            # For now, just blue.
            
        time_text.set_text(f'Time: {t[frame]:.2f} s\nAcc: {acc[frame]:.2f} m/s2')
        return lines + [time_text]

    # Downsample for animation speed
    skip = 10 # Increase skip to reduce file size
    frames = range(0, len(t), skip)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30)
    
    # Return the animation object and fig instead of saving immediately if called from GUI?
    # Or just save.
    # Plot Drift Angle of 1st Story
    # Drift = (u_3x - u_1x) / H
    # u_1x is 0
    # Note: n3.dof_indices[0] is correct for u_x
    drift_1 = [ u[n3.dof_indices[0]] / H for u in history_u ]
    
    try:
        # Lower resolution
        ani.save('simulation_result.gif', writer='pillow', fps=20, dpi=50)
        print("Animation saved to simulation_result.gif")
        
        # Save drift plot
        fig2 = plt.figure()
        plt.plot(t, drift_1)
        plt.title("1st Story Drift Angle")
        plt.xlabel("Time (s)")
        plt.ylabel("Drift (rad)")
        plt.grid(True)
        plt.savefig('drift_1st_story.png')
        plt.close(fig2)
    except Exception as e:
        print(f"Animation save failed: {e}")
        
    return 'simulation_result.gif', t, drift_1

if __name__ == "__main__":
    run_simulation()
