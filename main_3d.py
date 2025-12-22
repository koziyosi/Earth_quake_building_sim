import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from src.fem import Node
from src.fem_3d import BeamColumn3D
from src.solver import NewmarkBetaSolver
from src.earthquake import (generate_synthetic_wave, load_wave_from_file,
                             get_scenario_wave, generate_long_period_pulse,
                             EARTHQUAKE_SCENARIOS)
from src.builder import BuildingBuilder
from src.devices import BaseIsolator, OilDamper

def run_3d_simulation(duration=5.0, max_acc=400.0, dt=0.005, callback=None, 
                      builder_params=None, earthquake_file=None, layout=None, 
                      fast_mode=False, earthquake_scenario=None):
    """
    Run 3D Simulation.
    
    Args:
        duration: Simulation duration in seconds
        max_acc: Maximum acceleration in gal
        dt: Time step
        callback: Progress callback function(step, total_steps)
        builder_params: Building configuration dictionary
        earthquake_file: Path to earthquake data file
        layout: BuildingLayout object (optional)
        fast_mode: If True, use larger time step (4x faster)
        earthquake_scenario: Predefined scenario key (e.g., 'near_fault_pulse', 
                            'high_rise_killer', 'fling_step', 'distant_long_period')
    """
    # Fast mode uses larger time step for ~4x speedup
    # However, for strong earthquakes, large dt causes numerical instability
    if fast_mode:
        if max_acc > 800:
            # Strong earthquake: reduce dt for stability during nonlinear response
            dt = 0.01
            print(f"Warning: High acceleration ({max_acc} gal) detected.")
            print("Fast mode adjusted: dt=0.01s (2x speedup) for numerical stability.")
        elif max_acc > 500:
            # Moderate-strong: use intermediate dt
            dt = 0.015
            print(f"Fast mode adjusted for {max_acc} gal: dt=0.015s (3x speedup)")
        else:
            dt = 0.02  # 4x larger than default
            print("Fast mode enabled: dt=0.02s (4x speedup)")
    
    # --- 1. Model Definition (3D Frame) ---
    if layout:
        print("Building from Custom Layout...")
        nodes, elements = BuildingBuilder.build_from_layout(layout)
        # Calculate approximate total DOFs for array init
        dof_counter = 0
        for n in nodes:
            idx = max(n.dof_indices)
            if idx > dof_counter: dof_counter = idx
        dof_counter += 1
    else:
        if builder_params is None:
            builder_params = {}
        
        # Check if a template is selected
        template_key = builder_params.get('template')
        if template_key:
            from src.building_templates import get_template
            template = get_template(template_key)
            if template:
                nodes, elements = BuildingBuilder.build_from_template(template)
            else:
                print(f"Warning: Template '{template_key}' not found, using default")
                nodes, elements = BuildingBuilder.build_model(
                    floors=3, span_x=6.0, span_y=6.0, story_h=3.5,
                    soft_first_story=False, base_isolation=False, add_dampers=False
                )
        else:
            floors = builder_params.get('floors', 3)
            soft_story = builder_params.get('soft_story', False)
            isolation = builder_params.get('isolation', False)
            dampers = builder_params.get('dampers', False)
            
            nodes, elements = BuildingBuilder.build_model(
                floors=floors,
                span_x=6.0, span_y=6.0, story_h=3.5,
                soft_first_story=soft_story,
                base_isolation=isolation,
                add_dampers=dampers
            )
    
    # Assign DOFs manually if builder didn't? Builder does assign DOFs.
    # We just need to count total DOFs for array init
    dof_counter = 0
    for n in nodes:
        idx = max(n.dof_indices)
        if idx > dof_counter: dof_counter = idx
    dof_counter += 1
        
    print(f"Total DOFs: {dof_counter}")
            
    # --- 2. Input ---
    # Extract dominant period from builder_params if available
    dominant_period = None
    if builder_params and 'period' in builder_params:
        period = builder_params.get('period', 0)
        if period and period > 0:
            dominant_period = period
            print(f"Using dominant period: {dominant_period}s ({1/dominant_period:.2f} Hz)")
    
    scenario_info = None
    
    if earthquake_scenario:
        # Use predefined scenario
        print(f"\n=== Earthquake Scenario: {earthquake_scenario} ===")
        if earthquake_scenario in EARTHQUAKE_SCENARIOS:
            scenario = EARTHQUAKE_SCENARIOS[earthquake_scenario]
            print(f"  Name: {scenario['name']}")
            print(f"  Description: {scenario['description']}")
        
        # For resonance scenario, estimate building period first
        building_period = None
        if earthquake_scenario == 'resonance_attack':
            max_z = max(n.z for n in nodes) if nodes else 10.0
            building_period = 0.1 * (max_z / 3.5)  # Rough estimate: T ≈ 0.1N
            print(f"  Estimated building period: {building_period:.2f}s")
        
        t, acc_x, scenario_info = get_scenario_wave(
            earthquake_scenario, duration, dt, 
            building_period=building_period, 
            max_acc=max_acc
        )
        # Y component is 75% of X with phase shift
        acc_y = acc_x * 0.75
        print(f"  Peak acceleration: {np.max(np.abs(acc_x))*100:.1f} gal")
        
    elif earthquake_file:
        time_vals, acc_vals = load_wave_from_file(earthquake_file)
        # Interpolate
        t_new = np.arange(0, duration, dt)
        acc_x = np.interp(t_new, time_vals, acc_vals)
        # Generate Y as 75% of X
        acc_y = acc_x * 0.75
        t = t_new
    else:
        # Standard synthetic wave
        t, acc_x = generate_synthetic_wave(duration, dt, max_acc=max_acc, dominant_period=dominant_period)
        _, acc_y = generate_synthetic_wave(duration, dt, max_acc=max_acc*0.75, dominant_period=dominant_period)
    
    # Input vector for solver needs to be handled.
    # NewmarkBetaSolver currently takes `acc_g` scalar and uses `self.iota`.
    # We need to update Solver to handle multi-component input or combine them.
    # Solver.iota should be a matrix? Or F_ext = -M * (iota_x * ax + iota_y * ay + ...)
    
    # Patch Solver for multi-component?
    # Or just project `iota`?
    # Iota for X: 1 at dof indices 0.
    # Iota for Y: 1 at dof indices 1.
    
    # We'll need to manually compute F_ext in the loop or modify Solver.
    # Let's Modify Solver instance manually (hack) or subclass.
    
    # Prepare influence vectors
    iota_x = np.zeros(dof_counter)
    iota_y = np.zeros(dof_counter)
    
    for n in nodes:
        if n.dof_indices[0] != -1: iota_x[n.dof_indices[0]] = 1.0
        if n.dof_indices[1] != -1: iota_y[n.dof_indices[1]] = 1.0
        
    solver = NewmarkBetaSolver(nodes, elements, dt)
    
    # Adaptive damping based on building height for numerical stability
    max_z = max(n.z for n in nodes) if nodes else 10.0
    
    # Enable P-Delta for tall buildings where geometric effects are significant
    if max_z > 40:
        solver.p_delta_enabled = True
        print(f"P-Delta effects enabled for {max_z:.1f}m building")
    else:
        solver.p_delta_enabled = False
    
    if max_z <= 40:
        damping_ratio = 0.05  # 5% for low-rise
    elif max_z <= 100:
        damping_ratio = 0.08  # 8% for mid-rise
    elif max_z <= 200:
        damping_ratio = 0.12  # 12% for high-rise
    else:
        damping_ratio = 0.15  # 15% for super-tall (numerical stability)
    
    print(f"Building height: {max_z:.1f}m, damping ratio: {damping_ratio*100:.0f}%")
    
    # Use automatic Rayleigh damping based on actual building natural frequencies
    omega1, omega2 = solver.set_rayleigh_damping_auto(damping_ratio)
    print(f"Auto Rayleigh damping: omega1={omega1:.2f} rad/s ({omega1/(2*np.pi):.2f} Hz), omega2={omega2:.2f} rad/s ({omega2/(2*np.pi):.2f} Hz)")
    
    # --- 3. Run ---
    history_u = []
    history_damage = [] # Store damage index for each element at each step
    
    print("Starting 3D simulation...")
    
    # Initialize Solver State
    solver.u = np.zeros(solver.ndof)
    solver.v = np.zeros(solver.ndof)
    solver.a = np.zeros(solver.ndof)
    # Solver internal init (M, K, C are already built in init)
    solver.assemble_stiffness() # Initial setup
    
    acc_x_curr = 0.0
    acc_y_curr = 0.0
    
    for k in range(len(t)):
        ax = acc_x[k]
        ay = acc_y[k]
        
        d_ax = ax - acc_x_curr
        d_ay = ay - acc_y_curr
        
        # d_F_ext
        d_F_ext = -solver.M @ (iota_x * d_ax + iota_y * d_ay)
        
        # Solve Step (Newton-Raphson)
        u_new, v_new, a_new = solver.solve_newton_raphson(d_F_ext)
        
        # Collect Results
        current_step_damage = []
        for el in elements:
             if hasattr(el, 'damage_index'):
                 current_step_damage.append(el.damage_index)
             else:
                 current_step_damage.append(0.0)
             
        acc_x_curr = ax
        acc_y_curr = ay
        
        history_u.append(u_new.copy())
        history_damage.append(current_step_damage)
        
        if k % 100 == 0:
            if callback:
                callback(k, len(t))
        
        # More frequent progress for large models
        if k % 10 == 0 and dof_counter > 1000:
            print(f"  Step {k}/{len(t)} ({k*100//len(t)}%)")

    # --- 4. Visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_zlim(0, 12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Plot structure lines
    lines = []
    for el in elements:
        ln, = ax.plot([], [], [], 'o-', lw=2, color='gray')
        lines.append(ln)
        
    title = ax.set_title("3D Simulation")
    
    def update(frame):
        u = history_u[frame]
        dmg = history_damage[frame]
        scale = 1.0  # 1x scale for realistic GIF
        
        # Update Nodes
        node_pos = {}
        for n in nodes:
            dx = u[n.dof_indices[0]] if n.dof_indices[0] != -1 else 0
            dy = u[n.dof_indices[1]] if n.dof_indices[1] != -1 else 0
            dz = u[n.dof_indices[2]] if n.dof_indices[2] != -1 else 0
            node_pos[n.id] = (n.x + dx*scale, n.y + dy*scale, n.z + dz*scale)
            
        for i, el in enumerate(elements):
            p1 = node_pos[el.node_i.id]
            p2 = node_pos[el.node_j.id]
            lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            lines[i].set_3d_properties([p1[2], p2[2]])
            
            # Special Drawing for Devices
            if isinstance(el, BaseIsolator):
                lines[i].set_color('magenta')
                lines[i].set_linewidth(4)
                continue
            elif isinstance(el, OilDamper):
                lines[i].set_color('cyan')
                lines[i].set_linewidth(3)
                continue
            
            # Normal Elements (Beam/Column)
            # Color Map
            # 0.0 -> Blue (Safe)
            # 0.5 -> Green
            # 1.0 -> Yellow (Yield)
            # >1.0 -> Red (Danger)
            if hasattr(el, 'damage_index'): # Check if element supports damage
                d_idx = dmg[i]
                
                # Check for custom color (Walls)
                if hasattr(el, 'custom_color'):
                    # Blend with damage? Or just use custom?
                    # Let's outline or thickness?
                    # For now, just use custom color if damage is low.
                    if d_idx < 0.5:
                        lines[i].set_color(el.custom_color)
                        if el.custom_color == 'red': # Walls
                             lines[i].set_linewidth(1.5)
                        else:
                             lines[i].set_linewidth(2)
                    elif d_idx < 1.0:
                         lines[i].set_color('green')
                    elif d_idx < 1.5:
                         lines[i].set_color('orange')
                    else:
                         lines[i].set_color('red')
                else:
                    if d_idx < 0.5:
                        lines[i].set_color('blue')
                    elif d_idx < 1.0:
                        lines[i].set_color('green') # Pre-yield
                    elif d_idx < 1.5:
                        lines[i].set_color('orange') # Yielded
                    else:
                        lines[i].set_color('red') # High plastic
            else:
                lines[i].set_color('gray')
            
        return lines

    # Real-time sync: calculate interval so animation duration = simulation duration
    # interval (ms) = duration (s) * 1000 / num_frames
    skip = max(1, int(len(t) / 200))  # Limit to ~200 frames max for GIF
    frames = list(range(0, len(t), skip))
    num_frames = len(frames)
    
    # Real-time: total animation time should equal simulation duration
    real_time_interval = int(duration * 1000 / num_frames)  # ms per frame
    real_time_fps = 1000 / real_time_interval if real_time_interval > 0 else 20
    
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=real_time_interval)
    
    # Calculate axis limits based on actual building dimensions - EQUAL ASPECT RATIO
    if nodes:
        x_vals = [n.x for n in nodes]
        y_vals = [n.y for n in nodes]
        z_vals = [n.z for n in nodes]
        
        # Calculate ranges
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        z_range = max(z_vals) - min(z_vals)
        
        # Use the maximum range for all axes (equal aspect ratio)
        max_range = max(x_range, y_range, z_range) / 2 + 2  # margin
        
        mid_x = (max(x_vals) + min(x_vals)) / 2
        mid_y = (max(y_vals) + min(y_vals)) / 2
        mid_z = max(z_vals) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(0, max(z_vals) * 1.1)
    
    ani.save('simulation_3d.gif', writer='pillow', fps=real_time_fps, dpi=80)
    print(f"Saved simulation_3d.gif (real-time sync: {duration}s, {num_frames} frames, {real_time_fps:.1f} fps)")
    
    # Analyze element responses by type (columns, beams, walls)
    try:
        from src.element_analyzer import analyze_element_responses
        element_analysis = analyze_element_responses(elements, print_summary=True)
    except ImportError:
        element_analysis = None
        print("Note: Element analyzer not available")
    
    # Return full results as dictionary for 3D viewer integration
    return {
        'gif_path': 'simulation_3d.gif',
        'time': t,
        'acceleration_x': acc_x,  # Input motion
        'acceleration_y': acc_y,
        'displacement_history': history_u,
        'damage_history': history_damage,
        'nodes': nodes,
        'elements': elements,
        'duration': duration,
        'dt': dt,
        'element_analysis': element_analysis,
        'earthquake_scenario': scenario_info,  # Scenario details if used
    }

if __name__ == "__main__":
    run_3d_simulation()
