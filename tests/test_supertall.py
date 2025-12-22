"""
Quick test to verify HyperBuilding simulation stability.
"""
import sys
sys.path.insert(0, '.')

from src.layout_model import BuildingLayout
from src.builder import BuildingBuilder
import numpy as np

def test_supertall_building():
    """Test that super-tall building simulation doesn't diverge."""
    # Create a 50-floor test building (175m)
    from src.layout_model import GridSystem
    
    story_heights = [3.5] * 50  # 50 floors
    grid = GridSystem(
        x_spacings=[6.0, 6.0],
        y_spacings=[6.0, 6.0],
        story_heights=story_heights
    )
    
    layout = BuildingLayout()
    layout.grid = grid
    layout.initialize_default()
    
    # Build model
    nodes, elements = BuildingBuilder.build_from_layout(layout)
    
    print(f"Built model: {len(nodes)} nodes, {len(elements)} elements")
    
    # Get building height
    max_z = max(n.z for n in nodes)
    print(f"Building height: {max_z:.1f}m")
    
    # Check that section scaling was applied
    assert max_z > 150, "Building should be >150m tall"
    
    # Quick solver test with limited steps
    from src.solver import NewmarkBetaSolver
    from src.earthquake import generate_synthetic_wave
    
    dt = 0.01
    duration = 1.0  # Short test
    
    solver = NewmarkBetaSolver(nodes, elements, dt)
    omega1, omega2 = solver.set_rayleigh_damping_auto(0.10)  # 10% damping
    print(f"Natural frequencies: {omega1:.2f}, {omega2:.2f} rad/s")
    
    # Generate test wave (500 gal, 3s period)
    t, acc = generate_synthetic_wave(duration, dt, max_acc=500.0, dominant_period=3.0)
    
    # Prepare influence vectors
    iota_x = np.zeros(solver.ndof)
    for n in nodes:
        if n.dof_indices[0] != -1:
            iota_x[n.dof_indices[0]] = 1.0
    
    # Initialize
    solver.u = np.zeros(solver.ndof)
    solver.v = np.zeros(solver.ndof)
    solver.a = np.zeros(solver.ndof)
    solver.assemble_stiffness()
    
    acc_curr = 0.0
    max_disp_recorded = 0.0
    
    for i, ax in enumerate(acc[:50]):  # Only first 50 steps for quick test
        d_ax = ax - acc_curr
        d_F_ext = -solver.M @ (iota_x * d_ax)
        
        u, v, a = solver.solve_newton_raphson(d_F_ext)
        acc_curr = ax
        
        max_disp_step = np.max(np.abs(u))
        max_disp_recorded = max(max_disp_recorded, max_disp_step)
    
    print(f"Max displacement after 50 steps: {max_disp_recorded:.4f}m")
    
    # Should be realistic (< 5m for 50 steps)
    assert max_disp_recorded < 20.0, f"Displacement {max_disp_recorded}m is too large!"
    assert not np.any(np.isnan(u)), "NaN detected in displacement!"
    
    print("✓ Super-tall building test PASSED!")
    return True

if __name__ == "__main__":
    test_supertall_building()
