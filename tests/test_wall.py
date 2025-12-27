import numpy as np
from src.fem import Node
from src.wall import ShearWall3D

def test_wall_stiffness():
    # Test a cantilever wall element with Shear Deformation
    # Fixed at bottom, Load at top.
    
    L = 3.0
    E = 2.5e10
    G = 1.0e10 # E/2.5
    
    # Wall Dimensions: 2m wide, 0.2m thick
    t = 0.2
    W = 2.0
    A = t * W
    Iz = t * W**3 / 12.0 # Strong axis
    Iy = W * t**3 / 12.0 # Weak axis
    
    n1 = Node(1, 0, 0, 0)
    n2 = Node(2, 0, L, 0) # Vertical wall
    
    wall = ShearWall3D(1, n1, n2, E, G, A, Iz, Iy) # Note: Init uses I_strong=Iz, I_weak=Iy
    
    K = wall.get_stiffness_matrix()
    
    # Apply Force Fy at top node (index 7 in local/global since aligned)
    # Solve K * u = F
    # Partition K (indices 6-11 are free, 0-5 fixed)
    K_ff = K[6:12, 6:12]
    
    F = np.zeros(6)
    F_apply = 1000.0 # N
    
    # We want to test Strong Axis Bending (Iz).
    # Based on Element3D.get_transformation_matrix for a Vertical Element:
    # Local Z corresponds to Global X. (Bending about Local Z is "In-Plane" if we defined it that way?)
    # Wait, Bending about Strong Axis (Iz) usually resists shear in Local Y.
    # Local Y corresponds to Global Z (for Vertical element, see Element3D logic: z'=X, y'=Z).
    # So to engage Local Y (Strong Axis Shear), we must push in Global Z.

    # Push in Global Z (Index 2 in the 6-DOF node vector)
    F[2] = F_apply
    
    u = np.linalg.solve(K_ff, F)
    
    disp = u[2]
    
    # Theoretical Timoshenko Beam Deflection
    # D = PL^3/3EI + PL/GAs
    # As = 5/6 * A
    As = 5.0/6.0 * A
    
    term_bending = (F_apply * L**3) / (3 * E * Iz)
    term_shear = (F_apply * L) / (G * As)
    
    disp_theoretical = term_bending + term_shear
    
    print(f"Bending Term: {term_bending:.4e}")
    print(f"Shear Term:   {term_shear:.4e}")
    print(f"Total Theoretical: {disp_theoretical:.4e}")
    print(f"FEM Calculated:    {disp:.4e}")
    
    err = abs(disp - disp_theoretical) / disp_theoretical
    print(f"Error: {err:.2%}")
    
    assert err < 0.01

if __name__ == "__main__":
    test_wall_stiffness()
