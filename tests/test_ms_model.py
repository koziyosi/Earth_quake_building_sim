import numpy as np
import matplotlib.pyplot as plt
from src.ms_model import FiberSection, SimpleMaterial

def test_moment_curvature():
    # Define a Concrete Section (Rectangular)
    # Width 0.5m, Depth 0.5m
    # Discretize into fibers
    
    # Material
    Ec = 2.5e10
    Fc = 3.0e7 # 30 MPa
    conc_mat = SimpleMaterial(Ec, Fc)
    
    fibers = []
    # Grid of fibers
    ny, nz = 10, 10
    dy = 0.5 / ny
    dz = 0.5 / nz
    dA = dy * dz
    
    for i in range(ny):
        y = -0.25 + (i + 0.5) * dy
        for j in range(nz):
            z = -0.25 + (j + 0.5) * dz
            fibers.append((y, z, dA, conc_mat))
            
    section = FiberSection(fibers)
    
    # Cyclic Curvature History (Uniaxial Bending about Z)
    # curvature_y = 0
    # curvature_z varies
    
    phi_max = 0.02 # rad/m
    phis = np.linspace(0, phi_max, 50)
    phis = np.concatenate([phis, np.linspace(phi_max, -phi_max, 100)])
    
    moments = []
    curvature_history = []
    
    for phi in phis:
        # Assume N=0 (pure bending) - Solving for strain_centroid such that N=0
        # For linear elastic/symmetric, centroid strain is 0.
        # For simplified test, assume neutral axis at center (eps0 = 0).
        
        forces, k = section.get_response(0.0, 0.0, phi)
        # Forces: [N, My, Mz]
        moments.append(forces[2]) # Mz
        curvature_history.append(phi)
        
    plt.figure()
    plt.plot(curvature_history, moments)
    plt.title("Moment-Curvature (Fiber Section)")
    plt.xlabel("Curvature (rad/m)")
    plt.ylabel("Moment Mz (Nm)")
    plt.grid(True)
    plt.savefig('fiber_section_check.png')
    print("Saved fiber_section_check.png")

if __name__ == "__main__":
    test_moment_curvature()
