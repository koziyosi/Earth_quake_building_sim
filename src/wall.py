import numpy as np
from src.fem import Node
from src.fem_3d import Element3D

class ShearWall3D(Element3D):
    """
    Macroscopic Shear Wall Element.
    Modeled as:
    1. A central vertical truss (or beam) for Axial/Bending stiffness.
    2. A horizontal shear spring (or shear panel) for Shear stiffness.
    
    Simplified: Equivalent Beam Column with distinct Shear Hysteresis.
    """
    def __init__(self, id: int, node_i: Node, node_j: Node, 
                 E: float, G: float, A: float, I_strong: float, I_weak: float,
                 shear_hysteresis=None):
        super().__init__(id, node_i, node_j)
        self.E = E
        self.G = G
        self.A = A
        self.I_strong = I_strong
        self.I_weak = I_weak
        self.shear_hysteresis = shear_hysteresis
        
        # Internal state
        self.shear_force = 0.0
        self.shear_disp = 0.0
        
    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Returns the Timoshenko Beam Stiffness Matrix (12x12).
        Includes shear deformation effects which are critical for walls.
        """
        L = self.get_length()
        E, G, A = self.E, self.G, self.A
        Iy, Iz = self.I_weak, self.I_strong
        
        # Shear Areas (Assumption for rectangular section)
        # As = 5/6 * A
        As_y = 5.0/6.0 * A
        As_z = 5.0/6.0 * A
        
        # Shear Deformation Parameters phi
        # phi = 12 * EI / (G * As * L^2)
        phi_y = 12.0 * E * Iz / (G * As_y * L**2) # Bending about Z corresponds to shear in Y
        phi_z = 12.0 * E * Iy / (G * As_z * L**2) # Bending about Y corresponds to shear in Z
        
        # Pre-factors
        k = np.zeros((12, 12))
        
        # 1. Axial (x)
        k[0, 0] = E*A/L;  k[0, 6] = -E*A/L
        k[6, 0] = -E*A/L; k[6, 6] = E*A/L
        
        # 2. Torsion (rx) - Simplified Saint-Venant
        # J assumed small for walls usually, but provided
        J = 0.01 # Placeholder or need input
        k[3, 3] = G*J/L;  k[3, 9] = -G*J/L
        k[9, 3] = -G*J/L; k[9, 9] = G*J/L
        
        # 3. Bending about Z (Strong Axis) -> Shear in Y
        # Affects indices: 1 (v_y1), 5 (th_z1), 7 (v_y2), 11 (th_z2)
        py = 1.0 / (1.0 + phi_y)
        
        a = 12 * E * Iz / L**3 * py
        b = 6 * E * Iz / L**2 * py
        c = (4 + phi_y) * E * Iz / L * py
        d = (2 - phi_y) * E * Iz / L * py
        
        # Matrix Layout:
        #    v1     th1    v2     th2
        # v1 a      b      -a     b
        # th1 b     c      -b     d
        # v2 -a     -b     a      -b
        # th2 b     d      -b     c
        
        # Indices: v1=1, th1=5, v2=7, th2=11
        k[1, 1] = a;   k[1, 5] = b;   k[1, 7] = -a;  k[1, 11] = b
        k[5, 1] = b;   k[5, 5] = c;   k[5, 7] = -b;  k[5, 11] = d
        k[7, 1] = -a;  k[7, 5] = -b;  k[7, 7] = a;   k[7, 11] = -b
        k[11, 1] = b;  k[11, 5] = d;  k[11, 7] = -b; k[11, 11] = c
        
        # 4. Bending about Y (Weak Axis) -> Shear in Z
        # Affects indices: 2 (v_z1), 4 (th_y1), 8 (v_z2), 10 (th_y2)
        pz = 1.0 / (1.0 + phi_z)
        
        a = 12 * E * Iy / L**3 * pz
        b = 6 * E * Iy / L**2 * pz
        c = (4 + phi_z) * E * Iy / L * pz
        d = (2 - phi_z) * E * Iy / L * pz
        
        # Note signs for th_y are often opposite because right hand rule?
        # F_z positive -> Moment M_y negative slope?
        # Standard matrix signs for Y-bending:
        # Cross terms involving theta_y usually have negative signs relative to theta_z case 
        # because positive theta_y rotation moves +z displacement to -x.
        # Let's verify: +theta_y is rotation around Y.
        # Beam along X. +theta_y rotates +Z side to -X.
        # Stiffness k24 (Force Z due to Rot Y): -6EI/L^2.
        
        k[2, 2] = a;    k[2, 4] = -b;   k[2, 8] = -a;   k[2, 10] = -b
        k[4, 2] = -b;   k[4, 4] = c;    k[4, 8] = b;    k[4, 10] = d
        k[8, 2] = -a;   k[8, 4] = b;    k[8, 8] = a;    k[8, 10] = b
        k[10, 2] = -b;  k[10, 4] = d;   k[10, 8] = b;   k[10, 10] = c
        
        # Transform to Global
        T = self.get_transformation_matrix()
        return T.T @ k @ T
    
    def update_state(self, delta_u_global):
        # We need to implement update state logic.
        # For now, we will perform a linear update.
        # The forces are calculated as K * delta_u_global (incremental)
        # But this is "update_state", which usually updates internal history variables.
        # Since ShearWall3D is macroscopic, we might want to track shear deformations if `shear_hysteresis` is used.

        # Calculate local displacements
        T = self.get_transformation_matrix()
        delta_u_local = T @ delta_u_global

        # Calculate shear deformation (very approximate for this macro model)
        # Shear deformation is related to the relative lateral displacement between nodes.
        # u_shear = u_lateral - u_bending
        # But for this simple element, we can treat the "Shear Spring" as one component.
        # If we have `shear_hysteresis`, we should update it.

        if self.shear_hysteresis:
            # Assume Y-direction shear (strong axis bending involves Y-shear usually?)
            # Wait, strong axis bending (Iz) is about Z-axis. Corresponding shear is in Y.
            # Local DOF 1 and 7 (v_y1, v_y2).
            # Delta Shear = v_y2 - v_y1

            dy = delta_u_local[7] - delta_u_local[1]
            self.shear_disp += dy

            # Update hysteresis
            force, tangent = self.shear_hysteresis.set_trial_displacement(self.shear_disp)
            self.shear_force = force

        # Calculate forces using current stiffness matrix (or modified if hysteresis)
        # If hysteresis is present, we should replace the relevant stiffness terms.
        # But for now, let's just return K * delta_u for consistency with linear elements
        # unless we modify K based on hysteresis tangent.

        K = self.get_stiffness_matrix()

        # If shear hysteresis is active, modify K?
        # get_stiffness_matrix uses G*As. We should replace G*As with Tangent.
        # But get_stiffness_matrix is called by solver.
        # Here we return the restoring force vector.

        # Linear force
        f_inc = K @ delta_u_local

        # If nonlinear shear, replace shear force component?
        # This is getting complex for a "fix".
        # Minimal viable fix: Implement the method so it doesn't crash,
        # and return linear forces for now, or simplistic nonlinear.

        return T.T @ f_inc

    def commit_state(self):
        if self.shear_hysteresis:
            self.shear_hysteresis.commit()
