import numpy as np
from src.fem_3d import Element3D
from src.fem import Node
from src.hysteresis import Bilinear

class BaseIsolator(Element3D):
    """
    Base Isolation Bearing (e.g., Lead Rubber Bearing).
    Modeled as a short column with very low horizontal stiffness (post-yield) and high vertical stiffness.
    Uses Bilinear hysteresis for shear in both Y and Z directions.
    """
    def __init__(self, id: int, node_i: Node, node_j: Node, 
                 Kv: float, Kh_initial: float, Fy: float, r: float):
        super().__init__(id, node_i, node_j)
        self.Kv = Kv  # Vertical Stiffness
        self.Kh_0 = Kh_initial  # Initial Horizontal Stiffness
        self.Fy = Fy  # Yield Force
        self.r = r  # Post-yield stiffness ratio
        
        # Create Bilinear hysteresis models for Y and Z shear
        self.hysteresis_y = Bilinear(Kh_initial, Fy, r)
        self.hysteresis_z = Bilinear(Kh_initial, Fy, r)
        
        # Committed displacement tracking
        self.shear_disp_y = 0.0
        self.shear_disp_z = 0.0
    
    def get_length(self) -> float:
        """Calculate element length."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Returns 12x12 transformation matrix from global to local coordinates.
        Same logic as BeamColumn3D for consistency.
        """
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if L < 1e-10:
            return np.eye(12)
        
        cx = dx / L
        cy = dy / L
        cz = dz / L
        
        # Define local axes
        if np.abs(cz) < 0.999:
            # Not vertical
            D = np.sqrt(cx**2 + cy**2)
            xz = cy / D
            yz = -cx / D
            zz = 0.0
            xy = -cx * cz / D
            yy = -cy * cz / D
            zy = D
        else:
            # Vertical element
            if cz > 0:
                xy = -1; yy = 0; zy = 0
                xz = 0; yz = 1; zz = 0
            else:
                xy = 1; yy = 0; zy = 0
                xz = 0; yz = 1; zz = 0
        
        T_node = np.array([
            [cx, cy, cz],
            [xy, yy, zy],
            [xz, yz, zz]
        ])
        
        # Assemble 12x12 transformation matrix
        T = np.zeros((12, 12))
        T[0:3, 0:3] = T_node
        T[3:6, 3:6] = T_node
        T[6:9, 6:9] = T_node
        T[9:12, 9:12] = T_node
        
        return T
        
    @property
    def damage_index(self) -> float:
        """Calculate damage index based on ductility."""
        dy = self.Fy / self.Kh_0  # Yield displacement
        max_disp = max(abs(self.shear_disp_y), abs(self.shear_disp_z))
        return max_disp / dy if dy > 0 else 0.0
        
    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Returns the element stiffness matrix in global coordinates.
        Uses current tangent stiffness from hysteresis models.
        """
        k = np.zeros((12, 12))
        
        # Axial (Vertical) - Linear elastic
        k[0, 0] = self.Kv
        k[0, 6] = -self.Kv
        k[6, 0] = -self.Kv
        k[6, 6] = self.Kv
        
        # Shear Y - Use tangent from hysteresis
        ky = self.hysteresis_y.tangent
        k[1, 1] = ky
        k[1, 7] = -ky
        k[7, 1] = -ky
        k[7, 7] = ky
        
        # Shear Z - Use tangent from hysteresis
        kz = self.hysteresis_z.tangent
        k[2, 2] = kz
        k[2, 8] = -kz
        k[8, 2] = -kz
        k[8, 8] = kz
        
        # Rotational stiffness (high to emulate fixed connection)
        k_rot = self.Kv * 1.0
        k[3, 3] = k_rot
        k[9, 9] = k_rot
        k[4, 4] = k_rot
        k[10, 10] = k_rot
        k[5, 5] = k_rot
        k[11, 11] = k_rot
        
        # Transform to global coordinates
        T = self.get_transformation_matrix()
        return T.T @ k @ T

    def get_element_dof_indices(self):
        """Returns the global DOF indices for this element."""
        return self.node_i.dof_indices + self.node_j.dof_indices

    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        """
        Updates internal trial state based on displacement increment.
        Returns the restoring force vector in global coordinates.
        """
        T = self.get_transformation_matrix()
        
        # Extract element nodal displacements
        indices = self.get_element_dof_indices()
        u_ele_g = np.zeros(12)
        for k, idx in enumerate(indices):
            if idx != -1 and idx < len(delta_u_global):
                u_ele_g[k] = delta_u_global[idx]
                
        du_local = T @ u_ele_g
        
        # Shear displacements (relative between nodes)
        shear_disp_y_new = self.shear_disp_y + (du_local[7] - du_local[1])
        shear_disp_z_new = self.shear_disp_z + (du_local[8] - du_local[2])
        
        # Update hysteresis models with trial displacement
        force_y, _ = self.hysteresis_y.set_trial_displacement(shear_disp_y_new)
        force_z, _ = self.hysteresis_z.set_trial_displacement(shear_disp_z_new)
        
        # Store trial displacements
        self.trial_shear_disp_y = shear_disp_y_new
        self.trial_shear_disp_z = shear_disp_z_new
        
        # Construct local force vector
        f = np.zeros(12)
        
        # Axial force (linear elastic)
        axial_disp = du_local[6] - du_local[0]
        f_axial = self.Kv * axial_disp
        f[0] = -f_axial
        f[6] = f_axial
        
        # Shear Y
        f[1] = -force_y
        f[7] = force_y
        
        # Shear Z
        f[2] = -force_z
        f[8] = force_z
        
        # Moments from rotation (linear elastic)
        k_rot = self.Kv * 1.0
        f[3] = k_rot * du_local[3]
        f[9] = k_rot * du_local[9]
        f[4] = k_rot * du_local[4]
        f[10] = k_rot * du_local[10]
        f[5] = k_rot * du_local[5]
        f[11] = k_rot * du_local[11]
        
        return T.T @ f
        
    def commit_state(self):
        """Commits the current trial state to history."""
        self.hysteresis_y.commit()
        self.hysteresis_z.commit()
        
        if hasattr(self, 'trial_shear_disp_y'):
            self.shear_disp_y = self.trial_shear_disp_y
        if hasattr(self, 'trial_shear_disp_z'):
            self.shear_disp_z = self.trial_shear_disp_z

class OilDamper(Element3D):
    """
    Viscous Damper.
    Force depends on velocity, not displacement.
    """
    def __init__(self, id: int, node_i: Node, node_j: Node, C: float):
        super().__init__(id, node_i, node_j)
        self.C_val = C  # Damping Coefficient (N s/m)
    
    def get_length(self) -> float:
        """Calculate element length."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Returns 12x12 transformation matrix from global to local coordinates.
        """
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if L < 1e-10:
            return np.eye(12)
        
        cx = dx / L
        cy = dy / L
        cz = dz / L
        
        # Define local axes
        if np.abs(cz) < 0.999:
            D = np.sqrt(cx**2 + cy**2)
            xz = cy / D
            yz = -cx / D
            zz = 0.0
            xy = -cx * cz / D
            yy = -cy * cz / D
            zy = D
        else:
            if cz > 0:
                xy = -1; yy = 0; zy = 0
                xz = 0; yz = 1; zz = 0
            else:
                xy = 1; yy = 0; zy = 0
                xz = 0; yz = 1; zz = 0
        
        T_node = np.array([
            [cx, cy, cz],
            [xy, yy, zy],
            [xz, yz, zz]
        ])
        
        T = np.zeros((12, 12))
        T[0:3, 0:3] = T_node
        T[3:6, 3:6] = T_node
        T[6:9, 6:9] = T_node
        T[9:12, 9:12] = T_node
        
        return T
        
    @property
    def damage_index(self) -> float:
        """Dampers don't have damage in the traditional sense."""
        return 0.0
        
    def get_element_dof_indices(self):
        """Returns the global DOF indices for this element."""
        return self.node_i.dof_indices + self.node_j.dof_indices
        
    def get_stiffness_matrix(self) -> np.ndarray:
        """Stiffness is 0 for viscous damper."""
        return np.zeros((12, 12))
    
    def get_damping_matrix(self) -> np.ndarray:
        """
        Returns 12x12 damping matrix.
        Axial damping only (damper acts along its axis).
        """
        c = np.zeros((12, 12))
        val = self.C_val
        c[0, 0] = val
        c[0, 6] = -val
        c[6, 0] = -val
        c[6, 6] = val
        
        T = self.get_transformation_matrix()
        return T.T @ c @ T
        
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        """No restoring force from displacement for viscous damper."""
        return np.zeros(12)
        
    def commit_state(self):
        """Nothing to commit for viscous damper."""
        pass

