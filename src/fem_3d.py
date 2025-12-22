import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from src.fem import Node

class Element3D(ABC):
    """
    Abstract base class for 3D elements.
    """
    def __init__(self, id: int, node_i: Node, node_j: Node):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        
    @abstractmethod
    def commit_state(self):
        """
        Commits the current trial state to history.
        """
        pass

class BeamColumn3D(Element3D):
    """
    3D Beam-Column Element with Plastic Hinges (Takeda/Bilinear) at ends.
    Concentrated Plasticity Model.
    """
    # Element type constants
    TYPE_COLUMN = 'column'
    TYPE_BEAM = 'beam'
    TYPE_WALL = 'wall'
    TYPE_BRACE = 'brace'
    
    def __init__(self, id: int, node_i: Node, node_j: Node, 
                 E: float, G: float, A: float, Iy: float, Iz: float, J: float):
        super().__init__(id, node_i, node_j)
        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        
        self.forces_local = np.zeros(12) 
        
        # Hysteresis Models (Springs)
        # 4 Springs: My_i, My_j, Mz_i, Mz_j
        self.spring_my_i = None
        self.spring_my_j = None
        self.spring_mz_i = None
        self.spring_mz_j = None
        
        # Spring State tracking (Rotations)
        self.rot_my_i = 0.0; self.rot_my_j = 0.0
        self.rot_mz_i = 0.0; self.rot_mz_j = 0.0
        
        self.trial_rot_my_i = 0.0; self.trial_rot_my_j = 0.0
        self.trial_rot_mz_i = 0.0; self.trial_rot_mz_j = 0.0
        
        # Axial Force Tracking
        self.current_N = 0.0
        self.trial_N = 0.0
        
        # Element type (auto-detected or manually set)
        self._element_type = None
        
        # Response history tracking for individual element analysis
        self.track_history = False  # Enable to track full history
        self.history = {
            'axial_force': [],      # Axial force N
            'shear_y': [],          # Shear force in Y
            'shear_z': [],          # Shear force in Z
            'moment_y_i': [],       # Moment at node i (Y-axis)
            'moment_y_j': [],       # Moment at node j (Y-axis)
            'moment_z_i': [],       # Moment at node i (Z-axis)
            'moment_z_j': [],       # Moment at node j (Z-axis)
            'rotation_y_i': [],     # Plastic rotation at i (Y-axis)
            'rotation_y_j': [],     # Plastic rotation at j (Y-axis)
            'rotation_z_i': [],     # Plastic rotation at i (Z-axis)
            'rotation_z_j': [],     # Plastic rotation at j (Z-axis)
            'damage_index': [],     # Damage progression
            'drift_ratio': [],      # Story drift ratio (for columns)
        }
        
        # Peak values for quick access
        self.peak_axial_force = 0.0
        self.peak_shear = 0.0
        self.peak_moment = 0.0
        self.peak_rotation = 0.0
        self.max_damage_index = 0.0
    
    @property
    def element_type(self) -> str:
        """Get element type (column/beam/wall/brace), auto-detecting if not set."""
        if self._element_type:
            return self._element_type
        
        # Auto-detect based on geometry
        dz = abs(self.node_j.z - self.node_i.z)
        dx = abs(self.node_j.x - self.node_i.x)
        dy = abs(self.node_j.y - self.node_i.y)
        horizontal = max(dx, dy)
        
        if dz > 0.8 * self.get_length():
            return self.TYPE_COLUMN  # Primarily vertical
        elif horizontal > 0.9 * self.get_length():
            return self.TYPE_BEAM    # Primarily horizontal
        else:
            return self.TYPE_BRACE   # Diagonal
    
    @element_type.setter
    def element_type(self, value: str):
        """Manually set element type."""
        self._element_type = value

    def get_length(self) -> float:
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def get_transformation_matrix(self) -> np.ndarray:
        # 3D Beam Transformation
        # Local x' axis is along element axis
        # How to define y' and z'?
        # Standard convention: 
        # If vertical (along Z), y' is -X?
        # A common approach: use a reference vector.
        # Let's assume standard implementation.
        
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        cx = dx / L
        cy = dy / L
        cz = dz / L
        
        # Rotation Matrix R (3x3)
        # R = [nx ny nz] rows? Or columns?
        # Local to Global: u_g = R.T @ u_l ?
        # Usually u_l = T @ u_g. (T is 12x12).
        # T_3x3 maps Global to Local.
        
        # Algorithm to define local axes:
        # x' = (cx, cy, cz)
        # Need arbitrary perpendicular y' and z'.
        # If element is not vertical:
        # Let 'k' be global Z (0,0,1).
        # z' = x' cross k (horizontal vector)
        # y' = z' cross x' (vector in vertical plane)
        
        if np.abs(cz) < 0.999:
            # Not vertical
            # z' direction (perp to x' and Z)
            # z' = x' x K = (cy, -cx, 0)
            D = np.sqrt(cx**2 + cy**2)
            xz = cy / D
            yz = -cx / D
            zz = 0.0
            
            # y' = z' x x'
            # y'x = yz*cz - zz*cy = -cx*cz/D
            # y'y = zz*cx - xz*cz = -cy*cz/D
            # y'z = xz*cy - yz*cx = (cy^2 + cx^2)/D = D
            xy = -cx * cz / D
            yy = -cy * cz / D
            zy = D
            
        else:
            # Vertical element (Parallel to Z-axis)
            # x' = element axis direction
            # 
            # RIGHT-HAND RULE: x' × y' = z'
            # 
            # For upward column (cz > 0): x' = (0, 0, 1)
            #   Choose y' = Global X = (1, 0, 0)
            #   Then z' = x' × y' = (0,0,1) × (1,0,0) = (0*0-1*0, 1*1-0*0, 0*0-0*1) = (0, 1, 0) = Global Y ✓
            #
            # For downward column (cz < 0): x' = (0, 0, -1)
            #   Choose y' = Global X = (1, 0, 0)  
            #   Then z' = x' × y' = (0,0,-1) × (1,0,0) = (0*0-(-1)*0, (-1)*1-0*0, 0*0-0*1) = (0, -1, 0) = Global -Y
            
            if cz > 0:
                # x' = (0, 0, 1), y' = (1, 0, 0), z' = (0, 1, 0)
                xy = 1; yy = 0; zy = 0   # y' = Global X
                xz = 0; yz = 1; zz = 0   # z' = Global Y
            else:
                # x' = (0, 0, -1), y' = (1, 0, 0), z' = (0, -1, 0)
                xy = 1; yy = 0; zy = 0   # y' = Global X
                xz = 0; yz = -1; zz = 0  # z' = Global -Y
                
        # T_node = [cx cy cz]
        #          [xy yy zy]
        #          [xz yz zz]
        
        T_node = np.array([
            [cx, cy, cz],
            [xy, yy, zy],
            [xz, yz, zz]
        ])
        
        # Assemble 12x12
        T = np.zeros((12, 12))
        T[0:3, 0:3] = T_node
        T[3:6, 3:6] = T_node
        T[6:9, 6:9] = T_node
        T[9:12, 9:12] = T_node
        
        return T

    def get_element_dof_indices(self) -> List[int]:
        return self.node_i.dof_indices + self.node_j.dof_indices

    def set_yield_properties(self, My_y, My_z):
        from src.hysteresis import Takeda
        
        # Calculate initial stiffness of springs
        # Concentrated plasticity: K_spring should be high but not extreme
        # Too high (100x): Near-rigid, small rotations cause instability
        # Moderate (10x): Allows realistic plastic rotation development
        L = self.get_length()
        
        k_beam_y = 6 * self.E * self.Iy / L
        k_beam_z = 6 * self.E * self.Iz / L
        
        # Use 10x multiplier for better numerical stability
        k_sp_y = k_beam_y * 10
        k_sp_z = k_beam_z * 10
        
        # Create Takeda Models
        # r = 0.10 (post yield ratio) - slightly higher for stability
        self.spring_my_i = Takeda(k_sp_y, My_y, 0.10)
        self.spring_my_j = Takeda(k_sp_y, My_y, 0.10)
        
        self.spring_mz_i = Takeda(k_sp_z, My_z, 0.10)
        self.spring_mz_j = Takeda(k_sp_z, My_z, 0.10)

    def get_elastic_stiffness_matrix_local(self):
        # We need the "Secant" or "Tangent" matrix including springs?
        # get_stiffness_matrix usually returns Tangent Stiffness for Newton-Raphson.
        return self.get_tangent_stiffness_matrix_local()

    def get_tangent_stiffness_matrix_local(self):
        L = self.get_length()
        
        # 1. Axial (Assume Linear for now, or add hysteretic axial?)
        k_axial = self.E * self.A / L
        
        # 2. Torsion (Linear)
        k_torsion = self.G * self.J / L
        
        # 3. Bending Y (Weak Axis)
        # Series system: Spring_i -- Beam_y -- Spring_j
        # f_const = L / (6EIy)
        f_beam_y = (L / (6 * self.E * self.Iy)) * np.array([[2.0, -1.0], [-1.0, 2.0]])
        
        ks_i = self.spring_my_i.tangent if self.spring_my_i else 1e15
        ks_j = self.spring_my_j.tangent if self.spring_my_j else 1e15
        
        f_tot_y = f_beam_y + np.diag([1.0/ks_i, 1.0/ks_j])
        k_cond_y = np.linalg.inv(f_tot_y)
        
        # 4. Bending Z (Strong Axis)
        f_beam_z = (L / (6 * self.E * self.Iz)) * np.array([[2.0, -1.0], [-1.0, 2.0]])
        
        ks_zi = self.spring_mz_i.tangent if self.spring_mz_i else 1e15
        ks_zj = self.spring_mz_j.tangent if self.spring_mz_j else 1e15
        
        f_tot_z = f_beam_z + np.diag([1.0/ks_zi, 1.0/ks_zj])
        k_cond_z = np.linalg.inv(f_tot_z)
        
        # Assemble 12x12
        k = np.zeros((12, 12))
        
        # Axial (0, 6)
        k[0, 0] = k_axial; k[0, 6] = -k_axial
        k[6, 0] = -k_axial; k[6, 6] = k_axial
        
        # Torsion (3, 9)
        k[3, 3] = k_torsion; k[3, 9] = -k_torsion
        k[9, 3] = -k_torsion; k[9, 9] = k_torsion
        
        # Bending Z -> Forces in Y (1,7), Moments in Z (5,11)
        # k_cond_z relates [Mzi, Mzj, ...]
        # Map k_cond_z terms to 12x12
        # Standard relation:
        # V = -(Mi + Mj)/L
        # Rows: 1(Vy_i), 5(Mz_i), 7(Vy_j), 11(Mz_j)
        
        k11 = k_cond_z[0,0]; k12 = k_cond_z[0,1]
        k21 = k_cond_z[1,0]; k22 = k_cond_z[1,1]
        
        r1 = (k11 + k21)/L; r2 = (k12 + k22)/L; r3 = (k11+k12+k21+k22)/L**2
        
        # Row 1 (Vy_i)
        k[1,1] = r3;  k[1,5] = r1;  k[1,7] = -r3; k[1,11] = r2
        # Row 5 (Mz_i)
        k[5,1] = r1;  k[5,5] = k11; k[5,7] = -r1; k[5,11] = k12
        # Row 7 (Vy_j)
        k[7,1] = -r3; k[7,5] = -r1; k[7,7] = r3;  k[7,11] = -r2
        # Row 11 (Mz_j)
        k[11,1] = r2; k[11,5] = k21; k[11,7] = -r2; k[11,11] = k22
        
        # Bending Y -> Forces in Z (2,8), Moments in Y (4,10)
        # Sign convention: Must be consistent with Z-bending (rows 1,5,7,11)
        # 
        # For Z-bending: Shear-Rotation coupling is POSITIVE (k[1,5] = r1)
        # For Y-bending: Must follow the SAME sign pattern
        #
        # Standard 3D beam element convention:
        # - Positive Fz_i with positive theta_y_i relationship
        # - The coupling follows: V = (M_i + M_j) / L (with appropriate signs)
        
        ky11 = k_cond_y[0,0]; ky12 = k_cond_y[0,1]
        ky21 = k_cond_y[1,0]; ky22 = k_cond_y[1,1]
        
        ry1 = (ky11 + ky21)/L; ry2 = (ky12 + ky22)/L; ry3 = (ky11+ky12+ky21+ky22)/L**2
        
        # Row 2 (Fz_i) - FIXED: Same sign pattern as Z-bending
        k[2,2] = ry3;   k[2,4] = ry1;    k[2,8] = -ry3;  k[2,10] = ry2 
        # Row 4 (My_i) - FIXED: Symmetric with row 2
        k[4,2] = ry1;   k[4,4] = ky11;   k[4,8] = -ry1;  k[4,10] = ky12
        # Row 8 (Fz_j)
        k[8,2] = -ry3;  k[8,4] = -ry1;   k[8,8] = ry3;   k[8,10] = -ry2
        # Row 10 (My_j)
        k[10,2] = ry2;  k[10,4] = ky21;  k[10,8] = -ry2; k[10,10] = ky22
        
        return k

    def get_stiffness_matrix(self) -> np.ndarray:
        T = self.get_transformation_matrix()
        k_local = self.get_tangent_stiffness_matrix_local()
        return T.T @ k_local @ T
    
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        T = self.get_transformation_matrix()
        
        # Extract element nodal displacements (global coords)
        indices = self.get_element_dof_indices()
        u_ele_g = np.zeros(12)
        for k, idx in enumerate(indices):
            if idx != -1:
                # Check bounds just in case? Solver should guarantee n_dof match
                if idx < len(delta_u_global):
                    u_ele_g[k] = delta_u_global[idx]
                
        du_local = T @ u_ele_g
        L = self.get_length()
        
        # 1. Update Axial with TENSION CUTOFF for concrete
        # Concrete has very low tensile strength (typically ~10% of compressive)
        # Without cutoff, columns act like infinite rubber bands
        du_x = du_local[6] - du_local[0]
        k_axial = self.E * self.A / L
        dN = k_axial * du_x
        trial_N = self.current_N + dN
        
        # Tension cutoff: Limit tensile force to ~10% of yield capacity
        # This prevents unrealistic "pull-back" forces when building tries to lift
        max_tension = 0.1 * self.E * self.A * 0.002  # ~0.2% strain limit for tension
        if trial_N > max_tension:
            # In tension beyond capacity - force is limited
            self.trial_N = max_tension
            # Reduce axial stiffness for tensioned elements
            self._axial_softened = True
        else:
            self.trial_N = trial_N
            self._axial_softened = False
        
        # 2. Update Torsion (Elastic)
        d_theta_x = du_local[9] - du_local[3]
        dT = (self.G * self.J / L) * d_theta_x
        # Store T? For now just return it.
        Torque = dT # Assuming 0 initial? Or we should track it.
        # Let's assume T is elastic and not tracked in history (reset every step? No, must track or use total).
        # We need total deformation from somewhere if we don't track force.
        # But this method receives incremental delta.
        # Better to track `self.current_T`.
        if not hasattr(self, 'current_T'): self.current_T = 0.0
        if not hasattr(self, 'trial_T'): self.trial_T = 0.0
        self.trial_T = self.current_T + dT
        T_trial = self.trial_T
        
        # 3. Update Bending Z (Strong Axis) -> Disps y (1, 7), Rot z (5, 11)
        chord_rot_z = (du_local[7] - du_local[1]) / L
        dt_zi = du_local[5] - chord_rot_z
        dt_zj = du_local[11] - chord_rot_z
        
        # Condensation for Z
        ks_zi = self.spring_mz_i.tangent if self.spring_mz_i else 1e15
        ks_zj = self.spring_mz_j.tangent if self.spring_mz_j else 1e15
        f_beam_z = (L / (6 * self.E * self.Iz)) * np.array([[2.0, -1.0], [-1.0, 2.0]])
        f_tot_z = f_beam_z + np.diag([1.0/ks_zi, 1.0/ks_zj])
        k_ser_z = np.linalg.inv(f_tot_z)
        
        dM_z = k_ser_z @ np.array([dt_zi, dt_zj])
        
        dsp_zi = dM_z[0] / ks_zi
        dsp_zj = dM_z[1] / ks_zj
        
        mz_i, _ = self.spring_mz_i.set_trial_displacement(self.rot_mz_i + dsp_zi) if self.spring_mz_i else (0,0)
        mz_j, _ = self.spring_mz_j.set_trial_displacement(self.rot_mz_j + dsp_zj) if self.spring_mz_j else (0,0)
        
        self.trial_rot_mz_i = self.rot_mz_i + dsp_zi
        self.trial_rot_mz_j = self.rot_mz_j + dsp_zj
        
        # 4. Update Bending Y (Weak Axis) -> Disps z (2, 8), Rot y (4, 10)
        # Note signs.
        # Chord rot y.
        # d_z / L corresponds to -Rotation about Y?
        # R_y = -(u_z2 - u_z1)/L
        chord_rot_y = -(du_local[8] - du_local[2]) / L 
        # But wait, earlier I used positive chord rot subtraction?
        # Let's stick to the "Total Nodal Rotation - Chord" definition.
        # Rot Y is Rot Y.
        # If we defined Matrix such that (Fz, My) has negative cross terms,
        # It implies Positive My opposes Positive Fz-induced slope.
        # Slope = du_z/dx.
        # Let's assume standard definitions.
        
        dt_yi = du_local[4] - chord_rot_y
        dt_yj = du_local[10] - chord_rot_y
        
        # Condensation Y
        ks_yi = self.spring_my_i.tangent if self.spring_my_i else 1e15
        ks_yj = self.spring_my_j.tangent if self.spring_my_j else 1e15
        f_beam_y = (L / (6 * self.E * self.Iy)) * np.array([[2.0, -1.0], [-1.0, 2.0]])
        f_tot_y = f_beam_y + np.diag([1.0/ks_yi, 1.0/ks_yj])
        k_ser_y = np.linalg.inv(f_tot_y)
        
        dM_y = k_ser_y @ np.array([dt_yi, dt_yj])
        
        dsp_yi = dM_y[0] / ks_yi
        dsp_yj = dM_y[1] / ks_yj
        
        my_i, _ = self.spring_my_i.set_trial_displacement(self.rot_my_i + dsp_yi) if self.spring_my_i else (0,0)
        my_j, _ = self.spring_my_j.set_trial_displacement(self.rot_my_j + dsp_yj) if self.spring_my_j else (0,0)
        
        self.trial_rot_my_i = self.rot_my_i + dsp_yi
        self.trial_rot_my_j = self.rot_my_j + dsp_yj
        
        # 5. Assemble Global Force Vector
        # Reconstruct Shears from moment equilibrium
        # Sign convention must match stiffness matrix
        # For Z-bending: Vy = -(Mz_i + Mz_j) / L (moments create shear)
        Vy = -(mz_i + mz_j) / L
        # For Y-bending: Vz follows same pattern as Vy
        # With corrected stiffness signs, use consistent shear calculation
        Vz = (my_i + my_j) / L  # Positive coupling consistent with k[2,4]=+ry1
        
        # Local Forces 12-vector
        f_loc = np.zeros(12)
        f_loc[0] = -self.trial_N
        f_loc[6] = self.trial_N
        
        f_loc[3] = -T_trial # Torsion equilibrium
        f_loc[9] = T_trial
        
        # Y-Shear / Z-Moment
        f_loc[1] = Vy; f_loc[7] = -Vy
        f_loc[5] = mz_i; f_loc[11] = mz_j
        
        # Z-Shear / Y-Moment  
        f_loc[2] = Vz; f_loc[8] = -Vz
        f_loc[4] = my_i; f_loc[10] = my_j
        
        return T.T @ f_loc
    
    @property
    def damage_index(self) -> float:
        """
        Calculate damage index based on plastic rotation demand.
        
        Damage index = max(rotation / yield_rotation) across all hinges.
        0.0 = No damage (elastic)
        1.0 = Yield point reached
        >1.0 = Post-yield (plastic deformation)
        """
        max_rotation = max(
            abs(self.trial_rot_my_i),
            abs(self.trial_rot_my_j),
            abs(self.trial_rot_mz_i),
            abs(self.trial_rot_mz_j)
        )
        
        # Estimate yield rotation from spring properties
        if self.spring_my_i and hasattr(self.spring_my_i, 'dy'):
            yield_rotation = self.spring_my_i.dy
        else:
            # Estimate: θy ≈ My*L / (6*E*I)
            L = self.get_length()
            yield_rotation = 0.01  # Default 1% as yield rotation
            
        if yield_rotation > 0:
            return max_rotation / yield_rotation
        return 0.0
    
    def get_response_summary(self) -> dict:
        """Get current response state summary for this element."""
        return {
            'id': self.id,
            'type': self.element_type,
            'axial_force': self.current_N,
            'moment_y_i': self.spring_my_i.force if self.spring_my_i else 0,
            'moment_y_j': self.spring_my_j.force if self.spring_my_j else 0,
            'moment_z_i': self.spring_mz_i.force if self.spring_mz_i else 0,
            'moment_z_j': self.spring_mz_j.force if self.spring_mz_j else 0,
            'rotation_y_i': self.rot_my_i,
            'rotation_y_j': self.rot_my_j,
            'rotation_z_i': self.rot_mz_i,
            'rotation_z_j': self.rot_mz_j,
            'damage_index': self.damage_index,
            'peak_damage': self.max_damage_index,
        }

    def commit_state(self):
        if self.spring_my_i: self.spring_my_i.commit()
        if self.spring_my_j: self.spring_my_j.commit()
        if self.spring_mz_i: self.spring_mz_i.commit()
        if self.spring_mz_j: self.spring_mz_j.commit()
        
        self.rot_my_i = self.trial_rot_my_i
        self.rot_my_j = self.trial_rot_my_j
        self.rot_mz_i = self.trial_rot_mz_i
        self.rot_mz_j = self.trial_rot_mz_j
        
        self.current_N = self.trial_N
        
        # Safely handle torsional state (no trial logic implemented)
        if hasattr(self, 'trial_T'):
            self.current_T = self.trial_T
        elif not hasattr(self, 'current_T'):
            self.current_T = 0.0
        
        # Update peak values
        self.peak_axial_force = max(self.peak_axial_force, abs(self.current_N))
        current_moment = max(
            abs(self.spring_my_i.force) if self.spring_my_i else 0,
            abs(self.spring_my_j.force) if self.spring_my_j else 0,
            abs(self.spring_mz_i.force) if self.spring_mz_i else 0,
            abs(self.spring_mz_j.force) if self.spring_mz_j else 0,
        )
        self.peak_moment = max(self.peak_moment, current_moment)
        current_rotation = max(
            abs(self.rot_my_i), abs(self.rot_my_j),
            abs(self.rot_mz_i), abs(self.rot_mz_j)
        )
        self.peak_rotation = max(self.peak_rotation, current_rotation)
        self.max_damage_index = max(self.max_damage_index, self.damage_index)
        
        # Record history if tracking enabled
        if self.track_history:
            self.history['axial_force'].append(self.current_N)
            self.history['moment_y_i'].append(self.spring_my_i.force if self.spring_my_i else 0)
            self.history['moment_y_j'].append(self.spring_my_j.force if self.spring_my_j else 0)
            self.history['moment_z_i'].append(self.spring_mz_i.force if self.spring_mz_i else 0)
            self.history['moment_z_j'].append(self.spring_mz_j.force if self.spring_mz_j else 0)
            self.history['rotation_y_i'].append(self.rot_my_i)
            self.history['rotation_y_j'].append(self.rot_my_j)
            self.history['rotation_z_i'].append(self.rot_mz_i)
            self.history['rotation_z_j'].append(self.rot_mz_j)
            self.history['damage_index'].append(self.damage_index)
            
            # Drift ratio for columns
            if self.element_type == self.TYPE_COLUMN:
                L = self.get_length()
                # Estimate lateral drift from rotation
                drift = (abs(self.rot_my_i) + abs(self.rot_my_j)) / 2 if L > 0 else 0
                self.history['drift_ratio'].append(drift)

