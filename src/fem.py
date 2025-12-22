import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from src.hysteresis import Hysteresis

class Node:
    """
    Represents a node in the 3D frame.
    DOFs: [u_x, u_y, u_z, theta_x, theta_y, theta_z]
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0.0, mass: float = 0.0):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass
        # Global indices for [ux, uy, uz, tx, ty, tz]
        self.dof_indices = [-1] * 6 
        
    def set_dof_indices(self, indices: List[int]):
        if len(indices) != 6:
            raise ValueError("Node in 3D must have 6 DOF indices")
        self.dof_indices = indices

class Element2D(ABC):
    """
    Abstract base class for 2D elements.
    """
    def __init__(self, id: int, node_i: Node, node_j: Node):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        
    @abstractmethod
    def get_stiffness_matrix(self) -> np.ndarray:
        """Returns the element stiffness matrix in global coordinates."""
        pass
    
    @abstractmethod
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        """
        Updates internal TRIAL state based on global displacement increment.
        Returns the restoring force vector in global coordinates.
        Does NOT commit history.
        """
        pass

    @abstractmethod
    def commit_state(self):
        """
        Commits the current trial state to history.
        """
        pass

    def get_element_dof_indices(self) -> List[int]:
        """
        Returns the global DOF indices associated with this element.
        For 2D, maps 3D Node indices [0, 1, 5] (X, Y, ThetaZ).
        """
        # Indices: u_x, u_y, theta_z
        # Node DOFs: 0, 1, 2, 3, 4, 5
        # We take 0, 1, 5
        idx = []
        # Node i
        idx.append(self.node_i.dof_indices[0])
        idx.append(self.node_i.dof_indices[1])
        idx.append(self.node_i.dof_indices[5])
        # Node j
        idx.append(self.node_j.dof_indices[0])
        idx.append(self.node_j.dof_indices[1])
        idx.append(self.node_j.dof_indices[5])
        return idx
    
    def get_length(self):
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        return np.sqrt(dx**2 + dy**2)
    
    def get_transformation_matrix(self):
        """
        Returns transformation matrix T (6x6) from global to local.
        u_local = T @ u_global
        """
        L = self.get_length()
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        c = dx / L
        s = dy / L
        
        # T for one node (3x3)
        t_node = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        
        T = np.zeros((6, 6))
        T[0:3, 0:3] = t_node
        T[3:6, 3:6] = t_node
        return T

class BeamColumn2D(Element2D):
    """
    Beam-Column Element with Rotational Springs at ends (Giberson Model).
    """
    def __init__(self, id: int, node_i: Node, node_j: Node, 
                 E: float, A: float, I: float, 
                 spring_i: Optional[Hysteresis] = None, 
                 spring_j: Optional[Hysteresis] = None):
        super().__init__(id, node_i, node_j)
        self.E = E
        self.A = A
        self.I = I
        self.spring_i = spring_i
        self.spring_j = spring_j
        
        # Current forces (local system)
        # [N, V, M_i, N, V, M_j] -> Actually usually formulated as [N, M_i, M_j] for natural DOFs
        self.forces_local = np.zeros(6) 
        
    def get_elastic_stiffness_matrix_local(self):
        """
        Standard elastic stiffness matrix for frame element (Bernoulli-Euler).
        """
        L = self.get_length()
        E, A, I = self.E, self.A, self.I
        
        k = np.zeros((6, 6))
        
        # Axial
        k[0, 0] = E * A / L
        k[0, 3] = -E * A / L
        k[3, 0] = -E * A / L
        k[3, 3] = E * A / L
        
        # Bending / Shear
        k[1, 1] = 12 * E * I / L**3
        k[1, 2] = 6 * E * I / L**2
        k[1, 4] = -12 * E * I / L**3
        k[1, 5] = 6 * E * I / L**2
        
        k[2, 1] = 6 * E * I / L**2
        k[2, 2] = 4 * E * I / L
        k[2, 4] = -6 * E * I / L**2
        k[2, 5] = 2 * E * I / L
        
        k[4, 1] = -12 * E * I / L**3
        k[4, 2] = -6 * E * I / L**2
        k[4, 4] = 12 * E * I / L**3
        k[4, 5] = -6 * E * I / L**2
        
        k[5, 1] = 6 * E * I / L**2
        k[5, 2] = 2 * E * I / L
        k[5, 4] = -6 * E * I / L**2
        k[5, 5] = 4 * E * I / L
        
        return k

    def get_tangent_stiffness_matrix_local(self):
        """
        Returns tangent stiffness matrix modifying the elastic one with spring stiffnesses.
        Using static condensation or flexibility addition.
        """
        # Elastic Flexibility for rotations (2x2)
        # f = L / (6EI) * [[2, -1], [-1, 2]]
        L = self.get_length()
        f_const = L / (6 * self.E * self.I)
        f_elastic = f_const * np.array([[2.0, -1.0], [-1.0, 2.0]])
        
        # Spring flexibilities
        k_s_i = self.spring_i.stiffness if self.spring_i else float('inf')
        k_s_j = self.spring_j.stiffness if self.spring_j else float('inf')
        
        f_s_i = 1.0 / k_s_i if k_s_i > 1e-9 else 1e9 # High flexibility if stiffness is 0
        f_s_j = 1.0 / k_s_j if k_s_j > 1e-9 else 1e9
        
        f_total = f_elastic + np.diag([f_s_i, f_s_j])
        
        # Invert to get condensed rotational stiffness
        try:
            k_rot = np.linalg.inv(f_total)
        except np.linalg.LinAlgError:
            k_rot = np.zeros((2, 2))
            
        # Reconstruct full 6x6 local stiffness
        # We need to map [M_i, M_j] back to [u_y_i, theta_i, u_y_j, theta_j]
        # And add axial term.
        
        # Standard matrix from rotational stiffness terms k11, k12, k21, k22:
        # M_i = k11 * theta_i + k12 * theta_j - (k11+k12)/L * v_i + ...
        # This is derived from equilibrium.
        
        k11 = k_rot[0, 0]
        k12 = k_rot[0, 1]
        k21 = k_rot[1, 0]
        k22 = k_rot[1, 1]
        
        k = np.zeros((6, 6))
        
        # Axial (unchanged)
        EA_L = self.E * self.A / L
        k[0, 0] = EA_L; k[0, 3] = -EA_L
        k[3, 0] = -EA_L; k[3, 3] = EA_L
        
        # Rotational and Shear terms derived from k_rot
        # Shear V = -(Mi + Mj) / L
        
        r1 = (k11 + k21) / L
        r2 = (k12 + k22) / L
        r3 = (k11 + k12 + k21 + k22) / L**2
        
        # row 1 (v_i) -> index 1
        k[1, 1] = r3
        k[1, 2] = r1
        k[1, 4] = -r3
        k[1, 5] = r2
        
        # row 2 (theta_i) -> index 2
        k[2, 1] = r1
        k[2, 2] = k11
        k[2, 4] = -r1
        k[2, 5] = k12
        
        # row 4 (v_j) -> index 4
        k[4, 1] = -r3
        k[4, 2] = -r1
        k[4, 4] = r3
        k[4, 5] = -r2
        
        # row 5 (theta_j) -> index 5
        k[5, 1] = r2
        k[5, 2] = k21
        k[5, 4] = -r2
        k[5, 5] = k22
        
        return k

    def get_stiffness_matrix(self) -> np.ndarray:
        T = self.get_transformation_matrix()
        k_local = self.get_tangent_stiffness_matrix_local()
        return T.T @ k_local @ T
    
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        T = self.get_transformation_matrix()
        delta_u_local = T @ delta_u_global
        
        # We need cumulative local displacement for hysteresis trial?
        # Hysteresis models expect `total_displacement` or `incremental`?
        # My Hysteresis implementation expects `trial_disp` (Total).
        # We need to track total local rotation for springs.
        
        # Extract nodal rotations: theta_i, theta_j
        # Local DOFs: [u_xi, u_yi, theta_i, u_xj, u_yj, theta_j]
        
        # Calculate Beam Elastic Resisting Force (Linear)
        # Use simple "Series Spring" model logic again but formulated for Total Displacement.
        
        # Actually, simpler approach for this task:
        # Assume Springs are "Rotational Springs" at the nodes.
        # Total Rotation at node = Beam Rotation + Spring Rotation?
        # Or Node Rotation = Beam End Rotation + Spring Angle?
        # Physical model: Node is rigid. Connecting element has a spring.
        # Node Rotation = Theta_Node
        # Element End Rotation = Theta_Beam
        # Spring Deformation = Theta_Node - Theta_Beam
        # Moment M = K_spring * (Theta_Node - Theta_Beam)
        # Also M = K_beam * (Theta_Beam_i, Theta_Beam_j...)
        
        # This requires solving internal DoF (Theta_Beam).
        # Condensation again.
        
        # Let's perform condensation on the TRIAL state.
        
        # 1. Get Elastic Beam Stiffness (condensed) k_beam_elastic
        # 2. Get Spring Stiffness k_spr_i, k_spr_j (Trial)
        # But Spring Stiffness depends on Spring Deformation step.
        
        # Iteration needed at Element Level? 
        # For simplicity, let's assume "Series Limit":
        # M = min(M_elastic, M_yield)
        # But we have stiffness degrading.
        
        # Let's use the implementation: "Giberson Model"
        # dTheta_total = dTheta_beam + dTheta_spring
        # dTheta_spring = dM / K_spring
        # dM = K_beam * dTheta_beam
        
        # We need to solve for dM given dTheta_total.
        # dM = (K_beam^-1 + K_spring^-1)^-1 * dTheta_total
        
        # But Hysteresis is displacement driven. 
        # We need dTheta_spring to update Hysteresis.
        
        # Let's use the Previous Tangent of Spring to predict split?
        # Or iterate locally?
        # For efficiency, we assume the Spring dominates nonlinearity.
        # Hysteresis inputs: Spring Deformation (Theta_spring).
        
        # Current State tracking needed:
        # self.theta_spring_i, self.theta_spring_j (Total)
        
        if not hasattr(self, 'theta_spring_i'):
            self.theta_spring_i = 0.0
            self.theta_spring_j = 0.0
            
        # Get dTheta_total from input
        L = self.get_length()
        chord_rot_inc = (delta_u_local[4] - delta_u_local[1]) / L
        d_theta_i_tot = delta_u_local[2] - chord_rot_inc
        d_theta_j_tot = delta_u_local[5] - chord_rot_inc
        
        # Predict dTheta_spring using current tangents
        k_sp_i = self.spring_i.tangent if self.spring_i else 1e15
        k_sp_j = self.spring_j.tangent if self.spring_j else 1e15
        
        f_const = L / (6 * self.E * self.I)
        f_beam = f_const * np.array([[2.0, -1.0], [-1.0, 2.0]])
        k_beam = np.linalg.inv(f_beam)
        
        # Tangent Stiffness of Series System
        # F_series = F_beam + F_springs? No.
        # K_series = (F_beam + F_springs)^-1
        
        f_total = f_beam + np.diag([1.0/k_sp_i, 1.0/k_sp_j])
        k_series = np.linalg.inv(f_total)
        
        d_moments = k_series @ np.array([d_theta_i_tot, d_theta_j_tot])
        
        # Update Spring Estimations
        # dM = k_sp * dTheta_sp => dTheta_sp = dM / k_sp
        d_theta_sp_i = d_moments[0] / k_sp_i
        d_theta_sp_j = d_moments[1] / k_sp_j
        
        # Trial Update Springs
        m_i, k_i = self.spring_i.set_trial_displacement(self.theta_spring_i + d_theta_sp_i) if self.spring_i else (0, 1e15)
        m_j, k_j = self.spring_j.set_trial_displacement(self.theta_spring_j + d_theta_sp_j) if self.spring_j else (0, 1e15)
        
        self.trial_theta_spring_i = self.theta_spring_i + d_theta_sp_i
        self.trial_theta_spring_j = self.theta_spring_j + d_theta_sp_j
        
        # Re-Calculate Moments based on Trial Forces (Corrected)
        # M_i = m_i
        
        # Axial (Elastic)
        du_axial = delta_u_local[3] - delta_u_local[0]
        if not hasattr(self, 'current_N'): self.current_N = 0.0
        
        dN = (self.E * self.A / L) * du_axial
        N_trial = self.current_N + dN
        self.trial_N = N_trial
        
        # Shear
        V = -(m_i + m_j) / L
        
        forces = np.array([-N_trial, V, m_i, N_trial, -V, m_j])
        
        return T.T @ forces

    def commit_state(self):
        if self.spring_i: self.spring_i.commit()
        if self.spring_j: self.spring_j.commit()
        
        if hasattr(self, 'trial_theta_spring_i'):
            self.theta_spring_i = self.trial_theta_spring_i
            self.theta_spring_j = self.trial_theta_spring_j
            
        if hasattr(self, 'trial_N'):
            self.current_N = self.trial_N
