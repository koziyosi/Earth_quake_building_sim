"""
Extended Structural Elements Module.
Implements additional element types:
- #27: Brace elements (X-brace, V-brace, K-brace)
- #31: TMD (Tuned Mass Damper)
- #32: Friction damper
- #33: High damping rubber bearing
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .fem import Node
from .fem_3d import Element3D
from .hysteresis import Hysteresis, Bilinear


class BraceElement(Element3D):
    """
    Brace element for structural systems.
    Acts primarily in axial direction with possible buckling behavior.
    """
    
    def __init__(
        self, 
        id: int, 
        node_i: Node, 
        node_j: Node,
        E: float,           # Young's modulus (Pa)
        A: float,           # Cross-sectional area (m²)
        Fy_tension: float,  # Yield force in tension (N)
        Fy_compression: float = None,  # Yield/buckling force in compression
        r: float = 0.02     # Post-yield stiffness ratio
    ):
        super().__init__(id, node_i, node_j)
        self.E = E
        self.A = A
        self.Fy_tension = Fy_tension
        self.Fy_compression = Fy_compression or Fy_tension * 0.7  # Buckling reduction
        self.r = r
        
        L = self.get_length()
        self.k0 = E * A / L if L > 0 else 1e10
        
        # Asymmetric bilinear hysteresis for buckling
        self.hysteresis = AsymmetricBilinear(
            k0=self.k0,
            fy_pos=Fy_tension,
            fy_neg=self.Fy_compression,
            r=r
        )
        
        self.current_force = 0.0
        self.trial_force = 0.0
        self.current_deform = 0.0
        self.trial_deform = 0.0
        
    def get_length(self) -> float:
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Returns transformation matrix for axial element."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        L = self.get_length()
        
        if L < 1e-10:
            return np.eye(12)
        
        cx, cy, cz = dx/L, dy/L, dz/L
        
        # For brace, we mainly need axial transformation
        T = np.zeros((12, 12))
        
        # Direction cosines for each node
        for i, offset in enumerate([0, 6]):
            T[offset, offset] = cx
            T[offset, offset+1] = cy
            T[offset, offset+2] = cz
            T[offset+1, offset] = -cy if abs(cz) < 0.999 else 0
            T[offset+1, offset+1] = cx if abs(cz) < 0.999 else 0
            T[offset+1, offset+2] = 0
            T[offset+2, offset] = 0
            T[offset+2, offset+1] = cz if abs(cz) < 0.999 else -1
            T[offset+2, offset+2] = -cy if abs(cz) < 0.999 else 0
            
            # Rotational DOFs (simplified)
            for j in range(3):
                T[offset+3+j, offset+3+j] = 1.0
                
        return T
    
    @property
    def damage_index(self) -> float:
        """Damage based on ductility demand."""
        yield_def = self.Fy_tension / self.k0
        return abs(self.current_deform) / yield_def if yield_def > 0 else 0
    
    def get_stiffness_matrix(self) -> np.ndarray:
        """Returns 12x12 stiffness matrix."""
        k = np.zeros((12, 12))
        kt = self.hysteresis.tangent
        
        # Axial only
        k[0, 0] = kt
        k[0, 6] = -kt
        k[6, 0] = -kt
        k[6, 6] = kt
        
        T = self.get_transformation_matrix()
        return T.T @ k @ T
    
    def get_element_dof_indices(self) -> List[int]:
        return self.node_i.dof_indices + self.node_j.dof_indices
    
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        """Update element state."""
        indices = self.get_element_dof_indices()
        
        # Get local displacement
        T = self.get_transformation_matrix()
        delta_u_el = np.array([
            delta_u_global[i] if i >= 0 else 0.0 for i in indices
        ])
        delta_u_local = T @ delta_u_el
        
        # Axial deformation
        delta_axial = delta_u_local[6] - delta_u_local[0]
        self.trial_deform = self.current_deform + delta_axial
        
        # Calculate force from hysteresis
        self.hysteresis.set_trial_disp(self.trial_deform)
        self.trial_force = self.hysteresis.force
        
        # Return element forces in global
        f_local = np.zeros(12)
        f_local[0] = -self.trial_force
        f_local[6] = self.trial_force
        
        return T.T @ f_local
    
    def commit_state(self):
        self.current_deform = self.trial_deform
        self.current_force = self.trial_force
        self.hysteresis.commit()


class AsymmetricBilinear(Hysteresis):
    """Bilinear hysteresis with different tension/compression yield."""
    
    def __init__(self, k0: float, fy_pos: float, fy_neg: float, r: float):
        super().__init__(k0)
        self.fy_pos = fy_pos
        self.fy_neg = fy_neg
        self.r = r
        self.kp = k0 * r
        
        self.plastic_disp = 0.0
        self.plastic_disp_commit = 0.0
        
    def _calculate_trial_state(self):
        u = self.trial_disp
        f_trial = self.k0 * (u - self.plastic_disp_commit)
        
        if f_trial > self.fy_pos:
            self.trial_force = self.fy_pos + self.kp * (u - self.fy_pos/self.k0 - self.plastic_disp_commit)
            self.trial_tangent = self.kp
            self.plastic_disp = u - self.trial_force / self.k0
        elif f_trial < -self.fy_neg:
            self.trial_force = -self.fy_neg + self.kp * (u + self.fy_neg/self.k0 - self.plastic_disp_commit)
            self.trial_tangent = self.kp
            self.plastic_disp = u - self.trial_force / self.k0
        else:
            self.trial_force = f_trial
            self.trial_tangent = self.k0
            self.plastic_disp = self.plastic_disp_commit
            
    def _commit_history(self):
        self.plastic_disp_commit = self.plastic_disp


class TunedMassDamper(Element3D):
    """
    Tuned Mass Damper (TMD) element.
    Models a mass-spring-damper system attached to a structure.
    """
    
    def __init__(
        self,
        id: int,
        node_structure: Node,  # Node on main structure
        node_tmd: Node,        # TMD mass node
        mass: float,           # TMD mass (kg)
        stiffness: float,      # Spring stiffness (N/m)
        damping: float         # Damping coefficient (Ns/m)
    ):
        super().__init__(id, node_structure, node_tmd)
        self.mass = mass
        self.k = stiffness
        self.c = damping
        
        # Calculate natural frequency
        self.omega = np.sqrt(stiffness / mass)
        self.frequency = self.omega / (2 * np.pi)
        self.period = 1 / self.frequency if self.frequency > 0 else 0
        
    @classmethod
    def design_for_structure(
        cls,
        id: int,
        node_structure: Node,
        node_tmd: Node,
        structure_mass: float,
        structure_period: float,
        mass_ratio: float = 0.02,  # Typical 1-5%
        damping_ratio: float = 0.1
    ) -> 'TunedMassDamper':
        """
        Design TMD for a given structure.
        
        Uses Den Hartog's optimal tuning formulas.
        """
        m_tmd = structure_mass * mass_ratio
        omega_s = 2 * np.pi / structure_period
        
        # Optimal tuning ratio
        f_opt = 1 / (1 + mass_ratio)
        omega_tmd = f_opt * omega_s
        
        k_tmd = m_tmd * omega_tmd ** 2
        c_tmd = 2 * damping_ratio * m_tmd * omega_tmd
        
        return cls(id, node_structure, node_tmd, m_tmd, k_tmd, c_tmd)
    
    def get_length(self) -> float:
        return 0.0  # Zero-length element
    
    def get_transformation_matrix(self) -> np.ndarray:
        return np.eye(12)
    
    @property
    def damage_index(self) -> float:
        return 0.0  # TMD doesn't damage
    
    def get_stiffness_matrix(self) -> np.ndarray:
        k = np.zeros((12, 12))
        
        # X direction spring
        k[0, 0] = self.k
        k[0, 6] = -self.k
        k[6, 0] = -self.k
        k[6, 6] = self.k
        
        # Y direction spring
        k[1, 1] = self.k
        k[1, 7] = -self.k
        k[7, 1] = -self.k
        k[7, 7] = self.k
        
        return k
    
    def get_damping_matrix(self) -> np.ndarray:
        c = np.zeros((12, 12))
        
        # X direction damping
        c[0, 0] = self.c
        c[0, 6] = -self.c
        c[6, 0] = -self.c
        c[6, 6] = self.c
        
        # Y direction damping
        c[1, 1] = self.c
        c[1, 7] = -self.c
        c[7, 1] = -self.c
        c[7, 7] = self.c
        
        return c
    
    def get_mass_matrix(self) -> np.ndarray:
        m = np.zeros((12, 12))
        
        # Mass at TMD node only
        m[6, 6] = self.mass
        m[7, 7] = self.mass
        m[8, 8] = self.mass
        
        return m
    
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
    
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        # Linear spring force
        indices = self.get_element_dof_indices()
        f = np.zeros(12)
        
        du_rel_x = (delta_u_global[indices[6]] if indices[6] >= 0 else 0) - \
                   (delta_u_global[indices[0]] if indices[0] >= 0 else 0)
        du_rel_y = (delta_u_global[indices[7]] if indices[7] >= 0 else 0) - \
                   (delta_u_global[indices[1]] if indices[1] >= 0 else 0)
        
        f[0] = -self.k * du_rel_x
        f[1] = -self.k * du_rel_y
        f[6] = self.k * du_rel_x
        f[7] = self.k * du_rel_y
        
        return f
    
    def commit_state(self):
        pass


class FrictionDamper(Element3D):
    """
    Friction damper element.
    Uses Coulomb friction model.
    """
    
    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        friction_force: float,  # Slip force (N)
        initial_stiffness: float = 1e8  # Pre-slip stiffness (N/m)
    ):
        super().__init__(id, node_i, node_j)
        self.Ff = friction_force
        self.k_stick = initial_stiffness
        
        # State tracking
        self.slip_disp = 0.0
        self.trial_slip_disp = 0.0
        self.current_force = 0.0
        self.trial_force = 0.0
        self.is_slipping = False
        
    def get_length(self) -> float:
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_transformation_matrix(self) -> np.ndarray:
        # Same as brace
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        L = self.get_length()
        
        if L < 1e-10:
            return np.eye(12)
            
        cx, cy, cz = dx/L, dy/L, dz/L
        
        T = np.zeros((12, 12))
        for offset in [0, 6]:
            T[offset, offset] = cx
            T[offset, offset+1] = cy
            T[offset, offset+2] = cz
            for j in range(3):
                T[offset+3+j, offset+3+j] = 1.0
                
        return T
    
    @property
    def damage_index(self) -> float:
        return 0.0
    
    def get_stiffness_matrix(self) -> np.ndarray:
        k = np.zeros((12, 12))
        
        kt = 0.0 if self.is_slipping else self.k_stick
        
        k[0, 0] = kt
        k[0, 6] = -kt
        k[6, 0] = -kt
        k[6, 6] = kt
        
        T = self.get_transformation_matrix()
        return T.T @ k @ T
    
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
    
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        indices = self.get_element_dof_indices()
        T = self.get_transformation_matrix()
        
        delta_u_el = np.array([
            delta_u_global[i] if i >= 0 else 0.0 for i in indices
        ])
        delta_u_local = T @ delta_u_el
        
        delta_axial = delta_u_local[6] - delta_u_local[0]
        # Use committed slip displacement for trial calculation
        total_disp = self.slip_disp + delta_axial
        
        # Friction check - use committed slip_disp, not trial
        f_trial = self.k_stick * (total_disp - self.slip_disp)
        
        if abs(f_trial) > self.Ff:
            # Slipping
            self.is_slipping = True
            self.trial_force = np.sign(f_trial) * self.Ff
            self.trial_slip_disp = total_disp - self.trial_force / self.k_stick
        else:
            # Sticking
            self.is_slipping = False
            self.trial_force = f_trial
            
        f_local = np.zeros(12)
        f_local[0] = -self.trial_force
        f_local[6] = self.trial_force
        
        return T.T @ f_local
    
    def commit_state(self):
        """Commit trial state to current state."""
        self.slip_disp = self.trial_slip_disp
        self.current_force = self.trial_force
        self.is_slipping = False  # Reset for next step


class HighDampingRubberBearing(Element3D):
    """
    High Damping Rubber Bearing (HDRB).
    More detailed hysteresis than simple bilinear.
    """
    
    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        Kv: float,  # Vertical stiffness
        Kh: float,  # Horizontal stiffness
        Qd: float,  # Characteristic strength
        damping_ratio: float = 0.20  # Equivalent viscous damping
    ):
        super().__init__(id, node_i, node_j)
        self.Kv = Kv
        self.Kh = Kh
        self.Qd = Qd
        self.damping_ratio = damping_ratio
        
        # HDRB uses bilinear with high r
        r = 0.10  # Post-yield ratio for HDRB
        self.hysteresis_x = Bilinear(Kh, Qd, r)
        self.hysteresis_y = Bilinear(Kh, Qd, r)
        
        self.disp_x = 0.0
        self.disp_y = 0.0
        
    def get_length(self) -> float:
        return abs(self.node_j.z - self.node_i.z)
    
    def get_transformation_matrix(self) -> np.ndarray:
        return np.eye(12)
    
    @property
    def damage_index(self) -> float:
        dy = self.Qd / self.Kh
        max_disp = max(abs(self.disp_x), abs(self.disp_y))
        return max_disp / dy if dy > 0 else 0
    
    def get_stiffness_matrix(self) -> np.ndarray:
        k = np.zeros((12, 12))
        
        # Vertical
        k[2, 2] = self.Kv
        k[2, 8] = -self.Kv
        k[8, 2] = -self.Kv
        k[8, 8] = self.Kv
        
        # Horizontal X
        kx = self.hysteresis_x.tangent
        k[0, 0] = kx
        k[0, 6] = -kx
        k[6, 0] = -kx
        k[6, 6] = kx
        
        # Horizontal Y
        ky = self.hysteresis_y.tangent
        k[1, 1] = ky
        k[1, 7] = -ky
        k[7, 1] = -ky
        k[7, 7] = ky
        
        return k
    
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
    
    def update_state(self, delta_u_global: np.ndarray) -> np.ndarray:
        indices = self.get_element_dof_indices()
        
        # X displacement
        du_x = (delta_u_global[indices[6]] if indices[6] >= 0 else 0) - \
               (delta_u_global[indices[0]] if indices[0] >= 0 else 0)
        
        # Y displacement
        du_y = (delta_u_global[indices[7]] if indices[7] >= 0 else 0) - \
               (delta_u_global[indices[1]] if indices[1] >= 0 else 0)
        
        self.disp_x += du_x
        self.disp_y += du_y
        
        self.hysteresis_x.set_trial_disp(self.disp_x)
        self.hysteresis_y.set_trial_disp(self.disp_y)
        
        f = np.zeros(12)
        f[0] = -self.hysteresis_x.force
        f[1] = -self.hysteresis_y.force
        f[6] = self.hysteresis_x.force
        f[7] = self.hysteresis_y.force
        
        return f
    
    def commit_state(self):
        self.hysteresis_x.commit()
        self.hysteresis_y.commit()
