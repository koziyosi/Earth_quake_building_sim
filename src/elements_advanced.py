"""
Advanced Damper Elements Module.
Additional damper types beyond the basic ones.
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .fem import Node
from .fem_3d import Element3D


class ViscoelasticDamper(Element3D):
    """
    Viscoelastic damper element (VE damper).
    
    Force = K(γ) * d + C(γ) * v
    where K and C are strain-dependent.
    """
    
    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        K0: float,          # Initial stiffness (N/m)
        C0: float,          # Initial damping (Ns/m)
        thickness: float,   # VE layer thickness (m)
        shear_modulus: float = 1e6  # G (Pa)
    ):
        super().__init__(id, node_i, node_j)
        self.K0 = K0
        self.C0 = C0
        self.thickness = thickness
        self.G = shear_modulus
        
        # State
        self.disp = 0.0
        self.vel = 0.0
        self.trial_disp = 0.0
        self.trial_vel = 0.0
        self.force = 0.0
        
    def get_length(self) -> float:
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
        
    def get_transformation_matrix(self) -> np.ndarray:
        return np.eye(12)  # Simplified
        
    @property
    def damage_index(self) -> float:
        return 0.0  # Dampers don't damage
        
    def _strain_dependent_properties(self, gamma: float) -> Tuple[float, float]:
        """Get K and C as function of shear strain."""
        # VE material softening with strain
        if gamma < 0.01:
            factor = 1.0
        elif gamma < 0.1:
            factor = 1.0 - 0.3 * (gamma - 0.01) / 0.09
        else:
            factor = 0.7 - 0.2 * min(gamma - 0.1, 0.9)
            
        return self.K0 * factor, self.C0 * factor
        
    def get_stiffness_matrix(self) -> np.ndarray:
        gamma = abs(self.disp) / self.thickness if self.thickness > 0 else 0
        K, _ = self._strain_dependent_properties(gamma)
        
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = K
        k[0, 6] = k[6, 0] = -K
        return k
        
    def get_damping_matrix(self) -> np.ndarray:
        gamma = abs(self.disp) / self.thickness if self.thickness > 0 else 0
        _, C = self._strain_dependent_properties(gamma)
        
        c = np.zeros((12, 12))
        c[0, 0] = c[6, 6] = C
        c[0, 6] = c[6, 0] = -C
        return c
        
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
        
    def update_state(self, delta_u: np.ndarray) -> np.ndarray:
        indices = self.get_element_dof_indices()
        
        du = delta_u[indices[6]] - delta_u[indices[0]] if indices[0] >= 0 else 0
        self.trial_disp = self.disp + du
        
        gamma = abs(self.trial_disp) / self.thickness if self.thickness > 0 else 0
        K, C = self._strain_dependent_properties(gamma)
        
        # Simplified velocity estimate
        self.force = K * self.trial_disp
        
        f = np.zeros(12)
        f[0] = -self.force
        f[6] = self.force
        return f
        
    def commit_state(self):
        self.disp = self.trial_disp


class RotaryInertialDamper(Element3D):
    """
    Rotary Inertial Damper (回転慣性質量ダンパー).
    
    Uses rotational inertia to amplify apparent mass.
    F = m_app * a, where m_app >> actual mass
    """
    
    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        apparent_mass: float,   # Apparent mass (kg)
        damping_coef: float,    # Damping coefficient (Ns/m)
        stiffness: float = 0    # Optional spring (N/m)
    ):
        super().__init__(id, node_i, node_j)
        self.m_app = apparent_mass
        self.c = damping_coef
        self.k = stiffness
        
        self.disp = 0.0
        self.vel = 0.0
        
    def get_length(self) -> float:
        return 0.0
        
    def get_transformation_matrix(self) -> np.ndarray:
        return np.eye(12)
        
    @property
    def damage_index(self) -> float:
        return 0.0
        
    def get_stiffness_matrix(self) -> np.ndarray:
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = self.k
        k[0, 6] = k[6, 0] = -self.k
        return k
        
    def get_damping_matrix(self) -> np.ndarray:
        c = np.zeros((12, 12))
        c[0, 0] = c[6, 6] = self.c
        c[0, 6] = c[6, 0] = -self.c
        return c
        
    def get_mass_matrix(self) -> np.ndarray:
        """Rotary inertia contributes to mass matrix."""
        m = np.zeros((12, 12))
        # Apparent mass effect on relative motion
        m[0, 0] = self.m_app
        m[6, 6] = self.m_app
        m[0, 6] = m[6, 0] = -self.m_app
        return m
        
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
        
    def update_state(self, delta_u: np.ndarray) -> np.ndarray:
        indices = self.get_element_dof_indices()
        du = delta_u[indices[6]] - delta_u[indices[0]] if indices[0] >= 0 else 0
        self.disp += du
        
        f = np.zeros(12)
        f[0] = -self.k * self.disp
        f[6] = self.k * self.disp
        return f
        
    def commit_state(self):
        pass


class BucklingRestrainedBrace(Element3D):
    """
    Buckling Restrained Brace (BRB / 座屈拘束ブレース).
    
    Symmetric tension-compression behavior without buckling.
    """
    
    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        core_area: float,       # Core plate area (m²)
        steel_E: float = 2.05e11,  # Young's modulus (Pa)
        yield_stress: float = 235e6,  # Fy (Pa)
        strain_hardening: float = 0.01  # Post-yield ratio
    ):
        super().__init__(id, node_i, node_j)
        self.Ac = core_area
        self.E = steel_E
        self.Fy = yield_stress
        self.r = strain_hardening
        
        L = self.get_length()
        self.k0 = steel_E * core_area / L if L > 0 else 1e10
        self.Py = yield_stress * core_area
        self.dy = self.Py / self.k0
        
        # State
        self.disp = 0.0
        self.trial_disp = 0.0
        self.plastic_disp = 0.0
        self.force = 0.0
        self.tangent = self.k0
        
    def get_length(self) -> float:
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
        
    def get_transformation_matrix(self) -> np.ndarray:
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
        max_disp = max(abs(self.disp), abs(self.trial_disp))
        return max_disp / self.dy if self.dy > 0 else 0
        
    def get_stiffness_matrix(self) -> np.ndarray:
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = self.tangent
        k[0, 6] = k[6, 0] = -self.tangent
        
        T = self.get_transformation_matrix()
        return T.T @ k @ T
        
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
        
    def update_state(self, delta_u: np.ndarray) -> np.ndarray:
        indices = self.get_element_dof_indices()
        T = self.get_transformation_matrix()
        
        du_el = np.array([delta_u[i] if i >= 0 else 0 for i in indices])
        du_local = T @ du_el
        
        delta_axial = du_local[6] - du_local[0]
        self.trial_disp = self.disp + delta_axial
        
        # Symmetric bilinear
        elastic_disp = self.trial_disp - self.plastic_disp
        f_trial = self.k0 * elastic_disp
        
        if abs(f_trial) > self.Py:
            sign = np.sign(f_trial)
            self.force = self.Py * sign + self.k0 * self.r * (self.trial_disp - sign * self.dy - self.plastic_disp)
            self.tangent = self.k0 * self.r
            self.plastic_disp = self.trial_disp - self.force / self.k0
        else:
            self.force = f_trial
            self.tangent = self.k0
            
        f_local = np.zeros(12)
        f_local[0] = -self.force
        f_local[6] = self.force
        
        return T.T @ f_local
        
    def commit_state(self):
        self.disp = self.trial_disp


class WallPanelElement(Element3D):
    """
    Wall panel element for shear walls.
    
    Simplified shear panel behavior.
    """
    
    def __init__(
        self,
        id: int,
        nodes: list,  # 4 corner nodes
        thickness: float,
        shear_modulus: float = 10e9,  # G (Pa)
        yield_shear: float = 10e6     # τy (Pa)
    ):
        # Use first two nodes for base class
        super().__init__(id, nodes[0], nodes[1])
        self.all_nodes = nodes
        self.t = thickness
        self.G = shear_modulus
        self.tau_y = yield_shear
        
        # Calculate panel dimensions
        self.width = abs(nodes[1].x - nodes[0].x)
        self.height = abs(nodes[2].z - nodes[0].z)
        self.area = self.width * self.height
        
        # Stiffness
        self.k0 = shear_modulus * self.area * thickness / self.height if self.height > 0 else 1e10
        
        # State
        self.shear_strain = 0.0
        self.shear_force = 0.0
        
    def get_length(self) -> float:
        return self.height
        
    def get_transformation_matrix(self) -> np.ndarray:
        return np.eye(12)
        
    @property
    def damage_index(self) -> float:
        yield_strain = self.tau_y / self.G
        return abs(self.shear_strain) / yield_strain if yield_strain > 0 else 0
        
    def get_stiffness_matrix(self) -> np.ndarray:
        # Simplified 12x12 (connects first two nodes primarily)
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = self.k0
        k[0, 6] = k[6, 0] = -self.k0
        return k
        
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
        
    def update_state(self, delta_u: np.ndarray) -> np.ndarray:
        indices = self.get_element_dof_indices()
        
        du = delta_u[indices[6]] - delta_u[indices[0]] if indices[0] >= 0 else 0
        self.shear_strain += du / self.height if self.height > 0 else 0
        
        # Bilinear shear
        tau = self.G * self.shear_strain
        if abs(tau) > self.tau_y:
            tau = np.sign(tau) * self.tau_y
            
        self.shear_force = tau * self.area * self.t
        
        f = np.zeros(12)
        f[0] = -self.shear_force
        f[6] = self.shear_force
        return f
        
    def commit_state(self):
        pass
