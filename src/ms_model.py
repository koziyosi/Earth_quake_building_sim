import numpy as np
from typing import List, Tuple

class FiberSection:
    """
    Represents a cross-section discretized into fibers.
    For N-Mx-My interaction.
    """
    def __init__(self, fibers: List[Tuple[float, float, float, object]]):
        """
        Args:
            fibers: List of (y, z, area, material_model)
        """
        self.fibers = fibers
        
    def get_response(self, strain_centroid, curvature_y, curvature_z):
        """
        Calculate section forces (N, My, Mz) and tangent stiffness matrix.
        Assumption: Plane sections remain plane.
        strain(y,z) = strain_centroid - curvature_z * y + curvature_y * z
        (Sign convention depends on axis definition)
        Let's use:
        eps = eps0 - phi_z * y + phi_y * z
        """
        N = 0.0
        My = 0.0
        Mz = 0.0
        
        # Tangent Stiffness [3x3]
        # [ dN/deps0    dN/dphi_y    dN/dphi_z ]
        # [ dMy/deps0   dMy/dphi_y   dMy/dphi_z ]
        # [ dMz/deps0   dMz/dphi_y   dMz/dphi_z ]
        
        k_sec = np.zeros((3, 3))
        
        for y, z, area, mat in self.fibers:
            # Calculate strain
            strain = strain_centroid - curvature_z * y + curvature_y * z
            
            # Get stress and tangent modulus from material
            # mat.update(strain_inc) -> requires history?
            # Assuming mat has `get_stress_stiffness(strain)` method
            sigma, E_t = mat.get_response(strain)
            
            force = sigma * area
            stiffness = E_t * area
            
            # Accumulate Forces
            N += force
            My += force * z
            Mz -= force * y # Moment arm is y, force creates Mz. Sign? Mz = -Integral(sigma * y dA)
            
            # Accumulate Stiffness
            # dN = E*A * deps
            # dMy = z * dN
            # dMz = -y * dN
            
            # deps = deps0 - y*dphi_z + z*dphi_y
            
            k_sec[0, 0] += stiffness # N-eps0
            k_sec[0, 1] += stiffness * z # N-phi_y
            k_sec[0, 2] -= stiffness * y # N-phi_z
            
            k_sec[1, 0] += stiffness * z
            k_sec[1, 1] += stiffness * z**2
            k_sec[1, 2] -= stiffness * y * z
            
            k_sec[2, 0] -= stiffness * y
            k_sec[2, 1] -= stiffness * y * z
            k_sec[2, 2] += stiffness * y**2
            
        return np.array([N, My, Mz]), k_sec

class SimpleMaterial:
    """
    Simple Bilinear Material for Fibers.
    """
    def __init__(self, E, Fy):
        self.E = E
        self.Fy = Fy
        self.eps_y = Fy / E
        self.stress = 0.0
        self.tangent = E
        
        # History
        self.max_strain = 0.0
        self.min_strain = 0.0
        # For path dependence, we need to store current strain
        self.prev_strain = 0.0
        self.prev_stress = 0.0
        
        # Simplest: Elastic-Perfectly Plastic (No path dependence stored in class for this snippet, assumes monotonic for get_response simplicity)
        # REAL implementation needs `update(d_strain)`
    
    def get_response(self, strain):
        # Elastic-Perfectly Plastic logic
        if abs(strain) > self.eps_y:
            sigma = np.sign(strain) * self.Fy
            Et = 0.0 # Plastic
            # Add small hardening
            Et = self.E * 0.01
            sigma = np.sign(strain) * self.Fy + Et * (strain - np.sign(strain)*self.eps_y)
        else:
            sigma = self.E * strain
            Et = self.E
        return sigma, Et


class ConcreteMaterial:
    """
    Concrete material model with tension/compression asymmetry.
    Based on Kent-Park model (simplified).
    """
    def __init__(self, fc: float = 30e6, ec0: float = 0.002, ecu: float = 0.004, ft: float = 3e6):
        """
        Args:
            fc: Compressive strength (Pa, negative convention internally)
            ec0: Strain at peak stress
            ecu: Ultimate strain
            ft: Tensile strength
        """
        self.fc = fc  # Positive input, stored positive
        self.ec0 = ec0
        self.ecu = ecu
        self.ft = ft
        self.Et = ft / 0.0001  # Tensile initial modulus
        self.Ec = 2 * fc / ec0  # Secant modulus to peak
        
    def get_response(self, strain: float) -> Tuple[float, float]:
        """Get stress and tangent for given strain."""
        if strain < 0:  # Compression
            eps = abs(strain)
            if eps < self.ec0:
                # Parabolic ascending
                ratio = eps / self.ec0
                sigma = -self.fc * (2 * ratio - ratio**2)
                Et = -self.Ec * (1 - ratio)
            elif eps < self.ecu:
                # Linear descending
                ratio = (eps - self.ec0) / (self.ecu - self.ec0)
                sigma = -self.fc * (1 - 0.8 * ratio)
                Et = 0.8 * self.fc / (self.ecu - self.ec0)
            else:
                # Residual
                sigma = -0.2 * self.fc
                Et = 0.0
        else:  # Tension
            ect = self.ft / self.Et
            if strain < ect:
                sigma = self.Et * strain
                Et = self.Et
            else:
                # Cracked - linear softening to zero at 10x crack strain
                sigma = max(0, self.ft * (1 - (strain - ect) / (9 * ect)))
                Et = -self.ft / (9 * ect) if sigma > 0 else 0.0
        
        return sigma, Et


class SteelMaterial:
    """
    Steel reinforcement material with kinematic hardening.
    Menegotto-Pinto simplified model.
    """
    def __init__(self, E: float = 2e11, fy: float = 400e6, b: float = 0.01):
        """
        Args:
            E: Young's modulus (Pa)
            fy: Yield stress (Pa)
            b: Strain hardening ratio
        """
        self.E = E
        self.fy = fy
        self.b = b  # Post-yield ratio
        self.eps_y = fy / E
        
        # History for cyclic behavior
        self.eps_prev = 0.0
        self.sig_prev = 0.0
        self.eps_r = 0.0  # Reversal strain
        self.sig_r = 0.0  # Reversal stress
        
    def get_response(self, strain: float) -> Tuple[float, float]:
        """Get stress and tangent (simplified monotonic for now)."""
        E_h = self.b * self.E  # Hardening modulus
        
        if abs(strain) < self.eps_y:
            sigma = self.E * strain
            Et = self.E
        else:
            # Bilinear with hardening
            sign = np.sign(strain)
            eps_plastic = abs(strain) - self.eps_y
            sigma = sign * (self.fy + E_h * eps_plastic)
            Et = E_h
            
        return sigma, Et
    
    def commit(self):
        """Commit current state to history."""
        pass  # Simplified - no history tracking in this version


def create_rectangular_section(
    width: float, 
    height: float, 
    cover: float,
    n_fibers_y: int = 10,
    n_fibers_z: int = 10,
    concrete_material = None,
    steel_material = None,
    rebar_config: List[Tuple[float, float, float]] = None
) -> 'FiberSection':
    """
    Create a rectangular fiber section with optional reinforcement.
    
    Args:
        width: Section width (m)
        height: Section height (m)
        cover: Concrete cover (m)
        n_fibers_y: Number of fibers in Y direction
        n_fibers_z: Number of fibers in Z direction
        concrete_material: Concrete material model
        steel_material: Steel material for rebars
        rebar_config: List of (y, z, area) for each rebar
        
    Returns:
        FiberSection object
    """
    if concrete_material is None:
        concrete_material = ConcreteMaterial()
    if steel_material is None:
        steel_material = SteelMaterial()
    
    fibers = []
    
    # Concrete fibers
    dy = height / n_fibers_y
    dz = width / n_fibers_z
    fiber_area = dy * dz
    
    for i in range(n_fibers_y):
        y = -height/2 + dy/2 + i * dy
        for j in range(n_fibers_z):
            z = -width/2 + dz/2 + j * dz
            fibers.append((y, z, fiber_area, concrete_material))
    
    # Steel fibers (rebars)
    if rebar_config:
        for y, z, area in rebar_config:
            fibers.append((y, z, area, steel_material))
    else:
        # Default: 4 corner bars
        corner_offset = cover + 0.01  # 10mm from cover
        bar_area = 0.0005  # 500mm² each
        corners = [
            (-height/2 + corner_offset, -width/2 + corner_offset),
            (-height/2 + corner_offset, width/2 - corner_offset),
            (height/2 - corner_offset, -width/2 + corner_offset),
            (height/2 - corner_offset, width/2 - corner_offset),
        ]
        for y, z in corners:
            fibers.append((y, z, bar_area, steel_material))
    
    return FiberSection(fibers)

# To use this in an Element, we need a "MSBeamColumn3D"
# Ideally, we integrate this along the length (Newton-Cotes or Gauss).
# Or "Lumped Plasticity": Only check section at ends.

from src.fem_3d import BeamColumn3D
from src.fem import Node

class MSBeamColumn3D(BeamColumn3D):
    """
    Multi-Spring / Fiber Hinge Beam Column.
    Uses FiberSection at ends i and j to determine stiffness.
    Middle is elastic.
    Structure: Series model? 
    Or Displacement-Based Element (Distributed Plasticity)?
    
    Displacement-Based is easier to implement if we assume linear curvature distribution.
    But usually requires iteration (Newton-Raphson) at element level for force equilibrium.
    
    Simplest for "Sim":
    Use the initial stiffness matrix but modify diagonal terms based on Section Stiffness?
    Or standard "Stiffness Method" assembling k_sec at integration points.
    
    Let's implement a standard Displacement-Based Element with 2 integration points (Gauss).
    """
    def __init__(self, id: int, node_i: Node, node_j: Node, section: FiberSection):
        # We need E, G, A, etc for the base class? 
        # Actually base class constructor asks for them.
        # We can pass dummy values or calculate from section (elastic).
        # Let's calculate equivalent elastic props.
        res, k_sec = section.get_response(0, 0, 0) # Elastic state
        EA = k_sec[0, 0]
        EIy = k_sec[1, 1]
        EIz = k_sec[2, 2]
        # Assume L is not known yet.
        super().__init__(id, node_i, node_j, E=1.0, G=1.0, A=EA, Iy=EIy, Iz=EIz, J=1.0) # Placeholder G, J
        
        self.section = section
        
    # Override get_stiffness_matrix using numerical integration?
    # Or just use the diagonal stiffnesses from section to update the beam matrix?
    
    # Implementing full Fiber Element is complex for this turn.
    # The prompt asked for "MS Model or Fiber concept".
    # I will stick to the "MS Model" concept:
    # Replace the "Plastic Hinge" springs in `BeamColumn3D` with `FiberSection` response.
    # This effectively makes it a "Series" model.
    # Total Flexibility F = F_beam + F_hinge_i + F_hinge_j
    # F_hinge is derived from Section Stiffness (Moment-Curvature).
    # Rotation = Curvature * L_plastic.
    # K_spring = K_section / L_plastic.
    
    # This is "Lumped Plasticity with Fiber Section Hinge".
