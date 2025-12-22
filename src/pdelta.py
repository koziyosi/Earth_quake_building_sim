"""
P-Delta Effects Module.
Implements geometric stiffness for P-Δ and P-δ effects (#24, #34).
"""
import numpy as np
from typing import List, Optional, Tuple


def calculate_geometric_stiffness(
    nodes: List,
    elements: List,
    axial_forces: np.ndarray
) -> np.ndarray:
    """
    Calculate geometric stiffness matrix for P-Delta effects.
    
    The geometric stiffness accounts for the destabilizing effect
    of axial loads on lateral stiffness.
    
    Kg = (N/L) * [1, -1; -1, 1] for each element
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        axial_forces: Axial force in each element (negative = compression)
        
    Returns:
        Global geometric stiffness matrix
    """
    # Determine DOF count
    ndof = 0
    for node in nodes:
        ndof = max(ndof, max(node.dof_indices) + 1)
        
    Kg = np.zeros((ndof, ndof))
    
    for elem_idx, elem in enumerate(elements):
        N = axial_forces[elem_idx] if elem_idx < len(axial_forces) else 0
        
        # Only compression causes P-Delta instability
        if N >= 0:
            continue
            
        L = elem.get_length()
        if L <= 0:
            continue
            
        # Geometric stiffness coefficient
        kg = abs(N) / L
        
        # Get transformation info
        dx = elem.node_j.x - elem.node_i.x
        dy = elem.node_j.y - elem.node_i.y  
        dz = elem.node_j.z - elem.node_i.z
        
        # Direction cosines
        cx, cy, cz = dx/L, dy/L, dz/L
        
        # For vertical elements (columns), P-Delta affects lateral DOFs
        if abs(cz) > 0.9:  # Mostly vertical
            # Get lateral DOF indices
            dof_i_x = elem.node_i.dof_indices[0]
            dof_i_y = elem.node_i.dof_indices[1]
            dof_j_x = elem.node_j.dof_indices[0]
            dof_j_y = elem.node_j.dof_indices[1]
            
            # Add geometric stiffness contribution (X direction)
            if dof_i_x >= 0 and dof_j_x >= 0:
                Kg[dof_i_x, dof_i_x] += kg
                Kg[dof_i_x, dof_j_x] -= kg
                Kg[dof_j_x, dof_i_x] -= kg
                Kg[dof_j_x, dof_j_x] += kg
                
            # Add geometric stiffness contribution (Y direction)
            if dof_i_y >= 0 and dof_j_y >= 0:
                Kg[dof_i_y, dof_i_y] += kg
                Kg[dof_i_y, dof_j_y] -= kg
                Kg[dof_j_y, dof_i_y] -= kg
                Kg[dof_j_y, dof_j_y] += kg
                
    return Kg


def get_axial_forces_from_gravity(
    nodes: List,
    elements: List
) -> np.ndarray:
    """
    Estimate axial forces from gravity loads.
    
    Simplified approach: sum tributary masses above each column.
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        
    Returns:
        Axial force array (negative = compression)
    """
    g = 9.81  # m/s²
    
    axial_forces = np.zeros(len(elements))
    
    # Group nodes by floor level
    floor_nodes = {}
    for node in nodes:
        z = round(node.z, 2)
        if z not in floor_nodes:
            floor_nodes[z] = []
        floor_nodes[z].append(node)
        
    floor_levels = sorted(floor_nodes.keys())
    
    for elem_idx, elem in enumerate(elements):
        # Check if vertical element
        dz = elem.node_j.z - elem.node_i.z
        L = elem.get_length()
        
        if L > 0 and abs(dz / L) > 0.9:  # Vertical element
            # Find levels above this column
            z_top = max(elem.node_i.z, elem.node_j.z)
            
            # Sum masses above
            cumulative_mass = 0
            for z in floor_levels:
                if z >= z_top:
                    for node in floor_nodes[z]:
                        cumulative_mass += node.mass
                        
            # Axial force (negative for compression)
            axial_forces[elem_idx] = -cumulative_mass * g
            
    return axial_forces


def apply_pdelta_to_solver(
    K: np.ndarray,
    nodes: List,
    elements: List,
    axial_forces: np.ndarray = None,
    scale_factor: float = 1.0
) -> np.ndarray:
    """
    Apply P-Delta correction to stiffness matrix.
    
    K_effective = K + Kg
    
    Args:
        K: Original stiffness matrix
        nodes: List of nodes
        elements: List of elements
        axial_forces: Axial forces (or None to estimate from gravity)
        scale_factor: Scaling factor for geometric stiffness
        
    Returns:
        Modified stiffness matrix with P-Delta effects
    """
    if axial_forces is None:
        axial_forces = get_axial_forces_from_gravity(nodes, elements)
        
    Kg = calculate_geometric_stiffness(nodes, elements, axial_forces)
    
    return K + scale_factor * Kg


def estimate_pdelta_amplification(
    base_shear: float,
    roof_disp: float,
    total_weight: float,
    story_height: float
) -> Tuple[float, float]:
    """
    Estimate P-Delta amplification factor (stability coefficient).
    
    θ = P*Δ / (V*h)
    
    Where:
    - P = gravity load (weight)
    - Δ = lateral displacement
    - V = base shear
    - h = story height
    
    Amplification factor: 1 / (1 - θ)
    
    Args:
        base_shear: Base shear force (N)
        roof_disp: Roof displacement (m)
        total_weight: Total building weight (N)
        story_height: Effective story height (m)
        
    Returns:
        (stability_coefficient θ, amplification_factor)
    """
    if base_shear <= 0 or story_height <= 0:
        return (0.0, 1.0)
        
    theta = (total_weight * roof_disp) / (base_shear * story_height)
    
    # Amplification factor (bounded)
    if theta >= 1.0:
        amplification = float('inf')  # Unstable
    else:
        amplification = 1.0 / (1.0 - theta)
        
    return (theta, amplification)


def check_pdelta_significance(
    nodes: List,
    elements: List,
    max_drift: float,
    story_heights: List[float]
) -> dict:
    """
    Check if P-Delta effects are significant.
    
    Per ASCE 7, P-Delta is significant if θ > 0.10.
    
    Returns:
        Dictionary with significance check results
    """
    g = 9.81
    
    # Total weight
    total_weight = sum(node.mass * g for node in nodes)
    
    # Estimate base shear (from drift and approximate stiffness)
    # Rough estimate: V ≈ k * drift, k ≈ 3EI/L³
    avg_height = np.mean(story_heights) if story_heights else 3.5
    
    # Very rough estimate
    estimated_V = total_weight * 0.15  # Assume 15% base shear coefficient
    
    theta, amp = estimate_pdelta_amplification(
        estimated_V,
        max_drift * avg_height,
        total_weight,
        avg_height
    )
    
    return {
        'stability_coefficient': theta,
        'amplification_factor': amp,
        'is_significant': theta > 0.10,
        'is_critical': theta > 0.25,
        'recommendation': (
            'P-Delta effects should be considered' if theta > 0.10 else
            'P-Delta effects can be neglected'
        )
    }


class PDeltaAnalyzer:
    """
    Analyzer for P-Delta effects in time history analysis.
    """
    
    def __init__(
        self,
        nodes: List,
        elements: List,
        include_pdelta: bool = True
    ):
        self.nodes = nodes
        self.elements = elements
        self.include_pdelta = include_pdelta
        
        # Pre-calculate gravity axial forces
        self.axial_forces = get_axial_forces_from_gravity(nodes, elements)
        
    def get_effective_stiffness(self, K: np.ndarray) -> np.ndarray:
        """Get stiffness matrix with P-Delta correction."""
        if not self.include_pdelta:
            return K
            
        return apply_pdelta_to_solver(
            K, self.nodes, self.elements, self.axial_forces
        )
        
    def update_axial_forces(self, element_forces: np.ndarray):
        """Update axial forces from current state."""
        self.axial_forces = element_forces
