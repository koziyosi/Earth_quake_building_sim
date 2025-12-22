"""
Pushover Analysis Module.
Implements static nonlinear pushover analysis (#41).
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PushoverResult:
    """Results from pushover analysis."""
    base_shear: np.ndarray        # Base shear at each step (N)
    roof_disp: np.ndarray         # Roof displacement at each step (m)
    story_drifts: Dict[int, np.ndarray]  # Story drifts at each step
    yielded_elements: List[int]   # Element IDs that yielded
    capacity_curve: Tuple[np.ndarray, np.ndarray]  # (disp, shear)
    

class PushoverAnalyzer:
    """
    Static nonlinear pushover analysis.
    
    Features:
    - Displacement or force controlled
    - Multiple load patterns (uniform, triangular, mode-based)
    - Capacity curve generation
    - Bilinearization for capacity spectrum method
    """
    
    def __init__(
        self, 
        nodes: List,
        elements: List,
        fixed_dofs: List[int]
    ):
        """
        Initialize pushover analyzer.
        
        Args:
            nodes: List of Node objects
            elements: List of Element objects
            fixed_dofs: List of DOF indices that are fixed (supports)
        """
        self.nodes = nodes
        self.elements = elements
        self.fixed_dofs = set(fixed_dofs)
        
        # Determine DOF count
        self.ndof = 0
        for node in nodes:
            self.ndof = max(self.ndof, max(node.dof_indices) + 1)
            
        # Load pattern (will be set)
        self.load_pattern: np.ndarray = None
        
    def set_uniform_load_pattern(self, direction: str = 'x'):
        """Set uniform lateral load distribution."""
        self.load_pattern = np.zeros(self.ndof)
        
        dof_idx = {'x': 0, 'y': 1, 'z': 2}[direction.lower()]
        
        for node in self.nodes:
            if node.mass > 0 and node.dof_indices[dof_idx] >= 0:
                self.load_pattern[node.dof_indices[dof_idx]] = 1.0
                
        # Normalize
        self.load_pattern /= np.sum(self.load_pattern) 
        
    def set_triangular_load_pattern(self, direction: str = 'x'):
        """Set triangular (linear) lateral load distribution."""
        self.load_pattern = np.zeros(self.ndof)
        
        dof_idx = {'x': 0, 'y': 1, 'z': 2}[direction.lower()]
        
        # Find max height
        max_z = max(node.z for node in self.nodes)
        
        for node in self.nodes:
            if node.mass > 0 and node.dof_indices[dof_idx] >= 0:
                # Load proportional to height
                factor = node.z / max_z if max_z > 0 else 0
                self.load_pattern[node.dof_indices[dof_idx]] = factor * node.mass
                
        # Normalize
        total = np.sum(self.load_pattern)
        if total > 0:
            self.load_pattern /= total
            
    def set_mode_based_pattern(self, mode_shape: np.ndarray, direction: str = 'x'):
        """Set load pattern based on first mode shape."""
        self.load_pattern = np.zeros(self.ndof)
        
        dof_idx = {'x': 0, 'y': 1, 'z': 2}[direction.lower()]
        
        for node in self.nodes:
            if node.mass > 0 and node.dof_indices[dof_idx] >= 0:
                dof = node.dof_indices[dof_idx]
                if dof < len(mode_shape):
                    self.load_pattern[dof] = node.mass * mode_shape[dof]
                    
        # Normalize
        total = np.sum(np.abs(self.load_pattern))
        if total > 0:
            self.load_pattern /= total
            
    def run_displacement_control(
        self,
        target_disp: float,
        control_dof: int,
        n_steps: int = 100,
        tol: float = 1e-6,
        max_iter: int = 20
    ) -> PushoverResult:
        """
        Run displacement-controlled pushover analysis.
        
        Args:
            target_disp: Target displacement at control DOF (m)
            control_dof: DOF index for displacement control
            n_steps: Number of load steps
            tol: Convergence tolerance
            max_iter: Max Newton-Raphson iterations
            
        Returns:
            PushoverResult
        """
        if self.load_pattern is None:
            self.set_triangular_load_pattern()
            
        # Initialize
        u = np.zeros(self.ndof)
        base_shear_hist = []
        roof_disp_hist = []
        yielded = set()
        
        # Displacement increment
        d_target = target_disp / n_steps
        
        for step in range(n_steps):
            # Current target displacement
            u_target = (step + 1) * d_target
            
            # Newton-Raphson iteration
            for k in range(max_iter):
                # Assemble stiffness
                K = self._assemble_stiffness()
                
                # Apply boundary conditions (zero rows/cols for fixed DOFs)
                K_reduced = K.copy()
                for dof in self.fixed_dofs:
                    K_reduced[dof, :] = 0
                    K_reduced[:, dof] = 0
                    K_reduced[dof, dof] = 1.0
                    
                # Calculate internal forces
                F_int = self._calculate_internal_forces(u)
                
                # Displacement control: adjust load factor
                residual = u_target - u[control_dof]
                
                # Arc-length style displacement control
                if abs(residual) < tol:
                    break
                    
                # Solve for displacement increment
                load_factor = residual * K[control_dof, control_dof]
                F_ext = self.load_pattern * load_factor
                
                R = F_ext - F_int
                
                for dof in self.fixed_dofs:
                    R[dof] = 0
                    
                try:
                    du = np.linalg.solve(K_reduced, R)
                except np.linalg.LinAlgError:
                    du = np.linalg.lstsq(K_reduced, R, rcond=None)[0]
                    
                u += du
                
                # Update elements
                for elem in self.elements:
                    elem.update_state(du)
                    if hasattr(elem, 'damage_index') and elem.damage_index > 1.0:
                        yielded.add(elem.id)
                        
            # Commit states
            for elem in self.elements:
                elem.commit_state()
                
            # Record response
            base_shear_hist.append(np.sum(F_int[self._get_base_dofs()]))
            roof_disp_hist.append(u[control_dof])
            
        return PushoverResult(
            base_shear=np.array(base_shear_hist),
            roof_disp=np.array(roof_disp_hist),
            story_drifts={},  # Would need more complex tracking
            yielded_elements=list(yielded),
            capacity_curve=(np.array(roof_disp_hist), np.array(base_shear_hist))
        )
    
    def run_force_control(
        self,
        max_force: float,
        n_steps: int = 50,
        tol: float = 1e-6,
        max_iter: int = 20
    ) -> PushoverResult:
        """
        Run force-controlled pushover analysis.
        
        Args:
            max_force: Maximum total lateral force (N)
            n_steps: Number of load steps
            tol: Convergence tolerance
            max_iter: Max Newton-Raphson iterations
            
        Returns:
            PushoverResult
        """
        if self.load_pattern is None:
            self.set_triangular_load_pattern()
            
        # Initialize
        u = np.zeros(self.ndof)
        base_shear_hist = []
        roof_disp_hist = []
        yielded = set()
        converged_steps = 0
        
        # Force increment
        d_force = max_force / n_steps
        
        # Find control DOF (top node X)
        control_dof = max(n.dof_indices[0] for n in self.nodes if n.dof_indices[0] >= 0)
        
        for step in range(n_steps):
            current_force = (step + 1) * d_force
            F_ext = self.load_pattern * current_force
            
            # Newton-Raphson iteration
            converged = False
            for k in range(max_iter):
                # Assemble stiffness
                K = self._assemble_stiffness()
                
                # Apply boundary conditions
                K_reduced = K.copy()
                for dof in self.fixed_dofs:
                    K_reduced[dof, :] = 0
                    K_reduced[:, dof] = 0
                    K_reduced[dof, dof] = 1.0
                
                # Calculate internal forces
                F_int = self._calculate_internal_forces(u)
                
                # Residual
                R = F_ext - F_int
                for dof in self.fixed_dofs:
                    R[dof] = 0
                
                # Check convergence
                norm_R = np.linalg.norm(R)
                if norm_R < tol:
                    converged = True
                    break
                
                # Solve
                try:
                    du = np.linalg.solve(K_reduced, R)
                except np.linalg.LinAlgError:
                    du = np.linalg.lstsq(K_reduced, R, rcond=None)[0]
                
                u += du
                
                # Update elements
                for elem in self.elements:
                    elem.update_state(du)
                    if hasattr(elem, 'damage_index') and elem.damage_index > 1.0:
                        yielded.add(elem.id)
            
            if converged:
                converged_steps += 1
                
            # Commit states
            for elem in self.elements:
                elem.commit_state()
            
            # Record response
            base_shear_hist.append(current_force)
            roof_disp_hist.append(u[control_dof])
            
            # Stop if structure has failed (large displacement)
            if abs(u[control_dof]) > 1.0:  # 1m displacement limit
                break
        
        return PushoverResult(
            base_shear=np.array(base_shear_hist),
            roof_disp=np.array(roof_disp_hist),
            story_drifts={},
            yielded_elements=list(yielded),
            capacity_curve=(np.array(roof_disp_hist), np.array(base_shear_hist))
        )
    
    def _assemble_stiffness(self) -> np.ndarray:
        """Assemble global stiffness matrix."""
        K = np.zeros((self.ndof, self.ndof))
        
        for elem in self.elements:
            k_el = elem.get_stiffness_matrix()
            indices = elem.get_element_dof_indices()
            
            for i, idx_i in enumerate(indices):
                if idx_i < 0:
                    continue
                for j, idx_j in enumerate(indices):
                    if idx_j < 0:
                        continue
                    K[idx_i, idx_j] += k_el[i, j]
                    
        return K
    
    def _calculate_internal_forces(self, u: np.ndarray) -> np.ndarray:
        """Calculate internal forces from current displacement."""
        F_int = np.zeros(self.ndof)
        
        for elem in self.elements:
            f_el = elem.update_state(u)
            indices = elem.get_element_dof_indices()
            
            for i, idx in enumerate(indices):
                if idx >= 0:
                    F_int[idx] += f_el[i]
                    
        return F_int
    
    def _get_base_dofs(self) -> List[int]:
        """Get DOF indices at base level."""
        base_dofs = []
        for node in self.nodes:
            if node.z == 0:
                for dof in node.dof_indices[:3]:  # Translational only
                    if dof >= 0:
                        base_dofs.append(dof)
        return base_dofs


def bilinearize_capacity_curve(
    disp: np.ndarray,
    shear: np.ndarray,
    method: str = 'equal_energy'
) -> Tuple[float, float, float, float]:
    """
    Convert capacity curve to bilinear representation.
    
    Args:
        disp: Displacement array
        shear: Base shear array
        method: 'equal_energy' or 'equal_area'
        
    Returns:
        (K_eff, Fy, dy, du) - Effective stiffness, yield force, yield disp, ultimate disp
    """
    if len(disp) < 2 or len(shear) < 2:
        return (0, 0, 0, 0)
        
    # Ultimate point
    du = disp[-1]
    Vu = shear[-1]
    
    # Initial stiffness (secant to 60% of peak)
    peak_idx = np.argmax(shear)
    target_V = 0.6 * shear[peak_idx]
    
    # Find point where shear crosses 60%
    cross_idx = np.searchsorted(shear[:peak_idx+1], target_V)
    if cross_idx > 0:
        K0 = shear[cross_idx] / disp[cross_idx]
    else:
        K0 = shear[1] / disp[1] if disp[1] > 0 else 1e6
        
    if method == 'equal_energy':
        # Equal energy method
        # Area under actual curve
        area_actual = np.trapezoid(shear, disp)
        
        # Solve for Fy such that bilinear area equals actual
        # Bilinear: triangle + rectangle
        # Fy*dy/2 + Fy*(du-dy) = area_actual
        # dy = Fy/K0
        # Fy*(Fy/K0)/2 + Fy*(du - Fy/K0) = area
        # Fy²/(2K0) + Fy*du - Fy²/K0 = area
        # -Fy²/(2K0) + Fy*du = area
        # Fy² - 2K0*du*Fy + 2K0*area = 0
        
        a = 1
        b = -2 * K0 * du
        c = 2 * K0 * area_actual
        
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            Fy = (-b - np.sqrt(discriminant)) / (2*a)
        else:
            Fy = Vu
            
        dy = Fy / K0 if K0 > 0 else 0
        
    else:  # equal_area (simpler)
        Fy = 0.9 * np.max(shear)
        dy = Fy / K0 if K0 > 0 else 0
        
    return (K0, Fy, dy, du)


def plot_capacity_curve(
    result: PushoverResult,
    bilinear: Tuple[float, float, float, float] = None,
    ax = None
):
    """Plot capacity curve with optional bilinear representation."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    disp, shear = result.capacity_curve
    
    # Main curve
    ax.plot(disp * 100, shear / 1000, 'b-', linewidth=2, label='Capacity Curve')
    
    # Bilinear
    if bilinear:
        K0, Fy, dy, du = bilinear
        ax.plot([0, dy*100, du*100], [0, Fy/1000, Fy/1000], 
                'r--', linewidth=1.5, label='Bilinear Idealization')
        
    ax.set_xlabel('Roof Displacement (cm)')
    ax.set_ylabel('Base Shear (kN)')
    ax.set_title('Pushover Capacity Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax
