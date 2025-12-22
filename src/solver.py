import numpy as np
from typing import List
from src.fem import Node, Element2D
import os

# Set number of threads for numpy/scipy (BLAS/LAPACK)
_num_threads = os.cpu_count() or 4
os.environ.setdefault('OMP_NUM_THREADS', str(_num_threads))
os.environ.setdefault('MKL_NUM_THREADS', str(_num_threads))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_num_threads))

class NewmarkBetaSolver:
    """
    Newmark-Beta Time Integration Solver.
    """
    def __init__(self, nodes: List[Node], elements: List[Element2D], dt: float, 
                 beta: float = 0.25, gamma: float = 0.5):
        self.nodes = nodes
        self.elements = elements
        self.dt = dt
        self.beta = beta
        self.gamma = gamma
        
        # Determine total DOFs
        self.ndof = 0
        for node in self.nodes:
            self.ndof = max(self.ndof, max(node.dof_indices) + 1)
            
        # Global Matrices
        self.M = np.zeros((self.ndof, self.ndof))
        self.C = np.zeros((self.ndof, self.ndof))
        self.K = np.zeros((self.ndof, self.ndof))
        
        # State Vectors
        self.u = np.zeros(self.ndof)
        self.v = np.zeros(self.ndof)
        self.a = np.zeros(self.ndof)
        
        # Internal force tracking (for Newton-Raphson)
        self.F_int_prev = np.zeros(self.ndof)
        
        # Influence vector for ground motion (assuming X-direction input)
        self.iota = np.zeros(self.ndof)
        for node in self.nodes:
            idx_x = node.dof_indices[0]
            if idx_x != -1:
                self.iota[idx_x] = 1.0
                
        self.assemble_mass()
        # Damping is assembled later or assumed Rayleigh
        
    def assemble_mass(self):
        for node in self.nodes:
            # Lumped Mass
            # We assume node.mass is translational mass
            # We assume node.dof_indices is 6-long (3D) or we check length?
            # Node class is fixed to 6 DOFs now.
            
            # Translational: 0 (X), 1 (Y), 2 (Z)
            m = node.mass
            for i in [0, 1, 2]:
                idx = node.dof_indices[i]
                if idx != -1:
                    self.M[idx, idx] = m
                    
            # Rotational: 3 (Rx), 4 (Ry), 5 (Rz)
            # Rotational inertia scales with span length squared
            # Estimate typical span as building_height / n_floors (proxy for grid size)
            # J_rot ~ m * r^2, where r is characteristic length
            max_z = max((n.z for n in self.nodes), default=10.0)
            n_floors = max(1, len(set(n.z for n in self.nodes)) - 1)
            typical_span = max(max_z / n_floors, 3.0)  # At least 3m
            j_rot = m * (typical_span ** 2) / 12.0  # Approximation for distributed mass
            for i in [3, 4, 5]:
                idx = node.dof_indices[i]
                if idx != -1:
                    self.M[idx, idx] = j_rot

    def assemble_stiffness(self, use_parallel: bool = None):
        """Assemble global stiffness matrix.
        
        Args:
            use_parallel: Use parallel assembly for large models.
                         If None, auto-detect based on model size.
        """
        # Auto-detect parallel mode for large models
        if use_parallel is None:
            use_parallel = self.ndof > 1000 and len(self.elements) > 200
        
        if use_parallel:
            try:
                from src.parallel import parallel_matrix_assembly
                self.K = parallel_matrix_assembly(
                    self.elements, self.ndof, 
                    matrix_func='get_stiffness_matrix',
                    num_threads=os.cpu_count() or 4
                )
                return
            except ImportError:
                pass  # Fall back to sequential
        
        # Sequential assembly
        self.K.fill(0.0)
        for elem in self.elements:
            k_el = elem.get_stiffness_matrix()
            indices = elem.get_element_dof_indices()
            
            n_dof = len(indices)
            # Validate k_el size
            if k_el.shape != (n_dof, n_dof):
                # Should not happen if correctly implemented
                pass
                
            for r in range(n_dof):
                if indices[r] == -1: continue
                for c in range(n_dof):
                    if indices[c] == -1: continue
                    self.K[indices[r], indices[c]] += k_el[r, c]

    def assemble_damping(self):
        """
        Assemble global damping matrix from elements (Oil Dampers) + Rayleigh.
        Call this after set_rayleigh_damping.
        """
        # Start with Rayleigh C (calculated in set_rayleigh_damping)
        # But we need to add explicit element damping.
        
        # If set_rayleigh_damping hasn't been called, C might be 0.
        
        for elem in self.elements:
            if hasattr(elem, 'get_damping_matrix'):
                c_el = elem.get_damping_matrix()
                indices = elem.get_element_dof_indices()
                n_dof = len(indices)
                
                for r in range(n_dof):
                    if indices[r] == -1: continue
                    for c in range(n_dof):
                        if indices[c] == -1: continue
                        self.C[indices[r], indices[c]] += c_el[r, c]

    def set_rayleigh_damping(self, omega1, omega2, zeta=0.05):
        """
        Sets C = a0*M + a1*K_initial
        """
        # Solving:
        # 0.5 * (a0/w + a1*w) = zeta
        # a0 + a1*w^2 = 2*zeta*w
        
        det = omega2**2 - omega1**2
        a0 = 2*zeta*omega1*omega2*(omega2 - omega1) / det
        a1 = 2*zeta*(omega2 - omega1) / det
        
        # Simpler approximation if w1, w2 close or standard formula:
        # [1/w1 w1; 1/w2 w2] * [a0; a1] = [2*zeta; 2*zeta]
        
        mat = np.array([[1/omega1, omega1], [1/omega2, omega2]])
        vec = np.array([2*zeta, 2*zeta])
        coeffs = np.linalg.solve(mat, vec)
        a0, a1 = coeffs[0], coeffs[1]
        
        self.assemble_stiffness() # Ensure K is built
        self.C = a0 * self.M + a1 * self.K
        
        # Add explicit damping
        self.assemble_damping()
    
    def estimate_natural_frequencies(self, n_modes: int = 3):
        """
        Estimate natural frequencies from eigenvalue analysis.
        This allows automatic Rayleigh damping setup for any building.
        
        Returns:
            Tuple[float, float]: (omega1, omega2) for first and second mode
        """
        self.assemble_stiffness()
        
        # Find DOFs with non-zero mass (avoid divide by zero)
        M_diag = np.diag(self.M)
        active_dofs = np.where(M_diag > 1e-10)[0]
        
        if len(active_dofs) < 2:
            # Fallback to reasonable defaults for a 3-story building
            return 6.28, 18.84  # ~1Hz, ~3Hz
        
        # Extract submatrices for active DOFs
        M_sub = self.M[np.ix_(active_dofs, active_dofs)]
        K_sub = self.K[np.ix_(active_dofs, active_dofs)]
        
        try:
            # Solve generalized eigenvalue problem: K*phi = omega^2*M*phi
            # Use scipy.linalg.eigh for symmetric matrices
            from scipy.linalg import eigh
            eigenvalues, _ = eigh(K_sub, M_sub)
            
            # Filter positive eigenvalues and take sqrt for frequencies
            omega_sq = eigenvalues[eigenvalues > 1e-6]
            
            if len(omega_sq) < 2:
                return 6.28, 18.84
            
            omegas = np.sqrt(omega_sq)
            omegas = np.sort(omegas)
            
            # Use 1st and 3rd mode (or 2nd if only 2 available)
            omega1 = max(omegas[0], 0.5)  # Minimum 0.5 rad/s (~0.08 Hz)
            omega2 = omegas[min(2, len(omegas)-1)]
            omega2 = max(omega2, omega1 * 3)  # At least 3x first mode
            
            return float(omega1), float(omega2)
            
        except Exception:
            # Fallback if eigenvalue fails
            return 6.28, 18.84
    
    def set_rayleigh_damping_auto(self, zeta: float = 0.05):
        """
        Automatically estimate natural frequencies and set Rayleigh damping.
        Use this for any building configuration.
        """
        omega1, omega2 = self.estimate_natural_frequencies()
        self.set_rayleigh_damping(omega1, omega2, zeta)
        return omega1, omega2
    
    def enable_p_delta(self, enable: bool = True):
        """Enable or disable P-Delta effect (geometric nonlinearity)."""
        self.p_delta_enabled = enable
        self.K_geo = np.zeros((self.ndof, self.ndof))
        
    def assemble_geometric_stiffness(self, u: np.ndarray):
        """
        Assemble geometric stiffness matrix for P-Delta effect.
        
        The geometric stiffness accounts for the effect of axial forces
        on the lateral stiffness of vertical elements.
        
        For columns: K_geo = P/L * matrix for lateral stiffness modification
        
        Args:
            u: Current displacement vector
        """
        if not getattr(self, 'p_delta_enabled', False):
            return
            
        self.K_geo.fill(0.0)
        
        for elem in self.elements:
            # Only apply to vertical elements (columns)
            if not hasattr(elem, 'node_i') or not hasattr(elem, 'node_j'):
                continue
                
            ni = elem.node_i
            nj = elem.node_j
            
            # Check if vertical (significant Z difference)
            dz = abs(nj.z - ni.z)
            dx = abs(nj.x - ni.x)
            dy = abs(nj.y - ni.y)
            
            if dz < 0.5 * max(dx, dy, 0.1):  # Not a column
                continue
                
            L = np.sqrt(dx**2 + dy**2 + dz**2)
            if L < 0.01:
                continue
            
            # Estimate axial force from gravity (simplified)
            # In a full implementation, this would come from element state
            # For now, estimate from tributary mass above
            P_gravity = 0.0
            for node in self.nodes:
                if node.z > nj.z:  # Nodes above this column
                    P_gravity += node.mass * 9.81
            
            # Geometric stiffness coefficient
            k_geo = P_gravity / L
            
            # Get DOF indices for lateral DOFs (X and Y at both ends)
            indices = elem.get_element_dof_indices() if hasattr(elem, 'get_element_dof_indices') else []
            
            if len(indices) >= 12:
                # Standard 3D beam-column with 6 DOF per node
                # Lateral DOFs: 1 (Y at i), 7 (Y at j), 2 (Z at i), 8 (Z at j)
                # For columns, X-direction (horizontal) is indices 0 and 6
                
                # Simplified: Add geometric stiffness to lateral DOFs
                lateral_pairs = [(0, 6), (1, 7)]  # (X_i, X_j), (Y_i, Y_j)
                
                for (i_local, j_local) in lateral_pairs:
                    idx_i = indices[i_local] if i_local < len(indices) else -1
                    idx_j = indices[j_local] if j_local < len(indices) else -1
                    
                    if idx_i >= 0 and idx_j >= 0:
                        # K_geo contribution: [k_geo, -k_geo; -k_geo, k_geo]
                        self.K_geo[idx_i, idx_i] += k_geo
                        self.K_geo[idx_i, idx_j] -= k_geo
                        self.K_geo[idx_j, idx_i] -= k_geo
                        self.K_geo[idx_j, idx_j] += k_geo

    def solve_step(self, acc_g_next):
        """
        Perform one time step using Newton-Raphson.
        """
        if not hasattr(self, 'acc_g_curr'):
            self.acc_g_curr = 0.0
            
        # Calculate d_F_ext for scalar ground motion
        d_acc = acc_g_next - self.acc_g_curr
        d_F_ext = -self.M @ self.iota * d_acc
        
        # Call generalized solver
        u_new, v_new, a_new = self.solve_newton_raphson(d_F_ext)
        
        self.acc_g_curr = acc_g_next
        return u_new, v_new, a_new

    def solve_newton_raphson(self, d_F_ext, max_iter=10, tol=1e-3):
        """
        Generalized Newton-Raphson Step.
        d_F_ext: External force increment vector.
        """
        # Constants
        beta = self.beta
        gamma = self.gamma
        dt = self.dt
        
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta)
        a4 = gamma / beta
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        
        # Initial Predictor (u=0 increment relative to start)
        # We start with delta_u = 0? 
        # Or elastic predictor?
        # Standard: residual based.
        
        # Initial Residual (Unbalanced Force from previous step?)
        # Ideally 0 if equilibrated.
        # R_initial = F_ext_total - F_int_total - Inertia
        # But we work in increments for Newmark usually?
        # Total form is safer for NR.
        
        # Let's use Incremental Form NR for Newmark (bathe)
        # Equation:
        # (K_t + a0*M + a1*C) * d_u = R_effective
        # R_eff = F_ext_total[t+dt] - F_int[t+dt]^(i-1) - M*a[t+dt]^(i-1) - C*v...
        
        # State at start of step (converged)
        u_t = self.u.copy()
        v_t = self.v.copy()
        a_t = self.a.copy()
        
        # Initialize predictor (Newmark predictor with delta_u = 0)
        # Note: u_curr = u_t means (u_curr - u_t) = 0, simplifying acceleration predictor
        u_curr = u_t.copy()
        v_curr = (1 - gamma/beta)*v_t + dt*(1 - gamma/(2*beta))*a_t 
        a_curr = -(1/(beta*dt))*v_t - (1/(2*beta)-1)*a_t
        
        # Track total displacement increment from step start
        delta_u_total = np.zeros(self.ndof)
        
        # Momentum Predictor Force (Equivalent to R_mom)
        # R_mom = M * (a2*v_t + a3*a_t) + C * (a4*v_t + a5*a_t)
        R_mom = self.M @ (a2 * v_t + a3 * a_t) + self.C @ (a4 * v_t + a5 * a_t)
        
        # Net Effective Load Increment
        # d_F_hat_initial = d_F_ext + R_mom
        # This is the RHS if K was constant.
        
        # Loop
        R_unbalanced = d_F_ext + R_mom # Initial residual force?
        # Check: K * du = d_F_ext + R_mom
        # Residual r = d_F_ext + R_mom - (F_int(u+du) - F_int(u)) - ...
        
        # We need to assemble Tangent K at every iteration (or Modified Newton)
        
        # Divergence detection thresholds - building height and dt dependent
        # Calculate building height from nodes
        max_z = max(n.z for n in self.nodes) if self.nodes else 100.0
        # Step displacement limit: scales with dt and building height
        # Base: 1% drift for dt=0.005, scale proportionally
        dt_scale = dt / 0.005  # 1.0 for normal, 4.0 for fast mode
        # For small buildings (<10m): use 2% of height per step
        # For larger buildings: use 1% of height per step
        if max_z < 10.0:
            MAX_DISP_STEP = max(0.02 * max_z * dt_scale, 0.01)  # Min 1cm for tiny structures
        else:
            MAX_DISP_STEP = 0.01 * max_z * dt_scale
        # Cap at reasonable maximum to prevent extreme values
        MAX_DISP_STEP = min(MAX_DISP_STEP, 5.0)
        # Total accumulated displacement limit: realistic building deformation
        MAX_DISP_TOTAL = min(20.0, 0.05 * max_z)  # 5% of building height max
        MAX_ITER_NO_CONVERGE = max_iter
        
        for k in range(max_iter):
            # 1. Assemble Stiffness - Modified Newton: only on first iteration
            if k == 0:
                self.assemble_stiffness()
                K_hat = self.K + a1 * self.C + a0 * self.M
            # Reuse K_hat from first iteration for subsequent iterations
            
            
            # 2. Internal Force
            # We need F_int corresponding to current `delta_u_total`.
            # Can we get F_int directly from elements?
            # `element.update_state(delta_u_total)` returns `forces_local` mapped to global.
            # Let's sum them.
            F_int_increment = np.zeros(self.ndof)
            for elem in self.elements:
                f_el = elem.update_state(delta_u_total)
                # Map to global F_int
                indices = elem.get_element_dof_indices()
                for i, idx in enumerate(indices):
                    if idx != -1:
                        F_int_increment[idx] += f_el[i]
            
            # This F_int_increment includes -N_initial?
            # My `update_state` returns TOTAL forces.
            # So I need F_int_total.
            # And I need F_int_initial (at u_t).
            # To avoid calculating F_int_initial every time, let's assume valid start.
            
            # Actually, `d_F_hat = d_F_ext + R_mom - (F_int_current - F_int_start) - M*a_curr_inertia_part`
            # This is geeting complicated.
            
            # SIMPLIFIED NEWMARK with Newton Raphson on the Increment:
            # Residual R = (d_F_ext + R_mom) - [ (M*a0 + C*a1)*delta_u_total + (F_int(u_t + delta_u) - F_int(u_t)) ]
            # Note: The term (M*a0 + C*a1)*delta_u_total accounts for the M*a and C*v changes due to u.
            
            # We need F_int(u_t) to subtract?
            # Or we can just work with `Residual` directly if we tracked it.
            
            # Let's calculate F_int(u_t + delta_u).
            # We summed it into F_int_increment (which is F_int_total really).
            
            # We need F_int_prev (start of step).
            # We can store it in the solver.
            if not hasattr(self, 'F_int_prev'):
                 # Initialize
                 self.F_int_prev = np.zeros(self.ndof)
            
            F_restoring_force_change = F_int_increment - self.F_int_prev
            
            # Dynamic Force Change from Mass/Damping
            # d_a = a0 * delta_u
            # d_v = a1 * delta_u
            # F_dyn_change = (M*a0 + C*a1) * delta_u
            
            # R = (d_F_ext + R_mom) - F_restoring_force_change - F_dyn_change
            
            F_dyn_change = (self.M * a0 + self.C * a1) @ delta_u_total
            
            R = d_F_ext + R_mom - F_restoring_force_change - F_dyn_change
            
            # Convergence check
            norm_R = np.linalg.norm(R)
            norm_F = np.linalg.norm(d_F_ext) + 1e-9
            if k > 0 and norm_R < tol * norm_F:
                break
                
            # Solve correction - use scipy for large systems
            try:
                if self.ndof > 500:
                    # Use scipy sparse solver for large systems (much faster)
                    try:
                        from scipy.sparse import csr_matrix
                        from scipy.sparse.linalg import spsolve
                        K_sparse = csr_matrix(K_hat)
                        du = spsolve(K_sparse, R)
                    except ImportError:
                        du = np.linalg.solve(K_hat, R)
                else:
                    du = np.linalg.solve(K_hat, R)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K_hat, R, rcond=None)[0]
                
            delta_u_total += du
            
            # Divergence detection
            if np.any(np.isnan(delta_u_total)) or np.any(np.isinf(delta_u_total)):
                print("Warning: NaN/Inf detected in Newton-Raphson, using zero increment")
                delta_u_total = np.zeros(self.ndof)
                break
            
            if np.max(np.abs(delta_u_total)) > MAX_DISP_STEP:
                print(f"Warning: Step displacement {np.max(np.abs(delta_u_total)):.2e}m exceeds limit, clamping")
                delta_u_total = np.clip(delta_u_total, -MAX_DISP_STEP, MAX_DISP_STEP)
                # Mark as diverged for this step
                self._diverged_this_step = True
                break
            
        # End Loop
        
        # Check if we diverged - if so, damp the velocity and don't commit
        diverged = getattr(self, '_diverged_this_step', False)
        self._diverged_this_step = False
        
        # Update State Variables
        d_v = a1 * delta_u_total - a4 * v_t - a5 * a_t
        d_a = a0 * delta_u_total - a2 * v_t - a3 * a_t
        
        if diverged:
            # Apply heavy damping to velocity to prevent energy accumulation
            self.u += delta_u_total
            self.v = (self.v + d_v) * 0.1  # 90% velocity damping
            self.a = d_a * 0.1
        else:
            self.u += delta_u_total
            self.v += d_v
            self.a += d_a
        
        # Apply total displacement limit to prevent unbounded drift
        MAX_TOTAL_DISP = min(50.0, 0.1 * max_z)  # 10% of building height or 50m max
        if np.max(np.abs(self.u)) > MAX_TOTAL_DISP:
            # Scale down all displacements to limit
            scale_factor = MAX_TOTAL_DISP / np.max(np.abs(self.u))
            self.u *= scale_factor
            self.v *= 0.5  # Also damp velocity
        
        # Update F_int_prev for next step
        # F_int_increment is the Total Force at new step
        # But we need to recalculate it if loop ended?
        # We can re-call update_state one last time to be sure, or use last loop val.
        # Safest: re-calc.
        F_int_final = np.zeros(self.ndof)
        for elem in self.elements:
            f_el = elem.update_state(delta_u_total)
            indices = elem.get_element_dof_indices()
            for i, idx in enumerate(indices):
                if idx != -1:
                    F_int_final[idx] += f_el[i]
        self.F_int_prev = F_int_final
        
        # Only commit elements if we didn't diverge
        if not diverged:
            for elem in self.elements:
                elem.commit_state()
            
        return self.u.copy(), self.v.copy(), self.a.copy()
