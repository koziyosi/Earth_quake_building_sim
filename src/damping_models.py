"""
Advanced Damping Models.
Provides multiple damping formulations beyond Rayleigh damping.
"""
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class DampingType(Enum):
    """Available damping model types."""
    RAYLEIGH = "rayleigh"
    CAUGHEY = "caughey"
    MODAL = "modal"
    CONSTANT = "constant"
    WILSON_PENZIEN = "wilson_penzien"


class DampingModel:
    """Base class for damping models."""
    
    def __init__(self, M: np.ndarray, K: np.ndarray):
        """
        Initialize damping model.
        
        Args:
            M: Mass matrix
            K: Stiffness matrix
        """
        self.M = M
        self.K = K
        self.ndof = M.shape[0]
        
    def get_damping_matrix(self) -> np.ndarray:
        """Calculate and return the damping matrix."""
        raise NotImplementedError


class RayleighDamping(DampingModel):
    """
    Rayleigh (proportional) damping: C = a0*M + a1*K
    
    Standard approach where damping is proportional to mass and stiffness.
    Damping ratio varies with frequency; matches target at two frequencies.
    """
    
    def __init__(
        self,
        M: np.ndarray,
        K: np.ndarray,
        omega1: float,
        omega2: float,
        zeta: float = 0.05
    ):
        """
        Args:
            omega1: First target frequency (rad/s)
            omega2: Second target frequency (rad/s)
            zeta: Target damping ratio
        """
        super().__init__(M, K)
        self.omega1 = omega1
        self.omega2 = omega2
        self.zeta = zeta
        
        # Calculate coefficients
        mat = np.array([[1/omega1, omega1], [1/omega2, omega2]])
        vec = np.array([2*zeta, 2*zeta])
        self.a0, self.a1 = np.linalg.solve(mat, vec)
        
    def get_damping_matrix(self) -> np.ndarray:
        return self.a0 * self.M + self.a1 * self.K


class CaugheyDamping(DampingModel):
    """
    Caughey damping (Extended Rayleigh).
    C = M * Sum(a_k * (M^-1 * K)^k)
    
    Allows matching damping ratio at more than two frequencies.
    """
    
    def __init__(
        self,
        M: np.ndarray,
        K: np.ndarray,
        target_frequencies: List[float],
        target_zetas: List[float]
    ):
        """
        Args:
            target_frequencies: Natural frequencies to match (rad/s)
            target_zetas: Target damping ratios at those frequencies
        """
        super().__init__(M, K)
        self.target_frequencies = target_frequencies
        self.target_zetas = target_zetas
        
        assert len(target_frequencies) == len(target_zetas), \
            "Frequencies and zetas must have same length"
        
        self.coefficients = self._calculate_coefficients()
        
    def _calculate_coefficients(self) -> np.ndarray:
        """Calculate Caughey coefficients."""
        n = len(self.target_frequencies)
        
        # Build system of equations
        # zeta_j = 0.5 * sum(a_k * omega_j^(2k-1))
        
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for j, omega_j in enumerate(self.target_frequencies):
            for k in range(n):
                A[j, k] = 0.5 * omega_j**(2*k - 1)
            b[j] = self.target_zetas[j]
        
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fall back to least squares
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
        return coeffs
        
    def get_damping_matrix(self) -> np.ndarray:
        """Calculate Caughey damping matrix."""
        try:
            M_inv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(self.M)
            
        B = M_inv @ self.K  # M^-1 * K
        
        C = np.zeros_like(self.M)
        B_power = np.eye(self.ndof)  # (M^-1 * K)^0 = I
        
        for k, a_k in enumerate(self.coefficients):
            if k > 0:
                B_power = B_power @ B  # (M^-1 * K)^k
            C += a_k * B_power
        
        return self.M @ C


class ModalDamping(DampingModel):
    """
    Modal damping - specify damping ratio for each mode.
    
    C = M * Phi * diag(2*zeta_i*omega_i) * Phi^T * M
    
    Most accurate approach when mode shapes are available.
    """
    
    def __init__(
        self,
        M: np.ndarray,
        K: np.ndarray,
        modal_zetas: Optional[List[float]] = None,
        default_zeta: float = 0.05
    ):
        """
        Args:
            modal_zetas: Damping ratio for each mode (optional)
            default_zeta: Default damping ratio if modal_zetas not specified
        """
        super().__init__(M, K)
        self.modal_zetas = modal_zetas
        self.default_zeta = default_zeta
        
    def get_damping_matrix(self) -> np.ndarray:
        """Calculate modal damping matrix."""
        # Eigenvalue analysis
        try:
            M_inv = np.linalg.inv(self.M)
        except:
            M_inv = np.linalg.pinv(self.M)
            
        eigenvalues, eigenvectors = np.linalg.eig(M_inv @ self.K)
        
        # Clean up
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Natural frequencies
        omega = np.sqrt(np.maximum(eigenvalues, 0))
        
        # Damping ratios for each mode
        n_modes = len(omega)
        if self.modal_zetas is not None:
            zetas = np.zeros(n_modes)
            for i, z in enumerate(self.modal_zetas):
                if i < n_modes:
                    zetas[i] = z
                else:
                    break
            # Fill remaining with default
            zetas[len(self.modal_zetas):] = self.default_zeta
        else:
            zetas = np.full(n_modes, self.default_zeta)
        
        # Modal damping matrix in modal coordinates
        c_modal = np.diag(2 * zetas * omega)
        
        # Transform back to physical coordinates
        Phi = eigenvectors
        # C = M * Phi * c_modal * Phi^T * M^-1 * M = M * Phi * c_modal * Phi^T
        
        # Normalize mode shapes by mass
        for i in range(n_modes):
            m_i = Phi[:, i].T @ self.M @ Phi[:, i]
            if m_i > 1e-10:
                Phi[:, i] /= np.sqrt(m_i)
        
        C = self.M @ Phi @ c_modal @ Phi.T @ self.M
        
        return C


class ConstantDamping(DampingModel):
    """
    Constant (frequency-independent) damping.
    
    Useful for engineering estimates where constant damping ratio
    is desired across all frequencies.
    """
    
    def __init__(
        self,
        M: np.ndarray,
        K: np.ndarray,
        zeta: float = 0.05,
        omega_ref: float = 10.0
    ):
        """
        Args:
            zeta: Constant damping ratio
            omega_ref: Reference frequency for scaling (rad/s)
        """
        super().__init__(M, K)
        self.zeta = zeta
        self.omega_ref = omega_ref
        
    def get_damping_matrix(self) -> np.ndarray:
        """Calculate constant damping matrix approximation."""
        # C ≈ 2 * zeta * omega_ref * M (simplified)
        # This gives constant damping at omega_ref
        return 2 * self.zeta * self.omega_ref * self.M


def create_damping_model(
    damping_type: DampingType,
    M: np.ndarray,
    K: np.ndarray,
    **kwargs
) -> DampingModel:
    """
    Factory function to create damping models.
    
    Args:
        damping_type: Type of damping model
        M: Mass matrix
        K: Stiffness matrix
        **kwargs: Model-specific parameters
        
    Returns:
        DampingModel instance
    """
    if damping_type == DampingType.RAYLEIGH:
        return RayleighDamping(
            M, K,
            omega1=kwargs.get('omega1', 5.0),
            omega2=kwargs.get('omega2', 30.0),
            zeta=kwargs.get('zeta', 0.05)
        )
    elif damping_type == DampingType.CAUGHEY:
        return CaugheyDamping(
            M, K,
            target_frequencies=kwargs.get('frequencies', [5, 15, 30]),
            target_zetas=kwargs.get('zetas', [0.05, 0.05, 0.05])
        )
    elif damping_type == DampingType.MODAL:
        return ModalDamping(
            M, K,
            modal_zetas=kwargs.get('modal_zetas'),
            default_zeta=kwargs.get('zeta', 0.05)
        )
    elif damping_type == DampingType.CONSTANT:
        return ConstantDamping(
            M, K,
            zeta=kwargs.get('zeta', 0.05),
            omega_ref=kwargs.get('omega_ref', 10.0)
        )
    else:
        raise ValueError(f"Unknown damping type: {damping_type}")
