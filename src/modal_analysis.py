"""
Modal Analysis module.
Performs eigenvalue analysis to calculate natural frequencies and mode shapes.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModeResult:
    """Result for a single vibration mode."""
    mode_number: int
    frequency: float          # Hz
    period: float             # seconds
    omega: float              # rad/s
    mode_shape: np.ndarray    # Mode shape vector
    modal_mass: float         # Modal mass
    modal_participation: float  # Modal participation factor (X direction)


@dataclass
class ModalAnalysisResult:
    """Complete modal analysis result."""
    modes: List[ModeResult]
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    total_mass: float
    cumulative_mass_participation: List[float]


def perform_modal_analysis(
    M: np.ndarray,
    K: np.ndarray,
    n_modes: int = 10,
    normalize: str = 'mass'  # 'mass' or 'max'
) -> ModalAnalysisResult:
    """
    Perform eigenvalue analysis to find natural frequencies and mode shapes.
    
    Solves: K * phi = omega^2 * M * phi
    
    Args:
        M: Mass matrix (n x n)
        K: Stiffness matrix (n x n)
        n_modes: Number of modes to extract
        normalize: Normalization method ('mass' or 'max')
        
    Returns:
        ModalAnalysisResult with all mode information
    """
    n_dof = M.shape[0]
    n_modes = min(n_modes, n_dof)
    
    # Check for singularity
    if np.linalg.det(M) < 1e-20:
        # Make M invertible by adding small diagonal
        M = M + np.eye(n_dof) * 1e-6
    
    # Solve generalized eigenvalue problem
    # K * phi = lambda * M * phi
    # M^(-1) * K * phi = lambda * phi
    try:
        M_inv = np.linalg.inv(M)
        A = M_inv @ K
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        # Use least squares approach
        eigenvalues, eigenvectors = np.linalg.eig(K)
        eigenvalues = np.real(eigenvalues)
    
    # Clean up eigenvalues (remove imaginary, sort)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort by eigenvalue (ascending frequency)
    sort_idx = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Filter out zero/negative eigenvalues (rigid body modes)
    valid_idx = np.where(eigenvalues > 1e-10)[0]
    eigenvalues = eigenvalues[valid_idx]
    eigenvectors = eigenvectors[:, valid_idx]
    
    # Limit to n_modes
    n_available = min(n_modes, len(eigenvalues))
    eigenvalues = eigenvalues[:n_available]
    eigenvectors = eigenvectors[:, :n_available]
    
    # Calculate natural frequencies
    omegas = np.sqrt(np.abs(eigenvalues))
    frequencies = omegas / (2 * np.pi)
    periods = 1.0 / np.where(frequencies > 1e-10, frequencies, 1e-10)
    
    # Normalize mode shapes and calculate modal properties
    modes = []
    total_mass = np.trace(M)  # Approximation for lumped mass
    
    # Influence vector for X-direction (assuming DOF 0, 6, 12, ... are X)
    iota_x = np.zeros(n_dof)
    for i in range(0, n_dof, 6):  # Every 6 DOFs, first is X
        if i < n_dof:
            iota_x[i] = 1.0
    
    cumulative_participation = []
    running_mass = 0.0
    
    for i in range(n_available):
        phi = eigenvectors[:, i]
        
        # Normalize by mass
        modal_mass = phi @ M @ phi
        if normalize == 'mass' and modal_mass > 1e-20:
            phi = phi / np.sqrt(modal_mass)
            modal_mass = 1.0
        elif normalize == 'max':
            max_val = np.max(np.abs(phi))
            if max_val > 1e-20:
                phi = phi / max_val
            modal_mass = phi @ M @ phi
        
        # Modal participation factor
        L = phi @ M @ iota_x
        M_star = phi @ M @ phi
        if M_star > 1e-20:
            gamma = L / M_star
            effective_mass = L**2 / M_star
            participation_ratio = effective_mass / total_mass if total_mass > 0 else 0
        else:
            gamma = 0
            participation_ratio = 0
        
        running_mass += participation_ratio
        cumulative_participation.append(running_mass)
        
        mode = ModeResult(
            mode_number=i + 1,
            frequency=frequencies[i],
            period=periods[i],
            omega=omegas[i],
            mode_shape=phi,
            modal_mass=modal_mass,
            modal_participation=gamma
        )
        modes.append(mode)
    
    return ModalAnalysisResult(
        modes=modes,
        mass_matrix=M,
        stiffness_matrix=K,
        total_mass=total_mass,
        cumulative_mass_participation=cumulative_participation
    )


def plot_mode_shapes(
    result: ModalAnalysisResult,
    n_modes_plot: int = 4,
    node_coords: Optional[List[Tuple[float, float, float]]] = None
):
    """
    Plot mode shapes (requires matplotlib).
    
    Args:
        result: ModalAnalysisResult object
        n_modes_plot: Number of modes to plot
        node_coords: Optional list of (x, y, z) node coordinates
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    n_plot = min(n_modes_plot, len(result.modes))
    
    fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 6))
    if n_plot == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes[:n_plot]):
        mode = result.modes[i]
        phi = mode.mode_shape
        
        # Simple plot - every 6 DOFs
        n_nodes = len(phi) // 6 if len(phi) >= 6 else len(phi)
        
        # Extract X displacement component
        x_disp = phi[0::6][:n_nodes] if len(phi) >= 6 else phi
        
        ax.barh(range(len(x_disp)), x_disp)
        ax.set_title(f"Mode {mode.mode_number}\nT={mode.period:.3f}s\nf={mode.frequency:.2f}Hz")
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Mode Shape (X)')
        ax.set_ylabel('Node Level')
    
    plt.tight_layout()
    return fig


def get_modal_summary(result: ModalAnalysisResult) -> str:
    """
    Get a text summary of modal analysis results.
    
    Args:
        result: ModalAnalysisResult object
        
    Returns:
        Formatted string summary
    """
    lines = ["=" * 50]
    lines.append("MODAL ANALYSIS SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Total Mass: {result.total_mass:.2f} kg")
    lines.append("")
    lines.append(f"{'Mode':>4} {'Period(s)':>10} {'Freq(Hz)':>10} {'Cum.Mass%':>10}")
    lines.append("-" * 40)
    
    for i, mode in enumerate(result.modes):
        cum_mass = result.cumulative_mass_participation[i] * 100 if i < len(result.cumulative_mass_participation) else 0
        lines.append(f"{mode.mode_number:>4} {mode.period:>10.4f} {mode.frequency:>10.3f} {cum_mass:>10.1f}")
    
    lines.append("=" * 50)
    return "\n".join(lines)
