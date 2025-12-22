"""
Response Spectrum Analysis module.
Calculates acceleration, velocity, and displacement response spectra.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResponseSpectrumResult:
    """Container for response spectrum results."""
    periods: np.ndarray      # Natural periods (s)
    Sa: np.ndarray           # Spectral acceleration (m/s²)
    Sv: np.ndarray           # Spectral velocity (m/s)
    Sd: np.ndarray           # Spectral displacement (m)
    PSa: np.ndarray          # Pseudo spectral acceleration (m/s²)
    PSv: np.ndarray          # Pseudo spectral velocity (m/s)


def calculate_response_spectrum(
    time: np.ndarray,
    acceleration: np.ndarray,
    periods: Optional[np.ndarray] = None,
    damping_ratio: float = 0.05
) -> ResponseSpectrumResult:
    """
    Calculate response spectrum for a given ground motion.
    
    Uses Newmark-beta method (average acceleration) for each SDOF system.
    
    Args:
        time: Time array (s)
        acceleration: Ground acceleration array (m/s²)
        periods: Array of natural periods to calculate (s). 
                 If None, uses default range 0.01-10s.
        damping_ratio: Damping ratio (default 5%)
        
    Returns:
        ResponseSpectrumResult containing all spectral values
    """
    if periods is None:
        # Default period range: 0.01 to 10 seconds, log-spaced
        periods = np.logspace(-2, 1, 100)
    
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    n_steps = len(acceleration)
    
    Sa = np.zeros(len(periods))
    Sv = np.zeros(len(periods))
    Sd = np.zeros(len(periods))
    
    for i, T in enumerate(periods):
        if T <= 0:
            continue
            
        omega = 2 * np.pi / T
        omega_d = omega * np.sqrt(1 - damping_ratio**2)
        
        # Newmark-beta parameters (average acceleration)
        beta = 0.25
        gamma = 0.5
        
        # SDOF system integration
        u, v, a = _integrate_sdof(
            acceleration, dt, omega, damping_ratio, beta, gamma
        )
        
        # Maximum responses
        Sd[i] = np.max(np.abs(u))
        Sv[i] = np.max(np.abs(v))
        Sa[i] = np.max(np.abs(a + acceleration))  # Absolute acceleration
    
    # Pseudo spectral values
    omega_vec = 2 * np.pi / np.where(periods > 0, periods, 1e-10)
    PSv = omega_vec * Sd
    PSa = omega_vec**2 * Sd
    
    return ResponseSpectrumResult(
        periods=periods,
        Sa=Sa,
        Sv=Sv, 
        Sd=Sd,
        PSa=PSa,
        PSv=PSv
    )


def _integrate_sdof(
    acc_g: np.ndarray,
    dt: float,
    omega: float,
    zeta: float,
    beta: float = 0.25,
    gamma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate SDOF equation using Newmark-beta method.
    
    Equation: u'' + 2*zeta*omega*u' + omega^2*u = -acc_g
    
    Returns:
        Tuple of (displacement, velocity, acceleration) arrays
    """
    n = len(acc_g)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    
    # Initial acceleration
    a[0] = -acc_g[0]
    
    # Integration constants
    c = 2 * zeta * omega
    k = omega**2
    
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2 * beta) - 1.0)
    
    # Effective stiffness
    k_eff = k + a0 + a1 * c
    
    for i in range(n - 1):
        # Effective load
        p_eff = -acc_g[i+1] + a0 * u[i] + a2 * v[i] + a3 * a[i]
        p_eff += c * (a1 * u[i] + a4 * v[i] + a5 * a[i])
        
        # Solve for displacement
        u[i+1] = p_eff / k_eff
        
        # Update velocity and acceleration
        a[i+1] = a0 * (u[i+1] - u[i]) - a2 * v[i] - a3 * a[i]
        v[i+1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i+1])
    
    return u, v, a


def plot_response_spectrum(
    result: ResponseSpectrumResult,
    title: str = "Response Spectrum",
    log_scale: bool = True
):
    """
    Plot response spectrum (requires matplotlib).
    
    Args:
        result: ResponseSpectrumResult object
        title: Plot title
        log_scale: Use log scale for period axis
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Spectral Acceleration
    axes[0].plot(result.periods, result.Sa, 'b-', label='Sa')
    axes[0].plot(result.periods, result.PSa, 'b--', alpha=0.7, label='PSa')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    
    # Spectral Velocity
    axes[1].plot(result.periods, result.Sv, 'g-', label='Sv')
    axes[1].plot(result.periods, result.PSv, 'g--', alpha=0.7, label='PSv')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Spectral Displacement
    axes[2].plot(result.periods, result.Sd, 'r-', label='Sd')
    axes[2].set_ylabel('Displacement (m)')
    axes[2].set_xlabel('Period (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    if log_scale:
        for ax in axes:
            ax.set_xscale('log')
    
    plt.tight_layout()
    return fig
