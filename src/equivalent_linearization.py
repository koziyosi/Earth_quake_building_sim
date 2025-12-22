"""
Equivalent Linearization Module.
Linear approximation methods for nonlinear analysis.
"""
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class EquivalentLinearSystem:
    """Equivalent linear system properties."""
    T_eff: float           # Effective period
    zeta_eff: float        # Effective damping ratio
    K_eff: float           # Effective stiffness
    M_eff: float           # Effective mass
    Sd: float              # Spectral displacement demand
    Sa: float              # Spectral acceleration demand


def equivalent_linearization_bilinear(
    K0: float,
    Fy: float,
    r: float,
    Sa_spectrum: callable,
    mass: float = 1.0,
    damping_initial: float = 0.05,
    max_iterations: int = 20,
    tolerance: float = 0.01
) -> EquivalentLinearSystem:
    """
    Equivalent linearization for bilinear SDOF system.
    
    Uses secant stiffness and hysteretic damping.
    
    Args:
        K0: Initial stiffness
        Fy: Yield force
        r: Post-yield stiffness ratio (Kp = r*K0)
        Sa_spectrum: Function(T, zeta) -> Sa (m/s²)
        mass: System mass
        damping_initial: Initial damping ratio
        max_iterations: Max iterations
        tolerance: Convergence tolerance
        
    Returns:
        EquivalentLinearSystem
    """
    dy = Fy / K0  # Yield displacement
    T0 = 2 * np.pi * np.sqrt(mass / K0)
    
    # Initial guess: elastic response
    T_eff = T0
    zeta_eff = damping_initial
    
    for iteration in range(max_iterations):
        # Get spectral demand
        Sa = Sa_spectrum(T_eff, zeta_eff)
        omega = 2 * np.pi / T_eff
        Sd = Sa / omega**2  # Spectral displacement
        
        # Ductility
        mu = max(1.0, Sd / dy)
        
        if mu <= 1.0:
            # Elastic
            K_eff = K0
            zeta_h = 0
        else:
            # Inelastic - secant stiffness
            du = Sd
            Fu = Fy + r * K0 * (du - dy)
            K_eff = Fu / du
            
            # Hysteretic damping (based on area)
            # Area = 4 * Fy * (du - dy) * (1 - r)
            area = 4 * Fy * (du - dy) * (1 - r)
            E_elastic = 0.5 * K_eff * du**2
            zeta_h = area / (4 * np.pi * E_elastic) if E_elastic > 0 else 0
            
        # New effective period
        T_eff_new = 2 * np.pi * np.sqrt(mass / K_eff)
        
        # New effective damping
        zeta_eff_new = damping_initial + zeta_h
        zeta_eff_new = min(zeta_eff_new, 0.50)  # Cap at 50%
        
        # Check convergence
        if abs(T_eff_new - T_eff) / T_eff < tolerance:
            break
            
        T_eff = T_eff_new
        zeta_eff = zeta_eff_new
        
    return EquivalentLinearSystem(
        T_eff=T_eff,
        zeta_eff=zeta_eff,
        K_eff=K_eff,
        M_eff=mass,
        Sd=Sd,
        Sa=Sa
    )


def takeda_equivalent_damping(
    mu: float,
    alpha: float = 0.5,
    beta: float = 0.0
) -> float:
    """
    Equivalent viscous damping for Takeda hysteresis.
    
    Args:
        mu: Ductility demand
        alpha: Unloading stiffness parameter
        beta: Reloading stiffness parameter
        
    Returns:
        Equivalent damping ratio addition
    """
    if mu <= 1:
        return 0
        
    # Takeda formula (approximate)
    zeta_h = 0.05 * (1 - 1/mu) + 0.1 * (1 - 1/np.sqrt(mu))
    
    return min(zeta_h, 0.30)


def capacity_spectrum_method(
    capacity_curve: Tuple[np.ndarray, np.ndarray],
    demand_spectrum: callable,
    mass_eff: float,
    participation_factor: float = 1.0,
    initial_damping: float = 0.05
) -> Dict:
    """
    Capacity Spectrum Method (ATC-40 / FEMA 440).
    
    Finds performance point where capacity equals demand.
    
    Args:
        capacity_curve: (displacement, base_shear) arrays
        demand_spectrum: Function(T, zeta) -> (Sd, Sa)
        mass_eff: Effective modal mass
        participation_factor: Modal participation factor
        initial_damping: Initial damping ratio
        
    Returns:
        Performance point and equivalent system properties
    """
    disp, shear = capacity_curve
    
    # Convert to spectral coordinates
    Sd = disp / participation_factor
    Sa = shear / (mass_eff * participation_factor)
    
    # Find yield point (approximate)
    slope_initial = (Sa[1] - Sa[0]) / (Sd[1] - Sd[0]) if Sd[1] > Sd[0] else 1e6
    
    # Iterative solution
    T_eff = 0.5  # Initial guess
    zeta_eff = initial_damping
    
    for iteration in range(30):
        # Get demand at effective period
        Sd_demand, Sa_demand = demand_spectrum(T_eff, zeta_eff)
        
        # Find capacity at this displacement
        Sa_capacity = np.interp(Sd_demand, Sd, Sa)
        
        # Check convergence
        if abs(Sa_capacity - Sa_demand) / max(Sa_demand, 1e-6) < 0.02:
            break
            
        # Update effective period
        K_eff = Sa_capacity / Sd_demand if Sd_demand > 0 else slope_initial
        T_eff_new = 2 * np.pi / np.sqrt(K_eff)
        
        # Hysteretic damping (simplified)
        dy_approx = Sa[1] / slope_initial if slope_initial > 0 else Sd[0]
        mu = Sd_demand / dy_approx if dy_approx > 0 else 1.0
        zeta_h = takeda_equivalent_damping(max(1, mu))
        zeta_eff = initial_damping + zeta_h
        
        T_eff = 0.8 * T_eff + 0.2 * T_eff_new  # Damped update
        
    performance_point = {
        'Sd': Sd_demand,
        'Sa': Sa_capacity,
        'displacement': Sd_demand * participation_factor,
        'base_shear': Sa_capacity * mass_eff * participation_factor,
        'T_eff': T_eff,
        'zeta_eff': zeta_eff,
        'ductility': mu if 'mu' in dir() else 1.0
    }
    
    return performance_point


def plot_capacity_spectrum(
    capacity_curve: Tuple[np.ndarray, np.ndarray],
    demand_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    performance_point: Dict = None,
    ax = None
):
    """
    Plot capacity spectrum method diagram.
    
    Args:
        capacity_curve: (Sd, Sa) in spectral coordinates
        demand_curves: Dict of {label: (Sd, Sa)} demand spectra
        performance_point: Performance point dict
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    Sd_cap, Sa_cap = capacity_curve
    ax.plot(Sd_cap * 100, Sa_cap, 'b-', linewidth=2.5, label='Capacity')
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(demand_curves)))
    for i, (label, (Sd_dem, Sa_dem)) in enumerate(demand_curves.items()):
        ax.plot(Sd_dem * 100, Sa_dem, '--', color=colors[i], linewidth=1.5, label=f'Demand ({label})')
        
    if performance_point:
        ax.plot(performance_point['Sd'] * 100, performance_point['Sa'], 
                'go', markersize=12, label='Performance Point', zorder=5)
        ax.annotate(f"T={performance_point['T_eff']:.2f}s\nζ={performance_point['zeta_eff']:.1%}",
                   (performance_point['Sd'] * 100, performance_point['Sa']),
                   xytext=(10, 10), textcoords='offset points')
                   
    ax.set_xlabel('Spectral Displacement Sd (cm)')
    ax.set_ylabel('Spectral Acceleration Sa (g)')
    ax.set_title('Capacity Spectrum Method')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax
