"""
Energy Spectrum Module.
Energy-based seismic analysis tools.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnergyComponents:
    """Energy components in structural system."""
    kinetic: np.ndarray          # Kinetic energy E_k = 0.5*m*v²
    strain: np.ndarray           # Elastic strain E_s = 0.5*k*u²
    damping: np.ndarray          # Damping dissipation E_d = ∫c*v²dt
    hysteretic: np.ndarray       # Hysteretic dissipation E_h
    input: np.ndarray            # Input energy E_i = -∫m*ü_g*v dt
    total_dissipated: np.ndarray  # E_d + E_h


def calculate_energy_time_history(
    time: np.ndarray,
    disp: np.ndarray,
    vel: np.ndarray,
    acc_ground: np.ndarray,
    mass: float,
    stiffness: float,
    damping: float,
    force_history: np.ndarray = None
) -> EnergyComponents:
    """
    Calculate energy components over time for SDOF system.
    
    Energy balance: E_i = E_k + E_s + E_d + E_h
    
    Args:
        time: Time array
        disp: Displacement history
        vel: Velocity history
        acc_ground: Ground acceleration
        mass: System mass
        stiffness: System stiffness
        damping: Damping coefficient
        force_history: Optional restoring force history (for hysteretic)
        
    Returns:
        EnergyComponents dataclass
    """
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    n = len(time)
    
    # Kinetic energy
    E_k = 0.5 * mass * vel**2
    
    # Elastic strain energy (linear)
    E_s = 0.5 * stiffness * disp**2
    
    # Damping energy (cumulative)
    E_d = np.zeros(n)
    for i in range(1, n):
        E_d[i] = E_d[i-1] + damping * vel[i-1]**2 * dt
        
    # Hysteretic energy
    E_h = np.zeros(n)
    if force_history is not None:
        for i in range(1, n):
            # Plastic work = ∫(F - F_elastic)du
            du = disp[i] - disp[i-1]
            F_elastic = stiffness * disp[i-1]
            F_actual = force_history[i-1]
            E_h[i] = E_h[i-1] + max(0, (F_actual - F_elastic) * du)
    
    # Input energy
    E_i = np.zeros(n)
    for i in range(1, n):
        E_i[i] = E_i[i-1] - mass * acc_ground[i-1] * vel[i-1] * dt
        
    return EnergyComponents(
        kinetic=E_k,
        strain=E_s,
        damping=E_d,
        hysteretic=E_h,
        input=E_i,
        total_dissipated=E_d + E_h
    )


def calculate_input_energy_spectrum(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate input energy response spectrum.
    
    E_I/m = ∫(ü_g * v_rel) dt
    
    Args:
        acc: Ground acceleration (m/s²)
        dt: Time step
        periods: Array of periods to calculate
        damping: Damping ratio
        
    Returns:
        (periods, Ve) where Ve = √(2*E_I/m) equivalent velocity
    """
    Ve = np.zeros_like(periods)
    
    for i, T in enumerate(periods):
        if T <= 0:
            Ve[i] = 0
            continue
            
        omega = 2 * np.pi / T
        c = 2 * damping * omega
        k = omega**2
        
        # Time integration (Newmark)
        n = len(acc)
        u, v = 0.0, 0.0
        E_input = 0.0
        
        beta, gamma = 0.25, 0.5
        
        for j in range(1, n):
            a = -k * u - c * v - acc[j-1]
            
            du = dt * v + 0.5 * dt**2 * ((1 - 2*beta) * a)
            dv = dt * ((1 - gamma) * a)
            
            u_new = u + du
            v_new = v + dv
            
            a_new = -k * u_new - c * v_new - acc[j]
            
            u = u + dt * v + 0.5 * dt**2 * ((1 - 2*beta) * a + 2*beta * a_new)
            v = v + dt * ((1 - gamma) * a + gamma * a_new)
            
            # Input energy increment
            E_input += -acc[j] * v * dt
            
        # Equivalent velocity
        Ve[i] = np.sqrt(2 * max(0, E_input))
        
    return periods, Ve


def calculate_hysteretic_energy_spectrum(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    ductility: float = 2.0,
    damping: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate hysteretic energy demand spectrum.
    
    For bilinear SDOF with given target ductility.
    
    Args:
        acc: Ground acceleration
        dt: Time step
        periods: Period array
        ductility: Target ductility factor
        damping: Damping ratio
        
    Returns:
        (periods, Eh/m) hysteretic energy per unit mass
    """
    Eh = np.zeros_like(periods)
    
    for i, T in enumerate(periods):
        if T <= 0:
            Eh[i] = 0
            continue
            
        omega = 2 * np.pi / T
        k = omega**2
        c = 2 * damping * omega
        
        # Estimate yield displacement (iterative would be better)
        u_max = 0.1  # Initial guess
        uy = u_max / ductility
        Fy = k * uy
        
        # Run analysis with bilinear hysteresis
        r = 0.02  # Post-yield stiffness ratio
        
        n = len(acc)
        u, v = 0.0, 0.0
        up = 0.0  # Plastic displacement
        E_hyst = 0.0
        
        for j in range(1, n):
            # Trial elastic force
            F_trial = k * (u - up)
            
            if abs(F_trial) > Fy:
                # Yielding
                F = np.sign(F_trial) * Fy + r * k * (u - up - np.sign(F_trial) * uy)
                up = u - F / k
            else:
                F = F_trial
                
            a = -F - c * v - acc[j-1]
            
            u_new = u + dt * v + 0.5 * dt**2 * a
            v_new = v + dt * a
            
            # Hysteretic energy increment
            du = u_new - u
            E_hyst += max(0, (F - k * (u - up)) * du)
            
            u, v = u_new, v_new
            u_max = max(u_max, abs(u))
            
        Eh[i] = E_hyst
        
    return periods, Eh


def calculate_damping_effectiveness(
    input_energy: float,
    dissipated_by_damper: float,
    peak_displacement: float,
    target_displacement: float
) -> dict:
    """
    Evaluate damper effectiveness.
    
    Returns:
        Dictionary with effectiveness metrics
    """
    energy_ratio = dissipated_by_damper / input_energy if input_energy > 0 else 0
    disp_reduction = 1 - peak_displacement / target_displacement if target_displacement > 0 else 0
    
    # Effectiveness score (0-100)
    score = (energy_ratio * 0.4 + max(0, disp_reduction) * 0.6) * 100
    
    return {
        'energy_dissipation_ratio': energy_ratio,
        'displacement_reduction': disp_reduction,
        'effectiveness_score': score,
        'rating': 'Excellent' if score > 80 else 'Good' if score > 60 else 'Fair' if score > 40 else 'Poor'
    }


def plot_energy_decomposition(
    time: np.ndarray,
    components: EnergyComponents,
    ax = None
):
    """Plot energy decomposition over time."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    ax.fill_between(time, 0, components.kinetic, alpha=0.3, label='Kinetic')
    ax.fill_between(time, components.kinetic, 
                    components.kinetic + components.strain, alpha=0.3, label='Strain')
    ax.fill_between(time, components.kinetic + components.strain,
                    components.kinetic + components.strain + components.damping,
                    alpha=0.3, label='Damping')
    ax.fill_between(time, components.kinetic + components.strain + components.damping,
                    components.kinetic + components.strain + components.total_dissipated,
                    alpha=0.3, label='Hysteretic')
                    
    ax.plot(time, components.input, 'k--', linewidth=2, label='Input')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Energy Decomposition')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax
