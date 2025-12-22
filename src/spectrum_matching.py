"""
Spectrum Matching Module.
Generates artificial earthquake motions compatible with target response spectrum (#23).
"""
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TargetSpectrum:
    """Target response spectrum definition."""
    periods: np.ndarray
    accelerations: np.ndarray  # Spectral accelerations (m/s²)
    damping: float = 0.05


def generate_spectrum_compatible_motion(
    target: TargetSpectrum,
    duration: float = 20.0,
    dt: float = 0.01,
    n_iterations: int = 20,
    tolerance: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spectrum-compatible artificial ground motion.
    
    Uses iterative frequency-domain method:
    1. Generate random phase angles
    2. Calculate Fourier amplitudes from target spectrum
    3. Apply envelope function
    4. Iterate to match target spectrum
    
    Args:
        target: Target response spectrum
        duration: Motion duration (s)
        dt: Time step (s)
        n_iterations: Maximum iterations
        tolerance: Acceptable spectrum mismatch (fraction)
        
    Returns:
        (time array, acceleration array)
    """
    n_points = int(duration / dt)
    time = np.arange(n_points) * dt
    
    # Frequency array
    freqs = np.fft.rfftfreq(n_points, dt)
    n_freq = len(freqs)
    
    # Convert target periods to frequencies
    target_freqs = 1.0 / target.periods
    
    # Interpolate target spectrum to frequency array
    Sa_target = np.interp(
        freqs,
        np.flip(target_freqs),
        np.flip(target.accelerations),
        left=0,
        right=0
    )
    
    # Initial Fourier amplitudes from target spectrum
    # Sa ≈ ω² * Sd, so amplitude ≈ Sa / ω²
    omega = 2 * np.pi * freqs
    omega[0] = 1.0  # Avoid division by zero
    amplitudes = Sa_target / (omega ** 2) * n_points * dt
    amplitudes[0] = 0  # No DC component
    
    # Random phases
    phases = np.random.uniform(0, 2*np.pi, n_freq)
    
    # Construct complex Fourier coefficients
    F = amplitudes * np.exp(1j * phases)
    
    # Inverse FFT
    acc = np.fft.irfft(F, n_points)
    
    # Apply envelope function (compound envelope)
    envelope = _compound_envelope(time, duration)
    acc *= envelope
    
    # Normalize to approximate target PGA
    target_pga = np.max(target.accelerations) * 0.4  # Rough estimate
    current_pga = np.max(np.abs(acc))
    if current_pga > 0:
        acc *= target_pga / current_pga
    
    # Iterative spectrum matching (simplified)
    for iteration in range(n_iterations):
        # Calculate response spectrum
        _, Sa_current, _ = calculate_response_spectrum(acc, dt, target.periods, target.damping)
        
        # Calculate mismatch
        ratio = target.accelerations / (Sa_current + 1e-10)
        max_mismatch = np.max(np.abs(ratio - 1))
        
        if max_mismatch < tolerance:
            break
            
        # Adjust Fourier amplitudes
        ratio_freq = np.interp(
            freqs,
            np.flip(1.0 / target.periods),
            np.flip(ratio),
            left=1.0,
            right=1.0
        )
        
        F = np.fft.rfft(acc)
        F *= np.sqrt(ratio_freq)
        acc = np.fft.irfft(F, n_points)
        acc *= envelope
    
    # Baseline correction
    acc = _baseline_correct(acc, dt)
    
    return time, acc


def _compound_envelope(time: np.ndarray, duration: float) -> np.ndarray:
    """
    Generate compound envelope function.
    
    Based on Saragoni & Hart (1974) envelope: t^a * exp(-b*t)
    """
    t_max = duration * 0.3  # Peak at 30% of duration
    
    # Build-up phase
    a = 2.0
    b = a / t_max
    
    envelope = (time ** a) * np.exp(-b * time)
    
    # Normalize peak to 1
    envelope /= np.max(envelope)
    
    # Taper at end
    taper_start = 0.85 * duration
    taper_mask = time > taper_start
    taper = 1 - ((time[taper_mask] - taper_start) / (duration - taper_start)) ** 2
    envelope[taper_mask] *= np.maximum(taper, 0)
    
    return envelope


def _baseline_correct(acc: np.ndarray, dt: float) -> np.ndarray:
    """
    Apply baseline correction to acceleration.
    
    Removes linear trend and high-pass filters to eliminate drift.
    """
    from numpy.polynomial import polynomial as P
    
    n = len(acc)
    time = np.arange(n) * dt
    
    # Remove linear trend
    coeffs = np.polyfit(time, acc, 1)
    acc_corrected = acc - np.polyval(coeffs, time)
    
    # High-pass filter (simple moving average subtraction)
    window = int(2.0 / dt)  # 2 second window
    if window > 1 and window < n // 2:
        kernel = np.ones(window) / window
        baseline = np.convolve(acc_corrected, kernel, mode='same')
        acc_corrected -= baseline
    
    return acc_corrected


def calculate_response_spectrum(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate response spectrum for given acceleration.
    
    Returns:
        (periods, Sa, Sd)
    """
    Sa = np.zeros_like(periods)
    Sd = np.zeros_like(periods)
    
    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = np.max(np.abs(acc))
            Sd[i] = 0
            continue
            
        omega = 2 * np.pi / T
        c = 2 * damping * omega
        k = omega ** 2
        
        # Newmark-beta for SDOF
        u, v = 0.0, 0.0
        max_u, max_a = 0.0, 0.0
        
        beta, gamma = 0.25, 0.5
        
        for j in range(1, len(acc)):
            a_n = -k * u - c * v - acc[j-1]
            
            du = dt * v + 0.5 * dt**2 * ((1 - 2*beta) * a_n)
            dv = dt * ((1 - gamma) * a_n)
            
            u_new = u + du
            v_new = v + dv
            
            a_new = -k * u_new - c * v_new - acc[j]
            
            u = u + dt * v + 0.5 * dt**2 * ((1 - 2*beta) * a_n + 2*beta * a_new)
            v = v + dt * ((1 - gamma) * a_n + gamma * a_new)
            
            max_u = max(max_u, abs(u))
            max_a = max(max_a, abs(acc[j] + omega**2 * u + c * v))
        
        Sa[i] = max_a
        Sd[i] = max_u
    
    return periods, Sa, Sd


# Japanese design spectrum (simplified AI distribution)
def get_japanese_design_spectrum(
    soil_type: str = 'II',
    region: int = 1,
    damping: float = 0.05
) -> TargetSpectrum:
    """
    Get Japanese design response spectrum.
    
    Based on Building Standard Law notification.
    
    Args:
        soil_type: 'I', 'II', or 'III'
        region: Seismic zone (1-4)
        damping: Damping ratio
        
    Returns:
        TargetSpectrum object
    """
    # Zone coefficient
    Z = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7}[region]
    
    # Soil parameters
    soil_params = {
        'I': {'Tc': 0.4, 'Gs': 1.5},
        'II': {'Tc': 0.6, 'Gs': 1.5},
        'III': {'Tc': 0.8, 'Gs': 2.0}
    }
    Tc = soil_params[soil_type]['Tc']
    Gs = soil_params[soil_type]['Gs']
    
    # Generate periods
    periods = np.logspace(-2, 1, 100)  # 0.01 to 10 s
    
    # Calculate Rt (spectral shape)
    Sa = np.zeros_like(periods)
    for i, T in enumerate(periods):
        if T < Tc:
            Rt = 1.0
        else:
            Rt = 1.0 - 0.2 * ((T - Tc) / Tc) ** 2
            Rt = max(Rt, 0.4)
        
        # Convert to acceleration (assuming 1g base)
        Sa[i] = Z * Gs * Rt * 9.81
    
    # Damping correction
    if damping != 0.05:
        h_factor = 1.5 / (1 + 10 * damping)
        Sa *= h_factor
    
    return TargetSpectrum(
        periods=periods,
        accelerations=Sa,
        damping=damping
    )
