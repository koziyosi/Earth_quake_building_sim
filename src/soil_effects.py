"""
Soil Effects Module.
Ground amplification and liquefaction analysis.
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SoilType(Enum):
    """Japanese soil classification."""
    TYPE_I = "I"    # Hard rock/stiff soil
    TYPE_II = "II"  # Medium soil
    TYPE_III = "III"  # Soft soil


@dataclass
class SoilLayer:
    """Single soil layer properties."""
    thickness: float     # m
    density: float       # kg/m³
    shear_velocity: float  # m/s
    damping: float       # Damping ratio
    name: str = ""


@dataclass
class SoilProfile:
    """Complete soil profile."""
    layers: List[SoilLayer]
    groundwater_depth: float = 10.0  # m
    
    @property
    def total_depth(self) -> float:
        return sum(layer.thickness for layer in self.layers)
        
    @property
    def average_Vs(self) -> float:
        """Average shear wave velocity (Vs30 style)."""
        total_time = sum(l.thickness / l.shear_velocity for l in self.layers)
        return self.total_depth / total_time if total_time > 0 else 300
        
    def get_soil_type(self) -> SoilType:
        """Classify soil type based on Vs30."""
        vs30 = self.average_Vs
        if vs30 >= 600:
            return SoilType.TYPE_I
        elif vs30 >= 200:
            return SoilType.TYPE_II
        else:
            return SoilType.TYPE_III


def calculate_site_amplification_factor(
    soil_type: SoilType,
    period: float
) -> float:
    """
    Calculate site amplification factor.
    
    Based on Japanese building code amplification.
    
    Args:
        soil_type: Soil classification
        period: Structure natural period (s)
        
    Returns:
        Amplification factor Gs
    """
    # Characteristic period Tc
    Tc = {'I': 0.4, 'II': 0.6, 'III': 0.8}[soil_type.value]
    
    # Amplification (simplified)
    if period < 0.16:
        Gs = 1.0
    elif period < Tc:
        Gs = 1.0 + (period - 0.16) / (Tc - 0.16) * 0.6
    elif period < 2 * Tc:
        Gs = 1.6 - (period - Tc) / Tc * 0.4
    else:
        Gs = 1.2 * Tc / period
        
    return max(1.0, Gs)


def transfer_function_1d(
    profile: SoilProfile,
    frequencies: np.ndarray
) -> np.ndarray:
    """
    Calculate 1D site transfer function.
    
    Uses wave propagation theory.
    
    Args:
        profile: Soil profile
        frequencies: Frequency array (Hz)
        
    Returns:
        Amplification ratio at each frequency
    """
    n_freq = len(frequencies)
    transfer = np.ones(n_freq, dtype=complex)
    
    for layer in reversed(profile.layers):
        omega = 2 * np.pi * frequencies
        
        # Complex shear modulus
        G = layer.density * layer.shear_velocity**2
        G_complex = G * (1 + 2j * layer.damping)
        
        # Wave number
        k = omega / layer.shear_velocity * np.sqrt(1 + 2j * layer.damping)
        
        # Layer transfer matrix (simplified)
        cos_kh = np.cos(k * layer.thickness)
        sin_kh = np.sin(k * layer.thickness)
        
        # Accumulate transfer function
        transfer *= (cos_kh + 1j * layer.damping * sin_kh)
        
    # Amplitude
    return np.abs(transfer)


def apply_site_response(
    acc_rock: np.ndarray,
    dt: float,
    profile: SoilProfile
) -> np.ndarray:
    """
    Apply site response to rock outcrop motion.
    
    Args:
        acc_rock: Rock acceleration (m/s²)
        dt: Time step
        profile: Soil profile
        
    Returns:
        Surface acceleration
    """
    n = len(acc_rock)
    freqs = np.fft.rfftfreq(n, dt)
    
    # FFT of rock motion
    F_rock = np.fft.rfft(acc_rock)
    
    # Transfer function
    H = transfer_function_1d(profile, freqs)
    
    # Apply transfer function
    F_surface = F_rock * H
    
    # Inverse FFT
    acc_surface = np.fft.irfft(F_surface, n)
    
    return acc_surface


# ===== Liquefaction Assessment =====

@dataclass
class LiquefactionResult:
    """Liquefaction assessment result."""
    depth: float
    FL: float  # Factor of safety
    PL: float  # Liquefaction potential index
    liquefied: bool
    settlement: float  # Estimated settlement (m)


def calculate_liquefaction_FL(
    depth: float,
    N_value: int,
    fines_content: float,
    groundwater_depth: float,
    pga: float,  # Peak ground acceleration (g)
    magnitude: float = 7.0
) -> float:
    """
    Calculate liquefaction factor of safety FL.
    
    Based on simplified procedure (Seed & Idriss).
    
    Args:
        depth: Depth below surface (m)
        N_value: SPT N-value
        fines_content: Fines content (%)
        groundwater_depth: Groundwater depth (m)
        pga: Peak ground acceleration (g)
        magnitude: Earthquake magnitude
        
    Returns:
        Factor of safety FL (< 1 = liquefaction)
    """
    if depth < groundwater_depth:
        return float('inf')  # Above water table
        
    # Stress reduction factor
    rd = 1 - 0.00765 * depth if depth < 9.15 else 1.174 - 0.0267 * depth
    rd = max(0.5, min(1.0, rd))
    
    # CSR (Cyclic Stress Ratio)
    sigma_v = depth * 18000  # Total vertical stress (Pa), assume 18 kN/m³
    sigma_v_eff = (depth - groundwater_depth) * 10000  # Effective stress
    
    if sigma_v_eff <= 0:
        return float('inf')
        
    CSR = 0.65 * pga * (sigma_v / sigma_v_eff) * rd
    
    # Correct N for overburden
    CN = min(2.0, (100000 / sigma_v_eff) ** 0.5)
    N1_60 = N_value * CN
    
    # Correct for fines content
    if fines_content <= 5:
        N1_60_cs = N1_60
    elif fines_content <= 35:
        N1_60_cs = N1_60 + (fines_content - 5) * 0.2
    else:
        N1_60_cs = N1_60 + 6
        
    # CRR (Cyclic Resistance Ratio)
    if N1_60_cs < 30:
        CRR = 0.048 * N1_60_cs
    else:
        CRR = 2.0  # Not liquefiable
        
    # Magnitude scaling factor
    MSF = 10 ** (2.24 / magnitude ** 2.56)
    
    # Factor of safety
    FL = (CRR * MSF) / CSR if CSR > 0 else float('inf')
    
    return FL


def calculate_liquefaction_potential_index(
    FL_values: List[Tuple[float, float]]  # (depth, FL)
) -> float:
    """
    Calculate Liquefaction Potential Index (PL).
    
    PL = ∫₀²⁰ F(z) * W(z) dz
    
    where F(z) = 1 - FL if FL < 1, else 0
    and W(z) = 10 - 0.5z
    
    Args:
        FL_values: List of (depth, FL) tuples
        
    Returns:
        PL value (0=no risk, 5=low, 15=high, >50=very high)
    """
    if not FL_values:
        return 0
        
    # Sort by depth
    FL_values = sorted(FL_values, key=lambda x: x[0])
    
    PL = 0
    
    for i in range(len(FL_values) - 1):
        z1, FL1 = FL_values[i]
        z2, FL2 = FL_values[i + 1]
        
        if z1 >= 20 or z2 <= 0:
            continue
            
        # Average F and W in layer
        FL_avg = (FL1 + FL2) / 2
        F_avg = max(0, 1 - FL_avg) if FL_avg < 1 else 0
        
        z_mid = (z1 + z2) / 2
        W_avg = max(0, 10 - 0.5 * z_mid)
        
        dz = z2 - z1
        PL += F_avg * W_avg * dz
        
    return PL


def estimate_liquefaction_settlement(
    FL_values: List[Tuple[float, float]],
    Dr_values: List[float] = None
) -> float:
    """
    Estimate ground settlement due to liquefaction.
    
    Based on Ishihara & Yoshimine (1992).
    
    Args:
        FL_values: List of (depth, FL) tuples
        Dr_values: Relative density values (optional)
        
    Returns:
        Estimated settlement (m)
    """
    settlement = 0
    
    for i, (depth, FL) in enumerate(FL_values):
        if FL >= 1:
            continue  # No liquefaction
            
        # Volumetric strain (simplified)
        if FL < 0.5:
            ev = 0.05  # 5% strain
        elif FL < 0.8:
            ev = 0.03
        else:
            ev = 0.01
            
        # Layer thickness (approximate)
        if i < len(FL_values) - 1:
            dz = FL_values[i + 1][0] - depth
        else:
            dz = 1.0
            
        settlement += ev * dz
        
    return settlement


def assess_liquefaction_risk(
    profile: SoilProfile,
    N_values: List[int],
    fines: List[float],
    pga: float,
    magnitude: float = 7.0
) -> Dict:
    """
    Full liquefaction assessment for soil profile.
    
    Args:
        profile: Soil profile
        N_values: SPT N-value at each layer
        fines: Fines content (%) at each layer
        pga: Peak ground acceleration (g)
        magnitude: Earthquake magnitude
        
    Returns:
        Assessment results dictionary
    """
    FL_values = []
    current_depth = 0
    
    for i, layer in enumerate(profile.layers):
        mid_depth = current_depth + layer.thickness / 2
        
        N = N_values[i] if i < len(N_values) else 10
        fc = fines[i] if i < len(fines) else 15
        
        FL = calculate_liquefaction_FL(
            mid_depth, N, fc, profile.groundwater_depth, pga, magnitude
        )
        
        FL_values.append((mid_depth, FL))
        current_depth += layer.thickness
        
    PL = calculate_liquefaction_potential_index(FL_values)
    settlement = estimate_liquefaction_settlement(FL_values)
    
    # Risk classification
    if PL < 5:
        risk = "Very Low"
    elif PL < 15:
        risk = "Low"
    elif PL < 30:
        risk = "High"
    else:
        risk = "Very High"
        
    return {
        'FL_values': FL_values,
        'PL': PL,
        'settlement': settlement,
        'risk_level': risk,
        'critical_layers': [i for i, (d, fl) in enumerate(FL_values) if fl < 1]
    }
