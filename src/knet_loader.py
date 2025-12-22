"""
K-NET / KiK-net Data Loader.
Loads strong-motion records from Japanese networks (#51).
"""
import os
import re
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class StrongMotionRecord:
    """Strong motion record data."""
    station_code: str
    station_name: str
    lat: float
    lon: float
    origin_time: str
    magnitude: float
    depth: float
    max_acc: float  # gal
    sampling_rate: float  # Hz
    n_samples: int
    time: np.ndarray  # seconds
    acc: np.ndarray   # m/s²
    component: str    # NS, EW, UD
    
    
def load_knet_file(filepath: str) -> StrongMotionRecord:
    """
    Load K-NET format acceleration file.
    
    K-NET format:
    - Header section with metadata
    - Data section with acceleration values in gal
    
    Args:
        filepath: Path to K-NET file
        
    Returns:
        StrongMotionRecord object
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    # Parse header
    header = {}
    data_start = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('Memo'):
            data_start = i + 1
            break
            
        # Parse key-value pairs
        if ':' in line or '=' in line:
            parts = re.split(r'[:=]', line, 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                header[key] = value
                
    # Extract metadata
    station_code = header.get('Station Code', header.get('CODE', ''))
    station_name = header.get('Station Name', '')
    
    # Location
    lat = _parse_coordinate(header.get('Station Lat.', header.get('LAT', '0')))
    lon = _parse_coordinate(header.get('Station Long.', header.get('LON', '0')))
    
    # Event info
    origin_time = header.get('Origin Time', header.get('EVENT', ''))
    
    mag_str = header.get('Magnitude(JMA)', header.get('MAG', '0'))
    magnitude = float(re.findall(r'[\d.]+', mag_str)[0]) if re.findall(r'[\d.]+', mag_str) else 0
    
    depth_str = header.get('Depth(km)', header.get('DEPTH', '0'))
    depth = float(re.findall(r'[\d.]+', depth_str)[0]) if re.findall(r'[\d.]+', depth_str) else 0
    
    # Max acceleration
    max_acc_str = header.get('Max. Acc.(gal)', header.get('PMAX', '0'))
    max_acc = float(re.findall(r'[\d.]+', max_acc_str)[0]) if re.findall(r'[\d.]+', max_acc_str) else 0
    
    # Sampling rate
    sampling_str = header.get('Sampling Freq(Hz)', header.get('FREQ', '100'))
    sampling_rate = float(re.findall(r'[\d.]+', sampling_str)[0]) if re.findall(r'[\d.]+', sampling_str) else 100
    
    # Number of samples
    n_samples_str = header.get('Num. of Data', header.get('NDATA', '0'))
    n_samples = int(re.findall(r'\d+', n_samples_str)[0]) if re.findall(r'\d+', n_samples_str) else 0
    
    # Component
    component_str = header.get('Record Time', header.get('COMP', filepath))
    if 'NS' in component_str.upper() or 'N-S' in component_str:
        component = 'NS'
    elif 'EW' in component_str.upper() or 'E-W' in component_str:
        component = 'EW'
    elif 'UD' in component_str.upper() or 'U-D' in component_str:
        component = 'UD'
    else:
        # Infer from filename
        fname = os.path.basename(filepath).upper()
        if 'NS' in fname or '1' in fname[-5:]:
            component = 'NS'
        elif 'EW' in fname or '2' in fname[-5:]:
            component = 'EW'
        elif 'UD' in fname or '3' in fname[-5:]:
            component = 'UD'
        else:
            component = 'Unknown'
    
    # Parse data
    data_lines = lines[data_start:]
    acc_values = []
    
    for line in data_lines:
        # Split on whitespace
        values = line.strip().split()
        for v in values:
            try:
                acc_values.append(float(v))
            except ValueError:
                pass
                
    acc_gal = np.array(acc_values)
    
    # Convert to m/s²
    acc_mps2 = acc_gal * 0.01
    
    # Generate time array
    dt = 1.0 / sampling_rate
    n = len(acc_mps2) if n_samples == 0 else min(n_samples, len(acc_mps2))
    time = np.arange(n) * dt
    
    return StrongMotionRecord(
        station_code=station_code,
        station_name=station_name,
        lat=lat,
        lon=lon,
        origin_time=origin_time,
        magnitude=magnitude,
        depth=depth,
        max_acc=max_acc,
        sampling_rate=sampling_rate,
        n_samples=n,
        time=time[:n],
        acc=acc_mps2[:n],
        component=component
    )


def _parse_coordinate(coord_str: str) -> float:
    """Parse coordinate string to decimal degrees."""
    # Handle formats like "35.123" or "35 12 34.5"
    coord_str = coord_str.strip()
    
    if not coord_str:
        return 0.0
        
    # Simple decimal format
    try:
        return float(coord_str)
    except ValueError:
        pass
        
    # DMS format
    parts = re.findall(r'[\d.]+', coord_str)
    if len(parts) >= 1:
        deg = float(parts[0])
        min = float(parts[1]) if len(parts) > 1 else 0
        sec = float(parts[2]) if len(parts) > 2 else 0
        return deg + min/60 + sec/3600
        
    return 0.0


def load_kiknet_file(filepath: str) -> StrongMotionRecord:
    """
    Load KiK-net format file.
    Similar format to K-NET with minor differences.
    """
    # KiK-net uses same basic format
    return load_knet_file(filepath)


def load_knet_folder(folder_path: str) -> Dict[str, StrongMotionRecord]:
    """
    Load all K-NET files from a folder.
    
    Returns:
        Dictionary mapping component to record
    """
    records = {}
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        if not os.path.isfile(filepath):
            continue
            
        # K-NET files often have no extension or .EW/.NS/.UD
        try:
            record = load_knet_file(filepath)
            records[record.component] = record
        except Exception:
            pass
            
    return records


def combine_horizontal_components(
    ns_record: StrongMotionRecord,
    ew_record: StrongMotionRecord
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine NS and EW components.
    
    Returns:
        (time, acc_ns, acc_ew) with matched lengths
    """
    # Match time arrays
    n = min(len(ns_record.time), len(ew_record.time))
    
    return (
        ns_record.time[:n],
        ns_record.acc[:n],
        ew_record.acc[:n]
    )


def resample_record(
    record: StrongMotionRecord,
    target_dt: float
) -> StrongMotionRecord:
    """
    Resample record to different time step.
    
    Args:
        record: Original record
        target_dt: Target time step (seconds)
        
    Returns:
        Resampled record
    """
    # Calculate new time array
    duration = record.time[-1]
    new_time = np.arange(0, duration, target_dt)
    
    # Interpolate
    new_acc = np.interp(new_time, record.time, record.acc)
    
    return StrongMotionRecord(
        station_code=record.station_code,
        station_name=record.station_name,
        lat=record.lat,
        lon=record.lon,
        origin_time=record.origin_time,
        magnitude=record.magnitude,
        depth=record.depth,
        max_acc=np.max(np.abs(new_acc)) / 0.01,  # Convert back to gal
        sampling_rate=1.0 / target_dt,
        n_samples=len(new_acc),
        time=new_time,
        acc=new_acc,
        component=record.component
    )


def scale_record(
    record: StrongMotionRecord,
    target_pga: float  # Target PGA in m/s²
) -> StrongMotionRecord:
    """
    Scale record to target PGA.
    
    Args:
        record: Original record
        target_pga: Target peak ground acceleration (m/s²)
        
    Returns:
        Scaled record
    """
    current_pga = np.max(np.abs(record.acc))
    scale_factor = target_pga / current_pga if current_pga > 0 else 1.0
    
    scaled_acc = record.acc * scale_factor
    
    return StrongMotionRecord(
        station_code=record.station_code,
        station_name=record.station_name,
        lat=record.lat,
        lon=record.lon,
        origin_time=record.origin_time,
        magnitude=record.magnitude,
        depth=record.depth,
        max_acc=target_pga / 0.01,  # gal
        sampling_rate=record.sampling_rate,
        n_samples=record.n_samples,
        time=record.time.copy(),
        acc=scaled_acc,
        component=record.component
    )


def calculate_response_spectrum(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray = None,
    damping: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate response spectrum.
    
    Args:
        acc: Acceleration time history (m/s²)
        dt: Time step
        periods: Array of periods to evaluate
        damping: Damping ratio
        
    Returns:
        (periods, Sa, Sd) - Spectral acceleration and displacement
    """
    if periods is None:
        periods = np.logspace(-2, 1, 100)  # 0.01s to 10s
        
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
        
        # Single DOF response using Newmark-beta
        u = 0.0
        v = 0.0
        
        beta = 0.25
        gamma = 0.5
        
        a1 = 1.0 / (beta * dt**2)
        a2 = gamma / (beta * dt)
        a3 = 1.0 / (beta * dt)
        a4 = 1.0 / (2*beta) - 1
        
        max_u = 0.0
        max_a = 0.0
        
        for j in range(1, len(acc)):
            # Effective stiffness
            k_hat = k + a2*c + a1
            
            # Effective force
            p_hat = -(acc[j] - acc[j-1]) + c*(a2*u + (gamma/beta - 1)*v + dt*(gamma/(2*beta) - 1)*(-k*u - c*v)) + \
                    a1*u + a3*v + a4*(-k*u - c*v)
            
            # Actually simplified Newmark for SDOF
            a = -k*u - c*v - acc[j]
            du = dt * v + 0.5 * dt**2 * a
            dv = dt * a
            
            u += du
            v += dv
            
            max_u = max(max_u, abs(u))
            total_a = acc[j] + omega**2 * abs(u) + 2*damping*omega*abs(v)
            max_a = max(max_a, abs(total_a))
            
        Sa[i] = max_a
        Sd[i] = max_u
        
    return periods, Sa, Sd
