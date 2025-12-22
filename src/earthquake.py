import numpy as np

def generate_synthetic_wave(duration: float, dt: float, max_acc: float = 300.0, 
                            dominant_period: float = None):
    """
    Generates a synthetic earthquake wave (white noise with envelope).
    
    Args:
        duration (float): Total duration in seconds.
        dt (float): Time step in seconds.
        max_acc (float): Peak acceleration in gal (cm/s^2).
        dominant_period (float): Dominant period in seconds (optional).
                                If specified, filters the wave to have this
                                characteristic period. Common values:
                                - 0.5s: Short period (near-field, shallow quake)
                                - 1.0s: Medium period
                                - 2.0s: Long period (far-field, deep quake)
    
    Returns:
        np.array: Time vector.
        np.array: Acceleration vector (m/s^2).
    """
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    
    # Generate base signal
    if dominant_period is not None and dominant_period > 0:
        # Generate harmonic components around dominant frequency
        f0 = 1.0 / dominant_period  # Dominant frequency
        
        # Create a wave with the dominant frequency and harmonics
        # Using sum of sinusoids with random phases for realism
        np.random.seed(None)  # Ensure randomness
        acc = np.zeros(n_steps)
        
        # Main frequency component
        phase1 = np.random.uniform(0, 2*np.pi)
        acc += np.sin(2*np.pi*f0*t + phase1)
        
        # Add neighboring frequencies for natural variation (±30%)
        for f_mult in [0.7, 0.85, 1.15, 1.3]:
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.3, 0.6)
            acc += amplitude * np.sin(2*np.pi*f0*f_mult*t + phase)
        
        # Add small high-frequency noise for realism
        noise = np.random.normal(0, 0.15, n_steps)
        acc += noise
        
    else:
        # White noise (original behavior)
        acc = np.random.normal(0, 1, n_steps)
    
    # Envelope function (Jennings type)
    # 0 - t1: Rise (quadratic)
    # t1 - t2: Strong motion (constant)
    # t2 - t3: Decay (exponential)
    
    envelope = np.zeros(n_steps)
    t1 = duration * 0.15
    t2 = duration * 0.50
    t3 = duration
    
    for i, ti in enumerate(t):
        if ti < t1:
            envelope[i] = (ti / t1)**2
        elif ti < t2:
            envelope[i] = 1.0
        else:
            envelope[i] = np.exp(-2.0 * (ti - t2) / (t3 - t2))
            
    acc = acc * envelope
    
    # Normalize to max_acc
    current_max = np.max(np.abs(acc))
    if current_max > 0:
        acc = acc / current_max * max_acc
        
    # Convert gal to m/s^2 (1 gal = 0.01 m/s^2)
    acc_mps2 = acc * 0.01
    
    return t, acc_mps2

def load_wave_from_file(filepath: str):
    """
    Loads earthquake wave from a text file.
    Format: Single column of acceleration values, or two columns (time, acc).
    Assumes acceleration is in gal.
    
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (time array or None, acceleration array or None)
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Earthquake wave file not found: {filepath}")
    
    try:
        data = np.loadtxt(filepath)
        
        if data.size == 0:
            raise ValueError(f"Earthquake wave file is empty: {filepath}")
            
        if data.ndim == 1:
            # Single column: assume constant dt, return raw normalized data
            acc = data * 0.01  # gal to m/s^2
            return None, acc
        elif data.ndim == 2 and data.shape[1] >= 2:
            t = data[:, 0]
            acc = data[:, 1] * 0.01  # gal to m/s^2
            
            # Validate time array
            if len(t) < 2:
                raise ValueError("Time array must have at least 2 points")
            if not np.all(np.diff(t) > 0):
                raise ValueError("Time array must be monotonically increasing")
                
            return t, acc
        else:
            raise ValueError(f"Invalid data format in {filepath}. Expected 1 or 2 columns.")
            
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error loading earthquake file '{filepath}': {e}")


def generate_long_period_pulse(duration: float, dt: float, pulse_period: float = 4.0,
                                max_vel: float = 100.0, pulse_type: str = 'velocity'):
    """
    Generate long-period pulse motion (near-fault directivity effect).
    
    This simulates the "killer pulse" from near-fault earthquakes that can cause
    severe damage to individual columns and stories in tall buildings.
    
    Args:
        duration: Total duration in seconds
        dt: Time step in seconds
        pulse_period: Dominant period of the pulse (2-6s typical for near-fault)
        max_vel: Peak ground velocity in cm/s (typical: 50-150 cm/s)
        pulse_type: 'velocity' (directivity) or 'fling' (permanent displacement)
    
    Returns:
        t: Time array
        acc: Acceleration array (m/s^2)
        vel: Velocity array (m/s) - useful for understanding energy input
    """
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    
    # Pulse arrival time (centered in first half)
    t_pulse = duration * 0.25
    omega = 2 * np.pi / pulse_period
    
    vel = np.zeros(n_steps)
    acc = np.zeros(n_steps)
    
    if pulse_type == 'velocity':
        # Velocity pulse (Mavroeidis & Papageorgiou model simplified)
        # Creates a coherent velocity pulse that directly loads structures
        for i, ti in enumerate(t):
            tau = (ti - t_pulse) / pulse_period
            if -1.5 < tau < 1.5:
                # Gabor wavelet-like pulse
                envelope = np.exp(-2 * tau**2)
                vel[i] = max_vel * envelope * np.cos(2 * np.pi * tau)
        
        # Differentiate to get acceleration
        acc = np.gradient(vel, dt)
        
    elif pulse_type == 'fling':
        # Fling-step motion (permanent ground displacement)
        # Creates a one-sided pulse followed by static offset
        displacement = np.zeros(n_steps)
        max_disp = max_vel * pulse_period / (2 * np.pi)  # Approximate
        
        for i, ti in enumerate(t):
            tau = (ti - t_pulse) / pulse_period
            if tau < 0:
                displacement[i] = 0
            elif tau < 1:
                # Smooth ramp
                displacement[i] = max_disp * (1 - np.cos(np.pi * tau)) / 2
            else:
                displacement[i] = max_disp
        
        # Double differentiate to get acceleration
        vel = np.gradient(displacement, dt)
        acc = np.gradient(vel, dt)
    
    # Convert cm/s to m/s for velocity
    vel_mps = vel * 0.01
    # Acceleration is already in cm/s^2, convert to m/s^2
    acc_mps2 = acc * 0.01
    
    return t, acc_mps2, vel_mps


def generate_long_period_ground_motion(duration: float, dt: float, 
                                        period_range: tuple = (2.0, 8.0),
                                        max_acc: float = 100.0):
    """
    Generate long-period ground motion (far-field from large earthquakes).
    
    Simulates the type of motion that affects tall buildings far from
    large earthquakes (like what affected Tokyo from the 2011 Tohoku earthquake).
    
    Args:
        duration: Total duration in seconds (often 60-300s for long-period)
        dt: Time step
        period_range: (min_period, max_period) in seconds
        max_acc: Peak acceleration in gal
    
    Returns:
        t: Time array
        acc: Acceleration array (m/s^2)
    """
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    
    acc = np.zeros(n_steps)
    
    # Generate sum of long-period components
    n_components = 15
    periods = np.linspace(period_range[0], period_range[1], n_components)
    
    np.random.seed(None)
    
    for period in periods:
        freq = 1.0 / period
        phase = np.random.uniform(0, 2 * np.pi)
        # Amplitude decreases for shorter periods (characteristic of distant quakes)
        amplitude = np.random.uniform(0.5, 1.0) * (period / period_range[1])
        acc += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Long envelope with slow rise and decay (characteristic of distant sources)
    envelope = np.zeros(n_steps)
    t1 = duration * 0.1
    t2 = duration * 0.3
    t3 = duration * 0.7
    t4 = duration
    
    for i, ti in enumerate(t):
        if ti < t1:
            envelope[i] = (ti / t1) ** 1.5
        elif ti < t2:
            envelope[i] = 1.0
        elif ti < t3:
            envelope[i] = 1.0 - 0.3 * (ti - t2) / (t3 - t2)
        else:
            envelope[i] = 0.7 * np.exp(-3 * (ti - t3) / (t4 - t3))
    
    acc = acc * envelope
    
    # Normalize
    if np.max(np.abs(acc)) > 0:
        acc = acc / np.max(np.abs(acc)) * max_acc
    
    acc_mps2 = acc * 0.01
    
    return t, acc_mps2


def generate_resonance_pulse(duration: float, dt: float, 
                              building_period: float, max_acc: float = 200.0):
    """
    Generate a resonance-seeking pulse that maximizes energy input to a building.
    
    This is the "worst case" scenario where the ground motion period matches
    the building's natural period, causing maximum amplification.
    
    Args:
        duration: Total duration
        dt: Time step
        building_period: Building's natural period (T1)
        max_acc: Peak acceleration in gal
    
    Returns:
        t, acc (time and acceleration arrays)
    """
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    
    freq = 1.0 / building_period
    
    # Generate resonant input with gradual amplitude increase
    # This maximizes energy transfer to the building
    n_cycles = int(duration * freq)
    
    acc = np.zeros(n_steps)
    
    for i, ti in enumerate(t):
        cycle = ti * freq
        # Amplitude builds up over first few cycles, then sustains
        if cycle < 3:
            amp = cycle / 3
        elif cycle < n_cycles - 3:
            amp = 1.0
        else:
            amp = max(0, (n_cycles - cycle) / 3)
        
        # Add slight randomness to make it more realistic
        phase_noise = 0.1 * np.sin(2 * np.pi * 0.5 * ti)
        acc[i] = amp * np.sin(2 * np.pi * freq * ti + phase_noise)
    
    acc = acc * max_acc * 0.01  # Convert to m/s^2
    
    return t, acc


# Predefined earthquake scenarios
EARTHQUAKE_SCENARIOS = {
    'near_fault_pulse': {
        'name': 'Near-Fault Velocity Pulse',
        'description': 'High velocity pulse from nearby fault rupture. Period 3-5s. Severe for tall buildings.',
        'generator': 'long_period_pulse',
        'params': {'pulse_period': 4.0, 'max_vel': 120.0, 'pulse_type': 'velocity'}
    },
    'fling_step': {
        'name': 'Fling-Step Motion',
        'description': 'One-sided pulse with permanent displacement. Severe for column bases.',
        'generator': 'long_period_pulse', 
        'params': {'pulse_period': 3.0, 'max_vel': 80.0, 'pulse_type': 'fling'}
    },
    'distant_long_period': {
        'name': 'Distant Long-Period Motion',
        'description': 'Long-period motion from distant large earthquake (like 2011 Tohoku at Tokyo).',
        'generator': 'long_period_ground_motion',
        'params': {'period_range': (3.0, 7.0), 'max_acc': 80.0}
    },
    'resonance_attack': {
        'name': 'Resonance Match',
        'description': 'Ground motion tuned to building period. Maximum damage scenario.',
        'generator': 'resonance_pulse',
        'params': {}  # building_period set dynamically
    },
    'high_rise_killer': {
        'name': 'High-Rise Killer Pulse',
        'description': 'Long-period pulse (T=6s) targeting super-tall buildings.',
        'generator': 'long_period_pulse',
        'params': {'pulse_period': 6.0, 'max_vel': 100.0, 'pulse_type': 'velocity'}
    },
    'short_period': {
        'name': 'Short-Period Motion',
        'description': 'Typical shallow earthquake. Affects low-rise buildings more.',
        'generator': 'synthetic',
        'params': {'dominant_period': 0.5}
    }
}


def get_scenario_wave(scenario_key: str, duration: float, dt: float, 
                       building_period: float = None, max_acc: float = None):
    """
    Generate an earthquake wave for a predefined scenario.
    
    Args:
        scenario_key: Key from EARTHQUAKE_SCENARIOS
        duration: Duration in seconds
        dt: Time step
        building_period: Building's natural period (for resonance scenario)
        max_acc: Override max acceleration if desired
    
    Returns:
        t, acc, scenario_info
    """
    if scenario_key not in EARTHQUAKE_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}")
    
    scenario = EARTHQUAKE_SCENARIOS[scenario_key]
    params = scenario['params'].copy()
    
    if max_acc is not None:
        if 'max_acc' in params:
            params['max_acc'] = max_acc
        elif 'max_vel' in params:
            params['max_vel'] = max_acc  # Rough conversion
    
    generator = scenario['generator']
    
    if generator == 'long_period_pulse':
        t, acc, vel = generate_long_period_pulse(duration, dt, **params)
    elif generator == 'long_period_ground_motion':
        t, acc = generate_long_period_ground_motion(duration, dt, **params)
    elif generator == 'resonance_pulse':
        if building_period is None:
            building_period = 2.0  # Default
        t, acc = generate_resonance_pulse(duration, dt, building_period, 
                                          max_acc or 200.0)
    elif generator == 'synthetic':
        t, acc = generate_synthetic_wave(duration, dt, max_acc or 300.0, 
                                         params.get('dominant_period'))
    else:
        raise ValueError(f"Unknown generator: {generator}")
    
    return t, acc, scenario
