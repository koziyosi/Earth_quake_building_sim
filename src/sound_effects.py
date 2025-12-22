"""
Sound Effects Module.
Audio feedback for simulation events.
"""
import numpy as np
from typing import Optional
import wave
import io
import struct
import os


class SoundGenerator:
    """
    Generates audio feedback for simulation events.
    Uses pure Python - no external audio libraries required.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.sounds = {}
        self._generate_default_sounds()
        
    def _generate_default_sounds(self):
        """Generate default sound effects."""
        self.sounds['earthquake_start'] = self._create_rumble(duration=1.0)
        self.sounds['earthquake_peak'] = self._create_impact(duration=0.3)
        self.sounds['crack'] = self._create_crack(duration=0.2)
        self.sounds['collapse'] = self._create_collapse(duration=1.5)
        self.sounds['warning'] = self._create_warning(duration=0.5)
        self.sounds['complete'] = self._create_complete(duration=0.3)
        
    def _create_rumble(self, duration: float = 1.0) -> np.ndarray:
        """Low frequency rumble for earthquake."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Multiple low frequencies
        signal = np.zeros_like(t)
        for freq in [30, 45, 60, 80]:
            signal += np.sin(2 * np.pi * freq * t) * np.random.uniform(0.5, 1)
            
        # Add random noise
        signal += np.random.normal(0, 0.1, len(t))
        
        # Envelope
        envelope = np.exp(-t * 0.5)
        signal *= envelope
        
        return self._normalize(signal)
        
    def _create_impact(self, duration: float = 0.3) -> np.ndarray:
        """Impact sound for peak earthquake."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Initial burst
        signal = np.sin(2 * np.pi * 100 * t) * np.exp(-t * 20)
        signal += np.sin(2 * np.pi * 50 * t) * np.exp(-t * 10)
        
        # Add impact noise
        noise = np.random.normal(0, 0.3, len(t)) * np.exp(-t * 15)
        signal += noise
        
        return self._normalize(signal)
        
    def _create_crack(self, duration: float = 0.2) -> np.ndarray:
        """Cracking/breaking sound for structural damage."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Sharp initial crack
        signal = np.exp(-t * 50) * np.random.normal(0, 1, len(t))
        
        # Some resonance
        signal += 0.3 * np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 30)
        
        return self._normalize(signal)
        
    def _create_collapse(self, duration: float = 1.5) -> np.ndarray:
        """Collapse sound for building failure."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Initial impact
        signal = np.exp(-t * 5) * np.random.normal(0, 1, len(t))
        
        # Rolling debris
        for i in range(5):
            delay = i * 0.2
            mask = t > delay
            signal[mask] += 0.3 * np.exp(-(t[mask] - delay) * 3) * np.random.normal(0, 0.5, np.sum(mask))
            
        # Low rumble
        signal += 0.5 * np.sin(2 * np.pi * 40 * t) * np.exp(-t * 2)
        
        return self._normalize(signal)
        
    def _create_warning(self, duration: float = 0.5) -> np.ndarray:
        """Warning beep sound."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        freq = 880  # A5
        signal = np.sin(2 * np.pi * freq * t)
        
        # Pulse envelope
        pulse = np.sin(2 * np.pi * 4 * t)
        pulse = np.where(pulse > 0, 1, 0.3)
        signal *= pulse
        
        return self._normalize(signal)
        
    def _create_complete(self, duration: float = 0.3) -> np.ndarray:
        """Completion chime."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Two-tone chime
        freq1, freq2 = 523, 659  # C5 and E5
        signal = np.sin(2 * np.pi * freq1 * t)
        signal[len(t)//2:] = np.sin(2 * np.pi * freq2 * t[len(t)//2:])
        signal *= np.exp(-t * 3)
        
        return self._normalize(signal)
        
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]."""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.9
        return signal
        
    def get_sound(self, name: str) -> Optional[np.ndarray]:
        """Get a sound by name."""
        return self.sounds.get(name)
        
    def save_sound_wav(self, name: str, filepath: str):
        """Save a sound to WAV file."""
        if name not in self.sounds:
            raise ValueError(f"Sound '{name}' not found")
            
        signal = self.sounds[name]
        self._write_wav(signal, filepath)
        
    def _write_wav(self, signal: np.ndarray, filepath: str):
        """Write audio signal to WAV file."""
        # Convert to 16-bit integers
        audio_int = (signal * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int.tobytes())
            
    def play_sound_tk(self, name: str, root):
        """
        Play sound using Tkinter's winsound (Windows only).
        """
        try:
            import winsound
            
            # Create temporary WAV file
            temp_path = os.path.join(os.environ.get('TEMP', '.'), f'eq_sim_{name}.wav')
            self.save_sound_wav(name, temp_path)
            
            # Play asynchronously
            winsound.PlaySound(temp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass  # Silent fail on non-Windows


class EventSoundManager:
    """
    Manages sound playback based on simulation events.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.generator = SoundGenerator()
        self.root = None
        
    def set_root(self, root):
        """Set Tkinter root for audio playback."""
        self.root = root
        
    def on_simulation_start(self):
        """Called when simulation starts."""
        if self.enabled and self.root:
            self.generator.play_sound_tk('earthquake_start', self.root)
            
    def on_peak_acceleration(self):
        """Called at peak ground acceleration."""
        if self.enabled and self.root:
            self.generator.play_sound_tk('earthquake_peak', self.root)
            
    def on_element_yield(self, element_id: int):
        """Called when an element yields."""
        if self.enabled and self.root:
            self.generator.play_sound_tk('crack', self.root)
            
    def on_element_failure(self, element_id: int):
        """Called when an element fails."""
        if self.enabled and self.root:
            self.generator.play_sound_tk('collapse', self.root)
            
    def on_simulation_complete(self):
        """Called when simulation completes."""
        if self.enabled and self.root:
            self.generator.play_sound_tk('complete', self.root)
            
    def on_warning(self, message: str):
        """Called on warning events."""
        if self.enabled and self.root:
            self.generator.play_sound_tk('warning', self.root)
            
    def toggle(self):
        """Toggle sound effects on/off."""
        self.enabled = not self.enabled
        return self.enabled
