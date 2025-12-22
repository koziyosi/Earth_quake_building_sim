"""
Earthquake Database Connector.
Downloads and parses earthquake records from K-NET/KiK-net (NIED).
"""
import os
import io
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class EarthquakeRecord:
    """Container for earthquake record data."""
    station_code: str
    origin_time: str
    magnitude: float
    latitude: float
    longitude: float
    depth: float  # km
    pga_ns: float  # Peak Ground Acceleration NS (gal)
    pga_ew: float  # Peak Ground Acceleration EW (gal)
    pga_ud: float  # Peak Ground Acceleration UD (gal)
    sampling_freq: float  # Hz
    time_array: np.ndarray
    acc_ns: np.ndarray  # NS component (m/s²)
    acc_ew: np.ndarray  # EW component (m/s²)
    acc_ud: np.ndarray  # UD component (m/s²)


class KNETParser:
    """Parser for K-NET/KiK-net earthquake data files."""
    
    @staticmethod
    def parse_file(filepath: str) -> Optional[EarthquakeRecord]:
        """
        Parse a K-NET format earthquake file.
        
        K-NET format has a header section followed by acceleration data.
        Header contains station info, earthquake info, and scale factors.
        
        Args:
            filepath: Path to K-NET file
            
        Returns:
            EarthquakeRecord or None if parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
            
        # Parse header
        header = {}
        data_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('Memo'):
                data_start = i + 1
                break
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ''
                header[key] = value
        
        # Extract metadata
        station = header.get('Station Code', 'Unknown')
        origin_time = header.get('Origin Time', '')
        
        # Magnitude
        mag_str = header.get('Magnitude(JMA)', header.get('Magnitude', '0'))
        try:
            magnitude = float(mag_str.split()[0])
        except:
            magnitude = 0.0
            
        # Location
        try:
            lat = float(header.get('Lat.', '0'))
            lon = float(header.get('Long.', '0'))
            depth = float(header.get('Depth(km)', '0').replace('km', ''))
        except:
            lat, lon, depth = 0.0, 0.0, 0.0
            
        # Sampling
        try:
            freq_str = header.get('Sampling Freq(Hz)', '100')
            sampling_freq = float(freq_str.replace('Hz', ''))
        except:
            sampling_freq = 100.0
            
        # Scale factor (gal conversion)
        try:
            scale_str = header.get('Scale Factor', header.get('Scale', '1'))
            # Format often: "1E+3(gal)/1E+6" or similar
            if '/' in scale_str:
                nums = scale_str.replace('(gal)', '').split('/')
                scale = float(nums[0]) / float(nums[1].strip())
            else:
                scale = float(scale_str.replace('(gal)', ''))
        except:
            scale = 1.0
            
        # Parse acceleration data
        data_lines = lines[data_start:]
        acc_values = []
        
        for line in data_lines:
            parts = line.strip().split()
            for p in parts:
                try:
                    acc_values.append(float(p) * scale)  # Apply scale (result in gal)
                except:
                    pass
        
        acc_gal = np.array(acc_values)
        acc_mps2 = acc_gal * 0.01  # Convert gal to m/s²
        
        # Time array
        dt = 1.0 / sampling_freq
        time_array = np.arange(len(acc_mps2)) * dt
        
        # Detect component from filename
        filepath_lower = filepath.lower()
        is_ns = 'ns' in filepath_lower or 'n' in Path(filepath).stem[-2:]
        is_ew = 'ew' in filepath_lower or 'e' in Path(filepath).stem[-2:]
        is_ud = 'ud' in filepath_lower or 'u' in Path(filepath).stem[-2:]
        
        # Default to single component if unclear
        acc_ns = acc_mps2 if is_ns else np.zeros_like(acc_mps2)
        acc_ew = acc_mps2 if is_ew else np.zeros_like(acc_mps2)
        acc_ud = acc_mps2 if is_ud else np.zeros_like(acc_mps2)
        
        # PGA
        pga_ns = np.max(np.abs(acc_ns)) * 100  # Back to gal for record
        pga_ew = np.max(np.abs(acc_ew)) * 100
        pga_ud = np.max(np.abs(acc_ud)) * 100
        
        return EarthquakeRecord(
            station_code=station,
            origin_time=origin_time,
            magnitude=magnitude,
            latitude=lat,
            longitude=lon,
            depth=depth,
            pga_ns=pga_ns,
            pga_ew=pga_ew,
            pga_ud=pga_ud,
            sampling_freq=sampling_freq,
            time_array=time_array,
            acc_ns=acc_ns,
            acc_ew=acc_ew,
            acc_ud=acc_ud
        )
    
    @staticmethod
    def parse_directory(dirpath: str) -> List[EarthquakeRecord]:
        """Parse all K-NET files in a directory."""
        records = []
        
        for file in Path(dirpath).glob('*.ns*'):
            record = KNETParser.parse_file(str(file))
            if record:
                records.append(record)
                
        for file in Path(dirpath).glob('*.ew*'):
            record = KNETParser.parse_file(str(file))
            if record:
                records.append(record)
                
        return records


class EarthquakeDatabase:
    """Simple in-memory database for earthquake records."""
    
    def __init__(self):
        self.records: List[EarthquakeRecord] = []
        
    def add_record(self, record: EarthquakeRecord):
        """Add a record to the database."""
        self.records.append(record)
        
    def load_from_directory(self, dirpath: str):
        """Load all records from a directory."""
        records = KNETParser.parse_directory(dirpath)
        self.records.extend(records)
        
    def search(
        self,
        min_magnitude: Optional[float] = None,
        max_magnitude: Optional[float] = None,
        min_pga: Optional[float] = None,
        station_code: Optional[str] = None
    ) -> List[EarthquakeRecord]:
        """Search records by criteria."""
        results = []
        
        for r in self.records:
            if min_magnitude and r.magnitude < min_magnitude:
                continue
            if max_magnitude and r.magnitude > max_magnitude:
                continue
            if min_pga:
                max_pga = max(r.pga_ns, r.pga_ew, r.pga_ud)
                if max_pga < min_pga:
                    continue
            if station_code and station_code not in r.station_code:
                continue
            results.append(r)
            
        return results
    
    def get_by_index(self, idx: int) -> Optional[EarthquakeRecord]:
        """Get record by index."""
        if 0 <= idx < len(self.records):
            return self.records[idx]
        return None


# Built-in sample earthquakes (synthetic data for testing)
def get_sample_earthquakes() -> List[Dict]:
    """Get list of sample earthquake metadata."""
    return [
        {
            'name': '1995 Kobe (Synthetic)',
            'magnitude': 6.9,
            'pga': 818,
            'station': 'JMA Kobe',
        },
        {
            'name': '2011 Tohoku (Synthetic)',
            'magnitude': 9.0,
            'pga': 2700,
            'station': 'K-NET MYG004',
        },
        {
            'name': '2016 Kumamoto (Synthetic)',
            'magnitude': 7.0,
            'pga': 1580,
            'station': 'K-NET KMM004',
        }
    ]
