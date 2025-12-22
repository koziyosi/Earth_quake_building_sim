"""
JIS Standard Steel Section Database.
Provides standard Japanese steel sections (H-shapes, hollow sections, etc.)
with pre-calculated section properties.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import json


@dataclass 
class SteelSection:
    """Steel section with calculated properties."""
    name: str
    type: str  # 'H', 'BOX', 'PIPE', 'L', 'C'
    
    # Dimensions (mm)
    H: float  # Height
    B: float  # Width (flange)
    tw: float  # Web thickness
    tf: float  # Flange thickness
    r: float = 0.0  # Corner radius
    
    # Calculated properties (stored for quick access)
    A: float = 0.0      # Cross-sectional area (mm²)
    Ix: float = 0.0     # Moment of inertia about X (strong axis) (mm⁴)
    Iy: float = 0.0     # Moment of inertia about Y (weak axis) (mm⁴)
    Zx: float = 0.0     # Section modulus X (mm³)
    Zy: float = 0.0     # Section modulus Y (mm³)
    ix: float = 0.0     # Radius of gyration X (mm)
    iy: float = 0.0     # Radius of gyration Y (mm)
    J: float = 0.0      # Torsional constant (mm⁴)
    weight: float = 0.0  # Weight per meter (kg/m)
    
    def to_si_units(self) -> dict:
        """Convert properties to SI units (m, N) for FEM analysis."""
        return {
            'A': self.A * 1e-6,      # mm² -> m²
            'I_y': self.Iy * 1e-12,  # mm⁴ -> m⁴ (weak axis for columns)
            'I_z': self.Ix * 1e-12,  # mm⁴ -> m⁴ (strong axis)
            'J': self.J * 1e-12,     # mm⁴ -> m⁴
            'Z_x': self.Zx * 1e-9,   # mm³ -> m³
            'Z_y': self.Zy * 1e-9,   # mm³ -> m³
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'H': self.H, 'B': self.B, 'tw': self.tw, 'tf': self.tf, 'r': self.r,
            'A': self.A, 'Ix': self.Ix, 'Iy': self.Iy,
            'Zx': self.Zx, 'Zy': self.Zy, 'ix': self.ix, 'iy': self.iy,
            'J': self.J, 'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SteelSection':
        """Deserialize from dictionary."""
        return cls(**data)


# ============================================================================
# JIS H-Shape Steel Sections (Wide Flange)
# Based on JIS G 3192 / JIS A 5526
# ============================================================================

JIS_H_SECTIONS: Dict[str, SteelSection] = {
    # H-形鋼 (Wide Flange) - 一般的なサイズ
    
    # H-100シリーズ
    'H-100x100x6x8': SteelSection(
        name='H-100x100x6x8', type='H',
        H=100, B=100, tw=6, tf=8, r=10,
        A=2190, Ix=3.78e6, Iy=1.34e6, Zx=75.6e3, Zy=26.7e3,
        ix=41.5, iy=24.7, J=4.91e4, weight=17.2
    ),
    
    # H-150シリーズ
    'H-150x75x5x7': SteelSection(
        name='H-150x75x5x7', type='H',
        H=150, B=75, tw=5, tf=7, r=8,
        A=1770, Ix=6.66e6, Iy=0.496e6, Zx=88.8e3, Zy=13.2e3,
        ix=61.3, iy=16.7, J=3.52e4, weight=14.0
    ),
    'H-150x150x7x10': SteelSection(
        name='H-150x150x7x10', type='H',
        H=150, B=150, tw=7, tf=10, r=11,
        A=4010, Ix=16.4e6, Iy=5.63e6, Zx=219e3, Zy=75.1e3,
        ix=63.9, iy=37.5, J=15.2e4, weight=31.5
    ),
    
    # H-200シリーズ
    'H-200x100x5.5x8': SteelSection(
        name='H-200x100x5.5x8', type='H',
        H=200, B=100, tw=5.5, tf=8, r=11,
        A=2690, Ix=18.4e6, Iy=1.34e6, Zx=184e3, Zy=26.8e3,
        ix=82.6, iy=22.3, J=6.12e4, weight=21.3
    ),
    'H-200x200x8x12': SteelSection(
        name='H-200x200x8x12', type='H',
        H=200, B=200, tw=8, tf=12, r=13,
        A=6350, Ix=47.2e6, Iy=16.0e6, Zx=472e3, Zy=160e3,
        ix=86.2, iy=50.2, J=39.6e4, weight=49.9
    ),
    
    # H-250シリーズ
    'H-250x125x6x9': SteelSection(
        name='H-250x125x6x9', type='H',
        H=250, B=125, tw=6, tf=9, r=12,
        A=3610, Ix=40.9e6, Iy=2.94e6, Zx=327e3, Zy=47.0e3,
        ix=107, iy=28.5, J=10.8e4, weight=29.6
    ),
    'H-250x250x9x14': SteelSection(
        name='H-250x250x9x14', type='H', 
        H=250, B=250, tw=9, tf=14, r=13,
        A=9210, Ix=107e6, Iy=36.0e6, Zx=860e3, Zy=288e3,
        ix=108, iy=62.5, J=87.3e4, weight=72.4
    ),
    
    # H-300シリーズ
    'H-300x150x6.5x9': SteelSection(
        name='H-300x150x6.5x9', type='H',
        H=300, B=150, tw=6.5, tf=9, r=13,
        A=4680, Ix=72.0e6, Iy=5.08e6, Zx=480e3, Zy=67.7e3,
        ix=124, iy=32.9, J=13.0e4, weight=36.7
    ),
    'H-300x300x10x15': SteelSection(
        name='H-300x300x10x15', type='H',
        H=300, B=300, tw=10, tf=15, r=18,
        A=11900, Ix=201e6, Iy=67.5e6, Zx=1340e3, Zy=450e3,
        ix=130, iy=75.3, J=147e4, weight=93.0
    ),
    
    # H-350シリーズ
    'H-350x175x7x11': SteelSection(
        name='H-350x175x7x11', type='H',
        H=350, B=175, tw=7, tf=11, r=14,
        A=6310, Ix=136e6, Iy=9.84e6, Zx=777e3, Zy=112e3,
        ix=147, iy=39.5, J=25.2e4, weight=49.6
    ),
    'H-350x350x12x19': SteelSection(
        name='H-350x350x12x19', type='H',
        H=350, B=350, tw=12, tf=19, r=20,
        A=17400, Ix=403e6, Iy=136e6, Zx=2300e3, Zy=778e3,
        ix=152, iy=88.5, J=337e4, weight=137
    ),
    
    # H-400シリーズ
    'H-400x200x8x13': SteelSection(
        name='H-400x200x8x13', type='H',
        H=400, B=200, tw=8, tf=13, r=16,
        A=8410, Ix=237e6, Iy=17.4e6, Zx=1190e3, Zy=174e3,
        ix=168, iy=45.5, J=47.5e4, weight=66.0
    ),
    'H-400x400x13x21': SteelSection(
        name='H-400x400x13x21', type='H',
        H=400, B=400, tw=13, tf=21, r=22,
        A=21900, Ix=666e6, Iy=224e6, Zx=3330e3, Zy=1120e3,
        ix=174, iy=101, J=519e4, weight=172
    ),
    
    # H-500シリーズ
    'H-500x200x10x16': SteelSection(
        name='H-500x200x10x16', type='H',
        H=500, B=200, tw=10, tf=16, r=20,
        A=11200, Ix=472e6, Iy=21.4e6, Zx=1888e3, Zy=214e3,
        ix=205, iy=43.7, J=82.0e4, weight=89.6
    ),
    
    # H-600シリーズ
    'H-600x200x11x17': SteelSection(
        name='H-600x200x11x17', type='H',
        H=600, B=200, tw=11, tf=17, r=22,
        A=13400, Ix=769e6, Iy=22.8e6, Zx=2560e3, Zy=228e3,
        ix=239, iy=41.2, J=107e4, weight=106
    ),
}


# ============================================================================
# JIS Box/Hollow Sections (角形鋼管)
# Based on JIS G 3466
# ============================================================================

JIS_BOX_SECTIONS: Dict[str, SteelSection] = {
    # 正方形断面
    'BOX-100x100x3.2': SteelSection(
        name='BOX-100x100x3.2', type='BOX',
        H=100, B=100, tw=3.2, tf=3.2, r=9.6,
        A=1190, Ix=1.98e6, Iy=1.98e6, Zx=39.6e3, Zy=39.6e3,
        ix=40.8, iy=40.8, J=3.12e6, weight=9.34
    ),
    'BOX-125x125x4.5': SteelSection(
        name='BOX-125x125x4.5', type='BOX',
        H=125, B=125, tw=4.5, tf=4.5, r=13.5,
        A=2090, Ix=4.88e6, Iy=4.88e6, Zx=78.1e3, Zy=78.1e3,
        ix=48.3, iy=48.3, J=7.73e6, weight=16.4
    ),
    'BOX-150x150x6': SteelSection(
        name='BOX-150x150x6', type='BOX',
        H=150, B=150, tw=6, tf=6, r=18,
        A=3360, Ix=10.8e6, Iy=10.8e6, Zx=144e3, Zy=144e3,
        ix=56.7, iy=56.7, J=17.2e6, weight=26.4
    ),
    'BOX-175x175x6': SteelSection(
        name='BOX-175x175x6', type='BOX',
        H=175, B=175, tw=6, tf=6, r=18,
        A=3960, Ix=17.5e6, Iy=17.5e6, Zx=200e3, Zy=200e3,
        ix=66.5, iy=66.5, J=27.8e6, weight=31.1
    ),
    'BOX-200x200x6': SteelSection(
        name='BOX-200x200x6', type='BOX',
        H=200, B=200, tw=6, tf=6, r=18,
        A=4560, Ix=26.8e6, Iy=26.8e6, Zx=268e3, Zy=268e3,
        ix=76.6, iy=76.6, J=42.6e6, weight=35.8
    ),
    'BOX-200x200x9': SteelSection(
        name='BOX-200x200x9', type='BOX',
        H=200, B=200, tw=9, tf=9, r=27,
        A=6550, Ix=37.1e6, Iy=37.1e6, Zx=371e3, Zy=371e3,
        ix=75.2, iy=75.2, J=59.3e6, weight=51.4
    ),
    'BOX-250x250x9': SteelSection(
        name='BOX-250x250x9', type='BOX',
        H=250, B=250, tw=9, tf=9, r=27,
        A=8350, Ix=77.2e6, Iy=77.2e6, Zx=617e3, Zy=617e3,
        ix=96.2, iy=96.2, J=123e6, weight=65.5
    ),
    'BOX-300x300x9': SteelSection(
        name='BOX-300x300x9', type='BOX',
        H=300, B=300, tw=9, tf=9, r=27,
        A=10100, Ix=137e6, Iy=137e6, Zx=916e3, Zy=916e3,
        ix=117, iy=117, J=219e6, weight=79.7
    ),
    'BOX-300x300x12': SteelSection(
        name='BOX-300x300x12', type='BOX',
        H=300, B=300, tw=12, tf=12, r=36,
        A=13200, Ix=173e6, Iy=173e6, Zx=1150e3, Zy=1150e3,
        ix=115, iy=115, J=278e6, weight=103
    ),
    'BOX-350x350x12': SteelSection(
        name='BOX-350x350x12', type='BOX',
        H=350, B=350, tw=12, tf=12, r=36,
        A=15600, Ix=283e6, Iy=283e6, Zx=1620e3, Zy=1620e3,
        ix=135, iy=135, J=454e6, weight=123
    ),
    'BOX-400x400x12': SteelSection(
        name='BOX-400x400x12', type='BOX',
        H=400, B=400, tw=12, tf=12, r=36,
        A=18000, Ix=427e6, Iy=427e6, Zx=2140e3, Zy=2140e3,
        ix=154, iy=154, J=685e6, weight=141
    ),
    
    # 長方形断面
    'BOX-150x100x4.5': SteelSection(
        name='BOX-150x100x4.5', type='BOX',
        H=150, B=100, tw=4.5, tf=4.5, r=13.5,
        A=2090, Ix=5.49e6, Iy=2.88e6, Zx=73.2e3, Zy=57.6e3,
        ix=51.2, iy=37.1, J=6.64e6, weight=16.4
    ),
    'BOX-200x100x6': SteelSection(
        name='BOX-200x100x6', type='BOX',
        H=200, B=100, tw=6, tf=6, r=18,
        A=3360, Ix=12.7e6, Iy=4.48e6, Zx=127e3, Zy=89.6e3,
        ix=61.5, iy=36.5, J=12.4e6, weight=26.4
    ),
}


# ============================================================================
# JIS Pipe Sections (鋼管)
# Based on JIS G 3444 / JIS G 3452
# ============================================================================

JIS_PIPE_SECTIONS: Dict[str, SteelSection] = {
    'PIPE-114.3x4.5': SteelSection(
        name='PIPE-114.3x4.5', type='PIPE',
        H=114.3, B=114.3, tw=4.5, tf=4.5,
        A=1550, Ix=2.36e6, Iy=2.36e6, Zx=41.3e3, Zy=41.3e3,
        ix=39.0, iy=39.0, J=4.72e6, weight=12.2
    ),
    'PIPE-139.8x4.5': SteelSection(
        name='PIPE-139.8x4.5', type='PIPE',
        H=139.8, B=139.8, tw=4.5, tf=4.5,
        A=1910, Ix=4.49e6, Iy=4.49e6, Zx=64.3e3, Zy=64.3e3,
        ix=48.5, iy=48.5, J=8.98e6, weight=15.0
    ),
    'PIPE-165.2x5.0': SteelSection(
        name='PIPE-165.2x5.0', type='PIPE',
        H=165.2, B=165.2, tw=5.0, tf=5.0,
        A=2520, Ix=8.25e6, Iy=8.25e6, Zx=99.9e3, Zy=99.9e3,
        ix=57.2, iy=57.2, J=16.5e6, weight=19.8
    ),
    'PIPE-216.3x5.8': SteelSection(
        name='PIPE-216.3x5.8', type='PIPE',
        H=216.3, B=216.3, tw=5.8, tf=5.8,
        A=3840, Ix=20.7e6, Iy=20.7e6, Zx=192e3, Zy=192e3,
        ix=73.4, iy=73.4, J=41.4e6, weight=30.1
    ),
    'PIPE-267.4x6.6': SteelSection(
        name='PIPE-267.4x6.6', type='PIPE',
        H=267.4, B=267.4, tw=6.6, tf=6.6,
        A=5410, Ix=44.6e6, Iy=44.6e6, Zx=334e3, Zy=334e3,
        ix=90.8, iy=90.8, J=89.2e6, weight=42.4
    ),
    'PIPE-318.5x6.9': SteelSection(
        name='PIPE-318.5x6.9', type='PIPE',
        H=318.5, B=318.5, tw=6.9, tf=6.9,
        A=6760, Ix=81.1e6, Iy=81.1e6, Zx=509e3, Zy=509e3,
        ix=110, iy=110, J=162e6, weight=53.0
    ),
}


class SectionDatabase:
    """
    Unified section database with search and filtering capabilities.
    """
    
    def __init__(self):
        self.sections: Dict[str, SteelSection] = {}
        self._load_jis_sections()
    
    def _load_jis_sections(self):
        """Load all JIS standard sections."""
        self.sections.update(JIS_H_SECTIONS)
        self.sections.update(JIS_BOX_SECTIONS)
        self.sections.update(JIS_PIPE_SECTIONS)
    
    def get_section(self, name: str) -> Optional[SteelSection]:
        """Get section by name."""
        return self.sections.get(name)
    
    def get_sections_by_type(self, section_type: str) -> List[SteelSection]:
        """Get all sections of a specific type."""
        return [s for s in self.sections.values() if s.type == section_type]
    
    def get_sections_by_height_range(self, min_h: float, max_h: float) -> List[SteelSection]:
        """Get sections with height in specified range."""
        return [s for s in self.sections.values() if min_h <= s.H <= max_h]
    
    def search(self, query: str) -> List[SteelSection]:
        """Search sections by name (partial match)."""
        query_lower = query.lower()
        return [s for s in self.sections.values() if query_lower in s.name.lower()]
    
    def get_all_names(self) -> List[str]:
        """Get list of all section names."""
        return sorted(self.sections.keys())
    
    def get_column_recommendations(self, floor_count: int) -> List[str]:
        """Get recommended column sections based on building height."""
        if floor_count <= 3:
            return ['H-250x250x9x14', 'BOX-200x200x9', 'BOX-250x250x9']
        elif floor_count <= 6:
            return ['H-300x300x10x15', 'H-350x350x12x19', 'BOX-300x300x9']
        elif floor_count <= 10:
            return ['H-400x400x13x21', 'BOX-350x350x12', 'BOX-400x400x12']
        else:
            return ['H-400x400x13x21', 'BOX-400x400x12']
    
    def get_beam_recommendations(self, span: float) -> List[str]:
        """Get recommended beam sections based on span length (m)."""
        if span <= 6:
            return ['H-300x150x6.5x9', 'H-350x175x7x11']
        elif span <= 9:
            return ['H-400x200x8x13', 'H-500x200x10x16']
        else:
            return ['H-500x200x10x16', 'H-600x200x11x17']
    
    def add_custom_section(self, section: SteelSection):
        """Add a custom section to the database."""
        self.sections[section.name] = section
    
    def export_to_json(self, filepath: str):
        """Export database to JSON file."""
        data = {name: sec.to_dict() for name, sec in self.sections.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, filepath: str):
        """Import sections from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for name, sec_data in data.items():
            self.sections[name] = SteelSection.from_dict(sec_data)


# Global database instance
SECTION_DB = SectionDatabase()


def get_section_properties_for_fem(section_name: str, E: float = 2.05e11) -> dict:
    """
    Get section properties ready for FEM model creation.
    
    Args:
        section_name: Name of the section (e.g., 'H-300x300x10x15')
        E: Young's modulus (Pa), default for steel
        
    Returns:
        Dictionary with E, G, A, Iy, Iz, J in SI units
    """
    section = SECTION_DB.get_section(section_name)
    if not section:
        raise ValueError(f"Section '{section_name}' not found in database")
    
    si = section.to_si_units()
    G = E / (2 * (1 + 0.3))  # Assuming Poisson's ratio of 0.3
    
    return {
        'E': E,
        'G': G,
        'A': si['A'],
        'Iy': si['I_y'],  # Weak axis
        'Iz': si['I_z'],  # Strong axis
        'J': si['J'],
    }
