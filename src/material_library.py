"""
Material Library for Structural Analysis.
Provides Japanese standard materials with physical properties.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json


# ============================================================================
# Joint Properties (接合部特性)
# ============================================================================

@dataclass
class JointProperties:
    """Joint/Connection properties (接合部特性)."""
    name: str
    display_name: str = ""
    
    # 接合部タイプ
    joint_type: str = 'rigid'       # 'rigid', 'pinned', 'semi_rigid'
    
    # 剛性 (semi_rigid用)
    rotational_stiffness: float = 1e12  # 回転剛性 (N·m/rad)
    
    # 耐力
    moment_capacity: float = 0.0     # モーメント容量 (N·m) - 0 = 無制限
    shear_capacity: float = 0.0      # せん断容量 (N)
    
    # ヒステリシス特性
    hysteresis_type: str = 'takeda'  # 'takeda', 'bilinear', 'elastic'
    post_yield_ratio: float = 0.05   # 降伏後剛性比
    pinching_factor: float = 0.8     # ピンチング係数 (0-1)
    degradation_alpha: float = 0.4   # 剛性劣化指数 (Takeda)
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'joint_type': self.joint_type,
            'rotational_stiffness': self.rotational_stiffness,
            'moment_capacity': self.moment_capacity,
            'shear_capacity': self.shear_capacity,
            'hysteresis_type': self.hysteresis_type,
            'post_yield_ratio': self.post_yield_ratio,
            'pinching_factor': self.pinching_factor,
            'degradation_alpha': self.degradation_alpha
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'JointProperties':
        return cls(**data)


# ============================================================================
# Analysis Settings (解析設定)
# ============================================================================

@dataclass
class AnalysisSettings:
    """Global analysis parameters (解析設定)."""
    
    # 時間積分パラメータ
    time_step: float = 0.005         # 時間刻み (s)
    beta: float = 0.25               # Newmark β
    gamma: float = 0.5               # Newmark γ
    
    # Newton-Raphson設定
    max_iterations: int = 10         # 最大反復回数
    tolerance: float = 1e-3          # 収束判定誤差
    
    # 減衰設定
    damping_type: str = 'rayleigh'   # 'rayleigh', 'modal', 'caughey'
    damping_ratio: float = 0.05      # 減衰定数
    
    # 幾何学的非線形
    p_delta_enabled: bool = False    # P-Delta効果
    
    # 数値安定性
    line_search: bool = False        # ライン探索
    max_displacement_step: float = 0.5  # 最大変位増分 (m)
    
    # 質量マトリックス
    mass_type: str = 'lumped'        # 'lumped', 'consistent'
    
    # 出力設定
    output_interval: int = 1         # 出力間隔（ステップ）
    save_element_forces: bool = True # 要素力を保存
    
    def to_dict(self) -> dict:
        return {
            'time_step': self.time_step,
            'beta': self.beta,
            'gamma': self.gamma,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'damping_type': self.damping_type,
            'damping_ratio': self.damping_ratio,
            'p_delta_enabled': self.p_delta_enabled,
            'line_search': self.line_search,
            'max_displacement_step': self.max_displacement_step,
            'mass_type': self.mass_type,
            'output_interval': self.output_interval,
            'save_element_forces': self.save_element_forces
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnalysisSettings':
        return cls(**data)


# Default analysis settings
DEFAULT_ANALYSIS_SETTINGS = AnalysisSettings()


# ============================================================================
# Joint Library (接合部ライブラリ)
# ============================================================================

JOINT_LIBRARY: Dict[str, JointProperties] = {
    'rigid': JointProperties(
        name='rigid',
        display_name='剛接合',
        joint_type='rigid'
    ),
    'pinned': JointProperties(
        name='pinned',
        display_name='ピン接合',
        joint_type='pinned',
        rotational_stiffness=1e6
    ),
    'semi_rigid_rc': JointProperties(
        name='semi_rigid_rc',
        display_name='半剛接合(RC)',
        joint_type='semi_rigid',
        rotational_stiffness=1e10,
        hysteresis_type='takeda',
        degradation_alpha=0.4
    ),
    'semi_rigid_steel': JointProperties(
        name='semi_rigid_steel',
        display_name='半剛接合(S)',
        joint_type='semi_rigid',
        rotational_stiffness=5e10,
        hysteresis_type='bilinear',
        post_yield_ratio=0.03
    ),
    'base_fixed': JointProperties(
        name='base_fixed',
        display_name='固定端',
        joint_type='rigid',
        moment_capacity=0
    ),
    'base_pinned': JointProperties(
        name='base_pinned',
        display_name='ピン支点',
        joint_type='pinned'
    ),
}


class MaterialType(Enum):
    """Material type enumeration."""
    STEEL = "steel"
    CONCRETE = "concrete"
    REBAR = "rebar"
    WOOD = "wood"


@dataclass
class MaterialProperties:
    """
    Material properties for structural analysis.
    All values in SI units (Pa, kg/m³, etc.)
    """
    name: str
    type: MaterialType
    
    # Basic Properties
    E: float           # Young's modulus (Pa)
    G: float           # Shear modulus (Pa)
    nu: float          # Poisson's ratio
    rho: float         # Density (kg/m³)
    
    # Strength Properties
    Fy: float = 0.0    # Yield strength (Pa)
    Fu: float = 0.0    # Ultimate strength (Pa)
    Fc: float = 0.0    # Compressive strength (Pa) - for concrete
    
    # Thermal Properties
    alpha: float = 0.0  # Thermal expansion coefficient (1/°C)
    
    # Display
    color: str = '#808080'  # Color for visualization
    description: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'E': self.E, 'G': self.G, 'nu': self.nu, 'rho': self.rho,
            'Fy': self.Fy, 'Fu': self.Fu, 'Fc': self.Fc,
            'alpha': self.alpha,
            'color': self.color,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MaterialProperties':
        """Deserialize from dictionary."""
        data['type'] = MaterialType(data['type'])
        return cls(**data)


# ============================================================================
# Japanese Standard Steel Materials (JIS G 3101, G 3106, G 3136)
# ============================================================================

STEEL_MATERIALS: Dict[str, MaterialProperties] = {
    # 一般構造用圧延鋼材 (JIS G 3101)
    'SS400': MaterialProperties(
        name='SS400',
        type=MaterialType.STEEL,
        E=2.05e11,  # 205 GPa
        G=7.9e10,   # 79 GPa
        nu=0.3,
        rho=7850,   # kg/m³
        Fy=235e6,   # 235 MPa (板厚 16mm以下)
        Fu=400e6,   # 400-510 MPa
        alpha=12e-6,
        color='#4472C4',
        description='一般構造用圧延鋼材 (汎用)'
    ),
    
    # 溶接構造用圧延鋼材 (JIS G 3106)
    'SM400A': MaterialProperties(
        name='SM400A',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=235e6, Fu=400e6,
        alpha=12e-6,
        color='#5B9BD5',
        description='溶接構造用圧延鋼材 A種'
    ),
    'SM490A': MaterialProperties(
        name='SM490A',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=315e6,  # 315 MPa
        Fu=490e6,  # 490-610 MPa
        alpha=12e-6,
        color='#2E75B6',
        description='溶接構造用圧延鋼材 490N級'
    ),
    'SM490YA': MaterialProperties(
        name='SM490YA',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=325e6, Fu=490e6,
        alpha=12e-6,
        color='#1F4E79',
        description='溶接構造用圧延鋼材 490N級 降伏点指定'
    ),
    
    # 建築構造用圧延鋼材 (JIS G 3136)
    'SN400A': MaterialProperties(
        name='SN400A',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=235e6, Fu=400e6,
        alpha=12e-6,
        color='#70AD47',
        description='建築構造用圧延鋼材 400N級 A種 (主に非塑性化部材)'
    ),
    'SN400B': MaterialProperties(
        name='SN400B',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=235e6, Fu=400e6,
        alpha=12e-6,
        color='#548235',
        description='建築構造用圧延鋼材 400N級 B種 (塑性化部材用)'
    ),
    'SN400C': MaterialProperties(
        name='SN400C',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=235e6, Fu=400e6,
        alpha=12e-6,
        color='#375623',
        description='建築構造用圧延鋼材 400N級 C種 (板厚方向特性保証)'
    ),
    'SN490B': MaterialProperties(
        name='SN490B',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=325e6, Fu=490e6,
        alpha=12e-6,
        color='#C55A11',
        description='建築構造用圧延鋼材 490N級 B種'
    ),
    'SN490C': MaterialProperties(
        name='SN490C',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=325e6, Fu=490e6,
        alpha=12e-6,
        color='#833C0C',
        description='建築構造用圧延鋼材 490N級 C種'
    ),
    
    # 角形鋼管 (JIS G 3466)
    'STKR400': MaterialProperties(
        name='STKR400',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=235e6, Fu=400e6,
        alpha=12e-6,
        color='#7030A0',
        description='一般構造用角形鋼管'
    ),
    'STKR490': MaterialProperties(
        name='STKR490',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=315e6, Fu=490e6,
        alpha=12e-6,
        color='#5B285B',
        description='高張力構造用角形鋼管'
    ),
    
    # 建築構造用冷間成形角形鋼管 (BCR, BCP)
    'BCR295': MaterialProperties(
        name='BCR295',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=295e6, Fu=400e6,
        alpha=12e-6,
        color='#ED7D31',
        description='建築構造用冷間ロール成形角形鋼管'
    ),
    'BCP325': MaterialProperties(
        name='BCP325',
        type=MaterialType.STEEL,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=325e6, Fu=490e6,
        alpha=12e-6,
        color='#C00000',
        description='建築構造用冷間プレス成形角形鋼管'
    ),
}


# ============================================================================
# Japanese Standard Concrete Materials (JASS 5)
# ============================================================================

def create_concrete(fc: float, name: str = None) -> MaterialProperties:
    """
    Create concrete material properties based on compressive strength.
    
    Args:
        fc: Compressive strength in MPa (e.g., 24 for Fc24)
        name: Optional name override
        
    Returns:
        MaterialProperties for concrete
    """
    fc_pa = fc * 1e6  # Convert to Pa
    
    # Young's modulus: E = 3.35e4 * (γ/24)^2 * (Fc/60)^(1/3) [N/mm²]
    # Simplified: E ≈ 21000 * sqrt(Fc/10) [MPa] for normal weight concrete
    E = 2.1e10 * (fc / 10) ** 0.5  # Pa
    
    # Shear modulus G = E / (2 * (1 + nu))
    nu = 0.2
    G = E / (2 * (1 + nu))
    
    return MaterialProperties(
        name=name or f'Fc{int(fc)}',
        type=MaterialType.CONCRETE,
        E=E,
        G=G,
        nu=nu,
        rho=2400,  # kg/m³ for normal weight concrete
        Fc=fc_pa,
        Fy=0,  # Tensile strength neglected
        Fu=fc_pa,  # Ultimate = compressive
        alpha=10e-6,
        color='#A6A6A6',
        description=f'普通コンクリート 設計基準強度 {fc}N/mm²'
    )


CONCRETE_MATERIALS: Dict[str, MaterialProperties] = {
    'Fc18': create_concrete(18),
    'Fc21': create_concrete(21),
    'Fc24': create_concrete(24),
    'Fc27': create_concrete(27),
    'Fc30': create_concrete(30),
    'Fc33': create_concrete(33),
    'Fc36': create_concrete(36),
    'Fc40': create_concrete(40),
    'Fc42': create_concrete(42),
    'Fc45': create_concrete(45),
    'Fc50': create_concrete(50),
    'Fc60': create_concrete(60),
}


# ============================================================================
# Japanese Standard Reinforcing Steel (JIS G 3112)
# ============================================================================

REBAR_MATERIALS: Dict[str, MaterialProperties] = {
    'SD295A': MaterialProperties(
        name='SD295A',
        type=MaterialType.REBAR,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=295e6, Fu=440e6,
        alpha=12e-6,
        color='#00B050',
        description='異形棒鋼 SD295A (D10-D16)'
    ),
    'SD295B': MaterialProperties(
        name='SD295B',
        type=MaterialType.REBAR,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=295e6, Fu=440e6,
        alpha=12e-6,
        color='#00B050',
        description='異形棒鋼 SD295B (D19-D41)'
    ),
    'SD345': MaterialProperties(
        name='SD345',
        type=MaterialType.REBAR,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=345e6, Fu=490e6,
        alpha=12e-6,
        color='#FFC000',
        description='異形棒鋼 SD345 (主筋・せん断補強筋)'
    ),
    'SD390': MaterialProperties(
        name='SD390',
        type=MaterialType.REBAR,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=390e6, Fu=560e6,
        alpha=12e-6,
        color='#FF6600',
        description='異形棒鋼 SD390 (高強度)'
    ),
    'SD490': MaterialProperties(
        name='SD490',
        type=MaterialType.REBAR,
        E=2.05e11, G=7.9e10, nu=0.3, rho=7850,
        Fy=490e6, Fu=620e6,
        alpha=12e-6,
        color='#FF0000',
        description='異形棒鋼 SD490 (超高強度)'
    ),
}


# ============================================================================
# Material Library Class
# ============================================================================

class MaterialLibrary:
    """
    Unified material library with search and management capabilities.
    """
    
    def __init__(self):
        self.materials: Dict[str, MaterialProperties] = {}
        self._load_standard_materials()
    
    def _load_standard_materials(self):
        """Load all standard materials."""
        self.materials.update(STEEL_MATERIALS)
        self.materials.update(CONCRETE_MATERIALS)
        self.materials.update(REBAR_MATERIALS)
    
    def get_material(self, name: str) -> Optional[MaterialProperties]:
        """Get material by name."""
        return self.materials.get(name)
    
    def get_materials_by_type(self, material_type: MaterialType) -> List[MaterialProperties]:
        """Get all materials of a specific type."""
        return [m for m in self.materials.values() if m.type == material_type]
    
    def get_steel_materials(self) -> List[MaterialProperties]:
        """Get all steel materials."""
        return self.get_materials_by_type(MaterialType.STEEL)
    
    def get_concrete_materials(self) -> List[MaterialProperties]:
        """Get all concrete materials."""
        return self.get_materials_by_type(MaterialType.CONCRETE)
    
    def get_rebar_materials(self) -> List[MaterialProperties]:
        """Get all rebar materials."""
        return self.get_materials_by_type(MaterialType.REBAR)
    
    def search(self, query: str) -> List[MaterialProperties]:
        """Search materials by name (partial match)."""
        query_lower = query.lower()
        return [m for m in self.materials.values() if query_lower in m.name.lower()]
    
    def get_all_names(self) -> List[str]:
        """Get list of all material names."""
        return sorted(self.materials.keys())
    
    def add_custom_material(self, material: MaterialProperties):
        """Add a custom material to the library."""
        self.materials[material.name] = material
    
    def export_to_json(self, filepath: str):
        """Export library to JSON file."""
        data = {name: mat.to_dict() for name, mat in self.materials.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, filepath: str):
        """Import materials from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for name, mat_data in data.items():
            self.materials[name] = MaterialProperties.from_dict(mat_data)


# Global library instance
MATERIAL_LIB = MaterialLibrary()


# ============================================================================
# Helper Functions
# ============================================================================

def get_recommended_materials(structure_type: str) -> Dict[str, List[str]]:
    """
    Get recommended materials for a structure type.
    
    Args:
        structure_type: 'S' (Steel), 'RC' (Reinforced Concrete), 'SRC' (Composite)
        
    Returns:
        Dictionary with recommended materials for each element type
    """
    if structure_type == 'S':
        return {
            'column': ['SN490B', 'SN490C', 'BCP325', 'BCR295'],
            'beam': ['SN400B', 'SN490B', 'SS400'],
            'brace': ['SN400A', 'SS400'],
        }
    elif structure_type == 'RC':
        return {
            'concrete': ['Fc24', 'Fc27', 'Fc30', 'Fc36'],
            'main_rebar': ['SD345', 'SD390'],
            'shear_rebar': ['SD295A', 'SD345'],
        }
    elif structure_type == 'SRC':
        return {
            'steel': ['SN490B', 'SM490A'],
            'concrete': ['Fc27', 'Fc30', 'Fc36'],
            'rebar': ['SD345', 'SD390'],
        }
    else:
        return {}


def calculate_yield_moment(
    Z: float,  # Section modulus (m³)
    Fy: float  # Yield strength (Pa)
) -> float:
    """
    Calculate yield moment for steel section.
    
    My = Z * Fy
    
    Args:
        Z: Section modulus (m³)
        Fy: Yield strength (Pa)
        
    Returns:
        Yield moment (N·m)
    """
    return Z * Fy
