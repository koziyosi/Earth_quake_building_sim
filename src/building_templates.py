"""
Building Templates Module.
Predefined building configurations for quick setup.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class StructureType(Enum):
    """Building structure types."""
    MOMENT_FRAME = "moment_frame"
    BRACED_FRAME = "braced_frame"
    SHEAR_WALL = "shear_wall"
    DUAL_SYSTEM = "dual_system"
    BASE_ISOLATED = "base_isolated"


class OccupancyType(Enum):
    """Building occupancy categories."""
    RESIDENTIAL = "residential"
    OFFICE = "office"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    WAREHOUSE = "warehouse"
    FACTORY = "factory"


@dataclass
class BuildingTemplate:
    """Template for quick building creation."""
    name: str
    description: str
    structure_type: StructureType
    occupancy: OccupancyType
    
    # Dimensions
    n_stories: int
    n_bays_x: int
    n_bays_y: int
    story_height: float
    bay_width_x: float
    bay_width_y: float
    
    # Structural properties
    column_section: str
    beam_section: str
    material: str
    
    # Optional features
    soft_story: bool = False
    base_isolation: bool = False
    dampers: bool = False
    
    # Additional parameters
    params: Dict = field(default_factory=dict)


# ===== Predefined Templates =====

TEMPLATES = {
    'low_rise_residential': BuildingTemplate(
        name="低層住宅",
        description="3階建て集合住宅",
        structure_type=StructureType.MOMENT_FRAME,
        occupancy=OccupancyType.RESIDENTIAL,
        n_stories=3,
        n_bays_x=3,
        n_bays_y=2,
        story_height=3.0,
        bay_width_x=6.0,
        bay_width_y=6.0,
        column_section="H-300x300x10x15",
        beam_section="H-400x200x8x13",
        material="SN400B",
    ),
    
    'mid_rise_office': BuildingTemplate(
        name="中層オフィス",
        description="7階建てオフィスビル",
        structure_type=StructureType.MOMENT_FRAME,
        occupancy=OccupancyType.OFFICE,
        n_stories=7,
        n_bays_x=4,
        n_bays_y=3,
        story_height=3.5,
        bay_width_x=8.0,
        bay_width_y=8.0,
        column_section="H-400x400x13x21",
        beam_section="H-600x200x11x17",
        material="SN490B",
    ),
    
    'high_rise_office': BuildingTemplate(
        name="高層オフィス",
        description="20階建て高層ビル",
        structure_type=StructureType.DUAL_SYSTEM,
        occupancy=OccupancyType.OFFICE,
        n_stories=20,
        n_bays_x=5,
        n_bays_y=4,
        story_height=4.0,
        bay_width_x=10.0,
        bay_width_y=10.0,
        column_section="□-500x500x22",
        beam_section="H-700x300x13x24",
        material="SN490B",
        dampers=True,
    ),
    
    'hospital': BuildingTemplate(
        name="病院",
        description="5階建て病院（免震構造）",
        structure_type=StructureType.BASE_ISOLATED,
        occupancy=OccupancyType.HOSPITAL,
        n_stories=5,
        n_bays_x=6,
        n_bays_y=4,
        story_height=4.0,
        bay_width_x=8.0,
        bay_width_y=8.0,
        column_section="H-500x500x16x28",
        beam_section="H-600x300x12x20",
        material="SN490B",
        base_isolation=True,
    ),
    
    'school': BuildingTemplate(
        name="学校",
        description="4階建て校舎",
        structure_type=StructureType.BRACED_FRAME,
        occupancy=OccupancyType.SCHOOL,
        n_stories=4,
        n_bays_x=8,
        n_bays_y=2,
        story_height=3.5,
        bay_width_x=7.0,
        bay_width_y=10.0,
        column_section="H-350x350x12x19",
        beam_section="H-500x200x10x16",
        material="SS400",
    ),
    
    'warehouse': BuildingTemplate(
        name="倉庫",
        description="2階建て倉庫",
        structure_type=StructureType.BRACED_FRAME,
        occupancy=OccupancyType.WAREHOUSE,
        n_stories=2,
        n_bays_x=6,
        n_bays_y=3,
        story_height=6.0,
        bay_width_x=12.0,
        bay_width_y=10.0,
        column_section="H-400x400x13x21",
        beam_section="H-600x200x11x17",
        material="SS400",
    ),
    
    'piloti': BuildingTemplate(
        name="ピロティ建物",
        description="1階が柔らかい層（soft story）",
        structure_type=StructureType.MOMENT_FRAME,
        occupancy=OccupancyType.RESIDENTIAL,
        n_stories=5,
        n_bays_x=3,
        n_bays_y=2,
        story_height=3.5,
        bay_width_x=7.0,
        bay_width_y=7.0,
        column_section="H-350x350x12x19",
        beam_section="H-450x200x9x14",
        material="SN400B",
        soft_story=True,
        params={'soft_story_floor': 1, 'stiffness_ratio': 0.5}
    ),
    
    'tower': BuildingTemplate(
        name="超高層タワー",
        description="40階建てタワーマンション",
        structure_type=StructureType.DUAL_SYSTEM,
        occupancy=OccupancyType.RESIDENTIAL,
        n_stories=40,
        n_bays_x=3,
        n_bays_y=3,
        story_height=3.2,
        bay_width_x=12.0,
        bay_width_y=12.0,
        column_section="CFT-800x800x25",
        beam_section="H-800x300x14x26",
        material="SN490C",
        dampers=True,
        base_isolation=True,
    ),
}


def get_template(name: str) -> Optional[BuildingTemplate]:
    """Get a template by name."""
    return TEMPLATES.get(name)


def list_templates() -> List[str]:
    """List all available template names."""
    return list(TEMPLATES.keys())


def get_templates_by_type(structure_type: StructureType) -> List[BuildingTemplate]:
    """Get templates filtered by structure type."""
    return [t for t in TEMPLATES.values() if t.structure_type == structure_type]


def get_templates_by_occupancy(occupancy: OccupancyType) -> List[BuildingTemplate]:
    """Get templates filtered by occupancy."""
    return [t for t in TEMPLATES.values() if t.occupancy == occupancy]


def create_custom_template(
    name: str,
    n_stories: int,
    n_bays_x: int,
    n_bays_y: int,
    story_height: float = 3.5,
    bay_width: float = 8.0,
    **kwargs
) -> BuildingTemplate:
    """
    Create a custom template with default values.
    
    Args:
        name: Template name
        n_stories: Number of stories
        n_bays_x: Number of bays in X
        n_bays_y: Number of bays in Y
        story_height: Story height (m)
        bay_width: Bay width (m)
        **kwargs: Additional parameters
        
    Returns:
        BuildingTemplate object
    """
    # Auto-select sections based on height
    if n_stories <= 3:
        col = "H-300x300x10x15"
        beam = "H-400x200x8x13"
    elif n_stories <= 7:
        col = "H-400x400x13x21"
        beam = "H-500x200x10x16"
    elif n_stories <= 15:
        col = "□-500x500x22"
        beam = "H-600x300x12x20"
    else:
        col = "CFT-700x700x25"
        beam = "H-700x300x13x24"
        
    return BuildingTemplate(
        name=name,
        description=kwargs.get('description', f'{n_stories}階建て建物'),
        structure_type=kwargs.get('structure_type', StructureType.MOMENT_FRAME),
        occupancy=kwargs.get('occupancy', OccupancyType.OFFICE),
        n_stories=n_stories,
        n_bays_x=n_bays_x,
        n_bays_y=n_bays_y,
        story_height=story_height,
        bay_width_x=bay_width,
        bay_width_y=kwargs.get('bay_width_y', bay_width),
        column_section=kwargs.get('column_section', col),
        beam_section=kwargs.get('beam_section', beam),
        material=kwargs.get('material', 'SN400B'),
        soft_story=kwargs.get('soft_story', False),
        base_isolation=kwargs.get('base_isolation', False),
        dampers=kwargs.get('dampers', False),
    )


def estimate_period(template: BuildingTemplate) -> float:
    """
    Estimate natural period from template.
    
    Uses empirical formula: T = 0.02 * H (steel frame)
    """
    H = template.n_stories * template.story_height
    
    if template.structure_type == StructureType.SHEAR_WALL:
        return 0.015 * H
    elif template.structure_type == StructureType.BRACED_FRAME:
        return 0.018 * H
    else:  # Moment frame
        return 0.025 * H


def estimate_weight(template: BuildingTemplate) -> float:
    """
    Estimate total building weight from template.
    
    Based on typical floor loads.
    """
    floor_area = (template.n_bays_x * template.bay_width_x) * \
                 (template.n_bays_y * template.bay_width_y)
    
    # Load per floor (kN/m²)
    if template.occupancy == OccupancyType.WAREHOUSE:
        load = 15.0
    elif template.occupancy == OccupancyType.HOSPITAL:
        load = 10.0
    else:
        load = 8.0
        
    total_weight = floor_area * load * template.n_stories * 1000  # N
    
    return total_weight
