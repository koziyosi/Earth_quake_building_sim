from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class SectionProperties:
    """
    Physical properties for a structural section.
    """
    name: str
    E: float = 2.5e10
    area: float = 0.25
    I_y: float = 0.005 # Weak axis
    I_z: float = 0.005 # Strong axis 
    J: float = 0.002   # Torsion
    yield_moment: float = 300000.0
    color: str = 'green' # For GUI visualization

@dataclass
class GridSystem:
    """
    Defines the spatial grid.
    """
    x_spacings: List[float] = field(default_factory=lambda: [6.0, 6.0, 6.0]) # Spans in X
    y_spacings: List[float] = field(default_factory=lambda: [6.0, 6.0, 6.0]) # Spans in Y
    story_heights: List[float] = field(default_factory=lambda: [3.5, 3.5, 3.5]) # Height of each story
    
    def get_x_coords(self):
        coords = [0.0]
        for s in self.x_spacings:
            coords.append(coords[-1] + s)
        return coords
        
    def get_y_coords(self):
        coords = [0.0]
        for s in self.y_spacings:
            coords.append(coords[-1] + s)
        return coords
        
    def to_dict(self):
        return {
            'x_spacings': self.x_spacings,
            'y_spacings': self.y_spacings,
            'story_heights': self.story_heights
        }
    
    @classmethod
    def from_dict(cls, data):
        grid = cls()
        grid.x_spacings = data.get('x_spacings', [5.0]*3)
        grid.y_spacings = data.get('y_spacings', [5.0]*3)
        grid.story_heights = data.get('story_heights', [3.5]*3)
        return grid

@dataclass
class FloorLayout:
    """
    Layout for a specific floor.
    Contains definitions for Columns (vertical elements below this floor? or at this floor?)
    Convention: Floor i layout defines Columns between Floor i-1 and Floor i?
    Or Plan View at Floor i defines columns at that grid intersection?
    Usually: 
    1F Plan -> Columns standing on Base, supporting 1F Slab? No, 1F is usually ground.
    Let's use: 
    "1F Plan" = Structure at 1st Floor Level. 
    Columns defined at "1F" are usually the columns *below* the 1F slab (connecting Base to 1F) or *above*?
    Standard terminology: "1st Story Columns" connect Ground to 2nd Floor.
    The GUI shows "1F". 
    Let's assume:
    Floor 0 = Base.
    Floor 1 = 1st elevated level.
    "1F Layout" defines the columns BENEATH Floor 1 (connecting 0->1) and beams AT Floor 1.
    """
    floor_index: int
    columns: Dict[Tuple[int, int], str] = field(default_factory=dict) # (gx, gy) -> SectionName
    beams: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = field(default_factory=dict) # ((x1,y1), (x2,y2)) -> SectionName

    def add_column(self, gx, gy, section="C1"):
        self.columns[(gx, gy)] = section
        
    def remove_column(self, gx, gy):
        if (gx, gy) in self.columns:
            del self.columns[(gx, gy)]
            
    def add_beam(self, p1, p2, section="B1"):
        # p1, p2 are (gx, gy) tuples
        # Ensure consistent ordering for keys
        if p1 > p2: p1, p2 = p2, p1
        self.beams[(p1, p2)] = section
        
    def remove_beam(self, p1, p2):
        if p1 > p2: p1, p2 = p2, p1
        if (p1, p2) in self.beams:
            del self.beams[(p1, p2)]

    def to_dict(self):
        # Convert tuple keys to string representation or list for JSON
        # JSON only supports string keys.
        # But we can store as list of objects
        cols_data = []
        for (gx, gy), sec in self.columns.items():
            cols_data.append({'gx': gx, 'gy': gy, 'sec': sec})
            
        beams_data = []
        for ((gx1, gy1), (gx2, gy2)), sec in self.beams.items():
            beams_data.append({'p1': [gx1, gy1], 'p2': [gx2, gy2], 'sec': sec})
            
        return {
            'floor_index': self.floor_index,
            'columns': cols_data,
            'beams': beams_data
        }

    @classmethod
    def from_dict(cls, data):
        fl = cls(floor_index=data['floor_index'])
        if 'columns' in data:
            for item in data['columns']:
                fl.add_column(item['gx'], item['gy'], item['sec'])
        if 'beams' in data:
            for item in data['beams']:
                p1 = tuple(item['p1'])
                p2 = tuple(item['p2'])
                fl.add_beam(p1, p2, item['sec'])
        return fl

@dataclass
class BuildingLayout:
    """
    Entire Building Configuration.
    """
    grid: GridSystem = field(default_factory=GridSystem)
    floors: Dict[int, FloorLayout] = field(default_factory=dict)
    sections: Dict[str, SectionProperties] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)  # Building options (soft_story, base_isolation, dampers)
    
    def __post_init__(self):
        # Default sections
        self.sections["C1"] = SectionProperties("C1", I_y=0.008, I_z=0.008, color='green')
        self.sections["C2"] = SectionProperties("C2", I_y=0.012, I_z=0.012, color='darkgreen')
        self.sections["B1"] = SectionProperties("B1", I_y=0.004, I_z=0.008, color='blue') # Strong axis Z usually for beams
        self.sections["B2"] = SectionProperties("B2", I_y=0.006, I_z=0.012, color='navy')
        self.sections["W1"] = SectionProperties("W1", area=0.1, color='red') # Wall as lighter brace? Or heavy?
        
    def get_floor(self, index):
        if index not in self.floors:
            self.floors[index] = FloorLayout(index)
        return self.floors[index]
    
    def initialize_default(self):
        # Create a default regular grid based on grid definition
        n_x = len(self.grid.x_spacings) + 1
        n_y = len(self.grid.y_spacings) + 1
        n_stories = len(self.grid.story_heights)
        
        for f in range(1, n_stories + 1):
            fl = self.get_floor(f)
            # Add columns at all intersections
            for i in range(n_x):
                for j in range(n_y):
                    fl.add_column(i, j, "C1")
            
            # Add beams
            # X-beams
            for j in range(n_y):
                for i in range(n_x - 1):
                    fl.add_beam((i, j), (i+1, j), "B1")
            
            # Y-beams
            for i in range(n_x):
                for j in range(n_y - 1):
                    fl.add_beam((i, j), (i, j+1), "B1")

    def cleanup_elements(self):
        """
        Removes elements that are outside current grid bounds.
        """
        n_x = len(self.grid.x_spacings) + 1
        n_y = len(self.grid.y_spacings) + 1
        
        for f in self.floors.values():
            # Cleanup Columns
            cols_to_remove = []
            for (gx, gy) in f.columns:
                if gx >= n_x or gy >= n_y:
                    cols_to_remove.append((gx, gy))
            for k in cols_to_remove:
                del f.columns[k]
                
            # Cleanup Beams
            beams_to_remove = []
            for ((gx1, gy1), (gx2, gy2)) in f.beams:
                if gx1 >= n_x or gx2 >= n_x or gy1 >= n_y or gy2 >= n_y:
                    beams_to_remove.append(((gx1, gy1), (gx2, gy2)))
            for k in beams_to_remove:
                del f.beams[k]

    def to_dict(self):
        floors_data = [f.to_dict() for f in self.floors.values()]
        return {
            'grid': self.grid.to_dict(),
            'floors': floors_data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        layout = cls()
        if 'grid' in data:
            layout.grid = GridSystem.from_dict(data['grid'])
        
        if 'floors' in data:
            layout.floors = {}
            for fdata in data['floors']:
                fl = FloorLayout.from_dict(fdata)
                layout.floors[fl.floor_index] = fl
        else:
            layout.initialize_default()
        
        # Load metadata (building options)
        if 'metadata' in data:
            layout.metadata = data['metadata']
            
        return layout

    def save_to_file(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @staticmethod
    def load_from_file(filename):
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        return BuildingLayout.from_dict(data)
