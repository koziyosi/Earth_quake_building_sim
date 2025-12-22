"""
Project Manager Module.
Implements unified project file format (#92) and project management.
"""
import os
import json
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ProjectMetadata:
    """Project metadata."""
    name: str
    version: str = "1.0"
    created: str = ""
    modified: str = ""
    author: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()


class ProjectManager:
    """
    Manages earthquake simulation projects.
    
    Project file format (.eqbsim):
    - project.json: Metadata and structure
    - layout.json: Building layout data
    - sections.json: Section library
    - materials.json: Material library
    - analysis.json: Analysis settings
    - results/: Analysis results (optional)
    """
    
    FILE_EXTENSION = ".eqbsim"
    
    def __init__(self):
        self.metadata = ProjectMetadata(name="Untitled Project")
        self.layout_data: Dict = {}
        self.sections_data: Dict = {}
        self.materials_data: Dict = {}
        self.analysis_settings: Dict = {}
        self.results: Dict = {}
        
        self.project_path: Optional[str] = None
        self.is_modified = False
        
    def new_project(self, name: str = "Untitled Project"):
        """Create a new empty project."""
        self.metadata = ProjectMetadata(name=name)
        self.layout_data = {}
        self.sections_data = {}
        self.materials_data = {}
        self.analysis_settings = self._default_analysis_settings()
        self.results = {}
        self.project_path = None
        self.is_modified = False
        
    def _default_analysis_settings(self) -> Dict:
        """Get default analysis settings."""
        return {
            'solver': {
                'method': 'newmark_beta',
                'beta': 0.25,
                'gamma': 0.5,
                'dt': 0.01
            },
            'damping': {
                'type': 'rayleigh',
                'ratio': 0.05
            },
            'nonlinear': {
                'max_iter': 20,
                'tol': 1e-6,
                'p_delta': False
            },
            'output': {
                'save_all_steps': False,
                'save_interval': 10
            }
        }
        
    def set_layout(self, layout):
        """Set building layout from BuildingLayout object."""
        if hasattr(layout, 'to_dict'):
            self.layout_data = layout.to_dict()
        else:
            self.layout_data = layout
        self.is_modified = True
        
    def set_sections(self, sections: Dict):
        """Set section library."""
        self.sections_data = sections
        self.is_modified = True
        
    def set_materials(self, materials: Dict):
        """Set material library."""
        self.materials_data = materials
        self.is_modified = True
        
    def set_analysis_settings(self, settings: Dict):
        """Set analysis settings."""
        self.analysis_settings.update(settings)
        self.is_modified = True
        
    def save(self, filepath: str = None) -> str:
        """
        Save project to file.
        
        Args:
            filepath: Path to save to (uses existing path if None)
            
        Returns:
            Path of saved file
        """
        if filepath is None:
            if self.project_path is None:
                raise ValueError("No file path specified for new project")
            filepath = self.project_path
            
        # Ensure extension
        if not filepath.endswith(self.FILE_EXTENSION):
            filepath += self.FILE_EXTENSION
            
        # Update metadata
        self.metadata.modified = datetime.now().isoformat()
        
        # Create zip file
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Project metadata
            zf.writestr('project.json', json.dumps(asdict(self.metadata), indent=2))
            
            # Layout
            if self.layout_data:
                zf.writestr('layout.json', json.dumps(self.layout_data, indent=2))
                
            # Sections
            if self.sections_data:
                zf.writestr('sections.json', json.dumps(self.sections_data, indent=2))
                
            # Materials
            if self.materials_data:
                zf.writestr('materials.json', json.dumps(self.materials_data, indent=2))
                
            # Analysis settings
            zf.writestr('analysis.json', json.dumps(self.analysis_settings, indent=2))
            
            # Results (if any)
            if self.results:
                zf.writestr('results.json', json.dumps(self._serialize_results(), indent=2))
                
        self.project_path = filepath
        self.is_modified = False
        
        return filepath
    
    def _serialize_results(self) -> Dict:
        """Serialize results for JSON (convert numpy arrays)."""
        import numpy as np
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
            
        return convert(self.results)
    
    def load(self, filepath: str):
        """
        Load project from file.
        
        Args:
            filepath: Path to project file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Project file not found: {filepath}")
            
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Read files
            namelist = zf.namelist()
            
            # Project metadata
            if 'project.json' in namelist:
                data = json.loads(zf.read('project.json'))
                self.metadata = ProjectMetadata(**data)
                
            # Layout
            if 'layout.json' in namelist:
                self.layout_data = json.loads(zf.read('layout.json'))
                
            # Sections
            if 'sections.json' in namelist:
                self.sections_data = json.loads(zf.read('sections.json'))
                
            # Materials
            if 'materials.json' in namelist:
                self.materials_data = json.loads(zf.read('materials.json'))
                
            # Analysis settings
            if 'analysis.json' in namelist:
                self.analysis_settings = json.loads(zf.read('analysis.json'))
                
            # Results
            if 'results.json' in namelist:
                self.results = json.loads(zf.read('results.json'))
                
        self.project_path = filepath
        self.is_modified = False
        
    def export_layout_json(self, filepath: str):
        """Export just the layout to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.layout_data, f, indent=2, ensure_ascii=False)
            
    def import_layout_json(self, filepath: str):
        """Import layout from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.layout_data = json.load(f)
        self.is_modified = True
        
    def get_recent_projects(self, config_dir: str = None) -> List[str]:
        """Get list of recently opened projects."""
        if config_dir is None:
            config_dir = os.path.expanduser("~/.earthquake_sim")
            
        recent_file = os.path.join(config_dir, "recent_projects.json")
        
        if os.path.exists(recent_file):
            with open(recent_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_to_recent(self, filepath: str, config_dir: str = None, max_recent: int = 10):
        """Add a project to the recent list."""
        if config_dir is None:
            config_dir = os.path.expanduser("~/.earthquake_sim")
            
        os.makedirs(config_dir, exist_ok=True)
        recent_file = os.path.join(config_dir, "recent_projects.json")
        
        recent = self.get_recent_projects(config_dir)
        
        # Remove if already exists
        if filepath in recent:
            recent.remove(filepath)
            
        # Add to front
        recent.insert(0, filepath)
        
        # Limit size
        recent = recent[:max_recent]
        
        with open(recent_file, 'w') as f:
            json.dump(recent, f)


# Template projects
TEMPLATE_PROJECTS = {
    'steel_3story': {
        'name': '3-Story Steel Frame',
        'description': 'Standard 3-story steel moment frame building',
        'settings': {
            'stories': 3,
            'bays_x': 3,
            'bays_y': 2,
            'story_height': 3.5,
            'bay_width': 6.0,
            'structure_type': 'steel'
        }
    },
    'rc_5story': {
        'name': '5-Story RC Frame',
        'description': 'Reinforced concrete frame building',
        'settings': {
            'stories': 5,
            'bays_x': 4,
            'bays_y': 3,
            'story_height': 3.2,
            'bay_width': 5.0,
            'structure_type': 'rc'
        }
    },
    'isolated_building': {
        'name': 'Base-Isolated Building',
        'description': 'Building with base isolation system',
        'settings': {
            'stories': 4,
            'bays_x': 3,
            'bays_y': 3,
            'story_height': 3.5,
            'bay_width': 6.0,
            'base_isolation': True
        }
    }
}


def create_from_template(template_name: str) -> ProjectManager:
    """
    Create a new project from a template.
    
    Args:
        template_name: Name of template to use
        
    Returns:
        Configured ProjectManager
    """
    if template_name not in TEMPLATE_PROJECTS:
        raise ValueError(f"Unknown template: {template_name}")
        
    template = TEMPLATE_PROJECTS[template_name]
    
    pm = ProjectManager()
    pm.new_project(template['name'])
    pm.metadata.description = template['description']
    
    # Would populate layout_data based on template settings
    # This requires integration with the builder module
    
    return pm
