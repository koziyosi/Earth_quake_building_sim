"""
Configuration loader for EarthQuake Building Sim.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import os


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    duration: float = 5.0
    dt: float = 0.01
    max_acc: float = 400.0


@dataclass  
class BuildingConfig:
    """Building default parameters."""
    floors: int = 3
    span_x: float = 6.0
    span_y: float = 6.0
    story_height: float = 3.5


@dataclass
class DampingConfig:
    """Damping parameters."""
    omega1: float = 10.0
    omega2: float = 50.0
    zeta: float = 0.05


@dataclass
class OutputConfig:
    """Output parameters."""
    gif_fps: int = 20
    gif_dpi: int = 60
    animation_skip: int = 5


@dataclass
class LoggingConfig:
    """Logging parameters."""
    level: str = "INFO"
    file: Optional[str] = "simulation.log"
    console: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    damping: DampingConfig = field(default_factory=DampingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, searches default locations.
        
    Returns:
        AppConfig instance
    """
    config = AppConfig()
    
    # Search paths
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    
    # Default locations
    search_paths.extend([
        Path("config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
        Path.home() / ".earthquake_sim" / "config.yaml"
    ])
    
    # Try to load YAML
    try:
        import yaml
        
        for path in search_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data:
                        config = _parse_config(data)
                        print(f"Configuration loaded from: {path}")
                        break
    except ImportError:
        # PyYAML not installed, use defaults
        print("PyYAML not installed. Using default configuration.")
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
    
    return config


def _parse_config(data: Dict[str, Any]) -> AppConfig:
    """Parse YAML data into AppConfig."""
    config = AppConfig()
    
    if 'simulation' in data:
        sim = data['simulation']
        config.simulation = SimulationConfig(
            duration=sim.get('duration', 5.0),
            dt=sim.get('dt', 0.01),
            max_acc=sim.get('max_acc', 400.0)
        )
    
    if 'building' in data:
        bld = data['building']
        config.building = BuildingConfig(
            floors=bld.get('floors', 3),
            span_x=bld.get('span_x', 6.0),
            span_y=bld.get('span_y', 6.0),
            story_height=bld.get('story_height', 3.5)
        )
    
    if 'damping' in data:
        dmp = data['damping']
        config.damping = DampingConfig(
            omega1=dmp.get('omega1', 10.0),
            omega2=dmp.get('omega2', 50.0),
            zeta=dmp.get('zeta', 0.05)
        )
    
    if 'output' in data:
        out = data['output']
        config.output = OutputConfig(
            gif_fps=out.get('gif_fps', 20),
            gif_dpi=out.get('gif_dpi', 60),
            animation_skip=out.get('animation_skip', 5)
        )
    
    if 'logging' in data:
        log = data['logging']
        config.logging = LoggingConfig(
            level=log.get('level', 'INFO'),
            file=log.get('file'),
            console=log.get('console', True)
        )
    
    return config


# Global config instance (lazy loaded)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
