"""
Tests for BuildingBuilder.
"""
import pytest
import numpy as np
from src.builder import BuildingBuilder
from src.layout_model import BuildingLayout, GridSystem


class TestBuildingBuilder:
    """Tests for BuildingBuilder class."""
    
    def test_build_basic_model(self):
        """Test building a basic 3-story model."""
        nodes, elements = BuildingBuilder.build_model(
            floors=3,
            span_x=6.0,
            span_y=6.0,
            story_h=3.5,
            soft_first_story=False,
            base_isolation=False,
            add_dampers=False
        )
        
        # Should have nodes
        assert len(nodes) > 0
        
        # Should have elements
        assert len(elements) > 0
        
        # Check some node properties
        base_nodes = [n for n in nodes if n.z == 0.0]
        assert len(base_nodes) == 4  # 4 corner nodes at base
        
        # Base nodes should be fixed
        for n in base_nodes:
            assert all(idx == -1 for idx in n.dof_indices)
            
    def test_build_with_soft_story(self):
        """Test building with soft first story."""
        nodes, elements = BuildingBuilder.build_model(
            floors=3,
            span_x=6.0,
            span_y=6.0,
            story_h=3.5,
            soft_first_story=True,
            base_isolation=False,
            add_dampers=False
        )
        
        # First story should be taller (1.5x)
        # Find first floor nodes (should be at 3.5 * 1.5 = 5.25m)
        first_floor_z = 3.5 * 1.5
        first_floor_nodes = [n for n in nodes if abs(n.z - first_floor_z) < 0.01]
        assert len(first_floor_nodes) == 4
        
    def test_build_with_isolation(self):
        """Test building with base isolation."""
        nodes, elements = BuildingBuilder.build_model(
            floors=3,
            span_x=6.0,
            span_y=6.0,
            story_h=3.5,
            soft_first_story=False,
            base_isolation=True,
            add_dampers=False
        )
        
        # Should have BaseIsolator elements
        from src.devices import BaseIsolator
        isolators = [e for e in elements if isinstance(e, BaseIsolator)]
        assert len(isolators) == 4  # One under each column
        
    def test_build_with_dampers(self):
        """Test building with oil dampers."""
        nodes, elements = BuildingBuilder.build_model(
            floors=3,
            span_x=6.0,
            span_y=6.0,
            story_h=3.5,
            soft_first_story=False,
            base_isolation=False,
            add_dampers=True
        )
        
        # Should have OilDamper elements
        from src.devices import OilDamper
        dampers = [e for e in elements if isinstance(e, OilDamper)]
        assert len(dampers) > 0


class TestBuildFromLayout:
    """Tests for building from custom layout."""
    
    def test_build_from_default_layout(self):
        """Test building from default layout."""
        layout = BuildingLayout()
        layout.initialize_default()
        
        nodes, elements = BuildingBuilder.build_from_layout(layout)
        
        assert len(nodes) > 0
        assert len(elements) > 0
        
    def test_grid_system(self):
        """Test GridSystem coordinates."""
        grid = GridSystem(
            x_spacings=[5.0, 5.0],
            y_spacings=[4.0, 4.0],
            story_heights=[3.0, 3.0, 3.0]
        )
        
        x_coords = grid.get_x_coords()
        y_coords = grid.get_y_coords()
        
        assert x_coords == [0.0, 5.0, 10.0]
        assert y_coords == [0.0, 4.0, 8.0]
        
    def test_layout_serialization(self):
        """Test layout save/load."""
        layout = BuildingLayout()
        layout.initialize_default()
        
        # Convert to dict
        data = layout.to_dict()
        
        # Convert back
        layout2 = BuildingLayout.from_dict(data)
        
        # Check grid is preserved
        assert layout2.grid.x_spacings == layout.grid.x_spacings
        assert layout2.grid.y_spacings == layout.grid.y_spacings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
