"""
Tests for Response Analyzer Module.
"""
import pytest
import numpy as np
from src.response_analyzer import (
    calculate_inter_story_drift,
    calculate_ductility,
    calculate_base_shear_coefficient,
    calculate_center_of_mass,
    StoryResponse,
    GlobalResponse
)


class MockNode:
    """Mock node for testing."""
    def __init__(self, x, y, z, mass=1000.0, dof_start=0):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass
        self.dof_indices = [dof_start + i for i in range(6)]


class TestDuctility:
    def test_ductility_basic(self):
        """Test basic ductility calculation."""
        result = calculate_ductility(max_disp=0.02, yield_disp=0.01)
        assert result == 2.0
        
    def test_ductility_zero_yield(self):
        """Test ductility with zero yield displacement."""
        result = calculate_ductility(max_disp=0.02, yield_disp=0.0)
        assert result == 0.0
        
    def test_ductility_negative_disp(self):
        """Test ductility with negative displacement uses absolute value."""
        result = calculate_ductility(max_disp=-0.02, yield_disp=0.01)
        assert result == 2.0


class TestBaseShearCoefficient:
    def test_base_shear_basic(self):
        """Test basic base shear coefficient calculation."""
        base_shear = np.array([100, 200, 150, 180])
        total_weight = 1000.0
        
        Cb, max_Cb = calculate_base_shear_coefficient(base_shear, total_weight)
        
        assert len(Cb) == 4
        assert max_Cb == 0.2
        assert Cb[1] == 0.2
        
    def test_base_shear_zero_weight(self):
        """Test with zero weight returns zeros."""
        base_shear = np.array([100, 200])
        Cb, max_Cb = calculate_base_shear_coefficient(base_shear, 0.0)
        
        assert max_Cb == 0.0
        np.testing.assert_array_equal(Cb, np.zeros(2))


class TestCenterOfMass:
    def test_center_of_mass_symmetric(self):
        """Test center of mass for symmetric layout."""
        nodes = [
            MockNode(0, 0, 0, mass=100),
            MockNode(10, 0, 0, mass=100),
            MockNode(0, 10, 0, mass=100),
            MockNode(10, 10, 0, mass=100),
        ]
        
        x_cm, y_cm = calculate_center_of_mass(nodes)
        
        assert x_cm == 5.0
        assert y_cm == 5.0
        
    def test_center_of_mass_weighted(self):
        """Test center of mass with different weights."""
        nodes = [
            MockNode(0, 0, 0, mass=100),
            MockNode(10, 0, 0, mass=300),  # Heavier on right
        ]
        
        x_cm, y_cm = calculate_center_of_mass(nodes)
        
        assert x_cm == 7.5  # Weighted towards heavier node
        assert y_cm == 0.0


class TestStoryResponse:
    def test_story_response_creation(self):
        """Test StoryResponse dataclass."""
        sr = StoryResponse(
            story=1,
            max_drift=0.01,
            max_disp=0.05,
            max_accel=9.8,
            max_shear=50000,
            energy=10000,
            ductility=2.5
        )
        
        assert sr.story == 1
        assert sr.max_drift == 0.01
        assert sr.ductility == 2.5


class TestGlobalResponse:
    def test_global_response_creation(self):
        """Test GlobalResponse dataclass."""
        gr = GlobalResponse(
            max_base_shear=100000,
            base_shear_coef=0.15,
            max_top_disp=0.1,
            max_top_accel=15.0,
            total_energy=50000,
            period_T1=0.8,
            center_of_rigidity=(5.0, 5.0),
            center_of_mass=(4.8, 5.2)
        )
        
        assert gr.max_base_shear == 100000
        assert gr.period_T1 == 0.8
        assert gr.center_of_mass == (4.8, 5.2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
