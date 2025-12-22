"""
Tests for Pushover Analysis Module.
"""
import pytest
import numpy as np
from src.pushover import (
    PushoverAnalyzer,
    PushoverResult,
    bilinearize_capacity_curve
)


class MockNode:
    """Mock node for testing."""
    def __init__(self, x, y, z, mass=1000.0, dof_start=0):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass
        self.dof_indices = [dof_start + i for i in range(6)]


class MockElement:
    """Mock element for testing."""
    def __init__(self, id, node_i, node_j, k=1e6):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.k = k
        self.E = 2e11
        self.Iy = 1e-4
        self._damage_index = 0.0
        
    def get_length(self):
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
        
    def get_stiffness_matrix(self):
        K = np.zeros((12, 12))
        K[0, 0] = self.k
        K[0, 6] = -self.k
        K[6, 0] = -self.k
        K[6, 6] = self.k
        return K
        
    def get_element_dof_indices(self):
        return self.node_i.dof_indices + self.node_j.dof_indices
        
    def update_state(self, delta_u):
        indices = self.get_element_dof_indices()
        du = delta_u[indices[6]] - delta_u[indices[0]] if indices[0] >= 0 else 0
        f = np.zeros(12)
        f[0] = -self.k * du
        f[6] = self.k * du
        return f
        
    def commit_state(self):
        pass
        
    @property
    def damage_index(self):
        return self._damage_index


class TestBilinearizeCapacityCurve:
    def test_bilinearize_basic(self):
        """Test basic bilinearization."""
        disp = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05])
        shear = np.array([0, 100, 180, 220, 240, 250])
        
        K0, Fy, dy, du = bilinearize_capacity_curve(disp, shear)
        
        assert K0 > 0
        assert Fy > 0
        assert dy > 0
        assert du == 0.05
        
    def test_bilinearize_short_array(self):
        """Test with too short array."""
        disp = np.array([0])
        shear = np.array([0])
        
        K0, Fy, dy, du = bilinearize_capacity_curve(disp, shear)
        
        assert K0 == 0
        assert Fy == 0
        
    def test_bilinearize_equal_area_method(self):
        """Test equal area method."""
        disp = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05])
        shear = np.array([0, 100, 180, 220, 240, 250])
        
        K0, Fy, dy, du = bilinearize_capacity_curve(disp, shear, method='equal_area')
        
        assert K0 > 0
        assert Fy > 0


class TestPushoverResult:
    def test_result_creation(self):
        """Test PushoverResult dataclass."""
        result = PushoverResult(
            base_shear=np.array([0, 100, 200]),
            roof_disp=np.array([0, 0.01, 0.02]),
            story_drifts={1: np.array([0, 0.005, 0.01])},
            yielded_elements=[1, 3, 5],
            capacity_curve=(np.array([0, 0.01, 0.02]), np.array([0, 100, 200]))
        )
        
        assert len(result.base_shear) == 3
        assert len(result.yielded_elements) == 3
        assert 1 in result.story_drifts


class TestPushoverAnalyzer:
    def setup_method(self):
        """Setup simple 2-node system for testing."""
        self.node1 = MockNode(0, 0, 0, mass=0, dof_start=0)  # Fixed base
        self.node2 = MockNode(0, 0, 3.5, mass=10000, dof_start=6)
        
        self.nodes = [self.node1, self.node2]
        self.elements = [MockElement(1, self.node1, self.node2)]
        self.fixed_dofs = list(range(6))  # Fix base node
        
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = PushoverAnalyzer(
            nodes=self.nodes,
            elements=self.elements,
            fixed_dofs=self.fixed_dofs
        )
        
        assert analyzer.ndof == 12
        assert len(analyzer.fixed_dofs) == 6
        
    def test_uniform_load_pattern(self):
        """Test uniform load pattern generation."""
        analyzer = PushoverAnalyzer(
            nodes=self.nodes,
            elements=self.elements,
            fixed_dofs=self.fixed_dofs
        )
        
        analyzer.set_uniform_load_pattern('x')
        
        assert analyzer.load_pattern is not None
        assert analyzer.load_pattern[6] > 0  # Load at node 2 X-direction
        
    def test_triangular_load_pattern(self):
        """Test triangular load pattern generation."""
        # Add more nodes at different heights
        node3 = MockNode(0, 0, 7.0, mass=10000, dof_start=12)
        nodes = [self.node1, self.node2, node3]
        elements = [
            MockElement(1, self.node1, self.node2),
            MockElement(2, self.node2, node3)
        ]
        
        analyzer = PushoverAnalyzer(
            nodes=nodes,
            elements=elements,
            fixed_dofs=self.fixed_dofs
        )
        
        analyzer.set_triangular_load_pattern('x')
        
        # Higher nodes should have larger load
        assert analyzer.load_pattern[12] > analyzer.load_pattern[6]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
