"""
Tests for Extended Elements Module.
"""
import pytest
import numpy as np
from src.elements_extended import (
    BraceElement,
    AsymmetricBilinear,
    TunedMassDamper,
    FrictionDamper,
    HighDampingRubberBearing
)


class MockNode:
    """Mock node for testing."""
    def __init__(self, x, y, z, dof_start=0):
        self.x = x
        self.y = y
        self.z = z
        self.mass = 1000.0
        self.dof_indices = [dof_start + i for i in range(6)]


class TestAsymmetricBilinear:
    def test_elastic_range(self):
        """Test behavior in elastic range."""
        hyst = AsymmetricBilinear(k0=1000, fy_pos=100, fy_neg=80, r=0.1)
        
        hyst.set_trial_disp(0.05)  # Within elastic range
        
        assert abs(hyst.force - 50) < 1e-6  # k0 * disp = 1000 * 0.05
        assert hyst.tangent == 1000
        
    def test_positive_yield(self):
        """Test yielding in positive direction."""
        hyst = AsymmetricBilinear(k0=1000, fy_pos=100, fy_neg=80, r=0.1)
        
        hyst.set_trial_disp(0.2)  # Beyond yield
        
        assert hyst.force > 100
        assert hyst.tangent == 100  # k0 * r
        
    def test_negative_yield(self):
        """Test yielding in negative direction (buckling)."""
        hyst = AsymmetricBilinear(k0=1000, fy_pos=100, fy_neg=80, r=0.1)
        
        hyst.set_trial_disp(-0.2)  # Beyond yield in compression
        
        assert hyst.force < -80
        assert hyst.tangent == 100


class TestBraceElement:
    def test_brace_creation(self):
        """Test brace element creation."""
        node_i = MockNode(0, 0, 0, dof_start=0)
        node_j = MockNode(3, 4, 0, dof_start=6)
        
        brace = BraceElement(
            id=1,
            node_i=node_i,
            node_j=node_j,
            E=2.05e11,
            A=0.005,
            Fy_tension=500000
        )
        
        assert brace.get_length() == 5.0  # 3-4-5 triangle
        assert brace.Fy_compression == 350000  # 70% of tension
        
    def test_brace_stiffness_matrix_shape(self):
        """Test stiffness matrix has correct shape."""
        node_i = MockNode(0, 0, 0, dof_start=0)
        node_j = MockNode(5, 0, 0, dof_start=6)
        
        brace = BraceElement(
            id=1, node_i=node_i, node_j=node_j,
            E=2e11, A=0.01, Fy_tension=1e6
        )
        
        K = brace.get_stiffness_matrix()
        
        assert K.shape == (12, 12)
        assert np.allclose(K, K.T)  # Symmetric


class TestTunedMassDamper:
    def test_tmd_creation(self):
        """Test TMD creation."""
        node_struct = MockNode(0, 0, 10, dof_start=0)
        node_tmd = MockNode(0, 0, 10.5, dof_start=6)
        
        tmd = TunedMassDamper(
            id=1,
            node_structure=node_struct,
            node_tmd=node_tmd,
            mass=5000,
            stiffness=100000,
            damping=5000
        )
        
        assert tmd.mass == 5000
        assert tmd.k == 100000
        assert tmd.c == 5000
        assert tmd.frequency > 0
        
    def test_tmd_design_for_structure(self):
        """Test optimal TMD design."""
        node_struct = MockNode(0, 0, 10, dof_start=0)
        node_tmd = MockNode(0, 0, 10.5, dof_start=6)
        
        tmd = TunedMassDamper.design_for_structure(
            id=1,
            node_structure=node_struct,
            node_tmd=node_tmd,
            structure_mass=1000000,
            structure_period=1.0,
            mass_ratio=0.02
        )
        
        assert tmd.mass == 20000  # 2% of structure mass
        assert tmd.k > 0
        assert tmd.c > 0
        
    def test_tmd_matrices_shape(self):
        """Test TMD matrices have correct shape."""
        node_struct = MockNode(0, 0, 10, dof_start=0)
        node_tmd = MockNode(0, 0, 10.5, dof_start=6)
        
        tmd = TunedMassDamper(
            id=1, node_structure=node_struct, node_tmd=node_tmd,
            mass=5000, stiffness=100000, damping=5000
        )
        
        K = tmd.get_stiffness_matrix()
        C = tmd.get_damping_matrix()
        M = tmd.get_mass_matrix()
        
        assert K.shape == (12, 12)
        assert C.shape == (12, 12)
        assert M.shape == (12, 12)


class TestFrictionDamper:
    def test_friction_damper_sticking(self):
        """Test friction damper in sticking mode."""
        node_i = MockNode(0, 0, 0, dof_start=0)
        node_j = MockNode(5, 0, 0, dof_start=6)
        
        fd = FrictionDamper(
            id=1,
            node_i=node_i,
            node_j=node_j,
            friction_force=10000
        )
        
        # Small displacement - should stick
        delta_u = np.zeros(12)
        delta_u[6] = 0.00001  # Very small
        
        fd.update_state(delta_u)
        
        assert not fd.is_slipping
        
    def test_friction_damper_slipping(self):
        """Test friction damper in slipping mode."""
        node_i = MockNode(0, 0, 0, dof_start=0)
        node_j = MockNode(5, 0, 0, dof_start=6)
        
        fd = FrictionDamper(
            id=1,
            node_i=node_i,
            node_j=node_j,
            friction_force=100,
            initial_stiffness=1e6
        )
        
        # Large displacement - should slip
        delta_u = np.zeros(12)
        delta_u[6] = 0.01  # Large enough to exceed friction
        
        fd.update_state(delta_u)
        
        # Force should be capped at friction force
        assert abs(fd.trial_force) <= fd.Ff + 1  # Small tolerance


class TestHighDampingRubberBearing:
    def test_hdrb_creation(self):
        """Test HDRB creation."""
        node_i = MockNode(0, 0, 0, dof_start=0)
        node_j = MockNode(0, 0, 0.5, dof_start=6)
        
        hdrb = HighDampingRubberBearing(
            id=1,
            node_i=node_i,
            node_j=node_j,
            Kv=1e9,
            Kh=1e6,
            Qd=50000,
            damping_ratio=0.20
        )
        
        assert hdrb.Kv == 1e9
        assert hdrb.Kh == 1e6
        assert hdrb.damping_ratio == 0.20
        
    def test_hdrb_stiffness_matrix(self):
        """Test HDRB stiffness matrix."""
        node_i = MockNode(0, 0, 0, dof_start=0)
        node_j = MockNode(0, 0, 0.5, dof_start=6)
        
        hdrb = HighDampingRubberBearing(
            id=1, node_i=node_i, node_j=node_j,
            Kv=1e9, Kh=1e6, Qd=50000
        )
        
        K = hdrb.get_stiffness_matrix()
        
        assert K.shape == (12, 12)
        # Vertical stiffness should be much higher
        assert K[2, 2] > K[0, 0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
