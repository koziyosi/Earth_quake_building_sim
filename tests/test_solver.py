"""
Tests for NewmarkBetaSolver.
"""
import pytest
import numpy as np
from src.fem import Node, BeamColumn2D
from src.hysteresis import Takeda
from src.solver import NewmarkBetaSolver


class TestNewmarkBetaSolver:
    """Tests for NewmarkBetaSolver class."""
    
    def test_solver_initialization(self):
        """Test solver initializes correctly."""
        # Create simple 2-node model
        n1 = Node(1, 0, 0)
        n2 = Node(2, 0, 3.5, mass=1000.0)
        
        n1.set_dof_indices([-1, -1, -1, -1, -1, -1])
        n2.set_dof_indices([0, 1, -1, -1, -1, 2])
        
        elements = []
        
        solver = NewmarkBetaSolver([n1, n2], elements, dt=0.01)
        
        assert solver.ndof == 3
        assert solver.dt == 0.01
        assert solver.M.shape == (3, 3)
        assert solver.K.shape == (3, 3)
        
    def test_mass_matrix_assembly(self):
        """Test mass matrix is correctly assembled."""
        n1 = Node(1, 0, 0)
        n2 = Node(2, 0, 3.5, mass=2000.0)
        
        n1.set_dof_indices([-1, -1, -1, -1, -1, -1])
        n2.set_dof_indices([0, 1, 2, 3, 4, 5])
        
        solver = NewmarkBetaSolver([n1, n2], [], dt=0.01)
        
        # Check translational mass
        assert solver.M[0, 0] == 2000.0  # X
        assert solver.M[1, 1] == 2000.0  # Y
        assert solver.M[2, 2] == 2000.0  # Z
        
        # Check rotational inertia (calculated based on building size)
        # Should be positive and proportional to mass
        assert solver.M[3, 3] > 0  # Rx
        assert solver.M[4, 4] > 0  # Ry
        assert solver.M[5, 5] > 0  # Rz
        # Rotational inertia should be same for all rotation DOFs
        assert solver.M[3, 3] == solver.M[4, 4] == solver.M[5, 5]
        
    def test_rayleigh_damping(self):
        """Test Rayleigh damping setup."""
        n1 = Node(1, 0, 0)
        n2 = Node(2, 0, 3.5, mass=1000.0)
        
        n1.set_dof_indices([-1, -1, -1, -1, -1, -1])
        n2.set_dof_indices([0, 1, 2, 3, 4, 5])
        
        solver = NewmarkBetaSolver([n1, n2], [], dt=0.01)
        solver.set_rayleigh_damping(omega1=10.0, omega2=50.0, zeta=0.05)
        
        # C should not be all zeros after setting Rayleigh damping
        assert np.any(solver.C != 0)
        
    def test_single_step_no_force(self):
        """Test single step with no external force."""
        n1 = Node(1, 0, 0)
        n2 = Node(2, 0, 3.5, mass=1000.0)
        
        n1.set_dof_indices([-1, -1, -1, -1, -1, -1])
        n2.set_dof_indices([0, 1, 2, 3, 4, 5])
        
        solver = NewmarkBetaSolver([n1, n2], [], dt=0.01)
        solver.set_rayleigh_damping(omega1=10.0, omega2=50.0, zeta=0.05)
        
        # Run one step with zero acceleration
        u, v, a = solver.solve_step(0.0)
        
        # All should be close to zero
        assert np.allclose(u, 0.0, atol=1e-10)


class TestSolverWithBeamColumn:
    """Integration tests with BeamColumn elements."""
    
    def test_simple_frame_response(self):
        """Test response of simple single column."""
        # Simple cantilever
        n1 = Node(1, 0, 0)
        n2 = Node(2, 0, 3.5, mass=5000.0)
        
        n1.set_dof_indices([-1, -1, -1, -1, -1, -1])
        n2.set_dof_indices([0, 1, -1, -1, -1, 2])
        
        # Create column with hysteresis
        from src.hysteresis import Bilinear
        spring_i = Bilinear(1e8, 100000, 0.05)
        spring_j = Bilinear(1e8, 100000, 0.05)
        
        col = BeamColumn2D(1, n1, n2, E=2.5e10, A=0.25, I=0.005, 
                          spring_i=spring_i, spring_j=spring_j)
        
        solver = NewmarkBetaSolver([n1, n2], [col], dt=0.01)
        solver.set_rayleigh_damping(10.0, 50.0, 0.05)
        
        # Apply small ground acceleration
        u1, v1, a1 = solver.solve_step(0.0)
        u2, v2, a2 = solver.solve_step(0.5)  # 0.5 m/s^2
        u3, v3, a3 = solver.solve_step(1.0)  # 1.0 m/s^2
        
        # Should have some displacement in X direction
        assert u3[0] != 0.0  # X displacement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
