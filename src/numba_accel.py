"""
Numba-optimized functions for solver acceleration.
Uses JIT compilation for parallel processing of hot loops.
"""
import numpy as np
from numba import jit, prange
from numba.typed import List as NumbaList

@jit(nopython=True, cache=True)
def assemble_stiffness_fast(K_flat, k_elements, indices_list, ndof):
    """
    Fast stiffness matrix assembly using Numba.
    
    Note: Sequential accumulation to avoid race conditions.
    Parallel element computation would cause data races when multiple
    elements contribute to the same global DOF.
    
    Args:
        K_flat: Flattened global stiffness matrix (ndof * ndof)
        k_elements: List of flattened element stiffness matrices
        indices_list: List of DOF index arrays for each element
        ndof: Number of DOFs
    """
    n_elements = len(k_elements)
    
    # Sequential loop to avoid race conditions
    for e in range(n_elements):
        k_el = k_elements[e]
        indices = indices_list[e]
        n_dof_el = len(indices)
        
        for r in range(n_dof_el):
            if indices[r] < 0:
                continue
            for c in range(n_dof_el):
                if indices[c] < 0:
                    continue
                K_flat[indices[r] * ndof + indices[c]] += k_el[r * n_dof_el + c]


@jit(nopython=True, cache=True)
def compute_internal_forces_fast(f_int, f_elements, indices_list, ndof):
    """
    Fast internal force assembly using Numba.
    
    Note: Sequential to avoid race conditions when multiple elements
    contribute to the same DOF.
    
    Args:
        f_int: Global internal force vector (ndof)
        f_elements: List of element force vectors
        indices_list: List of DOF index arrays for each element
        ndof: Number of DOFs
    """
    n_elements = len(f_elements)
    
    # Sequential loop to avoid race conditions
    for e in range(n_elements):
        f_el = f_elements[e]
        indices = indices_list[e]
        n_dof_el = len(indices)
        
        for i in range(n_dof_el):
            if indices[i] >= 0:
                f_int[indices[i]] += f_el[i]


@jit(nopython=True, cache=True)
def update_positions_fast(node_positions, u, dof_indices, scale, n_nodes):
    """
    Fast node position update for visualization.
    
    Args:
        node_positions: Output array (n_nodes, 3) for updated positions
        u: Displacement vector
        dof_indices: DOF indices array (n_nodes, 6)
        scale: Displacement scale factor
        n_nodes: Number of nodes
    """
    for i in range(n_nodes):
        for j in range(3):  # X, Y, Z
            idx = dof_indices[i, j]
            if idx >= 0:
                node_positions[i, j] += u[idx] * scale


@jit(nopython=True, parallel=True, cache=True)
def matrix_vector_mult_fast(result, M, v):
    """Fast parallel matrix-vector multiplication."""
    n = len(v)
    for i in prange(n):
        s = 0.0
        for j in range(n):
            s += M[i, j] * v[j]
        result[i] = s


# Wrapper class for easy integration
class NumbaAccelerator:
    """Helper class to use Numba-optimized functions."""
    
    _initialized = False
    
    @classmethod
    def is_available(cls):
        """Check if Numba is available."""
        try:
            from numba import jit
            return True
        except ImportError:
            return False
    
    @classmethod
    def warmup(cls):
        """Warm up JIT compilation with small test data."""
        if cls._initialized:
            return
        
        # Small test to trigger JIT compilation
        test_K = np.zeros(9, dtype=np.float64)
        test_k = [np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)]
        test_idx = [np.array([0, 1, 2], dtype=np.int64)]
        
        try:
            assemble_stiffness_fast(test_K, test_k, test_idx, 3)
            cls._initialized = True
            print("Numba JIT warmup complete - acceleration enabled!")
        except Exception as e:
            print(f"Numba warmup failed: {e}")
