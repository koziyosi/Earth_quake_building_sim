"""
Parallel Processing utilities for earthquake simulation.
Provides multi-threading support for large-scale analyses.
"""
import numpy as np
from typing import List, Callable, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time


@dataclass
class BatchResult:
    """Result from a batch analysis."""
    case_id: int
    success: bool
    result_data: Any
    execution_time: float
    error_message: Optional[str] = None


class ParallelAnalyzer:
    """
    Parallel execution manager for batch analyses.
    
    Supports:
    - Multi-threading for I/O bound tasks
    - Multi-processing for CPU bound tasks (careful with pickling)
    """
    
    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        """
        Initialize parallel analyzer.
        
        Args:
            max_workers: Maximum number of parallel workers
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.results: List[BatchResult] = []
        
    def run_batch(
        self,
        analysis_func: Callable,
        parameter_sets: List[Dict],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """
        Run batch analysis with multiple parameter sets.
        
        Args:
            analysis_func: Function to run for each parameter set
                          Signature: (params: Dict) -> Any
            parameter_sets: List of parameter dictionaries
            progress_callback: Optional progress callback (current, total)
            
        Returns:
            List of BatchResult objects
        """
        self.results = []
        total = len(parameter_sets)
        completed = 0
        
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_id = {}
            for i, params in enumerate(parameter_sets):
                future = executor.submit(self._run_single, analysis_func, i, params)
                future_to_id[future] = i
            
            # Collect results
            for future in as_completed(future_to_id):
                case_id = future_to_id[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    self.results.append(BatchResult(
                        case_id=case_id,
                        success=False,
                        result_data=None,
                        execution_time=0,
                        error_message=str(e)
                    ))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        # Sort by case_id
        self.results.sort(key=lambda x: x.case_id)
        return self.results
    
    def _run_single(
        self,
        func: Callable,
        case_id: int,
        params: Dict
    ) -> BatchResult:
        """Run a single analysis case."""
        start_time = time.time()
        
        try:
            result_data = func(params)
            elapsed = time.time() - start_time
            
            return BatchResult(
                case_id=case_id,
                success=True,
                result_data=result_data,
                execution_time=elapsed
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return BatchResult(
                case_id=case_id,
                success=False,
                result_data=None,
                execution_time=elapsed,
                error_message=str(e)
            )
    
    def get_summary(self) -> str:
        """Get summary of batch results."""
        if not self.results:
            return "No results"
            
        success = sum(1 for r in self.results if r.success)
        failed = len(self.results) - success
        total_time = sum(r.execution_time for r in self.results)
        avg_time = total_time / len(self.results)
        
        return (
            f"Batch Analysis Summary:\n"
            f"  Total cases: {len(self.results)}\n"
            f"  Successful: {success}\n"
            f"  Failed: {failed}\n"
            f"  Total time: {total_time:.2f}s\n"
            f"  Average time: {avg_time:.2f}s per case"
        )


def parallel_matrix_assembly(
    elements: List,
    ndof: int,
    matrix_func: str = 'get_stiffness_matrix',
    num_threads: int = 4
) -> np.ndarray:
    """
    Parallel assembly of global matrix from element matrices.
    
    For large models, this can significantly speed up matrix assembly
    by computing element matrices in parallel.
    
    Args:
        elements: List of element objects
        ndof: Number of global DOFs
        matrix_func: Name of element method to call
        num_threads: Number of threads to use
        
    Returns:
        Assembled global matrix
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Pre-allocate result
    K_global = np.zeros((ndof, ndof))
    
    def compute_element_contribution(elem):
        """Compute single element contribution."""
        if not hasattr(elem, matrix_func):
            return None, None
            
        k_el = getattr(elem, matrix_func)()
        indices = elem.get_element_dof_indices() if hasattr(elem, 'get_element_dof_indices') else []
        
        return k_el, indices
    
    # Compute element matrices in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_element_contribution, elem) for elem in elements]
        
        for future in futures:
            k_el, indices = future.result()
            if k_el is None:
                continue
                
            # Assembly (must be sequential due to race conditions)
            n_local = len(indices)
            for r in range(n_local):
                if indices[r] == -1:
                    continue
                for c in range(n_local):
                    if indices[c] == -1:
                        continue
                    K_global[indices[r], indices[c]] += k_el[r, c]
    
    return K_global


def vectorized_response_spectrum(
    time_array: np.ndarray,
    acceleration: np.ndarray,
    periods: np.ndarray,
    damping: float = 0.05
) -> np.ndarray:
    """
    Vectorized response spectrum calculation for performance.
    
    Uses NumPy vectorization for faster computation of SDOF responses.
    
    Args:
        time_array: Time array
        acceleration: Ground acceleration
        periods: Array of periods to compute
        damping: Damping ratio
        
    Returns:
        Array of spectral acceleration values
    """
    dt = time_array[1] - time_array[0]
    n_steps = len(acceleration)
    n_periods = len(periods)
    
    # Pre-compute coefficients for all periods
    omega = 2 * np.pi / np.maximum(periods, 1e-10)
    zeta = damping
    
    # Newmark parameters
    beta = 0.25
    gamma = 0.5
    
    # Result array
    Sa = np.zeros(n_periods)
    
    # For each period (could be parallelized further)
    for i in range(n_periods):
        w = omega[i]
        c = 2 * zeta * w
        k = w**2
        
        # Integration constants
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2 * beta) - 1.0)
        
        k_eff = k + a0 + a1 * c
        
        u = 0.0
        v = 0.0
        a = -acceleration[0]
        
        max_abs_acc = 0.0
        
        for j in range(n_steps - 1):
            p_eff = -acceleration[j+1] + a0 * u + a2 * v + a3 * a
            p_eff += c * (a1 * u + a4 * v + a5 * a)
            
            u_new = p_eff / k_eff
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)
            
            # Absolute acceleration
            abs_acc = abs(a_new + acceleration[j+1])
            if abs_acc > max_abs_acc:
                max_abs_acc = abs_acc
            
            u, v, a = u_new, v_new, a_new
        
        Sa[i] = max_abs_acc
    
    return Sa
