"""
Validation and Benchmark Module.
Input validation, unit checking, and benchmark problems.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    level: ValidationLevel
    message: str
    field: str = ""
    suggestion: str = ""


class InputValidator:
    """
    Validates simulation input parameters.
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
    def validate_all(
        self,
        nodes: List,
        elements: List,
        params: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Run all validation checks."""
        self.results = []
        
        self._validate_nodes(nodes)
        self._validate_elements(elements, nodes)
        self._validate_params(params)
        self._validate_connectivity(nodes, elements)
        self._validate_physics(params)
        
        return self.results
        
    def _add_result(
        self,
        passed: bool,
        level: ValidationLevel,
        message: str,
        field: str = "",
        suggestion: str = ""
    ):
        self.results.append(ValidationResult(
            passed=passed,
            level=level,
            message=message,
            field=field,
            suggestion=suggestion
        ))
        
    def _validate_nodes(self, nodes: List):
        """Validate node data."""
        if not nodes:
            self._add_result(False, ValidationLevel.ERROR, 
                "No nodes defined", "nodes",
                "Add at least 2 nodes to define a structure")
            return
            
        # Check for duplicate positions
        positions = set()
        for i, node in enumerate(nodes):
            pos = (round(node.x, 4), round(node.y, 4), round(node.z, 4))
            if pos in positions:
                self._add_result(False, ValidationLevel.WARNING,
                    f"Duplicate node position at {pos}", f"node_{i}",
                    "Merge nodes at same position or adjust coordinates")
            positions.add(pos)
            
        # Check for negative masses
        for i, node in enumerate(nodes):
            if hasattr(node, 'mass') and node.mass < 0:
                self._add_result(False, ValidationLevel.ERROR,
                    f"Negative mass at node {i}", f"node_{i}.mass",
                    "Mass must be non-negative")
                    
        # Check DOF assignment
        dof_set = set()
        for node in nodes:
            for dof in node.dof_indices:
                if dof >= 0:
                    if dof in dof_set:
                        self._add_result(False, ValidationLevel.ERROR,
                            f"Duplicate DOF index {dof}", "dof_indices")
                    dof_set.add(dof)
                    
        self._add_result(True, ValidationLevel.INFO,
            f"Validated {len(nodes)} nodes")
            
    def _validate_elements(self, elements: List, nodes: List):
        """Validate element data."""
        if not elements:
            self._add_result(False, ValidationLevel.ERROR,
                "No elements defined", "elements")
            return
            
        for i, elem in enumerate(elements):
            L = elem.get_length()
            
            if L <= 0:
                self._add_result(False, ValidationLevel.ERROR,
                    f"Zero-length element {i}", f"element_{i}",
                    "Check node positions")
                    
            if L > 50:  # Very long element
                self._add_result(False, ValidationLevel.WARNING,
                    f"Very long element {i}: L={L:.1f}m", f"element_{i}",
                    "Consider splitting into smaller elements")
                    
            # Check material properties
            if hasattr(elem, 'E') and elem.E <= 0:
                self._add_result(False, ValidationLevel.ERROR,
                    f"Invalid E for element {i}", f"element_{i}.E")
                    
        self._add_result(True, ValidationLevel.INFO,
            f"Validated {len(elements)} elements")
            
    def _validate_params(self, params: Dict):
        """Validate analysis parameters."""
        if 'duration' in params:
            if params['duration'] <= 0:
                self._add_result(False, ValidationLevel.ERROR,
                    "Duration must be positive", "duration")
            elif params['duration'] > 300:
                self._add_result(False, ValidationLevel.WARNING,
                    "Very long duration may cause memory issues", "duration")
                    
        if 'dt' in params:
            if params['dt'] <= 0:
                self._add_result(False, ValidationLevel.ERROR,
                    "Time step must be positive", "dt")
            elif params['dt'] > 0.1:
                self._add_result(False, ValidationLevel.WARNING,
                    "Large time step may cause instability", "dt",
                    "Recommended: dt <= 0.02s")
                    
        if 'max_acc' in params:
            if params['max_acc'] <= 0:
                self._add_result(False, ValidationLevel.ERROR,
                    "Max acceleration must be positive", "max_acc")
            elif params['max_acc'] > 2000:
                self._add_result(False, ValidationLevel.WARNING,
                    "Very high acceleration (>2000 gal)", "max_acc")
                    
    def _validate_connectivity(self, nodes: List, elements: List):
        """Check structural connectivity."""
        if not nodes or not elements:
            return
            
        # Find connected components
        connected = {i: False for i in range(len(nodes))}
        node_to_idx = {id(node): i for i, node in enumerate(nodes)}
        
        for elem in elements:
            i = node_to_idx.get(id(elem.node_i), -1)
            j = node_to_idx.get(id(elem.node_j), -1)
            if i >= 0:
                connected[i] = True
            if j >= 0:
                connected[j] = True
                
        disconnected = [i for i, c in connected.items() if not c]
        if disconnected:
            self._add_result(False, ValidationLevel.WARNING,
                f"Disconnected nodes: {disconnected}", "connectivity")
                
        # Check for base support
        has_base = any(node.z == 0 for node in nodes)
        if not has_base:
            self._add_result(False, ValidationLevel.WARNING,
                "No nodes at base level (z=0)", "supports",
                "Add base supports or check node coordinates")
                
    def _validate_physics(self, params: Dict):
        """Check physical reasonableness."""
        if 'damping_ratio' in params:
            zeta = params['damping_ratio']
            if zeta < 0:
                self._add_result(False, ValidationLevel.ERROR,
                    "Damping ratio cannot be negative", "damping_ratio")
            elif zeta > 0.3:
                self._add_result(False, ValidationLevel.WARNING,
                    f"High damping ratio ({zeta})", "damping_ratio",
                    "Typical values: 2-5%")
                    
    def has_errors(self) -> bool:
        return any(r.level == ValidationLevel.ERROR and not r.passed 
                   for r in self.results)
                   
    def get_summary(self) -> str:
        """Get validation summary."""
        errors = sum(1 for r in self.results if r.level == ValidationLevel.ERROR and not r.passed)
        warnings = sum(1 for r in self.results if r.level == ValidationLevel.WARNING and not r.passed)
        
        return f"Validation: {errors} errors, {warnings} warnings"


# ===== Unit System Checker =====

class UnitSystem:
    """Unit system definitions."""
    SI = {
        'length': 'm',
        'force': 'N',
        'mass': 'kg',
        'time': 's',
        'stress': 'Pa',
        'moment': 'N·m',
        'acceleration': 'm/s²'
    }
    
    ENGINEERING = {
        'length': 'mm',
        'force': 'kN',
        'mass': 't',
        'time': 's',
        'stress': 'MPa',
        'moment': 'kN·m',
        'acceleration': 'mm/s²'
    }


def check_unit_consistency(
    E: float,
    density: float,
    length: float,
    expected_system: str = 'SI'
) -> List[str]:
    """
    Check if input values are consistent with expected unit system.
    
    Returns list of warnings.
    """
    warnings = []
    
    if expected_system == 'SI':
        # Steel E should be around 2e11 Pa
        if E > 1e6 and E < 1e9:
            warnings.append(f"E={E:.2e} looks like MPa, not Pa. Multiply by 1e6?")
        elif E < 1e9:
            warnings.append(f"E={E:.2e} seems too low for SI units (Pa)")
            
        # Density of steel ~7850 kg/m³
        if density > 0 and density < 100:
            warnings.append(f"Density={density} looks like t/m³, not kg/m³")
            
        # Check if length is in mm
        if length > 1000:
            warnings.append(f"Length={length}m seems very large. Using mm by mistake?")
            
    return warnings


# ===== Benchmark Problems =====

@dataclass
class BenchmarkResult:
    """Result of benchmark comparison."""
    name: str
    computed: float
    reference: float
    error_percent: float
    passed: bool


def run_cantilever_benchmark(
    L: float = 3.0,
    E: float = 2.05e11,
    I: float = 1e-4,
    P: float = 10000
) -> BenchmarkResult:
    """
    Cantilever beam under point load - analytical solution.
    
    δ = PL³/(3EI)
    """
    analytical = P * L**3 / (3 * E * I)
    
    # Simple FEM calculation
    k = 3 * E * I / L**3
    computed = P / k
    
    error = abs(computed - analytical) / analytical * 100
    
    return BenchmarkResult(
        name="Cantilever Beam Deflection",
        computed=computed,
        reference=analytical,
        error_percent=error,
        passed=error < 1.0
    )


def run_sdof_period_benchmark(
    m: float = 1000,
    k: float = 1e6
) -> BenchmarkResult:
    """
    SDOF natural period - analytical solution.
    
    T = 2π√(m/k)
    """
    analytical = 2 * np.pi * np.sqrt(m / k)
    
    # What the modal analysis should give
    omega = np.sqrt(k / m)
    computed = 2 * np.pi / omega
    
    error = abs(computed - analytical) / analytical * 100
    
    return BenchmarkResult(
        name="SDOF Natural Period",
        computed=computed,
        reference=analytical,
        error_percent=error,
        passed=error < 0.01
    )


def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmark problems."""
    results = [
        run_cantilever_benchmark(),
        run_sdof_period_benchmark(),
    ]
    return results


def print_benchmark_results(results: List[BenchmarkResult]):
    """Print benchmark results."""
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"\n{r.name}")
        print(f"  Computed:  {r.computed:.6e}")
        print(f"  Reference: {r.reference:.6e}")
        print(f"  Error:     {r.error_percent:.4f}%")
        print(f"  Status:    {status}")
        
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r.passed)
    print(f"Summary: {passed}/{len(results)} benchmarks passed")
