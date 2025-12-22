"""Test script for element response tracking."""
from src.fem import Node
from src.fem_3d import BeamColumn3D
from src.element_analyzer import ElementResponseAnalyzer

# Create nodes for columns and beams
n1 = Node(0, 0, 0, 0)
n2 = Node(1, 0, 0, 3.5, mass=20000)
n3 = Node(2, 6, 0, 3.5, mass=20000)  # Horizontal from n2

n1.set_dof_indices([-1]*6)
n2.set_dof_indices([0,1,2,3,4,5])
n3.set_dof_indices([6,7,8,9,10,11])

# Column (vertical)
col = BeamColumn3D(0, n1, n2, 2.5e10, 1e10, 0.25, 0.005, 0.005, 0.01)
col.set_yield_properties(200000, 200000)

# Beam (horizontal)
beam = BeamColumn3D(1, n2, n3, 2.5e10, 1e10, 0.20, 0.004, 0.004, 0.008)
beam.set_yield_properties(150000, 150000)

elements = [col, beam]

print("Element type detection:")
print(f"  Column (n1->n2): {col.element_type}")
print(f"  Beam (n2->n3): {beam.element_type}")

# Test analyzer
analyzer = ElementResponseAnalyzer(elements)
print(f"\nColumns found: {len(analyzer.columns)}")
print(f"Beams found: {len(analyzer.beams)}")

# Get summary
summary = analyzer.get_summary_by_type()
print(f"\nColumn stats: count={summary['columns']['count']}")
print(f"Beam stats: count={summary['beams']['count']}")

# Test damage index
print(f"\nColumn damage_index: {col.damage_index:.4f}")
print(f"Beam damage_index: {beam.damage_index:.4f}")

# Test response summary
resp = col.get_response_summary()
print(f"\nColumn response summary type: {resp['type']}")

# Enable history tracking
col.track_history = True
beam.track_history = True

print("\nHistory tracking enabled for both elements")

# Print full summary
analyzer.print_summary()

print("\n=== ALL TESTS PASSED ===")
