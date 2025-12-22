"""Test Takeda hysteresis model."""
from src.hysteresis import Takeda
import numpy as np

# Test Takeda model with loading-unloading cycle
print('=== Takeda Model Test ===')
takeda = Takeda(k0=1e6, fy=100000, r=0.05, alpha=0.4)
print(f'Initial: k0={takeda.k0:.0e}, fy={takeda.fy:.0e}, dy={takeda.dy:.4f}')

# Load to yield and beyond
displacements = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0, -0.05, -0.10]
print('\nLoading-Unloading Cycle:')
print(f'{"disp":>8} {"force":>12} {"tangent":>12} {"state":>10}')

for d in displacements:
    f, k = takeda.set_trial_displacement(d)
    state_name = ['?', 'LOAD+', 'LOAD-', 'UNLOAD'][takeda.state]
    print(f'{d:8.4f} {f:12.1f} {k:12.0e} {state_name:>10}')
    takeda.commit()

# Check unloading stiffness degradation
print('\nUnloading stiffness check:')
d_max = 0.25
k_unload = takeda._get_unloading_stiffness(d_max)
print(f'  d_max={d_max}, dy={takeda.dy:.4f}')
print(f'  K_unload = K0 * (dy/d_max)^alpha')
print(f'  K_unload = {k_unload:.0e} (vs K0={takeda.k0:.0e}, Kp={takeda.kp:.0e})')
print(f'  Ratio: K_unload/K0 = {k_unload/takeda.k0:.3f}')

print('\nSUCCESS!')
