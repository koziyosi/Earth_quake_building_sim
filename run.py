#!/usr/bin/env python3
import sys
import os

# === CRITICAL: Set parallel processing BEFORE importing numpy ===
# This enables multi-threaded BLAS/LAPACK operations
_num_threads = str(os.cpu_count() or 4)
os.environ['OMP_NUM_THREADS'] = _num_threads
os.environ['MKL_NUM_THREADS'] = _num_threads
os.environ['OPENBLAS_NUM_THREADS'] = _num_threads
os.environ['NUMEXPR_NUM_THREADS'] = _num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = _num_threads

# Ensure root is in path
sys.path.append(os.path.dirname(__file__))

# Warm up Numba JIT (optional but improves first-run performance)
try:
    from src.numba_accel import NumbaAccelerator
    NumbaAccelerator.warmup()
except ImportError:
    print("Note: Numba acceleration not available")
except Exception as e:
    print(f"Numba warmup skipped: {e}")

try:
    from src.gui import main
except ImportError as e:
    print(f"Error importing GUI: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print(f"Launching EarthQuake Building Sim GUI... (Using {_num_threads} CPU threads)")
    main()

