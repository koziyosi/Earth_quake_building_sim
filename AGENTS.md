# AGENTS.md

This file defines the coding conventions and physics standards for the **EarthQuake Building sim**.

## Physics Conventions
- **Units**: SI Units are used internally unless otherwise specified.
  - Length: meters (m)
  - Force: Newtons (N)
  - Mass: Kilograms (kg)
  - Time: Seconds (s)
  - Acceleration: m/s^2 (Note: Input earthquake data in 'gal' must be converted: 1 gal = 0.01 m/s^2)
- **Coordinate System (2D)**:
  - X: Horizontal
  - Y: Vertical (Up is positive)
  - Theta: Counter-clockwise rotation is positive.
- **Degrees of Freedom (DOF)**:
  - 2D Node: [u_x, u_y, theta_z] (3 DOFs per node)

## Code Structure
- **Docstrings**: All classes and public methods must have docstrings explaining inputs and units.
- **Type Hinting**: Use Python type hints for clarity.
- **Numpy**: Use `np.array` for all vector/matrix operations. Avoid Python lists for math.

## Hysteresis Rules
- **Yielding**: Defined by yield moment (My) and yield rotation (theta_y).
- **Stiffness**: Initial stiffness (K0), post-yield stiffness ratio (r).
