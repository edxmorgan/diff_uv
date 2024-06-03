# Diff_UV (Differentiable Underwater Vehicle System)
A differentiable Underwater vehicles dynamic model with actuation in all DOFs and can control the position and attitude in 6 DOFs

## Installation

The Diff_UV library has been implemented as a python3 pip package. To install and use this library in your own project, simply clone this
repository to your workspace:

```bash
cd path/to/src
git clone https://github.com/Eddy-Morgan/Diff_UV.git
pip install .
```

## Getting Started

All hydrodynamic terms implemented in this project have been defined using Fossen's equations
for hydrodynamics. The hydrodynamic terms implemented include:

- Mass: rigid body inertia and added mass
- Coriolis: centripetal, coriolis, and added coriolis
- Damping: linear and quadratic damping
- Restoring forces: buoyancy and gravitational forces

Each of the aforementioned terms provide their own distinct data class for independent use
or can be managed altogether within the `diffUV` class. 

## License

The Diff_UV library has been released under the MIT license.
