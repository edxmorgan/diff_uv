# Diff_UV (Differentiable Underwater Vehicle System)
A differentiable Underwater vehicles dynamic model with actuation in all DOFs and can control the position and attitude in 6 DOFs based on casadi operations.

These various matrices 𝑀, 𝐶(𝜈) and 𝐷(𝜈), and vector 𝑔(𝜂) in the dynamics contain more than 300 unknown parameters in total. As a result, estimation of all
parameters is infeasible. Yet, based on the features and operating speeds of the vehicle,
several assumptions can be made to simplify the dynamic model and reduce the number of
unknown parameters in the model. 

The assumptions that have been made for the dynamics
of a commond vehicle like the BlueROV2 Heavy are listed in the following:

BlueROV2 Heavy operates at relative low speeds (i.e. less than 2 m/s), lift
forces can be neglected.
2. BlueROV2 Heavy is assumed to have port-starboard symmetry and fore-aft
symmetry; and the centre of gravity (CG) is assumed to be located in the symmetry
planes.
3. BlueROV2 Heavy is assumed to be hydrodynamically symmetrical about 6-DoF.
Accordingly, the motions between DoFs of the vehicle in hydrodynamic can be
decoupled.
4. BlueROV2 Heavy is assumed to operate below the wave-affected zone. As a result,
disturbances of waves on the vehicle are negligible.

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
