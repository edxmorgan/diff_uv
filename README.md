# Diff_UV (Differentiable Underwater Vehicle System)
A differentiable Underwater vehicles dynamic model with actuation in all DOFs and can control the position and attitude in 6 DOFs based on casadi operations.

<!-- ![alt text]() -->
<img src="./images/BlueRobotics%202018b.png" width="420"/>

The matrices 𝑀, 𝐶(𝜈) and 𝐷(𝜈), and vector 𝑔(𝜂) in the dynamics contain more than 300 unknown parameters in total. As a result, estimation of all
parameters is infeasible. Yet, based on the features and operating speeds of the vehicle,
several assumptions can be made to simplify the dynamic model and reduce the number of
unknown parameters in the model. 

The assumptions that have been made for the dynamics
of a lightweight underwater vehicle are listed in the following:

* Operates at relative low speeds (i.e. less than 2 m/s), lift
forces can be neglected.
* Assumed to have port-starboard symmetry and fore-aft
symmetry; and the centre of gravity (CG) is assumed to be located in the symmetry
planes.
* Assumed to be hydrodynamically symmetrical about 6-DoF.
Accordingly, the motions between DoFs of the vehicle in hydrodynamic can be
decoupled.
* Assumed to operate below the wave-affected zone. As a result,
disturbances of waves on the vehicle are negligible.

## Getting Started
To use Diff_UV in your own project, simply clone this
repository to your workspace:

```bash
cd path/to/src
git clone https://github.com/Eddy-Morgan/Diff_UV.git
```

All hydrodynamic terms implemented in this project have been defined using Fossen's equations. The dynamic terms implemented include:
- Mass: rigid body inertia and added mass in body, ned and quaternion.
- Coriolis: centripetal, coriolis, and added coriolis in body, ned and quaternion.
- Damping: linear and quadratic damping in body, ned and quaternion.
- Restoring forces: buoyancy and gravitational forces in body, ned and quaternion.
- Forward dynamics: in body, ned and quaternion(to be updated).
- Inverse dynamics: in body, ned and quaternion(to be updated).

Each of the aforementioned terms provide their own distinct data class for independent use
or can be managed altogether within the `diffUV` class. 

```python
from diffUV import dyn_body,dyned_eul, kin
uv_dyn = dyn_body()
uv_dyned = dyned_eul()

inertia_mat = uv_dyn.body_inertia_matrix()
coriolis_mat = uv_dyn.body_coriolis_centripetal_matrix()
restoring_vec = uv_dyn.body_restoring_vector()
dampn_mat = uv_dyn.body_damping_matrix()

ned_accel = uv_dyned.ned_euler_forward_dynamics()
```
For detailed usage examples of the Diff_UV, see [Jupyter notebook](https://github.com/edxmorgan/Diff_UV/blob/main/usage.ipynb).

### Extending with CasADi Capabilities
All expressions obtained from the diffUV methods are of CasADi type. This allows them to be integrated with CasADi's advanced functionalities for optimization, symbolic computations, and numerical integrations.

### Symbolic Differentiation
Utilize CasADi's automatic differentiation to compute derivatives:
```python
from casadi import jacobian
accel_jacobian = jacobian(ned_accel, uv_dyned.ned_state_vector)
```

## References
Fossen, T.I. (2011) Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley & Sons, Inc., Chichester, UK.
https://doi.org/10.1002/9781119994138
