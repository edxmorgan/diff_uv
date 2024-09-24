# Diff_UV (Differentiable Underwater Vehicle Model)
A differentiable Underwater vehicles dynamic model in 6 DOFs based on casadi operations.

<!-- ![alt text]() -->
<img src="./images/BlueRobotics%202018b.png" width="420"/>

This library provides capabilities to generate the kinematics and dynamics models of an underwater vehicle in a symbolically.
This makes it possible to solve for gradients, hessians and jacobians using [casadi](https://web.casadi.org/). 
Through casadi, the dynamics can be interfaced with state-of-the-art codes such as SUNDIALS, IPOPT, FATROP etc for integration and optimisation. It is also possible to do code generation to be used from C++, Python or Matlab/Octave. 

The following assumptions have been made for the dynamics
of the underwater vehicle model:

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

All kinematics & hydrodynamic terms implemented in this project have been defined using Fossen's formulations. The terms implemented include:
- Kinematics : Rotation & Coordinate Transformation Matrices
- Mass: rigid body inertia and added mass in body, ned and quaternion.
- Coriolis: centripetal, coriolis, and added coriolis in body, ned and quaternion.
- Damping: linear and quadratic damping in body, ned and quaternion.
- Restoring forces: buoyancy and gravitational forces in body, ned and quaternion.
- Forward dynamics
- Inverse dynamics

Each of the aforementioned terms provide their own distinct data methods for independent use
and are managed altogether within the `diffUV` class. 

```python
from diffUV import dyn_body,dyned_eul, kin
uv_dyn = dyn_body()
uv_dyned = dyned_eul()

inertia_mat = uv_dyn.body_inertia_matrix()
coriolis_mat = uv_dyn.body_coriolis_centripetal_matrix()
restoring_vec = uv_dyn.body_restoring_vector()
dampn_mat = uv_dyn.body_damping_matrix()

v_dot = uv_dyn.body_forward_dynamics()
```
For detailed usage examples of the Diff_UV, see [Jupyter notebook](https://github.com/edxmorgan/Diff_UV/blob/main/usage).

### Extending with CasADi Capabilities
All expressions obtained from the diffUV methods are of CasADi type. This allows them to be integrated with CasADi's advanced functionalities for optimization, symbolic computations, and numerical integrations.

### Symbolic Differentiation
Utilize CasADi's automatic differentiation to compute derivatives:
```python
from casadi import jacobian
accel_jacobian = jacobian(v_dot, uv_dyn.body_state_vector)
```

### Code Generation
Expressions can be directly exported to MATLAB and C++ formats, for integration with external systems and applications.
```python
import os
from casadi import Function
from diffUV.utils.symbols import *

I_o = vertcat(I_x, I_y, I_z,I_xz) # rigid body inertia wrt body origin
decoupled_added_m = vertcat(X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr) # added mass in diagonals
coupled_added_m =  vertcat(X_dq, Y_dp, N_dp, M_du, K_dv) # effective added mass in non diagonals 


M_func = Function('M_b', [m, I_o, z_g, decoupled_added_m, coupled_added_m], [inertia_mat]) # for both numerical & symbolic use
M_func.generate("M_b.c")
os.system(f"gcc -fPIC -shared M_b.c -o libM_b.so")
```

```cpp
// C++ (and CasADi)
#include <casadi/casadi.hpp>
using namespace casadi;

void diffuv_usage_cplusplus(){
  std::cout << "---" << std::endl;
  std::cout << "Usage from CasADi C++:" << std::endl;
  std::cout << std::endl;

  // Use CasADi's "external" to load the compiled function
  Function f = external("M_b", "libM_b.so");

  // Use like any other CasADi function
  double m = 11.5;
  std::vector<double> Io = {0.16, 0.16, 0.16, 0};
  double z_g = 0.02;
  std::vector<double> added_m = {-5.5 , -12.7 , -14.57,  -0.12,  -0.12,  -0.12};
  std::vector<double> coupl_added_m = {0, 0, 0, 0, 0}; // assuming decoupling motion
  std::vector<DM> arg = {m, Io, z_g, added_m, coupl_added_m};
  std::vector<DM> res = f(arg);

  std::cout << "result (0): " << res.at(0) << std::endl;
  std::cout << "result (1): " << res.at(1) << std::endl;
}

int main()
{
    diffuv_usage_cplusplus();
    return 0;
}
```

## References
Fossen, T.I. (2011) Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley & Sons, Inc., Chichester, UK.
https://doi.org/10.1002/9781119994138
