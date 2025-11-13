# Diff_UV (Differentiable Underwater Vehicle Model)

A differentiable underwater vehicle dynamic model in 6 DOFs built entirely with CasADi operations.

🌊🐠🤖

<img src="./docs/image.png" width="840"/>

Diff_UV provides symbolic models for the kinematics and dynamics of an underwater vehicle. All expressions are CasADi based, which enables efficient computation of gradients, Jacobians, and Hessians. Through CasADi, these models integrate easily with state of the art solvers such as SUNDIALS, IPOPT, and FATROP for simulation, estimation, and optimization. Code generation is supported for C plus plus, Python, and MATLAB or Octave.

Differentiable models are central to underwater robotics because they make gradient based optimization and learning practical. They support high performance control, adaptation, parameter estimation, reinforcement learning, and real time MPC.

---

## 🌊 Model Assumptions

The underwater vehicle model relies on the following assumptions:

* Operates at relatively low speeds, below about 2 m per second, which allows neglecting lift forces.
* Port starboard symmetry and fore aft symmetry, with the center of gravity lying on these symmetry planes.
* Hydrodynamic symmetry about all 6 DOFs, which leads to decoupled hydrodynamic motions.
* Operates below the wave affected zone, so wave disturbances are negligible.

---

## 🚀 Getting Started

Clone the repository into your workspace:

```bash
cd path/to/src
git clone https://github.com/edxmorgan/diff_uv.git
```

All implemented kinematic and hydrodynamic terms follow Fossen's formulations. The library supports:

* Kinematics, rotation and coordinate transformations
* Mass, rigid body inertia and added mass in body, NED, and quaternion frames
* Coriolis, centripetal, and added mass Coriolis terms
* Damping, linear and quadratic damping models
* Restoring forces, buoyancy and gravity
* Forward dynamics
* Inverse dynamics

Each component is accessible through dedicated methods and is organized within the `diffUV` class.

```python
from diffUV import dyn_body, dyned_eul, kin

uv_dyn = dyn_body()
uv_dyned = dyned_eul()

inertia_mat = uv_dyn.body_inertia_matrix()
coriolis_mat = uv_dyn.body_coriolis_centripetal_matrix()
restoring_vec = uv_dyn.body_restoring_vector()
dampn_mat = uv_dyn.body_damping_matrix()

v_dot = uv_dyn.body_forward_dynamics()
```

---

## 📋 Todo / Implementation Status

* [x] Kinematics model
* [x] Forward dynamics
* [x] Simulation utilities
* [x] Inverse dynamics
* [x] Nonlinear PID controller
* [x] JIT support
* [ ] GPU support
* [x] C plus plus code generation

Example usage can be found in the project Jupyter notebook:
[https://github.com/edxmorgan/Diff_UV/blob/main/usage](https://github.com/edxmorgan/Diff_UV/blob/main/usage)

---

## ⚙️ Extending with CasADi Capabilities

All model outputs are CasADi symbolic expressions. They can be differentiated, optimized, integrated, or compiled.

### 🔧 Symbolic Differentiation Example

```python
from casadi import jacobian
accel_jacobian = jacobian(v_dot, uv_dyn.body_state_vector)
```

### 🛠️ Code Generation Example

```python
import os
from casadi import Function
from diffUV.utils.symbols import *

I_o = vertcat(I_x, I_y, I_z, I_xz)
decoupled_added_m = vertcat(X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr)
coupled_added_m = vertcat(X_dq, Y_dp, N_dp, M_du, K_dv)

M_func = Function('M_b', [m, I_o, z_g, decoupled_added_m, coupled_added_m], [inertia_mat])
M_func.generate("M_b.c")
os.system("gcc -fPIC -shared M_b.c -o libM_b.so")
```

### 🧩 C plus plus Usage Example

```cpp
#include <casadi/casadi.hpp>
using namespace casadi;

void diffuv_usage_cplusplus(){
  Function f = external("M_b", "libM_b.so");

  double m = 11.5;
  std::vector<double> Io = {0.16, 0.16, 0.16, 0};
  double z_g = 0.02;
  std::vector<double> added_m = {-5.5, -12.7, -14.57, -0.12, -0.12, -0.12};
  std::vector<double> coupl_added_m = {0, 0, 0, 0, 0};

  std::vector<DM> arg = {m, Io, z_g, added_m, coupl_added_m};
  std::vector<DM> res = f(arg);

  std::cout << "result (0): " << res.at(0) << std::endl;
}

int main(){
    diffuv_usage_cplusplus();
    return 0;
}
```

---

## 📚 References

Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley and Sons.
[https://doi.org/10.1002/9781119994138](https://doi.org/10.1002/9781119994138)

---

## 📑 Citing

```bibtex
@software{diffUV2024
  title = "diffUV: A Compact Library to retrieve symbolic representations of kinematics and dynamics of an underwater vehicle.",
  author = "Edward Morgan",
  year = "2024",
  url = {https://github.com/edxmorgan/diff_uv},
}
```

---

## 🤝 Contributing

If you find issues or documentation errors, please open an issue.
New features are welcome through pull requests.