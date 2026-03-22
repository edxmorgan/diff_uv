# diffUV
A differentiable 6-DOF underwater vehicle dynamics, with tooling for system identification, estimation, and control.

<img src="./resources/diff_uv_flow_diag.png" width="840"/>

<p align="center">
  <img src="./resources/dynamic_auv.gif" width="48%" alt="dynamic motions" />
</p>


<!-- ## Overview -->
The core library provides symbolic kinematics and dynamics derived from Fossen's
formulations. All major terms are represented as CasADi expressions, enabling
analytic derivatives (gradients, Jacobians, Hessians) for optimization, estimation,
and control workflows.

## Scope and modeling assumptions
- **Low-speed regime**: less than 2 m/s, lift forces neglected.
- **Symmetry**: port-starboard and fore-aft symmetry; CG in symmetry planes.
- **Hydrodynamic decoupling**: symmetric 6-DOF motions treated independently.
- **Below wave zone**: wave disturbances are negligible.

## Capabilities and implementation status
- [x] Symbolic kinematics and dynamics (body, NED, quaternion).
- [x] Forward and inverse dynamics.
- [x] Added mass, damping, Coriolis/centripetal, and restoring force models.
- [x] System identification utilities (CVXPY-based estimator).
- [x] EKF and nonlinear PID helpers.
- [x] CasADi code generation for C++/MATLAB/Python.
- [x] Gymnasium environment and RL training script.
- [x] JIT support.

<!-- ## Installation
Clone the repo and install the core library:

```bash
cd path/to/src
git clone https://github.com/edxmorgan/diff_uv.git
cd diff_uv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For notebooks, plotting, and system ID dependencies:

```bash
pip install -r requirements.txt
``` -->

## Core library usage

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

## CasADi integration
### Symbolic differentiation

```python
from casadi import jacobian
accel_jacobian = jacobian(v_dot, uv_dyn.body_state_vector)
```

### Code generation

```python
import os
from casadi import Function
from diffUV.utils.symbols import *

I_o = vertcat(I_x, I_y, I_z, I_xz)  # rigid body inertia wrt body origin
decoupled_added_m = vertcat(X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr)  # added mass in diagonals
coupled_added_m = vertcat(X_dq, Y_dp, N_dp, M_du, K_dv)  # effective added mass in non diagonals

M_func = Function('M_b', [m, I_o, z_g, decoupled_added_m, coupled_added_m], [inertia_mat])
M_func.generate("M_b.c")
os.system("gcc -fPIC -shared M_b.c -o libM_b.so")
```

## System identification
The core implementation is in `diffUV/system_id.py` (`MarineVehicleEstimator`).
The estimator formulates a convex fit in CVXPY with physical constraints; MOSEK is
used by default, but can be replaced in the solver call.

### Data requirements (vehicle log)
The identification notebook expects a single CSV with the following columns:
- `timestamp`
- `imu_roll_unwrap`, `imu_pitch_unwrap`, `imu_yaw_unwrap`
- `imu_ang_vel_x`, `imu_ang_vel_y`, `imu_ang_vel_z`
- `dvl_speed_x`, `dvl_speed_y`, `dvl_speed_z`
- `depth_from_pressure2`
- `base_x_force`, `base_y_force`, `base_z_force`
- `base_x_torque`, `base_y_torque`, `base_z_torque`

## References
Fossen, T.I. (2011) Handbook of Marine Craft Hydrodynamics and Motion Control.
John Wiley & Sons, Inc., Chichester, UK.
https://doi.org/10.1002/9781119994138