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
- [x] Nominal-dynamics disturbance EKF with generated C/shared-library export
  ([notebook](usage/uv_ekf.ipynb)). The ROS package owns runtime integration.
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

The current online workflow uses the same parameterization and physical
constraints, with a shared integral regressor and a native C++/MOSEK candidate
node in the ROS simulator.
It does not estimate a calibrated posterior or automatically update control.

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

### Integration rule for generated models

Keep kinematics, dynamics and estimator generation in the existing source
notebooks. SimLab and the simulator consume the exported `.casadi` functions
or compiled `.so` libraries through their existing adapters. Update the existing
export cells when fixing those models; do not add a parallel `export_*.py`
script, a source-directory import, or a new runtime Python package for them.
Retries, candidate selection, ROS integration and command validation stay in
the consuming adapter. Regenerate affected artifacts and verify their interfaces
and numerical behavior before deploying them.

Vehicle dynamics default to full immersion. Buoyancy is independent of absolute
depth; depth zero is a coordinate datum, not an air/water switch. Floating,
emergence and partial immersion are outside this model. With weight greater
than buoyancy, holding depth requires actuator effort.

Use the canonical export_vehicle function in usage/code_generation.ipynb for
deployment. It generates plant, EKF, control-model, identification and optional
CPU/CUDA artifacts from the same source, with a provenance manifest. The EKF
notebook is an analysis example rather than a separate deployment exporter.

Public SimLab serialized artifacts must be exported with stock CasADi 3.7.2.
In `usage/code_generation.ipynb`, use `export_vehicle(output, native=False)`
for the serialized models without compiling native libraries. The same entry
point also exports `rotation_log.casadi`. Custom CUDA generation can require a
newer CasADi toolchain; do not replace the public serialized bundle with that
build's files without testing the supported 3.7.2 reader.

### Exported vehicle sliding-mode controller

`Controller.sliding_mode` implements a model-based six-DOF sliding controller
for the fully submerged, constant-inertia, zero-current vehicle. Its design is
inspired by [Haber's pendulum example](https://aleksandarhaber.com/matlab-and-simulink-implementation-of-sliding-mode-controller-for-nonlinear-systems/),
with the vehicle's full inertia, Coriolis, damping and restoring terms and a
shortest SO(3) orientation error. It is not a direct copy of the scalar law.

With body-frame pose error `e`, transported reference twist `nu_d`, and positive
surface gains `Lambda`, define `nu_r = nu_d + Lambda*e`, `s = nu - nu_r`.
The reference derivative includes frame transport and the exact directional
derivative of the rotation logarithm. The wrench is

```
tau = M*nu_r_dot + C*nu + D*nu + g - K*s - rho*sat(s/phi)
```

For constant symmetric positive-definite `M`, exact nominal dynamics and matched
residual disturbance `d`, `V = s.T*M*s/2` gives
`V_dot = -s.T*K*s - sum(rho_i*s_i*sat(s_i/phi_i)) + s.T*d`.
Zero boundary width selects `sign(s)` and the ideal reaching condition requires
`rho_i > |d_i|`. Positive widths reduce chattering but give practical boundary
layers, not exact finite-time sliding. Sampling, actuator saturation, unmatched
model errors and the SO(3) half-turn cut limit these claims; no globally smooth
attitude controller or automatic robust-gain bound is claimed.

The existing `usage/code_generation.ipynb` `export_vehicle(...)` exports
`uv_sliding_mode_controller.casadi`. With `native=True`, it also exports
`libsliding_mode_controller.so` (symbol `sliding_mode_control`). Use stock
CasADi 3.7.2 for portable serialized assets. No runtime diffUV import is needed:

```python
import casadi as ca
smc = ca.Function.load('uv_sliding_mode_controller.casadi')
# Alternatively:
# smc = ca.external('sliding_mode_control', './libsliding_mode_controller.so')
result = smc(parameters=p27, state=state12, desired_pose=pose6,
             desired_twist=twist6, desired_acceleration=acceleration6,
             surface_gain=lambda6, damping_gain=k6,
             switching_gain=rho6, boundary=phi6)
wrench = result['wrench']
```

Contract: `state12=[NED position, roll/pitch/yaw, FRD linear/angular velocity]`;
`desired_pose` is NED position and attitude; desired twist is expressed in the
**desired body's axes**, and acceleration is the derivative of those components.
Output is current-body FRD force [N] and torque [Nm]. Diagnostics are `sliding`
[m/s, rad/s] and body-frame `pose_error` [m, rad]. Gains must be finite:
`lambda6>0` [1/s], `k6>=0` [N/(m/s), Nm/(rad/s)], `rho6>=0` [N, Nm],
`phi6>=0` [m/s, rad/s]. The exported numerical function assumes valid inputs;
the caller must validate gains, physical inertia and finite measurements.
This export does not select a controller or change active gains in SimLab.

### Optional artificial surface support

`Base(use_jit=False, artificial_surface_support=True)` restores the historical
simulation aid: zero buoyancy at negative NED depth, weight-balanced buoyancy at
depth zero, and a linear ramp from weight to full buoyancy over the next 3 m.
This is not a physical partial-immersion model; it has a depth-zero discontinuity.
The default is `False` (fully submerged).

Use `export_vehicle(output, artificial_surface_support=True)` in the existing
code-generation notebook to apply the same choice to plant, controller, EKF,
sensor and identification exports. The manifest records the choice. The legacy
`g_id` matrix has no depth input and remains the fully submerged matrix; use
`g_depth_id` for the selected restoring law. Regenerate/install a coordinated
bundle; this is an export-time option, not a live simulator menu toggle.
The standalone identification exporter was removed; the canonical notebook
already emits the same native regressor.
