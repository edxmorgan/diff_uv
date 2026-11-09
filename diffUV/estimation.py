"""Nominal-dynamics disturbance EKF, independent of simulator ground truth.

State: NED pose, body velocity, additive body wrench (18 coordinates).
Measurements: NED depth, roll/pitch/yaw, body-frame DVL velocity (legacy 7);
nonlinear sensor correction also uses body gyro and specific force (13).
Caller supplies synchronized measurements, preceding applied wrench estimate,
fixed nominal parameters and discrete process/measurement covariances.
"""
import math

import casadi as ca

from diffUV.base import Base
from diffUV.simulator import Simulator
from diffUV.utils.symbols import n, x_nb, tau_b, f_ext, v_c


def disturbance_ekf_covariance_prediction(prediction=None):
    """Predict mean and covariance from continuous process-noise intensity.

    ``Qc`` is a continuous covariance density, not a per-step covariance.
    Noise injected partway through the interval also changes subsequent motion:

      Qd = integral(Phi(dt,s) @ Qc @ Phi(dt,s).T, s=0..dt)
      Phi(dt,s) = A(dt) @ inverse(A(s)).

    A(s) is the existing RK mean predictor's analytic state transition along the
    same nominal trajectory, with actuator input and mean disturbance held as in
    that predictor. Three-point Gauss-Legendre quadrature preserves positive
    semidefiniteness without discarding navigation/wrench cross-covariance.
    This is numerical EKF covariance propagation, not exact nonlinear diffusion.
    Callers should subdivide intervals longer than 0.02 s, propagating both mean
    and covariance, and validate convergence for their dynamics/rate envelope.

    The optional prediction argument shares the graph with the existing export;
    legacy functions still accept *discrete* Qd and retain their original ABI.
    The caller validates finite inputs, nonnegative dt and symmetric PSD Qc/P.
    """
    if prediction is None:
        prediction, _ = disturbance_ekf_functions()
    nx, nu, nparams = (prediction.numel_in(i) for i in range(3))
    x = ca.MX.sym('x', nx)
    P = ca.MX.sym('P', nx, nx)
    u = ca.MX.sym('u_applied', nu)
    p = ca.MX.sym('p_nominal', nparams)
    dt = ca.MX.sym('dt')
    Qc = ca.MX.sym('Qc', nx, nx)
    predicted, A = prediction(x, u, p, dt)
    Qd = ca.MX.zeros(nx, nx)
    offset = math.sqrt(15.0) / 10.0
    for fraction, weight in ((0.5-offset, 5.0/18.0),
                             (0.5, 8.0/18.0),
                             (0.5+offset, 5.0/18.0)):
        _, A_s = prediction(x, u, p, dt*fraction)
        # Avoid forming an inverse. A(s) maps initial errors to s; undo that
        # map before propagating noise introduced at s to the endpoint.
        transition = ca.solve(A_s.T, A.T).T
        Qd += dt * weight * transition @ Qc @ transition.T
    Qd = (Qd + Qd.T) / 2
    prior_P = A @ P @ A.T + Qd
    return ca.Function('disturbance_ekf_predict_covariance',
        [x, P, u, p, dt, Qc],
        [predicted, (prior_P + prior_P.T)/2, A, Qd],
        ['x', 'P', 'u_applied', 'p_nominal', 'dt', 'Qc'],
        ['x_pred', 'P_pred', 'A', 'Qd'])



def _disturbance_ekf_smoothing_moment(full_covariance):
    """One backward RTS step and marginal disturbance-increment moments.

    Inputs describe two adjacent filter steps in the same NED/FRD coordinates.
    ``A_next`` maps errors from the filtered state to the next prior; the next
    smoothed state uses only observations in a completed, past window. The
    caller must discard windows spanning resets or invalid observations.

    With G = Pf A' Pprior^-1, the smoothed pair cross-covariance is
    Cov(x_k, x_{k+1} | window) = G Ps_next. Consequently the returned six
    values are E[(d_{k+1}-d_k)^2 | window], including posterior uncertainty,
    rather than squared differences of point estimates alone. They are raw
    second moments, not noise densities: the caller normalizes each interval
    by its dt before averaging and handles roundoff in the accumulated result.

    This supports causal marginal increment covariance matching. It is not
    exact full-state EM for integrated process noise: navigation/wrench Qd
    cross-terms and nonlinear dynamics also depend on the noise intensity.
    Smoothing is for learning from past data, not delayed operational output.
    Euler residuals use the nearest branch; the output remains on the filtered
    state's branch. This is a local EKF smoother, not a manifold-global solver.
    P_next_prior must be positive definite; no hidden jitter changes the model.
    """
    xf = ca.MX.sym('x_filtered', 18)
    Pf = ca.MX.sym('P_filtered', 18, 18)
    xp = ca.MX.sym('x_next_prior', 18)
    Pp = ca.MX.sym('P_next_prior', 18, 18)
    A = ca.MX.sym('A_next', 18, 18)
    xn = ca.MX.sym('x_next_smoothed', 18)
    Pn = ca.MX.sym('P_next_smoothed', 18, 18)
    G = ca.solve(Pp.T, A @ Pf.T).T
    delta = xn - xp
    for i in (3, 4, 5):
        delta[i] = ca.atan2(ca.sin(delta[i]), ca.cos(delta[i]))
    xs = xf + G @ delta
    Ps = Pf + G @ (Pn - Pp) @ G.T
    Ps = (Ps + Ps.T) / 2
    cross = G @ Pn
    increment = xn[12:18] - xs[12:18]
    full_moment = (increment @ increment.T + Ps[12:18, 12:18]
                   + Pn[12:18, 12:18] - cross[12:18, 12:18]
                   - cross[12:18, 12:18].T)
    full_moment = (full_moment + full_moment.T) / 2
    moments = full_moment if full_covariance else ca.diag(full_moment)
    name = ('disturbance_ekf_smoothing_full_moment' if full_covariance
            else 'disturbance_ekf_smoothing_moment')
    return ca.Function(name,
        [xf, Pf, xp, Pp, A, xn, Pn], [xs, Ps, moments],
        ['x_filtered', 'P_filtered', 'x_next_prior', 'P_next_prior',
         'A_next', 'x_next_smoothed', 'P_next_smoothed'],
        ['x_smoothed', 'P_smoothed', 'disturbance_increment_second_moment'])


def disturbance_ekf_smoothing_moment():
    """RTS step and six marginal increment moments; existing export ABI."""
    return _disturbance_ekf_smoothing_moment(False)


def disturbance_ekf_smoothing_full_moment():
    """RTS step and full 6x6 disturbance-increment second moment.

    E[delta_d delta_d'] includes the cross-axis posterior terms. It is symmetric
    without a hidden PSD projection. The caller must reject significant negative
    eigenvalues and may remove numerical roundoff only with a documented policy.
    This is still marginal moment matching, not exact full-state integrated-Q EM.
    """
    return _disturbance_ekf_smoothing_moment(True)


def disturbance_ekf_correction():
    """Measurement-only update; binary mask excludes absent observations exactly.

    Disturbance mean is frozen unless all measurement groups are available.
    Joseph covariance uses the same restricted gain (Schmidt-style update).
    Caller must replace nonfinite absent values before calling this function.
    """
    x = ca.MX.sym('x', 18)
    P = ca.MX.sym('P', 18, 18)
    y = ca.MX.sym('y', 7)
    R = ca.MX.sym('R', 7, 7)
    mask = ca.MX.sym('mask', 7)
    allow_disturbance = ca.MX.sym('allow_disturbance')
    D = ca.diag(mask)
    H = D @ ca.DM.eye(18)[2:9, :]
    innovation = y - x[2:9]
    for i in (1, 2, 3):
        innovation[i] = ca.atan2(ca.sin(innovation[i]), ca.cos(innovation[i]))
    innovation = D @ innovation
    covariance = D @ R @ D + ca.diag(1-mask)
    S = H @ P @ H.T + covariance
    K = ca.solve(S, H @ P).T
    K = ca.diag(ca.vertcat(ca.DM.ones(12), ca.repmat(allow_disturbance, 6, 1))) @ K
    residual = ca.MX.eye(18) - K @ H
    next_P = residual @ P @ residual.T + K @ covariance @ K.T
    return ca.Function('disturbance_ekf_correct', [x, P, y, R, mask, allow_disturbance],
        [x + K @ innovation, (next_P + next_P.T)/2])



def vehicle_sensor_model(*, artificial_surface_support=False):
    """Delivered depth/attitude/DVL/gyro/specific force in canonical NED/FRD.

    Accelerometers measure specific force, NOT body velocity derivatives.
    With R mapping body FRD to NED and an IMU at fixed body offset r:
      f = dv/dt + omega x v + domega/dt x r
          + omega x (omega x r) - R.T @ [0,0,g].
    Input u is the applied actuator wrench at the observation time. State d is
    the EKF's unknown external wrench, not a measured/input disturbance.
    Raw IMU readings must be rotated to body axes and bias calibrated by caller.
    """
    model = Base(use_jit=False, artificial_surface_support=artificial_surface_support)
    accel, parameters = model.body_forward_dynamics_reg()
    a = ca.Function('sensor_acceleration',
        [ca.vertcat(n, x_nb), tau_b, ca.vertcat(parameters, v_c), f_ext], [accel])
    rotation = ca.Function('sensor_rotation', [ca.vertcat(n, x_nb)], [model.R])
    x = ca.MX.sym('x', 18)
    u = ca.MX.sym('u_applied', 6)
    p = ca.MX.sym('p_nominal', 33)
    offset = ca.MX.sym('imu_offset_body', 3)
    gravity = ca.MX.sym('gravity')
    acceleration = a(x[:12], u, p, x[12:18])
    omega = x[9:12]
    specific = (acceleration[:3] + ca.cross(omega, x[6:9])
                + ca.cross(acceleration[3:], offset)
                + ca.cross(omega, ca.cross(omega, offset))
                - rotation(x[:12]).T @ ca.vertcat(0, 0, gravity))
    measurement = ca.vertcat(x[2:12], specific)
    return ca.Function('vehicle_sensor_model', [x, u, p, offset, gravity],
        [measurement, ca.jacobian(measurement, x), acceleration],
        ['x', 'u_applied', 'p_nominal', 'imu_offset_body', 'gravity'], ['y', 'H', 'acceleration'])


def disturbance_ekf_sensor_correction(*, artificial_surface_support=False):
    """Masked nonlinear correction using actual gyro/accelerometer observations.

    Order: depth, RPY, DVL velocity, body gyro, body specific force (13).
    Missing raw IMU channels can be masked without changing the legacy seven
    measurement model. Joseph form and the disturbance-availability gate match
    disturbance_ekf_correction. No numerical differentiation of measurements.
    """
    sensor = vehicle_sensor_model(artificial_surface_support=artificial_surface_support)
    x = ca.MX.sym('x', 18)
    P = ca.MX.sym('P', 18, 18)
    u = ca.MX.sym('u_applied', 6)
    p = ca.MX.sym('p_nominal', 33)
    y = ca.MX.sym('y', 13)
    R = ca.MX.sym('R', 13, 13)
    mask = ca.MX.sym('mask', 13)
    allow_disturbance = ca.MX.sym('allow_disturbance')
    offset = ca.MX.sym('imu_offset_body', 3)
    gravity = ca.MX.sym('gravity')
    expected, jacobian, _ = sensor(x, u, p, offset, gravity)
    D = ca.diag(mask)
    H = D @ jacobian
    innovation = y - expected
    for i in (1, 2, 3):
        innovation[i] = ca.atan2(ca.sin(innovation[i]), ca.cos(innovation[i]))
    innovation = D @ innovation
    covariance = D @ R @ D + ca.diag(1-mask)
    S = H @ P @ H.T + covariance
    K = ca.solve(S, H @ P).T
    K = ca.diag(ca.vertcat(ca.DM.ones(12), ca.repmat(allow_disturbance, 6, 1))) @ K
    residual = ca.MX.eye(18) - K @ H
    next_P = residual @ P @ residual.T + K @ covariance @ K.T
    return ca.Function('disturbance_ekf_sensor_correct',
        [x, P, u, p, y, R, mask, allow_disturbance, offset, gravity],
        [x + K @ innovation, (next_P + next_P.T)/2, innovation, S])


def disturbance_process_model():
    """Transport a load-reference random walk into the EKF's body-wrench state.

    B maps [force, torque] at the chosen load origin/axes to vehicle body
    generalized wrench. For a tool it is the vehicle block of J.T times the
    frame rotation. B_previous/B_current must be finite nonsingular 6x6 maps
    from consecutive geometry observations, never derived from injected loads.

    The EKF state remains [pose, body velocity, body wrench]. On a geometry
    change, B_current @ solve(B_previous, d) preserves the estimated wrench in
    load coordinates. Transform the entire covariance, including cross terms.
    At startup/reset use B_previous=B_current (no fictitious load change).

    Intensities are variance per second: wrench entries have units N^2/s or
    (N m)^2/s in load coordinates; navigation entries follow their state units.
    They must be finite and nonnegative; dt must be finite and positive. An OU
    prior with stationary sigma and correlation time tau has short-lag diffusion
    2*sigma^2/tau. This function uses that diffusion only: its mean is a random
    walk, not an OU mean-reverting model and not an injection-waveform tracker.

    Caller supplies geometry for the modeled interval. Do not silently select a
    tool-load prior for arbitrary loads or reuse stale geometry during motion.
    Outputs feed the existing prediction/update, which still receives measured
    motion, actuator input and nominal parameters, never disturbance truth.
    """
    x = ca.MX.sym('x', 18)
    P = ca.MX.sym('P', 18, 18)
    previous = ca.MX.sym('B_previous', 6, 6)
    current = ca.MX.sym('B_current', 6, 6)
    state_intensity = ca.MX.sym('state_intensity', 12)
    wrench_intensity = ca.MX.sym('wrench_intensity', 6)
    dt = ca.MX.sym('dt')
    transport = ca.diagcat(ca.MX.eye(12), current @ ca.solve(previous, ca.MX.eye(6)))
    covariance = transport @ P @ transport.T
    Q_d = dt * ca.diagcat(ca.diag(state_intensity),
                         current @ ca.diag(wrench_intensity) @ current.T)
    return ca.Function('disturbance_process_model',
        [x, P, previous, current, state_intensity, wrench_intensity, dt],
        [transport @ x, (covariance + covariance.T)/2, (Q_d + Q_d.T)/2],
        ['x', 'P', 'B_previous', 'B_current', 'state_intensity', 'wrench_intensity', 'dt'],
        ['x_transported', 'P_transported', 'Q_d'])


def disturbance_ekf_functions(*, artificial_surface_support=False):
    """Build code-generatable prediction/update functions with analytic AD.

    Reuses Simulator.model_reg's discretization; no independent RK settings.
    Q_d is a per-step covariance (not a continuous spectral density).
    d_next=d implements a random-walk mean; Q_d specifies its uncertainty.
    This Euler-angle model excludes pitch singularities. Defaults to full immersion. The optional artificial surface support must
    match the plant/sensor model; its depth-zero discontinuity is not smooth.
    """
    model = Base(use_jit=False, artificial_surface_support=artificial_surface_support)
    xn, xs, us, ps, ts, ds = Simulator(model).model_reg('euler')
    step = ca.Function('nominal_vehicle_step', [xs, us, ps, ts, ds], [xn])
    accel, lumped = model.body_forward_dynamics_reg()
    acceleration = ca.Function('nominal_vehicle_acceleration',
        [ca.vertcat(n, x_nb), tau_b, ca.vertcat(lumped, v_c), f_ext], [accel])

    x = ca.MX.sym('x', 18)
    u = ca.MX.sym('u_applied', 6)
    p = ca.MX.sym('p_nominal', ps.numel())
    dt = ca.MX.sym('dt')
    predicted = ca.vertcat(step(x[:12], u, p, dt, x[12:]), x[12:])
    A = ca.jacobian(predicted, x)
    prediction = ca.Function('disturbance_ekf_predict', [x, u, p, dt],
        [predicted, A], ['x', 'u_applied', 'p_nominal', 'dt'], ['x_pred', 'A'])

    P = ca.MX.sym('P', 18, 18)
    Q = ca.MX.sym('Q_d', 18, 18)
    R = ca.MX.sym('R', 7, 7)
    y = ca.MX.sym('y', 7)
    # Measurements are already in canonical frames: no hidden sign conversion.
    h = x[2:9]
    H = ca.jacobian(h, x)
    innovation = y - predicted[2:9]
    for i in (1, 2, 3):
        innovation[i] = ca.atan2(ca.sin(innovation[i]), ca.cos(innovation[i]))
    P_pred = A @ P @ A.T + Q
    S = H @ P_pred @ H.T + R
    K = ca.solve(S, H @ P_pred).T
    updated = predicted + K @ innovation
    # Joseph form protects covariance symmetry/PSD against roundoff.
    residual_map = ca.MX.eye(18) - K @ H
    P_next = residual_map @ P_pred @ residual_map.T + K @ R @ K.T
    P_next = (P_next + P_next.T) / 2
    a = acceleration(updated[:12], u, p, updated[12:])
    update = ca.Function('disturbance_ekf_update', [x, P, u, p, dt, y, Q, R],
        [updated, P_next, innovation, S, a],
        ['x', 'P', 'u_applied', 'p_nominal', 'dt', 'y', 'Q_d', 'R'],
        ['x_next', 'P_next', 'innovation', 'innovation_covariance', 'acceleration'])
    return prediction, update
