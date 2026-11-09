"""The optional historical surface law must remain consistent across consumers."""
import casadi as ca
import numpy as np
import pytest
from diffUV.base import Base
from diffUV.controller import Controller
from diffUV.estimation import disturbance_ekf_functions, vehicle_sensor_model
from diffUV.identification_data import IntegralVehicleRegressor
from diffUV.utils.symbols import n, x_nb, tau_b, v_c, f_ext, dx_nb, eul
from test_sliding_mode import parameters


def test_surface_support_is_explicit_and_defaults_to_submerged():
    default = Base(use_jit=False)
    supported = Base(use_jit=False, artificial_surface_support=True)
    for depth, expected in [(-1., 0.), (0., 105.), (.75, 103.25), (1.5, 101.5), (3., 98.), (4., 98.)]:
        assert float(default.surface_interaction(depth, 105., 98., 3.)) == 98.
        assert float(supported.surface_interaction(depth, 105., 98., 3.)) == expected
    with pytest.raises(TypeError):
        Base(use_jit=False, artificial_surface_support='false')


@pytest.mark.parametrize('enabled', [False, True])
def test_plant_observer_controller_and_regressor_share_surface_law(enabled):
    base = Base(use_jit=False, artificial_surface_support=enabled)
    acc, p = base.body_forward_dynamics_reg()
    f = ca.Function('physical_acc', [ca.vertcat(n, x_nb), tau_b, ca.vertcat(p, v_c), f_ext], [acc])
    predictor, _ = disturbance_ekf_functions(artificial_surface_support=enabled)
    sx, su, sp, step = ca.MX.sym("x",18), ca.MX.sym("u",6), ca.MX.sym("p",33), ca.MX.sym("dt")
    rate = ca.Function("prediction_rate", [sx,su,sp,step],
        [ca.jacobian(predictor(sx,su,sp,step)[0][6:12], step)])
    sensor = vehicle_sensor_model(artificial_surface_support=enabled)
    control = Controller(base)
    restoring = ca.Function('restoring', [n, p], [control.gn_Y, base.surface_restoring_reg()])
    reg = IntegralVehicleRegressor(artificial_surface_support=enabled)
    M = ca.Function('mass', [p], [base.build_sys_regressor()[5]])
    pars = np.r_[parameters(), np.zeros(6)]
    x = np.zeros(18); x[3:6] = [.1, -.2, .3]; x[6:12] = [.1, .03, -.04, .02, -.01, .03]
    u = np.arange(6)*.1
    for depth in [-.5, 0., .5, 1.5, 3., 4.]:
        x[2] = depth
        acceleration = np.asarray(f(x[:12], u, pars, x[12:])).ravel()
        np.testing.assert_allclose(sensor(x, u, pars, np.zeros(3), 9.80665)[2], acceleration[:, None], atol=1e-12)
        np.testing.assert_allclose(rate(x,u,pars,0.), acceleration[:,None], atol=2e-12)
        gc, gp = restoring(x[:6], pars[:27])
        np.testing.assert_array_equal(gc, gp)
        required = np.asarray(M(pars[:27]))@acceleration + np.asarray(reg.bias(depth,x[3:6],x[6:12]))@pars[:27]
        np.testing.assert_allclose(required, u, atol=2e-12)
