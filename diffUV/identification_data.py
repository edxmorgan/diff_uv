"""Shared integral regressor and completed-step input contract for identification."""
import numpy as np
import casadi as ca
from .base import Base
from .utils.symbols import z,eul,x_nb,dx_nb,v_c


def completed_step_impulse(times, effort):
    """Integrate recorded ZOH input: row k acted on (t[k-1], t[k]].

    Row zero belongs to the preceding interval and is not integrated.
    Callers must supply consecutive, synchronized completed-step records.
    """
    times = np.asarray(times, float)
    effort = np.asarray(effort, float)
    if (times.ndim != 1 or len(times) < 2 or effort.shape != (len(times), 6)
            or not np.isfinite(times).all() or not np.isfinite(effort).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid completed-step measurements')
    return (np.diff(times)[:, None] * effort[1:]).sum(axis=0)


PARAMETERS = (
    'm_x_du m_y_dv m_z_dw mz_g_x_dq mz_g_y_dp mz_g_k_dv mz_g_m_du '
    'i_x_k_dp i_y_m_dq i_z_n_dr weight buoyancy '
    'x_g_weight_minus_x_b_buoyancy y_g_weight_minus_y_b_buoyancy '
    'z_g_weight_minus_z_b_buoyancy x_u y_v z_w k_p m_q n_r '
    'x_uu y_vv z_ww k_pp m_qq n_rr'
).split()


class IntegralVehicleRegressor:
    """Zero-current integral form of the shared dynamics; no acceleration input.

    M (v_end-v_start) + integral(Cv+Dv+g) = integral(tau+d).
    Three-endpoint quadratic quadrature for smooth state-dependent terms;
    trapezoid fallback at input switches or strongly unequal sample periods; completed-step ZOH wrench
    (row k acted over the preceding interval). Measured velocity/attitude enter Y, so this does
    NOT make an errors-in-variables problem an exact Gaussian likelihood.
    Window endpoint noise and shared endpoints must be accounted for separately.
    """
    def __init__(self, *, artificial_surface_support=False):
        model=Base(use_jit=False, artificial_surface_support=artificial_surface_support)
        p=model.lumped_params
        M=model.build_sys_regressor()[5]
        self.inertia=ca.Function('integral_inertia',[dx_nb],[ca.jacobian(M@dx_nb,p)])
        Y=ca.substitute(model.identification_regressor(),dx_nb,ca.DM.zeros(6))
        Y=ca.substitute(Y,v_c,ca.DM.zeros(6))
        self.bias=ca.Function('integral_bias',[z,eul,x_nb],[Y])
        # Average over two completed intervals. Keep positive weights under jitter;
        # do not interpolate a smooth polynomial across a known ZOH input switch.
        h0, h1 = ca.SX.sym('h0'), ca.SX.sym('h1')
        smooth = ca.SX.sym('smooth')
        total = h0+h1
        quadratic = ca.vertcat((2*h0-h1)/(6*h0), total**2/(6*h0*h1),
                               (2*h1-h0)/(6*h1))
        trapezoid = ca.vertcat(h0, total, h1)/(2*total)
        use_quadratic = ca.logic_and(smooth, ca.logic_and(h1 <= 2*h0, h0 <= 2*h1))
        self.weights = ca.Function('integral_bias_weights', [h0,h1,smooth],
                                  [ca.if_else(use_quadratic, quadratic, trapezoid)])

    def window(self,times,depth,attitude,velocity,wrench):
        times=np.asarray(times,float); depth=np.asarray(depth,float)
        attitude=np.asarray(attitude,float); velocity=np.asarray(velocity,float); wrench=np.asarray(wrench,float)
        n=len(times)
        if n<2 or depth.shape!=(n,) or attitude.shape!=(n,3) or velocity.shape!=(n,6) or wrench.shape!=(n,6):
            raise ValueError('inconsistent integral window shapes')
        if not all(np.isfinite(a).all() for a in (times,depth,attitude,velocity,wrench)) or np.any(np.diff(times)<=0):
            raise ValueError('invalid integral window')
        bias=np.stack([np.asarray(self.bias(d,a,v)) for d,a,v in zip(depth,attitude,velocity)])
        dt=np.diff(times)
        Y=np.asarray(self.inertia(velocity[-1]-velocity[0]))
        for i in range(0, n-2, 2):
            weights=np.asarray(self.weights(dt[i],dt[i+1],
                np.array_equal(wrench[i+1],wrench[i+2]))).ravel()
            Y += (dt[i]+dt[i+1])*np.tensordot(weights,bias[i:i+3],axes=1)
        if (n-1)%2:
            Y += .5*dt[-1]*(bias[-2]+bias[-1])
        target=completed_step_impulse(times,wrench)
        return Y,target
