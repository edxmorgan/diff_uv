"""Simulation measurement boundary for interval-regressor identification.

Only public sensor/effort fields enter the fit. This does not apply parameters.
Zero-current, uncoupled vehicle in effort mode is an explicit deployment
precondition; unknown disturbances are not silently subtracted using EKF data.
"""
from collections import deque
import numpy as np
from .identification_data import IntegralVehicleRegressor


class LiveVehicleWindow:
    def __init__(self, *, max_gap, horizon=4200):
        if not isinstance(horizon,int) or horizon<5:
            raise ValueError('horizon must contain at least five samples')
        if not np.isfinite(max_gap) or max_gap<=0:
            raise ValueError('max_gap must be positive')
        self.max_gap=float(max_gap)
        self.rows=deque(maxlen=horizon)
        self.regressor=IntegralVehicleRegressor()
        self.identity=None
        self.epoch=0
        self.previous=None;self.older=None
        self.emitted=0

    def reset(self):
        self.rows.clear();self.previous=None;self.older=None;self.identity=None
        self.epoch+=1;self.emitted=0

    def push(self, state):
        """Deduplicate publications; never interpolate missing physics steps."""
        try:
            generation,step,time,depth,period=[float(state[k]) for k in
                ('sim_generation','sim_step','sim_time','depth_from_pressure2','sim_period')]
            angles=np.array([state['imu_'+k] for k in ('roll','pitch','yaw')],float)*[1,-1,-1]
            velocity=np.r_[np.array([state['dvl_speed_'+k] for k in 'xyz'],float),
                np.array([state['imu_angular_vel_'+k] for k in 'xyz'],float)*[1,-1,-1]]
            effort=np.array([state[k+'.'+a] for k in ('force','torque') for a in 'xyz'],float)
            if (not np.isfinite([generation,step,time,depth,period]).all()
                    or not all(np.isfinite(x).all() for x in (angles,velocity,effort))
                    or generation<0 or step<0 or generation!=int(generation) or step!=int(step)
                    or period<=0 or period>self.max_gap):
                raise ValueError('Invalid measurement values')
        except (KeyError,TypeError,ValueError):
            self.reset()
            raise ValueError('Missing/invalid synchronized simulation measurements') from None
        if self.identity is not None:
            gen,previous,stamp=self.identity
            if generation==gen and step==previous and time==stamp:
                return False
            if (generation!=gen or step!=previous+1 or time<=stamp
                    or time-stamp>self.max_gap):
                self.reset()
        self.identity=(generation,step,time)
        bias=np.asarray(self.regressor.bias(depth,angles,velocity))
        previous,older=self.previous,self.older
        self.older=previous
        self.previous=(velocity.copy(),bias,time,period,effort.copy())
        if older is None:
            return False
        # Same generated quadrature contract as the native LiveWindow.
        old_velocity,old_bias,start,_,_=older
        _,mid_bias,_,h0,mid_effort=previous
        weights=np.asarray(self.regressor.weights(h0,period,
            np.array_equal(mid_effort,effort))).ravel()
        Y=np.asarray(self.regressor.inertia(velocity-old_velocity))/(h0+period)
        Y+=weights[0]*old_bias+weights[1]*mid_bias+weights[2]*bias
        average_effort=(h0*mid_effort+period*effort)/(h0+period)
        self.rows.append((time,Y,average_effort,start))
        self.emitted+=1
        return True

    def snapshot(self):
        if len(self.rows)!=self.rows.maxlen:
            return None
        return (self.epoch,self.rows[0][3],self.rows[-1][0],
                np.vstack([r[1] for r in self.rows]),np.concatenate([r[2] for r in self.rows]))
