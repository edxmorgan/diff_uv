"""Notebook-equivalent measurement filtering, with a fixed-lag streaming form.

Inputs are already unwrapped Euler angles and DVL velocity in the notebook's
NED/FRD convention. No model parameters or EKF outputs are used. SG smoothing
is sample-index based, exactly as in the notebook; timestamps drive derivatives.
"""
from collections import deque
import numpy as np
from scipy.signal import savgol_filter


def notebook_motion(times, attitude, velocity, *, window=35, order=2):
    t = np.asarray(times, float)
    a = np.asarray(attitude, float)
    v = np.asarray(velocity, float)
    if (not isinstance(window, int) or window < 3 or window % 2 != 1
            or not isinstance(order, int) or not 0 <= order < window):
        raise ValueError('Invalid smoothing configuration')
    if (t.ndim != 1 or len(t) < window or a.shape != (len(t), 3)
            or v.shape != a.shape or not all(np.isfinite(x).all() for x in (t,a,v))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Require finite synchronized samples and increasing times')
    rates = savgol_filter(np.gradient(a, t, axis=0), window, order, axis=0)
    phi, pitch = a[:,0], a[:,1]
    body = np.column_stack((v,
        rates[:,0]-np.sin(pitch)*rates[:,2],
        np.cos(phi)*rates[:,1]+np.cos(pitch)*np.sin(phi)*rates[:,2],
        -np.sin(phi)*rates[:,1]+np.cos(pitch)*np.cos(phi)*rates[:,2]))
    body = savgol_filter(body, window, order, axis=0)
    acceleration = savgol_filter(np.gradient(body,t,axis=0,edge_order=2),
                                 window,order,axis=0)
    return body, acceleration


class DelayedNotebookMotion:
    """Emit only samples with complete filter support, never edge extrapolation.

    Three centered SG stages and two derivatives require lag 3*(window//2)+2
    samples. Wrench and attitude are returned at the SAME delayed timestamp.
    max_gap is an explicit caller policy in seconds; a gap resets the buffer.
    Reset explicitly on experiment generation/frame changes.
    """
    def __init__(self, *, max_gap, window=35, order=2):
        if not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('max_gap must be positive')
        notebook_motion(np.arange(max(window,3)), np.zeros((max(window,3),3)),
                        np.zeros((max(window,3),3)), window=window, order=order)
        self.window, self.order, self.max_gap = window, order, max_gap
        self.lag = 3*(window//2)+2
        self.samples = deque(maxlen=2*self.lag+1)

    def reset(self):
        self.samples.clear()

    def push(self, time, attitude, velocity, wrench):
        row = np.concatenate(([time],np.asarray(attitude),np.asarray(velocity),np.asarray(wrench)))
        if row.shape != (13,) or not np.isfinite(row).all():
            raise ValueError('Expected finite timestamp, attitude(3), velocity(3), wrench(6)')
        if self.samples:
            dt = time-self.samples[-1][0]
            if dt <= 0:
                raise ValueError('Nonmonotonic timestamp')
            if dt > self.max_gap:
                self.reset()
        self.samples.append(row.copy())
        if len(self.samples) < self.samples.maxlen:
            return None
        data = np.asarray(self.samples)
        body, acc = notebook_motion(data[:,0],data[:,1:4],data[:,4:7],
                                   window=self.window,order=self.order)
        k = self.lag
        return (float(data[k,0]), data[k,1:4].copy(), body[k].copy(),
                acc[k].copy(), data[k,7:].copy())
