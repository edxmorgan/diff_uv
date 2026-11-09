"""Reproduce the original notebook without executing plotting/export cells.

No ROS calls, parameter application, posterior claims, or truth-based fitting.
"""
import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter
import casadi as ca
import numpy as np
import pandas as pd
from diffUV.base import Base
from diffUV.system_id import MarineVehicleEstimator
from diffUV.identification_filter import notebook_motion, DelayedNotebookMotion
from diffUV.convex_vehicle_identification import ConvexVehicleIdentifier
from diffUV.utils.symbols import eul,x_nb,dx_nb,v_c


def reproduce(csv, *, stream_max_gap=None):
    df = pd.read_csv(csv)
    dt = df.timestamp.shift(-1)-df.timestamp
    df = df.loc[dt.fillna(0) != 0].reset_index(drop=True)
    t = df.timestamp.to_numpy(float)
    attitude = df[['imu_roll_unwrap','imu_pitch_unwrap','imu_yaw_unwrap']].to_numpy()*[1,-1,-1]
    velocity = df[['dvl_speed_x','dvl_speed_y','dvl_speed_z']].to_numpy()
    body, acc = notebook_motion(t,attitude,velocity)
    wrench = df[['base_x_force','base_y_force','base_z_force',
                 'base_x_torque','base_y_torque','base_z_torque']].to_numpy()
    model = Base(use_jit=False)
    reg = ca.Function('notebook_Y',[eul,x_nb,dx_nb,v_c],[model.build_sys_regressor()[0]])
    Y = np.vstack([np.asarray(reg(a,v,d,np.zeros(6))) for a,v,d in zip(attitude,body,acc)])
    initial = np.array([80,5,2,0,0,0,0,.5,.5,.5,0,0,0,0,0,
                        -30,-30,-75,-2,-2,-2,0,0,0,0,0,0])
    estimator = MarineVehicleEstimator(6,27,len(t),initial,body)
    runs=[]
    for _ in range(3):
        started=perf_counter()
        theta,_,_,seconds=estimator.estimate_vehicle_physical_parameters(Y,wrench.ravel(),True)
        runs.append({'solver_seconds': seconds,'wall_seconds':perf_counter()-started})
        if estimator.prob.status != 'optimal' or not np.isfinite(theta).all():
            raise RuntimeError('Notebook solve failed')
    compressed=ConvexVehicleIdentifier(weight_bounds=(105,150))
    compressed_runs=[]
    for _ in range(3):
        candidate=compressed.fit(Y,wrench.ravel())
        compressed_runs.append({'solver_seconds':candidate.solver_seconds,'wall_seconds':candidate.wall_seconds})
    original_cost=float(np.linalg.norm(Y@theta-wrench.ravel())**2)
    compressed_cost=float(np.linalg.norm(Y@candidate.parameters-wrench.ravel())**2)
    streaming=None
    if stream_max_gap is not None:
        stream=DelayedNotebookMotion(max_gap=stream_max_gap)
        rows,targets=deque(maxlen=500),deque(maxlen=500)
        emitted=0;stream_runs=[];velocity_error=0.;acceleration_error=0.;lags=[]
        for i in range(len(t)):
            delayed=stream.push(t[i],attitude[i],velocity[i],wrench[i])
            if delayed is None:
                rows.clear();targets.clear()
                continue
            stamp,a,v,d,u=delayed
            k=i-stream.lag
            velocity_error=max(velocity_error,float(np.max(np.abs(v-body[k]))))
            acceleration_error=max(acceleration_error,float(np.max(np.abs(d-acc[k]))))
            lags.append(float(t[i]-stamp))
            rows.append(np.asarray(reg(a,v,d,np.zeros(6))));targets.append(u)
            emitted+=1
            if len(rows)==500 and emitted%50==0:
                fit=compressed.fit(np.vstack(rows),np.concatenate(targets))
                stream_runs.append({'solver_seconds':fit.solver_seconds,'wall_seconds':fit.wall_seconds,
                                    'rank':fit.equality_reduced_rank})
        streaming={'emitted':emitted,'rolling_samples':500,'solve_stride_samples':50,
                   'runs':stream_runs,'max_batch_velocity_difference':velocity_error,
                   'max_batch_acceleration_difference':acceleration_error,
                   'max_filter_delay_seconds':max(lags,default=None),
                   'scope':'incremental CSV replay only; candidates not applied or certified'}
    return {'source':str(csv),'samples':len(t),'parameters':theta.tolist(),'runs':runs,
            'streaming':streaming,
            'compressed_runs':compressed_runs,'compressed_parameters':candidate.parameters.tolist(),
            'relative_objective_difference':abs(compressed_cost-original_cost)/max(original_cost,1.),
            'training_wrench_rms':np.sqrt(np.mean((Y@theta-wrench.ravel()).reshape(-1,6)**2,axis=0)).tolist(),
            'applied_parameters':False,'scope':'notebook reproduction; not a recovery or online acceptance test'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=Path(__file__).parent/'sample_data/best_data/20250914_152115/20250914_152115_robot_real_.csv')
    parser.add_argument('--stream-max-gap',type=float,help='Enable incremental replay with explicit gap reset threshold (seconds)')
    args=parser.parse_args()
    print(json.dumps(reproduce(args.csv,stream_max_gap=args.stream_max_gap),indent=2))
