"""A perfectly tracked moving reference must satisfy the actual plant equation."""
import casadi as ca
import numpy as np
from diffUV.base import Base
from diffUV.controller import Controller
from diffUV.utils.symbols import n,x_nb,nd,xb_d,des_acc_b,v_c,Kp,Ki,Kd,sum_e_buffer,dt
from test_sliding_mode import parameters


def test_moving_reference_acceleration_matches_complete_plant():
    base=Base(use_jit=False);control=Controller(base)
    u,_=control.trajectorytracking_pid()
    _,_,_,_,p,M,C,g,D=base.build_sys_regressor()
    acceleration=ca.solve(M,u-C@(x_nb-v_c)-D@(x_nb-v_c)-base.surface_restoring_reg())
    check=ca.Function('invdyn_acceleration',[p,ca.vertcat(n,x_nb),nd,xb_d,des_acc_b,Kp,Ki,Kd,sum_e_buffer,dt,v_c],[acceleration])
    rng=np.random.default_rng(734)
    for _ in range(100):
        x=rng.uniform(-.4,.4,12);x[2]=2
        desired_acc=rng.uniform(-.3,.3,6)
        result=check(parameters(),x,x[:6],x[6:],desired_acc,np.ones(6)*3,np.zeros(6),np.ones(6)*5,np.zeros(6),.005,np.zeros(6))
        np.testing.assert_allclose(np.array(result).ravel(),desired_acc,atol=2e-12,rtol=2e-12)


def test_roll_pitch_gains_do_not_change_with_world_heading():
    from scipy.spatial.transform import Rotation
    control=Controller(Base(use_jit=False))
    f=ca.Function('invdyn_heading',[control.lumped_params,ca.vertcat(n,x_nb),nd,xb_d,des_acc_b,Kp,Ki,Kd,sum_e_buffer,dt,v_c],list(control.trajectorytracking_pid()))
    rng=np.random.default_rng(314)
    kp=np.array([3.,3.,3.,.5,5.,.4]);kd=np.array([5.,5.,5.,1.5,10.,1.5])
    for _ in range(30):
        x=rng.uniform(-.3,.3,12);x[2]=4
        desired=x[:6]+rng.uniform(-.15,.15,6)
        vel=rng.uniform(-.2,.2,6);acc=rng.uniform(-.2,.2,6)
        def evaluate(state,target):return np.array(f(parameters(),state,target,vel,acc,kp,np.zeros(6),kd,np.zeros(6),.005,np.zeros(6))[0]).ravel()
        baseline=evaluate(x,desired)
        for yaw in [.6,np.pi/2,np.pi,-1.3]:
            Q=Rotation.from_euler('z',yaw);rotated=x.copy();target=desired.copy()
            rotated[:3]=Q.apply(x[:3]);target[:3]=Q.apply(desired[:3])
            rotated[3:6]=(Q*Rotation.from_euler('xyz',x[3:6])).as_euler('xyz')
            target[3:6]=(Q*Rotation.from_euler('xyz',desired[3:6])).as_euler('xyz')
            np.testing.assert_allclose(evaluate(rotated,target),baseline,atol=3e-12,rtol=3e-12)


def test_pid_angular_gains_and_integral_transport_are_heading_invariant():
    from scipy.spatial.transform import Rotation
    control=Controller(Base(use_jit=False))
    outputs=control.position_pid_reg()
    f=ca.Function('pid_heading',[control.lumped_params,ca.vertcat(n,x_nb),nd,xb_d,Kp,Ki,Kd,sum_e_buffer,dt],list(outputs))
    rng=np.random.default_rng(88)
    kp=np.array([3,3,3,.5,5,.4]);ki=np.array([.1,.1,.1,.2,.3,.4]);kd=np.array([2,2,2,2,3,5])
    for _ in range(40):
        x=rng.uniform(-.4,.4,12);x[2]=4;target=x[:6]+rng.uniform(-.2,.2,6);vel=rng.uniform(-.2,.2,6);integral=rng.uniform(-.1,.1,6)
        baseline=np.array(f(parameters(),x,target,vel,kp,ki,kd,integral,.005)[0]).ravel()
        for yaw in [np.pi/2,-1.,np.pi]:
            Q=Rotation.from_euler('z',yaw);s=x.copy();t=target.copy()
            s[:3]=Q.apply(x[:3]);t[:3]=Q.apply(target[:3]);s[3:6]=(Q*Rotation.from_euler('xyz',x[3:6])).as_euler('xyz');t[3:6]=(Q*Rotation.from_euler('xyz',target[3:6])).as_euler('xyz')
            buffer=np.r_[Q.apply(integral[:3]),Q.apply(integral[3:])]
            np.testing.assert_allclose(np.array(f(parameters(),s,t,vel,kp,ki,kd,buffer,.005)[0]).ravel(),baseline,atol=2e-12)
