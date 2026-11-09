"""Continuous-noise propagation: analytic solutions and nonlinear convergence."""
import casadi as ca
import numpy as np
from scipy.linalg import expm
from diffUV.estimation import disturbance_ekf_covariance_prediction, disturbance_ekf_functions


def linear_prediction(damping=3., mass=7.):
    x=ca.MX.sym('x',2);u=ca.MX.sym('u',1);p=ca.MX.sym('p',1);dt=ca.MX.sym('dt')
    a=ca.exp(-damping*dt)
    A=ca.vertcat(ca.horzcat(a,(1-a)/(damping*mass)),ca.horzcat(0,1))
    return ca.Function('linear_prediction',[x,u,p,dt],[A@x,A])


def van_loan(F,Q,dt):
    n=F.shape[0]
    block=np.block([[F,Q],[np.zeros_like(F),-F.T]])
    exponential=expm(block*dt)
    return exponential[:n,n:]@exponential[:n,:n].T


def test_linear_covariance_matches_van_loan_and_includes_cross_terms():
    prediction=linear_prediction();f=disturbance_ekf_covariance_prediction(prediction)
    F=np.array([[-3.,1/7.],[0.,0.]])
    for Q in [np.diag([0.,.8]),np.array([[.02,.01],[.01,.8]])]:
        for dt in [0.,.001,.005,.02]:
            x,P,A,Qd=f([.2,.1],np.eye(2),[0.],[0.],dt,Q)
            expected=van_loan(F,Q,dt)
            np.testing.assert_allclose(Qd,expected,atol=1e-12,rtol=2e-9)
            np.testing.assert_allclose(P,np.asarray(A)@np.asarray(A).T+expected,atol=1e-12,rtol=2e-9)
            assert np.linalg.eigvalsh(np.asarray(Qd)).min()>=-1e-14
            if dt>0: assert float(Qd[0,1])>0 and float(Qd[0,0])>0


def test_long_interval_subdivision_matches_van_loan():
    f=disturbance_ekf_covariance_prediction(linear_prediction())
    x=np.array([.2,.1]);P=np.zeros((2,2));Q=np.diag([0.,.8])
    for _ in range(25):x,P,_,_=f(x,P,[0.],[0.],.02,Q)
    np.testing.assert_allclose(P,van_loan(np.array([[-3.,1/7.],[0.,0.]]),Q,.5),atol=1e-11,rtol=1e-8)


def test_nonlinear_noise_uses_endpoint_transition_not_initial_transition():
    # xdot=-x² has flow x/(1+x*t), A(t)=(1+x*t)^-2. The noise at s must
    # propagate to dt, not from 0 to s. Its exact covariance integral is quartic.
    x=ca.MX.sym('x');u=ca.MX.sym('u');p=ca.MX.sym('p');dt=ca.MX.sym('dt')
    flow=x/(1+x*dt)
    pred=ca.Function('nonlinear_prediction',[x,u,p,dt],[flow,ca.jacobian(flow,x)])
    f=disturbance_ekf_covariance_prediction(pred)
    _,_,_,Qd=f([1.],np.zeros((1,1)),[0.],[0.],.5,np.ones((1,1)))
    exact=(1.5**5-1)/(5*1.5**4)
    np.testing.assert_allclose(Qd,exact,atol=1e-14)
    wrong=(1-1/1.5**3)/3
    assert abs(float(Qd)-wrong)>.02


def test_vehicle_prediction_unchanged_psd_and_step_refinement():
    prediction,_=disturbance_ekf_functions()
    f=disturbance_ekf_covariance_prediction(prediction)
    p=np.array([37.2028553,22.1828075,66.1734807,3.38909801,.641362046,.641362034,3.389098,1.39646394,.498032205,2.53118738,105.,97.8296453,.827479545,.136822559,4.25841171,-73.6416666,-33.6082112,-89.4055107,-2.98736214,-1.57921531,-3.39766499,-.000147912104,-.000516373030,-98.5522538,-.0305907788,-.127877517,-1.63514832,0,0,0,0,0,0])
    x=np.r_[0.,0.,2.,.1,-.2,.3,.2,.3,.1,.05,.07,.09,.1,.2,.3,.01,.02,.03]
    u=np.array([1.,2.,3.,.1,.2,.3]);Q=np.diag([0.]*12+[.4,.5,.6,.01,.02,.03]);P=np.eye(18)*.001
    got=f(x,P,u,p,.02,Q);expected=prediction(x,u,p,.02)
    np.testing.assert_allclose(got[0],expected[0],atol=0,rtol=0)
    np.testing.assert_allclose(got[2],expected[1],atol=0,rtol=0)
    assert np.linalg.eigvalsh(np.asarray(got[3])).min()>-1e-14
    assert np.linalg.norm(np.asarray(got[3])[6:12,12:])>0
    errors=[]
    reference_x=x.copy();reference_P=P.copy()
    for _ in range(16):reference_x,reference_P,_,_=f(reference_x,reference_P,u,p,.02/16,Q)
    for steps in [1,2,4]:
        xs=x.copy();Ps=P.copy()
        for _ in range(steps):xs,Ps,_,_=f(xs,Ps,u,p,.02/steps,Q)
        np.testing.assert_allclose(xs,reference_x,atol=1e-11,rtol=1e-11)
        errors.append(np.linalg.norm(np.asarray(Ps-reference_P)))
    assert errors[0]<1e-8
    assert errors[-1]<max(errors[0]*.01,1e-14)
