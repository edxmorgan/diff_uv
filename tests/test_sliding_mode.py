"""Source/export checks for the constant-M zero-current sliding controller."""
import casadi as ca
import numpy as np
from diffUV.base import Base
from diffUV.controller import Controller
from diffUV.utils.symbols import n,x_nb,nd,xb_d,des_acc_b,v_c
from diffUV.kinematics import Kinematics


def model():
    base=Base(use_jit=False);controller=Controller(base)
    gains=[ca.SX.sym(k,6) for k in ('lambda','damping','switching','width')]
    outputs=controller.sliding_mode(*gains)
    args=[controller.lumped_params,ca.vertcat(n,x_nb),nd,xb_d,des_acc_b]+gains
    f=ca.Function('smc_test',args,list(outputs))
    _,_,_,_,p,M,C,g,D=base.build_sys_regressor()
    bias=ca.substitute(C@x_nb+D@x_nb+base.surface_restoring_reg(),v_c,ca.SX.zeros(6))
    acceleration=ca.solve(M,outputs[0]-bias)
    J=Kinematics().J
    rate=ca.vertcat(J@x_nb,acceleration)
    Jd=ca.substitute(J,n,nd)
    sdot=(ca.jacobian(outputs[1],ca.vertcat(n,x_nb))@rate
          +ca.jacobian(outputs[1],nd)@Jd@xb_d
          +ca.jacobian(outputs[1],xb_d)@des_acc_b)
    energy_rate=ca.dot(outputs[1],M@sdot)
    check=ca.Function('smc_check',args,[sdot,energy_rate,M,bias])
    return f,check


def parameters():
    return np.array([37.2,22.2,66.2,3.389,.641,.641,3.389,1.396,.498,2.531,
        105,97.8,.827,.137,4.258,-73.64,-33.61,-89.41,-2.99,-1.58,-3.40,
        -.00015,-.00052,-98.55,-.031,-.128,-1.635])


def test_reaching_law_with_moving_reference():
    f,check=model();rng=np.random.default_rng(41)
    for width in (0.,.05):
        for _ in range(80):
            x=rng.uniform(-.6,.6,12);goal=rng.uniform(-.6,.6,6)
            args=[parameters(),x,goal,rng.uniform(-.3,.3,6),rng.uniform(-.2,.2,6),
                  np.ones(6)*1.5,np.ones(6)*2,np.ones(6)*3,np.ones(6)*width]
            u,s,e=f(*args);sdot,vdot,M,b=check(*args);s=np.array(s).ravel()
            sw=np.sign(s) if width==0 else np.clip(s/width,-1,1)
            expected=-2*s-3*sw
            np.testing.assert_allclose(np.array(M)@np.array(sdot).ravel(),expected,atol=2e-11)
            np.testing.assert_allclose(float(vdot),s@expected,atol=2e-11)
            assert float(vdot)<=1e-12


def test_equilibrium_and_yaw_representation():
    f,_=model();x=np.zeros(12);x[2]=2
    args=[parameters(),x,x[:6],np.zeros(6),np.zeros(6),np.ones(6),np.ones(6),np.ones(6),np.ones(6)*.05]
    u,s,e=f(*args);assert np.isfinite(u).all()
    np.testing.assert_allclose(s,0,atol=1e-14)
    args[2]=x[:6].copy();args[2][5]=2*np.pi
    np.testing.assert_allclose(f(*args)[0],u,atol=1e-12)


def test_static_closed_loop_with_matched_disturbance():
    from scipy.integrate import solve_ivp
    f,check=model();x=ca.SX.sym('x',12)
    J=ca.Function('J',[n],[Kinematics().J]);goal=np.array([.4,-.3,1.5,.15,-.1,.2])
    disturbance=np.array([.4,-.3,.2,.03,-.02,.02]);p=parameters()
    def rhs(t,state):
        args=[p,state,goal,np.zeros(6),np.zeros(6),np.ones(6),np.ones(6)*3,np.ones(6),np.ones(6)*.03]
        u,_,_=f(*args);_,_,M,b=check(*args)
        return np.r_[np.array(J(state[:6])@state[6:]).ravel(),np.linalg.solve(np.array(M),np.array(u-b).ravel()+disturbance)]
    initial=np.zeros(12);initial[2]=1
    sol=solve_ivp(rhs,(0,30),initial,rtol=1e-7,atol=1e-9,max_step=.05)
    assert sol.success
    assert np.linalg.norm(sol.y[:3,-1]-goal[:3])<.03
    assert np.linalg.norm(sol.y[3:6,-1]-goal[3:6])<.01
