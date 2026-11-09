"""Specific-force conventions and masked correction compatibility."""
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
from diffUV.estimation import vehicle_sensor_model, disturbance_ekf_sensor_correction, disturbance_ekf_correction

PARS=np.array([37.2028553,22.1828075,66.1734807,3.38909801,.641362046,.641362034,3.389098,1.39646394,.498032205,2.53118738,105.,97.8296453,.827479545,.136822559,4.25841171,-73.6416666,-33.6082112,-89.4055107,-2.98736214,-1.57921531,-3.39766499,-.000147912104,-.000516373030,-98.5522538,-.0305907788,-.127877517,-1.63514832,0,0,0,0,0,0])

def test_analytic_sensor_jacobian():
 f=vehicle_sensor_model();rng=np.random.default_rng(81)
 for _ in range(8):
  x=rng.normal(0,.15,18);x[2]=2.;u=rng.normal(size=6);r=np.array([.12,-.03,.06]);y,H,_=f(x,u,PARS,r,9.80665)
  numeric=np.column_stack([(np.asarray(f(x+np.eye(18)[i]*1e-6,u,PARS,r,9.80665)[0]).ravel()-np.asarray(f(x-np.eye(18)[i]*1e-6,u,PARS,r,9.80665)[0]).ravel())/2e-6 for i in range(18)])
  np.testing.assert_allclose(H,numeric,atol=2e-8,rtol=2e-7)

def test_gravity_and_lever_arm_terms():
 f=vehicle_sensor_model();x=np.zeros(18);x[2]=2.;u=np.zeros(6)
 # Gravity is independent of the dynamics' weight/buoyancy parameters.
 x[3:6]=[.3,-.4,.2];g=9.80665
 y0=np.asarray(f(x,u,PARS,np.zeros(3),0.)[0]).ravel();yg=np.asarray(f(x,u,PARS,np.zeros(3),g)[0]).ravel()
 R=Rotation.from_euler('xyz',x[3:6]).as_matrix()
 np.testing.assert_allclose(yg[10:]-y0[10:],-R.T@np.array([0,0,g]),atol=1e-12)
 # For alpha=0, adding an offset changes acceleration by the centripetal term.
 x[9:12]=[.2,-.3,.4];r=np.array([.1,.02,-.04])
 # Remove angular acceleration using an external wrench (only in the physical test).
 from diffUV.base import Base
 from diffUV.utils.symbols import n,x_nb,tau_b,v_c,f_ext
 model=Base(use_jit=False);a,p=model.body_forward_dynamics_reg()
 af=ca.Function('test_acceleration',[ca.vertcat(n,x_nb),tau_b,ca.vertcat(p,v_c),f_ext],[a])
 a0=np.asarray(af(x[:12],u,PARS,x[12:])).ravel()
 y1=np.asarray(f(x,u,PARS,r,g)[0]).ravel();y2=np.asarray(f(x,u,PARS,np.zeros(3),g)[0]).ravel()
 np.testing.assert_allclose(y1[10:]-y2[10:],np.cross(a0[3:],r)+np.cross(x[9:12],np.cross(x[9:12],r)),atol=1e-12)

def test_missing_raw_imu_is_exact_legacy_correction():
 new=disturbance_ekf_sensor_correction();old=disturbance_ekf_correction();rng=np.random.default_rng(17)
 x=rng.normal(0,.1,18);x[2]=2.;A=rng.normal(size=(18,18));P=A@A.T*.001
 y=rng.normal(0,.1,13);R=np.eye(13)*.01;mask=np.r_[1,1,1,1,0,1,1,np.zeros(6)]
 got=new(x,P,np.zeros(6),PARS,y,R,mask,1.,np.zeros(3),9.80665)
 expected=old(x,P,y[:7],R[:7,:7],mask[:7],1.)
 for a,b in zip(got[:2],expected):np.testing.assert_allclose(a,b,atol=1e-12)
 assert np.linalg.eigvalsh(np.asarray(got[1])).min()>-1e-12


def test_no_measurements_is_identity():
 f=disturbance_ekf_sensor_correction();x=np.zeros(18);x[2]=2.;P=np.eye(18)
 got=f(x,P,np.zeros(6),PARS,np.zeros(13),np.eye(13),np.zeros(13),1.,np.zeros(3),9.80665)
 np.testing.assert_array_equal(np.asarray(got[0]).ravel(),x);np.testing.assert_array_equal(got[1],P)


def test_submerged_dynamics_has_no_depth_switch():
 from diffUV.base import Base
 from diffUV.utils.symbols import n,x_nb,tau_b,v_c,f_ext
 model=Base(use_jit=False);acc,p=model.body_forward_dynamics_reg()
 f=ca.Function('submerged_acc',[ca.vertcat(n,x_nb),tau_b,ca.vertcat(p,v_c),f_ext],[acc])
 x=np.zeros(12);x[3:6]=[.2,-.3,.1]
 baseline=np.asarray(f(x,np.zeros(6),PARS,np.zeros(6)))
 sensor=vehicle_sensor_model()
 reference=np.asarray(sensor(np.r_[x,np.zeros(6)],np.zeros(6),PARS,np.zeros(3),9.80665)[0])
 for depth in [-1.,-1e-3,-1e-9,0.,1e-9,1e-3,1.,3.,10.]:
  x[2]=depth
  np.testing.assert_array_equal(f(x,np.zeros(6),PARS,np.zeros(6)),baseline)
  y,H,_=sensor(np.r_[x,np.zeros(6)],np.zeros(6),PARS,np.zeros(3),9.80665)
  np.testing.assert_array_equal(np.asarray(y)[1:],reference[1:])
  np.testing.assert_array_equal(np.asarray(H)[1:,2],np.zeros(12))
 # Negative buoyancy must produce downward acceleration without artificial support.
 x[:]=0.
 assert float(f(x,np.zeros(6),PARS,np.zeros(6))[2])>0.
 # Shared regressor predicts equilibrium effort at any coordinate depth.
 g=model.surface_restoring_reg()
 restoring=ca.Function('submerged_g',[n,ca.vertcat(p,v_c)],[g])
 for depth in [-.001,0.,.001,4.]:
  x[2]=depth
  np.testing.assert_allclose(f(x,restoring(x[:6],PARS),PARS,np.zeros(6)),0.,atol=1e-12)
