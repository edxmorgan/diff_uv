"""Physical covariance and moving-reference invariants, independent of ROS."""
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
from diffUV.estimation import disturbance_process_model


def mapping(r, rotation):
    x,y,z=r
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.block([[rotation,np.zeros((3,3))],[skew@rotation,rotation]])


def test_body_reference_preserves_existing_random_walk_contract():
    f=disturbance_process_model();rng=np.random.default_rng(10)
    x=rng.normal(size=18);a=rng.normal(size=(18,18));p=a@a.T
    q=np.r_[[7e-9]*12,[7e-4]*6];h=.005
    got=f(x,p,np.eye(6),np.eye(6),q[:12],q[12:],h)
    np.testing.assert_allclose(got[0].full().ravel(),x,atol=1e-14)
    np.testing.assert_allclose(got[1],p,atol=1e-14)
    np.testing.assert_allclose(got[2],np.diag(q)*h,atol=1e-14)


def test_load_reference_transport_preserves_mean_and_full_covariance():
    f=disturbance_process_model();rng=np.random.default_rng(46)
    for _ in range(30):
        old=mapping(rng.normal(size=3),Rotation.random(random_state=rng).as_matrix())
        new=mapping(rng.normal(size=3),Rotation.random(random_state=rng).as_matrix())
        x=rng.normal(size=18);a=rng.normal(size=(18,18));p=a@a.T
        xt,pt,qd=[v.full() for v in f(x,p,old,new,np.ones(12)*1e-8,np.ones(6)*.1,.01)]
        np.testing.assert_allclose(np.linalg.solve(new,xt[12:,0]),np.linalg.solve(old,x[12:]),atol=1e-12)
        np.testing.assert_allclose(np.linalg.solve(new,pt[12:,:12]),np.linalg.solve(old,p[12:,:12]),atol=1e-12)
        old_cov=np.linalg.solve(old,np.linalg.solve(old,p[12:,12:]).T).T
        new_cov=np.linalg.solve(new,np.linalg.solve(new,pt[12:,12:]).T).T
        np.testing.assert_allclose(new_cov,old_cov,atol=1e-11)
        assert np.linalg.eigvalsh(pt).min()>-1e-10
        assert np.linalg.eigvalsh(qd).min()>-1e-10


def test_force_noise_creates_correlated_moment_not_independent_torque():
    f=disturbance_process_model();b=mapping([.5,0.,-.2],np.eye(3))
    intensity=np.array([1.,0.,0.,0.,0.,0.])
    q=f(np.zeros(18),np.eye(18),b,b,np.zeros(12),intensity,.1)[2].full()[12:,12:]
    np.testing.assert_allclose(q,.1*np.outer(b[:,0],b[:,0]),atol=1e-14)
    # The apparent moment at the vehicle origin is entirely explained by force.
    recovered=np.linalg.solve(b,np.linalg.solve(b,q).T).T
    np.testing.assert_allclose(recovered,.1*np.diag(intensity),atol=1e-14)
    assert q[0,4]!=0.


def test_covariance_adds_over_time_and_roundtrip_export(tmp_path):
    f=disturbance_process_model();b=mapping([.2,.3,.1],Rotation.from_euler('xyz',[.3,-.2,.5]).as_matrix())
    args=[np.arange(18)*.01,np.eye(18),b,b,np.ones(12)*1e-8,np.r_[[1.]*3,[.01]*3]]
    f.save(str(tmp_path/'process.casadi'));loaded=ca.Function.load(str(tmp_path/'process.casadi'))
    for a,c in zip(f(*args,.005),loaded(*args,.005)):np.testing.assert_array_equal(a,c)
    np.testing.assert_allclose(f(*args,.01)[2],2*f(*args,.005)[2],atol=1e-14)
