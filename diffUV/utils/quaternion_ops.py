from casadi import cos, SX, sin, tan, skew, inv, vertcat, atan2, asin, sqrt
from diffUV.utils.operators import rot_diff, sympy2casadi

def linear_vel_Rq(uq):
    eta = uq[0]
    eps1 = uq[1]
    eps2 = uq[2]
    eps3 = uq[3]

    Rq = SX(3, 3)
    Rq[0,0] = 1 - 2*(eps2**2 + eps3**2)
    Rq[0,1] = 2*(eps1*eps2 - eps3*eta)
    Rq[0,2] = 2*(eps1*eps3 + eps2*eta)
    Rq[1,0] = 2*(eps1*eps2 + eps3*eta)
    Rq[1,1] = 1 - 2*(eps1**2 + eps3**2)
    Rq[1,2] = 2*(eps2*eps3 - eps1*eta)
    Rq[2,0] = 2*(eps1*eps3 - eps2*eta)
    Rq[2,1] = 2*(eps2*eps3 + eps1*eta)
    Rq[2,2] = 1 - 2*(eps1**2 + eps2**2)
    return Rq

def angular_vel_Tq(uq):
    eta = uq[0]
    eps1 = uq[1]
    eps2 = uq[2]
    eps3 = uq[3]

    Tq = SX(4,3)
    Tq[0,0] = -eps1
    Tq[0,1] = -eps2
    Tq[0,2] = -eps3
    Tq[1,0] = eta
    Tq[1,1] = -eps3
    Tq[1,2] = eps2
    Tq[2,0] = eps3
    Tq[2,1] = eta
    Tq[2,2] = -eps1
    Tq[3,0] = -eps2
    Tq[3,1] = eps1
    Tq[3,2] = eta
    return 0.5*Tq

def Jq_kin(uq):
    Rq = linear_vel_Rq(uq)
    Tq = angular_vel_Tq(uq)
    J = SX.zeros(7, 6)
    J[:3,:3] = Rq
    J[3:,3:] = Tq
    return J, Rq, Tq

def dTq(uq, w_nb):
    p = w_nb[0]
    q = w_nb[1]
    r = w_nb[2]

    n = uq[0]
    e1 = uq[1]
    e2 = uq[2]
    e3 = uq[3]

    dn = -0.5*(e1*p + e2*q + e3*r)
    de1 = 0.5*(n*p - e3*q + e2*r)
    de2 = 0.5*(e3*p + n*q - e1*r)
    de3 = 0.5*(-e2*p + e1*q + n*r)

    dTq = SX(4, 3)
    dTq[0,0] = -de1
    dTq[0,1] = -de2
    dTq[0,2] = -de3
    dTq[1,0] = dn
    dTq[1,1] = -de3
    dTq[1,2] = de2
    dTq[2,0] = de3
    dTq[2,1] = dn
    dTq[2,2] = -de1
    dTq[3,0] = -de2
    dTq[3,1] = de1
    dTq[3,2] = dn
    return dTq 

def Jq_dot(uq, w_nb):
    _,Rq,Tq = Jq_kin(uq)
    dRq = rot_diff(Rq, w_nb)
    dJq = SX.zeros(7, 6)
    _dTq = dTq(uq, w_nb)
    dJq[:3,:3] = dRq
    dJq[3:,3:] = _dTq
    return dJq, dRq, _dTq

def inv_Jq_kin(uq):
    Rq = linear_vel_Rq(uq)
    Tq = angular_vel_Tq(uq)
    J = SX.zeros(6, 7)
    J[:3,:3] = Rq.T
    J[3:,3:] = 4*Tq.T
    return J, Rq.T, 4*Tq.T


def euler2q(eul):
    phi, thet, psi = eul[0],eul[1],eul[2]
    q = SX(4, 1)
    q[0,0] = cos(0.5*psi)*cos(0.5*thet)*cos(0.5*phi) + sin(0.5*psi)*sin(0.5*thet)*sin(0.5*phi)
    q[1,0] = cos(0.5*psi)*cos(0.5*thet)*sin(0.5*phi) - sin(0.5*psi)*sin(0.5*thet)*cos(0.5*phi)
    q[2,0] = sin(0.5*psi)*cos(0.5*thet)*sin(0.5*phi) + cos(0.5*psi)*sin(0.5*thet)*cos(0.5*phi)
    q[3,0] = sin(0.5*psi)*cos(0.5*thet)*cos(0.5*phi) - cos(0.5*psi)*sin(0.5*thet)*sin(0.5*phi)
    return q

def q2euler(uq):
    norm_uq = uq/sqrt(uq.T@uq)
    eul_v = SX(3, 1)
    Rq = linear_vel_Rq(norm_uq)
    eul_v[0] = atan2(Rq[2,1], Rq[2,2])
    eul_v[1] = -asin(Rq[2,0])
    eul_v[2] = atan2(Rq[1,0], Rq[0,0])
    return eul_v