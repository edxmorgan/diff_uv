from casadi import cos, SX, sin, tan, skew, inv

def linear_vel_Rq(q):
    eta = q[0]
    eps1 = q[1]
    eps2 = q[2]
    eps3 = q[3]

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

def angular_vel_Tq(q):
    eta = q[0]
    eps1 = q[1]
    eps2 = q[2]
    eps3 = q[3]

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

def Jq_kin(q):
    Rq = linear_vel_Rq(q)
    Tq = angular_vel_Tq(q)
    J = SX.zeros(7, 6)
    J[:3,:3] = Rq
    J[3:,3:] = Tq
    return J, Rq, Tq

def inv_Jq_kin(q):
    Rq = linear_vel_Rq(q)
    Tq = angular_vel_Tq(q)
    J = SX.zeros(6, 7)
    J[:3,:3] = Rq.T
    J[3:,3:] = 4*Tq.T
    return J, Rq.T, 4*Tq.T