from casadi import cos, SX, sin, tan, skew, inv

def linear_vel_R(phi, thet, psi):
    R = SX(3, 3)
    R[0,0] = cos(psi)*cos(thet)
    R[0,1] = -sin(psi)*cos(phi) + cos(psi)*sin(thet)*sin(phi)
    R[0,2] = sin(psi)*cos(phi) + cos(psi)*cos(phi)*sin(thet)
    R[1,0] = sin(psi)*cos(thet)
    R[1,1] = cos(psi)*cos(phi) + sin(phi)*sin(thet)*sin(psi)
    R[1,2] = -cos(psi)*cos(phi) + sin(thet)*sin(psi)*sin(phi)
    R[2,0] = -sin(thet)
    R[2,1] = cos(thet)*sin(phi)
    R[2,2] = cos(thet)*cos(phi)
    return R

def inverse_linear_vel_R(phi, thet, psi):
    R = linear_vel_R(phi, thet, psi)
    return R.T

def angular_vel_T(phi, thet):
    #T(nb) is undefined for a pitch(psi) angle of θ = ± 90◦
    T = SX.eye(3)
    T[0,1] = sin(phi)*tan(thet)
    T[0,2] = cos(phi)*tan(thet)
    T[1,1] = cos(phi)
    T[1,2] = -sin(phi)
    T[2,1] = sin(phi)/cos(thet)
    T[2,2] = cos(phi)/cos(thet)
    return T

def inverse_angular_vel_T(phi, thet):
    T_1 = SX.eye(3)
    T_1[0,2] = -sin(thet)
    T_1[1,1] = cos(phi)
    T_1[1,2] = cos(thet)*sin(phi)
    T_1[2,1] = -sin(phi)
    T_1[2,2] = cos(thet)*cos(phi)
    return T_1

def T_diff(R_n,w_b):
    S = skew(w_b)
    dR_n = R_n@S
    return dR_n


def J_kin(phi, thet, psi):
    R = linear_vel_R(phi, thet, psi)
    T = angular_vel_T(phi, thet)
    J = SX.zeros(6, 6)
    J[:3,:3] = R
    J[3:,3:] = T
    return J,R,T