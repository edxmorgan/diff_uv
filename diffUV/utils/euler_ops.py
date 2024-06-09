from casadi import cos, SX, sin, tan, inv, vertcat
from diffUV.utils.operators import rot_diff, cross_pO, sympy2casadi

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

def inv_linear_vel_R(phi, thet, psi):
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

def inv_angular_vel_T(phi, thet):
    T_1 = SX.eye(3)
    T_1[0,2] = -sin(thet)
    T_1[1,1] = cos(phi)
    T_1[1,2] = cos(thet)*sin(phi)
    T_1[2,1] = -sin(phi)
    T_1[2,2] = cos(thet)*cos(phi)
    return T_1

def J_kin(eul):
    phi, thet, psi = eul[0],eul[1],eul[2]
    R = linear_vel_R(phi, thet, psi)
    T = angular_vel_T(phi, thet)
    J = SX.zeros(6, 6)
    J[:3,:3] = R
    J[3:,3:] = T
    return J,R,T

def J_dot(eul, deul,dT, eul_sp, w_nb):
    phi, thet, _ = eul[0],eul[1],eul[2]
    dthet, dphi, _ = deul[0],deul[1],deul[2]
    theta_sp, dtheta_sp, phi_sp, dphi_sp = eul_sp[0], eul_sp[1], eul_sp[2], eul_sp[3]
    _,R,T = J_kin(eul)
    dR = rot_diff(R, w_nb)
    dT = sympy2casadi(dT, [theta_sp, dtheta_sp, phi_sp, dphi_sp], vertcat(thet,dthet,phi,dphi))
    dJ = SX.zeros(6, 6)
    dJ[:3,:3] = dR
    dJ[3:,3:] = dT
    return dJ, dR, dT


def inv_J_kin(eul):
    phi, thet, psi = eul[0],eul[1],eul[2]
    RT = inv_linear_vel_R(phi, thet, psi)
    inv_T = inv_angular_vel_T(phi, thet)
    inv_J = SX.zeros(6, 6)
    inv_J[:3,:3] = RT
    inv_J[3:,3:] = inv_T
    return inv_J ,RT ,inv_T