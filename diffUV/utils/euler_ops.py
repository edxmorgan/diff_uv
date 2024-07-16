# Copyright 2024, Edward Morgan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from casadi import cos, SX, sin, tan, inv, vertcat
from diffUV.utils.operators import rot_diff, cross_pO, sympy2casadi

# Eq 6.3. Ocean current aceleration assuming constant and irrotational flow. 
def rel_acc(dx_nb, w_nb, v_c):
    v_cdot = SX.zeros(6,6) # actual 6.6 zeros. 
    S = cross_pO(w_nb)
    v_cdot[:3,:3] = -S
    v_cdot = v_cdot@v_c
    v_rdot = dx_nb - v_cdot # Relative accel
    return v_rdot, v_cdot


def linear_vel_R(eul):
    phi, thet, psi = eul[0],eul[1],eul[2]
    R = SX(3, 3)

    R[0,0] = cos(psi)*cos(thet)
    R[0,1] = -sin(psi)*cos(phi) + cos(psi)*sin(thet)*sin(phi)
    R[0,2] = sin(psi)*sin(phi) + cos(psi)*cos(phi)*sin(thet)

    R[1,0] = sin(psi)*cos(thet)
    R[1,1] = cos(psi)*cos(phi) + sin(phi)*sin(thet)*sin(psi)
    R[1,2] = -cos(psi)*sin(phi) + sin(thet)*sin(psi)*cos(phi)

    R[2,0] = -sin(thet)
    R[2,1] = cos(thet)*sin(phi)
    R[2,2] = cos(thet)*cos(phi)
    return R

def inv_linear_vel_R(eul):
    R = linear_vel_R(eul)
    return R.T

def angular_vel_T(eul):
    phi, thet, psi = eul[0],eul[1],eul[2]
    #T(nb) is undefined for a pitch(psi) angle of θ = ± 90◦
    T = SX.eye(3)
    T[0,1] = sin(phi)*tan(thet)
    T[0,2] = cos(phi)*tan(thet)
    T[1,1] = cos(phi)
    T[1,2] = -sin(phi)
    T[2,1] = sin(phi)/cos(thet)
    T[2,2] = cos(phi)/cos(thet)
    return T

def inv_angular_vel_T(eul):
    phi, thet, psi = eul[0],eul[1],eul[2]
    T_1 = SX.eye(3)
    T_1[0,2] = -sin(thet)
    T_1[1,1] = cos(phi)
    T_1[1,2] = cos(thet)*sin(phi)
    T_1[2,1] = -sin(phi)
    T_1[2,2] = cos(thet)*cos(phi)
    return T_1

def J_kin(eul):
    R = linear_vel_R(eul)
    T = angular_vel_T(eul)
    J = SX.zeros(6, 6)
    J[:3,:3] = R
    J[3:,3:] = T
    return J,R,T

def J_dot(eul, deul,dT, eul_sp, w_nb):
    phi, thet, _ = eul[0],eul[1],eul[2]
    dphi, dthet, _ = deul[0],deul[1],deul[2]
    theta_sp, dtheta_sp, phi_sp, dphi_sp = eul_sp[0], eul_sp[1], eul_sp[2], eul_sp[3]
    _,R,T = J_kin(eul)
    dR = rot_diff(R, w_nb)
    dT = sympy2casadi(dT, [theta_sp, dtheta_sp, phi_sp, dphi_sp], vertcat(thet,dthet,phi,dphi))
    dJ = SX.zeros(6, 6)
    dJ[:3,:3] = dR
    dJ[3:,3:] = dT
    return dJ, dR, dT


def inv_J_kin(eul):
    RT = inv_linear_vel_R(eul)
    inv_T = inv_angular_vel_T(eul)
    inv_J = SX.zeros(6, 6)
    inv_J[:3,:3] = RT
    inv_J[3:,3:] = inv_T
    return inv_J ,RT ,inv_T