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

import numpy as np

class Params:

    # Ocean current velocities. 
    v_flow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # (m/s). Assume irrotational, constant.

    # Thrust configuration matrix by BlueROV2 Heavy. Converts thrust force to body and vice versa. 
    # TODO Check Calcs to determine how to find. 
    thrust_config = np.array([
        [-0.7070, -0.7070,  0.7070,  0.7070,  0.0000,  0.0000,  0.0000, 0.0000],
        [ 0.7070, -0.7070,  0.7070, -0.7070,  0.0000,  0.0000,  0.0000, 0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -1.0000, -1.0000, 1.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000, -0.2180, -0.2180,  0.2180, 0.2180],
        [ 0.0000,  0.0000,  0.0000,  0.0000, -0.1200,  0.1200, -0.1200, 0.1200],
        [ 0.1888, -0.1888, -0.1888,  0.1888,  0.0000,  0.0000,  0.0000, 0.0000]])
    
    # Alternative thrust config matrix. 
    # thrust_config = np.array([[0.707, 0.707, -0.707, -0.707, 0.0, 0.0, 0.0, 0.0],
    #                     [-0.707, 0.707, -0.707, 0.707, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0,-1.0],
    #                     [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
    #                     [0.06, 0.06, -0.06, -0.06, 0.12, -0.12, 0.12, -0.12],
    #                     [-0.1888, 0.1888, 0.1888, -0.1888, 0.0, 0.0, 0.0, 0.0]])
    
    # Thrust coefficient matrix
    K = np.diag([40, 40, 40, 40, 40, 40, 40, 40])

    ### Parameters in rigid body dynamics and restoring forces
    # Based on BlueRobotics 2018b technical specs. 
    # Based on Table 5.1
    m = 11.5 #(kg)
    W = m*9.81 #(N). 112.8 N. Weight. 
    B = 114.8 #(N). Buoyant force assuming net buoyancy of 0.2 kg. 2N const up force. 
    rb  = np.array([0, 0, 0]) #(m). Placing the centre of the vehicles body frame at center of buoyancy (CoB). 
    rg = np.array([0, 0, 0.02]) #(m). Assumption that Center of Gravity (CoG) is this distance from CoB. 
    h = 0.254

    # Axis inertias. 
    # BAsed on Table 5.1.
    I_x = 0.16 #(kg m2)
    I_y = 0.16 #(kg m2)
    I_z = 0.16 #(kg m2)
    I_xz = 0
    Io = np.array([I_x, I_y, I_z, I_xz])

    # Added mass parameters.
    # Based on Table 5.2. 
    X_du = -5.5 #(kg). Surge. 
    Y_dv = -12.7 #(kg). Sway. 
    Z_dw = -14.57 #(kg). Heave. 
    K_dp = -0.12 #(kg m2/rad). Roll.
    M_dq = -0.12 #(kg m2/rad). Pitch. 
    N_dr = -0.12 #(kg m2/rad). Yaw. 
    added_m = np.array([X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr])

    coupl_added_m = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # ASSUMING decoupling motion

    # Linear damping coeffs. 
    Xu = -4.03 #(Ns/m). Surge. 
    Yv = -6.22 #(Ns/m). Sway.
    Zw = -5.18 #(Ns/m). Heave.  
    Kp = -0.07 #(Ns/rad). Roll.
    Mq = -0.07 #(Ns/rad). Pitch.
    Nr = -0.07 #(Ns/rad). Yaw. 
    linear_dc = np.array([Xu, Yv, Zw, Kp,  Mq, Nr])

    # Quadratic damping coeffs. 
    Xuu = -18.18 #(Ns2/m2). Surge. 
    Yvv = -21.66 #(Ns2/m2). Sway. 
    Zww = -36.99 #(Ns2/m2). Heave. 
    Kpp = -1.55 #(Ns2/rad2). Roll. 
    Mqq = -1.55 #(Ns2/rad2). Pitch. 
    Nrr = -1.55 #(Ns2/rad2). Yaw. 
    quadratic_dc = np.array([Xuu, Yvv, Zww, Kpp, Mqq, Nrr])