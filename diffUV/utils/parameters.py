import numpy as np

class Params:
    thrust_config = np.array([[-0.707, -0.707, 0.707, 0.707, 0.0, 0.0, 0.0, 0.0],
        [0.707, -0.707, 0.707, -0.707, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -0.218, -0.218, 0.218, 0.218],
        [0.0, 0.0, 0.0, 0.0, -0.12, 0.12, -0.12, 0.12],
        [0.1888, -0.1888, -0.1888, 0.1888, 0.0, 0.0, 0.0, 0.0]])
    
    #thrust coefficient matrix
    K = np.diag([40, 40, 40, 40, 40, 40, 40, 40])

    # parameters in rigid body dynamics and restoring forces
    m = 11.5 #(kg)
    W = 112.8 #(N)
    B = 114.8 #(N)
    rb  = np.array([[0], [0], [0]]) #(m)
    rg = np.array([[0], [0], [0.02]]) #(m)
    I_x = 0.16 #(kg m2)
    I_y = 0.16 #(kg m2)
    I_z = 0.16 #(kg m2)

    # added mass parameters
    X_du = -5.5 #(kg)
    Y_dv = -12.7 #(kg)
    Z_dw = -14.57 #(kg)
    K_dp = -0.12 #(kg m2/rad)
    M_dq = -0.12 #(kg m2/rad)
    N_dr = -0.12 #(kg m2/rad)

    Xu = -4.03 #(Ns/m) 
    Xuu = -18.18 #(Ns2/m2)
    Yv = -6.22 #(Ns/m) 
    Yvv = -21.66 #(Ns2/m2)
    Zw = -5.18 #(Ns/m) 
    Zww = -36.99 #(Ns2/m2)
    Kp  = -0.07 #(Ns/rad) 
    Kpp = -1.55 #(Ns2/rad2)
    Mq = -0.07 #(Ns/rad)
    Mqq = -1.55 #(Ns2/rad2)
    Nr = -0.07 #(Ns/rad) 
    Nrr = -1.55 #(Ns2/rad2)

