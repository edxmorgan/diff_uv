from casadi import skew, SX

def Smtrx(v):
    S = skew(v)
    return S

def coriolis_lag_param(M, x_nb):
    # coriolis_lagrange_parameterization
    v_nb, w_nb = x_nb[:3], x_nb[3:]
    # print(v_nb)
    # print(w_nb)
    M11 = M[:3, :3]
    M12 = M[:3, 3:]
    M21 = M[3:, :3]
    M22 = M[3:, 3:]
    C = SX.zeros(6, 6)
    C[3:, :3] = -Smtrx(M11@v_nb + M12@w_nb)
    C[:3, 3:] = -Smtrx(M11@v_nb + M12@w_nb)
    C[3:, 3:] = -Smtrx(M21@v_nb + M22@w_nb)
    return C