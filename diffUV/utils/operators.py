from casadi import skew, SX
import casadi

def cross_pO(v):
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
    C[3:, :3] = -cross_pO(M11@v_nb + M12@w_nb)
    C[:3, 3:] = -cross_pO(M11@v_nb + M12@w_nb)
    C[3:, 3:] = -cross_pO(M21@v_nb + M22@w_nb)
    return C

def rot_diff(R_n, w_b):
    S = cross_pO(w_b)
    dR_n = R_n@S
    return dR_n

def sympy2casadi(sympy_expr,sympy_var,casadi_var):
    assert casadi_var.is_vector()
    if casadi_var.shape[1]>1:
        casadi_var = casadi_var.T
    casadi_var = casadi.vertsplit(casadi_var)
    from sympy.utilities.lambdify import lambdify

    mapping = {'ImmutableDenseMatrix': casadi.blockcat,
                'MutableDenseMatrix': casadi.blockcat,
                'Abs':casadi.fabs
            }
    f = lambdify(sympy_var, sympy_expr,modules=[mapping, casadi])
    # print(casadi_var)
    return f(*casadi_var)