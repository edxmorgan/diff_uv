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

from casadi import skew, SX
import casadi

# Returns Skew symmetric matrix defined by 3 val vector v. 
# Based on Eq 2.13
def cross_pO(v):
    S = skew(v)
    return S

# coriolis_lagrange_parameterization
def coriolis_lag_param(M, x_nb):
    # Decompose into linear and angular vel.
    v_nb, w_nb = x_nb[:3], x_nb[3:]
    # print(v_nb)
    # print(w_nb)
    # Decompose into 4ths. 
    M11 = M[:3, :3] # Quad 1
    M12 = M[:3, 3:] # Quad 2
    M21 = M[3:, :3] # Quad 3
    M22 = M[3:, 3:] # Quad 4 

    # Create coriolis matrix. Based on Eq. 6.44. 
    C = SX.zeros(6, 6)
    C[:3, 3:] = -cross_pO(M11@v_nb + M12@w_nb) # Quad 2 
    C[3:, :3] = -cross_pO(M11@v_nb + M12@w_nb) # Quad 3 
    C[3:, 3:] = -cross_pO(M21@v_nb + M22@w_nb) # Quad 4 
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