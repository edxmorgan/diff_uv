# Copyright (C) 2024 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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