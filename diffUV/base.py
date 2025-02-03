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

"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from casadi import SX, inv, sin,cos, fabs, diag, pinv,substitute, if_else, logic_and
from platform import machine, system

from diffUV.utils import operators as ops
from diffUV.utils.operators import cross_pO, coriolis_lag_param # CHANGE? 
from diffUV.utils.symbols import *
# Repeats?
from diffUV.utils import euler_ops as T
from diffUV.utils import euler_ops as T_eul

class Base(object):
    func_opts = {}
    jit_func_opts = {"jit": True, "jit_options": {"flags": "-Ofast"}}
    # OS/CPU dependent specification of compiler
    # System checks the system/OS name. Machine checks machine type. 
    # https://docs.python.org/3/library/platform.html
    if system().lower() == "darwin" or machine().lower() == "aarch64":
        jit_func_opts["compiler"] = "shell"

    def __repr__(self) -> str:
        return "differentiable underwater dynamics"

    def __init__(self, func_opts=None, use_jit=True):
        if func_opts:
            self.func_opts = func_opts
        if use_jit:
            # NOTE: use_jit=True requires that CasADi is built with Clang
            for k, v in self.jit_func_opts.items():
                self.func_opts[k] = v

        self._initialize_inertia_matrix()
        # 1x6 vector. Xyz, rpy. 
        self.body_state_vector = x_nb
        self.J, self.R, self.T = T_eul.J_kin(eul)
        self.v_rdot, self.v_cdot = T_eul.rel_acc(dx_nb, w_nb, v_c)
    
    # Follow 6.2. 
    # Mass matrix already made in symbolic. Rigid body made here. 
    def _initialize_inertia_matrix(self):
        """Internal method to compute the UV inertia matrix based on vehicle parameters."""
        self._initialize_mass_rb()
        # self._initialize_mass_ma()
        __M = (self.M_rb + MA) 
        # Apply symmetry considerations. 
        self.M = __M* sb_fft_config

    def _initialize_mass_rb(self):
        # ASSUMPTIONS. Ixy = Iyz = 0. yg = 0. 
        # Making matrix/Eq 8.8. 
        M_rb = SX(6,6) # representative 6x6 0's 
        S = cross_pO(r_g) # Eq 2.13 (skew symmetric) where lambda is CoG wrt CO
        M_rb[:3,:3] = m*SX.eye(3) # Quad 1 
        M_rb[:3,3:] = -m*S # Quad 2 
        M_rb[3:,:3] = m*S # Quad 3
        # Quad 4 Apply yg= 0 and Ixy=Iyz=0
        M_rb[3:,3:] = Ib_b 
        M_rb = substitute(M_rb, I_xy, SX(0))
        M_rb = substitute(M_rb, I_yz, SX(0))
        M_rb = substitute(M_rb, y_g, SX(0))
        self.M_rb = M_rb # save

    def body_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments."""
        # M = Function("M", syms , [M], self.func_opts)
        return self.M
    
    # According to Eq. 10.7, Coriolis must be split
    # into rigid body and hydrodynamic terms and added. 
    def body_coriolis_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        M = self.body_inertia_matrix()
        C = coriolis_lag_param(M, v_r)
        return C

    def body_restoring_vector(self):
        """Compute and return the hydrostatic restoring forces."""

        # A positively buoyant object doesn't "fly out" of water when it reaches the surface because once it reaches the water level,
        # the buoyant force acting on it becomes equal to its weight, creating a balance and preventing further upward movement; 
        # essentially, the upward force of buoyancy is counteracted by the downward force of gravity, resulting in a stable floating
        # position at the water surface.

        # Define buoyancy_conditions
        b_condition1 = z == 0.0
        b_condition2 = z < 0.0

        # Define mB using nested if_else
        mB = if_else(b_condition1, W, if_else(b_condition2, 0.0, B))

        g = SX(6, 1)
        g[0, 0] = (W - mB)*sin(thet)
        g[1, 0] = -(W - mB)*cos(thet)*sin(phi)
        g[2, 0] = -(W - mB)*cos(thet)*cos(phi)
        g[3, 0] = -(y_g*W - y_b*mB)*cos(thet)*cos(phi) + \
            (z_g*W - z_b*mB)*cos(thet)*sin(phi)
        g[4, 0] = (z_g*W - z_b*mB)*sin(thet) + \
            (x_g*W - x_b*mB)*cos(thet)*cos(phi)
        g[5, 0] = -(x_g*W - x_b*mB)*cos(thet) * \
            sin(phi) - (y_g*W - y_b*mB)*sin(thet)

        return g

     # D(v_r) Eq 8.10. Vehicle is performing non-coupled motion. 
    def body_damping_matrix(self):
        """Compute and return the total damping forces, including both linear and nonlinear components in body"""
        linear_damping = -diag(vertcat(X_u,Y_v,Z_w,K_p,M_q,N_r))
        # fabs: absolute value. vertcat: makes a column. diag distributes a row or column across a diagonal. 
        # Damping depends on Vr.
        nonlinear_damping = -diag(vertcat(X_uu,Y_vv,Z_ww,K_pp,M_qq,N_rr)*fabs(v_r)) # Quadratic
        D_v = linear_damping + nonlinear_damping
        return D_v

    # Solved for accel based on inv dyn. 
    def body_forward_dynamics(self):
        acc = inv(self.body_inertia_matrix())@(tau_b + f_ext - self.body_coriolis_centripetal_matrix()@v_r - self.body_damping_matrix()@v_r - self.body_restoring_vector())
        return acc

    # G_0 missing bc underwater vehicle. No ballast control applicable. Eq. 6.4. 
    def body_inverse_dynamics(self):
        resultant_torque = -f_ext + self.body_inertia_matrix()@self.v_rdot + self.body_coriolis_centripetal_matrix()@v_r + self.body_damping_matrix()@(v_r) + self.body_restoring_vector() 
        return resultant_torque
    
    def control_Allocation(self):
        # u = inv(K)@pinv(Tc)@tau_b
        thruster_F = pinv(Tc)@tau_b
        return thruster_F
    
    def thruster_input2generalized_Forces(self):
        # tau = K@Tc@u
        tau = Tc@thru_u
        return tau