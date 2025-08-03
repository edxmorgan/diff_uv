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

"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from casadi import SX, inv, sin,cos, fabs, diag, pinv,substitute
from platform import machine, system

from diffUV.utils import operators as ops
from diffUV.utils.operators import cross_pO, coriolis_lag_param 
from diffUV.utils.symbols import *
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

    def _initialize_inertia_matrix(self):
        """Internal method to compute the UV inertia matrix based on vehicle parameters."""
        self._initialize_mass_rb()
        # self._initialize_mass_ma()
        __M = (self.M_rb + MA) 
        # Apply symmetry considerations. 
        self.M = __M* sb_fft_config
        
    def body_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments."""
        return self.M
    
    def body_coriolis_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        M = self.body_inertia_matrix()
        C = coriolis_lag_param(M, v_r)
        return C

    def body_restoring_vector(self):
        """Compute and return the hydrostatic restoring forces with a smooth dynamic transition.
        
        This implementation keeps the original if_else structure but makes the effective values 
        in the buoyancy computation dynamic. In particular, the constant B is replaced by 
        a dynamic variable that smoothly transitions from W at the water surface (z = 0) 
        to B when above the water, over a small region defined by B_eps.
        """
        g = SX(6, 1)
        g[0, 0] = (W - B) * sin(thet)
        g[1, 0] = -(W - B) * cos(thet) * sin(phi)
        g[2, 0] = -(W - B) * cos(thet) * cos(phi)
        g[3, 0] = -(y_g * W - y_b * B) * cos(thet) * cos(phi) + (z_g * W - z_b * B) * cos(thet) * sin(phi)
        g[4, 0] = (z_g * W - z_b * B) * sin(thet) + (x_g * W - x_b * B) * cos(thet) * cos(phi)
        g[5, 0] = -(x_g * W - x_b * B) * cos(thet) * sin(phi) - (y_g * W - y_b * B) * sin(thet)

        return g


    def body_damping_matrix(self):
        """Compute and return the total damping forces, including both linear and nonlinear components in body"""
        linear_damping = -diag(vertcat(X_u,Y_v,Z_w,K_p,M_q,N_r))
        nonlinear_damping = -diag(vertcat(X_uu,Y_vv,Z_ww,K_pp,M_qq,N_rr)*fabs(v_r))
        D_v = linear_damping + nonlinear_damping
        return D_v

    def get_bias(self):
        C = self.body_coriolis_centripetal_matrix()@v_r
        g = self.body_restoring_vector()
        d = self.body_damping_matrix()@v_r
        B = C + g + d - f_ext
        return B
    
    def body_forward_dynamics(self):
        """
        Calculate body accelerations based on inverse dynamics.
        """
        acc = inv(self.body_inertia_matrix())@(tau_b + f_ext - self.body_coriolis_centripetal_matrix()@v_r - self.body_damping_matrix()@v_r - self.body_restoring_vector())
        return acc
    
    def body_inverse_dynamics(self):
        """
        Calculate the required torque (resultant torque) based on the desired acceleration,
        using inverse dynamics.
        """
        resultant_torque = -f_ext + self.body_inertia_matrix()@self.v_rdot + self.body_coriolis_centripetal_matrix()@v_r + self.body_damping_matrix()@(v_r) + self.body_restoring_vector()
        return resultant_torque 
    
    def control_Allocation(self):
        thruster_F = pinv(Tc)@tau_b
        return thruster_F
    
    def thruster_input2generalized_Forces(self):
        tau = Tc@thru_u
        return tau