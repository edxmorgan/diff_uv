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
from casadi import SX, inv, sin,cos, fabs, diag, pinv,substitute, vertsplit, jacobian, if_else
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
        self.lumped_params = SX.sym('lumped_params', 27)
    
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

    def body_restoring_vector(self, mB=B):
        """Compute and return the hydrostatic restoring forces with a smooth dynamic transition.
        Note: mB is the smooth dynamic buoyancy force computed in the simulator class."""
        g = SX(6, 1)
        g[0, 0] = (W - mB) * sin(thet)
        g[1, 0] = -(W - mB) * cos(thet) * sin(phi)
        g[2, 0] = -(W - mB) * cos(thet) * cos(phi)
        g[3, 0] = -(y_g * W - y_b * mB) * cos(thet) * cos(phi) + (z_g * W - z_b * mB) * cos(thet) * sin(phi)
        g[4, 0] = (z_g * W - z_b * mB) * sin(thet) + (x_g * W - x_b * mB) * cos(thet) * cos(phi)
        g[5, 0] = -(x_g * W - x_b * mB) * cos(thet) * sin(phi) - (y_g * W - y_b * mB) * sin(thet)
        return g

    def surface_interaction(self, z, W, B, B_eps):
        # When the object is just above the water surface (0 <= z < eps),
        # B_dynamic linearly interpolates between W (at z=0) and B (at z=eps).
        B_dynamic = if_else(z < B_eps, W + (B - W) * (z / B_eps), B)

        # Use the original structure:
        #   - If z == 0, set mB to the (dynamic) W value (which is just W).
        #   - If z < 0, then mB is 0.
        #   - Otherwise (z > 0), use the smooth dynamic value B_dynamic.
        mB = if_else(z == 0.0, W, if_else(z < 0.0, 0.0, B_dynamic))
        return mB


    def body_restoring_vector_jacobian_wrt_params(self):
        # restoring force jacobian
        YG = SX.zeros(6,5)
        sTh, cTh = sin(thet), cos(thet)
        sPh, cPh = sin(phi),   cos(phi)

        YG[0,0], YG[0,1] =  sTh, -sTh
        YG[1,0], YG[1,1] = -cTh*sPh,  cTh*sPh
        YG[2,0], YG[2,1] = -cTh*cPh,  cTh*cPh

        YG[3,3], YG[3,4] = -cTh*cPh,  cTh*sPh
        YG[4,2], YG[4,4] =  cTh*cPh,  sTh
        YG[5,2], YG[5,3] = -cTh*sPh, -sTh
        return YG
    
    def body_damping_matrix(self):
        """Compute and return the total damping forces, including both linear and nonlinear components in body"""
        linear_damping = -diag(vertcat(X_u,Y_v,Z_w,K_p,M_q,N_r))
        nonlinear_damping = -diag(vertcat(X_uu,Y_vv,Z_ww,K_pp,M_qq,N_rr)*fabs(v_r))
        D_v = linear_damping + nonlinear_damping
        return D_v

    def get_bias(self):
        mB = self.surface_interaction(z, W, B, B_eps=3.0)
        C = self.body_coriolis_centripetal_matrix()@v_r
        g = self.body_restoring_vector(mB)
        d = self.body_damping_matrix()@v_r
        bias = C + g + d - f_ext
        return bias
    
    def body_forward_dynamics(self):
        """
        Calculate body accelerations based on inverse dynamics.
        """
        mB = self.surface_interaction(z, W, B, B_eps=3.0)
        acc = inv(self.body_inertia_matrix())@(tau_b + f_ext 
                                               - self.body_coriolis_centripetal_matrix()@v_r 
                                               - self.body_damping_matrix()@v_r 
                                               - self.body_restoring_vector(mB))
        return acc
    
    def body_inverse_dynamics(self):
        """
        Calculate the required torque (resultant torque) based on the desired acceleration,
        using inverse dynamics.
        """
        mB = self.surface_interaction(z, W, B, B_eps=3.0)
        resultant_torque = -f_ext + self.body_inertia_matrix()@self.v_rdot + self.body_coriolis_centripetal_matrix()@v_r + self.body_damping_matrix()@(v_r) + self.body_restoring_vector(mB)
        return resultant_torque 
    
    def control_Allocation(self):
        thruster_F = pinv(Tc)@tau_b
        return thruster_F
    
    def thruster_input2generalized_Forces(self):
        tau = Tc@thru_u
        return tau
    
    def build_sys_regressor(self):
        # β = [m-X_du, m-Y_dv, m-Z_dw, m*z_g-X_dq, -m*z_g+Y_dp,
        #      -m*z_g+K_dv, m*z_g-M_du, I_x-K_dp, I_y-M_dq, I_z-N_dr , W, B, x_g*W - x_b*B, y_g*W - y_b*B , z_g*W - z_b*B, X_u,Y_v,Z_w,K_p,M_q,N_r, X_uu,Y_vv,Z_ww,K_pp,M_qq,N_rr]
        lumped_params = self.lumped_params 
        (m_X_du, m_Y_dv, m_Z_dw, mz_g_X_dq, _mz_g_Y_dp, _mz_g_K_dv, mz_g_M_du, I_x_K_dp, I_y_M_dq, I_z_N_dr, 
         W_, B_, x_gW_x_bB, y_gW_y_bB , z_gW_z_bB, 
         X_u_,Y_v_,Z_w_,K_p_,M_q_,N_r_, X_uu_,Y_vv_,Z_ww_,K_pp_,M_qq_,N_rr_)  = vertsplit(lumped_params)
        
        # I_z_N_dr, N_r_ , N_rr_, yaw first

        # inertia and coriolis force
        inertia_mat_id = SX.zeros(6,6)
        inertia_mat_id[0,0] = m_X_du
        inertia_mat_id[1,1] = m_Y_dv
        inertia_mat_id[2,2] = m_Z_dw
        inertia_mat_id[0,4] = mz_g_X_dq            # u–q
        inertia_mat_id[1,3] = _mz_g_Y_dp            # v–p
        inertia_mat_id[3,1] = _mz_g_K_dv            # p–v
        inertia_mat_id[4,0] = mz_g_M_du            # q–u
        inertia_mat_id[3,3] = I_x_K_dp
        inertia_mat_id[4,4] = I_y_M_dq
        inertia_mat_id[5,5] = I_z_N_dr

        C_id = coriolis_lag_param(inertia_mat_id, v_r)

        YMC = jacobian((inertia_mat_id@self.v_rdot + C_id@v_r), vertcat(m_X_du, m_Y_dv, m_Z_dw, mz_g_X_dq, _mz_g_Y_dp, _mz_g_K_dv, mz_g_M_du, I_x_K_dp, I_y_M_dq, I_z_N_dr))

        YG = self.body_restoring_vector_jacobian_wrt_params()
        g_id = YG@vertcat(W_, B_, x_gW_x_bB, y_gW_y_bB , z_gW_z_bB,)

        # damping force
        linear_damping_coeff = vertcat(X_u_,Y_v_,Z_w_,K_p_,M_q_,N_r_)
        quad_damping_coeff = vertcat(X_uu_,Y_vv_,Z_ww_,K_pp_,M_qq_,N_rr_)
        linear_damping = -diag(linear_damping_coeff)
        
        nonlinear_damping = -diag(quad_damping_coeff*fabs(v_r))
        body_damping_matrix_id = linear_damping + nonlinear_damping
        D_v_F = body_damping_matrix_id@v_r
        YD = jacobian(D_v_F, vertcat(linear_damping_coeff, quad_damping_coeff))
    
        Y = horzcat(YMC, YG, YD)
        return Y , YMC, YG, YD, lumped_params, inertia_mat_id, C_id, g_id, body_damping_matrix_id
        
    def body_forward_dynamics_reg(self):
        """
        Calculate body accelerations based on inverse dynamics based on regressor lump parameter.
        """
        Y , YMC, YG, YD, lumped_params, inertia_mat_id, C_id, g_id, body_damping_matrix_id = self.build_sys_regressor()
        (m_X_du, m_Y_dv, m_Z_dw, mz_g_X_dq, _mz_g_Y_dp, _mz_g_K_dv, mz_g_M_du, I_x_K_dp, I_y_M_dq, I_z_N_dr, 
         W_, B_, x_gW_x_bB, y_gW_y_bB , z_gW_z_bB, 
         X_u_,Y_v_,Z_w_,K_p_,M_q_,N_r_, X_uu_,Y_vv_,Z_ww_,K_pp_,M_qq_,N_rr_)  = vertsplit(lumped_params)
        
        mB = self.surface_interaction(z, W_, B_, B_eps=3.0)
        g_id_z_lock = YG@vertcat(W_, mB, x_gW_x_bB, y_gW_y_bB , z_gW_z_bB,)
        
        acc = inv(inertia_mat_id)@(tau_b + f_ext - C_id@v_r - body_damping_matrix_id@v_r - g_id_z_lock)
        return acc, lumped_params
    
