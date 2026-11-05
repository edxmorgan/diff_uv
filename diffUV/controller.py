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

"""This module contains a class for implementing a MIMO Nonlinear Controllers
"""
from diffUV.kinematics import Kinematics as kin
from diffUV.base import Base as dyn_body
from diffUV.utils.symbols import *
import casadi as ca
from diffUV.utils.rotation_error import rotation_log
from diffUV.utils import euler_ops as T_eul
from diffUV.utils.operators import coriolis_lag_param 

class Controller():
    def __init__(self, uv_body: dyn_body = None):
        # ned kinematic transformation
        Kinematics = kin()
        self.J_ = Kinematics.J
        # body representaion
        uv_body =  uv_body if uv_body is not None else dyn_body()
        mB = uv_body.surface_interaction(z, W, B, B_eps=3.0)
        self.gn = uv_body.body_restoring_vector(mB)

        Y , YMC, YG, YD, self.lumped_params, inertia_mat_id, C_id, g_id, body_damping_matrix_id = uv_body.build_sys_regressor()
        (m_X_du, m_Y_dv, m_Z_dw, mz_g_X_dq, _mz_g_Y_dp, _mz_g_K_dv, mz_g_M_du, I_x_K_dp, I_y_M_dq, I_z_N_dr, 
         W_, B_, x_gW_x_bB, y_gW_y_bB , z_gW_z_bB, 
         X_u_,Y_v_,Z_w_,K_p_,M_q_,N_r_, X_uu_,Y_vv_,Z_ww_,K_pp_,M_qq_,N_rr_)  = ca.vertsplit(self.lumped_params)

        # inertia and coriolis force
        self.inertia_mat_id = SX.zeros(6,6)
        self.inertia_mat_id[0,0] = m_X_du
        self.inertia_mat_id[1,1] = m_Y_dv
        self.inertia_mat_id[2,2] = m_Z_dw
        self.inertia_mat_id[0,4] = mz_g_X_dq            # u–q
        self.inertia_mat_id[1,3] = _mz_g_Y_dp            # v–p
        self.inertia_mat_id[3,1] = _mz_g_K_dv            # p–v
        self.inertia_mat_id[4,0] = mz_g_M_du            # q–u
        self.inertia_mat_id[3,3] = I_x_K_dp
        self.inertia_mat_id[4,4] = I_y_M_dq
        self.inertia_mat_id[5,5] = I_z_N_dr

        self.C_id = coriolis_lag_param(self.inertia_mat_id, v_r)

        mB_ = uv_body.surface_interaction(z, W_, B_, B_eps=3.0)
        self.gn_Y = YG@vertcat(W_, mB_, x_gW_x_bB, y_gW_y_bB , z_gW_z_bB)

    def __repr__(self) -> str:
        return f'{super().__repr__()} Simulator'
    
    def geometric_reference(self):
        """World position/rotation-vector errors; physical body twist inputs."""
        R = T_eul.linear_vel_R(eul)
        Rd = T_eul.linear_vel_R(nd[3:6])
        G = ca.diagcat(R, R)
        Gd = ca.diagcat(Rd, Rd)
        error = ca.vertcat(nd[:3]-n[:3], R @ rotation_log(R.T @ Rd))
        integral = sum_e_buffer + error*dt
        return R, Rd, G, Gd, error, integral

    def position_pid(self):
        _, _, G, Gd, error, integral = self.geometric_reference()
        feedback = diag(Kp)@error + diag(Ki)@integral + diag(Kd)@(Gd@xb_d-G@x_nb)
        return self.gn + G.T@feedback, integral

    def position_pid_reg(self):
        _, _, G, Gd, error, integral = self.geometric_reference()
        feedback = diag(Kp)@error + diag(Ki)@integral + diag(Kd)@(Gd@xb_d-G@x_nb)
        return self.gn_Y + G.T@feedback, integral

    def trajectorytracking_pid(self):
        R, Rd, G, _, error, integral = self.geometric_reference()
        A = R.T @ Rd
        Adot = -ca.skew(x_nb[3:6])@A + A@ca.skew(xb_d[3:6])
        transport = ca.diagcat(A, A)
        # Reference body-component derivatives transported to the current body.
        acceleration = transport@des_acc_b + ca.diagcat(Adot, Adot)@xb_d
        velocity = transport@xb_d + G.T@(diag(Kp)@error + diag(Ki)@integral)
        tracking_acc = acceleration + diag(Kd)@(velocity-x_nb)
        tracking_u = self.inertia_mat_id@tracking_acc + self.C_id@v_r + self.gn_Y
        return tracking_u, integral
