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
from diffUV.utils import euler_ops as T_eul
from diffUV.utils.operators import coriolis_lag_param 

class Controller():
    def __init__(self):
        # ned kinematic transformation
        Kinematics = kin()
        self.J_ = Kinematics.J
        # body representaion
        uv_body = dyn_body()
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
    
    def position_pid(self):
        ne = nd - n

        eul_d = nd[3:6]
        J_d, _, _ = T_eul.J_kin(eul_d)

        i_buffer = sum_e_buffer + ne*dt

        pid = diag(Kp)@ne + diag(Ki)@i_buffer + diag(Kd)@(J_d@xb_d - self.J_@x_nb)

        pid_controller = self.gn + self.J_.T@pid

        return pid_controller, i_buffer
    
    def position_pid_reg(self):
        ne = nd - n

        eul_d = nd[3:6]
        J_d, _, _ = T_eul.J_kin(eul_d)
        
        i_buffer = sum_e_buffer + ne*dt

        pid = diag(Kp)@ne + diag(Ki)@i_buffer + diag(Kd)@(J_d@xb_d - self.J_@x_nb)

        pid_controller = self.gn_Y + self.J_.T@pid

        return pid_controller, i_buffer
    
    
    def trajectorytracking_pid(self):
        #trajectorytracking using inverse dynamics computed torque
        ne = nd - n

        inv_J ,RT ,inv_T = T_eul.inv_J_kin(eul)
        
        i_buffer = sum_e_buffer + ne*dt

        desired_vel_fb = inv_J@(diag(Kp)@ne + diag(Ki)@i_buffer)
        
        v_ref_b = xb_d + desired_vel_fb
        v_err_b = v_ref_b - x_nb
        tracking_acc = des_acc_b + diag(Kd) @ v_err_b

        tracking_u = (self.inertia_mat_id @ tracking_acc
                  + self.C_id @ v_r
                  + self.gn_Y)
        
        return tracking_u, i_buffer