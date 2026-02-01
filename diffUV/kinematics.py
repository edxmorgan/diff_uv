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
from diffUV.base import Base
from diffUV.utils import euler_ops as T_eul
from diffUV.utils import quaternion_ops as T_quat
from diffUV.utils.symbols import *

class Kinematics():
    def __init__(self):
        self.J, self.R, self.T = T_eul.J_kin(eul)
        self.J_INV, self.R_INV, self.T_INV = T_eul.inv_J_kin(eul)
        
        _n_dot     = self.ned_euler_vel_from_body()
        _deul      = _n_dot[3:6]           # Euler rates
        self.J_dot, self.dR, self.dT = T_eul.J_dot(eul,_deul,dT_sp,eul_sp,w_nb)

        self.Jq, self.Rq, self.Tq = T_quat.Jq_kin(uq)
        self.Jq_INV, _,_ = T_quat.inv_Jq_kin(uq)
        self.Jq_INV_T = self.Jq_INV.T
        self.Jq_dot, self.dRq ,self.dTq = T_quat.Jq_dot(uq, w_nb)

        self.v_rdot, self.v_cdot = T_eul.rel_acc(dx_nb, w_nb, v_c)

    def __repr__(self) -> str:
        return f'{super().__repr__()} Kinematics'
    
    def ned_euler_vel_from_body(self):
        _dn = self.J@v_r
        return _dn

    def ned_euler_acc_from_body(self):
        _ddn = self.J@self.v_rdot + self.J_dot@v_r
        return _ddn
    
    def ned_quat_vel_from_body(self):
        _dn_q = self.Jq@v_r
        return _dn_q

    def ned_quat_acc_from_body(self):
        _ddn_q = self.Jq@v_r + self.Jq_dot@v_r
        return _ddn_q
    
    def body_vel_from_ned_euler(self):
        v = self.J_INV@dn
        return v
    
    def body_acc_from_ned_euler(self):
        dv = self.J_INV@(ddn - self.J_dot@self.J_INV@dn)
        return dv
    
    def body_vel_from_ned_quat(self):
        v = self.Jq_INV@dnq
        return v
    
    def body_acc_from_ned_quat(self):
        dv = self.Jq_INV@(ddnq - self.Jq_dot@self.Jq_INV@dnq)
        return dv
