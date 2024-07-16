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
from diffUV.base import Base
from diffUV.utils import euler_ops as T_eul
from diffUV.utils import quaternion_ops as T_quat
from diffUV.utils.symbols import *

class Kinematics():
    def __init__(self):
        self.J, self.R, self.T = T_eul.J_kin(eul)
        self.J_INV, self.R_INV, self.T_INV = T_eul.inv_J_kin(eul)
        self.J_dot, self.dR, self.dT = T_eul.J_dot(eul,deul,dT_sp,eul_sp,w_nb)
        self.ned_state_vector = vertcat(n,dn)

        self.Jq, self.Rq, self.Tq = T_quat.Jq_kin(uq)
        self.Jq_INV, _,_ = T_quat.inv_Jq_kin(uq)
        self.Jq_INV_T = self.Jq_INV.T
        self.Jq_dot, self.dRq ,self.dTq = T_quat.Jq_dot(uq, w_nb)
        
        self.v_rdot = T_eul.rel_acc(dx_nb, w_nb, v_c)

    def __repr__(self) -> str:
        return f'{super().__repr__()} Kinematics'
    
    def ned_euler_vel(self):
        _dn = self.J@v_r
        return _dn

    def ned_euler_acc(self):
        _ddn = self.J@self.v_rdot + self.J_dot@v_r
        return _ddn
    
    def ned_quat_vel(self):
        _dn_q = self.Jq@v_r
        return _dn_q

    def ned_quat_acc(self):
        _ddn_q = self.Jq@v_r + self.Jq_dot@v_r
        return _ddn_q
    
    def body_position_from_euler(self):
        v = self.J_INV@dn
        return v
    
    def body_vel_from_euler(self):
        dv = self.J_INV@(ddn - self.J_dot@self.J_INV@dn)
        return dv
    
    def body_position_from_quat(self):
        v = self.Jq_INV@dn
        return v
    
    def body_vel_from_quat(self):
        dv = self.Jq_INV@(ddn - self.Jq_dot@self.Jq_INV@dn)
        return dv
