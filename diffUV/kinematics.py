"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from diffUV.base import Base
from diffUV.utils import euler_ops as T_eul
from diffUV.utils import quaternion_ops as T_quat
from diffUV.utils.symbol import *

class Kinematics():
    def __init__(self):
        self.J, self.R, self.T = T_eul.J_kin(eul)
        self.J_inv, self.R_inv, self.T_inv = T_eul.inv_J_kin(eul)
        self.J_dot, _, _ = T_eul.J_dot(eul,deul,dT_sp,eul_sp,w_nb)
        self.ned_state_vector = vertcat(n,dn)

    def __repr__(self) -> str:
        return f'{super().__repr__()} Kinematics'
    
    def ned_vel(self):
        _dn = self.J@(x_nb)
        return _dn

    def ned_acc(self):
        _ddn = self.J@dx_nb + self.J_dot@x_nb
        return _ddn
    
    def body_position(self):
        v = self.J_inv@dn
        return v
    
    def body_vel(self):
        dv = self.J_inv@(ddn - self.J_dot@self.J_inv@dn)
        return dv
