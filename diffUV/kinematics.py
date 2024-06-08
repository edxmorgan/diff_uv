"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from diffUV.base import Base
from diffUV.utils import transformation_matrix as Tm
from diffUV.utils.symbol import *

class Kinematics(Base):
    def __init__(self):
        self.J, self.R, self.T = Tm.J_kin(phi, thet, psi)
        self.J_inv, self.R_inv, self.T_inv = Tm.inv_J_kin(phi, thet, psi)
        self.dR = Tm.T_diff(self.R, v_nb)
        self.dT = Tm.T_diff(self.T, w_nb)
        self.dJ = SX.zeros(6, 6)
        self.dJ[:3,:3] = self.dR
        self.dJ[3:,3:] = self.dT

    def __repr__(self) -> str:
        return f'{super().__repr__()} Kinematics'
    
    def ned_vel(self):
        _dn = self.J@(x_nb)
        return _dn

    def ned_acc(self):
        _ddn = self.J@dx_nb + self.dJ@x_nb
        return _ddn
    
    def body_position(self):
        v = self.J_inv@dn
        return v
    
    def body_vel(self):
        dv = self.J_inv@(ddn - self.dJ@self.J_inv@dn)
        return dv
