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

from casadi import inv
from diffUV.base import Base
from diffUV.utils.symbols import *
from diffUV.utils import euler_ops as T_eul

class DynamicsEuler(Base):
    def __init__(self):
        super().__init__()
        self.J, self.R, self.T = T_eul.J_kin(eul)
        self.J_INV, self.R_INV, self.T_INV = T_eul.inv_J_kin(eul)
        self.J_INV_T = self.J_INV.T
        self.J_dot, self.dR, self.dT = T_eul.J_dot(eul,deul,dT_sp,eul_sp,w_nb)

    def __repr__(self) -> str:
        """Euler representation of the Dynamics instance  in ned frame"""
        return f'{super().__repr__()} --> (euler in ned frame)'

    def ned_euler_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments in ned"""
        M = self.body_inertia_matrix()
        M_ned = self.J_INV_T@M@self.J_INV
        return M_ned

    def ned_euler_coriolis_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        C = self.body_coriolis_centripetal_matrix()
        M = self.body_inertia_matrix()
        C_ned = self.J_INV_T@(C - M@self.J_INV@self.J_dot)@self.J_INV
        return C_ned
    
    def ned_euler_restoring_vector(self):
        g = self.body_restoring_vector()
        g_ned = self.J_INV_T@g
        return g_ned

    def ned_euler_damping(self):
        D_v = self.body_damping_matrix()
        D = self.J_INV_T@D_v@self.J_INV
        return D