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
from diffUV.utils import quaternion_ops as T_quat

class DynamicsQuat(Base):
    def __init__(self):
        super().__init__()
        self.Jq, self.Rq, self.Tq = T_quat.Jq_kin(uq)
        self.Jq_INV, _,_ = T_quat.inv_Jq_kin(uq)
        self.Jq_INV_T = self.Jq_INV.T
        self.Jq_dot, self.dRq ,self.dTq = T_quat.Jq_dot(uq, w_nb)
        # self.state_vector = vertcat(uq,..)

    def __repr__(self) -> str:
        """Quaternion representation of the Dynamics instance in ned frame"""
        return f'{super().__repr__()} --> (quat in ned frame)'
    
    def ned_quat_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments in ned for quaternion"""
        M = self.body_inertia_matrix()
        M_ned_q = self.Jq_INV_T@M@self.Jq_INV
        return M_ned_q

    def ned_quat_coriolis_ned_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        C = self.body_coriolis_centripetal_matrix()
        M = self.body_inertia_matrix()
        C_ned_q = self.Jq_INV_T@(C - M@self.Jq_INV@self.Jq_dot)@self.Jq_INV
        return C_ned_q
    
    def ned_quat_restoring_vector(self):
        g = self.body_restoring_vector()
        g_ned = self.Jq_INV_T@g
        return g_ned

    def ned_quat_damping(self):
        D_v = self.body_damping_matrix()
        D = self.Jq_INV_T@D_v@self.Jq_INV
        return D