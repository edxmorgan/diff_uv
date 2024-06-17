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