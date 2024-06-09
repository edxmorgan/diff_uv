from casadi import inv
from diffUV.base import Base
from diffUV.utils.symbol import *
import diffUV.utils.quaternion_ops as Tquat

class DynamicsQuat(Base):
    def __init__(self):
        super().__init__()
        self.Jq_INV, _,_ = Tquat.inv_Jq_kin(uq)
        self.Jq_INV_T = self.Jq_INV.T
        # self.state_vector = vertcat(uq,..)

    def __repr__(self) -> str:
        """Quaternion representation of the Dynamics instance in ned frame"""
        return f'{super().__repr__()} --> (quat in ned frame)'
    
    def quat_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments in ned for quaternion"""
        M = self.body_inertia_matrix()
        M_ned_q = self.Jq_INV_T@M@self.Jq_INV
        return M_ned_q

    # def quat_coriolis_ned_centripetal_matrix_quat(self):
    #     """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
    #     C = self.body_coriolis_centripetal_matrix()
    #     M = self.body_inertia_matrix()
    #     C_ned_q = self.Jq_INV_T@(C - M@self.Jq_INV@self.Jq_dot)@self.Jq_INV
    #     return C_ned_q
    
    def gvect_ned_quat(self):
        g = self.body_restoring_vector()
        g_ned = self.Jq_INV_T@g
        return g_ned

    def damping_ned_quat(self):
        D_v = self.body_damping_matrix()
        D = self.Jq_INV_T@D_v@self.Jq_INV
        return D

    #####incorrect formulations
    # def forward_dynamics_ned_quat(self):
    #     ned_acc_quat = inv(self.get_ned_inertia_matrix_quat())@(self.Jq_INV_T@tau_body - self.coriolis_ned_centripetal_matrix_quat()@x_nb - self.gvect_ned_quat() -self.damping_ned_quat()@x_nb)
    #     return ned_acc_quat
    
    # def inverse_dynamics_ned_quat(self):
    #     resultant_torque = self.get_body_inertia_matrix()@ddn + self.coriolis_body_centripetal_matrix()@dn + self.gvect_body() + self.damping_body()@dn
    #     return resultant_torque