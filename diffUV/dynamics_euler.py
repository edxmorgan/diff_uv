from casadi import inv
from diffUV.base import Base
from diffUV.utils.symbol import *
import diffUV.utils.euler_ops as T_eul

class DynamicsEuler(Base):
    def __init__(self):
        super().__init__()
        self.J, R, T = T_eul.J_kin(eul)
        self.J_INV, _,_ = T_eul.inv_J_kin(eul)
        self.J_INV_T = self.J_INV.T
        self.state_vector = vertcat(n,dn)
        self.J_dot, _, _ = T_eul.J_dot(eul,deul,dT_sp,eul_sp,w_nb)

    def __repr__(self) -> str:
        """Euler representation of the Dynamics instance  in ned frame"""
        return f'{super().__repr__()} --> (euler in ned frame)'

    def get_ned_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments in ned"""
        M = self.get_body_inertia_matrix()
        M_ned = self.J_INV_T@M@self.J_INV
        return M_ned

    def coriolis_ned_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        C = self.coriolis_body_centripetal_matrix()
        M = self.get_body_inertia_matrix()
        C_ned = self.J_INV_T@(C - M@self.J_INV@self.J_dot)@self.J_INV
        return C_ned
    
    def gvect_ned(self):
        g = self.gvect_body()
        g_ned = self.J_INV_T@g
        return g_ned

    def damping_ned(self):
        D_v = self.damping_body()
        D = self.J_INV_T@D_v@self.J_INV
        return D
    
    def forward_dynamics_ned(self):
        ned_acc = inv(self.get_ned_inertia_matrix())@(self.J_INV_T@tau_body - self.coriolis_ned_centripetal_matrix()@dn - self.gvect_ned() -self.damping_ned()@dn)
        return ned_acc

    def inverse_dynamics_ned(self):
        resultant_torque = self.get_ned_inertia_matrix()@ddn + self.coriolis_ned_centripetal_matrix()@dn + self.gvect_ned() + self.damping_ned()@dn
        return resultant_torque