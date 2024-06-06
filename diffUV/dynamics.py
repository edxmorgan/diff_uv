from casadi import SX, horzcat, inv, sin,cos, fabs, Function, diag, pinv
from diffUV.base import Base
from diffUV.utils.symbol import *
from diffUV.utils.operators import cross_product
import diffUV.utils.transformation_matrix as T_eul
import diffUV.utils.dual_quaternion as Tquat

class Dynamics(Base):
    def __init__(self):
        super().__init__()
        self._M = None
        self.J_INV, _,_ = T_eul.inv_J_kin(phi, thet, psi)
        self.J_INV_T = self.J_INV.T

        self.Jq_INV, _,_ = Tquat.inv_Jq_kin(uq)
        self.Jq_INV_T = self.J_INV.T

        self.state_vector = vertcat(n,dn)


    def __repr__(self) -> str:
        """String representation of the Dynamics instance."""
        return f'{super().__repr__()} Dynamics'

    def _initialize_inertia_matrix(self):
        """Internal method to compute the UV inertia matrix based on vehicle parameters."""
        self._M = SX(6, 6)
        self._M[0, :] = horzcat(
            m - X_du, -X_dv, -X_dw, -X_dp, m*z_g - X_dq, -m*y_g - X_dr)
        self._M[1, :] = horzcat(-X_dv, m-Y_dv, -Y_dw, -
                                m*z_g-Y_dp, -Y_dq, m*x_g - Y_dr)
        self._M[2, :] = horzcat(-X_dw, -Y_dw, m - Z_dw,
                                m*y_g - Z_dp, -m*x_g - Z_dq, -Z_dr)
        self._M[3, :] = horzcat(-X_dp, -m*z_g-Y_dp, m*y_g -
                                Z_dp, I_x - K_dp, -I_yx - K_dq, -I_zx - K_dr)
        self._M[4, :] = horzcat(
            m*z_g - X_dq, -Y_dq, -m*x_g - Z_dq, -I_yx - K_dq, I_y - M_dq, -I_zy - M_dr)
        self._M[5, :] = horzcat(-m*y_g - X_dr, m*x_g -
                                Y_dr, -Z_dr, -I_zx - K_dr, -I_zy - M_dr, I_z - N_dr)

    def get_body_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments."""
        self._initialize_inertia_matrix()
        # syms = [q] 
        M = self._M * star_board_config
        # M = Function("M", syms , [M], self.func_opts)
        return M

    def get_ned_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments in ned"""
        M = self.get_body_inertia_matrix()
        M_ned = self.J_INV_T@M@self.J_INV
        return M_ned
    
    def get_ned_inertia_matrix_quat(self):
        """Compute and return the UV inertia matrix with configuration adjustments in ned for quaternion"""
        M = self.get_body_inertia_matrix()
        M_ned_q = self.Jq_INV_T@M@self.Jq_INV
        return M_ned_q

    def coriolis_body_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        M = self.get_body_inertia_matrix()
        M11 = M[:3, :3]
        M12 = M[:3, 3:]
        M21 = M[3:, :3]
        M22 = M[3:, 3:]
        C = SX.zeros(6, 6)
        C[3:, :3] = -cross_product(M11@v_nb + M12@w_nb)
        C[:3, 3:] = -cross_product(M11@v_nb + M12@w_nb)
        C[3:, 3:] = -cross_product(M21@v_nb + M22@w_nb)
        return C

    def coriolis_ned_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        C = self.coriolis_body_centripetal_matrix()
        M = self.get_body_inertia_matrix()
        C_ned = self.J_INV_T@(C - M@self.J_INV@self.J_dot)@self.J_INV
        return C_ned

    def coriolis_ned_centripetal_matrix_quat(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        C = self.coriolis_body_centripetal_matrix()
        M = self.get_body_inertia_matrix()
        C_ned_q = self.Jq_INV_T@(C - M@self.Jq_INV@self.Jq_dot)@self.Jq_INV
        return C_ned_q
    
    def gvect_body(self):
        """Compute and return the hydrostatic restoring forces."""
        g = SX(6, 1)
        g[0, 0] = (W - B)*sin(thet)
        g[1, 0] = -(W - B)*cos(thet)*sin(phi)
        g[2, 0] = -(W - B)*cos(thet)*cos(phi)
        g[3, 0] = -(y_g*W - y_b*B)*cos(thet)*cos(phi) + \
            (z_g*W - z_b*B)*cos(thet)*sin(phi)
        g[4, 0] = (z_g*W - z_b*B)*sin(thet) + \
            (x_g*W - x_b*B)*cos(thet)*cos(phi)
        g[5, 0] = -(x_g*W - x_b*B)*cos(thet) * \
            sin(phi) - (y_g*W - y_b*B)*sin(thet)
        # For neutrally buoyant vehicles W = B
        return g
    
    def gvect_ned(self):
        g = self.gvect_body()
        g_ned = self.J_INV_T@g
        return g_ned
    
    def gvect_ned_quat(self):
        g = self.gvect_body()
        g_ned = self.Jq_INV_T@g
        return g_ned

    # def gvect_quat(self):
    #     """Compute and return the hydrostatic restoring forces using quaternions."""
    #     g_quat = SX(6, 1)
    #     g_quat[0, 0] = (B-W)*(2*eps1*eps3 - 2*eps2*eta)
    #     g_quat[1, 0] = (B-W)*(2*eps2*eps3 - 2*eps1*eta)
    #     g_quat[2, 0] = (W-B)*(2*eps1**2 - 2*eps2**2 -1)
    #     g_quat[3, 0] = z_g*W*(2*eps2*eps3 + 2*eps1*eta)
    #     g_quat[4, 0] = z_g*W*(2*eps1*eps3 + 2*eps2*eta)
    #     g_quat[5, 0] = 0
    #     # For neutrally buoyant vehicles W = B
    #     return g_quat


    def damping_body(self):
        """Compute and return the total damping forces, including both linear and nonlinear components in body"""
        linear_damping = -diag(vertcat(X_u,Y_v,Z_w,K_p,M_q,N_r))
        nonlinear_damping = -diag(vertcat(X_uu,Y_vv,Z_ww,K_pp,M_qq,N_rr))@fabs(x_nb)
        D_v = linear_damping + nonlinear_damping
        return D_v
    
    def damping_ned_quat(self):
        D_v = self.damping_body()
        D = self.Jq_INV_T@D_v@self.Jq_INV
        return D

    def damping_ned(self):
        D_v = self.damping_body()
        D = self.J_INV_T@D_v@self.J_INV
        return D

    def forward_dynamics_body(self):
        body_acc = inv(self.get_body_inertia_matrix())@(tau_body - self.coriolis_body_centripetal_matrix()@x_nb - self.gvect_body() -self.damping_body()@x_nb)
        return body_acc
    
    def forward_dynamics_ned(self):
        ned_acc = inv(self.get_ned_inertia_matrix())@(self.J_INV_T@tau_body - self.coriolis_ned_centripetal_matrix()@dn - self.gvect_ned() -self.damping_ned()@dn)
        return ned_acc
    
    # def forward_dynamics_ned_quat(self):
    #     ned_acc_quat = inv(self.get_ned_inertia_matrix_quat())@(self.Jq_INV_T@tau_body - self.coriolis_ned_centripetal_matrix_quat()@x_nb - self.gvect_ned_quat() -self.damping_ned_quat()@x_nb)
    #     return ned_acc_quat
    
    def inverse_dynamics_body(self):
        resultant_torque = self.get_body_inertia_matrix()@dx_nb + self.coriolis_body_centripetal_matrix()@x_nb + self.gvect_body() + self.damping_body()@x_nb
        return resultant_torque

    def inverse_dynamics_ned(self):
        resultant_torque = self.get_body_inertia_matrix()@ddn + self.coriolis_body_centripetal_matrix()@dn + self.gvect_body() + self.damping_body()@dn
        return resultant_torque
    
    def control_Allocation(self):
        u = inv(K)@pinv(T)@tau_body
        return u