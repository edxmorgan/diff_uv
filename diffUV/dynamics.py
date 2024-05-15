"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from diffUV.base import Base
from casadi import SX, horzcat, sin,cos
from diffUV.utils.symbol import *

class Dynamics(Base):
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return f'{super().__repr__()} Dynamics'

    def general_uv_inertia_matrix(self):
        self._M = SX(6, 6)
        self._M[0, :] = horzcat(m - X_du, -X_dv, -X_dw, -X_dp, m*z_g - X_dq, -m*y_g - X_dr)
        self._M[1, :] = horzcat(-X_dv, m-Y_dv, -Y_dw, -m*z_g-Y_dp, -Y_dq, m*x_g - Y_dr)
        self._M[2, :] = horzcat(-X_dw, -Y_dw, m - Z_dw, m*y_g - Z_dp, -m*x_g - Z_dq, -Z_dr)
        self._M[3, :] = horzcat(-X_dp, -m*z_g-Y_dp, m*y_g - Z_dp, I_x - K_dp, -I_yx - K_dq, -I_zx - K_dr)
        self._M[4, :] = horzcat(m*z_g - X_dq, -Y_dq, -m*x_g - Z_dq, -I_yx - K_dq, I_y - M_dq, -I_zy - M_dr)
        self._M[5, :] = horzcat(-m*y_g - X_dr, m*x_g - Y_dr, -Z_dr, -I_zx - K_dr, -I_zy - M_dr, I_z - N_dr)
    
    def UV_inertia_matrix(self):
        self.general_uv_inertia_matrix()
        M = self._M*star_board_config
        return M

    def gvect(self):
        # Hydrostatics of Submerged Vehicles
        # restoring forces
        g = SX(6, 1)
        g[0, 0] = (W - B)*sin(thet)
        g[1, 0] = -(W - B)*cos(thet)*sin(phi)
        g[2, 0] = -(W - B)*cos(thet)*cos(phi)
        g[3, 0] = -(y_g*W - y_b*B)*cos(thet)*cos(phi) + \
            (z_g*W - z_b*B)*cos(thet)*sin(phi)
        g[4, 0] = (z_g*W - z_b*B)*sin(thet) + (x_g*W - x_b*B)*cos(thet)*cos(phi)
        g[5, 0] = -(x_g*W - x_b*B)*cos(thet)*sin(phi) - (y_g*W - y_b*B)*sin(thet)
        # For neutrally buoyant vehicles W = B
        return g