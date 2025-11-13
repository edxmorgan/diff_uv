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

"""This module contains a class for implementing a MIMO Nonlinear Controllers
"""
from diffUV.kinematics import Kinematics as kin
from diffUV.base import Base as dyn_body
from diffUV.utils.symbols import *
import casadi as ca

class Controller():
    def __init__(self):
        # ned kinematic transformation
        Kinematics = kin()
        self.J_ = Kinematics.J
        # body representaion
        uv_body = dyn_body()
        mB = uv_body.surface_interaction(z, W, B, B_eps=3.0)
        self.gn = uv_body.body_restoring_vector(mB)

    def __repr__(self) -> str:
        return f'{super().__repr__()} Simulator'
    
    def position_pid(self):
        ne = nd - n

        i_buffer = sum_e_buffer + ne*dt

        pid = diag(Kp)@ne + diag(Ki)@i_buffer - diag(Kd)@(self.J_@x_nb)

        pid_controller = self.gn + self.J_.T@pid

        return pid_controller, i_buffer