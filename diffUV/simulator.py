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

"""This module contains a class for implementing time integration of the forward dynamics from 
fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from diffUV.kinematics import Kinematics as kin
from diffUV.utils.symbols import *
from diffUV.base import Base as dyn_body
from diffUV.utils.quaternion_ops import q2euler
import casadi as ca

class Simulator():
    def __init__(self):
        # ned kinematic transformation
        Kinematics = kin()
        self.Jq_ = Kinematics.Jq
        self.J_ = Kinematics.J
        # body representaion
        self.uv_body = dyn_body()

    def __repr__(self) -> str:
        return f'{super().__repr__()} Simulator'
    
    def model(self, rot_type='euler'):
        if rot_type == 'quat':
            xd = self.Jq_@x_nb
            xdd = self.uv_body.body_forward_dynamics()

            rhs = vertcat(xd, xdd) #the complete ODE vector
            f_rhs = ca.Function('Odefunc', [m, W, B, r_g, r_b, I_o,
                                    decoupled_added_m, coupled_added_m,
                                    linear_dc, quadratic_dc,
                                    x_nb, v_c, eul, uq, tau_b, z], [rhs])

            ode = f_rhs(m, W, B, r_g, r_b, I_o, decoupled_added_m,
                                coupled_added_m, linear_dc, quadratic_dc, x_nb, v_c, q2euler(uq), uq,  tau_b, z)

            xS0 = vertcat(p_n, uq, x_nb)

            # integrator to discretize the system
            sys = {}
            sys['x'] = xS0
            sys['u'] = tau_b
            sys['p'] = ode_p
            sys['ode'] = ode*dt # Time scaling

            intg = ca.integrator('intg', 'rk', sys, 0, 1 , {'simplify':True, 'number_of_finite_elements':5})

            res = intg(x0=xS0,u=tau_b, p=ode_p) #evaluate with symbols
            x_next = res['xf']

            x_next[3:7] = x_next[3:7]/ca.sqrt(x_next[3:7].T@x_next[3:7])  #quaternions requires normalization

            # x_next[9] = ca.if_else(x_next[2] < 0, 0,  x_next[9]) # if vehicle on surface, no more up motion
            # x_next[2] = ca.if_else(x_next[2] < 0, 0,  x_next[2]) # if vehicle on surface, keep on surface and not go up

        else:
            xd = self.J_@x_nb
            xdd = self.uv_body.body_forward_dynamics()

            ode = vertcat(xd, xdd) #the complete ODE vector

            xS0 = vertcat(n, x_nb) #states

            # integrator to discretize the system
            sys = {}
            sys['x'] = xS0
            sys['u'] = tau_b
            sys['p'] = ode_p
            sys['ode'] = ode*dt # Time scaling

            intg = ca.integrator('intg', 'rk', sys, 0, 1 , {'simplify':True, 'number_of_finite_elements':5})

            res = intg(x0=xS0,u=tau_b, p=ode_p) #evaluate with symbols
            x_next = res['xf']

            # x_next[8] = ca.if_else(x_next[2] < 0, 0,  x_next[8]) # if vehicle on surface, no more up motion
            # x_next[2] = ca.if_else(x_next[2] < 0, 0,  x_next[2]) # if vehicle on surface, keep on surface and not go up

        return x_next, xS0, tau_b, ode_p