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
        ode_p = ca.vertcat(sim_p, dt, f_ext)
        xdd = self.uv_body.body_forward_dynamics()
        if rot_type == 'quat':
            xd = self.Jq_@x_nb
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

            intg = ca.integrator('intg', 'rk', sys, 0, 1 , {'simplify':True, 'number_of_finite_elements':3})

            res = intg(x0=xS0,u=tau_b, p=ode_p) #evaluate with symbols
            x_next = res['xf']

            x_next[3:7] = x_next[3:7]/ca.sqrt(x_next[3:7].T@x_next[3:7])  #quaternions requires normalization
        else:
            xd = self.J_@x_nb
            ode = vertcat(xd, xdd) #the complete ODE vector
            xS0 = vertcat(n, x_nb) #states

            # integrator to discretize the system
            sys = {}
            sys['x'] = xS0
            sys['u'] = tau_b
            sys['p'] = ode_p
            sys['ode'] = ode*dt # Time scaling

            intg = ca.integrator('intg', 'rk', sys, 0, 1 , {'simplify':True, 'number_of_finite_elements':3})

            res = intg(x0=xS0,u=tau_b, p= ode_p) #evaluate with symbols
            x_next = res['xf']

        return x_next, xS0, tau_b, sim_p, dt, f_ext
    
    def model_reg(self, rot_type='euler'):
        xdd, lumped_params = self.uv_body.body_forward_dynamics_reg()
        sim_p = vertcat(lumped_params, v_c)
        ode_p = ca.vertcat(sim_p, dt, f_ext)
        if rot_type == 'quat':
            xd = self.Jq_@x_nb
            rhs = vertcat(xd, xdd) #the complete ODE vector
            f_rhs = ca.Function('Odefunc', [*lumped_params,
                                    x_nb, v_c, eul, uq, tau_b, z], [rhs])
            ode = f_rhs(*lumped_params, x_nb, v_c, q2euler(uq), uq,  tau_b, z)
            xS0 = vertcat(p_n, uq, x_nb)

            # integrator to discretize the system
            sys = {}
            sys['x'] = xS0
            sys['u'] = tau_b
            sys['p'] = ode_p
            sys['ode'] = ode*dt # Time scaling

            intg = ca.integrator('intg', 'rk', sys, 0, 1 , {'simplify':True, 'number_of_finite_elements':30})

            res = intg(x0=xS0,u=tau_b, p=ode_p) #evaluate with symbols
            x_next = res['xf']

            x_next[3:7] = x_next[3:7]/ca.sqrt(x_next[3:7].T@x_next[3:7])  #quaternions requires normalization
        else:
            xd = self.J_@x_nb
            ode = vertcat(xd, xdd) #the complete ODE vector

            xS0 = vertcat(n, x_nb) #states

            # integrator to discretize the system
            sys = {}
            sys['x'] = xS0
            sys['u'] = tau_b
            sys['p'] = ode_p
            sys['ode'] = ode*dt # Time scaling

            intg = ca.integrator('intg', 'rk', sys, 0, 1 , {'simplify':True, 'number_of_finite_elements':30})

            res = intg(x0=xS0,u=tau_b, p= ode_p) #evaluate with symbols
            x_next = res['xf']

        return x_next, xS0, tau_b, sim_p, dt, f_ext
    

