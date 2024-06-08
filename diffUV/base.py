"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion_control
"""
from casadi import SX, horzcat, inv, sin,cos, fabs, Function, diag, pinv,substitute
from platform import machine, system
import diffUV.utils.operators as ops
import diffUV.utils.transformation_matrix as T
from diffUV.utils.operators import Smtrx, coriolis_lag_param
from diffUV.utils.symbol import *


class Base(object):
    func_opts = {}
    jit_func_opts = {"jit": True, "jit_options": {"flags": "-Ofast"}}
    # OS/CPU dependent specification of compiler
    if system().lower() == "darwin" or machine().lower() == "aarch64":
        jit_func_opts["compiler"] = "shell"

    def __init__(self, func_opts=None, use_jit=True):
        if func_opts:
            self.func_opts = func_opts
        if use_jit:
            # NOTE: use_jit=True requires that CasADi is built with Clang
            for k, v in self.jit_func_opts.items():
                self.func_opts[k] = v
        self._initialize_inertia_matrix()

    def __repr__(self) -> str:
        return "differentiable underwater dynamics"
    
    def _initialize_inertia_matrix(self):
        """Internal method to compute the UV inertia matrix based on vehicle parameters."""
        M_rb = SX(6,6)
        S = Smtrx(r_g)
        M_rb[:3,:3] = m*SX.eye(3)
        M_rb[:3,3:] = -m*S
        M_rb[3:,:3] = m*S
        M_rb[3:,3:] = Ib_b
        __M = (M_rb + MA) 
        # apply symmetry considerations
        self.M = __M* sb_fft_config
        # apply yg= 0 and Ixy=Iyz=0
        self.M = substitute(self.M, y_g, SX(0))
        self.M = substitute(self.M, I_xy, SX(0))
        self.M = substitute(self.M, I_yz, SX(0))


    def get_body_inertia_matrix(self):
        """Compute and return the UV inertia matrix with configuration adjustments."""
        # M = Function("M", syms , [M], self.func_opts)
        return self.M
    
    def coriolis_body_centripetal_matrix(self):
        """Compute and return the Coriolis and centripetal matrix based on current vehicle state in body"""
        M = self.get_body_inertia_matrix()
        C_rb = M
        CA = coriolis_lag_param(MA, x_nb)
        C = C_rb+CA
        return C

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

    def damping_body(self):
        """Compute and return the total damping forces, including both linear and nonlinear components in body"""
        linear_damping = -diag(vertcat(X_u,Y_v,Z_w,K_p,M_q,N_r))
        nonlinear_damping = -diag(vertcat(X_uu,Y_vv,Z_ww,K_pp,M_qq,N_rr))@fabs(x_nb)
        D_v = linear_damping + nonlinear_damping
        return D_v

    def forward_dynamics_body(self):
        body_acc = inv(self.get_body_inertia_matrix())@(tau_body - self.coriolis_body_centripetal_matrix()@x_nb - self.gvect_body() -self.damping_body()@x_nb)
        return body_acc

    def inverse_dynamics_body(self):
        resultant_torque = self.get_body_inertia_matrix()@dx_nb + self.coriolis_body_centripetal_matrix()@x_nb + self.gvect_body() + self.damping_body()@x_nb
        return resultant_torque
    
    # def control_Allocation(self):
    #     u = inv(K)@pinv(T)@tau_body
    #     return u
    
    