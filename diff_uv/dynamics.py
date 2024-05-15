"""This module contains a class for implementing fossen_thor_i_handbook_of_marine_craft_hydrodynamics_and_motion
"""
from casadi import SX, inv, vertcat, horzcat, Function, integrator, sin, cos, diag, skew, substitute, DM
from platform import machine, system

class Dynamics(object):
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