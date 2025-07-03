from dataclasses import dataclass
from typing import Optional

import casadi as ca

from diffUV.utils.operators import cross_pO, coriolis_lag_param
from .torch_dynamics import VehicleParams


class CasadiDynamics:
    """CasADi-based implementation of the underwater vehicle dynamics."""

    def __init__(self, params: VehicleParams):
        self.mass = params.mass
        self.inertia = ca.DM(params.inertia)
        self.cog = ca.DM(params.cog)
        self.cob = ca.DM(params.cob)
        self.added_mass = ca.DM(params.added_mass)
        self.linear_damping = ca.DM(params.linear_damping)
        self.quadratic_damping = ca.DM(params.quadratic_damping)
        self.weight = params.weight
        self.buoyancy = params.buoyancy

    def _inertia_matrix(self) -> ca.DM:
        S = cross_pO(self.cog)
        M_rb = ca.DM.zeros(6, 6)
        M_rb[:3, :3] = self.mass * ca.DM.eye(3)
        M_rb[:3, 3:] = -self.mass * S
        M_rb[3:, :3] = self.mass * S
        M_rb[3:, 3:] = self.inertia
        return M_rb + self.added_mass

    def body_inertia_matrix(self) -> ca.DM:
        return self._inertia_matrix()

    def body_coriolis_centripetal_matrix(self, v_r: ca.DM) -> ca.DM:
        M = self.body_inertia_matrix()
        return coriolis_lag_param(M, v_r)

    def body_restoring_vector(self, angles: ca.DM) -> ca.DM:
        phi, thet, _ = angles
        W = self.weight
        B = self.buoyancy
        x_g, y_g, z_g = self.cog
        x_b, y_b, z_b = self.cob

        g = ca.DM.zeros(6, 1)
        g[0] = (W - B) * ca.sin(thet)
        g[1] = -(W - B) * ca.cos(thet) * ca.sin(phi)
        g[2] = -(W - B) * ca.cos(thet) * ca.cos(phi)
        g[3] = -(y_g * W - y_b * B) * ca.cos(thet) * ca.cos(phi) + (z_g * W - z_b * B) * ca.cos(thet) * ca.sin(phi)
        g[4] = (z_g * W - z_b * B) * ca.sin(thet) + (x_g * W - x_b * B) * ca.cos(thet) * ca.cos(phi)
        g[5] = -(x_g * W - x_b * B) * ca.cos(thet) * ca.sin(phi) - (y_g * W - y_b * B) * ca.sin(thet)
        return g

    def body_damping_matrix(self, v_r: ca.DM) -> ca.DM:
        lin = -ca.diag(self.linear_damping)
        non = -ca.diag(self.quadratic_damping * ca.fabs(v_r))
        return lin + non

    def body_forward_dynamics(
        self,
        v_r: ca.DM,
        tau: ca.DM,
        angles: ca.DM,
        f_ext: Optional[ca.DM] = None,
    ) -> ca.DM:
        if f_ext is None:
            f_ext = ca.DM.zeros(6, 1)
        M = self.body_inertia_matrix()
        C = self.body_coriolis_centripetal_matrix(v_r) @ v_r
        D = self.body_damping_matrix(v_r) @ v_r
        g = self.body_restoring_vector(angles)
        rhs = tau + f_ext - C - D - g
        acc = ca.solve(M, rhs)
        return acc
