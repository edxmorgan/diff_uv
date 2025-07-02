from dataclasses import dataclass
import torch

from diffUV.utils.torch_ops import cross_pO, coriolis_lag_param


@dataclass
class VehicleParams:
    mass: float
    inertia: torch.Tensor  # 3x3 tensor
    cog: torch.Tensor  # 3 vector
    cob: torch.Tensor  # 3 vector
    added_mass: torch.Tensor = torch.zeros(6, 6)
    linear_damping: torch.Tensor = torch.zeros(6)
    quadratic_damping: torch.Tensor = torch.zeros(6)
    weight: float = 0.0
    buoyancy: float = 0.0


class TorchDynamics:
    """PyTorch implementation of the underwater vehicle dynamics."""

    def __init__(self, params: VehicleParams):
        self.p = params
        self.M = self._initialize_inertia_matrix()

    def _initialize_inertia_matrix(self) -> torch.Tensor:
        S = cross_pO(self.p.cog)
        M_rb = torch.zeros(6, 6, dtype=self.p.inertia.dtype)
        M_rb[:3, :3] = self.p.mass * torch.eye(3, dtype=self.p.inertia.dtype)
        M_rb[:3, 3:] = -self.p.mass * S
        M_rb[3:, :3] = self.p.mass * S
        M_rb[3:, 3:] = self.p.inertia
        return M_rb + self.p.added_mass

    def body_inertia_matrix(self) -> torch.Tensor:
        return self.M

    def body_coriolis_centripetal_matrix(self, v_r: torch.Tensor) -> torch.Tensor:
        return coriolis_lag_param(self.M, v_r)

    def body_restoring_vector(self, angles: torch.Tensor) -> torch.Tensor:
        phi, thet, _ = angles
        W = self.p.weight
        B = self.p.buoyancy
        x_g, y_g, z_g = self.p.cog
        x_b, y_b, z_b = self.p.cob

        g = torch.zeros(6, dtype=angles.dtype)
        g[0] = (W - B) * torch.sin(thet)
        g[1] = -(W - B) * torch.cos(thet) * torch.sin(phi)
        g[2] = -(W - B) * torch.cos(thet) * torch.cos(phi)
        g[3] = -(y_g * W - y_b * B) * torch.cos(thet) * torch.cos(phi) + (
            z_g * W - z_b * B
        ) * torch.cos(thet) * torch.sin(phi)
        g[4] = (z_g * W - z_b * B) * torch.sin(thet) + (
            x_g * W - x_b * B
        ) * torch.cos(thet) * torch.cos(phi)
        g[5] = -(x_g * W - x_b * B) * torch.cos(thet) * torch.sin(phi) - (
            y_g * W - y_b * B
        ) * torch.sin(thet)
        return g

    def body_damping_matrix(self, v_r: torch.Tensor) -> torch.Tensor:
        lin = -torch.diag(self.p.linear_damping)
        non = -torch.diag(self.p.quadratic_damping * torch.abs(v_r))
        return lin + non

    def body_forward_dynamics(
        self,
        v_r: torch.Tensor,
        tau: torch.Tensor,
        angles: torch.Tensor,
        f_ext: torch.Tensor = None,
    ) -> torch.Tensor:
        if f_ext is None:
            f_ext = torch.zeros_like(v_r)
        M = self.body_inertia_matrix()
        C = self.body_coriolis_centripetal_matrix(v_r) @ v_r
        D = self.body_damping_matrix(v_r) @ v_r
        g = self.body_restoring_vector(angles)
        rhs = tau + f_ext - C - D - g
        acc = torch.linalg.solve(M, rhs)
        return acc

    def body_inverse_dynamics(
        self,
        desired_acc: torch.Tensor,
        v_r: torch.Tensor,
        angles: torch.Tensor,
        f_ext: torch.Tensor = None,
    ) -> torch.Tensor:
        if f_ext is None:
            f_ext = torch.zeros_like(v_r)
        M = self.body_inertia_matrix()
        C = self.body_coriolis_centripetal_matrix(v_r) @ v_r
        D = self.body_damping_matrix(v_r) @ v_r
        g = self.body_restoring_vector(angles)
        tau = M @ desired_acc + C + D + g - f_ext
        return tau
