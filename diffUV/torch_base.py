from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from diffUV.utils.torch_ops import cross_pO, coriolis_lag_param


@dataclass
class VehicleParams:
    """Container for vehicle parameters."""

    mass: float
    inertia: torch.Tensor  # 3x3 tensor
    cog: torch.Tensor  # 3 vector
    cob: torch.Tensor  # 3 vector
    added_mass: torch.Tensor = torch.zeros(6, 6)
    linear_damping: torch.Tensor = torch.zeros(6)
    quadratic_damping: torch.Tensor = torch.zeros(6)
    weight: float = 0.0
    buoyancy: float = 0.0


class VehicleTorchDynamics(nn.Module):
    """PyTorch implementation of the underwater vehicle dynamics.
    Parameters are stored as :class:`~torch.nn.Parameter` so they can be
    optimized online.  The ``trainable`` flag controls whether gradients are
    collected for these parameters.
    """

    def __init__(self, params: VehicleParams, trainable: bool = False):
        super().__init__()

        dtype = params.inertia.dtype
        device = params.inertia.device

        self.mass = nn.Parameter(torch.tensor(params.mass, dtype=dtype, device=device), requires_grad=trainable)
        self.inertia = nn.Parameter(params.inertia.clone().to(device), requires_grad=trainable)
        self.cog = nn.Parameter(params.cog.clone().to(device), requires_grad=trainable)
        self.cob = nn.Parameter(params.cob.clone().to(device), requires_grad=trainable)
        self.added_mass = nn.Parameter(params.added_mass.clone().to(device), requires_grad=trainable)
        self.linear_damping = nn.Parameter(params.linear_damping.clone().to(device), requires_grad=trainable)
        self.quadratic_damping = nn.Parameter(params.quadratic_damping.clone().to(device), requires_grad=trainable)
        self.weight = nn.Parameter(torch.tensor(params.weight, dtype=dtype, device=device), requires_grad=trainable)
        self.buoyancy = nn.Parameter(torch.tensor(params.buoyancy, dtype=dtype, device=device), requires_grad=trainable)

    @property
    def params(self) -> VehicleParams:
        """Return the current parameters as a :class:`VehicleParams` dataclass."""
        return VehicleParams(
            mass=float(self.mass.detach()),
            inertia=self.inertia.detach().clone(),
            cog=self.cog.detach().clone(),
            cob=self.cob.detach().clone(),
            added_mass=self.added_mass.detach().clone(),
            linear_damping=self.linear_damping.detach().clone(),
            quadratic_damping=self.quadratic_damping.detach().clone(),
            weight=float(self.weight.detach()),
            buoyancy=float(self.buoyancy.detach()),
        )

    def update_params(self, params: VehicleParams) -> None:
        """Load new parameter values inplace."""
        with torch.no_grad():
            self.mass.copy_(torch.tensor(params.mass, dtype=self.mass.dtype, device=self.mass.device))
            self.inertia.copy_(params.inertia.to(self.inertia.device))
            self.cog.copy_(params.cog.to(self.cog.device))
            self.cob.copy_(params.cob.to(self.cob.device))
            self.added_mass.copy_(params.added_mass.to(self.added_mass.device))
            self.linear_damping.copy_(params.linear_damping.to(self.linear_damping.device))
            self.quadratic_damping.copy_(params.quadratic_damping.to(self.quadratic_damping.device))
            self.weight.copy_(torch.tensor(params.weight, dtype=self.weight.dtype, device=self.weight.device))
            self.buoyancy.copy_(torch.tensor(params.buoyancy, dtype=self.buoyancy.dtype, device=self.buoyancy.device))
            
    def _initialize_inertia_matrix(self) -> torch.Tensor:
        S = cross_pO(self.cog)
        dtype = self.inertia.dtype
        device = self.inertia.device
        M_rb = torch.zeros(6, 6, dtype=dtype, device=device)
        M_rb[:3, :3] = self.mass * torch.eye(3, dtype=dtype, device=device)
        M_rb[:3, 3:] = -self.mass * S
        M_rb[3:, :3] = self.mass * S
        M_rb[3:, 3:] = self.inertia
        return M_rb + self.added_mass

    def body_inertia_matrix(self) -> torch.Tensor:
        return self._initialize_inertia_matrix()
    
    def body_coriolis_centripetal_matrix(self, v_r: torch.Tensor) -> torch.Tensor:
        M = self.body_inertia_matrix()
        return coriolis_lag_param(M, v_r)