import torch


def cross_pO(v: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric matrix for vector ``v``."""
    x, y, z = v
    return torch.tensor([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=v.dtype, device=v.device)
    
def coriolis_lag_param(M: torch.Tensor, x_nb: torch.Tensor) -> torch.Tensor:
    """Compute the Coriolis matrix using the lagrangian parameterization."""
    v_nb, w_nb = x_nb[:3], x_nb[3:]
    M11 = M[:3, :3]
    M12 = M[:3, 3:]
    M21 = M[3:, :3]
    M22 = M[3:, 3:]

    C = torch.zeros(6, 6, dtype=M.dtype, device=M.device)
    tmp_v = M11 @ v_nb + M12 @ w_nb
    tmp_w = M21 @ v_nb + M22 @ w_nb
    C[:3, 3:] = -cross_pO(tmp_v)
    C[3:, :3] = -cross_pO(tmp_v)
    C[3:, 3:] = -cross_pO(tmp_w)
    return C