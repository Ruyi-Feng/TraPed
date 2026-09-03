"""Winner-take-all Laplace NLL + mode classification + smoothness."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def multimodal_loss(
    out: dict,
    future: torch.Tensor,
    future_valid: torch.Tensor,
    *,
    cls_w: float = 0.5,
    smooth_w: float = 0.02,
) -> dict:
    traj = out["traj"]                 # [B, K, T, 2]
    scale = out["scale"]
    pi = out["pi"]
    gt = future.unsqueeze(1)           # [B, 1, T, 2]
    valid = future_valid.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
    denom = valid.sum(dim=(2, 3)).clamp_min(1.0)     # [B, 1]

    ade = ((traj - gt).abs() * valid).sum(dim=(2, 3)) / denom   # [B, K]
    winner = ade.argmin(dim=-1)
    b = torch.arange(traj.size(0), device=traj.device)
    w_traj = traj[b, winner]
    w_scale = scale[b, winner]
    nll = torch.log(2.0 * w_scale) + (future - w_traj).abs() / w_scale
    nll = (nll * future_valid.unsqueeze(-1)).sum() / future_valid.sum().clamp_min(1.0)

    cls = F.cross_entropy(pi, winner)
    acc = w_traj[:, 2:, :] - 2.0 * w_traj[:, 1:-1, :] + w_traj[:, :-2, :]
    smooth = (acc.pow(2) * future_valid[:, 2:].unsqueeze(-1)).mean()
    loss = nll + cls_w * cls + smooth_w * smooth
    return {
        "loss": loss,
        "nll": nll.detach(),
        "cls": cls.detach(),
        "smooth": smooth.detach(),
        "minade": ade.min(dim=-1).values.mean().detach(),
    }
