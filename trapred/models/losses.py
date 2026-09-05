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
    endpoint_weight: float = 1.0,
    diversity_w: float = 0.0,
    soft_cls_temp: float = 0.0,
    winner_fde_weight: float = 0.0,
    diversity_margin: float = 2.0,
    kl_weight: float = 0.0,
    kl_free_bits: float = 0.0,
    confidence_regret_weight: float = 0.0,
) -> dict:
    traj = out["traj"]                 # [B, K, T, 2]
    scale = out["scale"]
    pi = out["pi"]
    gt = future.unsqueeze(1)           # [B, 1, T, 2]
    valid = future_valid.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
    valid_t = future_valid.unsqueeze(1)
    dist = torch.linalg.vector_norm(traj - gt, dim=-1)
    ade = (dist * valid_t).sum(dim=-1) / valid_t.sum(dim=-1).clamp_min(1.0)
    last_idx = (future_valid.cumsum(dim=-1) * future_valid).argmax(dim=-1)
    b = torch.arange(traj.size(0), device=traj.device)
    mode = torch.arange(traj.size(1), device=traj.device)
    fde = dist[b[:, None], mode[None, :], last_idx[:, None]]
    l1_ade = ((traj - gt).abs() * valid).sum(dim=(2, 3)) / valid.sum(
        dim=(2, 3)
    ).clamp_min(1.0)
    winner_cost = l1_ade + winner_fde_weight * fde
    winner = winner_cost.argmin(dim=-1)
    b = torch.arange(traj.size(0), device=traj.device)
    w_traj = traj[b, winner].float()
    # Floor scale in fp32 to keep Laplace NLL stable under AMP.
    w_scale = scale[b, winner].float().clamp_min(0.1)
    nll = torch.log(2.0 * w_scale) + (future.float() - w_traj).abs() / w_scale
    time_weight = torch.linspace(
        1.0, max(1.0, endpoint_weight), future.size(1),
        device=future.device, dtype=torch.float32,
    )
    weighted_valid = future_valid * time_weight.unsqueeze(0)
    nll = (nll * weighted_valid.unsqueeze(-1)).sum() / weighted_valid.sum().clamp_min(1.0)

    if soft_cls_temp > 0.0:
        target = torch.softmax(-winner_cost.detach() / soft_cls_temp, dim=-1)
        cls = -(target * F.log_softmax(pi, dim=-1)).sum(dim=-1).mean()
    else:
        cls = F.cross_entropy(pi, winner)
    acc = w_traj[:, 2:, :] - 2.0 * w_traj[:, 1:-1, :] + w_traj[:, :-2, :]
    smooth = (acc.pow(2) * future_valid[:, 2:].unsqueeze(-1)).mean()

    if traj.size(1) > 1:
        endpoints = traj[b[:, None], mode[None, :], last_idx[:, None]]
        pair_dist = torch.cdist(endpoints, endpoints)
        off_diag = ~torch.eye(traj.size(1), dtype=torch.bool, device=traj.device)
        diversity = F.relu(diversity_margin - pair_dist[:, off_diag]).mean()
    else:
        diversity = traj.new_zeros(())
    confidence_regret = (
        torch.softmax(pi, dim=-1) * winner_cost.detach()
    ).sum(dim=-1).mean()
    kl_per_dim = out.get("latent_kl_per_dim")
    if kl_per_dim is None:
        kl = traj.new_zeros(())
    else:
        kl = kl_per_dim.float().clamp_min(float(kl_free_bits)).sum(dim=-1).mean()
    loss = (
        nll
        + cls_w * cls
        + smooth_w * smooth
        + diversity_w * diversity
        + kl_weight * kl
        + confidence_regret_weight * confidence_regret
    )
    return {
        "loss": loss,
        "nll": nll.detach(),
        "cls": cls.detach(),
        "smooth": smooth.detach(),
        "diversity": diversity.detach(),
        "kl": kl.detach(),
        "confidence_regret": confidence_regret.detach(),
        "minade": ade.min(dim=-1).values.mean().detach(),
    }
