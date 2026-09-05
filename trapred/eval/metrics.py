"""ADE / FDE / miss-rate on ego future xy (meters, ego frame)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _masked_ade_fde(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """pred [B, T, 2] or [B, K, T, 2]. Returns per-sample ADE, FDE."""
    if pred.ndim == 3:
        pred = pred[:, None]
    d = np.linalg.norm(pred - gt[:, None], axis=-1)
    v = valid[:, None]
    ade = (d * v).sum(axis=-1) / np.clip(v.sum(axis=-1), 1.0, None)
    t_last = (valid.cumsum(axis=-1) * valid).argmax(axis=-1).astype(np.int64)
    b = np.arange(gt.shape[0])
    k = np.arange(pred.shape[1])
    fde = d[b[:, None], k[None, :], t_last[:, None]]
    return ade, fde


def batch_metrics(out: dict, future: torch.Tensor, future_valid: torch.Tensor) -> Dict[str, float]:
    traj = out["traj"].detach().cpu().numpy()
    pi = out["pi"].detach().cpu().numpy()
    gt = future.detach().cpu().numpy()
    valid = (future_valid.detach().cpu().numpy() > 0.5).astype(np.float64)
    ade, fde = _masked_ade_fde(traj, gt, valid)
    k_ml = pi.argmax(axis=-1)
    b = np.arange(gt.shape[0])
    ranked = np.argsort(-pi, axis=-1)
    k3 = ranked[:, :min(3, ranked.shape[1])]
    k5 = ranked[:, :min(5, ranked.shape[1])]
    ade3 = np.take_along_axis(ade, k3, axis=1).min(axis=-1)
    fde3 = np.take_along_axis(fde, k3, axis=1).min(axis=-1)
    ade5 = np.take_along_axis(ade, k5, axis=1).min(axis=-1)
    fde5 = np.take_along_axis(fde, k5, axis=1).min(axis=-1)
    min_ade = ade.min(axis=-1)
    min_fde = fde.min(axis=-1)
    ml_ade = ade[b, k_ml]
    ml_fde = fde[b, k_ml]
    return {
        "minADE": float(min_ade.mean()),
        "minFDE": float(min_fde.mean()),
        "mlADE": float(ml_ade.mean()),
        "mlFDE": float(ml_fde.mean()),
        "top3ADE": float(ade3.mean()),
        "top3FDE": float(fde3.mean()),
        "top5ADE": float(ade5.mean()),
        "top5FDE": float(fde5.mean()),
        "selectionGapADE": float((ml_ade - min_ade).mean()),
        "selectionGapFDE": float((ml_fde - min_fde).mean()),
        "oracleModeRate": float((k_ml == ade.argmin(axis=-1)).mean()),
        "MR2": float((fde.min(axis=-1) > 2.0).mean()),
        "MR5": float((fde.min(axis=-1) > 5.0).mean()),
    }


def cv_metrics(cv_pred: torch.Tensor, future: torch.Tensor, future_valid: torch.Tensor) -> Dict[str, float]:
    pred = cv_pred.detach().cpu().numpy()
    gt = future.detach().cpu().numpy()
    valid = (future_valid.detach().cpu().numpy() > 0.5).astype(np.float64)
    ade, fde = _masked_ade_fde(pred, gt, valid)
    return {
        "cvADE": float(ade.mean()),
        "cvFDE": float(fde.mean()),
        "cvMR2": float((fde[:, 0] > 2.0).mean()),
    }
