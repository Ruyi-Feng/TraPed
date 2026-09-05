"""Render one test sample as a two-panel multi-trajectory comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trapred.config import load_config
from trapred.data.dataset import SceneDataset, build_cache
from trapred.data.windows import IDX_VALID
from trapred.eval.metrics import _masked_ade_fde
from trapred.models.factory import load_model_from_ckpt
from trapred.train import _device


def _batch(item: dict, device: torch.device) -> dict:
    return {key: value.unsqueeze(0).to(device) for key, value in item.items()}


@torch.no_grad()
def _prediction(model, item: dict, device: torch.device) -> dict:
    out = model(_batch(item, device))
    traj = out["traj"][0].float().cpu().numpy()
    probability = torch.softmax(out["pi"][0].float(), dim=-1).cpu().numpy()
    gt = item["future"].numpy()
    valid = (item["future_valid"].numpy() > 0.5).astype(np.float64)
    ade, fde = _masked_ade_fde(traj[None], gt[None], valid[None])
    selected = int(probability.argmax())
    oracle = int(ade[0].argmin())
    return {
        "traj": traj,
        "probability": probability,
        "selected": selected,
        "oracle": oracle,
        "ade": ade[0],
        "fde": fde[0],
    }


@torch.no_grad()
def _pick_selector_case(
    model, ds: SceneDataset, device: torch.device, *, scan: int, seed: int,
) -> int:
    """Find a valid case where candidate generation works but ranking matters."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(scan, len(ds)), replace=False)
    best_index = int(indices[0])
    best_score = -np.inf
    for index in indices:
        item = ds[int(index)]
        if int((item["future_valid"] > 0.5).sum()) < 10:
            continue
        pred = _prediction(model, item, device)
        selected = pred["selected"]
        oracle = pred["oracle"]
        oracle_ade = float(pred["ade"][oracle])
        if not np.isfinite(oracle_ade) or oracle_ade > 3.0:
            continue
        gap = float(
            pred["ade"][selected] - oracle_ade
            + 0.25 * (pred["fde"][selected] - pred["fde"][oracle])
        )
        # Prefer a visible selector gap, but retain cases with useful candidates.
        score = gap + 0.05 * float(np.std(pred["traj"][:, -1], axis=0).sum())
        if score > best_score:
            best_score = score
            best_index = int(index)
    return best_index


def _bounds(item: dict, predictions: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    agents = item["agents"].numpy()
    gt = item["future"].numpy()
    valid = item["future_valid"].numpy() > 0.5
    points = [gt[valid]]
    for agent in agents:
        mask = agent[:, IDX_VALID] > 0.5
        if mask.any():
            points.append(agent[mask, :2])
    for pred in predictions:
        points.append(pred["traj"].reshape(-1, 2))
    xy = np.concatenate(points, axis=0)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, np.array([45.0, 24.0]))
    center = 0.5 * (lo + hi)
    margin = np.array([8.0, 5.0])
    lo = center - 0.5 * span - margin
    hi = center + 0.5 * span + margin
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _draw_map(ax, item: dict) -> None:
    points = item["map_pts"].numpy()
    valid = item["map_valid"].numpy() > 0.5
    source = item["map_src"].numpy()
    colors = {0: "#94a3b8", 1: "#16a34a", 2: "#7c3aed"}
    styles = {0: "-", 1: "-", 2: "--"}
    widths = {0: 0.9, 1: 0.75, 2: 0.75}
    for index in range(points.shape[0]):
        if not valid[index]:
            continue
        src = int(source[index])
        ax.plot(
            points[index, :, 0], points[index, :, 1],
            color=colors.get(src, "#94a3b8"), ls=styles.get(src, "-"),
            lw=widths.get(src, 0.7), alpha=0.42, zorder=1,
        )


def _draw_panel(
    ax, item: dict, pred: dict, label: str,
    xlim: tuple[float, float], ylim: tuple[float, float],
) -> None:
    _draw_map(ax, item)
    agents = item["agents"].numpy()
    gt = item["future"].numpy()
    valid = item["future_valid"].numpy() > 0.5

    for index, agent in enumerate(agents):
        mask = agent[:, IDX_VALID] > 0.5
        if mask.sum() < 2:
            continue
        ax.plot(
            agent[mask, 0], agent[mask, 1],
            color="#475569" if index == 0 else "#cbd5e1",
            lw=2.0 if index == 0 else 0.8,
            alpha=0.95 if index == 0 else 0.65,
            zorder=3,
        )

    selected = pred["selected"]
    oracle = pred["oracle"]
    other_labeled = False
    for mode, trajectory in enumerate(pred["traj"]):
        if mode in (selected, oracle):
            continue
        ax.plot(
            trajectory[:, 0], trajectory[:, 1],
            color="#60a5fa", lw=0.85, alpha=0.38, zorder=4,
            label="other candidates" if not other_labeled else None,
        )
        other_labeled = True
    if oracle != selected:
        trajectory = pred["traj"][oracle]
        ax.plot(
            trajectory[:, 0], trajectory[:, 1], color="#16a34a",
            lw=2.1, ls="--", label="oracle candidate", zorder=6,
        )
    trajectory = pred["traj"][selected]
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], color="#dc2626",
        lw=2.6, label="selected", zorder=7,
    )
    ax.plot(
        gt[valid, 0], gt[valid, 1], color="#111827",
        lw=2.8, label="ground truth", zorder=8,
    )
    ax.scatter([0.0], [0.0], s=38, color="#f59e0b", edgecolor="#92400e", zorder=9)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.set_xlabel("longitudinal displacement [m]")
    ax.set_ylabel("lateral displacement [m]")
    ax.set_title(
        f"{label}\n"
        f"selected ADE/FDE={pred['ade'][selected]:.2f}/{pred['fde'][selected]:.2f} m  "
        f"oracle={pred['ade'][oracle]:.2f}/{pred['fde'][oracle]:.2f} m  "
        f"p={pred['probability'][selected]:.2f}",
        fontsize=10,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--left-ckpt", type=Path, required=True)
    parser.add_argument("--right-ckpt", type=Path, required=True)
    parser.add_argument("--left-label", default="Model A")
    parser.add_argument("--right-label", default="Model B")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--scan", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ds = SceneDataset(build_cache(cfg, force=False), args.split)
    device = _device()
    left = load_model_from_ckpt(args.left_ckpt, device)
    right = load_model_from_ckpt(args.right_ckpt, device)
    index = args.sample_index
    if index < 0:
        index = _pick_selector_case(right, ds, device, scan=args.scan, seed=args.seed)
    if index >= len(ds):
        raise IndexError(f"sample index {index} outside split of length {len(ds)}")

    item = ds[index]
    left_pred = _prediction(left, item, device)
    right_pred = _prediction(right, item, device)
    xlim, ylim = _bounds(item, [left_pred, right_pred])
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharex=True, sharey=True)
    _draw_panel(axes[0], item, left_pred, args.left_label, xlim, ylim)
    _draw_panel(axes[1], item, right_pred, args.right_label, xlim, ylim)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    row = ds.rows[ds.index[index]]
    fig.suptitle(
        f"One-scene multi-trajectory comparison — {row.get('site', '')}, "
        f"ego={row.get('ego_id', '')}, frame={row.get('start_frame', '')}, "
        f"sample={index}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "sample_index": index,
        "site": row.get("site", ""),
        "ego_id": row.get("ego_id", ""),
        "start_frame": row.get("start_frame", ""),
        "left": {
            "label": args.left_label,
            "selected_mode": left_pred["selected"],
            "selected_probability": float(left_pred["probability"][left_pred["selected"]]),
            "selected_ADE": float(left_pred["ade"][left_pred["selected"]]),
            "selected_FDE": float(left_pred["fde"][left_pred["selected"]]),
            "oracle_ADE": float(left_pred["ade"][left_pred["oracle"]]),
            "oracle_FDE": float(left_pred["fde"][left_pred["oracle"]]),
        },
        "right": {
            "label": args.right_label,
            "selected_mode": right_pred["selected"],
            "selected_probability": float(right_pred["probability"][right_pred["selected"]]),
            "selected_ADE": float(right_pred["ade"][right_pred["selected"]]),
            "selected_FDE": float(right_pred["fde"][right_pred["selected"]]),
            "oracle_ADE": float(right_pred["ade"][right_pred["oracle"]]),
            "oracle_FDE": float(right_pred["fde"][right_pred["oracle"]]),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
