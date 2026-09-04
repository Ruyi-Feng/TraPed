"""Case visualizations: BEV (bbox + history + lane fill) + s–t longitudinal plot."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPolygon

from trapred.config import load_config
from trapred.data.dataset import SceneDataset, build_cache
from trapred.data.windows import IDX_EGO, IDX_VALID
from trapred.eval.metrics import _masked_ade_fde
from trapred.models.factory import ARCHES, ckpt_path, load_model_from_ckpt
from trapred.models.mat import constant_velocity
from trapred.train import _device

# semi-transparent lane fills (one color per polygon token)
LANE_FILLS = [
    "#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3", "#e0e7ff",
    "#ccfbf1", "#ffedd5", "#f3e8ff", "#fee2e2", "#ecfccb",
    "#cffafe", "#fae8ff",
]
PRED_STYLE = {
    "gt_future": ("GT", "#111827", "-", 2.4),
    "gt_hist": ("ego hist", "#6b7280", "-", 1.6),
    "mat": ("MAT", "#2563eb", "-", 2.0),
    "transformer": ("Transformer", "#7c3aed", "-.", 1.7),
    "lstm": ("LSTM", "#ea580c", ":", 1.7),
    "cv": ("CV", "#f59e0b", "--", 1.3),
}


def _heading_from_feat(row: np.ndarray) -> float:
    return float(math.atan2(row[8], row[9]))


def _bbox_corners(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    lc, wc = length * 0.5, width * 0.5
    local = np.array([[-lc, -wc], [lc, -wc], [lc, wc], [-lc, wc]], dtype=np.float64)
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _draw_vehicle(ax, row: np.ndarray, *, is_ego: bool, alpha: float = 0.85) -> None:
    if row[IDX_VALID] < 0.5:
        return
    cx, cy = float(row[0]), float(row[1])
    length = max(float(row[11]), 3.5)
    width = max(float(row[12]), 1.6)
    h = _heading_from_feat(row)
    corners = _bbox_corners(cx, cy, length, width, h)
    face = "#fecaca" if is_ego else "#e2e8f0"
    edge = "#dc2626" if is_ego else "#475569"
    ax.add_patch(MplPolygon(
        corners, closed=True, facecolor=face, edgecolor=edge,
        linewidth=1.2 if is_ego else 0.8, alpha=alpha, zorder=8,
    ))


def _draw_lane_fills(ax, item, xlim, ylim) -> None:
    mp = item["map_pts"].numpy()
    mv = item["map_valid"].numpy() > 0.5
    src = item["map_src"].numpy()
    lane_i = 0
    for t in range(mp.shape[0]):
        if not mv[t] or int(src[t]) != 0:
            continue
        poly = mp[t, :, :2]
        if not _poly_intersects_view(poly, xlim, ylim):
            continue
        color = LANE_FILLS[lane_i % len(LANE_FILLS)]
        ax.add_patch(MplPolygon(
            poly, closed=True, facecolor=color, edgecolor="none", alpha=0.55, zorder=1,
        ))
        lane_i += 1


def _poly_intersects_view(poly: np.ndarray, xlim, ylim) -> bool:
    return (
        poly[:, 0].max() >= xlim[0] and poly[:, 0].min() <= xlim[1]
        and poly[:, 1].max() >= ylim[0] and poly[:, 1].min() <= ylim[1]
    )


def _draw_map_lines(ax, item, xlim, ylim) -> None:
    mp = item["map_pts"].numpy()
    mv = item["map_valid"].numpy() > 0.5
    src = item["map_src"].numpy()
    for t in range(mp.shape[0]):
        if not mv[t]:
            continue
        s = int(src[t])
        if s == 0:
            continue
        xy = mp[t, :, :2]
        if not _poly_intersects_view(xy, xlim, ylim):
            continue
        if s == 1:
            ax.plot(xy[:, 0], xy[:, 1], color="#bbf7d0", lw=0.6, alpha=0.35, zorder=2)
        elif s == 2:
            ax.plot(xy[:, 0], xy[:, 1], color="#ddd6fe", lw=0.7, ls="--", alpha=0.30, zorder=2)


@torch.no_grad()
def _collect_preds(item, models, device, dt, t_out) -> dict[str, np.ndarray]:
    batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
    gt = item["future"].numpy()
    valid = item["future_valid"].numpy() > 0.5
    preds = {"cv": constant_velocity(batch["agents"], t_out, dt)[0].cpu().numpy()}
    for arch in ("mat", "transformer", "lstm"):
        if arch not in models:
            continue
        out = models[arch](batch)
        traj = out["traj"][0].cpu().numpy()
        ade, _ = _masked_ade_fde(
            traj[None], gt[None], valid.astype(np.float64)[None],
        )
        preds[arch] = traj[int(ade[0].argmin())]
    return preds


def _view_bounds(agents, gt, valid, preds, margin: float = 12.0):
    pts = [agents[0, agents[0, :, IDX_VALID] > 0.5, :2]]
    pts.append(gt[valid])
    for p in preds.values():
        pts.append(p)
    for n in range(agents.shape[0]):
        m = agents[n, :, IDX_VALID] > 0.5
        if m.any():
            pts.append(agents[n, m, :2])
    allp = np.vstack(pts)
    x0, x1 = float(allp[:, 0].min() - margin), float(allp[:, 0].max() + margin)
    y0, y1 = float(allp[:, 1].min() - margin), float(allp[:, 1].max() + margin)
    # keep a minimum window
    if x1 - x0 < 40:
        c = 0.5 * (x0 + x1)
        x0, x1 = c - 20, c + 20
    if y1 - y0 < 30:
        c = 0.5 * (y0 + y1)
        y0, y1 = c - 15, c + 15
    return (x0, x1), (y0, y1)


def _time_axes(t_in: int, t_out: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    hist_t = np.arange(-(t_in - 1), 1) * dt
    fut_t = np.arange(1, t_out + 1) * dt
    return hist_t, fut_t


def plot_spatial(ax, item, preds, *, title: str) -> None:
    agents = item["agents"].numpy()
    gt = item["future"].numpy()
    valid = item["future_valid"].numpy() > 0.5
    xlim, ylim = _view_bounds(agents, gt, valid, preds)

    _draw_lane_fills(ax, item, xlim, ylim)
    _draw_map_lines(ax, item, xlim, ylim)

    # neighbor history (light)
    segs = []
    for n in range(1, agents.shape[0]):
        m = agents[n, :, IDX_VALID] > 0.5
        if m.sum() < 2:
            continue
        xy = agents[n, m, :2]
        segs.append(xy)
    for xy in segs:
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", lw=0.9, alpha=0.7, zorder=3)

    # ego history
    ego_m = agents[0, :, IDX_VALID] > 0.5
    ego_hist = agents[0, ego_m, :2]
    ax.plot(ego_hist[:, 0], ego_hist[:, 1], color="#64748b", lw=2.0, zorder=5, label="ego hist")

    # vehicles at last frame
    for n in range(agents.shape[0]):
        if agents[n, -1, IDX_VALID] < 0.5:
            continue
        _draw_vehicle(ax, agents[n, -1], is_ego=(n == 0))

    # futures
    ax.plot(gt[valid, 0], gt[valid, 1], **dict(
        color=PRED_STYLE["gt_future"][1], lw=PRED_STYLE["gt_future"][3], zorder=9, label="GT",
    ))
    for key in ("cv", "lstm", "transformer", "mat"):
        if key not in preds:
            continue
        name, color, ls, lw = PRED_STYLE[key]
        ax.plot(preds[key][:, 0], preds[key][:, 1], color=color, ls=ls, lw=lw, zorder=10, label=name)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("longitudinal Δx [m]")
    ax.set_ylabel("lateral Δy [m]")
    ax.grid(alpha=0.2, zorder=0)


def plot_st(ax, item, preds, dt: float, t_in: int, t_out: int) -> None:
    agents = item["agents"].numpy()
    gt = item["future"].numpy()
    valid = item["future_valid"].numpy() > 0.5
    hist_t, fut_t = _time_axes(t_in, t_out, dt)

    ego_m = agents[0, :, IDX_VALID] > 0.5
    s_hist = agents[0, ego_m, 0]
    t_hist = hist_t[ego_m]
    ax.plot(t_hist, s_hist, color=PRED_STYLE["gt_hist"][1], lw=2.0, label="ego hist")
    ax.plot(fut_t[valid], gt[valid, 0], color=PRED_STYLE["gt_future"][1],
            lw=2.4, label="GT")

    for key in ("cv", "lstm", "transformer", "mat"):
        if key not in preds:
            continue
        name, color, ls, lw = PRED_STYLE[key]
        ax.plot(fut_t, preds[key][:, 0], color=color, ls=ls, lw=lw, label=name)

    ax.axvline(0.0, color="#94a3b8", ls=":", lw=1.0)
    ax.text(0.02, 0.03, "t=0 (last obs)", transform=ax.transAxes, fontsize=8, color="#64748b")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("longitudinal s [m]")
    ax.set_title("longitudinal s–t (ego frame +x)", fontsize=10)
    ax.grid(alpha=0.25)


def _sample_scores(models, item, device, dt, t_out) -> dict | None:
    gt = item["future"].numpy()
    valid = item["future_valid"].numpy() > 0.5
    if valid.sum() < 10:
        return None
    preds = _collect_preds(item, models, device, dt, t_out)
    v = valid.astype(np.float64)
    out = {}
    for key in ("mat", "transformer", "lstm", "cv"):
        if key not in preds:
            continue
        ade, _ = _masked_ade_fde(preds[key][None], gt[None], v[None])
        out[f"{key}_ade"] = float(ade[0, 0])
    if "mat_ade" not in out:
        return None
    if "transformer_ade" in out:
        out["mat_gain"] = out["transformer_ade"] - out["mat_ade"]
    return out


def _sample_error(models, item, device, dt, t_out) -> float:
    s = _sample_scores(models, item, device, dt, t_out)
    return s["mat_ade"] if s else 1e9


def _pick_cases(
    ds,
    models,
    device,
    dt,
    t_out,
    *,
    n: int,
    seed: int,
    fixed: list[int] | None = None,
) -> list[int]:
    """Pick cases: 2 baseline (easy/medium MAT) + rest where MAT beats Transformer."""
    if "transformer" not in models:
        raise SystemExit("need transformer checkpoint for MAT vs Transformer cases")

    scored = []
    for i in range(len(ds)):
        s = _sample_scores(models, ds[i], device, dt, t_out)
        if s is None:
            continue
        scored.append((i, s))

    by_mat = sorted(scored, key=lambda t: t[1]["mat_ade"])
    easy = by_mat[0][0]
    medium = by_mat[len(by_mat) // 3][0]

    chosen = []
    if fixed:
        chosen.extend(fixed)
    else:
        chosen.extend([easy, medium])

    # MAT wins: gain > 0.15 m, MAT ADE in reasonable range, diverse ego
    mat_wins = [
        (i, s) for i, s in scored
        if s.get("mat_gain", 0) > 0.15
        and 0.3 < s["mat_ade"] < 4.0
        and i not in chosen
    ]
    mat_wins.sort(key=lambda t: t[1]["mat_gain"], reverse=True)

    used_egos = {ds.rows[ds.index[i]]["ego_id"] for i in chosen}
    for i, s in mat_wins:
        if len(chosen) >= n:
            break
        ego = ds.rows[ds.index[i]]["ego_id"]
        if ego in used_egos:
            continue
        chosen.append(i)
        used_egos.add(ego)

    # fill if needed
    if len(chosen) < n:
        for i, s in mat_wins:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= n:
                break
    return chosen[:n]


def plot_case_figure(item, models, device, dt, t_in, t_out, *, title: str, out: Path) -> None:
    preds = _collect_preds(item, models, device, dt, t_out)
    fig, (ax_bev, ax_st) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    plot_spatial(ax_bev, item, preds, title=title)
    plot_st(ax_st, item, preds, dt, t_in, t_out)
    handles, labels = ax_bev.get_legend_handles_labels()
    st_h, st_l = ax_st.get_legend_handles_labels()
    for h, l in zip(st_h, st_l):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/expresswayA_sample.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--only", type=str, default="",
        help="comma-separated 1-based case numbers to render, e.g. 3,4",
    )
    parser.add_argument(
        "--keep-indices", type=str, default="786,732",
        help="dataset indices to keep as case 1,2 (empty to auto-pick)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = _device()
    cache_dir = build_cache(cfg, force=False)
    ds = SceneDataset(cache_dir, args.split)
    dt = 1.0 / float(ds.meta.get("fps", cfg.window.target_fps))
    t_in = int(ds.meta["t_in"])
    t_out = int(ds.meta["t_out"])

    models = {}
    for arch in ARCHES:
        p = ckpt_path(cfg.out_path, arch)
        if p.exists():
            models[arch] = load_model_from_ckpt(p, device)
    if "mat" not in models:
        raise SystemExit("need MAT checkpoint")

    fixed = None
    if args.keep_indices.strip():
        fixed = [int(x) for x in args.keep_indices.split(",") if x.strip()]

    chosen = _pick_cases(
        ds, models, device, dt, t_out, n=args.n, seed=args.seed, fixed=fixed,
    )

    only_cases = None
    if args.only.strip():
        only_cases = {int(x) for x in args.only.split(",") if x.strip()}

    out_dir = cfg.out_path
    for j, idx in enumerate(chosen):
        case_no = j + 1
        if only_cases is not None and case_no not in only_cases:
            continue
        item = ds[int(idx)]
        row = ds.rows[ds.index[int(idx)]]
        sc = _sample_scores(models, item, device, dt, t_out) or {}
        mat_a = sc.get("mat_ade", float("nan"))
        tr_a = sc.get("transformer_ade", float("nan"))
        gain = sc.get("mat_gain", float("nan"))
        title = (
            f"ego={row['ego_id']}  frame={row['start_frame']}\n"
            f"MAT={mat_a:.2f}m  Transformer={tr_a:.2f}m  gain={gain:+.2f}m"
        )
        out = out_dir / f"case_{args.split}_{case_no}.png"
        plot_case_figure(item, models, device, dt, t_in, t_out, title=title, out=out)
        print("wrote", out)
        print(
            f"  case{case_no}: idx={idx} ego={row['ego_id']} frame={row['start_frame']} "
            f"MAT={mat_a:.2f} Trans={tr_a:.2f} gain={gain:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
