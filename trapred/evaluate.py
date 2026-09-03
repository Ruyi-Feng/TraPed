"""Evaluate a checkpoint on val/test and write qualitative plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from trapred.config import load_config
from trapred.data.dataset import SceneDataset, build_cache
from trapred.models.factory import ARCHES, ckpt_path, load_model_from_ckpt
from trapred.models.mat import constant_velocity
from trapred.train import _device, _move, evaluate_loader


def load_model(ckpt_file: Path, device: torch.device):
    return load_model_from_ckpt(ckpt_file, device)


@torch.no_grad()
def plot_samples(model, ds: SceneDataset, out_png: Path, device, n: int = 6) -> None:
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    dt = 1.0 / float(ds.meta.get("fps", 10.0))
    for ax, i in zip(axes, idxs):
        item = ds[int(i)]
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
        out = model(batch)
        traj = out["traj"][0].cpu().numpy()
        pi = torch.softmax(out["pi"][0], dim=-1).cpu().numpy()
        gt = item["future"].numpy()
        valid = item["future_valid"].numpy() > 0.5
        agents = item["agents"].numpy()
        cv = constant_velocity(batch["agents"], model.t_out, dt)[0].cpu().numpy()
        # neighbors last frame
        last = agents[:, -1]
        ax.scatter(last[:, 0], last[:, 1], s=12, c="#94a3b8", label="agents")
        ax.plot(0, 0, "o", color="#dc2626", markersize=7, label="ego")
        mp = item["map_pts"].numpy()
        mv = item["map_valid"].numpy() > 0.5
        src = item["map_src"].numpy()
        colors = {0: "#0ea5e9", 1: "#16a34a", 2: "#a855f7"}
        for t in range(mp.shape[0]):
            if not mv[t]:
                continue
            ax.plot(mp[t, :, 0], mp[t, :, 1], color=colors.get(int(src[t]), "#64748b"),
                    lw=0.8, alpha=0.7)
        ax.plot(gt[valid, 0], gt[valid, 1], "k-", lw=2.0, label="gt")
        ax.plot(cv[:, 0], cv[:, 1], color="#f59e0b", ls="--", lw=1.2, label="CV")
        k_best = int(pi.argmax())
        for k in range(traj.shape[0]):
            kw = dict(lw=1.8, color="#2563eb") if k == k_best else dict(lw=0.7, color="#93c5fd", alpha=0.7)
            ax.plot(traj[k, :, 0], traj[k, :, 1], **kw)
        ax.set_aspect("equal")
        ax.set_title(f"p={pi[k_best]:.2f}")
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/expresswayA_sample.yaml"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--arch", choices=ARCHES, default="mat")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = _device()
    cache_dir = build_cache(cfg, force=False)
    ckpt = args.ckpt or ckpt_path(cfg.out_path, args.arch)
    model = load_model(ckpt, device)
    ds = SceneDataset(cache_dir, args.split)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    dt = 1.0 / float(ds.meta.get("fps", cfg.window.target_fps))
    metrics = evaluate_loader(model, loader, device, dt)
    out_json = cfg.out_path / f"metrics_{args.split}_{args.arch}.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    plot_samples(model, ds, cfg.out_path / f"qual_{args.split}_{args.arch}.png", device)
    print("plots ->", cfg.out_path / f"qual_{args.split}_{args.arch}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
