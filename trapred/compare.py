"""Compare MAT vs Transformer / LSTM baselines (and constant-velocity) on one split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from trapred.config import load_config
from trapred.data.dataset import SceneDataset, build_cache
from trapred.models.factory import ARCHES, ckpt_path, load_model_from_ckpt
from trapred.models.mat import constant_velocity
from trapred.train import _device, evaluate_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/expresswayA_sample.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = _device()
    cache_dir = build_cache(cfg, force=False)
    ds = SceneDataset(cache_dir, args.split)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    dt = 1.0 / float(ds.meta.get("fps", cfg.window.target_fps))

    rows = {}
    # Constant velocity from any batch via evaluate_loader's cv_* keys;
    # run MAT if present so cv metrics are filled, else compute from one batch.
    for arch in ARCHES:
        path = ckpt_path(cfg.out_path, arch)
        if not path.exists():
            print(f"skip {arch}: missing {path}")
            continue
        model = load_model_from_ckpt(path, device)
        m = evaluate_loader(model, loader, device, dt)
        rows[arch] = m
        print(f"{arch:12s}  minADE={m['minADE']:.3f}  minFDE={m['minFDE']:.3f}  "
              f"mlADE={m['mlADE']:.3f}  vs CV {m['cvADE']:.3f}")

    if not rows:
        raise SystemExit("no checkpoints found — train with --arch first")

    cv = next(iter(rows.values()))
    rows["constant_velocity"] = {
        "minADE": cv["cvADE"], "minFDE": cv["cvFDE"],
        "mlADE": cv["cvADE"], "mlFDE": cv["cvFDE"],
        "MR2": cv["cvMR2"], "MR5": float("nan"),
        "cvADE": cv["cvADE"], "cvFDE": cv["cvFDE"], "cvMR2": cv["cvMR2"],
    }

    out_json = cfg.out_path / f"compare_{args.split}.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    order = ["constant_velocity", "lstm", "transformer", "mat"]
    names = [k for k in order if k in rows]
    ade = [rows[k]["minADE"] for k in names]
    fde = [rows[k]["minFDE"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    w = 0.35
    ax.bar([i - w / 2 for i in x], ade, w, label="minADE", color="#2563eb")
    ax.bar([i + w / 2 for i in x], fde, w, label="minFDE", color="#dc2626")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("meters")
    ax.set_title(f"5 s trajectory prediction — {args.split}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = cfg.out_path / f"compare_{args.split}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("wrote", out_json)
    print("wrote", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
