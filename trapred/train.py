"""Train the map-aware trajectory Transformer."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trapred.config import Cfg, dump_config, load_config
from trapred.data.dataset import SceneDataset, build_cache
from trapred.eval.metrics import batch_metrics, cv_metrics
from trapred.models.factory import ARCHES, build_model, ckpt_path, history_filename, model_kwargs
from trapred.models.losses import multimodal_loss
from trapred.models.mat import constant_velocity


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loader(model, loader, device, dt: float) -> dict:
    model.eval()
    acc = {}
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        out = model(batch)
        m = batch_metrics(out, batch["future"], batch["future_valid"])
        cv = constant_velocity(batch["agents"], model.t_out, dt)
        c = cv_metrics(cv, batch["future"], batch["future_valid"])
        bs = batch["future"].size(0)
        for k, v in {**m, **c}.items():
            acc[k] = acc.get(k, 0.0) + v * bs
        n += bs
    return {k: v / max(n, 1) for k, v in acc.items()}


def train_one_epoch(model, loader, opt, scaler, device, amp: bool) -> dict:
    model.train()
    tot = {}
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        opt.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(batch)
                stats = multimodal_loss(out, batch["future"], batch["future_valid"])
            scaler.scale(stats["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            out = model(batch)
            stats = multimodal_loss(out, batch["future"], batch["future_valid"])
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = batch["future"].size(0)
        for k, v in stats.items():
            tot[k] = tot.get(k, 0.0) + float(v) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in tot.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/expresswayA_sample.yaml"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--arch", choices=ARCHES, default="mat",
                        help="mat = map-aware transformer; transformer/lstm = no-map baselines")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_windows is not None:
        cfg.train.max_windows = args.max_windows
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    set_seed(cfg.train.seed)
    device = _device()
    out_dir = cfg.out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    cache_dir = build_cache(cfg, force=args.force_cache)
    train_ds = SceneDataset(cache_dir, "train")
    val_ds = SceneDataset(cache_dir, "val")
    t_out = train_ds.future.shape[1]
    fps = float(train_ds.meta.get("fps", cfg.window.target_fps))
    dt = 1.0 / fps

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= cfg.train.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
    )

    model = build_model(args.arch, t_out=t_out, dt=dt, model=cfg.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch}  device={device}  params={n_params/1e6:.2f}M  "
          f"train={len(train_ds)}  val={len(val_ds)}  t_out={t_out}")

    best = 1e9
    history = []
    ckpt_file = ckpt_path(out_dir, args.arch)
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, scaler, device, cfg.train.amp)
        va = evaluate_loader(model, val_loader, device, dt)
        sched.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **va,
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(
            f"epoch {epoch:02d}  loss={tr['loss']:.3f}  "
            f"val minADE={va['minADE']:.3f} minFDE={va['minFDE']:.3f}  "
            f"cvADE={va['cvADE']:.3f}  {row['sec']:.1f}s"
        )
        if va["minADE"] < best:
            best = va["minADE"]
            torch.save({
                "arch": args.arch,
                "model": model.state_dict(),
                "cfg": model_kwargs(cfg.model, t_out=t_out, dt=dt),
                "epoch": epoch,
                "val": va,
            }, ckpt_file)
    hist_path = out_dir / history_filename(args.arch)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best val minADE={best:.4f}  saved {ckpt_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
