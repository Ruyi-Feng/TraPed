"""Train the map-aware trajectory Transformer."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from trapred.ablations import ABLATION_PROFILES, ablation_dict, ablation_run_name, apply_ablation
from trapred.config import Cfg, dump_config, load_config
from trapred.data.dataset import SceneDataset, build_cache
from trapred.eval.metrics import batch_metrics, cv_metrics
from trapred.models.factory import (
    ARCHES, build_model, ckpt_path, history_filename, model_kwargs, torch_load,
)
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


def _float_outputs(out: dict) -> dict:
    """Cast model outputs to fp32 so Laplace NLL stays numerically stable under AMP."""
    return {
        k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
        for k, v in out.items()
    }


def _grads_finite(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _limited_subset(dataset, limit: int | None, seed: int):
    """Deterministic subset used by parameter sweeps without rebuilding cache."""
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _checkpoint_score(metrics: dict, name: str) -> float:
    if name == "minADE":
        return float(metrics["minADE"])
    if name == "mlADE":
        return float(metrics["mlADE"])
    if name == "reliable":
        # Selected trajectory quality, with extra emphasis on its endpoint.
        return float(metrics["mlADE"] + 0.25 * metrics["mlFDE"])
    raise ValueError(f"unknown checkpoint metric {name!r}")


def train_one_epoch(
    model, loader, opt, scaler, device, amp: bool, *,
    amp_dtype: torch.dtype = torch.float16,
    grad_accum: int = 1, grad_clip: float = 1.0,
    loss_kwargs: dict | None = None,
) -> dict:
    model.train()
    tot = {}
    n = 0
    skipped = 0
    grad_accum = max(1, int(grad_accum))
    loss_kwargs = loss_kwargs or {}
    use_scaler = bool(amp and device.type == "cuda" and scaler.is_enabled())
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = _move(batch, device)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(
                    batch, use_posterior=True
                ) if getattr(model, "is_generative", False) else model(batch)
            # Loss outside autocast: log/div on Laplace scale is unsafe in fp16.
            stats = multimodal_loss(
                _float_outputs(out),
                batch["future"].float(),
                batch["future_valid"],
                **loss_kwargs,
            )
            loss = stats["loss"] / grad_accum
        else:
            out = model(
                batch, use_posterior=True
            ) if getattr(model, "is_generative", False) else model(batch)
            stats = multimodal_loss(
                out, batch["future"], batch["future_valid"], **loss_kwargs
            )
            loss = stats["loss"] / grad_accum
        do_step = (step + 1) % grad_accum == 0 or step + 1 == len(loader)
        if not torch.isfinite(loss).all():
            skipped += 1
            if do_step:
                opt.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.update()
            continue
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if do_step:
            if use_scaler:
                scaler.unscale_(opt)
            if not _grads_finite(model):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.update()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        bs = batch["future"].size(0)
        for k, v in stats.items():
            tot[k] = tot.get(k, 0.0) + float(v) * bs
        n += bs
    out_stats = {k: v / max(n, 1) for k, v in tot.items()}
    out_stats["skipped_steps"] = float(skipped)
    return out_stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/expresswayA_sample.yaml"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None,
                        help="deterministic training subset; cache is unchanged")
    parser.add_argument("--val-samples", type=int, default=None,
                        help="deterministic validation subset; cache is unchanged")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--temporal-layers", type=int, default=None)
    parser.add_argument("--social-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--map-layers", type=int, default=None)
    parser.add_argument("--n-modes", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--endpoint-weight", type=float, default=None)
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--soft-cls-temp", type=float, default=None)
    parser.add_argument("--winner-fde-weight", type=float, default=None)
    parser.add_argument("--cls-weight", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--kl-free-bits", type=float, default=None)
    parser.add_argument("--kl-warmup-epochs", type=int, default=None)
    parser.add_argument("--confidence-regret-weight", type=float, default=None)
    parser.add_argument("--diversity-margin", type=float, default=None)
    parser.add_argument(
        "--checkpoint-metric", choices=("minADE", "mlADE", "reliable"),
        default=None,
        help="metric used to save best checkpoint; reliable=mlADE+0.25*mlFDE",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--amp", dest="amp", action="store_true", default=None,
        help="enable CUDA AMP (default: config)",
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false",
        help="disable CUDA AMP",
    )
    parser.add_argument(
        "--amp-dtype", choices=("auto", "fp16", "bf16"), default="auto",
        help="AMP compute dtype; auto prefers bf16 when supported",
    )
    parser.add_argument("--arch", choices=ARCHES, default="mat",
                        help="mat_cvae = conditional generative multi-trajectory model")
    parser.add_argument(
        "--ablation", choices=tuple(ABLATION_PROFILES), default=None,
        help="named MAT-v2 component ablation (requires --arch mat_v2)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_windows is not None:
        cfg.train.max_windows = args.max_windows
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.amp is not None:
        cfg.train.amp = bool(args.amp)
    train_overrides = {
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "loss_endpoint_weight": args.endpoint_weight,
        "loss_diversity_weight": args.diversity_weight,
        "loss_soft_cls_temp": args.soft_cls_temp,
        "loss_winner_fde_weight": args.winner_fde_weight,
        "loss_cls_weight": args.cls_weight,
        "loss_kl_weight": args.kl_weight,
        "loss_kl_free_bits": args.kl_free_bits,
        "loss_kl_warmup_epochs": args.kl_warmup_epochs,
        "loss_confidence_regret_weight": args.confidence_regret_weight,
        "loss_diversity_margin": args.diversity_margin,
        "checkpoint_metric": args.checkpoint_metric,
    }
    for name, value in train_overrides.items():
        if value is not None:
            setattr(cfg.train, name, value)
    model_overrides = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "n_temporal_layers": args.temporal_layers,
        "n_social_layers": args.social_layers,
        "n_decoder_layers": args.decoder_layers,
        "n_map_layers": args.map_layers,
        "n_modes": args.n_modes,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
    }
    for name, value in model_overrides.items():
        if value is not None:
            setattr(cfg.model, name, value)
    if args.ablation is not None:
        if args.arch != "mat_v2":
            parser.error("--ablation requires --arch mat_v2")
        apply_ablation(cfg, args.ablation)

    set_seed(cfg.train.seed)
    device = _device()
    run_name = args.run_name
    if run_name is None and args.ablation is not None:
        run_name = ablation_run_name(args.ablation)
    out_dir = cfg.out_path / run_name if run_name else cfg.out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    cache_dir = build_cache(cfg, force=args.force_cache)
    full_train_ds = SceneDataset(cache_dir, "train")
    full_val_ds = SceneDataset(cache_dir, "val")
    t_out = full_train_ds.future.shape[1]
    fps = float(full_train_ds.meta.get("fps", cfg.window.target_fps))
    dt = 1.0 / fps
    train_ds = _limited_subset(full_train_ds, args.train_samples, cfg.train.seed)
    val_ds = _limited_subset(full_val_ds, args.val_samples, cfg.train.seed + 1)

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
    warmup = max(0, int(cfg.train.warmup_epochs))
    def lr_factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, cfg.train.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    # GradScaler is only useful for fp16; bf16 shares fp32 exponent range.
    use_grad_scaler = (
        cfg.train.amp and device.type == "cuda" and amp_dtype == torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch}  device={device}  params={n_params/1e6:.2f}M  "
          f"train={len(train_ds)}  val={len(val_ds)}  t_out={t_out}  "
          f"batch={cfg.train.batch_size}x{cfg.train.grad_accum} "
          f"(effective={cfg.train.batch_size * cfg.train.grad_accum})  "
          f"amp={cfg.train.amp} dtype={str(amp_dtype).replace('torch.', '')}  "
          f"lr={cfg.train.lr:g} warmup={cfg.train.warmup_epochs} "
          f"grad_clip={cfg.train.grad_clip} best_by={cfg.train.checkpoint_metric}")

    best = 1e9
    history = []
    start_epoch = 1
    ckpt_file = ckpt_path(out_dir, args.arch)
    last_file = out_dir / "last.pt"
    if args.resume and last_file.exists():
        state = torch_load(last_file, device)
        saved_ablation = state.get("ablation")
        if saved_ablation is not None and saved_ablation != args.ablation:
            raise ValueError(
                f"cannot resume {last_file}: checkpoint ablation "
                f"{saved_ablation!r} != requested {args.ablation!r}"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best = float(state.get("best", best))
        history = list(state.get("history", []))
        start_epoch = int(state["epoch"]) + 1
        print(f"resumed {last_file} at epoch {start_epoch}")
    loss_kwargs = {
        "cls_w": cfg.train.loss_cls_weight,
        "endpoint_weight": cfg.train.loss_endpoint_weight,
        "diversity_w": cfg.train.loss_diversity_weight,
        "diversity_margin": cfg.train.loss_diversity_margin,
        "soft_cls_temp": cfg.train.loss_soft_cls_temp,
        "winner_fde_weight": cfg.train.loss_winner_fde_weight,
        "kl_free_bits": cfg.train.loss_kl_free_bits,
        "confidence_regret_weight": cfg.train.loss_confidence_regret_weight,
    }
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t0 = time.time()
        epoch_loss_kwargs = dict(loss_kwargs)
        if cfg.train.loss_kl_warmup_epochs > 0:
            kl_factor = min(1.0, epoch / cfg.train.loss_kl_warmup_epochs)
        else:
            kl_factor = 1.0
        epoch_loss_kwargs["kl_weight"] = cfg.train.loss_kl_weight * kl_factor
        tr = train_one_epoch(
            model, train_loader, opt, scaler, device, cfg.train.amp,
            amp_dtype=amp_dtype,
            grad_accum=cfg.train.grad_accum, grad_clip=cfg.train.grad_clip,
            loss_kwargs=epoch_loss_kwargs,
        )
        va = evaluate_loader(model, val_loader, device, dt)
        sched.step()
        score = _checkpoint_score(va, cfg.train.checkpoint_metric)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **va,
               "checkpoint_score": score, "sec": round(time.time() - t0, 1)}
        history.append(row)
        skipped = int(tr.get("skipped_steps", 0.0))
        loss_v = tr.get("loss", float("nan"))
        print(
            f"epoch {epoch:02d}  loss={loss_v:.3f}  "
            f"val minADE={va['minADE']:.3f} minFDE={va['minFDE']:.3f}  "
            f"mlADE={va['mlADE']:.3f} score={score:.3f}  "
            f"cvADE={va['cvADE']:.3f}  skip={skipped}  {row['sec']:.1f}s",
            flush=True,
        )
        if not math.isfinite(loss_v) or not math.isfinite(va["minADE"]):
            print(
                f"non-finite metrics at epoch {epoch}; "
                f"stopping early (last.pt kept for diagnosis)",
                flush=True,
            )
            break
        if score < best:
            best = score
            torch.save({
                "arch": args.arch,
                "ablation": args.ablation,
                "ablation_flags": ablation_dict(cfg),
                "model": model.state_dict(),
                "cfg": model_kwargs(cfg.model, t_out=t_out, dt=dt),
                "epoch": epoch,
                "checkpoint_metric": cfg.train.checkpoint_metric,
                "checkpoint_score": score,
                "val": va,
            }, ckpt_file)
        torch.save({
            "arch": args.arch,
            "ablation": args.ablation,
            "ablation_flags": ablation_dict(cfg),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": model_kwargs(cfg.model, t_out=t_out, dt=dt),
            "epoch": epoch,
            "best": best,
            "history": history,
        }, last_file)
    hist_path = out_dir / history_filename(args.arch)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"best val {cfg.train.checkpoint_metric} score={best:.4f}  "
        f"saved {ckpt_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
