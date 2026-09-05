"""Run and summarize a controlled MAT-v2 ablation suite.

Every profile uses the same data split, seed, optimizer, and model size from the
selected YAML. Only the component named by the profile is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from trapred.ablations import (
    ABLATION_PROFILES,
    DEFAULT_ABLATIONS,
    ablation_run_name,
)
from trapred.config import load_config


SUMMARY_METRICS = ("minADE", "minFDE", "mlADE", "mlFDE", "MR2", "MR5")


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _add_value(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _write_summaries(
    config: Path, profiles: list[str], split: str,
) -> tuple[Path, Path, Path]:
    cfg = load_config(config)
    output_root = cfg.out_path.resolve()
    rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    raw: dict[str, dict] = {}
    for profile in profiles:
        metric_path = (
            output_root / ablation_run_name(profile)
            / f"metrics_{split}_mat_v2.json"
        )
        if not metric_path.exists():
            print(f"warning: missing metrics for {profile}: {metric_path}")
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        raw[profile] = metrics
        rows.append({
            "profile": profile,
            **{name: metrics.get(name) for name in SUMMARY_METRICS},
        })
        for site, site_metrics in metrics.get("by_site", {}).items():
            site_rows.append({
                "profile": profile,
                "site": site,
                **{name: site_metrics.get(name) for name in SUMMARY_METRICS},
            })

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"ablation_{split}.json"
    csv_path = output_root / f"ablation_{split}.csv"
    site_csv_path = output_root / f"ablation_{split}_by_site.csv"
    json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("profile", *SUMMARY_METRICS))
        writer.writeheader()
        writer.writerows(rows)
    with site_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("profile", "site", *SUMMARY_METRICS)
        )
        writer.writeheader()
        writer.writerows(site_rows)
    return json_path, csv_path, site_csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and summarize MAT-v2 ablations."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multisite_ablation.yaml")
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=tuple(ABLATION_PROFILES),
        default=list(DEFAULT_ABLATIONS),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = args.config.resolve()
    for index, profile in enumerate(args.profiles):
        if not args.skip_train:
            command = [
                sys.executable, "-m", "trapred.train", "--config", str(config),
                "--arch", "mat_v2", "--ablation", profile,
            ]
            _add_value(command, "--epochs", args.epochs)
            _add_value(command, "--batch-size", args.batch_size)
            _add_value(command, "--grad-accum", args.grad_accum)
            _add_value(command, "--num-workers", args.num_workers)
            _add_value(command, "--max-windows", args.max_windows)
            if args.resume:
                command.append("--resume")
            if args.force_cache and index == 0:
                command.append("--force-cache")
            _run(command, dry_run=args.dry_run)

        if not args.skip_eval:
            command = [
                sys.executable, "-m", "trapred.evaluate", "--config", str(config),
                "--arch", "mat_v2", "--ablation", profile,
                "--split", args.split,
            ]
            _add_value(command, "--batch-size", args.batch_size)
            _run(command, dry_run=args.dry_run)

    if args.dry_run:
        return 0
    paths = _write_summaries(config, args.profiles, args.split)
    print("ablation summaries:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
