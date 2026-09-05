"""Collect validation histories from a generative parameter sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--glob", default="gen-*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = []
    for run_dir in sorted(path for path in args.root.glob(args.glob) if path.is_dir()):
        history_path = run_dir / "history_mat_cvae.json"
        config_path = run_dir / "config.yaml"
        if not history_path.exists() or not config_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model = config.get("model", {})
        train = config.get("train", {})
        best = min(
            history,
            key=lambda item: float(
                item.get(
                    "checkpoint_score",
                    item.get("mlADE", 1e9) + 0.25 * item.get("mlFDE", 1e9),
                )
            ),
        )
        rows.append({
            "run": run_dir.name,
            "epoch": best.get("epoch"),
            "score": best.get(
                "checkpoint_score",
                best.get("mlADE", 1e9) + 0.25 * best.get("mlFDE", 1e9),
            ),
            "minADE": best.get("minADE"),
            "minFDE": best.get("minFDE"),
            "mlADE": best.get("mlADE"),
            "mlFDE": best.get("mlFDE"),
            "selectionGapADE": best.get("selectionGapADE"),
            "top3ADE": best.get("top3ADE"),
            "oracleModeRate": best.get("oracleModeRate"),
            "lr": train.get("lr"),
            "d_model": model.get("d_model"),
            "n_modes": model.get("n_modes"),
            "latent_dim": model.get("latent_dim"),
            "temporal_layers": model.get("n_temporal_layers"),
            "social_layers": model.get("n_social_layers"),
            "decoder_layers": model.get("n_decoder_layers"),
            "kl_weight": train.get("loss_kl_weight"),
            "diversity_weight": train.get("loss_diversity_weight"),
            "confidence_regret_weight": train.get("loss_confidence_regret_weight"),
        })

    rows.sort(key=lambda item: float(item["score"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["run", "score"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        print("rank  run                         score   mlADE  mlFDE  minADE  gapADE")
        for rank, row in enumerate(rows, 1):
            print(
                f"{rank:>4}  {row['run']:<27} "
                f"{float(row['score']):>6.3f}  {float(row['mlADE']):>6.3f} "
                f"{float(row['mlFDE']):>6.3f}  {float(row['minADE']):>6.3f} "
                f"{float(row['selectionGapADE']):>6.3f}"
            )
    else:
        print("no completed sweep histories found")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
