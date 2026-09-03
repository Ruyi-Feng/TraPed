"""Dump a lane-marking overlay (solid vs dashed) for a site folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trapred.data.citysim import discover_site
from trapred.data.lane_marking import MARK_NAME, extract_from_site, load_polygons_px


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/expresswayAsample"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    files = discover_site(args.data_dir)
    if files.lanes_npy is None or files.background is None:
        raise SystemExit("need *Lanes.npy and *Background.png")
    marks = extract_from_site(files.lanes_npy, files.background)
    bg = cv2.cvtColor(cv2.imread(str(files.background)), cv2.COLOR_BGR2RGB)
    polys = load_polygons_px(files.lanes_npy)
    colors = {1: "#dc2626", 2: "#2563eb", 3: "#f59e0b", 0: "#64748b"}
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(bg)
    for i, p in enumerate(polys):
        closed = np.vstack([p, p[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="#94a3b8", lw=0.6, alpha=0.6)
        c = p.mean(axis=0)
        ax.text(c[0], c[1], str(i), color="white", fontsize=8)
    for m in marks:
        ax.plot(m.xy_px[:, 0], m.xy_px[:, 1], color=colors[m.marking], lw=2.0,
                label=MARK_NAME[m.marking])
        print(f"lanes {m.upper_idx}|{m.lower_idx}  {MARK_NAME[m.marking]:7s}  "
              f"duty={m.duty:.2f} gaps={m.gaps}")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")
    ax.set_title("lane markings (red=solid, blue=dashed, amber=mixed)")
    ax.set_axis_off()
    fig.tight_layout()
    out = args.out or Path("outputs") / "lane_markings.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
