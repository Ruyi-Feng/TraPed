"""Build / load a tensor cache of scene windows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from trapred.config import Cfg
from trapred.data.basemap import LaneEdgeBasemap
from trapred.data.citysim import Recording, discover_site, load_recording
from trapred.data.lane_marking import extract_from_site
from trapred.data.map_tokens import SiteMap
from trapred.data.windows import SceneSample, WindowBuilder


def _stratified_take(samples: List[SceneSample], k: int) -> List[SceneSample]:
    by = {}
    for s in samples:
        by.setdefault(s.split, []).append(s)
    # keep split proportions, at least 1 per nonempty split
    total = len(samples)
    out: List[SceneSample] = []
    rng = np.random.default_rng(0)
    for split, grp in by.items():
        n = max(1, int(round(k * len(grp) / total)))
        n = min(n, len(grp))
        idx = rng.choice(len(grp), size=n, replace=False)
        out.extend(grp[i] for i in sorted(idx.tolist()))
    return out[:k]


def _cache_paths(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "agents": root / "agents.npy",
        "future": root / "future.npy",
        "future_valid": root / "future_valid.npy",
        "map_pts": root / "map_pts.npy",
        "map_src": root / "map_src.npy",
        "map_mark": root / "map_mark.npy",
        "map_valid": root / "map_valid.npy",
        "meta": root / "meta.json",
    }


def build_site_map(cfg: Cfg, rec: Recording) -> Tuple[SiteMap, list]:
    files = discover_site(cfg.data_path)
    if files.lanes_npy is None:
        raise FileNotFoundError("site is missing a *Lanes.npy basemap")
    bm = LaneEdgeBasemap.from_npy(
        files.lanes_npy, pixel_per_meter=rec.pixel_per_meter
    )
    marks = []
    if files.background is not None:
        marks = extract_from_site(files.lanes_npy, files.background)
    site_map = SiteMap.build(bm, marks, cfg.window.n_pts_polyline)
    return site_map, marks


def build_cache(cfg: Cfg, *, force: bool = False) -> Path:
    cache_dir = cfg.out_path / "cache"
    paths = _cache_paths(cache_dir)
    if paths["agents"].exists() and not force:
        return cache_dir

    files = discover_site(cfg.data_path)
    ppm = cfg.site.pixel_per_meter
    samples: List[SceneSample] = []
    mark_summary = []
    for csv in files.csvs:
        rec = load_recording(
            csv,
            source_fps=cfg.site.source_fps,
            pixel_per_meter=ppm,
            savgol_window=cfg.window.savgol_window,
            savgol_poly=cfg.window.savgol_poly,
        )
        ppm = rec.pixel_per_meter
        site_map, marks = build_site_map(cfg, rec)
        mark_summary = [
            {
                "pair": [m.upper_idx, m.lower_idx],
                "type": int(m.marking),
                "duty": round(m.duty, 3),
                "gaps": int(m.gaps),
            }
            for m in marks
        ]
        builder = WindowBuilder(cfg.window, rec.source_fps)
        it = builder.iter_samples(
            rec, site_map,
            train_frac=cfg.split.train,
            val_frac=cfg.split.val,
        )
        for s in tqdm(it, desc=f"windows {csv.name}", unit="win"):
            samples.append(s)

    if not samples:
        raise RuntimeError("no scene windows produced — check FPS / horizon vs recording length")

    if cfg.train.max_windows is not None and len(samples) > cfg.train.max_windows:
        samples = _stratified_take(samples, cfg.train.max_windows)

    n = len(samples)
    na, tin, fa = samples[0].agents.shape
    tout = samples[0].future.shape[0]
    nm, p, fm = samples[0].map_pts.shape
    agents = np.zeros((n, na, tin, fa), np.float32)
    future = np.zeros((n, tout, 2), np.float32)
    future_valid = np.zeros((n, tout), np.float32)
    map_pts = np.zeros((n, nm, p, fm), np.float32)
    map_src = np.zeros((n, nm), np.int64)
    map_mark = np.zeros((n, nm), np.int64)
    map_valid = np.zeros((n, nm), np.float32)
    meta = {
        "n": n,
        "pixel_per_meter": float(ppm),
        "t_in": tin,
        "t_out": tout,
        "n_agents": na,
        "f_a": fa,
        "markings": mark_summary,
        "fps": cfg.window.target_fps,
        "rows": [],
    }
    for i, s in enumerate(samples):
        agents[i] = s.agents
        future[i] = s.future
        future_valid[i] = s.future_valid
        map_pts[i] = s.map_pts
        map_src[i] = s.map_src
        map_mark[i] = s.map_mark
        map_valid[i] = s.map_valid
        meta["rows"].append({
            "i": i,
            "split": s.split,
            "ego_id": s.ego_id,
            "start_frame": s.start_frame,
            "t_last": s.t_last,
            "csv": s.csv_stem,
        })

    np.save(paths["agents"], agents)
    np.save(paths["future"], future)
    np.save(paths["future_valid"], future_valid)
    np.save(paths["map_pts"], map_pts)
    np.save(paths["map_src"], map_src)
    np.save(paths["map_mark"], map_mark)
    np.save(paths["map_valid"], map_valid)
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    counts = {}
    for r in meta["rows"]:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"cache {cache_dir}  n={n}  splits={counts}  ppm={ppm:.4f}  markings={len(mark_summary)}")
    return cache_dir


class SceneDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str) -> None:
        cache_dir = Path(cache_dir)
        paths = _cache_paths(cache_dir)
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.index = [r["i"] for r in meta["rows"] if r["split"] == split]
        if not self.index:
            raise RuntimeError(f"split {split!r} is empty in {cache_dir}")
        self.agents = np.load(paths["agents"], mmap_mode="r")
        self.future = np.load(paths["future"], mmap_mode="r")
        self.future_valid = np.load(paths["future_valid"], mmap_mode="r")
        self.map_pts = np.load(paths["map_pts"], mmap_mode="r")
        self.map_src = np.load(paths["map_src"], mmap_mode="r")
        self.map_mark = np.load(paths["map_mark"], mmap_mode="r")
        self.map_valid = np.load(paths["map_valid"], mmap_mode="r")
        self.rows = {r["i"]: r for r in meta["rows"]}
        self.meta = meta
        self.split = split

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        i = self.index[idx]
        return {
            "agents": torch.from_numpy(np.array(self.agents[i], copy=True)),
            "future": torch.from_numpy(np.array(self.future[i], copy=True)),
            "future_valid": torch.from_numpy(np.array(self.future_valid[i], copy=True)),
            "map_pts": torch.from_numpy(np.array(self.map_pts[i], copy=True)),
            "map_src": torch.from_numpy(np.array(self.map_src[i], copy=True)),
            "map_mark": torch.from_numpy(np.array(self.map_mark[i], copy=True)),
            "map_valid": torch.from_numpy(np.array(self.map_valid[i], copy=True)),
        }
