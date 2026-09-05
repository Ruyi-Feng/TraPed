"""Build / load a tensor cache of scene windows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from trapred.config import Cfg, SiteCfg
from trapred.data.basemap import LaneEdgeBasemap
from trapred.data.citysim import Recording, discover_site, load_recording
from trapred.data.lane_marking import extract_from_site
from trapred.data.map_tokens import SiteMap
from trapred.data.windows import SceneSample, WindowBuilder


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


def _discover_configured_site(site: SiteCfg):
    return discover_site(
        site.data_dir,
        trajectory_dir=site.trajectory_dir,
        csv_glob=site.csv_glob,
        exclude_globs=site.exclude_globs,
        lanes_npy=site.lanes_npy,
        background=site.background,
        lane_png=site.lane_png,
    )


def build_site_map(
    cfg_or_site: Cfg | SiteCfg,
    rec: Recording,
    *,
    files=None,
    marks: Optional[list] = None,
) -> Tuple[SiteMap, list]:
    site = cfg_or_site.site_configs[0] if isinstance(cfg_or_site, Cfg) else cfg_or_site
    files = files or _discover_configured_site(site)
    if files.lanes_npy is None:
        raise FileNotFoundError("site is missing a *Lanes.npy basemap")
    bm = LaneEdgeBasemap.from_npy(
        files.lanes_npy, pixel_per_meter=rec.pixel_per_meter
    )
    if marks is None:
        marks = []
        if files.background is not None:
            marks = extract_from_site(files.lanes_npy, files.background)
    window = cfg_or_site.window if isinstance(cfg_or_site, Cfg) else None
    n_pts = window.n_pts_polyline if window is not None else 20
    site_map = SiteMap.build(bm, marks, n_pts)
    return site_map, marks


def _split_capacities(total: int, train_frac: float, val_frac: float) -> Dict[str, int]:
    train = int(round(total * train_frac))
    val = int(round(total * val_frac))
    train = min(max(train, 0), total)
    val = min(max(val, 0), total - train)
    return {"train": train, "val": val, "test": total - train - val}


def _bucket_capacities(cfg: Cfg) -> Dict[Tuple[str, str], int]:
    total = cfg.train.max_windows
    if total is None or total <= 0:
        raise ValueError("train.max_windows must be a positive integer for disk streaming")
    keys = [s.name for s in cfg.site_configs] if cfg.train.balance_sites else ["*"]
    per_site = [total // len(keys)] * len(keys)
    for i in range(total % len(keys)):
        per_site[i] += 1
    out: Dict[Tuple[str, str], int] = {}
    for key, count in zip(keys, per_site):
        for split, cap in _split_capacities(count, cfg.split.train, cfg.split.val).items():
            out[(key, split)] = cap
    return out


class _DiskReservoir:
    """Bounded, split/site-stratified reservoir backed by .npy memmaps."""

    def __init__(self, paths: Dict[str, Path], cfg: Cfg, first: SceneSample) -> None:
        self.paths = paths
        self.balance_sites = cfg.train.balance_sites
        self.capacities = _bucket_capacities(cfg)
        self.offsets: Dict[Tuple[str, str], int] = {}
        offset = 0
        for key, cap in self.capacities.items():
            self.offsets[key] = offset
            offset += cap
        self.capacity = offset
        na, tin, fa = first.agents.shape
        tout = first.future.shape[0]
        nm, npts, fmap = first.map_pts.shape
        open_mm = np.lib.format.open_memmap
        self.arrays = {
            "agents": open_mm(paths["agents"], mode="w+", dtype=np.float32,
                              shape=(offset, na, tin, fa)),
            "future": open_mm(paths["future"], mode="w+", dtype=np.float32,
                              shape=(offset, tout, 2)),
            "future_valid": open_mm(paths["future_valid"], mode="w+", dtype=np.float32,
                                    shape=(offset, tout)),
            "map_pts": open_mm(paths["map_pts"], mode="w+", dtype=np.float32,
                               shape=(offset, nm, npts, fmap)),
            "map_src": open_mm(paths["map_src"], mode="w+", dtype=np.int64,
                               shape=(offset, nm)),
            "map_mark": open_mm(paths["map_mark"], mode="w+", dtype=np.int64,
                                shape=(offset, nm)),
            "map_valid": open_mm(paths["map_valid"], mode="w+", dtype=np.float32,
                                 shape=(offset, nm)),
        }
        self.shapes = {"t_in": tin, "t_out": tout, "n_agents": na, "f_a": fa}
        self.seen = {key: 0 for key in self.capacities}
        self.kept = {key: 0 for key in self.capacities}
        self.rows: Dict[int, dict] = {}
        self.rng = np.random.default_rng(cfg.train.seed)

    def add(self, sample: SceneSample) -> None:
        site_key = sample.site if self.balance_sites else "*"
        key = (site_key, sample.split)
        cap = self.capacities.get(key, 0)
        if cap <= 0:
            return
        self.seen[key] += 1
        if self.kept[key] < cap:
            local = self.kept[key]
            self.kept[key] += 1
        else:
            local = int(self.rng.integers(0, self.seen[key]))
            if local >= cap:
                return
        slot = self.offsets[key] + local
        self.arrays["agents"][slot] = sample.agents
        self.arrays["future"][slot] = sample.future
        self.arrays["future_valid"][slot] = sample.future_valid
        self.arrays["map_pts"][slot] = sample.map_pts
        self.arrays["map_src"][slot] = sample.map_src
        self.arrays["map_mark"][slot] = sample.map_mark
        self.arrays["map_valid"][slot] = sample.map_valid
        self.rows[slot] = {
            "i": slot,
            "split": sample.split,
            "site": sample.site,
            "ego_id": sample.ego_id,
            "start_frame": sample.start_frame,
            "t_last": sample.t_last,
            "csv": sample.csv_stem,
        }

    def finish(self) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
        for arr in self.arrays.values():
            arr.flush()
        rows = [self.rows[i] for i in sorted(self.rows)]
        seen = {f"{k[0]}/{k[1]}": v for k, v in self.seen.items()}
        kept = {f"{k[0]}/{k[1]}": v for k, v in self.kept.items()}
        return rows, seen, kept


def _save_in_memory(samples: List[SceneSample], paths: Dict[str, Path]) -> Tuple[List[dict], dict]:
    n = len(samples)
    na, tin, fa = samples[0].agents.shape
    tout = samples[0].future.shape[0]
    nm, p, fm = samples[0].map_pts.shape
    arrays = {
        "agents": np.zeros((n, na, tin, fa), np.float32),
        "future": np.zeros((n, tout, 2), np.float32),
        "future_valid": np.zeros((n, tout), np.float32),
        "map_pts": np.zeros((n, nm, p, fm), np.float32),
        "map_src": np.zeros((n, nm), np.int64),
        "map_mark": np.zeros((n, nm), np.int64),
        "map_valid": np.zeros((n, nm), np.float32),
    }
    rows = []
    for i, sample in enumerate(samples):
        for name in arrays:
            arrays[name][i] = getattr(sample, name)
        rows.append({
            "i": i, "split": sample.split, "site": sample.site,
            "ego_id": sample.ego_id, "start_frame": sample.start_frame,
            "t_last": sample.t_last, "csv": sample.csv_stem,
        })
    for name, arr in arrays.items():
        np.save(paths[name], arr)
    return rows, {"t_in": tin, "t_out": tout, "n_agents": na, "f_a": fa}


def build_cache(cfg: Cfg, *, force: bool = False) -> Path:
    cache_dir = cfg.out_path / "cache"
    paths = _cache_paths(cache_dir)
    required = list(paths.values())
    if all(path.exists() for path in required) and not force:
        return cache_dir
    if force:
        for path in required:
            if path.exists():
                path.unlink()

    samples: List[SceneSample] = []
    reservoir: Optional[_DiskReservoir] = None
    mark_summary: Dict[str, list] = {}
    ppm_summary: Dict[str, Dict[str, float]] = {}
    source_counts: Dict[str, int] = {}

    for site in cfg.site_configs:
        files = _discover_configured_site(site)
        if files.lanes_npy is None:
            raise FileNotFoundError(f"{site.name}: missing lane polygon .npy")
        marks = extract_from_site(files.lanes_npy, files.background) if files.background else []
        mark_summary[site.name] = [
            {"pair": [m.upper_idx, m.lower_idx], "type": int(m.marking),
             "duty": round(m.duty, 3), "gaps": int(m.gaps)}
            for m in marks
        ]
        ppm_summary[site.name] = {}
        source_counts[site.name] = len(files.csvs)
        for csv in files.csvs:
            rec = load_recording(
                csv, source_fps=site.source_fps,
                pixel_per_meter=site.pixel_per_meter,
                savgol_window=cfg.window.savgol_window,
                savgol_poly=cfg.window.savgol_poly,
            )
            ppm_summary[site.name][csv.stem] = round(rec.pixel_per_meter, 6)
            site_map, _ = build_site_map(cfg, rec, files=files, marks=marks)
            builder = WindowBuilder(cfg.window, rec.source_fps)
            iterator = builder.iter_samples(
                rec, site_map, train_frac=cfg.split.train, val_frac=cfg.split.val,
            )
            for sample in tqdm(iterator, desc=f"windows {site.name}/{csv.name}", unit="win"):
                sample.site = site.name
                if cfg.train.max_windows is None:
                    samples.append(sample)
                else:
                    if reservoir is None:
                        reservoir = _DiskReservoir(paths, cfg, sample)
                    reservoir.add(sample)

    if reservoir is None and not samples:
        raise RuntimeError("no scene windows produced — check FPS / horizon vs recording length")

    if reservoir is not None:
        rows, seen, kept = reservoir.finish()
        shapes = reservoir.shapes
        capacity = reservoir.capacity
    else:
        rows, shapes = _save_in_memory(samples, paths)
        seen = {}
        kept = {}
        capacity = len(rows)
    counts: Dict[str, int] = {}
    site_counts: Dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        site_counts[row["site"]] = site_counts.get(row["site"], 0) + 1
    meta = {
        "cache_version": 2,
        "n": len(rows),
        "capacity": capacity,
        "pixel_per_meter": ppm_summary,
        **shapes,
        "markings": mark_summary,
        "fps": cfg.window.target_fps,
        "source_csv_counts": source_counts,
        "split_counts": counts,
        "site_counts": site_counts,
        "reservoir_seen": seen,
        "reservoir_kept": kept,
        "rows": rows,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cache {cache_dir}  n={len(rows)}  splits={counts}  sites={site_counts}")
    return cache_dir


class SceneDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str, site: Optional[str] = None) -> None:
        cache_dir = Path(cache_dir)
        paths = _cache_paths(cache_dir)
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.index = [
            r["i"] for r in meta["rows"]
            if r["split"] == split and (site is None or r.get("site", "") == site)
        ]
        if not self.index:
            suffix = f" for site {site!r}" if site is not None else ""
            raise RuntimeError(f"split {split!r}{suffix} is empty in {cache_dir}")
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
        self.site = site

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
