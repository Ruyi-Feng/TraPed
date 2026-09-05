"""Sliding ego-centric scene windows with neighbors + map tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from trapred.config import WindowCfg
from trapred.data.citysim import Recording
from trapred.data.map_tokens import SiteMap, extract_tokens, lane_context_batch, pack_tokens
from trapred.data.splits import assign_split

# Feature layout of agents[n, t, :]
# 0-10 motion, 11-12 geom, 13-18 lane context, 19 is_ego, 20 valid
F_A = 21
IDX_VALID = 20
IDX_EGO = 19
SLICE_LANE = slice(13, 19)


@dataclass
class SceneSample:
    agents: np.ndarray
    future: np.ndarray          # [T_out, 2] ego xy in last-obs ego frame
    future_valid: np.ndarray    # [T_out]
    map_pts: np.ndarray
    map_src: np.ndarray
    map_mark: np.ndarray
    map_valid: np.ndarray
    ego_id: int
    start_frame: int
    t_last: int
    split: str
    csv_stem: str
    site: str = ""


def _downsample_track(tr: dict, stride: int, frame0: int) -> dict:
    frames = tr["frames"]
    keep = (frames - frame0) % stride == 0
    if not np.any(keep):
        return {}
    out = {}
    for k, v in tr.items():
        if isinstance(v, np.ndarray) and v.shape[0] == frames.shape[0]:
            out[k] = v[keep]
        else:
            out[k] = v
    # Re-index to 10 Hz grid: frameIdx = (frame - frame0) // stride
    out["idx"] = ((out["frames"] - frame0) // stride).astype(np.int64)
    return out


def _rot(heading: float) -> np.ndarray:
    c, s = math.cos(-heading), math.sin(-heading)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class WindowBuilder:
    def __init__(self, cfg: WindowCfg, source_fps: float) -> None:
        self.cfg = cfg
        self.source_fps = source_fps
        self.stride = max(1, int(round(source_fps / cfg.target_fps)))
        self.fps = source_fps / self.stride
        self.t_in = int(round(cfg.t_input_s * self.fps))
        self.t_out = int(round(cfg.t_horizon_s * self.fps))
        self.stride_frames = max(1, int(round(cfg.stride_s * self.fps)))

    def iter_samples(
        self,
        rec: Recording,
        site_map: SiteMap,
        *,
        train_frac: float,
        val_frac: float,
    ) -> Iterator[SceneSample]:
        frame0 = rec.frame_num_min
        tracks: Dict[int, dict] = {}
        lookups: Dict[int, Dict[int, int]] = {}
        for cid, tr in rec.tracks.items():
            ds = _downsample_track(tr, self.stride, frame0)
            if not ds or ds["idx"].size < self.t_in:
                continue
            tracks[cid] = ds
            lookups[cid] = {int(f): i for i, f in enumerate(ds["idx"])}

        n_idx = 0
        for tr in tracks.values():
            if tr["idx"].size:
                n_idx = max(n_idx, int(tr["idx"].max()) + 1)

        csv_stem = rec.path.stem
        for t_last in range(self.t_in - 1, n_idx - self.t_out, self.stride_frames):
            split = assign_split(
                t_last,
                n_idx,
                fps=self.fps,
                t_input_s=self.cfg.t_input_s,
                t_horizon_s=self.cfg.t_horizon_s,
                train_frac=train_frac,
                val_frac=val_frac,
            )
            if split == "discard":
                continue
            egos = [cid for cid, lk in lookups.items() if t_last in lk]
            for ego_id in egos:
                sample = self._build_one(
                    ego_id, t_last, tracks, lookups, site_map, split, csv_stem, frame0
                )
                if sample is None:
                    continue
                yield sample

    def _build_one(
        self,
        ego_id: int,
        t_last: int,
        tracks: Dict[int, dict],
        lookups: Dict[int, Dict[int, int]],
        site_map: SiteMap,
        split: str,
        csv_stem: str,
        frame0: int,
    ) -> Optional[SceneSample]:
        cfg = self.cfg
        t_in, t_out = self.t_in, self.t_out
        ego = tracks[ego_id]
        lk_e = lookups[ego_id]
        ego_rows = [lk_e.get(t_last - (t_in - 1) + k) for k in range(t_in)]
        if any(r is None for r in ego_rows):
            return None
        fut_rows = [lk_e.get(t_last + 1 + k) for k in range(t_out)]
        if sum(r is not None for r in fut_rows) < int(0.8 * t_out):
            return None

        er = lk_e[t_last]
        ego_xy = np.array([ego["x"][er], ego["y"][er]], dtype=np.float64)
        ego_h = float(ego["heading"][er])
        R = _rot(ego_h)

        neigh: List[Tuple[int, float]] = []
        for cid, lk in lookups.items():
            row = lk.get(t_last)
            if row is None:
                continue
            tr = tracks[cid]
            d = math.hypot(tr["x"][row] - ego_xy[0], tr["y"][row] - ego_xy[1])
            if cid == ego_id:
                d = -1.0
            elif d > cfg.max_neighbor_m:
                continue
            neigh.append((cid, d))
        neigh.sort(key=lambda t: t[1])
        neigh = neigh[: cfg.n_agents]

        agents = np.zeros((cfg.n_agents, t_in, F_A), dtype=np.float32)
        ctx_xy, ctx_h, ctx_idx = [], [], []
        for slot, (cid, _) in enumerate(neigh):
            tr = tracks[cid]
            lk = lookups[cid]
            is_ego = 1.0 if cid == ego_id else 0.0
            for k in range(t_in):
                gf = t_last - (t_in - 1) + k
                row = lk.get(gf)
                if row is None:
                    continue
                self._fill_motion(
                    agents[slot, k], tr, row, ego_xy, R, ego_h, is_ego
                )
                ctx_xy.append([tr["x"][row], tr["y"][row]])
                ctx_h.append(tr["heading"][row])
                ctx_idx.append((slot, k))
        if ctx_xy:
            lc = lane_context_batch(
                np.asarray(ctx_xy, dtype=np.float64),
                np.asarray(ctx_h, dtype=np.float64),
                site_map,
            )
            for (slot, k), feat in zip(ctx_idx, lc):
                agents[slot, k, 13:19] = feat

        future = np.zeros((t_out, 2), dtype=np.float32)
        future_valid = np.zeros((t_out,), dtype=np.float32)
        for k in range(t_out):
            row = fut_rows[k]
            if row is None:
                continue
            rel = R @ np.array(
                [ego["x"][row] - ego_xy[0], ego["y"][row] - ego_xy[1]],
                dtype=np.float64,
            )
            future[k] = rel.astype(np.float32)
            future_valid[k] = 1.0

        tokens = extract_tokens(
            site_map, ego_xy, ego_h,
            radius_m=cfg.map_radius_m, max_tokens=cfg.n_map_tokens,
        )
        map_pts, map_src, map_mark, map_valid = pack_tokens(
            tokens, cfg.n_map_tokens, cfg.n_pts_polyline
        )
        start_frame = int(ego["frames"][lk_e[t_last - (t_in - 1)]])
        return SceneSample(
            agents=agents,
            future=future,
            future_valid=future_valid,
            map_pts=map_pts,
            map_src=map_src,
            map_mark=map_mark,
            map_valid=map_valid,
            ego_id=int(ego_id),
            start_frame=start_frame,
            t_last=int(t_last),
            split=split,
            csv_stem=csv_stem,
        )

    @staticmethod
    def _fill_motion(
        out: np.ndarray,
        tr: dict,
        row: int,
        ego_xy: np.ndarray,
        R: np.ndarray,
        ego_h: float,
        is_ego: float,
    ) -> None:
        xy = np.array([tr["x"][row], tr["y"][row]], dtype=np.float64)
        rel = R @ (xy - ego_xy)
        v = R @ np.array([tr["vx"][row], tr["vy"][row]], dtype=np.float64)
        a = R @ np.array([tr["ax"][row], tr["ay"][row]], dtype=np.float64)
        h_rel = tr["heading"][row] - ego_h
        out[0] = rel[0]
        out[1] = rel[1]
        out[2] = v[0]
        out[3] = v[1]
        out[4] = a[0]
        out[5] = a[1]
        out[6] = math.hypot(tr["vx"][row], tr["vy"][row])
        out[7] = math.hypot(tr["ax"][row], tr["ay"][row])
        out[8] = math.sin(h_rel)
        out[9] = math.cos(h_rel)
        out[10] = tr["yaw_rate"][row]
        out[11] = tr["length"]
        out[12] = tr["width"]
        out[19] = is_ego
        out[20] = 1.0
