"""Lane-edge polygons (.npy, pixel) → metric polylines.

Adapted from ``ref_code/safetyfm/adapters/basemap.py``: geometric
centerlines via PCA midline, drivable-edge resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


def polygon_long_edges(
    poly_xy: np.ndarray,
    *,
    min_aspect_ratio: float = 2.0,
) -> List[np.ndarray]:
    """The two long sides of an elongated polygon (pixel or metric)."""
    pts = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return []
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    if S[1] < 1e-6 or S[0] / S[1] < min_aspect_ratio:
        return []
    rotated = centered @ Vt.T
    edges = []
    for mask in (rotated[:, 1] >= 0.0, rotated[:, 1] < 0.0):
        if mask.sum() < 2:
            continue
        side = rotated[mask]
        side = side[np.argsort(side[:, 0])]
        world = side @ Vt + centroid
        if np.linalg.norm(world[-1] - world[0]) < 5.0:
            continue
        edges.append(world)
    return edges


def polygon_midline(
    poly_xy: np.ndarray,
    *,
    n_samples: int = 20,
    min_aspect_ratio: float = 2.0,
) -> Optional[np.ndarray]:
    pts = np.asarray(poly_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    if S[1] < 1e-6 or S[0] / S[1] < min_aspect_ratio:
        return None
    rotated = centered @ Vt.T
    upper_mask = rotated[:, 1] >= 0.0
    lower_mask = ~upper_mask
    if upper_mask.sum() < 2 or lower_mask.sum() < 2:
        return None
    upper = rotated[upper_mask]
    lower = rotated[lower_mask]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    x_lo = max(upper[0, 0], lower[0, 0])
    x_hi = min(upper[-1, 0], lower[-1, 0])
    if x_hi - x_lo < 1.0:
        return None
    x_samples = np.linspace(x_lo, x_hi, n_samples)
    y_upper = np.interp(x_samples, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_samples, lower[:, 0], lower[:, 1])
    y_mid = 0.5 * (y_upper + y_lower)
    midline_world = np.stack([x_samples, y_mid], axis=1) @ Vt + centroid
    tx = np.gradient(midline_world[:, 0])
    ty = np.gradient(midline_world[:, 1])
    theta = np.arctan2(ty, tx)
    return np.stack([midline_world[:, 0], midline_world[:, 1], theta], axis=1)


def resample_closed(poly: np.ndarray, n_samples: int) -> Optional[np.ndarray]:
    pts = np.vstack([poly, poly[:1]])
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-6:
        return None
    s = np.linspace(0.0, total, n_samples)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    theta = np.arctan2(np.gradient(y), np.gradient(x))
    return np.stack([x, y, theta], axis=1)


def resample_open(xy: np.ndarray, n_samples: int) -> Optional[np.ndarray]:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return None
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-6:
        return None
    s = np.linspace(0.0, total, n_samples)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    theta = np.arctan2(np.gradient(y), np.gradient(x))
    return np.stack([x, y, theta], axis=1)


@dataclass
class LaneEdgeBasemap:
    polygons_m: List[np.ndarray]
    pixel_per_meter: float

    @classmethod
    def from_npy(
        cls, npy_path: str | Path, *, pixel_per_meter: float
    ) -> "LaneEdgeBasemap":
        if pixel_per_meter <= 0:
            raise ValueError("pixel_per_meter must be > 0")
        raw = np.load(str(npy_path), allow_pickle=True)
        polygons_m: List[np.ndarray] = []
        for item in raw:
            arr = np.asarray(item).reshape(-1, 2).astype(np.float64)
            if arr.shape[0] < 3:
                continue
            polygons_m.append(arr / pixel_per_meter)
        if not polygons_m:
            raise ValueError(f"{npy_path}: no usable polygons")
        return cls(polygons_m=polygons_m, pixel_per_meter=float(pixel_per_meter))

    def n_lanes(self) -> int:
        return len(self.polygons_m)

    def as_drivable_edge_polylines(self, n_samples: int = 20) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for poly in self.polygons_m:
            pl = resample_closed(poly, n_samples)
            if pl is not None:
                out.append(pl)
        return out

    def as_lane_centerline_polylines(
        self, n_samples: int = 20, min_aspect_ratio: float = 2.0
    ) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for poly in self.polygons_m:
            mid = polygon_midline(
                poly, n_samples=n_samples, min_aspect_ratio=min_aspect_ratio
            )
            if mid is not None:
                out.append(mid)
        return out
