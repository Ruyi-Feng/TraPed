"""Detect solid vs dashed lane markings from an orthophoto.

Signal method from ``ref_code/lane_line_probe2.py``:

1. Find adjacent lane-polygon pairs and the shared boundary (mid of the
   facing edges at each x).
2. Dewarp a ±BAND strip along that boundary.
3. Threshold the 1-D intensity; duty cycle + gap count → SOLID / DASHED.

No VLM — the probe's "gen" branch is the objective labeler.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from trapred.data.basemap import polygon_long_edges, resample_open

SOLID, DASHED, MIXED, UNKNOWN = 1, 2, 3, 0
MARK_NAME = {0: "unknown", 1: "solid", 2: "dashed", 3: "mixed"}

BAND = 26
STEP = 2
GAP_RUN = 4
DUTY_SOLID = 0.90
GRAY_THR = 200


def lane_span_at_x(poly_px: np.ndarray, x: float) -> Optional[Tuple[float, float]]:
    """Min/max y of a closed polygon along the vertical line at ``x``."""
    pts = np.asarray(poly_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    ys: List[float] = []
    n = pts.shape[0]
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if abs(x1 - x0) < 1e-6:
            if abs(x0 - x) < 0.75:
                ys.extend([float(y0), float(y1)])
            continue
        lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
        if x < lo or x > hi:
            continue
        t = (x - x0) / (x1 - x0)
        ys.append(float(y0 + t * (y1 - y0)))
    if len(ys) < 2:
        return None
    return min(ys), max(ys)


def _boundary_y(
    upper: np.ndarray, lower: np.ndarray, x: float
) -> Optional[float]:
    su = lane_span_at_x(upper, x)
    sl = lane_span_at_x(lower, x)
    if su is None or sl is None:
        return None
    return 0.5 * (su[1] + sl[0])


def _classify_signal(sig: Sequence[float]) -> Tuple[int, float, int]:
    arr = np.asarray(sig, dtype=np.float64)
    if arr.size < 40:
        return UNKNOWN, 0.0, 0
    thr = GRAY_THR
    if np.percentile(arr, 90) < 160:
        thr = float(np.percentile(arr, 70))
    binary = arr > thr
    duty = float(binary.mean())
    gaps, run = 0, 0
    for v in binary:
        if not v:
            run += 1
        else:
            if run >= GAP_RUN:
                gaps += 1
            run = 0
    if run >= GAP_RUN:
        gaps += 1
    if duty < 0.08:
        return UNKNOWN, duty, gaps
    if duty > DUTY_SOLID and gaps == 0:
        return SOLID, duty, gaps
    if gaps >= 3 and duty >= 0.10:
        return DASHED, duty, gaps
    return MIXED, duty, gaps


def find_adjacent_pairs(
    polygons_px: Sequence[np.ndarray],
    *,
    max_gap_px: float = 18.0,
    min_overlap_frac: float = 0.25,
    step: int = 20,
) -> List[Tuple[int, int]]:
    """Pairs (upper_idx, lower_idx) whose y-spans nearly touch over x-overlap."""
    polys = [np.asarray(p).reshape(-1, 2) for p in polygons_px]
    n = len(polys)
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            xi = polys[i][:, 0]
            xj = polys[j][:, 0]
            x0 = int(max(xi.min(), xj.min()))
            x1 = int(min(xi.max(), xj.max()))
            if x1 - x0 < 80:
                continue
            close = 0
            total = 0
            i_above = 0
            for x in range(x0, x1, step):
                si = lane_span_at_x(polys[i], x)
                sj = lane_span_at_x(polys[j], x)
                if si is None or sj is None:
                    continue
                total += 1
                if si[1] <= sj[0]:
                    gap = sj[0] - si[1]
                    if 0.0 <= gap <= max_gap_px:
                        close += 1
                        i_above += 1
                elif sj[1] <= si[0]:
                    gap = si[0] - sj[1]
                    if 0.0 <= gap <= max_gap_px:
                        close += 1
                else:
                    # overlap in y — not a painted separator
                    pass
            if total < 8:
                continue
            if close / total < min_overlap_frac:
                continue
            if i_above >= close / 2:
                pairs.append((i, j))
            else:
                pairs.append((j, i))
    return pairs


@dataclass
class LaneMarking:
    """Shared boundary between two lane polygons, with line type."""

    upper_idx: int
    lower_idx: int
    xy_px: np.ndarray          # [K, 2] pixel samples along the boundary
    marking: int
    duty: float
    gaps: int

    def as_polyline_m(
        self, pixel_per_meter: float, n_samples: int
    ) -> Optional[np.ndarray]:
        return resample_open(self.xy_px / pixel_per_meter, n_samples)


def extract_lane_markings(
    polygons_px: Sequence[np.ndarray],
    background_bgr: np.ndarray,
    *,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[LaneMarking]:
    gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    polys = [np.asarray(p).reshape(-1, 2) for p in polygons_px]
    if pairs is None:
        pairs = find_adjacent_pairs(polys)
    out: List[LaneMarking] = []
    for u, l in pairs:
        xs = np.concatenate([polys[u][:, 0], polys[l][:, 0]])
        x0, x1 = int(xs.min()), int(xs.max())
        cols_xy: List[Tuple[float, float]] = []
        sig: List[float] = []
        for x in range(x0, x1, STEP):
            yb = _boundary_y(polys[u], polys[l], x)
            if yb is None:
                continue
            yi = int(round(yb))
            if yi - 5 < 0 or yi + 6 > h or x < 0 or x >= w:
                continue
            sig.append(float(gray[max(0, yi - 5):min(h, yi + 6), x].max()))
            cols_xy.append((float(x), float(yb)))
        if len(sig) < 80:
            continue
        marking, duty, gaps = _classify_signal(sig)
        if marking == UNKNOWN:
            continue
        out.append(LaneMarking(
            upper_idx=u, lower_idx=l,
            xy_px=np.asarray(cols_xy, dtype=np.float64),
            marking=marking, duty=duty, gaps=gaps,
        ))
    return out


def load_polygons_px(npy_path: str | Path) -> List[np.ndarray]:
    raw = np.load(str(npy_path), allow_pickle=True)
    return [np.asarray(item).reshape(-1, 2).astype(np.float64) for item in raw]


def _classify_polyline_on_image(
    xy_px: np.ndarray, gray: np.ndarray, *, step: int = 2
) -> Tuple[int, float, int, np.ndarray]:
    h, w = gray.shape
    sampled = []
    sig = []
    for i in range(0, len(xy_px), max(1, step)):
        x, y = xy_px[i]
        xi, yi = int(round(x)), int(round(y))
        if yi - 5 < 0 or yi + 6 > h or xi < 0 or xi >= w:
            continue
        sig.append(float(gray[yi - 5:yi + 6, xi].max()))
        sampled.append((float(x), float(y)))
    if len(sig) < 80:
        return UNKNOWN, 0.0, 0, np.zeros((0, 2))
    marking, duty, gaps = _classify_signal(sig)
    return marking, duty, gaps, np.asarray(sampled, dtype=np.float64)


def extract_outer_edge_markings(
    polygons_px: Sequence[np.ndarray],
    background_bgr: np.ndarray,
    existing: Sequence[LaneMarking],
    *,
    min_dist_px: float = 12.0,
) -> List[LaneMarking]:
    """Classify each polygon's long sides; keep edges not already covered."""
    gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    covered = [m.xy_px for m in existing if m.xy_px.size]
    extra: List[LaneMarking] = []
    for i, poly in enumerate(polygons_px):
        for edge in polygon_long_edges(np.asarray(poly).reshape(-1, 2)):
            mid = edge[len(edge) // 2]
            skip = False
            for cov in covered:
                d = np.hypot(cov[:, 0] - mid[0], cov[:, 1] - mid[1]).min()
                if d < min_dist_px:
                    skip = True
                    break
            if skip:
                continue
            dens = resample_open(edge, max(200, int(np.linalg.norm(edge[-1] - edge[0]) / 2)))
            if dens is None:
                continue
            marking, duty, gaps, xy = _classify_polyline_on_image(dens[:, :2], gray)
            if xy.shape[0] < 80 or marking == UNKNOWN:
                continue
            extra.append(LaneMarking(
                upper_idx=i, lower_idx=-1, xy_px=xy,
                marking=marking, duty=duty, gaps=gaps,
            ))
            covered.append(xy)
    return extra


def extract_from_site(
    npy_path: str | Path,
    background_path: str | Path,
) -> List[LaneMarking]:
    polys = load_polygons_px(npy_path)
    bg = cv2.imread(str(background_path))
    if bg is None:
        raise FileNotFoundError(background_path)
    marks = extract_lane_markings(polys, bg)
    marks.extend(extract_outer_edge_markings(polys, bg, marks))
    return marks
