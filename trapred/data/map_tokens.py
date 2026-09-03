"""World-frame map polylines → ego-centric tokens + per-agent lane context.

Channels follow ``ref_code/safetyfm/adapters/map_extractor.py`` with an
extra LANE_MARKING channel that carries solid/dashed type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from trapred.data.basemap import LaneEdgeBasemap
from trapred.data.lane_marking import (
    DASHED,
    SOLID,
    UNKNOWN,
    LaneMarking,
)

SRC_EDGE, SRC_CENTER, SRC_MARK = 0, 1, 2
N_SRC = 3
N_MARK = 4  # unknown, solid, dashed, mixed


@dataclass
class PolylineToken:
    points: np.ndarray          # [P, 4] x, y, sinθ, cosθ in ego frame
    source_type: int
    marking_type: int = UNKNOWN
    cluster_id: int = -1


@dataclass
class SiteMap:
    """Static site geometry in metric image coordinates."""

    edges: List[np.ndarray]
    centerlines: List[np.ndarray]
    markings: List[np.ndarray]          # [P, 3] x,y,theta
    marking_types: List[int]
    pixel_per_meter: float

    @classmethod
    def build(
        cls,
        basemap: LaneEdgeBasemap,
        lane_marks: Sequence[LaneMarking],
        n_samples: int,
    ) -> "SiteMap":
        marks_pl: List[np.ndarray] = []
        types: List[int] = []
        for m in lane_marks:
            pl = m.as_polyline_m(basemap.pixel_per_meter, n_samples)
            if pl is None:
                continue
            marks_pl.append(pl)
            types.append(int(m.marking))
        return cls(
            edges=basemap.as_drivable_edge_polylines(n_samples),
            centerlines=basemap.as_lane_centerline_polylines(n_samples),
            markings=marks_pl,
            marking_types=types,
            pixel_per_meter=basemap.pixel_per_meter,
        )


def world_to_ego_tokens(
    polylines: Sequence[np.ndarray],
    source_type: int,
    ego_xy: np.ndarray,
    ego_heading: float,
    *,
    radius_m: float,
    marking_types: Optional[Sequence[int]] = None,
) -> List[PolylineToken]:
    cos_e, sin_e = np.cos(-ego_heading), np.sin(-ego_heading)
    R = np.array([[cos_e, -sin_e], [sin_e, cos_e]], dtype=np.float64)
    tokens: List[PolylineToken] = []
    for cid, pl in enumerate(polylines):
        xy = pl[:, :2] - ego_xy[None, :]
        xy_rot = xy @ R.T
        if np.min(np.linalg.norm(xy_rot, axis=1)) > radius_m:
            continue
        theta_rel = pl[:, 2] - ego_heading
        pts = np.stack(
            [xy_rot[:, 0], xy_rot[:, 1], np.sin(theta_rel), np.cos(theta_rel)],
            axis=1,
        ).astype(np.float32)
        mt = int(marking_types[cid]) if marking_types is not None else UNKNOWN
        tokens.append(PolylineToken(
            points=pts, source_type=source_type, marking_type=mt, cluster_id=cid
        ))
    return tokens


def extract_tokens(
    site_map: SiteMap,
    ego_xy: np.ndarray,
    ego_heading: float,
    *,
    radius_m: float = 100.0,
    max_tokens: int = 48,
) -> List[PolylineToken]:
    tokens: List[PolylineToken] = []
    tokens.extend(world_to_ego_tokens(
        site_map.edges, SRC_EDGE, ego_xy, ego_heading, radius_m=radius_m
    ))
    tokens.extend(world_to_ego_tokens(
        site_map.centerlines, SRC_CENTER, ego_xy, ego_heading, radius_m=radius_m
    ))
    tokens.extend(world_to_ego_tokens(
        site_map.markings, SRC_MARK, ego_xy, ego_heading,
        radius_m=radius_m, marking_types=site_map.marking_types,
    ))
    # Keep nearest-to-ego tokens if we overflow.
    if len(tokens) > max_tokens:
        dist = [float(np.min(np.linalg.norm(t.points[:, :2], axis=1))) for t in tokens]
        order = np.argsort(dist)[:max_tokens]
        tokens = [tokens[i] for i in order]
    return tokens


def pack_tokens(
    tokens: Sequence[PolylineToken], max_tokens: int, n_pts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.zeros((max_tokens, n_pts, 4), dtype=np.float32)
    src = np.zeros((max_tokens,), dtype=np.int64)
    mark = np.zeros((max_tokens,), dtype=np.int64)
    valid = np.zeros((max_tokens,), dtype=np.float32)
    for i, t in enumerate(tokens[:max_tokens]):
        p = t.points
        n = min(n_pts, p.shape[0])
        pts[i, :n] = p[:n]
        if p.shape[0] < n_pts and p.shape[0] > 0:
            pts[i, n:] = p[-1]
        src[i] = t.source_type
        mark[i] = t.marking_type
        valid[i] = 1.0
    return pts, src, mark, valid


def lane_context_batch(
    xy: np.ndarray,
    heading: np.ndarray,
    site_map: SiteMap,
) -> np.ndarray:
    """Vectorized 6-D lane features for V positions, ``[V, 6]``."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    heading = np.asarray(heading, dtype=np.float64).reshape(-1)
    v = xy.shape[0]
    out = np.zeros((v, 6), dtype=np.float32)
    out[:, 1] = 1.0
    left_dir = np.stack([-np.sin(heading), np.cos(heading)], axis=1)

    if site_map.centerlines:
        cl = np.concatenate(site_map.centerlines, axis=0)
        d = np.hypot(cl[None, :, 0] - xy[:, None, 0], cl[None, :, 1] - xy[:, None, 1])
        j = d.argmin(axis=1)
        pt = cl[j]
        out[:, 0] = np.clip(((pt[:, :2] - xy) * left_dir).sum(axis=1), -8.0, 8.0)
        out[:, 1] = np.cos(pt[:, 2] - heading)

    left_d = np.full(v, 12.0)
    right_d = np.full(v, 12.0)
    left_dash = np.full(v, 0.5)
    right_dash = np.full(v, 0.5)
    for pl, mt in zip(site_map.markings, site_map.marking_types):
        d = np.hypot(pl[None, :, 0] - xy[:, None, 0], pl[None, :, 1] - xy[:, None, 1])
        j = d.argmin(axis=1)
        dist = d[np.arange(v), j]
        lat = ((pl[j, :2] - xy) * left_dir).sum(axis=1)
        dash = 1.0 if mt == DASHED else (0.0 if mt == SOLID else 0.5)
        take_l = (lat >= 0.0) & (dist < left_d)
        take_r = (lat < 0.0) & (dist < right_d)
        left_d = np.where(take_l, dist, left_d)
        right_d = np.where(take_r, dist, right_d)
        left_dash = np.where(take_l, dash, left_dash)
        right_dash = np.where(take_r, dash, right_dash)
    out[:, 2] = np.clip(left_d, 0.0, 12.0)
    out[:, 3] = np.clip(right_d, 0.0, 12.0)
    out[:, 4] = left_dash
    out[:, 5] = right_dash
    return out


def lane_context(xy: np.ndarray, heading: float, site_map: SiteMap) -> np.ndarray:
    return lane_context_batch(xy.reshape(1, 2), np.array([heading]), site_map)[0]
