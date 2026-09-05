"""Load CitySim / NBDT trajectory CSVs into a metric internal schema."""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


FRAME_ALIASES = ("frameNUM", "frameNum", "frame")
CAR_ALIASES = ("carID", "carId", "car_id")
LANE_ALIASES = ("laneNumber", "laneId", "lane_id")


@dataclass
class SiteFiles:
    data_dir: Path
    csvs: List[Path]
    lanes_npy: Optional[Path]
    background: Optional[Path]
    lane_png: Optional[Path]


@dataclass
class Recording:
    """One CSV after metric conversion + per-track kinematics."""

    path: Path
    source_fps: float
    pixel_per_meter: float
    tracks: Dict[int, dict]
    n_frames: int
    frame_num_min: int


def discover_site(
    data_dir: str | Path,
    *,
    trajectory_dir: str | Path | None = None,
    csv_glob: str = "*.csv",
    exclude_globs: Sequence[str] = ("metadata.csv", "._*.csv", "*_located.csv"),
    lanes_npy: str | Path | None = None,
    background: str | Path | None = None,
    lane_png: str | Path | None = None,
) -> SiteFiles:
    root = Path(data_dir)
    traj_root = _resolve(root, trajectory_dir) if trajectory_dir else root
    csvs = sorted(
        p for p in traj_root.glob(csv_glob)
        if p.is_file() and not any(
            fnmatch(p.name.lower(), pat.lower()) for pat in exclude_globs
        )
    )
    npy = _resolve(root, lanes_npy) if lanes_npy else _first(
        root, ["*Lanes.npy", "*lanes.npy", "*.npy"]
    )
    bg = _resolve(root, background) if background else _first(
        root, ["*Background.png", "*background.png", "*Background.jpg"]
    )
    lane_img = _resolve(root, lane_png) if lane_png else _first(
        root, ["*Lane.png", "*lane.png"]
    )
    if not csvs:
        raise FileNotFoundError(
            f"no trajectory CSV matching {csv_glob!r} under {traj_root}"
        )
    for label, path in (("lanes_npy", npy), ("background", bg)):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return SiteFiles(
        data_dir=root, csvs=csvs, lanes_npy=npy, background=bg, lane_png=lane_img
    )


def _resolve(root: Path, path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else root / out


def _first(root: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def _col(df: pd.DataFrame, names: Tuple[str, ...]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)}")


def infer_pixel_per_meter(df: pd.DataFrame) -> float:
    """Recover px/m.

    Prefer NBDT ``carCenterX / carCenterXm``. Otherwise fit pixel ~ local
    ENU from lat/lon (CitySim drone ortho). ExpresswayA is ~17.91 px/m.
    """
    if {"carCenterX", "carCenterXm"}.issubset(df.columns):
        rx = (df["carCenterX"] / df["carCenterXm"]).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(rx) > 10:
            return float(np.median(np.abs(rx.to_numpy())))

    # CitySim FreewayC supplies world-aligned feet rather than lat/lon or metres.
    # Both axes use the same image origin, so pixel / (feet * 0.3048) is px/m.
    if {"carCenterX", "carCenterY", "carCenterXft", "carCenterYft"}.issubset(df.columns):
        ratios = []
        for px_col, ft_col in (("carCenterX", "carCenterXft"),
                               ("carCenterY", "carCenterYft")):
            metres = df[ft_col].to_numpy(np.float64) * 0.3048
            pixels = df[px_col].to_numpy(np.float64)
            valid = np.isfinite(metres) & np.isfinite(pixels) & (np.abs(metres) > 1.0)
            if np.any(valid):
                ratios.append(np.abs(pixels[valid] / metres[valid]))
        if ratios:
            ppm = float(np.median(np.concatenate(ratios)))
            if 1.0 <= ppm <= 200.0:
                return ppm

    need = {"carCenterX", "carCenterY", "carCenterLat", "carCenterLon"}
    if not need.issubset(df.columns):
        return 17.9129

    sample = df.sample(n=min(len(df), 4000), random_state=0)
    lat0 = float(sample["carCenterLat"].mean())
    lon0 = float(sample["carCenterLon"].mean())
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(lat0))
    east = (sample["carCenterLon"].to_numpy() - lon0) * m_per_lon
    north = (sample["carCenterLat"].to_numpy() - lat0) * m_per_lat
    A = np.stack([east, north], axis=1)
    A = np.concatenate([A, np.ones((len(A), 1))], axis=1)
    coef_x, _, _, _ = np.linalg.lstsq(A, sample["carCenterX"].to_numpy(), rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A, sample["carCenterY"].to_numpy(), rcond=None)
    lin = np.array([[coef_x[0], coef_x[1]], [coef_y[0], coef_y[1]]], dtype=np.float64)
    svals = np.linalg.svd(lin, compute_uv=False)
    ppm = float(np.mean(svals))
    if not np.isfinite(ppm) or ppm < 1.0 or ppm > 200.0:
        return 17.9129
    return ppm


def _bbox_lw_heading(
    bx: np.ndarray, by: np.ndarray
) -> Tuple[float, float, float]:
    """Length, width (m already) and long-axis heading from 4 corners."""
    pts = np.stack([bx, by], axis=1)
    edges = np.roll(pts, -1, axis=0) - pts
    elen = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(elen))
    heading = float(np.arctan2(edges[i, 1], edges[i, 0]))
    length = float(elen[i])
    width = float(elen[(i + 1) % 4])
    if length < 0.5:
        length = 4.5
    if width < 0.3:
        width = 1.8
    if width > length:
        length, width = width, length
        heading += 0.5 * np.pi
    return length, width, heading


def load_recording(
    csv_path: str | Path,
    *,
    source_fps: float,
    pixel_per_meter: Optional[float] = None,
    savgol_window: int = 11,
    savgol_poly: int = 3,
) -> Recording:
    path = Path(csv_path)
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    frame_c = _col(header, FRAME_ALIASES)
    car_c = _col(header, CAR_ALIASES)
    lane_c = _col(header, LANE_ALIASES) if any(c in columns for c in LANE_ALIASES) else None

    needed = {frame_c, car_c}
    if lane_c:
        needed.add(lane_c)
    for group in (
        ("carCenterX", "carCenterY", "carCenterXm", "carCenterYm",
         "carCenterXft", "carCenterYft", "carCenterLat", "carCenterLon"),
        tuple(f"boundingBox{i}{suffix}" for i in range(1, 5)
              for suffix in ("X", "Y", "Xm", "Ym", "Xft", "Yft")),
    ):
        needed.update(c for c in group if c in columns)
    df = pd.read_csv(path, usecols=sorted(needed))
    frame_c = _col(df, FRAME_ALIASES)
    car_c = _col(df, CAR_ALIASES)
    lane_c = _col(df, LANE_ALIASES) if any(c in df.columns for c in LANE_ALIASES) else None
    # Some distributed CitySim CSVs contain large blank tails.  Drop them before
    # integer conversion and also reject partially missing geometry below.
    df = df.dropna(subset=[frame_c, car_c]).copy()

    ppm = float(pixel_per_meter) if pixel_per_meter else infer_pixel_per_meter(df)

    if "carCenterXm" in df.columns and "carCenterYm" in df.columns:
        x_m = df["carCenterXm"].to_numpy(np.float64)
        y_m = df["carCenterYm"].to_numpy(np.float64)
        bb = np.stack(
            [
                df["boundingBox1Xm"], df["boundingBox1Ym"],
                df["boundingBox2Xm"], df["boundingBox2Ym"],
                df["boundingBox3Xm"], df["boundingBox3Ym"],
                df["boundingBox4Xm"], df["boundingBox4Ym"],
            ],
            axis=1,
        ).astype(np.float64)
    elif "carCenterXft" in df.columns and "carCenterYft" in df.columns:
        ft_to_m = 0.3048
        x_m = df["carCenterXft"].to_numpy(np.float64) * ft_to_m
        y_m = df["carCenterYft"].to_numpy(np.float64) * ft_to_m
        bb = np.stack(
            [
                df["boundingBox1Xft"], df["boundingBox1Yft"],
                df["boundingBox2Xft"], df["boundingBox2Yft"],
                df["boundingBox3Xft"], df["boundingBox3Yft"],
                df["boundingBox4Xft"], df["boundingBox4Yft"],
            ],
            axis=1,
        ).astype(np.float64) * ft_to_m
    else:
        x_m = df["carCenterX"].to_numpy(np.float64) / ppm
        y_m = df["carCenterY"].to_numpy(np.float64) / ppm
        bb = np.stack(
            [
                df["boundingBox1X"], df["boundingBox1Y"],
                df["boundingBox2X"], df["boundingBox2Y"],
                df["boundingBox3X"], df["boundingBox3Y"],
                df["boundingBox4X"], df["boundingBox4Y"],
            ],
            axis=1,
            dtype=np.float64,
        ) / ppm

    finite = np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(bb).all(axis=1)
    if not np.all(finite):
        df = df.loc[finite].copy()
        x_m = x_m[finite]
        y_m = y_m[finite]
        bb = bb[finite]
    if len(df) == 0:
        raise ValueError(f"{path}: no valid trajectory rows")

    rec = pd.DataFrame({
        "frame": df[frame_c].astype(np.int64),
        "car_id": df[car_c].astype(np.int64),
        "x": x_m,
        "y": y_m,
        "lane": df[lane_c].fillna(-1).astype(np.int64) if lane_c else -1,
        "bb": list(bb),
    })
    rec.sort_values(["car_id", "frame"], inplace=True)

    dt = 1.0 / source_fps
    win = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
    tracks: Dict[int, dict] = {}
    for cid, g in rec.groupby("car_id", sort=False):
        frames = g["frame"].to_numpy(np.int64)
        x = g["x"].to_numpy(np.float64)
        y = g["y"].to_numpy(np.float64)
        n = len(frames)
        if n >= win:
            vx = savgol_filter(x, win, savgol_poly, deriv=1, delta=dt, mode="interp")
            vy = savgol_filter(y, win, savgol_poly, deriv=1, delta=dt, mode="interp")
            ax = savgol_filter(x, win, savgol_poly, deriv=2, delta=dt, mode="interp")
            ay = savgol_filter(y, win, savgol_poly, deriv=2, delta=dt, mode="interp")
        else:
            vx = np.gradient(x, dt) if n >= 2 else np.zeros(n)
            vy = np.gradient(y, dt) if n >= 2 else np.zeros(n)
            ax = np.gradient(vx, dt) if n >= 2 else np.zeros(n)
            ay = np.gradient(vy, dt) if n >= 2 else np.zeros(n)

        speed = np.hypot(vx, vy)
        heading_v = np.arctan2(vy, vx)
        bb0 = np.asarray(g["bb"].iloc[0], dtype=np.float64).reshape(4, 2)
        length, width, heading_bb = _bbox_lw_heading(bb0[:, 0], bb0[:, 1])

        heading = heading_v.copy()
        slow = speed < 1.0
        heading[slow] = heading_bb
        # Flip 180° if bbox heading disagrees with motion.
        fast = ~slow
        c, s = np.cos(heading[fast]), np.sin(heading[fast])
        align = c * vx[fast] + s * vy[fast]
        heading[fast] = np.where(align < 0.0, heading[fast] + np.pi, heading[fast])
        heading = np.unwrap(heading)
        if n >= win:
            yaw_rate = savgol_filter(heading, win, savgol_poly, deriv=1, delta=dt, mode="interp")
        else:
            yaw_rate = np.gradient(heading, dt) if n >= 2 else np.zeros(n)

        tracks[int(cid)] = {
            "frames": frames,
            "x": x, "y": y,
            "vx": vx, "vy": vy,
            "ax": ax, "ay": ay,
            "heading": heading,
            "yaw_rate": yaw_rate,
            "speed": speed,
            "length": float(length),
            "width": float(width),
            "lane": g["lane"].to_numpy(np.int64),
        }

    n_frames = int(rec["frame"].max()) + 1 if len(rec) else 0
    return Recording(
        path=path,
        source_fps=float(source_fps),
        pixel_per_meter=float(ppm),
        tracks=tracks,
        n_frames=n_frames,
        frame_num_min=int(rec["frame"].min()) if len(rec) else 0,
    )
