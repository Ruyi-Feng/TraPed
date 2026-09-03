import math
from pathlib import Path

import numpy as np

from trapred.config import WindowCfg
from trapred.data.basemap import LaneEdgeBasemap
from trapred.data.citysim import Recording
from trapred.data.map_tokens import SiteMap
from trapred.data.windows import WindowBuilder, F_A, IDX_VALID


def _track(cid, n, x0, y0, vx, dt=1 / 30):
    frames = np.arange(n, dtype=np.int64)
    t = frames * dt
    x = x0 + vx * t
    y = np.full(n, y0)
    heading = np.full(n, 0.0)
    return cid, {
        "frames": frames,
        "x": x, "y": y,
        "vx": np.full(n, vx), "vy": np.zeros(n),
        "ax": np.zeros(n), "ay": np.zeros(n),
        "heading": heading, "yaw_rate": np.zeros(n),
        "speed": np.full(n, vx),
        "length": 4.5, "width": 1.8,
        "lane": np.zeros(n, dtype=np.int64),
    }


def test_ego_origin_and_neighbor_in_local_frame(tmp_path: Path):
    n = 400
    tracks = dict([
        _track(1, n, 0.0, 0.0, 20.0),
        _track(2, n, 30.0, 4.0, 18.0),
    ])
    rec = Recording(
        path=tmp_path / "t.csv",
        source_fps=30.0,
        pixel_per_meter=10.0,
        tracks=tracks,
        n_frames=n,
        frame_num_min=0,
    )
    poly = np.array([[-10, -2], [200, -2], [200, 2], [-10, 2]], dtype=float)
    npy = tmp_path / "lanes.npy"
    arr = np.empty(1, dtype=object)
    arr[0] = (poly * 10).reshape(-1, 1, 2)
    np.save(npy, arr, allow_pickle=True)
    bm = LaneEdgeBasemap.from_npy(npy, pixel_per_meter=10.0)
    site_map = SiteMap.build(bm, [], n_samples=16)
    cfg = WindowCfg(
        target_fps=10.0, t_input_s=2.0, t_horizon_s=3.0,
        n_agents=4, stride_s=1.0, max_neighbor_m=80.0,
        n_map_tokens=8, n_pts_polyline=16,
    )
    builder = WindowBuilder(cfg, 30.0)
    sample = next(builder.iter_samples(rec, site_map, train_frac=0.7, val_frac=0.15))
    assert sample.agents.shape[-1] == F_A
    last = cfg.t_input_s * builder.fps
    last = int(round(last)) - 1
    assert np.allclose(sample.agents[0, last, :2], 0.0, atol=1e-3)
    assert math.isclose(float(sample.agents[0, last, 8]), 0.0, abs_tol=1e-3)
    assert math.isclose(float(sample.agents[0, last, 9]), 1.0, abs_tol=1e-3)
    assert sample.agents[0, last, IDX_VALID] == 1.0
    assert sample.agents[1, last, IDX_VALID] == 1.0
    assert sample.agents[1, last, 0] > 10.0
    assert abs(float(sample.agents[1, last, 1]) - 4.0) < 0.5
    assert sample.future.shape[0] == builder.t_out
    assert sample.map_valid.sum() >= 1
