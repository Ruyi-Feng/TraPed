from pathlib import Path

import numpy as np
import pandas as pd

from trapred.config import Cfg, SiteCfg, load_config
from trapred.data.citysim import discover_site, load_recording
from trapred.data.dataset import _DiskReservoir, _cache_paths
from trapred.data.windows import SceneSample


def _freeway_c_frame(n: int = 6, ppm: float = 10.0) -> pd.DataFrame:
    rows = []
    for frame in range(n):
        x_ft = 100.0 + frame
        y_ft = 50.0
        row = {
            "frameNum": frame, "carId": 1, "laneId": 2,
            "carCenterXft": x_ft, "carCenterYft": y_ft,
            "carCenterX": x_ft * 0.3048 * ppm,
            "carCenterY": y_ft * 0.3048 * ppm,
        }
        corners = [
            (x_ft + 2.0, y_ft + 1.0), (x_ft - 2.0, y_ft + 1.0),
            (x_ft - 2.0, y_ft - 1.0), (x_ft + 2.0, y_ft - 1.0),
        ]
        for i, (x, y) in enumerate(corners, 1):
            row[f"boundingBox{i}Xft"] = x
            row[f"boundingBox{i}Yft"] = y
        rows.append(row)
    return pd.DataFrame(rows)


def test_freeway_c_feet_are_loaded_as_metres(tmp_path: Path):
    csv = tmp_path / "FreewayC-01.csv"
    frame = _freeway_c_frame()
    frame.loc[len(frame)] = np.nan
    frame.to_csv(csv, index=False)
    rec = load_recording(csv, source_fps=30.0, savgol_window=5)
    assert np.isclose(rec.pixel_per_meter, 10.0, atol=1e-5)
    assert np.isclose(rec.tracks[1]["x"][0], 30.48, atol=1e-5)
    assert np.isclose(rec.tracks[1]["length"], 4.0 * 0.3048, atol=1e-5)


def test_discovery_uses_trajectory_subdir_and_excludes_located(tmp_path: Path):
    traj = tmp_path / "Trajectories"
    traj.mkdir()
    for name in ("FreewayC-01.csv", "FreewayC-01_located.csv", "._junk.csv"):
        (traj / name).write_text("frameNum,carId\n", encoding="utf-8")
    (tmp_path / "FreewayCLanes.npy").write_bytes(b"npy")
    (tmp_path / "freewayCbackground.png").write_bytes(b"png")
    files = discover_site(
        tmp_path, trajectory_dir="Trajectories", csv_glob="FreewayC-*.csv",
        lanes_npy="FreewayCLanes.npy", background="freewayCbackground.png",
    )
    assert [p.name for p in files.csvs] == ["FreewayC-01.csv"]


def test_multisite_yaml_is_parsed(tmp_path: Path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        "site:\n  name: MultiSite\n"
        "sites:\n"
        "  - name: A\n    data_dir: D:/a\n    csv_glob: 'a-*.csv'\n"
        "  - name: C\n    data_dir: F:/c\n    trajectory_dir: Trajectories\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert [site.name for site in cfg.site_configs] == ["A", "C"]
    assert cfg.site_configs[1].trajectory_dir == "Trajectories"


def test_nested_ablation_yaml_is_parsed(tmp_path: Path):
    cfg_path = tmp_path / "ablation.yaml"
    cfg_path.write_text(
        "model:\n  ablation:\n    use_map_gate: false\n"
        "    use_marking_embedding: false\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert not cfg.model.ablation.use_map_gate
    assert not cfg.model.ablation.use_marking_embedding
    assert cfg.model.ablation.use_map


def test_disk_reservoir_balances_sites_and_splits(tmp_path: Path):
    cfg = Cfg(sites=[SiteCfg(name="A"), SiteCfg(name="B")])
    cfg.train.max_windows = 20
    cfg.train.balance_sites = True
    cfg.split.train = 0.6
    cfg.split.val = 0.2
    paths = _cache_paths(tmp_path / "cache")

    def sample(site: str, split: str, value: float) -> SceneSample:
        return SceneSample(
            agents=np.full((2, 3, 21), value, np.float32),
            future=np.full((4, 2), value, np.float32),
            future_valid=np.ones(4, np.float32),
            map_pts=np.full((3, 5, 4), value, np.float32),
            map_src=np.zeros(3, np.int64), map_mark=np.zeros(3, np.int64),
            map_valid=np.ones(3, np.float32), ego_id=1, start_frame=0,
            t_last=2, split=split, csv_stem="x", site=site,
        )

    reservoir = _DiskReservoir(paths, cfg, sample("A", "train", 0.0))
    for site in ("A", "B"):
        for split in ("train", "val", "test"):
            for i in range(20):
                reservoir.add(sample(site, split, float(i)))
    rows, _, kept = reservoir.finish()
    assert len(rows) == 20
    assert kept == {
        "A/train": 6, "A/val": 2, "A/test": 2,
        "B/train": 6, "B/val": 2, "B/test": 2,
    }
    assert np.load(paths["agents"], mmap_mode="r").shape[0] == 20
