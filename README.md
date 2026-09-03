# TraPed

Map-aware trajectory prediction for merge-area drone recordings
(CitySim-style CSV + lane-polygon `.npy` + orthophoto).

Each sample is **ego-centric** (last observed pose = origin), includes
nearby vehicles, and encodes road structure: drivable edges, geometric
centerlines (from the basemap, same construction as `ref_code/safetyfm/adapters/basemap.py`),
and **solid/dashed lane markings** from the background image (signal
method in `ref_code/lane_line_probe2.py`).

Horizon is **5 s at 10 Hz**. At highway speed that is ~100–150 m, which
covers a merge / lane-change decision; 3 s of history is the observation
window. Splits are **time-ordered with a horizon buffer**, so a train
window's future frames never appear in val/test history.

## Sample run (`conda` env `torch231`)

```bat
conda activate torch231
python -m pytest tests -q
python -m trapred.visualize_markings --data-dir data/expresswayAsample
python -m trapred.train --config configs/expresswayA_sample.yaml --arch mat
python -m trapred.train --config configs/expresswayA_sample.yaml --arch transformer
python -m trapred.train --config configs/expresswayA_sample.yaml --arch lstm
python -m trapred.compare --config configs/expresswayA_sample.yaml --split test
```

Outputs go to `outputs/ExpresswayA/` (`best.pt`, metrics JSON, qualitative plots).

A new site is a folder with trajectory `*.csv`, `*Lanes.npy`, and
`*Background.png`. Point `site.data_dir` at it; pixel-to-meter scale is
inferred from lat/lon when metric columns are absent.
