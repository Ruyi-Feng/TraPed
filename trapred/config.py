"""YAML + dataclass config."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class WindowCfg:
    target_fps: float = 10.0
    t_input_s: float = 3.0
    t_horizon_s: float = 5.0
    n_agents: int = 12
    stride_s: float = 1.0
    max_neighbor_m: float = 80.0
    map_radius_m: float = 100.0
    n_map_tokens: int = 48
    n_pts_polyline: int = 20
    savgol_window: int = 11
    savgol_poly: int = 3


@dataclass
class SplitCfg:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass
class AblationCfg:
    """Switches used to ablate individual components of MAT-v2."""

    use_map: bool = True
    use_polyline_order: bool = True
    use_map_gate: bool = True
    use_map_source_embedding: bool = True
    use_marking_embedding: bool = True
    use_social_context: bool = True
    use_cv_anchor: bool = True
    use_cumulative_residual: bool = True


@dataclass
class ModelCfg:
    d_model: int = 128
    nhead: int = 4
    n_temporal_layers: int = 2
    n_social_layers: int = 2
    n_decoder_layers: int = 1
    n_modes: int = 6
    dropout: float = 0.1
    ffn_mult: int = 4
    n_map_layers: int = 2
    latent_dim: int = 24
    ablation: AblationCfg = field(default_factory=AblationCfg)


@dataclass
class TrainCfg:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    max_windows: Optional[int] = None
    amp: bool = True
    grad_accum: int = 1
    balance_sites: bool = True
    loss_endpoint_weight: float = 1.0
    loss_diversity_weight: float = 0.0
    loss_soft_cls_temp: float = 0.0
    loss_winner_fde_weight: float = 0.0
    loss_cls_weight: float = 0.5
    loss_kl_weight: float = 0.0
    loss_kl_free_bits: float = 0.0
    loss_kl_warmup_epochs: int = 0
    loss_confidence_regret_weight: float = 0.0
    loss_diversity_margin: float = 2.0
    checkpoint_metric: str = "minADE"


@dataclass
class SiteCfg:
    name: str = "ExpresswayA"
    data_dir: str = "data/expresswayAsample"
    source_fps: float = 30.0
    pixel_per_meter: Optional[float] = None
    trajectory_dir: Optional[str] = None
    csv_glob: str = "*.csv"
    exclude_globs: List[str] = field(default_factory=lambda: [
        "metadata.csv", "._*.csv", "*_located.csv",
    ])
    lanes_npy: Optional[str] = None
    background: Optional[str] = None
    lane_png: Optional[str] = None


@dataclass
class Cfg:
    site: SiteCfg = field(default_factory=SiteCfg)
    sites: List[SiteCfg] = field(default_factory=list)
    window: WindowCfg = field(default_factory=WindowCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    output_dir: str = "outputs"

    @property
    def data_path(self) -> Path:
        return Path(self.site.data_dir)

    @property
    def out_path(self) -> Path:
        return Path(self.output_dir) / self.site.name

    @property
    def site_configs(self) -> List[SiteCfg]:
        return self.sites if self.sites else [self.site]


def _merge(dc, raw: dict) -> None:
    for k, v in raw.items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: str | Path) -> Cfg:
    cfg = Cfg()
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_sites = raw.pop("sites", None)
    _merge(cfg, raw)
    if raw_sites is not None:
        cfg.sites = []
        for item in raw_sites:
            site = SiteCfg()
            _merge(site, item or {})
            cfg.sites.append(site)
    return cfg


def dump_config(cfg: Cfg, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
