"""Named, reproducible ablation profiles for the MAT-v2 main model."""
from __future__ import annotations

from dataclasses import asdict

from trapred.config import Cfg


_FULL_MODEL = {
    "use_map": True,
    "use_polyline_order": True,
    "use_map_gate": True,
    "use_map_source_embedding": True,
    "use_marking_embedding": True,
    "use_social_context": True,
    "use_cv_anchor": True,
    "use_cumulative_residual": True,
}

# Each model profile starts from the complete method, so results do not depend
# on unrelated switches that happen to be present in a YAML file.
ABLATION_PROFILES: dict[str, dict[str, dict[str, object]]] = {
    "full": {"model": _FULL_MODEL},
    "no_map": {"model": {**_FULL_MODEL, "use_map": False}},
    "no_polyline_order": {
        "model": {**_FULL_MODEL, "use_polyline_order": False},
    },
    "no_map_gate": {"model": {**_FULL_MODEL, "use_map_gate": False}},
    "no_map_semantics": {
        "model": {
            **_FULL_MODEL,
            "use_map_source_embedding": False,
            "use_marking_embedding": False,
        },
    },
    "no_marking_type": {
        "model": {**_FULL_MODEL, "use_marking_embedding": False},
    },
    "no_social_context": {
        "model": {**_FULL_MODEL, "use_social_context": False},
    },
    "no_cv_anchor": {"model": {**_FULL_MODEL, "use_cv_anchor": False}},
    "no_cumulative_residual": {
        "model": {**_FULL_MODEL, "use_cumulative_residual": False},
    },
    "standard_loss": {
        "model": _FULL_MODEL,
        "train": {
            "loss_endpoint_weight": 1.0,
            "loss_diversity_weight": 0.0,
            "loss_soft_cls_temp": 0.0,
            "loss_winner_fde_weight": 0.0,
        },
    },
}

# Default suite: full method, broad context removals, each MAT-v2 addition,
# lane-marking semantics, motion decoding choices, and the enhanced objective.
DEFAULT_ABLATIONS = (
    "full",
    "no_map",
    "no_polyline_order",
    "no_map_gate",
    "no_marking_type",
    "no_social_context",
    "no_cv_anchor",
    "no_cumulative_residual",
    "standard_loss",
)


def apply_ablation(cfg: Cfg, profile: str) -> None:
    """Apply a named profile in-place after normal CLI/config overrides."""
    if profile not in ABLATION_PROFILES:
        expected = ", ".join(ABLATION_PROFILES)
        raise ValueError(f"unknown ablation {profile!r}; expected one of {expected}")
    overrides = ABLATION_PROFILES[profile]
    for name, value in overrides.get("model", {}).items():
        setattr(cfg.model.ablation, name, value)
    for name, value in overrides.get("train", {}).items():
        setattr(cfg.train, name, value)


def ablation_run_name(profile: str) -> str:
    if profile not in ABLATION_PROFILES:
        raise ValueError(f"unknown ablation {profile!r}")
    return f"ablation-{profile.replace('_', '-')}"


def ablation_dict(cfg: Cfg) -> dict[str, bool]:
    return asdict(cfg.model.ablation)
