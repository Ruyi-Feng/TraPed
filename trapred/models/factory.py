"""Build / serialize models by architecture name."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from trapred.config import ModelCfg
from trapred.models.baselines import AgentLSTM, AgentTransformer
from trapred.models.mat import MapAwareAgentTransformer

ARCHES = ("mat", "transformer", "lstm")


def ckpt_filename(arch: str) -> str:
    return "best.pt" if arch == "mat" else f"best_{arch}.pt"


def ckpt_path(out_dir: Path, arch: str) -> Path:
    return Path(out_dir) / ckpt_filename(arch)


def history_filename(arch: str) -> str:
    return "history.json" if arch == "mat" else f"history_{arch}.json"


def model_kwargs(model: ModelCfg, *, t_out: int, dt: float) -> dict[str, Any]:
    return {
        "t_out": t_out,
        "d_model": model.d_model,
        "nhead": model.nhead,
        "n_temporal_layers": model.n_temporal_layers,
        "n_social_layers": model.n_social_layers,
        "n_decoder_layers": model.n_decoder_layers,
        "n_modes": model.n_modes,
        "dropout": model.dropout,
        "ffn_mult": model.ffn_mult,
        "dt": dt,
    }


def build_model(arch: str, *, t_out: int, dt: float, model: ModelCfg) -> nn.Module:
    kw = model_kwargs(model, t_out=t_out, dt=dt)
    if arch == "mat":
        return MapAwareAgentTransformer(**kw)
    if arch == "transformer":
        return AgentTransformer(**kw)
    if arch == "lstm":
        return AgentLSTM(**kw)
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")


def load_model_from_ckpt(ckpt_path: Path, device) -> nn.Module:
    blob = torch_load(ckpt_path, device)
    arch = blob.get("arch", "mat")
    cfg = dict(blob["cfg"])
    t_out = cfg.pop("t_out")
    dummy = ModelCfg()
    for k, v in cfg.items():
        if hasattr(dummy, k):
            setattr(dummy, k, v)
    dt = float(cfg.get("dt", 0.1))
    net = build_model(arch, t_out=t_out, dt=dt, model=dummy)
    net.load_state_dict(blob["model"])
    return net.to(device).eval()


def torch_load(path: Path, device):
    import torch
    return torch.load(path, map_location=device, weights_only=False)
