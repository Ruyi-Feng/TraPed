import torch

from trapred.config import ModelCfg
from trapred.data.windows import F_A
from trapred.models.factory import ARCHES, build_model


def _batch(b=2, n=4, t=8, m=6, p=10, t_out=12):
    agents = torch.zeros(b, n, t, F_A)
    agents[..., -1] = 1.0
    agents[:, 0, -1, 2] = 10.0  # vx
    return {
        "agents": agents,
        "future": torch.randn(b, t_out, 2),
        "future_valid": torch.ones(b, t_out),
        "map_pts": torch.randn(b, m, p, 4),
        "map_src": torch.zeros(b, m, dtype=torch.long),
        "map_mark": torch.zeros(b, m, dtype=torch.long),
        "map_valid": torch.ones(b, m),
    }


def test_all_arches_forward_shapes():
    t_out = 12
    cfg = ModelCfg(d_model=32, nhead=2, n_temporal_layers=1, n_social_layers=1,
                   n_decoder_layers=1, n_modes=3, dropout=0.0)
    batch = _batch(t_out=t_out)
    for arch in ARCHES:
        net = build_model(arch, t_out=t_out, dt=0.1, model=cfg)
        out = net(batch)
        assert out["traj"].shape == (2, 3, t_out, 2)
        assert out["pi"].shape == (2, 3)
        assert out["scale"].shape == (2, 3, t_out, 2)
