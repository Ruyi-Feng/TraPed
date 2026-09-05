import torch
import pytest

from trapred.ablations import ABLATION_PROFILES, apply_ablation
from trapred.config import Cfg, ModelCfg
from trapred.data.windows import F_A
from trapred.models.factory import (
    ARCHES, build_model, load_model_from_ckpt, model_kwargs,
)
from trapred.models.losses import multimodal_loss
from trapred.eval.metrics import batch_metrics


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


def test_optimized_model_backward_with_enhanced_loss():
    t_out = 12
    cfg = ModelCfg(
        d_model=32, nhead=2, n_temporal_layers=1, n_social_layers=1,
        n_decoder_layers=1, n_map_layers=1, n_modes=3, dropout=0.0,
    )
    batch = _batch(t_out=t_out)
    net = build_model("mat_v2", t_out=t_out, dt=0.1, model=cfg)
    out = net(batch)
    stats = multimodal_loss(
        out, batch["future"], batch["future_valid"],
        endpoint_weight=2.0, diversity_w=0.02,
        soft_cls_temp=2.0, winner_fde_weight=0.5,
    )
    stats["loss"].backward()
    assert torch.isfinite(stats["loss"])


def test_generative_model_prior_posterior_and_reliability_loss():
    cfg = ModelCfg(
        d_model=32, nhead=2, n_temporal_layers=1, n_social_layers=1,
        n_decoder_layers=1, n_map_layers=1, n_modes=5, latent_dim=8,
        dropout=0.0,
    )
    batch = _batch(b=2, n=3, t=5, m=4, p=6, t_out=7)
    net = build_model("mat_cvae", t_out=7, dt=0.1, model=cfg)

    net.train()
    posterior = net(batch, use_posterior=True)
    assert posterior["traj"].shape == (2, 5, 7, 2)
    assert posterior["latent_kl_per_dim"].shape == (2, 8)
    stats = multimodal_loss(
        posterior, batch["future"], batch["future_valid"],
        kl_weight=0.01, kl_free_bits=0.02,
        diversity_w=0.02, confidence_regret_weight=0.1,
    )
    stats["loss"].backward()
    assert torch.isfinite(stats["loss"])
    assert stats["kl"] >= 0

    # Evaluation uses only the conditional prior and is deterministic.
    inference_batch = {
        key: value for key, value in batch.items()
        if key not in ("future", "future_valid")
    }
    net.eval()
    first = net(inference_batch)
    second = net(inference_batch)
    assert torch.equal(first["traj"], second["traj"])
    assert "latent_kl_per_dim" not in first


def test_multimodal_metrics_report_selector_quality():
    future = torch.zeros(1, 3, 2)
    valid = torch.ones(1, 3)
    traj = torch.tensor([[
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
    ]])
    out = {
        "traj": traj,
        "pi": torch.tensor([[0.0, 1.0, 2.0]]),
    }
    metrics = batch_metrics(out, future, valid)
    assert metrics["minADE"] == pytest.approx(0.0)
    assert metrics["mlADE"] == pytest.approx(2.0)
    assert metrics["top3ADE"] == pytest.approx(0.0)
    assert metrics["selectionGapADE"] == pytest.approx(2.0)
    assert metrics["oracleModeRate"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "profile",
    [
        "no_map", "no_polyline_order", "no_map_gate", "no_map_semantics",
        "no_marking_type", "no_social_context", "no_cv_anchor",
        "no_cumulative_residual",
    ],
)
def test_mat_v2_architecture_ablation_forward(profile):
    cfg = Cfg()
    cfg.model = ModelCfg(
        d_model=16, nhead=2, n_temporal_layers=1, n_social_layers=1,
        n_decoder_layers=1, n_map_layers=1, n_modes=2, dropout=0.0,
    )
    apply_ablation(cfg, profile)
    batch = _batch(b=1, n=3, t=5, m=4, p=6, t_out=7)
    if profile == "no_map":
        # Removing the map must remove the data dependency, not merely zero it.
        batch = {k: v for k, v in batch.items() if not k.startswith("map_")}
    net = build_model("mat_v2", t_out=7, dt=0.1, model=cfg.model)
    out = net(batch)
    assert out["traj"].shape == (1, 2, 7, 2)
    out["traj"].sum().backward()


def test_standard_loss_profile_only_changes_loss_terms():
    cfg = Cfg()
    cfg.train.loss_endpoint_weight = 2.0
    cfg.train.loss_diversity_weight = 0.02
    cfg.train.loss_soft_cls_temp = 2.0
    cfg.train.loss_winner_fde_weight = 0.5
    apply_ablation(cfg, "standard_loss")
    assert cfg.train.loss_endpoint_weight == 1.0
    assert cfg.train.loss_diversity_weight == 0.0
    assert cfg.train.loss_soft_cls_temp == 0.0
    assert cfg.train.loss_winner_fde_weight == 0.0
    assert all(vars(cfg.model.ablation).values())


def test_ablation_checkpoint_round_trip(tmp_path):
    cfg = Cfg()
    cfg.model = ModelCfg(
        d_model=16, nhead=2, n_temporal_layers=1, n_social_layers=1,
        n_decoder_layers=1, n_map_layers=1, n_modes=2, dropout=0.0,
    )
    apply_ablation(cfg, "no_map")
    net = build_model("mat_v2", t_out=7, dt=0.1, model=cfg.model)
    ckpt = tmp_path / "ablation.pt"
    torch.save({
        "arch": "mat_v2",
        "model": net.state_dict(),
        "cfg": model_kwargs(cfg.model, t_out=7, dt=0.1),
    }, ckpt)
    loaded = load_model_from_ckpt(ckpt, torch.device("cpu"))
    assert loaded.map_encoder is None
    batch = _batch(b=1, n=3, t=5, m=4, p=6, t_out=7)
    batch = {k: v for k, v in batch.items() if not k.startswith("map_")}
    assert loaded(batch)["traj"].shape == (1, 2, 7, 2)


def test_all_named_profiles_are_applicable():
    for profile in ABLATION_PROFILES:
        cfg = Cfg()
        apply_ablation(cfg, profile)
