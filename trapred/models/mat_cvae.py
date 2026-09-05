"""Conditional latent-variable MAT for diverse trajectories and reliable selection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trapred.models.mat import MLP, constant_velocity
from trapred.models.mat_v2 import MapAwareAgentTransformerV2


class GenerativeMapTrajectoryTransformer(MapAwareAgentTransformerV2):
    """Generate K futures from a learned conditional prior, then rank them.

    During training a future-conditioned posterior supplies latent samples and
    a KL term aligns it with the scene-conditioned prior. Evaluation never
    reads the future: it uses deterministic samples from the prior so metrics
    and plots remain reproducible. ``stochastic=True`` can be used for fresh
    Monte-Carlo samples at inference time.
    """

    is_generative = True

    def __init__(self, *, latent_dim: int = 24, **kwargs) -> None:
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")

        d = self.d_model
        self.future_encoder = nn.GRU(input_size=3, hidden_size=d, batch_first=True)
        self.prior_head = MLP([d, d, 2 * self.latent_dim], kwargs.get("dropout", 0.1))
        self.posterior_head = MLP(
            [2 * d, d, 2 * self.latent_dim], kwargs.get("dropout", 0.1)
        )
        self.latent_proj = MLP(
            [self.latent_dim, d, d], kwargs.get("dropout", 0.1)
        )
        # Candidate token + endpoint + aleatoric scale + prior energy.
        self.pi_head = MLP([d + 5, d, 1], kwargs.get("dropout", 0.1))
        self.register_buffer(
            "latent_anchors", self._make_anchors(self.n_modes, self.latent_dim),
            persistent=False,
        )

    @staticmethod
    def _make_anchors(n_modes: int, latent_dim: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260905)
        anchors = torch.randn(n_modes, latent_dim, generator=generator)
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        anchors = anchors / anchors.std(dim=0, keepdim=True).clamp_min(0.25)
        anchors[0].zero_()  # Include the conditional prior mean as one candidate.
        return anchors

    @staticmethod
    def _gaussian_params(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = raw.chunk(2, dim=-1)
        return mu, logvar.clamp(-6.0, 3.0)

    def _encode_future(self, batch: dict) -> torch.Tensor:
        future = batch["future"]
        valid = batch["future_valid"].to(future.dtype).unsqueeze(-1)
        previous = torch.cat([future.new_zeros(future.size(0), 1, 2), future[:, :-1]], dim=1)
        delta = (future - previous) * valid
        features = torch.cat([delta, valid], dim=-1)
        _, hidden = self.future_encoder(features)
        return hidden[-1]

    @staticmethod
    def _kl_per_dim(
        q_mu: torch.Tensor, q_logvar: torch.Tensor,
        p_mu: torch.Tensor, p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = q_logvar.exp()
        p_var = p_logvar.exp()
        return 0.5 * (
            p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var - 1.0
        )

    def forward(
        self, batch: dict, *, use_posterior: bool | None = None,
        stochastic: bool | None = None,
    ) -> dict:
        agents, h, context, context_pad = self.encode_scene(batch)
        b = agents.size(0)
        p_mu, p_logvar = self._gaussian_params(self.prior_head(h[:, 0]))

        if use_posterior is None:
            use_posterior = self.training and "future" in batch
        if stochastic is None:
            stochastic = bool(use_posterior)
        q_mu = q_logvar = None
        if use_posterior:
            if "future" not in batch or "future_valid" not in batch:
                raise ValueError("posterior sampling requires future and future_valid")
            future_state = self._encode_future(batch)
            q_mu, q_logvar = self._gaussian_params(
                self.posterior_head(torch.cat([h[:, 0], future_state], dim=-1))
            )
            source_mu, source_logvar = q_mu, q_logvar
        else:
            source_mu, source_logvar = p_mu, p_logvar

        if stochastic:
            eps = torch.randn(
                b, self.n_modes, self.latent_dim,
                device=agents.device, dtype=agents.dtype,
            )
        else:
            eps = self.latent_anchors.to(dtype=agents.dtype).unsqueeze(0).expand(b, -1, -1)
        std = (0.5 * source_logvar).exp()
        z = source_mu.unsqueeze(1) + std.unsqueeze(1) * eps

        query = (
            self.mode_q.expand(b, -1, -1)
            + h[:, :1]
            + self.latent_proj(z)
        )
        query = self.decoder(query, context, memory_key_padding_mask=context_pad)

        step_residual = self.step_head(query).view(
            b, self.n_modes, self.t_out, 2
        )
        residual = (
            step_residual.cumsum(dim=2)
            if self.ablation["use_cumulative_residual"] else step_residual
        )
        if self.ablation["use_cv_anchor"]:
            residual = residual + constant_velocity(
                agents, self.t_out, self.dt
            ).unsqueeze(1)
        traj = residual
        scale = F.softplus(
            self.scale_head(query).view(b, self.n_modes, self.t_out, 2)
        ) + 0.1

        prior_std = (0.5 * p_logvar).exp().unsqueeze(1).clamp_min(1e-4)
        prior_energy = ((z - p_mu.unsqueeze(1)) / prior_std).pow(2).mean(
            dim=-1, keepdim=True
        )
        score_features = torch.cat([
            query,
            traj[:, :, -1],
            scale.mean(dim=2),
            prior_energy,
        ], dim=-1)
        pi = self.pi_head(score_features).squeeze(-1)
        out = {
            "traj": traj,
            "scale": scale,
            "pi": pi,
            "prior_mu": p_mu,
            "prior_logvar": p_logvar,
        }
        if q_mu is not None and q_logvar is not None:
            out["posterior_mu"] = q_mu
            out["posterior_logvar"] = q_logvar
            out["latent_kl_per_dim"] = self._kl_per_dim(
                q_mu, q_logvar, p_mu, p_logvar
            )
        return out
