"""Second-generation map-aware trajectory Transformer.

Compared with the original MAT, this version preserves point order inside each
map polyline, uses gated agent-to-map attention, and predicts cumulative motion
residuals over a constant-velocity anchor for smoother trajectories.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trapred.data.map_tokens import N_MARK, N_SRC
from trapred.data.windows import F_A, IDX_VALID
from trapred.models.mat import MLP, constant_velocity


class PolylineEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float,
        n_layers: int,
        use_order: bool = True,
        max_points: int = 64,
    ) -> None:
        super().__init__()
        self.use_order = bool(use_order)
        self.point_in = MLP([4, d_model, d_model], dropout)
        if self.use_order:
            self.point_pe = nn.Parameter(torch.zeros(1, max_points, d_model))
            nn.init.normal_(self.point_pe, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_ff, dropout, batch_first=True,
                activation="gelu", norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, max(1, n_layers))
            self.pool = MLP([2 * d_model, d_model, d_model], dropout)
        else:
            self.register_parameter("point_pe", None)
            self.encoder = None
            self.pool = None

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        b, m, p, _ = points.shape
        if self.use_order and p > self.point_pe.size(1):
            raise ValueError(
                f"map polyline has {p} points, maximum is {self.point_pe.size(1)}"
            )
        x = self.point_in(points).reshape(b * m, p, -1)
        if not self.use_order:
            return x.amax(dim=1).reshape(b, m, -1)
        x = x + self.point_pe[:, :p]
        x = self.encoder(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
        return self.pool(pooled).reshape(b, m, -1)


class GatedMapBlock(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_ff: int, dropout: float,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_gate = bool(use_gate)
        self.q_norm = nn.LayerNorm(d_model)
        self.m_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(2 * d_model, d_model) if self.use_gate else None
        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, agents: torch.Tensor, map_tokens: torch.Tensor,
        map_padding: torch.Tensor,
    ) -> torch.Tensor:
        cross, _ = self.attn(
            self.q_norm(agents), self.m_norm(map_tokens), self.m_norm(map_tokens),
            key_padding_mask=map_padding, need_weights=False,
        )
        if self.gate is not None:
            cross = torch.sigmoid(
                self.gate(torch.cat([agents, cross], dim=-1))
            ) * cross
        x = agents + self.dropout(cross)
        return x + self.ffn(self.out_norm(x))


class MapAwareAgentTransformerV2(nn.Module):
    """Order-aware map encoder plus gated social/map fusion."""

    def __init__(
        self,
        *,
        t_out: int,
        d_model: int = 192,
        nhead: int = 8,
        n_temporal_layers: int = 3,
        n_social_layers: int = 3,
        n_decoder_layers: int = 2,
        n_map_layers: int = 2,
        n_modes: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        max_t: int = 64,
        dt: float = 0.1,
        ablation: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.t_out = t_out
        self.dt = float(dt)
        flags = {
            "use_map": True,
            "use_polyline_order": True,
            "use_map_gate": True,
            "use_map_source_embedding": True,
            "use_marking_embedding": True,
            "use_social_context": True,
            "use_cv_anchor": True,
            "use_cumulative_residual": True,
        }
        unknown = set(ablation or {}) - set(flags)
        if unknown:
            raise ValueError(f"unknown MAT-v2 ablation flags: {sorted(unknown)}")
        flags.update(ablation or {})
        self.ablation = flags
        dim_ff = d_model * ffn_mult

        self.agent_norm = nn.LayerNorm(F_A)
        self.agent_in = MLP([F_A, d_model, d_model], dropout)
        self.time_pe = nn.Parameter(torch.zeros(1, max_t, d_model))
        nn.init.normal_(self.time_pe, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, max(1, n_temporal_layers)
        )
        social_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.social = (
            nn.TransformerEncoder(social_layer, max(1, n_social_layers))
            if flags["use_social_context"] else None
        )

        if flags["use_map"]:
            self.map_encoder = PolylineEncoder(
                d_model, nhead, dim_ff, dropout, n_map_layers,
                use_order=flags["use_polyline_order"],
            )
            self.src_emb = (
                nn.Embedding(N_SRC, d_model)
                if flags["use_map_source_embedding"] else None
            )
            self.mark_emb = (
                nn.Embedding(N_MARK, d_model)
                if flags["use_marking_embedding"] else None
            )
            self.map_ln = nn.LayerNorm(d_model)
            self.map_cross = nn.ModuleList([
                GatedMapBlock(
                    d_model, nhead, dim_ff, dropout,
                    use_gate=flags["use_map_gate"],
                )
                for _ in range(max(1, n_social_layers))
            ])
        else:
            self.map_encoder = None
            self.src_emb = None
            self.mark_emb = None
            self.map_ln = None
            self.map_cross = nn.ModuleList()

        self.mode_q = nn.Parameter(torch.randn(1, n_modes, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, max(1, n_decoder_layers)
        )
        self.step_head = MLP([d_model, d_model, t_out * 2], dropout)
        self.scale_head = MLP([d_model, d_model, t_out * 2], dropout)
        self.pi_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.step_head.net[-1].weight)
        nn.init.zeros_(self.step_head.net[-1].bias)

    def encode_scene(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return agents, fused agent states, decoder memory and its padding mask."""
        agents = batch["agents"]
        b, n, t, _ = agents.shape
        valid_t = agents[..., IDX_VALID] > 0.5
        agent_ok = valid_t.any(dim=-1)

        x = self.agent_in(self.agent_norm(agents)).reshape(b * n, t, self.d_model)
        x = x + self.time_pe[:, :t]
        t_pad = ~valid_t.reshape(b * n, t)
        all_pad = t_pad.all(dim=-1)
        t_pad = t_pad.clone()
        t_pad[all_pad] = False
        x = self.temporal(x, src_key_padding_mask=t_pad)
        last = valid_t.reshape(b * n, t).long().cumsum(dim=-1).argmax(dim=-1)
        idx = torch.arange(b * n, device=x.device)
        h = x[idx, last].reshape(b, n, self.d_model)
        h = h * agent_ok.unsqueeze(-1).to(h.dtype)

        if self.social is not None:
            agent_pad = ~agent_ok
            agent_mask = agent_pad.clone()
            agent_mask[agent_mask.all(dim=-1)] = False
            h = self.social(h, src_key_padding_mask=agent_mask)
            h = h * agent_ok.unsqueeze(-1).to(h.dtype)
        else:
            # A true social-context ablation: neighbors are removed from both
            # the social encoder and the final decoder memory.
            h = h[:, :1]
            agent_ok = agent_ok[:, :1]
            agent_pad = ~agent_ok

        if self.map_encoder is not None:
            map_ok = batch["map_valid"] > 0.5
            m = self.map_encoder(batch["map_pts"])
            if self.src_emb is not None:
                m = m + self.src_emb(batch["map_src"].clamp(0, N_SRC - 1))
            if self.mark_emb is not None:
                m = m + self.mark_emb(batch["map_mark"].clamp(0, N_MARK - 1))
            m = self.map_ln(m)
            m = m * map_ok.unsqueeze(-1).to(m.dtype)
            map_pad = ~map_ok
            map_pad = map_pad.clone()
            map_pad[map_pad.all(dim=-1)] = False
            for layer in self.map_cross:
                h = layer(h, m, map_pad)
            h = h * agent_ok.unsqueeze(-1).to(h.dtype)
            context = torch.cat([h, m], dim=1)
            context_pad = torch.cat([agent_pad, map_pad], dim=1)
        else:
            context = h
            context_pad = agent_pad
        return agents, h, context, context_pad

    def forward(self, batch: dict) -> dict:
        agents, h, context, context_pad = self.encode_scene(batch)
        b = agents.size(0)
        query = self.mode_q.expand(b, -1, -1) + h[:, :1]
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
        pi = self.pi_head(query).squeeze(-1)
        return {"traj": traj, "scale": scale, "pi": pi}
