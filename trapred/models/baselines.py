"""Architecture baselines: agent Transformer (no map) and social LSTM (no map).

Both share the same ego-centric agent tensor, multimodal residual-CV
decoder, and loss as the map-aware model so the comparison isolates
sequence encoder + map channel — not data or training protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trapred.data.windows import F_A, IDX_VALID
from trapred.models.mat import MLP, constant_velocity


def _agent_pad_mask(agent_ok: torch.Tensor) -> torch.Tensor | None:
    a_pad = ~agent_ok
    if a_pad.all():
        return None
    a_fix = a_pad.clone()
    a_fix[a_fix.all(dim=-1)] = False
    return a_fix


def _last_valid(seq: torch.Tensor, valid_t: torch.Tensor) -> torch.Tensor:
    """seq [B*N, T, D], valid_t [B*N, T] → [B*N, D] at last valid step."""
    last = valid_t.long().cumsum(dim=-1).argmax(dim=-1)
    idx = torch.arange(seq.size(0), device=seq.device)
    return seq[idx, last]


class ResidualDecoder(nn.Module):
    def __init__(self, d_model: int, t_out: int, n_modes: int, dropout: float, dt: float) -> None:
        super().__init__()
        self.t_out = t_out
        self.n_modes = n_modes
        self.dt = float(dt)
        self.mode_q = nn.Parameter(torch.randn(1, n_modes, d_model) * 0.02)
        self.traj_head = MLP([d_model, d_model, t_out * 2], dropout)
        self.scale_head = MLP([d_model, d_model, t_out * 2], dropout)
        self.pi_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.traj_head.net[-1].weight)
        nn.init.zeros_(self.traj_head.net[-1].bias)

    def forward(self, fused: torch.Tensor, agents: torch.Tensor) -> dict:
        """fused [B, D] ego-centric context."""
        b = fused.size(0)
        q = self.mode_q.expand(b, -1, -1) + fused.unsqueeze(1)
        residual = self.traj_head(q).view(b, self.n_modes, self.t_out, 2)
        cv = constant_velocity(agents, self.t_out, self.dt).unsqueeze(1)
        scale = F.softplus(self.scale_head(q).view(b, self.n_modes, self.t_out, 2)) + 0.1
        return {
            "traj": cv + residual,
            "scale": scale,
            "pi": self.pi_head(q).squeeze(-1),
        }


class AgentLSTM(nn.Module):
    """Per-agent LSTM encoder + neighbor mean-pool (no map)."""

    def __init__(
        self,
        *,
        t_out: int,
        d_model: int = 128,
        n_modes: int = 6,
        dropout: float = 0.1,
        dt: float = 0.1,
        n_lstm_layers: int = 2,
        **_ignored,
    ) -> None:
        super().__init__()
        self.t_out = t_out
        self.d_model = d_model
        self.n_modes = n_modes
        self.dt = float(dt)
        self.agent_in = MLP([F_A, d_model, d_model], dropout)
        self.encoder = nn.LSTM(
            d_model, d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.social = MLP([d_model * 2, d_model, d_model], dropout)
        self.dec = ResidualDecoder(d_model, t_out, n_modes, dropout, dt)

    def forward(self, batch: dict) -> dict:
        agents = batch["agents"]
        b, n, t, _ = agents.shape
        valid_t = agents[..., IDX_VALID] > 0.5
        agent_ok = valid_t.any(dim=-1)
        x = self.agent_in(agents).reshape(b * n, t, self.d_model)
        packed, _ = self.encoder(x)
        h = _last_valid(packed, valid_t.reshape(b * n, t)).reshape(b, n, self.d_model)
        h = h * agent_ok.unsqueeze(-1).to(h.dtype)
        ego = h[:, 0]
        neigh_w = agent_ok[:, 1:].to(h.dtype).unsqueeze(-1)
        neigh = (h[:, 1:] * neigh_w).sum(dim=1) / neigh_w.sum(dim=1).clamp_min(1.0)
        fused = self.social(torch.cat([ego, neigh], dim=-1))
        return self.dec(fused, agents)


class AgentTransformer(nn.Module):
    """Temporal + social Transformer over agents only (no map tokens)."""

    def __init__(
        self,
        *,
        t_out: int,
        d_model: int = 128,
        nhead: int = 4,
        n_temporal_layers: int = 2,
        n_social_layers: int = 2,
        n_decoder_layers: int = 1,
        n_modes: int = 6,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        max_t: int = 64,
        dt: float = 0.1,
        **_ignored,
    ) -> None:
        super().__init__()
        self.t_out = t_out
        self.d_model = d_model
        self.n_modes = n_modes
        self.dt = float(dt)
        dim_ff = d_model * ffn_mult
        self.agent_in = MLP([F_A, d_model, d_model], dropout)
        self.time_pe = nn.Parameter(torch.zeros(1, max_t, d_model))
        nn.init.normal_(self.time_pe, std=0.02)
        tlayer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(tlayer, n_temporal_layers)
        slayer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.social = nn.TransformerEncoder(slayer, n_social_layers)
        dlayer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dlayer, n_decoder_layers)
        self.mode_q = nn.Parameter(torch.randn(1, n_modes, d_model) * 0.02)
        self.traj_head = MLP([d_model, d_model, t_out * 2], dropout)
        self.scale_head = MLP([d_model, d_model, t_out * 2], dropout)
        self.pi_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.traj_head.net[-1].weight)
        nn.init.zeros_(self.traj_head.net[-1].bias)

    def forward(self, batch: dict) -> dict:
        agents = batch["agents"]
        b, n, t, _ = agents.shape
        valid_t = agents[..., IDX_VALID] > 0.5
        agent_ok = valid_t.any(dim=-1)
        x = self.agent_in(agents).reshape(b * n, t, self.d_model)
        x = x + self.time_pe[:, :t]
        t_pad = ~valid_t.reshape(b * n, t)
        all_pad = t_pad.all(dim=-1)
        t_pad = t_pad.clone()
        t_pad[all_pad] = False
        x = self.temporal(x, src_key_padding_mask=t_pad)
        h = _last_valid(x, valid_t.reshape(b * n, t)).reshape(b, n, self.d_model)
        h = h * agent_ok.unsqueeze(-1).to(h.dtype)
        h = self.social(h, src_key_padding_mask=_agent_pad_mask(agent_ok))
        h = h * agent_ok.unsqueeze(-1).to(h.dtype)

        a_pad = ~agent_ok
        empty = a_pad.all(dim=-1)
        a_pad = a_pad.clone()
        a_pad[empty] = False
        q = self.mode_q.expand(b, -1, -1) + h[:, :1, :]
        q = self.decoder(q, h, memory_key_padding_mask=a_pad)
        residual = self.traj_head(q).view(b, self.n_modes, self.t_out, 2)
        cv = constant_velocity(agents, self.t_out, self.dt).unsqueeze(1)
        scale = F.softplus(self.scale_head(q).view(b, self.n_modes, self.t_out, 2)) + 0.1
        return {
            "traj": cv + residual,
            "scale": scale,
            "pi": self.pi_head(q).squeeze(-1),
        }
