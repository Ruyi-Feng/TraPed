"""Map-aware Agent Transformer (HiVT / QCNet-style, ego-centric)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trapred.data.map_tokens import N_MARK, N_SRC
from trapred.data.windows import F_A, IDX_VALID


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MapAwareAgentTransformer(nn.Module):
    """Encode neighbor histories + map polylines (with dashed/solid), decode K modes."""

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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.t_out = t_out
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

        self.map_pt = MLP([4, d_model, d_model], dropout)
        self.src_emb = nn.Embedding(N_SRC, d_model)
        self.mark_emb = nn.Embedding(N_MARK, d_model)
        self.map_ln = nn.LayerNorm(d_model)

        self.map_cross = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu",
                norm_first=True,
            )
            for _ in range(max(1, n_social_layers))
        ])

        self.mode_q = nn.Parameter(torch.randn(1, n_modes, d_model) * 0.02)
        dlayer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dlayer, n_decoder_layers)
        self.traj_head = MLP([d_model, d_model, t_out * 2])
        self.scale_head = MLP([d_model, d_model, t_out * 2])
        self.pi_head = nn.Linear(d_model, 1)
        last = self.traj_head.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, batch: dict) -> dict:
        agents = batch["agents"]                    # [B, N, T, F]
        map_pts = batch["map_pts"]                  # [B, M, P, 4]
        b, n, t, _ = agents.shape
        valid_t = agents[..., IDX_VALID] > 0.5      # [B, N, T]
        agent_ok = valid_t.any(dim=-1)              # [B, N]
        map_ok = batch["map_valid"] > 0.5           # [B, M]

        x = self.agent_in(agents).reshape(b * n, t, self.d_model)
        x = x + self.time_pe[:, :t]
        t_pad = ~valid_t.reshape(b * n, t)
        # TransformerEncoder errors if a row is all-padded; mark those as all-valid then zero.
        all_pad = t_pad.all(dim=-1)
        t_pad = t_pad.clone()
        t_pad[all_pad] = False
        x = self.temporal(x, src_key_padding_mask=t_pad)
        # last valid timestep
        last = valid_t.reshape(b * n, t).long().cumsum(dim=-1)
        last = last.argmax(dim=-1)
        idx = torch.arange(b * n, device=x.device)
        h = x[idx, last].reshape(b, n, self.d_model)
        h = h * agent_ok.unsqueeze(-1).to(h.dtype)

        a_pad = ~agent_ok
        if a_pad.all():
            a_mask = None
        else:
            a_fix = a_pad.clone()
            empty = a_fix.all(dim=-1)
            a_fix[empty] = False
            a_mask = a_fix
        h = self.social(h, src_key_padding_mask=a_mask)

        m = self.map_pt(map_pts).max(dim=2).values
        src = batch["map_src"].clamp(0, N_SRC - 1)
        mark = batch["map_mark"].clamp(0, N_MARK - 1)
        m = self.map_ln(m + self.src_emb(src) + self.mark_emb(mark))
        m = m * map_ok.unsqueeze(-1).to(m.dtype)
        m_pad = ~map_ok
        empty_m = m_pad.all(dim=-1)
        m_pad = m_pad.clone()
        m_pad[empty_m] = False
        for layer in self.map_cross:
            h = layer(h, m, memory_key_padding_mask=m_pad)
        h = h * agent_ok.unsqueeze(-1).to(h.dtype)

        ctx = torch.cat([h, m], dim=1)
        ctx_pad = torch.cat([~agent_ok, m_pad], dim=1)
        q = self.mode_q.expand(b, -1, -1) + h[:, :1, :]
        q = self.decoder(q, ctx, memory_key_padding_mask=ctx_pad)

        residual = self.traj_head(q).view(b, self.n_modes, self.t_out, 2)
        cv = constant_velocity(agents, self.t_out, self.dt).unsqueeze(1)
        traj = cv + residual
        scale = F.softplus(self.scale_head(q).view(b, self.n_modes, self.t_out, 2)) + 0.1
        pi = self.pi_head(q).squeeze(-1)
        return {"traj": traj, "scale": scale, "pi": pi}


def constant_velocity(agents: torch.Tensor, t_out: int, dt: float) -> torch.Tensor:
    """CV baseline in the ego frame: last observed ego velocity × time."""
    vx = agents[:, 0, -1, 2]
    vy = agents[:, 0, -1, 3]
    steps = torch.arange(1, t_out + 1, device=agents.device, dtype=agents.dtype) * dt
    pred = torch.stack([vx.unsqueeze(-1) * steps, vy.unsqueeze(-1) * steps], dim=-1)
    return pred
