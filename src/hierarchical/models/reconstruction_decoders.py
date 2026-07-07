from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedPositionQuery(nn.Module):
    def __init__(self, model_dim: int, *, max_day_of_week: int = 7, max_time_of_day: int = 288):
        super().__init__()
        self.day_embedding = nn.Embedding(max_day_of_week, model_dim)
        self.time_embedding = nn.Embedding(max_time_of_day, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, day_of_week: torch.Tensor, time_of_day: torch.Tensor) -> torch.Tensor:
        query = self.day_embedding(day_of_week.long()) + self.time_embedding(time_of_day.long())
        return self.norm(query)


class LocalReconstructionDecoder(nn.Module):
    """Cross-attend target positions to context outputs and predict input embeddings."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position_query = FactorizedPositionQuery(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        self.predictor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, input_dim),
        )

    def forward(
        self,
        context_outputs: torch.Tensor,
        target_day_of_week: torch.Tensor,
        target_time_of_day: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.position_query(target_day_of_week, target_time_of_day)
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        if key_padding_mask is not None:
            all_missing = key_padding_mask.all(dim=1)
            if bool(all_missing.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_missing, 0] = False
        attended, _ = self.attn(
            query=query,
            key=context_outputs,
            value=context_outputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.predictor(self.norm(attended + query))


class GlobalReconstructionDecoder(nn.Module):
    """Reconstruct targets from a mean-pooled global week bottleneck."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position_query = FactorizedPositionQuery(model_dim)
        self.predictor = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, input_dim),
        )

    def forward(
        self,
        context_outputs: torch.Tensor,
        target_day_of_week: torch.Tensor,
        target_time_of_day: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_mask is None:
            global_context = context_outputs.mean(dim=1)
        else:
            mask = context_mask.bool().unsqueeze(-1)
            global_context = (context_outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        query = self.position_query(target_day_of_week, target_time_of_day)
        global_tokens = global_context.unsqueeze(1).expand(-1, query.shape[1], -1)
        return self.predictor(torch.cat([global_tokens, query], dim=-1))


class DualDecoderHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local = LocalReconstructionDecoder(
            input_dim=input_dim,
            model_dim=model_dim,
            nhead=nhead,
            dropout=dropout,
        )
        self.global_decoder = GlobalReconstructionDecoder(
            input_dim=input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )

    def forward(
        self,
        context_outputs: torch.Tensor,
        target_day_of_week: torch.Tensor,
        target_time_of_day: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_local = self.local(
            context_outputs,
            target_day_of_week,
            target_time_of_day,
            context_mask=context_mask,
        )
        pred_global = self.global_decoder(
            context_outputs,
            target_day_of_week,
            target_time_of_day,
            context_mask=context_mask,
        )
        return pred_local, pred_global
