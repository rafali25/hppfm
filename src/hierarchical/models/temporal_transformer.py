from __future__ import annotations

import torch
import torch.nn as nn


class WeekTemporalTransformer(nn.Module):
    """WavesFM-style temporal encoder over 5-minute week tokens."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_time_of_day: int = 288,
        max_day_of_week: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.max_time_of_day = max_time_of_day
        self.max_day_of_week = max_day_of_week

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.missing_token = nn.Parameter(torch.zeros(model_dim))
        self.day_embedding = nn.Embedding(max_day_of_week, model_dim)
        self.time_embedding = nn.Embedding(max_time_of_day, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        day_of_week: torch.Tensor,
        time_of_day: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, D], got {tuple(x.shape)}")
        valid_mask = valid_mask.bool()
        projected = self.input_projection(x)
        missing = self.missing_token.view(1, 1, -1).expand_as(projected)
        projected = torch.where(valid_mask.unsqueeze(-1), projected, missing)
        projected = projected + self.day_embedding(day_of_week.long())
        projected = projected + self.time_embedding(time_of_day.long())

        key_padding_mask = ~valid_mask
        all_missing = key_padding_mask.all(dim=1)
        if bool(all_missing.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing, 0] = False

        encoded = self.encoder(projected, src_key_padding_mask=key_padding_mask)
        return self.norm(encoded)

    @staticmethod
    def mean_pool(outputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.bool().unsqueeze(-1)
        summed = (outputs * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        return summed / denom

