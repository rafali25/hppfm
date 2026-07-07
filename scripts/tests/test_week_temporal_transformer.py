#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.models import WeekTemporalTransformer


def main() -> None:
    batch_size = 2
    seq_len = 2016
    input_dim = 64
    x = torch.randn(batch_size, seq_len, input_dim)
    valid_mask = torch.rand(batch_size, seq_len) > 0.15
    day_of_week = torch.arange(7).repeat_interleave(288).unsqueeze(0).repeat(batch_size, 1)
    time_of_day = torch.arange(288).repeat(7).unsqueeze(0).repeat(batch_size, 1)

    model = WeekTemporalTransformer(
        input_dim=input_dim,
        model_dim=256,
        num_layers=1,
        nhead=4,
        dim_feedforward=512,
    )
    out = model(x, valid_mask, day_of_week, time_of_day)
    pooled = model.mean_pool(out, valid_mask)
    assert out.shape == (batch_size, seq_len, 256), out.shape
    assert pooled.shape == (batch_size, 256), pooled.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(pooled).all()
    print("WeekTemporalTransformer test passed")


if __name__ == "__main__":
    main()

