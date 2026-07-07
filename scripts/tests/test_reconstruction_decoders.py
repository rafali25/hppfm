#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.models import DualDecoderHead


def main() -> None:
    batch_size = 2
    n_context = 128
    n_target = 64
    model_dim = 256
    input_dim = 96
    context_outputs = torch.randn(batch_size, n_context, model_dim)
    context_mask = torch.rand(batch_size, n_context) > 0.1
    target_day = torch.randint(0, 7, (batch_size, n_target))
    target_time = torch.randint(0, 288, (batch_size, n_target))

    decoders = DualDecoderHead(input_dim=input_dim, model_dim=model_dim, nhead=4)
    pred_local, pred_global = decoders(
        context_outputs,
        target_day,
        target_time,
        context_mask=context_mask,
    )
    assert pred_local.shape == (batch_size, n_target, input_dim), pred_local.shape
    assert pred_global.shape == (batch_size, n_target, input_dim), pred_global.shape
    assert torch.isfinite(pred_local).all()
    assert torch.isfinite(pred_global).all()
    print("reconstruction decoder tests passed")


if __name__ == "__main__":
    main()
