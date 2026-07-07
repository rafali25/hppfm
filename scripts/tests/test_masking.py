#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.training import sample_context_target_mask


def main() -> None:
    valid = torch.ones(2, 2016, dtype=torch.bool)
    context, target = sample_context_target_mask(valid, context_ratio=0.25, patch_sizes=(1, 4, 12))
    assert context.shape == valid.shape
    assert target.shape == valid.shape
    assert torch.all(context <= valid)
    assert torch.all(target <= valid)
    assert not bool((context & target).any())
    assert bool(context.any())
    assert bool(target.any())

    empty = torch.zeros(2, 2016, dtype=torch.bool)
    context, target = sample_context_target_mask(empty)
    assert not bool(context.any())
    assert not bool(target.any())

    sparse = torch.zeros(1, 2016, dtype=torch.bool)
    sparse[0, [3, 100, 700]] = True
    context, target = sample_context_target_mask(sparse, context_ratio=0.25, patch_sizes=(12,))
    assert torch.all(context <= sparse)
    assert torch.all(target <= sparse)
    assert int((context | target).sum()) == 3
    print("masking tests passed")


if __name__ == "__main__":
    main()

