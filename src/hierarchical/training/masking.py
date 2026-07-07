from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def sample_context_target_mask(
    valid_mask: torch.Tensor,
    context_ratio: float = 0.25,
    patch_sizes: Sequence[int] = (1, 4, 12),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample sparse multi-scale context patches from valid week tokens."""
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected valid_mask with shape [B, T], got {tuple(valid_mask.shape)}")
    if not 0.0 < context_ratio <= 1.0:
        raise ValueError("context_ratio must be in (0, 1]")
    if not patch_sizes:
        raise ValueError("patch_sizes cannot be empty")

    valid = valid_mask.bool()
    batch_size, seq_len = valid.shape
    context = torch.zeros_like(valid)

    for batch_idx in range(batch_size):
        valid_count = int(valid[batch_idx].sum().item())
        if valid_count == 0:
            continue

        patch_size = int(random.choice(tuple(patch_sizes)))
        patch_size = max(1, patch_size)
        patch_starts = list(range(0, seq_len, patch_size))
        random.shuffle(patch_starts)
        target_count = max(1, int(round(valid_count * context_ratio)))
        if valid_count > 1:
            target_count = min(target_count, valid_count - 1)

        selected_valid = 0
        for start in patch_starts:
            stop = min(seq_len, start + patch_size)
            patch_valid = valid[batch_idx, start:stop]
            count = int(patch_valid.sum().item())
            if count == 0:
                continue
            context[batch_idx, start:stop] |= patch_valid
            selected_valid += count
            if selected_valid >= target_count:
                break

    target = valid & ~context
    return context, target

