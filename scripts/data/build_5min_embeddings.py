#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.data import build_5min_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate short-window embeddings into 5-minute bins.")
    parser.add_argument("--embeddings", required=True, help="Input embeddings.npy with shape [N, D].")
    parser.add_argument("--metadata", required=True, help="Segment metadata CSV with start_time.")
    parser.add_argument("--out-dir", required=True, help="Output directory for five_min files.")
    parser.add_argument("--min-valid-segments", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings, metadata_path = build_5min_embeddings(
        args.embeddings,
        args.metadata,
        args.out_dir,
        min_valid_segments=args.min_valid_segments,
    )
    print("Created 5-min embeddings")
    print(f"Shape: {list(embeddings.shape)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

