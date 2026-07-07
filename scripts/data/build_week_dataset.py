#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.data import build_week_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2016-token week tensors from 5-minute embeddings.")
    parser.add_argument("--five-min-embeddings", required=True)
    parser.add_argument("--five-min-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-valid-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_week_dataset(
        args.five_min_embeddings,
        args.five_min_metadata,
        args.out_dir,
        min_valid_fraction=args.min_valid_fraction,
    )
    print(f"Train weeks: {counts.get('train', 0)}")
    print(f"Val weeks: {counts.get('val', 0)}")
    print(f"Test weeks: {counts.get('test', 0)}")
    print("Each week shape: [2016, D]")


if __name__ == "__main__":
    main()
