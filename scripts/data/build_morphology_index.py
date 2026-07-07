#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import build_morphology_records, write_morphology_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PaPaGei-style morphology labels for a Pulse-PPG folder tree."
    )
    parser.add_argument("--data-root", required=True, help="Folder with train/val/test subfolders.")
    parser.add_argument(
        "--output-csv",
        default="datasets/morphology/pulseppg_morphology_index.csv",
        help="Where to write the morphology index CSV.",
    )
    parser.add_argument("--fs", type=int, default=50, help="Sampling frequency of input .npy files.")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=10.0,
        help="Center-crop length used to compute morphology labels. Use 0 for full file.",
    )
    parser.add_argument("--channel", type=int, default=0, help="PPG channel index.")
    parser.add_argument("--svri-bins", type=int, default=8, help="Number of sVRI bins.")
    parser.add_argument(
        "--keep-unfiltered",
        action="store_true",
        help="Keep rows outside PaPaGei's default morphology sanity ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segment_seconds = None if args.segment_seconds <= 0 else args.segment_seconds
    records, bin_edges = build_morphology_records(
        args.data_root,
        fs=args.fs,
        segment_seconds=segment_seconds,
        channel=args.channel,
        num_svri_bins=args.svri_bins,
        keep_unfiltered=args.keep_unfiltered,
    )
    write_morphology_index(records, args.output_csv, bin_edges=bin_edges)
    print(f"Wrote {len(records)} morphology rows to {args.output_csv}")
    print(f"Wrote sVRI bin edges to {Path(args.output_csv).with_suffix('.svri_bin_edges.npy')}")


if __name__ == "__main__":
    main()
