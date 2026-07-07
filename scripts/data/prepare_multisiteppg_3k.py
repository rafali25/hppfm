#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SPLITS = {
    "train": [f"P{i}" for i in range(1, 15)],
    "val": [f"P{i}" for i in range(15, 18)],
    "test": [f"P{i}" for i in range(18, 21)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Multi-site PPG from Hugging Face and build 3k timestamped 30s segments."
    )
    parser.add_argument("--repo-id", default="snowballlab/Multisite-PPG")
    parser.add_argument("--output-root", default="datasets/multisite_ppg_3k")
    parser.add_argument("--cache-dir", default="datasets/hf_cache")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--sites", nargs="+", default=["Watch"])
    parser.add_argument("--ppg-channel", choices=["green", "ir"], default="green")
    parser.add_argument("--segment-seconds", type=int, default=30)
    parser.add_argument("--target-fs", type=int, default=50)
    parser.add_argument("--source-step-seconds", type=int, default=30)
    parser.add_argument("--min-hr", type=float, default=40.0)
    parser.add_argument("--max-hr", type=float, default=180.0)
    return parser.parse_args()


def split_targets(max_samples: int) -> dict[str, int]:
    train = int(round(max_samples * 0.70))
    val = int(round(max_samples * 0.15))
    test = max_samples - train - val
    return {"train": train, "val": val, "test": test}


def timestamp_ms_to_iso(value_ms: float) -> str:
    return datetime.fromtimestamp(float(value_ms) / 1000.0, tz=timezone.utc).isoformat()


def resample_linear(signal: np.ndarray, fs_original: int, fs_target: int) -> np.ndarray:
    if fs_original == fs_target:
        return signal.astype(np.float32)
    duration = signal.shape[0] / fs_original
    old_t = np.arange(signal.shape[0], dtype=np.float64) / fs_original
    new_t = np.arange(0, duration, 1.0 / fs_target, dtype=np.float64)
    return np.interp(new_t, old_t, signal).astype(np.float32)


def zscore(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    return (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-8)


def download_npz(repo_id: str, participant: str, site: str, cache_dir: str | Path) -> Path:
    filename = f"ppg_windowed_data/{participant}/alignment_windows_{participant}_{site}.npz"
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=cache_dir,
        )
    )


def valid_start_indices(t0_ms: np.ndarray, segment_windows: int, step_windows: int) -> list[int]:
    starts: list[int] = []
    max_start = len(t0_ms) - segment_windows
    for start in range(0, max_start + 1, step_windows):
        times = t0_ms[start : start + segment_windows]
        if times.shape[0] != segment_windows:
            continue
        diffs = np.diff(times)
        if np.all((diffs >= 750.0) & (diffs <= 1250.0)):
            starts.append(start)
    return starts


def process_file(
    npz_path: Path,
    *,
    participant: str,
    site: str,
    split: str,
    remaining: int,
    output_root: Path,
    ppg_channel: str,
    segment_seconds: int,
    target_fs: int,
    source_step_seconds: int,
    min_hr: float,
    max_hr: float,
    global_offset: int,
) -> list[dict[str, str | int | float]]:
    data = np.load(npz_path)
    ppg_key = "ppg_green" if ppg_channel == "green" else "ppg_ir"
    windows = np.asarray(data[ppg_key], dtype=np.float32)
    hr = np.asarray(data["hr_gt"], dtype=np.float32)
    t0_ms = np.asarray(data["t0_ms"], dtype=np.float64)
    source_fs = int(round(float(data["ppg_fs"])))
    samples_per_second = source_fs
    segment_windows = int(segment_seconds)
    step_windows = int(source_step_seconds)
    starts = valid_start_indices(t0_ms, segment_windows, step_windows)

    subject_id = f"{participant}_{site}"
    out_dir = output_root / "segments" / split / subject_id / "hour_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []

    for start in starts:
        if len(rows) >= remaining:
            break
        hr_values = hr[start : start + segment_windows]
        if not np.all(np.isfinite(hr_values)):
            continue
        mean_hr = float(np.mean(hr_values))
        if mean_hr < min_hr or mean_hr > max_hr:
            continue

        segment = windows[start : start + segment_windows, :samples_per_second].reshape(-1)
        segment = zscore(resample_linear(segment, fs_original=source_fs, fs_target=target_fs))
        local_index = global_offset + len(rows)
        path = out_dir / f"ts_{local_index:06d}.npy"
        np.save(path, segment[:, None].astype(np.float32))
        rows.append(
            {
                "split": split,
                "subject_id": subject_id,
                "participant_id": participant,
                "site": site,
                "path": str(path),
                "start_time": timestamp_ms_to_iso(t0_ms[start]),
                "end_time": timestamp_ms_to_iso(t0_ms[start] + segment_seconds * 1000.0),
                "hr": mean_hr,
                "source_index": int(start),
                "source_file": str(npz_path),
            }
        )
    return rows


def write_labels(output_root: Path, rows: list[dict[str, str | int | float]]) -> Path:
    labels_path = output_root / "segment_metadata.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "subject_id",
                "participant_id",
                "site",
                "path",
                "start_time",
                "end_time",
                "hr",
                "source_index",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return labels_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = split_targets(args.max_samples)
    counts = {split: 0 for split in targets}
    all_rows: list[dict[str, str | int | float]] = []

    for split, participants in DEFAULT_SPLITS.items():
        for participant in participants:
            for site in args.sites:
                if counts[split] >= targets[split]:
                    continue
                remaining = targets[split] - counts[split]
                npz_path = download_npz(args.repo_id, participant, site, args.cache_dir)
                rows = process_file(
                    npz_path,
                    participant=participant,
                    site=site,
                    split=split,
                    remaining=remaining,
                    output_root=output_root,
                    ppg_channel=args.ppg_channel,
                    segment_seconds=args.segment_seconds,
                    target_fs=args.target_fs,
                    source_step_seconds=args.source_step_seconds,
                    min_hr=args.min_hr,
                    max_hr=args.max_hr,
                    global_offset=len(all_rows),
                )
                all_rows.extend(rows)
                counts[split] += len(rows)
                print(f"{participant} {site} {split}: wrote {len(rows)} segments")
                if counts[split] >= targets[split]:
                    break
            if counts[split] >= targets[split]:
                break

    labels_path = write_labels(output_root, all_rows)
    print(f"Wrote {len(all_rows)} 30s segments to {output_root / 'segments'}")
    print(f"Split counts: {counts}")
    print(f"Metadata: {labels_path}")
    if len(all_rows) < args.max_samples:
        raise RuntimeError(f"Only wrote {len(all_rows)} segments, requested {args.max_samples}")


if __name__ == "__main__":
    main()

