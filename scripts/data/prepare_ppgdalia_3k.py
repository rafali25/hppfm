#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


PPGDALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
DEFAULT_SOURCE_PPG_FS = 64
DEFAULT_TARGET_PPG_FS = 50
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_STRIDE_SECONDS = 30.0
DEFAULT_BASE_DATETIME = "2024-01-01T00:00:00"
PPGDALIA_HR_LABEL_PERIOD_SECONDS = 2.0
PPGDALIA_ACTIVITY_FS = 4
SUBJECT_SPLITS = {
    "train": {f"S{i}" for i in range(1, 11)},
    "val": {"S11", "S12"},
    "test": {"S13", "S14", "S15"},
}
_LAST_DOWNLOAD_PERCENT = -1

ACTIVITY_NAMES = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}
ACTIVE_ACTIVITY_IDS = {2, 3, 4, 7}
SEDENTARY_ACTIVITY_IDS = {1, 5, 6, 8}
SEGMENT_METADATA_COLUMNS = [
    "segment_id",
    "subject_id",
    "start_time",
    "ppg_path",
    "split",
    "hr_mean",
    "hr_std",
    "activity",
    "activity_name",
    "activity_binary",
    "source_start_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PPG-DaLiA and prepare a 3k-window Pulse-PPG-style sample."
    )
    parser.add_argument("--output-root", default="datasets/ppgdalia_3k")
    parser.add_argument("--archive-path", default="datasets/archives/ppg_dalia.zip")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--source-ppg-fs", type=int, default=DEFAULT_SOURCE_PPG_FS)
    parser.add_argument("--fs-target", type=int, default=DEFAULT_TARGET_PPG_FS)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--base-datetime", default=DEFAULT_BASE_DATETIME)
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument(
        "--include-transient-activity",
        action="store_true",
        help="Keep PPG-DaLiA activity 0 transient windows. Defaults to skipping them.",
    )
    parser.add_argument(
        "--keep-extracted-pickles",
        action="store_true",
        help="Keep extracted subject pickle files under output-root/raw_pickles.",
    )
    return parser.parse_args()


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    global _LAST_DOWNLOAD_PERCENT
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    percent = downloaded / total_size * 100
    percent_int = int(percent)
    if percent_int == _LAST_DOWNLOAD_PERCENT and downloaded < total_size:
        return
    _LAST_DOWNLOAD_PERCENT = percent_int
    mb_done = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"Downloading PPG-DaLiA: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")


def download_archive(archive_path: Path, redownload: bool) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and not redownload:
        print(f"Using existing archive: {archive_path}")
        return
    print(f"Downloading {PPGDALIA_URL}")
    urlretrieve(PPGDALIA_URL, archive_path, progress_hook)


def subject_from_member(member_name: str) -> str | None:
    parts = Path(member_name).parts
    for part in parts:
        if part.startswith("S") and part[1:].isdigit():
            return part
    name = Path(member_name).stem
    if name.startswith("S") and name[1:].isdigit():
        return name
    return None


def find_split(subject: str) -> str | None:
    for split, subjects in SUBJECT_SPLITS.items():
        if subject in subjects:
            return split
    return None


def extract_subject_pickles(archive_path: Path, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob("S*/S*.pkl"))
    if existing:
        print(f"Using {len(existing)} existing subject pickle files under {raw_dir}")
        return existing

    with zipfile.ZipFile(archive_path) as outer_zip:
        nested_data_members = [name for name in outer_zip.namelist() if name.endswith("data.zip")]
        if nested_data_members:
            print("Extracting subject pickle files from nested data.zip")
            with tempfile.TemporaryDirectory(prefix="ppgdalia_", dir=raw_dir.parent) as tmp_dir:
                nested_path = Path(tmp_dir) / "data.zip"
                with outer_zip.open(nested_data_members[0]) as source, nested_path.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
                with zipfile.ZipFile(nested_path) as data_zip:
                    return _extract_pickles_from_zip(data_zip, raw_dir)
        print("Extracting subject pickle files from archive")
        return _extract_pickles_from_zip(outer_zip, raw_dir)


def _extract_pickles_from_zip(zip_obj: zipfile.ZipFile, raw_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    members = sorted(
        name
        for name in zip_obj.namelist()
        if name.endswith(".pkl") and "PPG_FieldStudy" in name
    )
    if not members:
        raise FileNotFoundError("Could not find PPG_FieldStudy subject .pkl files in archive")

    for member in members:
        subject = subject_from_member(member)
        if subject is None:
            continue
        out_dir = raw_dir / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject}.pkl"
        if not out_path.exists():
            out_path.write_bytes(zip_obj.read(member))
        extracted.append(out_path)
    print(f"Extracted {len(extracted)} subject pickle files")
    return sorted(extracted)


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def resample_linear(signal: np.ndarray, fs_original: int, fs_target: int) -> np.ndarray:
    if fs_original == fs_target:
        return signal.astype(np.float32)
    target_len = int(round(signal.shape[0] * fs_target / fs_original))
    old_t = np.arange(signal.shape[0], dtype=np.float64) / fs_original
    new_t = np.arange(target_len, dtype=np.float64) / fs_target
    return np.interp(new_t, old_t, signal).astype(np.float32)


def crop_or_pad_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] == target_len:
        return signal.astype(np.float32)
    if signal.shape[0] > target_len:
        return signal[:target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[: signal.shape[0]] = signal.astype(np.float32)
    return out


def parse_base_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse --base-datetime '{value}' as ISO datetime") from exc


def source_start_seconds_from_sample(source_start_sample_index: int, source_ppg_fs: int) -> float:
    return float(source_start_sample_index) / float(source_ppg_fs)


def synthetic_start_time(base_datetime: datetime, source_start_seconds: float) -> datetime:
    return base_datetime + timedelta(seconds=float(source_start_seconds))


def safe_timestamp_for_segment_id(start_time: datetime) -> str:
    return start_time.strftime("%Y-%m-%dT%H-%M-%S")


def format_source_start_seconds_for_id(source_start_seconds: float) -> str:
    return f"start{int(round(source_start_seconds)):08d}s"


def make_ppgdalia_segment_id(
    *,
    subject_id: str,
    start_time: datetime,
    source_start_seconds: float,
) -> str:
    safe_time = safe_timestamp_for_segment_id(start_time)
    safe_source_start = format_source_start_seconds_for_id(source_start_seconds)
    return f"{subject_id}_{safe_time}_{safe_source_start}"


def activity_binary_label(activity_id: int) -> str:
    if activity_id in ACTIVE_ACTIVITY_IDS:
        return "active"
    if activity_id in SEDENTARY_ACTIVITY_IDS:
        return "sedentary"
    return "other"


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_subject_pickle(pickle_path: Path) -> dict:
    try:
        return pd.read_pickle(pickle_path)
    except UnicodeDecodeError:
        with pickle_path.open("rb") as f:
            return pickle.load(f, encoding="latin1")


def mode_int(values: np.ndarray) -> int:
    values = np.asarray(values).astype(np.int64)
    if values.size == 0:
        return -1
    uniques, counts = np.unique(values, return_counts=True)
    return int(uniques[np.argmax(counts)])


def split_targets(max_samples: int) -> dict[str, int]:
    train = int(round(max_samples * 0.70))
    val = int(round(max_samples * 0.15))
    test = max_samples - train - val
    return {"train": train, "val": val, "test": test}


def process_subject(
    pickle_path: Path,
    output_segments_root: Path,
    split: str,
    remaining: int,
    *,
    source_ppg_fs: int,
    fs_target: int,
    window_seconds: float,
    stride_seconds: float,
    base_datetime: datetime,
    include_transient_activity: bool,
) -> list[dict[str, str | int | float]]:
    df = read_subject_pickle(pickle_path)
    subject = pickle_path.stem

    bvp = np.asarray(df["signal"]["wrist"]["BVP"], dtype=np.float32).reshape(-1)
    bvp = zscore(bvp)
    labels_hr = np.asarray(df["label"], dtype=np.float32).reshape(-1)
    activity = np.asarray(df["activity"]).reshape(-1)
    source_window_samples = int(round(window_seconds * source_ppg_fs))
    source_stride_samples = int(round(stride_seconds * source_ppg_fs))
    target_window_samples = int(round(window_seconds * fs_target))
    if source_window_samples <= 0 or source_stride_samples <= 0:
        raise ValueError("window-seconds and stride-seconds must produce positive source samples")

    rows: list[dict[str, str | int | float]] = []
    out_dir = output_segments_root / split / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    for source_start_sample_index in range(0, len(bvp) - source_window_samples + 1, source_stride_samples):
        if len(rows) >= remaining:
            break
        source_start_seconds = source_start_seconds_from_sample(
            source_start_sample_index,
            source_ppg_fs=source_ppg_fs,
        )
        source_stop_seconds = source_start_seconds + window_seconds

        hr_start = int(np.floor(source_start_seconds / PPGDALIA_HR_LABEL_PERIOD_SECONDS))
        hr_stop = int(np.ceil(source_stop_seconds / PPGDALIA_HR_LABEL_PERIOD_SECONDS))
        hr_window = labels_hr[hr_start:hr_stop]
        hr_window = hr_window[np.isfinite(hr_window)]
        if hr_window.size == 0:
            continue

        act_start = int(np.floor(source_start_seconds * PPGDALIA_ACTIVITY_FS))
        act_stop = int(np.ceil(source_stop_seconds * PPGDALIA_ACTIVITY_FS))
        activity_window = activity[act_start:act_stop]
        if activity_window.size == 0:
            continue
        activity_id = mode_int(activity_window)
        if activity_id == 0 and not include_transient_activity:
            continue

        source_segment = bvp[
            source_start_sample_index : source_start_sample_index + source_window_samples
        ]
        saved_segment = resample_linear(
            source_segment,
            fs_original=source_ppg_fs,
            fs_target=fs_target,
        )
        if saved_segment.shape[0] != target_window_samples:
            saved_segment = crop_or_pad_1d(saved_segment, target_window_samples)
        start_time = synthetic_start_time(base_datetime, source_start_seconds)
        segment_id = make_ppgdalia_segment_id(
            subject_id=subject,
            start_time=start_time,
            source_start_seconds=source_start_seconds,
        )
        path = out_dir / f"{segment_id}.npy"
        np.save(path, saved_segment[:, None].astype(np.float32))

        rows.append(
            {
                "segment_id": segment_id,
                "subject_id": subject,
                "start_time": start_time.isoformat(timespec="seconds"),
                "ppg_path": relative_path(path),
                "split": split,
                "hr_mean": float(np.mean(hr_window, dtype=np.float64)),
                "hr_std": float(np.std(hr_window, dtype=np.float64)),
                "activity": activity_id,
                "activity_name": ACTIVITY_NAMES.get(activity_id, f"activity_{activity_id}"),
                "activity_binary": activity_binary_label(activity_id),
                "source_start_seconds": int(round(source_start_seconds)),
            }
        )
    return rows


def write_labels(labels_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    archive_path = Path(args.archive_path)
    raw_dir = output_root / "raw_pickles"
    segments_root = output_root / f"segments_{int(round(args.window_seconds))}s_{args.fs_target}Hz"
    base_datetime = parse_base_datetime(args.base_datetime)

    download_archive(archive_path, args.redownload)
    pickle_paths = extract_subject_pickles(archive_path, raw_dir)

    targets = split_targets(args.max_samples)
    counts = {split: 0 for split in targets}
    rows: list[dict[str, str | int | float]] = []

    for pickle_path in pickle_paths:
        subject = pickle_path.stem
        split = find_split(subject)
        if split is None or counts[split] >= targets[split]:
            continue
        remaining = targets[split] - counts[split]
        subject_rows = process_subject(
            pickle_path,
            segments_root,
            split,
            remaining,
            source_ppg_fs=args.source_ppg_fs,
            fs_target=args.fs_target,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            base_datetime=base_datetime,
            include_transient_activity=args.include_transient_activity,
        )
        rows.extend(subject_rows)
        counts[split] += len(subject_rows)
        print(f"{subject}: wrote {len(subject_rows)} {split} segments")
        if all(counts[split] >= targets[split] for split in targets):
            break

    labels_path = output_root / "segment_labels.csv"
    manifest_path = output_root / f"{segments_root.name}_manifest.csv"
    compatibility_labels_path = output_root / "labels.csv"
    write_labels(labels_path, rows)
    write_labels(manifest_path, rows)
    write_labels(compatibility_labels_path, rows)
    print(f"Wrote {len(rows)} segments to {segments_root}")
    print(f"Split counts: {counts}")
    print(f"Segment labels: {labels_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Labels: {compatibility_labels_path}")

    if not args.keep_extracted_pickles:
        # Keep the archive, remove only extracted intermediates.
        for path in sorted(raw_dir.glob("S*/S*.pkl")):
            path.unlink()
        for path in sorted(raw_dir.glob("S*"), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        try:
            raw_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
