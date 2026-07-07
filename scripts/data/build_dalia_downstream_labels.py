#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.helpers import skewness_sqi
from src.hierarchical.data.build_5min import floor_to_five_minutes, parse_datetime


SEDENTARY_ACTIVITY_IDS = {1, 5, 6, 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5-minute downstream labels for timestamped PPG-DaLiA.")
    parser.add_argument("--segment-metadata", required=True)
    parser.add_argument("--five-min-metadata", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--morphology-index", default=None)
    parser.add_argument("--fs", type=int, default=50)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def path_keys(value: str) -> list[str]:
    path = Path(value)
    keys = [path.as_posix()]
    try:
        keys.append(path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix())
    except ValueError:
        pass
    keys.append((PROJECT_ROOT / path).resolve().as_posix())
    return list(dict.fromkeys(keys))


def load_sqi_by_path(index_csv: str | Path | None, *, fs: int) -> dict[str, float]:
    if index_csv is None:
        return {}
    sqi_by_path: dict[str, float] = {}
    for row in read_csv(index_csv):
        value = float(row["sqi"])
        for key in path_keys(row["path"]):
            sqi_by_path[key] = value
    return sqi_by_path


def lookup_sqi(row: dict[str, str], sqi_by_path: dict[str, float], *, fs: int) -> float:
    path_value = row.get("ppg_path") or row.get("path")
    if path_value is None:
        return float("nan")
    for key in path_keys(path_value):
        if key in sqi_by_path:
            return sqi_by_path[key]
    path = Path(path_value)
    if not path.exists():
        path = PROJECT_ROOT / path
    signal = np.load(path).astype(np.float32).reshape(-1)
    return float(skewness_sqi(signal, fs=fs))


def read_five_min_rows(path: str | Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = {}
    for row in read_csv(path):
        key = (row["subject_id"], row["split"], parse_datetime(row["bin_start_time"]).isoformat())
        rows[key] = row
    return rows


def majority_activity(rows: list[dict[str, str]]) -> tuple[int | str, int | str, str]:
    values = [int(row["activity"]) for row in rows if row.get("activity", "") != ""]
    if not values:
        return "", "", ""
    activity_id = Counter(values).most_common(1)[0][0]
    name = next((row.get("activity_name", "") for row in rows if int(row["activity"]) == activity_id), "")
    return int(activity_id in SEDENTARY_ACTIVITY_IDS), activity_id, name


def main() -> None:
    args = parse_args()
    five_min_rows = read_five_min_rows(args.five_min_metadata)
    segment_rows = read_csv(args.segment_metadata)
    sqi_by_path = load_sqi_by_path(args.morphology_index, fs=args.fs)

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in segment_rows:
        start = parse_datetime(row["start_time"])
        bin_start = floor_to_five_minutes(start).isoformat()
        key = (row["subject_id"], row["split"], bin_start)
        if key in five_min_rows:
            grouped[key].append(row)

    out_rows: list[dict[str, str | int | float]] = []
    for key, rows in sorted(grouped.items()):
        subject_id, split, bin_start = key
        five_min_row = five_min_rows[key]
        hr_values = np.asarray([float(row["hr_mean"]) for row in rows], dtype=np.float64)
        sqi_values = np.asarray([lookup_sqi(row, sqi_by_path, fs=args.fs) for row in rows], dtype=np.float64)
        sqi_values = sqi_values[np.isfinite(sqi_values)]
        sedentary, activity9, activity_label = majority_activity(rows)
        out_rows.append(
            {
                "subject_id": subject_id,
                "split": split,
                "bin_start_time": bin_start,
                "five_min_index": int(five_min_row["five_min_index"]),
                "valid_mask": int(five_min_row["valid_mask"]),
                "segment_count": len(rows),
                "hr": float(np.mean(hr_values)),
                "sqi": "" if sqi_values.size == 0 else float(np.mean(sqi_values)),
                "sedentary": sedentary,
                "activity9": activity9,
                "activity_label": activity_label,
                "activity_label_count": len(rows),
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "split",
                "bin_start_time",
                "five_min_index",
                "valid_mask",
                "segment_count",
                "hr",
                "sqi",
                "sedentary",
                "activity9",
                "activity_label",
                "activity_label_count",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} downstream label rows to {out_path}")
    print(dict(Counter(row["activity_label"] for row in out_rows if row["activity_label"] != "")))


if __name__ == "__main__":
    main()
