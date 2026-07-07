from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .build_5min import parse_datetime


BINS_PER_HOUR = 12
BINS_PER_DAY = 24 * BINS_PER_HOUR
BINS_PER_WEEK = 7 * BINS_PER_DAY


@dataclass(frozen=True)
class FiveMinuteMetadata:
    subject_id: str
    split: str
    bin_start_time: datetime
    five_min_index: int
    valid_mask: bool


def monday_week_start(value: datetime) -> datetime:
    start = value - timedelta(days=value.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def time_of_day_index(value: datetime) -> int:
    return value.hour * BINS_PER_HOUR + value.minute // 5


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def read_five_min_metadata(metadata_csv: str | Path) -> list[FiveMinuteMetadata]:
    rows: list[FiveMinuteMetadata] = []
    with Path(metadata_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            subject_id = row.get("subject_id") or row.get("subject")
            split = row.get("split")
            bin_start_time = row.get("bin_start_time")
            if subject_id is None or split is None or bin_start_time is None:
                raise ValueError("five_min_metadata.csv must include subject_id, split, bin_start_time")
            index_raw = row.get("five_min_index") or row.get("embedding_index") or str(row_idx)
            valid_raw = row.get("valid_mask", "1")
            rows.append(
                FiveMinuteMetadata(
                    subject_id=subject_id,
                    split=split,
                    bin_start_time=parse_datetime(bin_start_time),
                    five_min_index=int(index_raw),
                    valid_mask=valid_raw.lower() in {"1", "true", "yes"},
                )
            )
    return rows


def build_week_dataset(
    five_min_embeddings_path: str | Path,
    five_min_metadata_csv: str | Path,
    out_dir: str | Path,
    *,
    min_valid_fraction: float = 0.2,
) -> Counter:
    embeddings = np.load(five_min_embeddings_path).astype(np.float32)
    metadata = read_five_min_metadata(five_min_metadata_csv)
    if metadata and max(row.five_min_index for row in metadata) >= embeddings.shape[0]:
        raise IndexError("five_min_metadata references an index outside five_min_embeddings.npy")

    groups: dict[tuple[str, str, datetime], list[FiveMinuteMetadata]] = defaultdict(list)
    for row in metadata:
        groups[(row.subject_id, row.split, monday_week_start(row.bin_start_time))].append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    feature_dim = embeddings.shape[1]
    min_valid = int(np.ceil(BINS_PER_WEEK * min_valid_fraction))

    for (subject_id, split, week_start), rows in sorted(groups.items(), key=lambda item: item[0]):
        x = np.zeros((BINS_PER_WEEK, feature_dim), dtype=np.float32)
        valid_mask = np.zeros(BINS_PER_WEEK, dtype=bool)
        day_of_week = np.repeat(np.arange(7, dtype=np.int64), BINS_PER_DAY)
        time_of_day = np.tile(np.arange(BINS_PER_DAY, dtype=np.int64), 7)

        for row in rows:
            day_idx = row.bin_start_time.weekday()
            tod_idx = time_of_day_index(row.bin_start_time)
            week_idx = day_idx * BINS_PER_DAY + tod_idx
            if 0 <= week_idx < BINS_PER_WEEK and row.valid_mask:
                x[week_idx] = embeddings[row.five_min_index]
                valid_mask[week_idx] = True

        if int(valid_mask.sum()) < min_valid:
            continue

        counts[split] += 1
        file_name = f"{safe_name(split)}__{safe_name(subject_id)}__{week_start.date().isoformat()}.npz"
        np.savez_compressed(
            out_dir / file_name,
            x=x,
            valid_mask=valid_mask,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            subject_id=np.asarray(subject_id),
            week_start_time=np.asarray(week_start.isoformat()),
            split=np.asarray(split),
        )

    return counts
