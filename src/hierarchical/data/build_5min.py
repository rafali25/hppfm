from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SegmentMetadata:
    subject_id: str
    split: str
    start_time: datetime
    embedding_index: int


def parse_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(f"Could not parse start_time '{value}' as an ISO datetime") from exc


def floor_to_five_minutes(value: datetime) -> datetime:
    minute = (value.minute // 5) * 5
    return value.replace(minute=minute, second=0, microsecond=0)


def _first_present(row: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def read_segment_metadata(metadata_csv: str | Path) -> list[SegmentMetadata]:
    rows: list[SegmentMetadata] = []
    with Path(metadata_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            subject_id = _first_present(row, ("subject_id", "subject", "participant_id"))
            split = _first_present(row, ("split", "set"))
            start_time = _first_present(row, ("start_time", "segment_start_time", "timestamp"))
            if subject_id is None or split is None or start_time is None:
                raise ValueError(
                    "segment metadata must include subject_id/subject, split, and start_time columns"
                )
            embedding_index_raw = _first_present(row, ("embedding_index", "segment_index", "index"))
            embedding_index = row_idx if embedding_index_raw is None else int(embedding_index_raw)
            rows.append(
                SegmentMetadata(
                    subject_id=subject_id,
                    split=split,
                    start_time=parse_datetime(start_time),
                    embedding_index=embedding_index,
                )
            )
    return rows


def build_5min_embeddings(
    embeddings_path: str | Path,
    metadata_csv: str | Path,
    out_dir: str | Path,
    *,
    min_valid_segments: int = 3,
) -> tuple[np.ndarray, Path]:
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata = read_segment_metadata(metadata_csv)
    if len(metadata) == 0:
        raise ValueError("No segment metadata rows found")
    if max(row.embedding_index for row in metadata) >= embeddings.shape[0]:
        raise IndexError("segment metadata references an embedding_index outside embeddings.npy")

    groups: dict[tuple[str, str, datetime], list[int]] = defaultdict(list)
    for row in metadata:
        key = (row.subject_id, row.split, floor_to_five_minutes(row.start_time))
        groups[key].append(row.embedding_index)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_embeddings: list[np.ndarray] = []
    output_rows: list[dict[str, str | int]] = []
    for five_min_index, (key, indices) in enumerate(sorted(groups.items(), key=lambda item: item[0])):
        subject_id, split, bin_start = key
        values = embeddings[np.asarray(indices, dtype=np.int64)]
        output_embeddings.append(values.mean(axis=0).astype(np.float32))
        valid_count = len(indices)
        output_rows.append(
            {
                "subject_id": subject_id,
                "bin_start_time": bin_start.isoformat(),
                "split": split,
                "valid_count": valid_count,
                "valid_mask": int(valid_count >= min_valid_segments),
                "five_min_index": five_min_index,
            }
        )

    five_min_embeddings = np.stack(output_embeddings).astype(np.float32)
    embeddings_out = out_dir / "five_min_embeddings.npy"
    metadata_out = out_dir / "five_min_metadata.csv"
    np.save(embeddings_out, five_min_embeddings)
    with metadata_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "bin_start_time",
                "split",
                "valid_count",
                "valid_mask",
                "five_min_index",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return five_min_embeddings, metadata_out

