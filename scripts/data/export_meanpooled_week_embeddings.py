#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export mean-pooled week embedding baseline.")
    parser.add_argument("--week-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = data[key]
    return str(value.item() if value.shape == () else value)


def mean_pool(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.sum() == 0:
        return np.zeros(x.shape[1], dtype=np.float32)
    return x[valid_mask.astype(bool)].mean(axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    week_embeddings = []
    week_rows = []
    for path in sorted(Path(args.week_dir).glob("*.npz")):
        data = np.load(path)
        embedding_index = len(week_embeddings)
        week_embeddings.append(mean_pool(data["x"].astype(np.float32), data["valid_mask"].astype(bool)))
        week_rows.append(
            {
                "subject_id": scalar(data, "subject_id"),
                "week_start_time": scalar(data, "week_start_time"),
                "split": scalar(data, "split"),
                "embedding_index": embedding_index,
            }
        )

    week_array = np.stack(week_embeddings).astype(np.float32)
    np.save(out_dir / "meanpooled_week_embeddings.npy", week_array)
    with (out_dir / "meanpooled_week_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "week_start_time", "split", "embedding_index"])
        writer.writeheader()
        writer.writerows(week_rows)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in week_rows:
        groups[(str(row["subject_id"]), str(row["split"]))].append(int(row["embedding_index"]))
    subject_embeddings = []
    subject_rows = []
    for embedding_index, ((subject_id, split), indices) in enumerate(sorted(groups.items())):
        subject_embeddings.append(week_array[np.asarray(indices)].mean(axis=0))
        subject_rows.append({"subject_id": subject_id, "split": split, "embedding_index": embedding_index})
    np.save(out_dir / "subject_embeddings.npy", np.stack(subject_embeddings).astype(np.float32))
    with (out_dir / "subject_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "split", "embedding_index"])
        writer.writeheader()
        writer.writerows(subject_rows)
    print(f"Exported {len(week_rows)} mean-pooled week embeddings to {out_dir}")


if __name__ == "__main__":
    main()
