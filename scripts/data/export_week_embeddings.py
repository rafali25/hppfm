#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.models import WeekTemporalTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-II week and subject embeddings.")
    parser.add_argument("--week-dir", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-subject-average", action="store_true")
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = data[key]
    return str(value.item() if value.shape == () else value)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    model = WeekTemporalTransformer(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    week_embeddings: list[np.ndarray] = []
    week_rows: list[dict[str, str | int]] = []
    with torch.no_grad():
        for path in sorted(Path(args.week_dir).glob("*.npz")):
            data = np.load(path)
            x = torch.from_numpy(data["x"].astype(np.float32)).unsqueeze(0).to(device)
            valid_mask = torch.from_numpy(data["valid_mask"].astype(bool)).unsqueeze(0).to(device)
            day = torch.from_numpy(data["day_of_week"].astype(np.int64)).unsqueeze(0).to(device)
            tod = torch.from_numpy(data["time_of_day"].astype(np.int64)).unsqueeze(0).to(device)
            outputs = model(x, valid_mask, day, tod)
            pooled = model.mean_pool(outputs, valid_mask).squeeze(0).cpu().numpy().astype(np.float32)
            embedding_index = len(week_embeddings)
            week_embeddings.append(pooled)
            week_rows.append(
                {
                    "subject_id": scalar(data, "subject_id"),
                    "week_start_time": scalar(data, "week_start_time"),
                    "split": scalar(data, "split"),
                    "embedding_index": embedding_index,
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    week_array = np.stack(week_embeddings).astype(np.float32)
    np.save(out_dir / "week_embeddings.npy", week_array)
    with (out_dir / "week_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "week_start_time", "split", "embedding_index"])
        writer.writeheader()
        writer.writerows(week_rows)

    if not args.no_subject_average:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in week_rows:
            groups[(str(row["subject_id"]), str(row["split"]))].append(int(row["embedding_index"]))
        subject_embeddings = []
        subject_rows = []
        for embedding_index, ((subject_id, split), indices) in enumerate(sorted(groups.items())):
            subject_embeddings.append(week_array[np.asarray(indices)].mean(axis=0))
            subject_rows.append(
                {"subject_id": subject_id, "split": split, "embedding_index": embedding_index}
            )
        np.save(out_dir / "subject_embeddings.npy", np.stack(subject_embeddings).astype(np.float32))
        with (out_dir / "subject_metadata.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject_id", "split", "embedding_index"])
            writer.writeheader()
            writer.writerows(subject_rows)

    print(f"Exported {len(week_rows)} week embeddings to {out_dir}")


if __name__ == "__main__":
    main()
