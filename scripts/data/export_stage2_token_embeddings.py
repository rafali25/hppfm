#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.data.build_5min import parse_datetime
from src.hierarchical.models import WeekTemporalTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-II per-token embeddings for valid 5-minute bins.")
    parser.add_argument("--week-dir", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-invalid", action="store_true")
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

    embeddings: list[np.ndarray] = []
    rows: list[dict[str, str | int]] = []
    with torch.no_grad():
        for path in sorted(Path(args.week_dir).glob("*.npz")):
            data = np.load(path)
            subject_id = scalar(data, "subject_id")
            split = scalar(data, "split")
            week_start = parse_datetime(scalar(data, "week_start_time"))
            x = torch.from_numpy(data["x"].astype(np.float32)).unsqueeze(0).to(device)
            valid_mask_np = data["valid_mask"].astype(bool)
            valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0).to(device)
            day = torch.from_numpy(data["day_of_week"].astype(np.int64)).unsqueeze(0).to(device)
            tod = torch.from_numpy(data["time_of_day"].astype(np.int64)).unsqueeze(0).to(device)
            outputs = model(x, valid_mask, day, tod).squeeze(0).detach().cpu().numpy().astype(np.float32)

            token_indices = range(outputs.shape[0]) if args.include_invalid else np.flatnonzero(valid_mask_np)
            for token_index in token_indices:
                embedding_index = len(rows)
                embeddings.append(outputs[int(token_index)])
                rows.append(
                    {
                        "subject_id": subject_id,
                        "split": split,
                        "bin_start_time": (week_start + timedelta(minutes=5 * int(token_index))).isoformat(),
                        "week_start_time": week_start.isoformat(),
                        "token_index": int(token_index),
                        "valid_mask": int(valid_mask_np[int(token_index)]),
                        "embedding_index": embedding_index,
                    }
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_array = np.stack(embeddings).astype(np.float32)
    np.save(out_dir / "token_embeddings.npy", embeddings_array)
    with (out_dir / "token_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "split",
                "bin_start_time",
                "week_start_time",
                "token_index",
                "valid_mask",
                "embedding_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported Stage-II token embeddings: {embeddings_array.shape}")
    print(f"Metadata: {out_dir / 'token_metadata.csv'}")


if __name__ == "__main__":
    main()
