#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder import PulsePPGResNet1D
from src.heads import MorphologyAwarePulsePPG


def resolve_signal_path(row: dict[str, str]) -> Path:
    path_value = row.get("path") or row.get("ppg_path")
    if not path_value:
        raise KeyError("metadata row must include path or ppg_path")
    path = Path(path_value)
    if path.exists():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return path


class SegmentDataset(Dataset):
    def __init__(self, metadata_csv: str | Path):
        with Path(metadata_csv).open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"No metadata rows found in {metadata_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        signal = np.load(resolve_signal_path(row)).astype(np.float32).reshape(-1)
        signal = (signal - np.nanmean(signal)) / (np.nanstd(signal) + 1e-8)
        return {
            "signal": torch.from_numpy(signal[None, :].astype(np.float32)),
            "row": row,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "signal": torch.stack([item["signal"] for item in batch], dim=0),
        "rows": [item["row"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PulsePPG+PaPaGei segment embeddings.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", default="datasets/runs/ppgdalia_3k_gpu_smoke/checkpoint_best.pt")
    parser.add_argument("--model-args", default="datasets/runs/ppgdalia_3k_gpu_smoke/args.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> MorphologyAwarePulsePPG:
    model_args = json.loads(Path(args.model_args).read_text(encoding="utf-8"))
    encoder = PulsePPGResNet1D(
        in_channels=1,
        base_filters=int(model_args.get("base_filters", 128)),
        kernel_size=int(model_args.get("kernel_size", 11)),
        stride=int(model_args.get("stride", 2)),
        groups=1,
        n_block=int(model_args.get("n_block", 12)),
        final_pool=model_args.get("final_pool", "max"),
        use_dropout=not bool(model_args.get("no_dropout", False)),
    )
    model = MorphologyAwarePulsePPG(
        encoder=encoder,
        embedding_dim=int(model_args.get("embedding_dim", 512)),
        n_experts=int(model_args.get("n_experts", 3)),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = SegmentDataset(args.metadata)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    model = load_model(args, device)
    embeddings: list[np.ndarray] = []
    rows_out: list[dict[str, str | int]] = []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            outputs = model(signal)
            emb = outputs["embedding"].detach().cpu().numpy().astype(np.float32)
            start = len(rows_out)
            embeddings.append(emb)
            for offset, row in enumerate(batch["rows"]):
                out_row = dict(row)
                out_row["embedding_index"] = start + offset
                rows_out.append(out_row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_array = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(out_dir / "embeddings.npy", embeddings_array)
    with (out_dir / "segment_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Exported embeddings: {embeddings_array.shape}")
    print(f"Metadata: {out_dir / 'segment_metadata.csv'}")


if __name__ == "__main__":
    main()
