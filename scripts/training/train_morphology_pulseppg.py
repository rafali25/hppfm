#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import (
    PulsePPGMorphologyDataset,
    build_morphology_records,
    load_morphology_index,
    write_morphology_index,
)
from src.encoder import PulsePPGResNet1D
from src.heads import MorphologyAwarePulsePPG
from src.helpers import get_device, set_seed, write_json
from src.trainer import MorphologyTrainConfig, fit_morphology_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Pulse-PPG encoder with PaPaGei-S morphology-aware SSL."
    )
    parser.add_argument("--data-root", required=True, help="Folder with train/val/test subfolders.")
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Precomputed morphology index. If omitted, use --build-index.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build morphology index before training.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/runs/morphology_pulseppg",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--fs", type=int, default=50)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--svri-bins", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-sqi-loss", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--base-filters", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-block", type=int, default=12)
    parser.add_argument("--final-pool", choices=["max", "avg"], default="max")
    parser.add_argument("--no-dropout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))

    segment_seconds = None if args.segment_seconds <= 0 else args.segment_seconds
    index_csv = args.index_csv
    if args.build_index:
        index_csv = str(output_dir / "morphology_index.csv")
        records, bin_edges = build_morphology_records(
            args.data_root,
            fs=args.fs,
            segment_seconds=segment_seconds,
            channel=args.channel,
            num_svri_bins=args.svri_bins,
        )
        write_morphology_index(records, index_csv, bin_edges=bin_edges)
    elif index_csv is None:
        raise ValueError("Pass --index-csv or use --build-index")

    records = load_morphology_index(index_csv)
    train_dataset = PulsePPGMorphologyDataset(
        records,
        split="train",
        fs=args.fs,
        segment_seconds=segment_seconds,
        channel=args.channel,
        random_crop=args.random_crop,
        noise_std=args.noise_std,
    )
    try:
        val_dataset = PulsePPGMorphologyDataset(
            records,
            split="val",
            fs=args.fs,
            segment_seconds=segment_seconds,
            channel=args.channel,
            random_crop=False,
            noise_std=0.0,
        )
    except ValueError:
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )

    encoder = PulsePPGResNet1D(
        in_channels=1,
        base_filters=args.base_filters,
        kernel_size=args.kernel_size,
        stride=args.stride,
        groups=1,
        n_block=args.n_block,
        final_pool=args.final_pool,
        use_dropout=not args.no_dropout,
    )
    model = MorphologyAwarePulsePPG(
        encoder=encoder,
        embedding_dim=args.embedding_dim,
        n_experts=args.n_experts,
    )

    config = MorphologyTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        temperature=args.temperature,
        use_sqi_loss=not args.no_sqi_loss,
        grad_clip=args.grad_clip,
    )
    device = get_device(args.device)
    print(f"Training on {device} with {len(train_dataset)} train records")
    fit_morphology_model(
        model,
        train_loader,
        val_loader,
        device=device,
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
