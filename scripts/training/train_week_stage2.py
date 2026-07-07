#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical.models import DualDecoderHead, WeekTemporalTransformer
from src.hierarchical.training import sample_context_target_mask


class WeekDataset(Dataset):
    def __init__(self, week_dir: str | Path, *, split: str | None = None):
        self.files = sorted(Path(week_dir).glob("*.npz"))
        if split is not None:
            self.files = [path for path in self.files if _npz_scalar(path, "split") == split]
        if not self.files:
            raise FileNotFoundError(f"No week .npz files found in {week_dir} for split={split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        data = np.load(path)
        return {
            "x": torch.from_numpy(data["x"].astype(np.float32)),
            "valid_mask": torch.from_numpy(data["valid_mask"].astype(bool)),
            "day_of_week": torch.from_numpy(data["day_of_week"].astype(np.int64)),
            "time_of_day": torch.from_numpy(data["time_of_day"].astype(np.int64)),
            "path": str(path),
        }


def _npz_scalar(path: Path, key: str) -> str:
    with np.load(path) as data:
        value = data[key]
        return str(value.item() if value.shape == () else value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WavesFM-style Stage-II masked reconstruction.")
    parser.add_argument("--week-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-ratio", type=float, default=0.25)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[1, 4, 12])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Train for two batches only.")
    return parser.parse_args()


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def masked_reconstruction_loss(
    pred_local: torch.Tensor,
    pred_global: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    mask = target_mask.bool().unsqueeze(-1)
    if int(mask.sum().item()) == 0:
        return (pred_local.sum() + pred_global.sum()) * 0.0
    target_norm = F.normalize(target, dim=-1)
    local_loss = ((F.normalize(pred_local, dim=-1) - target_norm) ** 2)[mask.expand_as(target)].mean()
    global_loss = ((F.normalize(pred_global, dim=-1) - target_norm) ** 2)[mask.expand_as(target)].mean()
    return local_loss + global_loss


def run_epoch(
    model: WeekTemporalTransformer,
    decoders: DualDecoderHead,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> float:
    train = optimizer is not None
    model.train(mode=train)
    decoders.train(mode=train)
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            batch = to_device(batch, device)
            context_mask, target_mask = sample_context_target_mask(
                batch["valid_mask"],
                context_ratio=args.context_ratio,
                patch_sizes=args.patch_sizes,
            )
            outputs = model(
                batch["x"],
                context_mask,
                batch["day_of_week"],
                batch["time_of_day"],
            )
            pred_local, pred_global = decoders(
                outputs,
                batch["day_of_week"],
                batch["time_of_day"],
                context_mask=context_mask,
            )
            loss = masked_reconstruction_loss(pred_local, pred_global, batch["x"], target_mask)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = int(batch["x"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            if args.dry_run and batch_idx >= 1:
                break

    return total_loss / max(1, total_items)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    model: WeekTemporalTransformer,
    decoders: DualDecoderHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "decoders": decoders.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_args": {
                "input_dim": args.input_dim,
                "model_dim": args.model_dim,
                "num_layers": args.layers,
                "nhead": args.heads,
                "dim_feedforward": args.ff_dim,
                "dropout": args.dropout,
            },
            "decoder_args": {
                "input_dim": args.input_dim,
                "model_dim": args.model_dim,
                "nhead": args.heads,
                "dropout": args.dropout,
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", vars(args))

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_dataset = WeekDataset(args.week_dir, split="train")
    try:
        val_dataset = WeekDataset(args.week_dir, split="val")
    except FileNotFoundError:
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = WeekTemporalTransformer(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_layers=args.layers,
        nhead=args.heads,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    decoders = DualDecoderHead(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        nhead=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(decoders.parameters()), lr=args.lr)
    history: list[dict] = []
    best_loss = float("inf")

    print(f"Training Stage II on {device} with {len(train_dataset)} train weeks")
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = run_epoch(
            model,
            decoders,
            train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            desc=f"train {epoch + 1}/{args.epochs}",
        )
        row = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - start}
        if val_loader is not None:
            val_loss = run_epoch(
                model,
                decoders,
                val_loader,
                optimizer=None,
                device=device,
                args=args,
                desc=f"val {epoch + 1}/{args.epochs}",
            )
            row["val_loss"] = val_loss
            score = val_loss
        else:
            score = train_loss
        history.append(row)
        write_json(out_dir / "history.json", history)
        save_checkpoint(
            out_dir / "checkpoint_latest.pt",
            model=model,
            decoders=decoders,
            optimizer=optimizer,
            epoch=epoch,
            metrics=row,
            args=args,
        )
        if score < best_loss:
            best_loss = score
            save_checkpoint(
                out_dir / "checkpoint_best.pt",
                model=model,
                decoders=decoders,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                args=args,
            )
        metric_text = " | ".join(f"{key}={value:.4f}" for key, value in row.items())
        print(metric_text)
        if args.dry_run:
            break


if __name__ == "__main__":
    main()

