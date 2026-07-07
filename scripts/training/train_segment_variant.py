#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import PulsePPGMorphologyDataset, load_morphology_index
from src.encoder import PulsePPGResNet1D
from src.heads import MorphologyAwarePulsePPG, morphology_ssl_loss
from src.helpers import get_device, set_seed, write_json


@dataclass
class SegmentVariantConfig:
    variant: str
    epochs: int
    lr: float
    batch_size: int
    alpha: float
    morphology_temperature: float
    pulse_temperature: float
    pulse_weight: float
    papagei_weight: float
    use_sqi_loss: bool
    grad_clip: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train segment-level PulsePPG/PaPaGei representation variants."
    )
    parser.add_argument(
        "--variant",
        choices=["pulseppg_only", "papagei_only", "pulseppg_papagei"],
        required=True,
    )
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fs", type=int, default=50)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--morphology-temperature", type=float, default=0.1)
    parser.add_argument("--pulse-temperature", type=float, default=0.2)
    parser.add_argument("--pulse-weight", type=float, default=1.0)
    parser.add_argument("--papagei-weight", type=float, default=1.0)
    parser.add_argument("--no-sqi-loss", action="store_true")
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def augment_signal(signal: torch.Tensor, *, noise_std: float = 0.03, max_shift: int = 25) -> torch.Tensor:
    x = signal
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = x * scale
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
        x = torch.stack([torch.roll(item, int(shift.item()), dims=-1) for item, shift in zip(x, shifts)], dim=0)
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


def unsupervised_nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    batch_size = z1.shape[0]
    if batch_size <= 1:
        return (z1.sum() + z2.sum()) * 0.0
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )
    exp_logits = torch.exp(logits) * (~eye).float()
    numerator = torch.exp(logits[torch.arange(2 * batch_size, device=z.device), positives])
    denominator = exp_logits.sum(dim=1).clamp_min(1e-12)
    return -torch.log(numerator / denominator).mean()


def variant_weights(args: argparse.Namespace) -> tuple[float, float]:
    if args.variant == "pulseppg_only":
        return args.pulse_weight, 0.0
    if args.variant == "papagei_only":
        return 0.0, args.papagei_weight
    return args.pulse_weight, args.papagei_weight


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(
    model: MorphologyAwarePulsePPG,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: SegmentVariantConfig,
    desc: str,
    dry_run: bool,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)
    totals = {
        "loss": 0.0,
        "pulse": 0.0,
        "papagei": 0.0,
        "morph_contrastive": 0.0,
        "ipa": 0.0,
        "sqi": 0.0,
    }
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            batch = to_device(batch, device)
            loss = batch["signal"].sum() * 0.0
            pulse_loss = loss.detach()
            papagei_loss = loss.detach()
            morph_contrastive = loss.detach()
            ipa = loss.detach()
            sqi = loss.detach()

            if config.pulse_weight > 0:
                view1 = augment_signal(batch["signal"])
                view2 = augment_signal(batch["signal"])
                out1 = model(view1)
                out2 = model(view2)
                pulse_loss = unsupervised_nt_xent_loss(
                    out1["embedding"],
                    out2["embedding"],
                    temperature=config.pulse_temperature,
                )
                loss = loss + config.pulse_weight * pulse_loss

            if config.papagei_weight > 0:
                outputs = model(batch["signal"])
                morph = morphology_ssl_loss(
                    outputs,
                    batch["svri_bin"],
                    batch["ipa"],
                    batch["sqi"],
                    alpha=config.alpha,
                    temperature=config.morphology_temperature,
                    use_sqi_loss=config.use_sqi_loss,
                )
                papagei_loss = morph.loss
                morph_contrastive = morph.contrastive
                ipa = morph.ipa
                sqi = morph.sqi
                loss = loss + config.papagei_weight * morph.loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            batch_size = int(batch["signal"].shape[0])
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["pulse"] += float(pulse_loss.detach().cpu()) * batch_size
            totals["papagei"] += float(papagei_loss.detach().cpu()) * batch_size
            totals["morph_contrastive"] += float(morph_contrastive.detach().cpu()) * batch_size
            totals["ipa"] += float(ipa.detach().cpu()) * batch_size
            totals["sqi"] += float(sqi.detach().cpu()) * batch_size
            count += batch_size
            if dry_run and batch_idx >= 1:
                break

    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    model: MorphologyAwarePulsePPG,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: SegmentVariantConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(config),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))

    pulse_weight, papagei_weight = variant_weights(args)
    config = SegmentVariantConfig(
        variant=args.variant,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        morphology_temperature=args.morphology_temperature,
        pulse_temperature=args.pulse_temperature,
        pulse_weight=pulse_weight,
        papagei_weight=papagei_weight,
        use_sqi_loss=not args.no_sqi_loss,
        grad_clip=args.grad_clip,
    )
    write_json(output_dir / "train_config.json", asdict(config))

    segment_seconds = None if args.segment_seconds <= 0 else args.segment_seconds
    records = load_morphology_index(args.index_csv)
    train_dataset = PulsePPGMorphologyDataset(
        records,
        split="train",
        fs=args.fs,
        segment_seconds=segment_seconds,
        channel=args.channel,
    )
    try:
        val_dataset = PulsePPGMorphologyDataset(
            records,
            split="val",
            fs=args.fs,
            segment_seconds=segment_seconds,
            channel=args.channel,
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
    val_loader = None if val_dataset is None else DataLoader(
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
    device = get_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    best_score = float("inf")

    print(f"Training {args.variant} on {device} with {len(train_dataset)} train records")
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            desc=f"{args.variant} train {epoch + 1}/{args.epochs}",
            dry_run=args.dry_run,
        )
        row = {f"train_{key}": value for key, value in train_metrics.items()}
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                config=config,
                desc=f"{args.variant} val {epoch + 1}/{args.epochs}",
                dry_run=args.dry_run,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = val_metrics["loss"]
        else:
            score = train_metrics["loss"]
        row["epoch"] = float(epoch)
        row["seconds"] = time.time() - start
        history.append(row)
        write_json(output_dir / "history.json", history)
        save_checkpoint(
            output_dir / "checkpoint_latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=row,
            config=config,
        )
        if score < best_score:
            best_score = score
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                config=config,
            )
        metric_text = " | ".join(f"{key}={value:.4f}" for key, value in row.items() if key != "epoch")
        print(f"epoch={epoch + 1} | {metric_text}")
        if args.dry_run:
            break


if __name__ == "__main__":
    main()
