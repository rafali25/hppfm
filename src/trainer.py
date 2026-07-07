from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .heads import MorphologyAwarePulsePPG, morphology_ssl_loss
from .helpers import write_json


@dataclass
class MorphologyTrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 64
    alpha: float = 0.6
    temperature: float = 0.1
    use_sqi_loss: bool = True
    grad_clip: float | None = None
    save_every: int = 1


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_morphology_epoch(
    model: MorphologyAwarePulsePPG,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: MorphologyTrainConfig,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)

    totals = {"loss": 0.0, "contrastive": 0.0, "ipa": 0.0, "sqi": 0.0}
    count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = _to_device(batch, device)
            outputs = model(batch["signal"])
            loss_out = morphology_ssl_loss(
                outputs,
                batch["svri_bin"],
                batch["ipa"],
                batch["sqi"],
                alpha=config.alpha,
                temperature=config.temperature,
                use_sqi_loss=config.use_sqi_loss,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss_out.loss.backward()
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            batch_size = int(batch["signal"].shape[0])
            totals["loss"] += float(loss_out.loss.detach().cpu()) * batch_size
            totals["contrastive"] += float(loss_out.contrastive.cpu()) * batch_size
            totals["ipa"] += float(loss_out.ipa.cpu()) * batch_size
            totals["sqi"] += float(loss_out.sqi.cpu()) * batch_size
            count += batch_size

    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    *,
    model: MorphologyAwarePulsePPG,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: MorphologyTrainConfig,
) -> None:
    path = Path(path)
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


def fit_morphology_model(
    model: MorphologyAwarePulsePPG,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: torch.device,
    output_dir: str | Path,
    config: MorphologyTrainConfig,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_config.json", asdict(config))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(config.epochs):
        start = time.time()
        train_metrics = run_morphology_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            desc=f"train {epoch + 1}/{config.epochs}",
        )
        row: dict[str, float] = {f"train_{k}": v for k, v in train_metrics.items()}

        if val_loader is not None:
            val_metrics = run_morphology_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                config=config,
                desc=f"val {epoch + 1}/{config.epochs}",
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            score = val_metrics["loss"]
        else:
            score = train_metrics["loss"]

        row["epoch"] = float(epoch)
        row["seconds"] = time.time() - start
        history.append(row)
        write_json(output_dir / "history.json", history)

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                output_dir / "checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                config=config,
            )
        if score < best_val:
            best_val = score
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                config=config,
            )

        metric_text = " | ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch")
        print(f"epoch={epoch + 1} | {metric_text}")

    return history
