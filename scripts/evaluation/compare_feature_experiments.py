#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    embeddings: Path
    metadata: Path
    index_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple feature sets on downstream 5-minute tasks.")
    parser.add_argument(
        "--feature",
        action="append",
        required=True,
        help="Feature spec formatted as name:embeddings.npy:metadata.csv",
    )
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metadata_index_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("metadata is empty")
    if "embedding_index" in rows[0]:
        return "embedding_index"
    if "five_min_index" in rows[0]:
        return "five_min_index"
    raise ValueError("metadata must contain embedding_index or five_min_index")


def parse_feature_spec(value: str) -> FeatureSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--feature must look like name:embeddings.npy:metadata.csv")
    name, embeddings, metadata = parts
    rows = read_csv(metadata)
    return FeatureSpec(name=name, embeddings=Path(embeddings), metadata=Path(metadata), index_column=metadata_index_column(rows))


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["subject_id"], row["split"], row["bin_start_time"])


def load_joined_features(spec: FeatureSpec, labels_by_key: dict[tuple[str, str, str], dict[str, str]]):
    embeddings = np.load(spec.embeddings).astype(np.float32)
    rows = read_csv(spec.metadata)
    examples = []
    for row in rows:
        label = labels_by_key.get(key(row))
        if label is None:
            continue
        if label.get("valid_mask", "1") not in {"1", "true", "True"}:
            continue
        index = int(row[spec.index_column])
        merged = dict(label)
        merged["_embedding_index"] = index
        examples.append(merged)
    if not examples:
        raise ValueError(f"No joined examples for {spec.name}")
    x = np.stack([embeddings[int(row["_embedding_index"])] for row in examples]).astype(np.float32)
    return x, examples


def split_arrays(x: np.ndarray, rows: list[dict[str, str]], target: str):
    y_values = []
    keep = []
    for idx, row in enumerate(rows):
        value = row.get(target, "")
        if value == "":
            continue
        y_values.append(float(value))
        keep.append(idx)
    keep_arr = np.asarray(keep, dtype=np.int64)
    y = np.asarray(y_values, dtype=np.float32)
    x = x[keep_arr]
    rows = [rows[i] for i in keep]
    masks = {split: np.asarray([row["split"] == split for row in rows]) for split in ("train", "val", "test")}
    return x, y, rows, masks


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(train_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, classes: np.ndarray) -> float:
    scores = []
    for cls in classes:
        true = y_true == cls
        pred = y_pred == cls
        tp = float((true & pred).sum())
        fp = float((~true & pred).sum())
        fn = float((true & ~pred).sum())
        if tp + fp + fn == 0:
            scores.append(0.0)
            continue
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        scores.append(2 * precision * recall / max(1e-8, precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def train_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    task_type: str,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> TaskHead:
    model = TaskHead(train_x.shape[1], output_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if task_type == "regression":
        y_mean = float(train_y.mean())
        y_std = float(train_y.std() + 1e-6)
        train_targets = ((train_y - y_mean) / y_std).astype(np.float32)[:, None]
        criterion = nn.MSELoss()
    else:
        y_mean = 0.0
        y_std = 1.0
        train_targets = train_y.astype(np.int64)
        criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_targets))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_score = float("inf")

    for _ in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y if task_type == "classification" else batch_y.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if val_x.shape[0] > 0:
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(val_x.astype(np.float32)).to(device))
                if task_type == "regression":
                    pred_np = pred.squeeze(-1).cpu().numpy() * y_std + y_mean
                    score = float(np.mean(np.abs(pred_np - val_y)))
                else:
                    pred_np = pred.argmax(dim=1).cpu().numpy()
                    score = 1.0 - macro_f1(val_y.astype(int), pred_np, classes=np.unique(val_y.astype(int)))
            if score < best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std  # type: ignore[attr-defined]
    return model


def evaluate(
    model: TaskHead,
    x: np.ndarray,
    y: np.ndarray,
    *,
    task_type: str,
    device: torch.device,
    class_count: int | None = None,
) -> dict[str, float | int | list[int]]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x.astype(np.float32)).to(device))
    if task_type == "regression":
        pred_np = pred.squeeze(-1).cpu().numpy() * float(model.y_std) + float(model.y_mean)  # type: ignore[attr-defined]
        err = pred_np - y
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "n": int(y.shape[0]),
        }
    pred_np = pred.argmax(dim=1).cpu().numpy()
    present_classes = np.unique(y.astype(int))
    result: dict[str, float | int | list[int]] = {
        "macro_f1": macro_f1(y.astype(int), pred_np, classes=present_classes),
        "accuracy": float(np.mean(pred_np == y.astype(int))),
        "n": int(y.shape[0]),
        "classes_present": [int(item) for item in present_classes.tolist()],
    }
    if class_count is not None:
        result["macro_f1_all_classes"] = macro_f1(y.astype(int), pred_np, classes=np.arange(class_count))
    return result


def run_task(
    x_raw: np.ndarray,
    rows: list[dict[str, str]],
    target: str,
    *,
    task_type: str,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    x, y, rows, masks = split_arrays(x_raw, rows, target)
    if not bool(masks["train"].any()) or not bool(masks["test"].any()):
        return {"skipped": True, "reason": "missing train or test examples", "n": int(y.shape[0])}
    if task_type == "classification" and len(np.unique(y[masks["train"]].astype(int))) < 2:
        return {"skipped": True, "reason": "train split has fewer than two classes", "n": int(y.shape[0])}
    x_std = standardize(x[masks["train"]], x)
    model = train_head(
        x_std[masks["train"]],
        y[masks["train"]],
        x_std[masks["val"]],
        y[masks["val"]],
        task_type=task_type,
        output_dim=output_dim,
        args=args,
        device=device,
    )
    result = evaluate(
        model,
        x_std[masks["test"]],
        y[masks["test"]],
        task_type=task_type,
        device=device,
        class_count=output_dim if task_type == "classification" else None,
    )
    result.update(
        {
            "train_n": int(masks["train"].sum()),
            "val_n": int(masks["val"].sum()),
            "test_n": int(masks["test"].sum()),
        }
    )
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    labels = read_csv(args.labels_csv)
    labels_by_key = {key(row): row for row in labels}
    specs = [parse_feature_spec(value) for value in args.feature]
    task_defs = {
        "hr": ("regression", 1),
        "sqi": ("regression", 1),
        "sedentary": ("classification", 2),
        "activity9": ("classification", 9),
    }

    results = {}
    for spec in specs:
        x, rows = load_joined_features(spec, labels_by_key)
        results[spec.name] = {
            "joined_rows": len(rows),
            "feature_dim": int(x.shape[1]),
            "tasks": {},
        }
        for task, (task_type, output_dim) in task_defs.items():
            results[spec.name]["tasks"][task] = run_task(
                x,
                rows,
                task,
                task_type=task_type,
                output_dim=output_dim,
                args=args,
                device=device,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
