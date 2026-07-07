#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream linear probes on exported embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--level", choices=["week", "subject"], default="subject")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_arrays(embeddings_path: str | Path, metadata_path: str | Path, labels_path: str | Path):
    embeddings = np.load(embeddings_path).astype(np.float64)
    metadata = read_csv(metadata_path)
    labels = {row["subject_id"]: float(row["label"]) for row in read_csv(labels_path)}
    xs, ys, splits = [], [], []
    for row_idx, row in enumerate(metadata):
        subject_id = row["subject_id"]
        if subject_id not in labels:
            continue
        index = int(row.get("embedding_index", row_idx))
        xs.append(embeddings[index])
        ys.append(labels[subject_id])
        splits.append(row["split"])
    if not xs:
        raise ValueError("No embeddings matched labels by subject_id")
    return np.stack(xs), np.asarray(ys), np.asarray(splits)


def standardize(train_x: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return tuple((arr - mean) / std for arr in arrays)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logistic_fit(train_x: np.ndarray, train_y: np.ndarray, *, steps: int = 1000, lr: float = 0.05):
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    y = train_y.astype(np.float64)
    for _ in range(steps):
        pred = sigmoid(x @ weights)
        grad = x.T @ (pred - y) / max(1, x.shape[0])
        weights -= lr * grad
    return weights


def logistic_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_bias = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return sigmoid(x_bias @ weights)


def ridge_fit(train_x: np.ndarray, train_y: np.ndarray, alpha: float = 1.0):
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    eye = np.eye(x.shape[1])
    eye[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + alpha * eye, x.T @ train_y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_bias = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return x_bias @ weights


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = y_true[order].astype(np.float64)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    precision = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / positives)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = scores >= 0.5
    y = y_true.astype(bool)
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    tn = float((~pred & ~y).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "auroc": binary_auc(y_true.astype(int), scores),
        "auprc": average_precision(y_true.astype(int), scores),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(1.0, tp + tn + fp + fn)),
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y_true
    if y_true.size < 2 or np.std(pred) < 1e-8 or np.std(y_true) < 1e-8:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, pred)[0, 1])
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "pearson": pearson,
    }


def main() -> None:
    args = parse_args()
    x, y, splits = build_arrays(args.embeddings, args.metadata, args.labels_csv)
    train_mask = splits == "train"
    eval_mask = splits == "test"
    if not bool(eval_mask.any()):
        eval_mask = splits == "val"
    if not bool(train_mask.any()) or not bool(eval_mask.any()):
        raise ValueError("Need train and val/test rows after joining labels")
    train_x, eval_x = standardize(x[train_mask], x[train_mask], x[eval_mask])
    train_y, eval_y = y[train_mask], y[eval_mask]

    if args.task == "classification":
        weights = logistic_fit(train_x, train_y)
        scores = logistic_predict(eval_x, weights)
        results = classification_metrics(eval_y, scores)
    else:
        weights = ridge_fit(train_x, train_y)
        pred = ridge_predict(eval_x, weights)
        results = regression_metrics(eval_y, pred)

    results.update(
        {
            "train_n": int(train_mask.sum()),
            "eval_n": int(eval_mask.sum()),
            "eval_split": "test" if bool((splits == "test").any()) else "val",
            "task": args.task,
            "level": args.level,
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
