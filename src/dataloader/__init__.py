from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from ..helpers import (
    compute_morphology,
    crop_or_pad_1d,
    digitize,
    ensure_time_channel,
    filter_morphology_values,
    iter_npy_files,
    make_bin_edges,
    zscore,
)

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


SPLITS = ("train", "val", "test")


@dataclass
class MorphologyRecord:
    path: str
    split: str
    svri: float
    sqi: float
    ipa: float
    svri_bin: int


def infer_split(path: str | Path) -> str:
    parts = Path(path).parts
    for split in SPLITS:
        if split in parts:
            return split
    return "train"


def select_ppg_channel(signal: np.ndarray, channel: int = 0) -> np.ndarray:
    arr = ensure_time_channel(signal)
    if channel < 0 or channel >= arr.shape[1]:
        raise IndexError(f"Channel {channel} is out of bounds for signal shape {arr.shape}")
    return arr[:, channel]


def load_ppg_npy(
    path: str | Path,
    *,
    channel: int = 0,
    segment_samples: int | None = None,
    random_crop: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    signal = np.load(path).astype(np.float32)
    ppg = select_ppg_channel(signal, channel=channel)
    ppg = crop_or_pad_1d(ppg, segment_samples, random_crop=random_crop)
    if normalize:
        ppg = zscore(ppg)
    return ppg.astype(np.float32)


def scan_split_files(data_root: str | Path, splits: Iterable[str] = SPLITS) -> list[Path]:
    root = Path(data_root)
    files: list[Path] = []
    for split in splits:
        split_root = root / split
        if split_root.exists():
            files.extend(iter_npy_files(split_root))
    return sorted(files)


def build_morphology_records(
    data_root: str | Path,
    *,
    fs: int,
    segment_seconds: float | None = 10.0,
    channel: int = 0,
    num_svri_bins: int = 8,
    splits: Iterable[str] = SPLITS,
    keep_unfiltered: bool = False,
) -> tuple[list[MorphologyRecord], np.ndarray]:
    segment_samples = None if segment_seconds is None else int(round(segment_seconds * fs))
    files = scan_split_files(data_root, splits=splits)
    if not files:
        raise FileNotFoundError(f"No .npy files found under {data_root}")

    pending: list[dict] = []
    for path in tqdm(files, desc="Computing morphology"):
        ppg = load_ppg_npy(
            path,
            channel=channel,
            segment_samples=segment_samples,
            random_crop=False,
            normalize=True,
        )
        values = compute_morphology(ppg, fs=fs)
        if keep_unfiltered or filter_morphology_values(values):
            pending.append({"path": str(path), "split": infer_split(path), **values})

    if not pending:
        raise ValueError("All morphology rows were filtered out")

    train_svri = [row["svri"] for row in pending if row["split"] == "train"]
    if not train_svri:
        train_svri = [row["svri"] for row in pending]
    bin_edges = make_bin_edges(train_svri, num_svri_bins)
    svri_bins = digitize([row["svri"] for row in pending], bin_edges)

    records = [
        MorphologyRecord(
            path=row["path"],
            split=row["split"],
            svri=float(row["svri"]),
            sqi=float(row["sqi"]),
            ipa=float(row["ipa"]),
            svri_bin=int(svri_bin),
        )
        for row, svri_bin in zip(pending, svri_bins)
    ]
    return records, bin_edges


def write_morphology_index(
    records: list[MorphologyRecord],
    output_csv: str | Path,
    *,
    bin_edges: np.ndarray | None = None,
) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "split", "svri", "sqi", "ipa", "svri_bin"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": record.path,
                    "split": record.split,
                    "svri": record.svri,
                    "sqi": record.sqi,
                    "ipa": record.ipa,
                    "svri_bin": record.svri_bin,
                }
            )

    if bin_edges is not None:
        np.save(output_csv.with_suffix(".svri_bin_edges.npy"), np.asarray(bin_edges))


def load_morphology_index(index_csv: str | Path) -> list[MorphologyRecord]:
    records: list[MorphologyRecord] = []
    with Path(index_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                MorphologyRecord(
                    path=row["path"],
                    split=row.get("split") or infer_split(row["path"]),
                    svri=float(row["svri"]),
                    sqi=float(row["sqi"]),
                    ipa=float(row["ipa"]),
                    svri_bin=int(row["svri_bin"]),
                )
            )
    return records


class PulsePPGMorphologyDataset(Dataset):
    """Pulse-PPG folder data with PaPaGei-S morphology labels."""

    def __init__(
        self,
        records: list[MorphologyRecord],
        *,
        split: str,
        fs: int,
        segment_seconds: float | None = 10.0,
        channel: int = 0,
        random_crop: bool = False,
        noise_std: float = 0.0,
        normalize: bool = True,
    ):
        if torch is None:
            raise ImportError("PulsePPGMorphologyDataset requires torch. Install torch to train models.")
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"No records found for split '{split}'")
        self.split = split
        self.fs = fs
        self.segment_samples = None if segment_seconds is None else int(round(segment_seconds * fs))
        self.channel = channel
        self.random_crop = random_crop
        self.noise_std = noise_std
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        ppg = load_ppg_npy(
            record.path,
            channel=self.channel,
            segment_samples=self.segment_samples,
            random_crop=self.random_crop,
            normalize=self.normalize,
        )
        if self.noise_std > 0:
            ppg = ppg + np.random.normal(0.0, self.noise_std, size=ppg.shape).astype(np.float32)

        signal = torch.from_numpy(ppg[None, :].astype(np.float32))
        return {
            "signal": signal,
            "svri_bin": torch.tensor(record.svri_bin, dtype=torch.long),
            "sqi": torch.tensor(record.sqi, dtype=torch.float32),
            "ipa": torch.tensor(record.ipa, dtype=torch.float32),
            "path": record.path,
        }
