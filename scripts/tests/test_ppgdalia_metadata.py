#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

HELPER_PATH = PROJECT_ROOT / "scripts" / "data" / "prepare_ppgdalia_3k.py"
spec = importlib.util.spec_from_file_location("prepare_ppgdalia_3k", HELPER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not import helper module from {HELPER_PATH}")
prepare_ppgdalia_3k = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prepare_ppgdalia_3k
spec.loader.exec_module(prepare_ppgdalia_3k)

SEGMENT_METADATA_COLUMNS = prepare_ppgdalia_3k.SEGMENT_METADATA_COLUMNS
make_ppgdalia_segment_id = prepare_ppgdalia_3k.make_ppgdalia_segment_id
parse_base_datetime = prepare_ppgdalia_3k.parse_base_datetime
process_subject = prepare_ppgdalia_3k.process_subject
resample_linear = prepare_ppgdalia_3k.resample_linear
source_start_seconds_from_sample = prepare_ppgdalia_3k.source_start_seconds_from_sample
synthetic_start_time = prepare_ppgdalia_3k.synthetic_start_time
write_labels = prepare_ppgdalia_3k.write_labels


def test_timestamp_helpers() -> None:
    source_start_seconds = source_start_seconds_from_sample(5760, source_ppg_fs=64)
    start_time = synthetic_start_time(
        parse_base_datetime("2024-01-01T00:00:00"),
        source_start_seconds,
    )
    segment_id = make_ppgdalia_segment_id(
        subject_id="S1",
        start_time=start_time,
        source_start_seconds=source_start_seconds,
    )

    assert source_start_seconds == 90
    assert start_time.isoformat(timespec="seconds") == "2024-01-01T00:01:30"
    assert segment_id == "S1_2024-01-01T00-01-30_start00000090s"


def test_resampled_30_second_segment_length() -> None:
    source_segment = np.linspace(-1.0, 1.0, num=30 * 64, dtype=np.float32)
    saved_segment = resample_linear(source_segment, fs_original=64, fs_target=50)

    assert saved_segment.dtype == np.float32
    assert saved_segment.shape == (1500,)


def test_metadata_csv_columns_are_exact(tmp_dir: Path) -> None:
    start_time = datetime.fromisoformat("2024-01-01T00:01:30")
    segment_id = make_ppgdalia_segment_id(
        subject_id="S1",
        start_time=start_time,
        source_start_seconds=90,
    )
    out_csv = tmp_dir / "segment_labels.csv"
    write_labels(
        out_csv,
        [
            {
                "segment_id": segment_id,
                "subject_id": "S1",
                "start_time": start_time.isoformat(timespec="seconds"),
                "ppg_path": f"segments_30s_50Hz/train/S1/{segment_id}.npy",
                "split": "train",
                "hr_mean": 46.4,
                "hr_std": 2.1,
                "activity": 1,
                "activity_name": "sitting",
                "activity_binary": "sedentary",
                "source_start_seconds": 90,
            }
        ],
    )

    with out_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == SEGMENT_METADATA_COLUMNS
    assert rows[0]["segment_id"] == segment_id
    assert rows[0]["start_time"] == "2024-01-01T00:01:30"
    assert rows[0]["source_start_seconds"] == "90"
    assert rows[0]["ppg_path"].endswith(f"{segment_id}.npy")


def test_process_subject_uses_full_interval_activity_majority(tmp_dir: Path) -> None:
    subject_dir = tmp_dir / "S1"
    subject_dir.mkdir()
    pickle_path = subject_dir / "S1.pkl"
    payload = {
        "signal": {"wrist": {"BVP": np.ones(30 * 64, dtype=np.float32)}},
        "label": np.full(15, 70.0, dtype=np.float32),
        "activity": np.concatenate(
            [
                np.full(40, 1, dtype=np.int64),
                np.full(80, 2, dtype=np.int64),
            ]
        ),
    }
    with pickle_path.open("wb") as f:
        pickle.dump(payload, f)

    rows = process_subject(
        pickle_path,
        tmp_dir / "segments_30s_50Hz",
        "train",
        1,
        source_ppg_fs=64,
        fs_target=50,
        window_seconds=30,
        stride_seconds=30,
        base_datetime=parse_base_datetime("2024-01-01T00:00:00"),
        include_transient_activity=False,
    )

    assert len(rows) == 1
    assert rows[0]["activity"] == 2
    assert rows[0]["activity_name"] == "stairs"
    saved = np.load(PROJECT_ROOT / rows[0]["ppg_path"])
    assert saved.shape == (1500, 1)


def main() -> None:
    test_timestamp_helpers()
    test_resampled_30_second_segment_length()
    with tempfile.TemporaryDirectory(prefix="ppgdalia_metadata_") as tmp:
        tmp_dir = Path(tmp)
        test_metadata_csv_columns_are_exact(tmp_dir)
        test_process_subject_uses_full_interval_activity_majority(tmp_dir)
    print("PPG-DaLiA metadata tests passed")


if __name__ == "__main__":
    main()
