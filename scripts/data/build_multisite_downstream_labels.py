#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.helpers import skewness_sqi
from src.hierarchical.data.build_5min import floor_to_five_minutes, parse_datetime


ACTIVITY_NAMES = [
    "sitting_rest",
    "walking",
    "running",
    "stairs",
    "driving_transit",
    "eating_drinking",
    "talking_social",
    "work_study_typing",
    "household_exercise_other",
]
ACTIVITY_TO_ID = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}
SEDENTARY_ACTIVITY_IDS = {
    ACTIVITY_TO_ID["sitting_rest"],
    ACTIVITY_TO_ID["driving_transit"],
    ACTIVITY_TO_ID["eating_drinking"],
    ACTIVITY_TO_ID["talking_social"],
    ACTIVITY_TO_ID["work_study_typing"],
}


@dataclass(frozen=True)
class ActivityEvent:
    timestamp: datetime
    action: str
    activity_id: int | None
    raw_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5-minute downstream labels for Multi-site PPG.")
    parser.add_argument("--segment-metadata", required=True)
    parser.add_argument("--five-min-metadata", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--repo-id", default="snowballlab/Multisite-PPG")
    parser.add_argument("--cache-dir", default="datasets/hf_cache")
    parser.add_argument("--fs", type=int, default=50)
    return parser.parse_args()


def parse_log_time(unix_ms: str, local_time: str) -> datetime:
    value = unix_ms.strip().lower()
    if value not in {"", "n/a", "na", "none"}:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    return datetime.strptime(local_time.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("watching talk", "talking")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def action_from_text(text: str) -> str:
    if re.search(r"\b(end|stop|stopped|stoped|done|ended|disconnected|charging break)\b", text):
        if re.search(r"\b(start|started|walking|running|sitting|driving|talking|eating|typing)\b", text):
            return "transition"
        return "end"
    if re.search(r"\b(start|started|walking|running|sitting|driving|talking|eating|typing|stairs)\b", text):
        return "start"
    return "start"


def activity_id_from_text(text: str) -> int | None:
    text = normalize_text(text)
    if "run" in text or "jog" in text:
        return ACTIVITY_TO_ID["running"]
    if "stair" in text:
        return ACTIVITY_TO_ID["stairs"]
    if "walk" in text:
        return ACTIVITY_TO_ID["walking"]
    if any(word in text for word in ["drive", "driving", "bus", "rail", "transit"]):
        return ACTIVITY_TO_ID["driving_transit"]
    if any(word in text for word in ["eat", "drink", "dinner", "smoothie", "tea", "water"]):
        return ACTIVITY_TO_ID["eating_drinking"]
    if any(word in text for word in ["talk", "chat", "call", "partner", "friend"]):
        return ACTIVITY_TO_ID["talking_social"]
    if any(word in text for word in ["study", "typing", "working", "work", "class"]):
        return ACTIVITY_TO_ID["work_study_typing"]
    if any(word in text for word in ["cook", "clean", "tidy", "workout", "weights", "dogs", "playing"]):
        return ACTIVITY_TO_ID["household_exercise_other"]
    if any(word in text for word in ["sit", "sleep", "nap", "watch tv", "watching tv", "movie", "gaming", "lay"]):
        return ACTIVITY_TO_ID["sitting_rest"]
    return None


def download_activity_log(repo_id: str, participant_id: str, cache_dir: str | Path) -> Path | None:
    candidates = [
        f"raw_data/{participant_id}/{participant_id}_activity_log.txt",
        f"raw_data/{participant_id}/{participant_id}_activity_log1.txt",
    ]
    for filename in candidates:
        try:
            return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, cache_dir=cache_dir))
        except Exception:
            continue
    return None


def read_activity_events(path: Path) -> list[ActivityEvent]:
    events: list[ActivityEvent] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            text = ",".join(row[2:]).strip()
            try:
                timestamp = parse_log_time(row[0], row[1])
            except Exception:
                continue
            events.append(
                ActivityEvent(
                    timestamp=timestamp,
                    action=action_from_text(normalize_text(text)),
                    activity_id=activity_id_from_text(text),
                    raw_text=text,
                )
            )
    return sorted(events, key=lambda event: event.timestamp)


def active_activity_at(events: list[ActivityEvent], timestamp: datetime) -> int | None:
    current: int | None = None
    for event in events:
        if event.timestamp > timestamp:
            break
        if event.action == "end":
            current = None
        elif event.activity_id is not None:
            current = event.activity_id
    return current


def read_five_min_rows(path: str | Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["subject_id"], row["split"], parse_datetime(row["bin_start_time"]).isoformat())
            rows[key] = row
    return rows


def main() -> None:
    args = parse_args()
    five_min_rows = read_five_min_rows(args.five_min_metadata)

    with Path(args.segment_metadata).open("r", newline="", encoding="utf-8") as f:
        segment_rows = list(csv.DictReader(f))

    participants = sorted({row["participant_id"] for row in segment_rows})
    activity_events = {}
    for participant in participants:
        log_path = download_activity_log(args.repo_id, participant, args.cache_dir)
        activity_events[participant] = [] if log_path is None else read_activity_events(log_path)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in segment_rows:
        start = parse_datetime(row["start_time"])
        bin_start = floor_to_five_minutes(start).isoformat()
        row = dict(row)
        signal = np.load(row["path"]).astype(np.float32).reshape(-1)
        row["sqi"] = float(skewness_sqi(signal, fs=args.fs))
        activity_id = active_activity_at(activity_events[row["participant_id"]], start)
        row["activity9"] = "" if activity_id is None else int(activity_id)
        grouped[(row["subject_id"], row["split"], bin_start)].append(row)

    out_rows: list[dict[str, str | int | float]] = []
    for key, rows in sorted(grouped.items()):
        subject_id, split, bin_start = key
        five_min_row = five_min_rows.get(key)
        if five_min_row is None:
            continue
        hr_values = np.asarray([float(row["hr"]) for row in rows], dtype=np.float64)
        sqi_values = np.asarray([float(row["sqi"]) for row in rows], dtype=np.float64)
        activity_values = [int(row["activity9"]) for row in rows if row["activity9"] != ""]
        if activity_values:
            counts = Counter(activity_values)
            activity9 = int(counts.most_common(1)[0][0])
            sedentary = int(activity9 in SEDENTARY_ACTIVITY_IDS)
            activity_label = ACTIVITY_NAMES[activity9]
        else:
            activity9 = ""
            sedentary = ""
            activity_label = ""
        out_rows.append(
            {
                "subject_id": subject_id,
                "split": split,
                "bin_start_time": bin_start,
                "five_min_index": int(five_min_row["five_min_index"]),
                "valid_mask": int(five_min_row["valid_mask"]),
                "segment_count": len(rows),
                "hr": float(np.mean(hr_values)),
                "sqi": float(np.mean(sqi_values)),
                "sedentary": sedentary,
                "activity9": activity9,
                "activity_label": activity_label,
                "activity_label_count": len(activity_values),
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "split",
                "bin_start_time",
                "five_min_index",
                "valid_mask",
                "segment_count",
                "hr",
                "sqi",
                "sedentary",
                "activity9",
                "activity_label",
                "activity_label_count",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    activity_counts = Counter(row["activity_label"] for row in out_rows if row["activity_label"] != "")
    print(f"Wrote {len(out_rows)} downstream label rows to {out_path}")
    print(f"Activity-labeled rows: {sum(activity_counts.values())}")
    print(dict(activity_counts))


if __name__ == "__main__":
    main()
