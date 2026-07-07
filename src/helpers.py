from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if torch is None:
        raise ImportError("get_device requires torch. Install torch to train models.")
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_time_channel(signal: np.ndarray) -> np.ndarray:
    """Return a signal as (time, channels)."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D signal, got shape {arr.shape}")
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        return arr.T.copy()
    return arr


def zscore(signal: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    return (arr - np.nanmean(arr)) / (np.nanstd(arr) + eps)


def crop_or_pad_1d(
    signal: np.ndarray,
    target_length: int | None,
    *,
    random_crop: bool = False,
) -> np.ndarray:
    if target_length is None or target_length <= 0:
        return np.asarray(signal, dtype=np.float32)

    arr = np.asarray(signal, dtype=np.float32)
    length = arr.shape[0]
    if length == target_length:
        return arr
    if length > target_length:
        if random_crop:
            start = np.random.randint(0, length - target_length + 1)
        else:
            start = (length - target_length) // 2
        return arr[start : start + target_length]

    pad_total = target_length - length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(arr, (pad_left, pad_right), mode="constant")


def extract_svri(single_waveform: np.ndarray) -> float:
    """Stress-induced vascular response index from PaPaGei."""
    x = np.asarray(single_waveform, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    scale = x_max - x_min
    if scale <= EPS:
        return 0.0

    x = (x - x_min) / scale
    max_index = int(np.argmax(x))
    if max_index <= 0 or max_index >= x.size:
        return 0.0

    pre = float(np.mean(x[:max_index]))
    post = float(np.mean(x[max_index:]))
    if abs(pre) <= EPS:
        return 0.0
    return post / pre


def skewness(x: np.ndarray, axis: int | None = None, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    centered = arr - mean
    m2 = np.nanmean(centered**2, axis=axis)
    m3 = np.nanmean(centered**3, axis=axis)
    return m3 / np.power(m2 + eps, 1.5)


def skewness_sqi(signal: np.ndarray, fs: int, window_seconds: int = 5) -> float:
    """PaPaGei SQI: mean skewness over 5-second windows by default."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0

    win = max(1, int(fs * window_seconds))
    usable = (x.size // win) * win
    if usable < win:
        return float(skewness(x))

    windows = x[:usable].reshape(-1, win)
    return float(np.nanmean(skewness(windows, axis=1)))


def _local_extrema(signal: np.ndarray, order: int, mode: str) -> np.ndarray:
    try:
        from scipy.signal import argrelmax, argrelmin

        if mode == "max":
            return argrelmax(signal, order=order)[0]
        return argrelmin(signal, order=order)[0]
    except Exception:
        x = np.asarray(signal)
        indices: list[int] = []
        for idx in range(order, len(x) - order):
            window = x[idx - order : idx + order + 1]
            if mode == "max" and x[idx] == np.max(window):
                indices.append(idx)
            elif mode == "min" and x[idx] == np.min(window):
                indices.append(idx)
        return np.asarray(indices, dtype=int)


def compute_ipa(signal: np.ndarray, fs: int) -> float:
    """Inflection point area ratio following the PaPaGei implementation."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < max(8, fs // 2):
        return 0.0

    order = max(1, int(fs) // 5)
    minima_index = _local_extrema(x, order=order, mode="min")
    if minima_index.size < 2:
        return 0.0

    start, end = int(minima_index[0]), int(minima_index[1])
    if end <= start + 2:
        return 0.0

    single_beat = x[start:end]
    beat_minima = _local_extrema(single_beat, order=1, mode="min")
    beat_minima = beat_minima[(beat_minima > 0) & (beat_minima < single_beat.size - 1)]
    if beat_minima.size == 0:
        return 0.0

    notch = int(beat_minima[0])
    sys_values = single_beat[:notch]
    dias_values = single_beat[notch:]
    if sys_values.size < 2 or dias_values.size < 2:
        return 0.0

    trapz = getattr(np, "trapezoid", None)
    if trapz is None:
        trapz = _trapz
    sys_area = float(trapz(sys_values, dx=1.0))
    dias_area = float(trapz(dias_values, dx=1.0))
    if abs(dias_area) <= EPS:
        return 0.0
    return sys_area / dias_area


def _trapz(values: np.ndarray, dx: float = 1.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(np.sum((arr[:-1] + arr[1:]) * 0.5 * dx))


def compute_morphology(signal: np.ndarray, fs: int) -> dict[str, float]:
    x = zscore(np.asarray(signal, dtype=np.float32).reshape(-1))
    return {
        "svri": safe_float(extract_svri(x)),
        "sqi": safe_float(skewness_sqi(x, fs=fs)),
        "ipa": safe_float(compute_ipa(x, fs=fs)),
    }


def safe_float(value: float, default: float = 0.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return default
    return value


def filter_morphology_values(
    values: dict[str, float],
    *,
    svri_range: tuple[float, float] = (0.0, 2.0),
    ipa_range: tuple[float, float] = (-10.0, 10.0),
    sqi_range: tuple[float, float] = (-3.0, 3.0),
) -> bool:
    svri = values["svri"]
    ipa = values["ipa"]
    sqi = values["sqi"]
    return (
        svri_range[0] < svri < svri_range[1]
        and ipa_range[0] < ipa < ipa_range[1]
        and sqi_range[0] < sqi < sqi_range[1]
    )


def make_bin_edges(values: Sequence[float], num_bins: int) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size == 0:
        raise ValueError("Cannot create bins from an empty value list")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")

    vmin = float(np.min(values_arr))
    vmax = float(np.max(values_arr))
    if abs(vmax - vmin) <= EPS:
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, num_bins + 1, dtype=np.float64)[1:-1]


def digitize(values: Sequence[float] | np.ndarray, bin_edges: Sequence[float]) -> np.ndarray:
    labels = np.digitize(np.asarray(values, dtype=np.float64), np.asarray(bin_edges), right=False)
    return labels.astype(np.int64)


def iter_npy_files(root: str | Path) -> Iterable[Path]:
    yield from sorted(Path(root).rglob("*.npy"))


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
