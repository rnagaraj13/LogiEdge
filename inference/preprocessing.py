"""Six-feature preprocessing used by both training and the inference container."""

from __future__ import annotations

import math
from collections import deque

FEATURE_NAMES = (
    "temperature_mean",
    "temperature_std",
    "temperature_rate_c_per_min",
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis",
)


def moving_average(values, width=5):
    if width < 1:
        raise ValueError("Moving-average width must be positive.")
    queue, total, result = deque(), 0.0, []
    for raw in values:
        value = float(raw)
        queue.append(value)
        total += value
        if len(queue) > width:
            total -= queue.popleft()
        result.append(total / len(queue))
    return result


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def extract_feature_vector(temperature_values, vibration_values, temperature_hz=1.0):
    if not temperature_values or not vibration_values:
        raise ValueError("A feature window needs temperature and vibration samples.")
    temp = moving_average(temperature_values)
    vibration = moving_average(vibration_values)
    temp_mean = sum(temp) / len(temp)
    temp_std = _std(temp)
    duration_minutes = (len(temp) - 1) / temperature_hz / 60.0
    temp_rate = (temp[-1] - temp[0]) / duration_minutes if duration_minutes > 0 else 0.0
    vibration_rms = math.sqrt(sum(v * v for v in vibration) / len(vibration))
    vibration_peak = max(abs(v) for v in vibration)
    vib_mean = sum(vibration) / len(vibration)
    variance = sum((v - vib_mean) ** 2 for v in vibration) / len(vibration)
    kurtosis = (
        sum((v - vib_mean) ** 4 for v in vibration) / len(vibration) / variance**2
        if variance > 1e-12 else 0.0
    )
    return [temp_mean, temp_std, temp_rate, vibration_rms, vibration_peak, kurtosis]


def normalize(features, stats):
    import numpy as np

    values = np.asarray(features, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if values.shape[-1] != 6 or mean.shape != (6,) or std.shape != (6,):
        raise ValueError("Expected six feature values and six-element mean/std arrays.")
    return (values - mean) / np.where(std < 1e-6, 1.0, std)


def shifted_stats(stats, sigma=3.0):
    return {
        "mean": [m + sigma * s for m, s in zip(stats["mean"], stats["std"])],
        "std": list(stats["std"]),
    }
