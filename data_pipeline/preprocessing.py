"""Compatibility exports for the shared inference/training preprocessor."""

from inference.preprocessing import (
    FEATURE_NAMES,
    extract_feature_vector,
    moving_average,
    normalize,
    shifted_stats,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_feature_vector",
    "moving_average",
    "normalize",
    "shifted_stats",
]


def sliding_windows(samples, window_seconds=30, step_seconds=10):
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Window and step lengths must be positive.")
    rows = sorted(samples, key=lambda row: float(row["timestamp"]))
    if not rows:
        return []
    start_time = float(rows[0]["timestamp"])
    end_time = float(rows[-1]["timestamp"])
    window_end = start_time + window_seconds
    result = []
    while window_end <= end_time + 1e-9:
        window_start = window_end - window_seconds
        current = [
            row for row in rows
            if window_start < float(row["timestamp"]) <= window_end
        ]
        temps = [float(row["value"]) for row in current if "temperature" in str(row.get("topic", ""))]
        vibrations = [
            float(row["value"]) for row in current
            if "vibration_rms" in str(row.get("topic", ""))
        ]
        # Dataset generation stores the stream name before calling this helper.
        if not temps:
            temps = [float(row["temperature"]) for row in current if "temperature" in row]
        if not vibrations:
            vibrations = [float(row["vibration_rms"]) for row in current if "vibration_rms" in row]
        if temps and vibrations:
            result.append({
                "timestamp": window_end,
                "features": extract_feature_vector(temps, vibrations),
                "temperature_end": temps[-1],
                "vibration_rms_window": sum(v * v for v in vibrations) ** 0.5 / len(vibrations) ** 0.5,
            })
        window_end += step_seconds
    return result
