#!/usr/bin/env python3
"""Build confidence reference bins from 300 clean, simulated Normal windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.preprocessing import sliding_windows
from data_pipeline.simulator import generate_stream
from monitoring.psi import BIN_EDGES, proportions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="training/models/m2_int8.tflite")
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if args.count < 300:
        raise ValueError("The assignment requires at least 300 Normal reference windows.")
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    stats = np.load(ROOT / "data_pipeline/training_stats.npy")
    if stats.shape != (2, 6):
        raise ValueError("Expected frozen training_stats.npy with shape (2, 6).")

    duration_minutes = int(np.ceil((30 + 10 * (args.count - 1)) / 60)) + 1
    samples = [
        {**payload, "topic": topic}
        for topic, payload in generate_stream(duration_minutes, "none", seed=179)
        if topic.endswith(("/temperature", "/vibration_rms"))
    ]
    windows = sliding_windows(samples)[:args.count]
    if len(windows) < args.count:
        raise RuntimeError(f"Only {len(windows)} clean windows generated; requested {args.count}.")
    x = np.asarray([row["features"] for row in windows], dtype=np.float32)
    x = (x - stats[0]) / np.where(stats[1] < 1e-6, 1.0, stats[1])

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    confidences = []
    for row in x:
        values = row
        scale, zero = input_info["quantization"]
        if input_info["dtype"] == np.int8:
            if scale <= 0:
                raise ValueError("Model has invalid input quantization.")
            values = np.clip(np.round(row / scale + zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_info["index"], values.reshape(1, 6))
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0]
        out_scale, out_zero = output_info["quantization"]
        if output_info["dtype"] == np.int8 and out_scale > 0:
            scores = (scores.astype(np.float32) - out_zero) * out_scale
        confidences.append(float(np.max(scores)))

    result = {
        "status": "measured",
        "model": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "sample_count": len(confidences),
        "bin_edges": list(BIN_EDGES),
        "reference_proportions": proportions(confidences),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "simulator": "normal, seed 179; independent long clean stream",
    }
    out = ROOT / "monitoring/reference_dist.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
