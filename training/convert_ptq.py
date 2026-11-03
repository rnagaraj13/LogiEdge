#!/usr/bin/env python3
"""Convert the FP32 baseline to full-integer INT8 using representative samples."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_model import FEATURE_NAMES
from training.tflite_utils import convert_full_int8


def main():
    model_path = ROOT / "training/models/m1_fp32.keras"
    stats_path = ROOT / "data_pipeline/training_stats.npy"
    data_path = ROOT / "data_pipeline/generated/windows.csv"
    out_dir = ROOT / "training/models"
    if not model_path.is_file():
        raise FileNotFoundError("Train the FP32 baseline first.")
    rows = list(csv.DictReader(data_path.open(encoding="utf-8")))
    raw = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    stats = np.load(stats_path)
    normalized = (raw - stats[0]) / np.where(stats[1] < 1e-6, 1.0, stats[1])
    if len(normalized) < 200:
        raise ValueError(f"Need >=200 calibration examples; dataset contains {len(normalized)}.")
    model = tf.keras.models.load_model(model_path)
    output = out_dir / "m2_int8.tflite"
    convert_full_int8(model, normalized[:max(200, len(normalized))], output)
    print(f"Created full INT8 model with {len(normalized)} representative examples: {output}")


if __name__ == "__main__":
    main()
