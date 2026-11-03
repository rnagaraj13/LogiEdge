#!/usr/bin/env python3
"""Stage a trained TFLite model and frozen stats into the Docker build context."""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "m2": ROOT / "training/models/m2_int8.tflite",
    "m3": ROOT / "training/models/m3_pruned_int8.tflite",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="m2")
    args = parser.parse_args()
    model = VARIANTS[args.variant]
    stats = ROOT / "data_pipeline/training_stats.npy"
    if not model.is_file() or not stats.is_file():
        raise FileNotFoundError("Required model or training stats are missing; run training first.")
    shutil.copy2(model, ROOT / "inference/model.tflite")
    shutil.copy2(stats, ROOT / "inference/training_stats.npy")
    print(f"Staged {model.name} and frozen training stats for Docker.")


if __name__ == "__main__":
    main()
