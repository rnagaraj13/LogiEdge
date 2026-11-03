#!/usr/bin/env python3
"""Measure required latency, p95, model size, accuracy, and CPU-energy metrics."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "temperature_mean", "temperature_std", "temperature_rate_c_per_min",
    "vibration_rms", "vibration_peak", "vibration_kurtosis",
)
VARIANTS = {
    "M1_FP32": ROOT / "training/models/m1_fp32.keras",
    "M2_PTQ_INT8": ROOT / "training/models/m2_int8.tflite",
    "M3_PRUNED_PTQ_INT8": ROOT / "training/models/m3_pruned_int8.tflite",
}


class TFLiteRunner:
    def __init__(self, path):
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def __call__(self, row):
        scale, zero = self.input["quantization"]
        values = row
        if self.input["dtype"] == np.int8:
            if scale <= 0:
                raise ValueError("Invalid input quantization scale.")
            values = np.clip(np.round(row / scale + zero), -128, 127).astype(np.int8)
        else:
            values = row.astype(self.input["dtype"])
        self.interpreter.set_tensor(self.input["index"], values.reshape(1, 6))
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output["index"])[0]
        out_scale, out_zero = self.output["quantization"]
        if self.output["dtype"] == np.int8 and out_scale > 0:
            scores = (scores.astype(np.float32) - out_zero) * out_scale
        return scores.astype(np.float32)


def size_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def load_validation():
    rows = list(csv.DictReader((ROOT / "data_pipeline/generated/windows.csv").open()))
    raw = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    stats = np.load(ROOT / "data_pipeline/training_stats.npy")
    normalized = (raw - stats[0]) / np.where(stats[1] < 1e-6, 1.0, stats[1])
    return train_test_split(
        normalized, labels, test_size=0.20, random_state=631, stratify=labels
    )[1::2]


def main():
    if not (ROOT / "data_pipeline/generated/windows.csv").is_file():
        raise FileNotFoundError("Run data_pipeline/generate_dataset.py first.")
    x_val, y_val = load_validation()
    results_dir = ROOT / "optimisation/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tdp_watts = float(os.getenv("LOGIBRIDGE_LAPTOP_TDP_WATTS", "25"))
    warmups, timed_runs = 10, 200
    process = psutil.Process()
    rows = []
    for name, path in VARIANTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Required model variant missing: {path}")
        if name == "M1_FP32":
            model = tf.keras.models.load_model(path)
            predict = lambda feature: model(feature[None, :], training=False).numpy()[0]
        else:
            predict = TFLiteRunner(path)
        for i in range(warmups):
            predict(x_val[i % len(x_val)])
        latencies = []
        process.cpu_percent(interval=None)
        cpu_before = process.cpu_times()
        wall_before = time.perf_counter()
        timed_features = [x_val[i % len(x_val)] for i in range(timed_runs)]
        for feature in timed_features:
            started = time.perf_counter()
            predict(feature)
            latencies.append((time.perf_counter() - started) * 1000)
        wall_elapsed = time.perf_counter() - wall_before
        cpu_after = process.cpu_times()
        cpu_percent = process.cpu_percent(interval=None)
        cpu_seconds = (
            (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
        )
        total_cpu_fraction = min(
            max(cpu_seconds / wall_elapsed / max(psutil.cpu_count(logical=True) or 1, 1), 0.0),
            1.0,
        )
        mean_latency = float(np.mean(latencies))
        cpu_fraction_from_psutil = min(
            max(cpu_percent / 100 / max(psutil.cpu_count(logical=True) or 1, 1), 0.0), 1.0
        )
        # Use psutil's sampled process CPU% for the TDP-based energy estimate.
        energy_mj = tdp_watts * cpu_fraction_from_psutil * mean_latency
        all_scores = np.asarray([predict(feature) for feature in x_val])
        predicted = all_scores.argmax(axis=1)
        accuracy = float(accuracy_score(y_val, predicted)) * 100
        critical_recall = float(recall_score(
            y_val, predicted, labels=[2], average=None, zero_division=0
        )[0]) * 100
        rows.append({
            "variant": name,
            "mean_latency_ms_200_runs": mean_latency,
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "model_size_kb": size_bytes(path) / 1000,
            "validation_accuracy_percent": accuracy,
            "critical_recall_percent": critical_recall,
            "energy_mj_per_inference_estimated": energy_mj,
            "psutil_process_cpu_percent": cpu_percent,
            "cpu_fraction_for_energy_estimate": cpu_fraction_from_psutil,
            "cpu_fraction_crosscheck": total_cpu_fraction,
            "cpu_tdp_estimate_watts": tdp_watts,
            "warmups_excluded": warmups,
            "timed_runs": timed_runs,
        })
    csv_path = results_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "benchmark_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    plot_pareto(rows, results_dir / "pareto_chart.png")
    print(json.dumps(rows, indent=2))
    print(f"Saved measured benchmark and Pareto plot under {results_dir}.")
    recommended = min(
        (row for row in rows if row["critical_recall_percent"] > 95.0),
        key=lambda row: (row["mean_latency_ms_200_runs"], row["model_size_kb"]),
        default=None,
    )
    if recommended:
        print("Measured feasible candidate (>95% Class 2 recall):", recommended["variant"])
    else:
        print("NO VARIANT PASSED the required >95% Class 2 recall gate; do not recommend deployment.")


def dominates(a, b):
    metrics = (
        ("mean_latency_ms_200_runs", 1),
        ("model_size_kb", 1),
        ("energy_mj_per_inference_estimated", 1),
        ("validation_accuracy_percent", -1),
    )
    no_worse, strictly_better = True, False
    for key, direction in metrics:
        av, bv = a[key] * direction, b[key] * direction
        no_worse &= av <= bv
        strictly_better |= av < bv
    return no_worse and strictly_better


def plot_pareto(rows, path):
    pareto = [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]
    fig, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        axis.scatter(row["model_size_kb"], row["mean_latency_ms_200_runs"], s=70)
        axis.annotate(
            f"{row['variant']}\nAcc {row['validation_accuracy_percent']:.1f}%, "
            f"C2 recall {row['critical_recall_percent']:.1f}%",
            (row["model_size_kb"], row["mean_latency_ms_200_runs"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
        )
    ordered = sorted(pareto, key=lambda row: row["model_size_kb"])
    if len(ordered) > 1:
        axis.plot(
            [row["model_size_kb"] for row in ordered],
            [row["mean_latency_ms_200_runs"] for row in ordered],
            linestyle="--", color="gray", label="Non-dominated variants",
        )
    axis.set_xlabel("Measured model size (KB)")
    axis.set_ylabel("Mean inference latency (ms; 200 timed runs)")
    axis.set_title("LogiEdge validation-set latency/size Pareto view")
    axis.grid(True, alpha=0.3)
    if len(ordered) > 1:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
