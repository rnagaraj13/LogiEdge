#!/usr/bin/env python3
"""Measure required latency, p95, model size, accuracy, and CPU-energy metrics."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Redraw pareto_chart.png from the existing benchmark_results.csv without re-measuring.",
    )
    args = parser.parse_args()
    results_dir = ROOT / "optimisation/results"
    if args.plot_only:
        csv_path = results_dir / "benchmark_results.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"No measured results to plot: {csv_path}")
        rows = [
            {key: (value if key == "variant" else float(value)) for key, value in row.items()}
            for row in csv.DictReader(csv_path.open(encoding="utf-8"))
        ]
        plot_pareto(rows, results_dir / "pareto_chart.png")
        print(f"Redrew {results_dir / 'pareto_chart.png'} from {csv_path}.")
        return
    if not (ROOT / "data_pipeline/generated/windows.csv").is_file():
        raise FileNotFoundError("Run data_pipeline/generate_dataset.py first.")
    x_val, y_val = load_validation()
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
    short = {"M1_FP32": "M1 FP32", "M2_PTQ_INT8": "M2 PTQ INT8", "M3_PRUNED_PTQ_INT8": "M3 Pruned + PTQ INT8"}
    # Tags point in different directions so near-identical points (M2/M3) never share label space.
    tag_offsets = [(-70, -40), (45, 40), (45, -45)]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for index, row in enumerate(rows):
        name = short.get(row["variant"], row["variant"])
        size, latency = row["model_size_kb"], row["mean_latency_ms_200_runs"]
        axis.scatter(
            size, latency, s=80, zorder=3,
            label=(f"{name}: {size:.2f} KB, {latency:.4f} ms, "
                   f"Acc {row['validation_accuracy_percent']:.1f}%, C2 recall {row['critical_recall_percent']:.1f}%"),
        )
        axis.annotate(
            name.split()[0], (size, latency),
            xytext=tag_offsets[index % len(tag_offsets)], textcoords="offset points",
            fontsize=10, fontweight="bold", ha="center",
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1},
        )
    ordered = sorted(pareto, key=lambda row: row["model_size_kb"])
    if len(ordered) > 1:
        axis.plot(
            [row["model_size_kb"] for row in ordered],
            [row["mean_latency_ms_200_runs"] for row in ordered],
            linestyle="--", color="gray", zorder=2, label="Pareto frontier (non-dominated)",
        )
    latencies = [row["mean_latency_ms_200_runs"] for row in rows]
    sizes = [row["model_size_kb"] for row in rows]
    axis.set_yscale("log")
    axis.set_ylim(min(latencies) / 4, max(latencies) * 4)
    axis.set_xlim(0, max(sizes) * 1.25)
    axis.set_xlabel("Measured model size (KB)")
    axis.set_ylabel("Mean inference latency (ms, log scale; 200 timed runs)")
    axis.set_title("LogiEdge latency vs size Pareto view (validation set)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
