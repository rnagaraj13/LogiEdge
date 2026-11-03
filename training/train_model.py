#!/usr/bin/env python3
"""Train the specified 32/16 ReLU MLP and record real validation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "temperature_mean", "temperature_std", "temperature_rate_c_per_min",
    "vibration_rms", "vibration_peak", "vibration_kurtosis",
)
DATA_PATH = ROOT / "data_pipeline/generated/windows.csv"
STATS_PATH = ROOT / "data_pipeline/training_stats.npy"
MODEL_DIR = ROOT / "training/models"
RESULTS_DIR = ROOT / "training/results"


def load_data():
    if not DATA_PATH.is_file() or not STATS_PATH.is_file():
        raise FileNotFoundError("Run `python data_pipeline/generate_dataset.py` before training.")
    rows = list(csv.DictReader(DATA_PATH.open(encoding="utf-8")))
    x = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    stats = np.load(STATS_PATH)
    if stats.shape != (2, 6):
        raise ValueError(f"Expected training_stats.npy shape (2, 6), got {stats.shape}.")
    x = (x - stats[0]) / np.where(stats[1] < 1e-6, 1.0, stats[1])
    return x, y, stats


def main():
    np.random.seed(631)
    tf.keras.utils.set_random_seed(631)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x, y, stats = load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.20, random_state=631, stratify=y
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,), name="features"),
        tf.keras.layers.Dense(32, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(16, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(3, activation="softmax", name="class_probabilities"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=100, batch_size=16, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ],
    )
    probabilities = model.predict(x_val, verbose=0)
    predicted = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(y_val, predicted))
    critical_recall = float(recall_score(y_val, predicted, labels=[2], average=None)[0])
    metrics = {
        "validation_accuracy": accuracy,
        "validation_critical_recall": critical_recall,
        "validation_confusion_matrix_labels_0_1_2": confusion_matrix(
            y_val, predicted, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": classification_report(
            y_val, predicted, labels=[0, 1, 2], target_names=["Normal", "Warning", "Critical"],
            output_dict=True, zero_division=0,
        ),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "epochs_run": len(history.history["loss"]),
        "normalization_stats": stats.tolist(),
        "seed": 631,
    }
    model.save(MODEL_DIR / "m1_fp32.keras")
    (RESULTS_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if accuracy < 0.88:
        raise SystemExit(
            f"FAIL: held-out accuracy {accuracy:.3%} is below the required 88%; revise and retrain."
        )
    print(f"PASS: held-out accuracy {accuracy:.3%} meets the 88% gate.")
    shifted_stats = stats.copy()
    shifted_stats[0] += 3 * shifted_stats[1]
    raw_x = np.asarray([
        [float(row[name]) for name in FEATURE_NAMES]
        for row in csv.DictReader(DATA_PATH.open(encoding="utf-8"))
    ], dtype=np.float32)
    raw_y = np.asarray([
        int(row["label"]) for row in csv.DictReader(DATA_PATH.open(encoding="utf-8"))
    ], dtype=np.int64)
    _, raw_validation, _, y_validation = train_test_split(
        raw_x, raw_y, test_size=0.20, random_state=631, stratify=raw_y
    )
    shifted_x = (raw_validation - shifted_stats[0]) / np.where(
        shifted_stats[1] < 1e-6, 1.0, shifted_stats[1]
    )
    shifted_accuracy = float(accuracy_score(y_validation, model.predict(shifted_x, verbose=0).argmax(axis=1)))
    experiment = {
        "correct_stats_accuracy": accuracy,
        "mean_shifted_by_3_sigma_accuracy": shifted_accuracy,
        "accuracy_change_percentage_points": (shifted_accuracy - accuracy) * 100,
        "interpretation": "Measured on the same held-out split; not a live-data normalization update.",
    }
    (RESULTS_DIR / "normalization_shift_experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n"
    )
    print("Three-sigma normalization experiment:", json.dumps(experiment))


if __name__ == "__main__":
    main()
