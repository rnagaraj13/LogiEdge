#!/usr/bin/env python3
"""PolynomialDecay-scheduled structured-neuron pruning, compaction and INT8 PTQ."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_model import FEATURE_NAMES
from training.tflite_utils import convert_full_int8


class StructuredPolynomialPruner(tf.keras.callbacks.Callback):
    """Progressively mask whole Dense hidden units using TFMOT's polynomial schedule."""

    def __init__(self, total_steps: int, final_sparsity: float = 0.35):
        super().__init__()
        self.schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=final_sparsity,
            begin_step=0, end_step=total_steps, power=3.0,
        )
        self.step = 0
        self.order = {}

    def on_train_begin(self, logs=None):
        for layer_name in ("dense_1", "dense_2"):
            layer = self.model.get_layer(layer_name)
            kernel, bias = layer.get_weights()
            importance = np.linalg.norm(kernel, axis=0)
            self.order[layer_name] = np.argsort(importance)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        # TFMOT PolynomialDecay returns (should_update, sparsity) on current stacks.
        scheduled = self.schedule(self.step)
        sparsity_tensor = scheduled[1] if isinstance(scheduled, tuple) else scheduled
        sparsity = float(sparsity_tensor.numpy())
        for layer_name in ("dense_1", "dense_2"):
            layer = self.model.get_layer(layer_name)
            kernel, bias = layer.get_weights()
            count = min(int(np.floor(kernel.shape[1] * sparsity)), kernel.shape[1] - 1)
            if count:
                kernel[:, self.order[layer_name][:count]] = 0.0
                bias[self.order[layer_name][:count]] = 0.0
                layer.set_weights([kernel, bias])
            self.model.get_layer("dense_2" if layer_name == "dense_1" else "class_probabilities")

    def on_train_end(self, logs=None):
        self.final_step = self.step


def compact_model(model, target_fraction=0.35):
    dense1, dense2, output = (
        model.get_layer("dense_1"), model.get_layer("dense_2"),
        model.get_layer("class_probabilities"),
    )
    w1, b1 = dense1.get_weights()
    w2, b2 = dense2.get_weights()
    w3, b3 = output.get_weights()
    keep1 = np.flatnonzero(np.any(np.abs(w1) > 0, axis=0) | (np.abs(b1) > 0))
    keep2 = np.flatnonzero(np.any(np.abs(w2) > 0, axis=0) | (np.abs(b2) > 0))
    if not len(keep1) or not len(keep2):
        raise RuntimeError("Structured pruning removed every hidden unit.")
    if len(keep1) > int(32 * (1 - target_fraction)) + 1:
        raise RuntimeError("Dense_1 did not reach the 35% structured pruning target.")
    compact = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,), name="features"),
        tf.keras.layers.Dense(len(keep1), activation="relu", name="dense_1"),
        tf.keras.layers.Dense(len(keep2), activation="relu", name="dense_2"),
        tf.keras.layers.Dense(3, activation="softmax", name="class_probabilities"),
    ])
    compact.get_layer("dense_1").set_weights([w1[:, keep1], b1[keep1]])
    compact.get_layer("dense_2").set_weights([w2[np.ix_(keep1, keep2)], b2[keep2]])
    compact.get_layer("class_probabilities").set_weights([w3[keep2, :], b3])
    compact.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    return compact


def main():
    source = ROOT / "training/models/m1_fp32.keras"
    if not source.is_file():
        raise FileNotFoundError("Train the FP32 baseline first.")
    data_path = ROOT / "data_pipeline/generated/windows.csv"
    stats = np.load(ROOT / "data_pipeline/training_stats.npy")
    rows = list(csv.DictReader(data_path.open(encoding="utf-8")))
    raw = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    x = (raw - stats[0]) / np.where(stats[1] < 1e-6, 1.0, stats[1])
    x_train, x_val, y_train, _ = train_test_split(
        x, labels, test_size=0.20, random_state=631, stratify=labels
    )
    model = tf.keras.models.load_model(source)
    batch_size = 16
    epochs = 10
    steps = int(np.ceil(len(x_train) / batch_size)) * epochs
    pruner = StructuredPolynomialPruner(total_steps=steps, final_sparsity=0.35)
    model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        verbose=2, callbacks=[pruner],
    )
    compact = compact_model(model, 0.35)
    compact.fit(x_train, y_train, epochs=8, batch_size=batch_size, verbose=2)
    models_dir = ROOT / "training/models"
    compact_path = models_dir / "m3_pruned_fp32.keras"
    compact.save(compact_path)
    normalized_calibration = x
    output = models_dir / "m3_pruned_int8.tflite"
    convert_full_int8(compact, normalized_calibration, output)
    details = {
        "schedule": "tensorflow_model_optimization PolynomialDecay",
        "structured_unit_sparsity_target": 0.35,
        "final_hidden_units": [
            int(compact.get_layer("dense_1").units),
            int(compact.get_layer("dense_2").units),
        ],
        "pruning_steps": pruner.step,
        "calibration_samples": len(normalized_calibration),
        "validation_samples_reserved": len(x_val),
        "note": "The reduced dense dimensions are physically compacted before PTQ.",
    }
    (ROOT / "training/results/pruning.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
