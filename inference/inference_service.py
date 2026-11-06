#!/usr/bin/env python3
"""MQTT-to-TFLite edge inference with local logging and offline alert persistence."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
from preprocessing import extract_feature_vector

TRUCK_ID = os.getenv("TRUCK_ID", "truck-001")
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model.tflite"))
STATS_PATH = Path(os.getenv("STATS_PATH", "/app/training_stats.npy"))
LOG_PATH = Path(os.getenv("ALERT_LOG_PATH", "/var/lib/logibridge/alerts.jsonl"))
CLASS_NAMES = ("Normal", "Warning", "Critical")
TOPIC_ROOT = f"logibridge/trucks/{TRUCK_ID}"
WINDOW_SECONDS = 30
STEP_SECONDS = 10
lock = threading.Lock()
temperatures: list[tuple[float, float]] = []
vibrations: list[tuple[float, float]] = []
next_window_end: float | None = None
interpreter = None
input_detail = None
output_detail = None
stats = None
client = None


def load_runtime():
    global interpreter, input_detail, output_detail, stats
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeError("tflite-runtime is not installed in the inference image.") from exc
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")
    if not STATS_PATH.is_file():
        raise FileNotFoundError(f"Frozen normalization stats not found: {STATS_PATH}")
    stats_array = np.load(STATS_PATH)
    if stats_array.shape != (2, 6):
        raise ValueError(f"Expected normalization array shape (2, 6), got {stats_array.shape}.")
    stats = {"mean": stats_array[0], "std": stats_array[1]}
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"])[-1] != 6:
        raise ValueError(f"Model must accept six features; shape={input_detail['shape']}.")


def predict(feature_vector):
    normalized = (np.asarray(feature_vector, dtype=np.float32) - stats["mean"]) / np.where(
        stats["std"] < 1e-6, 1.0, stats["std"]
    )
    scale, zero = input_detail["quantization"]
    if input_detail["dtype"] == np.int8:
        if scale <= 0:
            raise ValueError("Invalid input quantization scale.")
        values = np.clip(np.round(normalized / scale + zero), -128, 127).astype(np.int8)
    else:
        values = normalized.astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], values.reshape(1, 6))
    started = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - started) * 1000
    probabilities = interpreter.get_tensor(output_detail["index"])[0]
    out_scale, out_zero = output_detail["quantization"]
    if output_detail["dtype"] == np.int8 and out_scale > 0:
        probabilities = (probabilities.astype(np.float32) - out_zero) * out_scale
    probabilities = probabilities.astype(np.float64)
    probabilities = np.clip(probabilities, 0, None)
    probabilities /= probabilities.sum() if probabilities.sum() > 0 else 1.0
    return probabilities, latency_ms


def publish_and_log(topic, payload, qos=1):
    encoded = json.dumps(payload, separators=(",", ":"))
    if client is not None and client.is_connected():
        result = client.publish(topic, encoded, qos=qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[LOCAL MQTT PUBLISH ERROR] topic={topic} rc={result.rc}", flush=True)
    if topic.endswith("/alerts") or topic.endswith("/inference"):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(encoded + "\n")


def infer_window(window_end):
    start = window_end - WINDOW_SECONDS
    temp = [value for timestamp, value in temperatures if start < timestamp <= window_end]
    vib = [value for timestamp, value in vibrations if start < timestamp <= window_end]
    if len(temp) < 25 or len(vib) < 10:
        return
    features = extract_feature_vector(temp, vib)
    probs, latency_ms = predict(features)
    class_id = int(np.argmax(probs))
    event_time = time.time()
    payload = {
        "truck_id": TRUCK_ID,
        "timestamp_utc": event_time,
        "sensor_window_end_seconds": window_end,
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": float(np.max(probs)),
        "probabilities": [float(x) for x in probs],
        "inference_latency_ms": latency_ms,
        "model_path": str(MODEL_PATH),
        "features": features,
    }
    publish_and_log(f"{TOPIC_ROOT}/inference", payload)
    if class_id in (1, 2):
        alert = {**payload, "severity": CLASS_NAMES[class_id].upper()}
        publish_and_log(f"{TOPIC_ROOT}/alerts", alert, qos=1)
        print(f"[LOCAL {alert['severity']} ALERT] class={class_id} confidence={payload['confidence']:.3f}")


def on_connect(mqtt_client, _userdata, _flags, reason_code, _properties):
    if reason_code.is_failure:
        raise ConnectionError(f"MQTT broker connection rejected: {reason_code}")
    mqtt_client.subscribe(f"{TOPIC_ROOT}/sensor/+", qos=1)
    print(f"Subscribed to {TOPIC_ROOT}/sensor/+", flush=True)


def on_message(_client, _userdata, message):
    global next_window_end
    try:
        data = json.loads(message.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[INVALID SENSOR MESSAGE] {message.topic}: {exc}", flush=True)
        return
    timestamp = float(data["timestamp"])
    if next_window_end is not None and timestamp < next_window_end - 2 * WINDOW_SECONDS:
        # Sensor clock restarted (new simulator run or gateway reboot): start fresh windows.
        with lock:
            temperatures.clear()
            vibrations.clear()
            next_window_end = None
        print(f"[SENSOR CLOCK RESET] timestamp={timestamp}; window buffers cleared", flush=True)
    if message.topic.endswith("/temperature"):
        temperatures.append((timestamp, float(data["value"])))
    elif message.topic.endswith("/vibration_rms"):
        vibrations.append((timestamp, float(data["value"])))
    elif message.topic.endswith("/door_event"):
        if data.get("event") == "OPEN":
            publish_and_log(
                f"{TOPIC_ROOT}/alerts",
                {
                    "truck_id": TRUCK_ID,
                    "timestamp_utc": time.time(),
                    "severity": "DOOR_EVENT",
                    "event": "OPEN",
                    "sensor_timestamp": timestamp,
                },
            )
        return
    else:
        return
    if next_window_end is None:
        next_window_end = timestamp + WINDOW_SECONDS
    with lock:
        while timestamp >= next_window_end:
            infer_window(next_window_end)
            next_window_end += STEP_SECONDS
        cutoff = timestamp - 2 * WINDOW_SECONDS
        temperatures[:] = [(t, v) for t, v in temperatures if t >= cutoff]
        vibrations[:] = [(t, v) for t, v in vibrations if t >= cutoff]


def main():
    global client
    load_runtime()
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"logibridge-inference-{TRUCK_ID}",
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
