#!/usr/bin/env python3
"""Monitor 100 recent MQTT confidence scores against the clean reference with PSI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

from monitoring.psi import calculate_psi, proportions

WINDOW_SIZE = 100
CHECK_INTERVAL_SECONDS = 60


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truck-id", default=os.getenv("TRUCK_ID", "truck-001"))
    parser.add_argument("--reference", default="monitoring/reference_dist.json")
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the live warm-up/countdown status line (it is shown only on interactive terminals).",
    )
    args = parser.parse_args()
    reference_path = Path(args.reference)
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Build the measured 300-window confidence reference first: "
            "python monitoring/build_reference.py"
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("status") != "measured" or reference.get("sample_count", 0) < 300:
        raise ValueError("Reference must contain measured confidence bins from at least 300 clean windows.")
    expected = reference["reference_proportions"]
    recent = deque(maxlen=WINDOW_SIZE)
    arrivals = deque(maxlen=10)
    last_check = 0.0
    was_drifting = False
    show_progress = sys.stdout.isatty() and not args.no_progress
    output_lock = threading.Lock()

    def emit(line):
        with output_lock:
            if show_progress:
                sys.stdout.write("\r\033[K")
            print(line, flush=True)

    def status_text():
        now = time.monotonic()
        count = len(recent)
        if not arrivals:
            return "Waiting for the first prediction..."
        idle = now - arrivals[-1]
        if idle > 30:
            return f"No new predictions for {_format_duration(idle)} - is the simulator running?"
        if count < WINDOW_SIZE:
            text = f"Warm-up: {count}/{WINDOW_SIZE} predictions collected"
            if len(arrivals) >= 2:
                gap = (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1)
                text += f" | first PSI check in ~{_format_duration((WINDOW_SIZE - count) * gap)}"
            return text
        remaining = CHECK_INTERVAL_SECONDS - (now - last_check)
        if remaining > 0:
            return f"Window n={count} | next PSI check in {_format_duration(remaining)}"
        return f"Window n={count} | PSI check at the next prediction"

    def show_status():
        while True:
            with output_lock:
                sys.stdout.write("\r\033[K" + status_text())
                sys.stdout.flush()
            time.sleep(1)

    def on_connect(client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            raise ConnectionError(f"MQTT connection rejected: {reason_code}")
        client.subscribe(f"logibridge/trucks/{args.truck_id}/inference", qos=1)
        emit(f"Monitoring confidence drift for {args.truck_id}.")

    def on_message(_client, _userdata, message):
        nonlocal last_check, was_drifting
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            confidence = float(payload["confidence"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            emit(f"[INVALID CONFIDENCE MESSAGE] {exc}")
            return
        if not 0 <= confidence <= 1:
            emit(f"[INVALID CONFIDENCE] outside [0,1]: {confidence}")
            return
        recent.append(confidence)
        now = time.monotonic()
        arrivals.append(now)
        if len(recent) < WINDOW_SIZE or now - last_check < CHECK_INTERVAL_SECONDS:
            return
        current = proportions(recent)
        value = calculate_psi(expected, current)
        emit(f"[LOGIBRIDGE PSI] PSI={value:.3f} n={len(recent)} bins={current}")
        if value > 0.25:
            emit(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}")
            was_drifting = True
        elif was_drifting and value < 0.10:
            emit(f"[LOGIBRIDGE DRIFT RECOVERED] PSI={value:.3f}")
            was_drifting = False
        last_check = now

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"logibridge-psi-{args.truck_id}",
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(os.getenv("MQTT_HOST", "localhost"), int(os.getenv("MQTT_PORT", "1883")))
    if show_progress:
        threading.Thread(target=show_status, daemon=True).start()
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        emit("Drift monitor stopped.")


if __name__ == "__main__":
    main()
