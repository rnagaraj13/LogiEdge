#!/usr/bin/env python3
"""Monitor 100 recent MQTT confidence scores against the clean reference with PSI."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

from monitoring.psi import calculate_psi, proportions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truck-id", default=os.getenv("TRUCK_ID", "truck-001"))
    parser.add_argument("--reference", default="monitoring/reference_dist.json")
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
    recent = deque(maxlen=100)
    last_check = 0.0
    was_drifting = False

    def on_connect(client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            raise ConnectionError(f"MQTT connection rejected: {reason_code}")
        client.subscribe(f"logibridge/trucks/{args.truck_id}/inference", qos=1)
        print(f"Monitoring confidence drift for {args.truck_id}.", flush=True)

    def on_message(_client, _userdata, message):
        nonlocal last_check, was_drifting
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            confidence = float(payload["confidence"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"[INVALID CONFIDENCE MESSAGE] {exc}", flush=True)
            return
        if not 0 <= confidence <= 1:
            print(f"[INVALID CONFIDENCE] outside [0,1]: {confidence}", flush=True)
            return
        recent.append(confidence)
        now = time.monotonic()
        if len(recent) < 100 or now - last_check < 60:
            return
        current = proportions(recent)
        value = calculate_psi(expected, current)
        print(f"[LOGIBRIDGE PSI] PSI={value:.3f} n={len(recent)} bins={current}", flush=True)
        if value > 0.25:
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
            was_drifting = True
        elif was_drifting and value < 0.10:
            print(f"[LOGIBRIDGE DRIFT RECOVERED] PSI={value:.3f}", flush=True)
            was_drifting = False
        last_check = now

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"logibridge-psi-{args.truck_id}",
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(os.getenv("MQTT_HOST", "localhost"), int(os.getenv("MQTT_PORT", "1883")))
    client.loop_forever()


if __name__ == "__main__":
    main()
