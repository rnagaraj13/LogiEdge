#!/usr/bin/env python3
"""Generate the assignment's synthetic cold-chain streams and publish via MQTT."""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Iterator

ANOMALIES = ("none", "temp_drift", "vibration", "combined")


def generate_stream(
    duration_minutes: float,
    anomaly: str = "none",
    *,
    seed: int = 631,
    truck_id: str = "truck-001",
    inject_after_seconds: int | None = None,
    anomaly_after: str | None = None,
    recover_after_seconds: int | None = None,
    recovery_mode: str = "none",
) -> Iterator[tuple[str, dict[str, object]]]:
    if anomaly not in ANOMALIES:
        raise ValueError(f"Unknown anomaly {anomaly!r}; choose from {ANOMALIES}.")
    if duration_minutes <= 0:
        raise ValueError("Duration must be positive.")
    if anomaly_after is not None and anomaly_after not in ANOMALIES:
        raise ValueError(f"Unknown injected anomaly {anomaly_after!r}.")
    if recovery_mode not in ANOMALIES:
        raise ValueError(f"Unknown recovery mode {recovery_mode!r}.")
    if (anomaly_after is None) != (inject_after_seconds is None):
        raise ValueError("Set both --inject-after-seconds and --anomaly-after.")
    if recover_after_seconds is not None and inject_after_seconds is None:
        raise ValueError("Recovery timing requires an anomaly injection time.")
    rng = random.Random(seed)
    seconds = int(duration_minutes * 60)
    next_door_event = rng.randint(30, 60)
    door_open = False
    door_close_at = -1
    for second in range(seconds):
        timestamp = float(second)
        active_mode = anomaly
        if inject_after_seconds is not None and second >= inject_after_seconds:
            active_mode = anomaly_after
        if (
            recover_after_seconds is not None
            and second >= inject_after_seconds + recover_after_seconds
        ):
            active_mode = recovery_mode
        temp_drift = active_mode in {"temp_drift", "combined"}
        vib_anomaly = active_mode in {"vibration", "combined"}
        temperature = rng.gauss(4.0, 0.3) + (0.08 * second if temp_drift else 0.0)
        yield (
            f"logibridge/trucks/{truck_id}/sensor/temperature",
            {
                "truck_id": truck_id, "timestamp": timestamp,
                "value": temperature, "unit": "C", "anomaly": active_mode,
            },
        )
        if second % 2 == 0:
            vibration = rng.gauss(1.2, 0.15) if vib_anomaly else rng.gauss(0.45, 0.05)
            yield (
                f"logibridge/trucks/{truck_id}/sensor/vibration_rms",
                {
                    "truck_id": truck_id, "timestamp": timestamp,
                    "value": max(0.0, vibration), "unit": "g", "anomaly": active_mode,
                },
            )
        if second >= next_door_event:
            if not door_open:
                door_open = True
                door_close_at = second + 10
                event = "OPEN"
            else:
                door_open = False
                event = "CLOSE"
                next_door_event = second + rng.randint(290, 430)
            yield (
                f"logibridge/trucks/{truck_id}/sensor/door_event",
                {
                    "truck_id": truck_id, "timestamp": timestamp,
                    "event": event, "anomaly": anomaly,
                },
            )
        elif door_open and second >= door_close_at:
            door_open = False
            yield (
                f"logibridge/trucks/{truck_id}/sensor/door_event",
                {
                    "truck_id": truck_id, "timestamp": timestamp,
                    "event": "CLOSE", "anomaly": anomaly,
                },
            )


def publish_local_mqtt(messages, host: str, port: int, keepalive: int = 30) -> None:
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to publish to the local Mosquitto broker.") from exc
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="logibridge-simulator")
    connected = False

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        nonlocal connected
        if reason_code.is_failure:
            raise ConnectionError(f"MQTT broker rejected connection: {reason_code}")
        connected = True

    client.on_connect = on_connect
    client.connect(host, port, keepalive)
    client.loop_start()
    deadline = time.monotonic() + 10
    while not connected and time.monotonic() < deadline:
        time.sleep(0.05)
    if not connected:
        client.loop_stop()
        raise TimeoutError(f"Could not connect to MQTT broker at {host}:{port}.")
    try:
        previous_timestamp = None
        for topic, payload in messages:
            timestamp = float(payload["timestamp"])
            if previous_timestamp is not None and timestamp > previous_timestamp:
                time.sleep(timestamp - previous_timestamp)
            previous_timestamp = timestamp
            qos = 1 if topic.endswith("/door_event") else 0
            info = client.publish(topic, json.dumps(payload), qos=qos)
            info.wait_for_publish(timeout=5)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise ConnectionError(f"MQTT publish failed for {topic}: rc={info.rc}")
    finally:
        client.disconnect()
        client.loop_stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly", choices=ANOMALIES, default="none")
    parser.add_argument("--duration-minutes", type=float, default=2.0)
    parser.add_argument("--truck-id", default="truck-001")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--seed", type=int, default=631)
    parser.add_argument(
        "--inject-after-seconds", type=int,
        help="Switch from --anomaly to --anomaly-after at this simulated second.",
    )
    parser.add_argument("--anomaly-after", choices=ANOMALIES)
    parser.add_argument(
        "--recover-after-seconds", type=int,
        help="Return to --recovery-mode this many seconds after injection.",
    )
    parser.add_argument("--recovery-mode", choices=ANOMALIES, default="none")
    args = parser.parse_args()
    messages = generate_stream(
        args.duration_minutes,
        args.anomaly,
        seed=args.seed,
        truck_id=args.truck_id,
        inject_after_seconds=args.inject_after_seconds,
        anomaly_after=args.anomaly_after,
        recover_after_seconds=args.recover_after_seconds,
        recovery_mode=args.recovery_mode,
    )
    publish_local_mqtt(messages, args.mqtt_host, args.mqtt_port)
    print(
        f"Published {args.anomaly} stream for {args.truck_id} "
        f"({args.duration_minutes:g} minutes) to {args.mqtt_host}:{args.mqtt_port}."
    )


if __name__ == "__main__":
    main()
