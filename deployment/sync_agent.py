#!/usr/bin/env python3
"""Persist local edge events and forward them to operations when cellular returns."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

TRUCK_ID = os.getenv("TRUCK_ID", "truck-001")
LOCAL_HOST = os.getenv("MQTT_HOST", "mosquitto")
LOCAL_PORT = int(os.getenv("MQTT_PORT", "1883"))
REMOTE_HOST = os.getenv("OPS_MQTT_HOST", "").strip()
REMOTE_PORT = int(os.getenv("OPS_MQTT_PORT", "8883"))
OUTBOX_PATH = Path(os.getenv("OUTBOX_PATH", "/var/lib/logibridge/outbox.sqlite3"))
EVENT_LOG = Path(os.getenv("LOCAL_EVENT_LOG", "/var/lib/logibridge/alerts.jsonl"))
lock = threading.Lock()
database = None


def open_outbox():
    global database
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(OUTBOX_PATH, check_same_thread=False)
    database.execute(
        "CREATE TABLE IF NOT EXISTS outbox "
        "(event_id TEXT PRIMARY KEY, topic TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    database.commit()


def enqueue(topic, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    event_id = str(payload.get("event_id") or hashlib.sha256(
        f"{topic}|{encoded}".encode("utf-8")
    ).hexdigest())
    with lock:
        database.execute(
            "INSERT OR IGNORE INTO outbox(event_id, topic, payload, created_at) VALUES (?, ?, ?, ?)",
            (event_id, topic, encoded, time.time()),
        )
        database.commit()


def on_connect(client, _userdata, _flags, reason_code, _properties):
    if reason_code.is_failure:
        raise ConnectionError(f"Local MQTT connection failed: {reason_code}")
    client.subscribe(f"logibridge/trucks/{TRUCK_ID}/#", qos=1)


def on_message(_client, _userdata, message):
    try:
        payload = json.loads(message.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[OUTBOX INVALID MESSAGE] {message.topic}: {exc}", flush=True)
        return
    if message.topic.endswith("/inference") or message.topic.endswith("/alerts"):
        enqueue(message.topic, payload)


def replay_log():
    if not EVENT_LOG.is_file():
        return
    with EVENT_LOG.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[OUTBOX INVALID LOG ROW] {exc}", flush=True)
                continue
            topic_suffix = "alerts" if "severity" in payload else "inference"
            enqueue(f"logibridge/trucks/{TRUCK_ID}/{topic_suffix}", payload)


def main():
    open_outbox()
    local = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"logibridge-outbox-{TRUCK_ID}",
    )
    local.on_connect = on_connect
    local.on_message = on_message
    local.connect(LOCAL_HOST, LOCAL_PORT, keepalive=30)
    local.loop_start()

    remote = None
    if REMOTE_HOST:
        remote = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"logibridge-uplink-{TRUCK_ID}",
        )
        username = os.getenv("OPS_MQTT_USERNAME", "")
        password = os.getenv("OPS_MQTT_PASSWORD", "")
        if username:
            remote.username_pw_set(username, password)
        if os.getenv("OPS_MQTT_TLS", "true").lower() == "true":
            remote.tls_set()
        remote.connect_async(REMOTE_HOST, REMOTE_PORT, keepalive=30)
        remote.loop_start()
    else:
        print("[UPLINK OFFLINE] OPS_MQTT_HOST is unset; events remain queued locally.", flush=True)

    try:
        while True:
            replay_log()
            if remote is not None and remote.is_connected():
                with lock:
                    pending = database.execute(
                        "SELECT event_id, topic, payload FROM outbox ORDER BY created_at LIMIT 100"
                    ).fetchall()
                for event_id, topic, payload in pending:
                    remote_topic = f"operations/{topic}"
                    info = remote.publish(remote_topic, payload, qos=1)
                    info.wait_for_publish(timeout=10)
                    if info.rc != mqtt.MQTT_ERR_SUCCESS:
                        print(f"[UPLINK PUBLISH ERROR] {remote_topic} rc={info.rc}", flush=True)
                        break
                    with lock:
                        database.execute("DELETE FROM outbox WHERE event_id=?", (event_id,))
                        database.commit()
                    print(f"[UPLINK SYNCED] event={event_id}", flush=True)
            time.sleep(5)
    finally:
        local.disconnect()
        local.loop_stop()
        if remote is not None:
            remote.disconnect()
            remote.loop_stop()
        database.close()


if __name__ == "__main__":
    main()
