# C1/C3 — Local MQTT topics, QoS and sensor fusion

The broker runs on the truck's edge computer. The cellular uplink is a separate synchronisation client; the model and local alarm do not wait for cloud delivery.

```text
logibridge/
└── trucks/{truck_id}/
    ├── sensor/
    │   ├── temperature       JSON {timestamp, value, unit, anomaly}
    │   ├── vibration_rms     JSON {timestamp, value, unit, anomaly}
    │   └── door_event        JSON {timestamp, event: OPEN|CLOSE}
    ├── inference             JSON {class_id, class_name, confidence, probabilities, ...}
    └── alerts                JSON {severity, class, timestamp, ...}
```

| Topic | QoS | Retain | Rationale |
|---|---:|---|---|
| `sensor/temperature` | 0 | No | Periodic 1 Hz data; a later sample supersedes a lost one. |
| `sensor/vibration_rms` | 0 | No | Aggregate at 0.5 Hz; bounded sensor loss is preferable to back-pressure. |
| `sensor/door_event` | 1 | No | Discrete state transition should be delivered at least once; consumers deduplicate by event ID. |
| `inference` | 1 | No | Useful for traceability and PSI; local broker and disk journal retain operational evidence. |
| `alerts` | 1 | No | Safety event must be delivered at least once; durable journal and IDs make retry safe. |

Planned but not implemented in this submission: a retained `health` topic (model version, queue depth, uptime) for reconnecting operations clients, and a QoS 1 `ota/ack` topic for model activation and rollback acknowledgements. In the current build, OTA success is verified by the Ansible play's final running-container check.

Temperature and vibration are filtered independently with a five-sample moving average. Each aligned 30-second window advances every 10 seconds. Temperature produces mean, standard deviation, and rate-of-change; vibration produces RMS, peak, and Pearson kurtosis. The independently computed values are concatenated in a fixed six-feature schema. This **feature-level fusion** preserves cross-sensor relations for the MLP while sending only six values per inference window. Raw data-level fusion would require clock alignment and move much more 500 Hz data through the model; decision-level fusion would require separate models and make temperature/vibration interactions harder to learn. Door transitions are separately timestamped and logged; they remain outside the required six-feature vector, so the model does not infer a door-dependent class unless the feature contract is explicitly expanded and retrained.

The 0.5 Hz vibration stream is a sensor-side RMS aggregate from the 500 Hz raw tri-axis input. Production deployment must document the sensor's aggregation/filtering firmware and synchronise timestamps across sensor buses.
