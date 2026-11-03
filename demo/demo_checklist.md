# Demo video checklist (15–20 minutes)

Record your own live run. Keep the student's terminal, commands, timestamps and actual outputs visible; do not use another person's recording or present fabricated results.

1. **0:00–1:00 — Context:** identify the 85-truck pilot, operational classes, offline requirement, and chosen hardware.
2. **1:00–3:00 — Architecture:** show `scenario_architecture/system_architecture.svg` and explain local alarm vs eventual cloud sync.
3. **3:00–5:00 — MQTT/simulator:** show Mosquitto running; run `simulator.py` with `none`, then `combined`; show temperature, vibration RMS and door event topics.
4. **5:00–7:00 — Preprocessing/data:** show feature order, 30 s/10 s windows, saved clean stats, dataset class counts and the documented drift-label conflict.
5. **7:00–9:00 — Training:** show validation accuracy, confusion matrix and Class 2 recall from your own `training/results/training_metrics.json`.
6. **9:00–11:00 — Model variants:** show M1/M2/M3 artifacts, calibration sample count, actual benchmark results, Pareto chart and the eligibility decision.
7. **11:00–13:00 — Docker/OTA:** build the image, show dependencies cached before model layer, change only model bytes, rebuild, and calculate savings from measured byte counts.
8. **13:00–15:00 — PSI:** show reference sample count 300, clean value, combined anomaly crossing (or honest measured failure), then clean recovery.
9. **15:00–17:00 — Ansible:** show both runs and actual output; the second run should report `changed=0` if inputs are unchanged.
10. **17:00–19:00 — Offline/close:** disconnect uplink while local broker remains, show local alert and queued outbox, reconnect and show sync; state limitations and answer questions.

Before recording, start the monitor before publishing anomaly data. The PSI sample window requires 100 inferences and evaluations are once per minute; the requested five-minute crossing may not occur for every trained model at a 10-second inference cadence. Preserve the actual result and explain it rather than editing logs.
