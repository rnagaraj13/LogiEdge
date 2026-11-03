# E3 — OTA strategy comparison

Assumption for arithmetic: 280 KB is decimal `0.28 MB`; SIM price is ₹0.10/MB; one model download per truck per update, ignoring protocol headers/retries. Fleet-wide completed rollout cost is the same regardless of staged strategy.

| Strategy | Transfer arithmetic | Pilot transfer | Cost/update cycle | Cold-chain trade-off |
|---|---|---:|---:|---|
| Full replacement | `85 × 0.28 MB` | 23.8 MB | ₹2.38 | Fastest rollout and simplest state, but a defect can affect every truck at once; a rural outage can leave a truck on an older model without a deliberate canary gate. |
| Canary (10 first, then 75 after validation) | Initial `10 × 0.28 = 2.8 MB`; remainder `75 × 0.28 = 21.0 MB` | 2.8 MB first phase; 23.8 MB after full rollout | ₹0.28 initial; ₹2.10 second phase; ₹2.38 total | Recommended. Validate health, drift, alert delivery, and model version on ten trucks before releasing to the remaining fleet. Pause/rollback on a safety regression. Store signed packages locally and resume downloads after rural coverage returns. |
| Shadow mode | Candidate model still transferred to all 85 to run side-by-side | 23.8 MB | ₹2.38, plus unquantified shadow telemetry | Useful for comparing candidate outputs without replacing active decisions; however, it doubles inference workload temporarily and still needs version distribution. Measure additional CPU/power and telemetry before enabling on a 10 W node. |

The update cadence is every six weeks. A canary reduces blast radius, not total completed model bytes. A shadow experiment may be combined with a 10-truck canary; it is not a zero-bandwidth alternative. The local alert model remains active throughout; installation is atomic, signature-verified, health-checked, and rollback-capable. Actual transfer volumes and costs must use registry/device logs, not the nominal file size.
