# GROUPNO_LogiEdge_Final

> **Draft template — not ready for LMS submission.** Replace every `[STUDENT: ...]` and `[MEASURE: ...]` item using your own runs, include your own technical reflection, verify the word count (2,500–3,000 excluding figures/tables/references), and export to `GROUPNO_LogiEdge_Final.pdf`. Do not claim that a simulation validates real pharmaceutical safety. The supplied assignment requires original analysis and measured evidence.

**Group:** [STUDENT: enter group number and members]  
**Repository:** [STUDENT: enter private GitHub URL and verify instructor collaborator access]  
**Demo video:** [STUDENT: enter tested unlisted/anyone-with-link URL]

## 1 — FreightBridge Deployment Context (target 400–500 words)

FreightBridge's pilot monitors 85 refrigerated vehicles and may scale to 265. A refrigeration failure can raise cargo temperature by 1°C per minute, so LogiEdge must detect and alert within 90 seconds. That is an end-to-end sensor-to-local-alert SLA, not simply model invocation time. The specified rural routes have seven signal gaps lasting 35–90 minutes. A cloud-only system therefore cannot guarantee a timely decision or complete chain-of-custody record during those gaps. The truck must make classification and alarm decisions locally; cellular is used for eventual synchronisation and controlled model updates.

For a lower-bound bandwidth calculation, assume 32-bit sensor values and exclude timestamps and protocol framing. One 1 Hz temperature stream uses 4 bytes/s. Raw tri-axis vibration at 500 Hz uses `500 × 3 × 4 = 6,000 bytes/s`. The combined 6,004 bytes/s over a day is 518,745,600 bytes, or 518.75 decimal MB/truck-day. At ₹0.10/MB that is ₹51.87 per truck-day and about ₹4,409 per day for the 85-truck pilot. Real JSON/MQTT/TLS and retransmission overhead increases it. Edge summaries avoid transmitting every sample: an illustrative ten 1 KB alerts/day is 10 KB/day, approximately ₹0.001/truck-day, but alert frequency must be measured rather than assumed. The simulator's 0.5 Hz vibration RMS is an already aggregated value; the production sensor gateway must document the raw 500 Hz preprocessing.

The edge node buffers timestamped events, features, model versions and decisions in a durable local outbox. If cellular disappears, local MQTT, filtering, inference and alarms continue. When a secure uplink returns, a retrying sync service transfers deduplicated event IDs. On-device inference also limits access to detailed cargo readings, though confidentiality requires encryption, protected keys, least-privilege credentials, signed OTA and audited retention; edge processing alone does not prove contractual compliance.

The Constraint Triangle compares performance/latency, power/thermal envelope and cost. The selected pilot candidate is Raspberry Pi 5 (8 GB) plus AI HAT+ at the specified 7.5 W and ₹15,000 per truck. This is below the 10 W AI budget and costs ₹12.75 lakh for 85 trucks or ₹39.75 lakh for 265. Jetson Orin Nano Super is stated at 15 W and ₹45,000, so it exceeds power and costs ₹38.25 lakh pilot / ₹1.1925 crore at scale. STM32H7 costs ₹3,500 and 0.4 W but is not evidenced to support the selected model/runtime, memory, OTA and SLA. Pi/HAT is conditional on measured sustained power, thermal behavior, sensor-to-alert latency, and runtime compatibility; 13 TOPS is not a timing result.

The supplied arithmetic intensity is `45 MFLOPs / 18 MB = 2.5 FLOP/byte`. The ridge point is `16 GFLOP/s / 12 GB/s = 1.33 FLOP/byte`, so the ideal Roofline classifies it as compute-bound. Peak lower bounds are 2.81 ms compute and 1.50 ms memory per inference; they are not measured latencies. The architecture is shown in [Figure 1](../scenario_architecture/system_architecture.svg).

**[MEASURE: target hardware, end-to-end run count, p95/worst latency, power, and offline test evidence; explain if the 90-second SLA passed.]**

## 2 — Sensor Pipeline and MQTT Design (target 500–600 words)

Temperature publishes at 1 Hz; vibration arrives as 0.5 Hz RMS aggregated from the raw tri-axis 500 Hz signal; door contacts publish discrete OPEN/CLOSE events. The local Mosquitto tree is `logibridge/trucks/{truck_id}/sensor/{temperature|vibration_rms|door_event}`. The service publishes `.../inference`, `.../alerts`, `.../health`, and `.../ota/ack`. Temperature/vibration use QoS 0 because each periodic value is superseded, while door events and decisions use QoS 1 for at-least-once delivery. Consumers deduplicate using event IDs; alert and outbox persistence are local. Health may be retained so a reconnecting client sees the latest status; event topics are not retained. The complete tree and justifications are in `data_pipeline/mqtt_architecture.md`.

The preprocessing path applies a five-sample moving average separately to temperature and vibration. A 30-second window advances every ten seconds. Temperature produces mean, standard deviation and rate in °C/min; vibration produces RMS, peak and Pearson kurtosis. These independently computed features are concatenated as a stable six-value vector. Feature-level fusion is appropriate because it captures joint sensor relations without requiring an MLP to consume the raw high-rate waveform. Data-level fusion would require time-aligning streams with very different rates and would transport/process far more values. Decision-level fusion would require independent classifiers and a later rule/model to combine outputs, obscuring cross-sensor relations. Door events remain logged and alerted separately because the assigned model input has exactly six values; they are not silently represented as an unrequested seventh feature.

The training normalization parameters are calculated from ten minutes of clean Normal simulation and stored in `training_stats.npy`. Inference loads that frozen file; it never learns statistics from live data, because adapting to a drift could hide the drift itself. A required sensitivity test compares the held-out model accuracy under correct statistics and statistics whose mean is shifted by 3σ while preserving the original scale. The problem also has a label inconsistency: +0.08°C per second crosses the Critical >3°C deviation after 38 seconds, although the dataset table assigns an entire 15-minute `temp_drift` run to Warning. This implementation preserves the specified ramp and labels windows by severity, treats vibration-only bearing anomaly as Warning, and treats combined as Critical. The resulting mode policy is recorded in the generation manifest and should be confirmed with the instructor.

| Normalization case | Held-out accuracy | Change from correct stats |
|---|---:|---:|
| Correct ten-minute clean mean/std | [MEASURE: %] | Baseline |
| Mean shifted by 3σ | [MEASURE: %] | [MEASURE: percentage points] |

**[STUDENT: add the actual measured values, seed, split sizes, and your explanation of the change.]**

## 3 — Model Deployment and Pipeline Mapping (target 400–500 words)

The training script generates the Normal, Warning and Critical windows, freezes clean-data statistics, trains the two-hidden-layer 32/16 ReLU MLP, and uses a stratified 20% held-out split. It records accuracy, confusion matrix and class recall, and stops if accuracy is below the required 88%. The deployment candidate must additionally exceed 95% Class 2 recall. PTQ conversion uses at least 200 representative windows and full integer TFLite input/output. The third candidate uses a TensorFlow Model Optimization Toolkit PolynomialDecay schedule to remove whole hidden units, compacts their dimensions, then performs full INT8 PTQ; report measured size and accuracy to verify the optimization did not trade away safety-critical recall.

The ten lifecycle stages are mapped specifically in `reports/pipeline_mapping.md`: acquisition, validation, feature engineering, frozen normalization, training, validation/selection, conversion/optimization, packaging, edge deployment/inference, and monitoring/update. The assignment statement does not list the official ten stage names; compare the table labels with the exact Module 4 lecture terms before final export.

The Docker image uses Python 3.11 slim and installs pip dependencies before `COPY model.tflite .`. This preserves dependency layers when the model alone changes. `MODEL_PATH` selects a mounted TFLite variant without rebuilding. The inference service subscribes locally, executes preprocessing and model inference, writes persistent local events, and publishes decisions without the cellular uplink. The sync agent keeps a SQLite outbox and forwards events over configured TLS when coverage returns.

The OTA nominal calculation treats the supplied 280 KB model as 0.28 decimal MB. Full fleet transfer for 85 trucks is 23.8 MB and ₹2.38. A ten-truck canary costs ₹0.28 for its 2.8 MB initial phase; the remaining 75 trucks cost ₹2.10, for the same ₹2.38 after complete rollout. Shadow mode also transfers the candidate and adds unquantified compute/telemetry. A staged canary is preferred for blast-radius control and rural retry behavior, not for reducing total completed transfer bytes.

**[MEASURE: image bytes and model-layer bytes before/after rebuild; layers actually rebuilt; measured 85-device byte/cost savings; Docker output evidence.]**

## 4 — Optimisation and Pareto Analysis (target 400–500 words)

The benchmarking script performs ten excluded warm-ups and 200 measured calls per variant. It reports mean inference latency, p95 latency, model size, held-out accuracy and energy/inference. The energy estimate uses `psutil` process CPU percentage and an explicitly configured laptop TDP estimate; it is a comparative estimate, not a calibrated edge-board power measurement. M1 is the FP32 Keras baseline, M2 uses full INT8 post-training quantization and M3 combines structured hidden-unit pruning with PTQ INT8. The Pareto chart uses measured latency and size for visualization; non-dominance is calculated across latency, size, energy and accuracy. An attractive size/latency point is not eligible if its Critical recall fails the stated threshold.

| Variant | Mean latency ms (200) | p95 ms | Size KB | Accuracy % | Energy mJ/inference |
|---|---:|---:|---:|---:|---:|
| M1 FP32 baseline | [MEASURE] | [MEASURE] | [MEASURE] | [MEASURE] | [MEASURE] |
| M2 full INT8 PTQ | [MEASURE] | [MEASURE] | [MEASURE] | [MEASURE] | [MEASURE] |
| M3 structured prune + PTQ INT8 | [MEASURE] | [MEASURE] | [MEASURE] | [MEASURE] | [MEASURE] |

Class 2 recall is separately evaluated from `recall = true critical predictions / all actual critical validation windows`. Record each numerator and denominator from the confusion matrix, not a recollection. **[MEASURE: recommended variant Critical recall, TP/actual-Critical count, and whether it is strictly above 95%.]**

The 90-second SLA includes the 30-second feature window and ten-second step. A newly appearing fault may wait for the next window step; the implementation must measure the actual worst-case sensor-to-alert path and demonstrate it remains below 90 seconds. The Pi/HAT option fits the stated 10 W budget on paper, but actual memory/flash/RSS, sustained power, runtime and accelerator usage need target-device measurement. TOPS and ideal Roofline bounds are not evidence of deployment speed. My recommendation is **[STUDENT: name only a variant that passes both 88% overall accuracy and >95% Critical recall; otherwise say do not deploy and explain remediation]**. Include [Pareto chart](../optimisation/results/pareto_chart.png) after the real benchmark command has generated it.

## 5 — MLOps Monitoring and Reflection (target 300–400 words)

The clean confidence reference is generated by running the selected model on 300 independent Normal simulation windows. Maximum class confidence is binned as [0,.25), [.25,.5), [.5,.75), and [.75,1]. PSI compares the reference proportions to the most recent 100 inferences every 60 seconds with a small epsilon for empty bins. PSI above 0.25 logs a drift alert; after restoring clean data, a result below 0.10 indicates recovery. The injected mode is a covariate/operational distribution shift in simulated inputs; confidence-only PSI detects a change in output confidence distribution, not the root cause or correctness of labels.

| Phase | PSI / time / sample evidence |
|---|---|
| Clean baseline | [MEASURE: reference bin proportions and PSI] |
| Combined drift injection | [MEASURE: time to >0.25; attach terminal log] |
| Restored clean input | [MEASURE: time until <0.10; attach terminal log] |

The Ansible play has exactly seven tasks and is designed to avoid stopping an unchanged container. Run it twice with immutable image/artifact inputs. **[STUDENT: paste the two actual PLAY RECAPs; confirm second `changed=0` or explain any real update.]** The sync agent queues events locally and replays them over authenticated TLS when uplink returns; locally generated alerts do not wait on that path.

**Genuine technical difficulty:** [STUDENT: describe a problem you personally encountered, the logs/tests that exposed it, and how you fixed or mitigated it. Do not invent a difficulty.]  
**One architectural change:** [STUDENT: propose a change and justify it using your observed latency, reliability, energy, cost or drift evidence.]

The outputs are synthetic training/engineering evidence. A successful lab run is not certification for pharmaceutical deployment; a real fleet pilot requires qualified sensors, traceable calibration, hardware/environment tests, security review, operating procedures, and clinical/customer approval.

## Conclusion

**[STUDENT: summarize measured results, explicit go/no-go for the pilot, and next validation needed.]**

## References (excluded from word count)

1. FreightBridge LogiEdge Mini-Project Problem Statement — Student, supplied course document.
2. TensorFlow Lite documentation: https://www.tensorflow.org/lite
3. Mosquitto documentation: https://mosquitto.org/documentation/
4. Ansible documentation: https://docs.ansible.com/
5. TensorFlow Model Optimization documentation: https://www.tensorflow.org/model_optimization
