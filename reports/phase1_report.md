# Phase 1 — Architecture and hardware design

## Context and constraints

FreightBridge's pilot monitors 85 refrigerated vehicles with a planned expansion to 265. Temperature-sensitive cargo can warm by 1°C/minute after refrigeration failure, so LogiEdge must detect and locally alert within 90 seconds of a fault signature. This is an end-to-end requirement that includes sensor collection, windowing, inference, and local annunciation. Rural routes have seven documented cellular gaps lasting 35–90 minutes; network round-trip time is variable and cannot be a safety dependency. The selected architecture makes decisions on the truck and uses cellular only for eventual synchronisation and controlled OTA.

Assuming 32-bit values and excluding timestamps and transport overhead, raw temperature is 4 bytes/s. Three-axis vibration at 500 Hz is 6,000 bytes/s. Together this is 6,004 bytes/s × 86,400 = 518,745,600 bytes/day, or 518.75 decimal MB/truck-day. At ₹0.10/MB, that is ₹51.87/truck-day and approximately ₹4,409/day across 85 trucks. A 10-alert/day illustration at 1 KB/event would be 10 KB/day per truck and about ₹0.001/day; that alert count is an assumption, not measured evidence. The simulator receives 0.5 Hz aggregated RMS vibration, while a production gateway must do the 500 Hz aggregation locally. JSON, timestamps, MQTT/TLS and retries add overhead to the raw estimate.

During a complete cellular outage, cloud-only inference cannot return timely decisions and can leave the chain-of-custody record incomplete. Each edge node therefore runs local MQTT, preprocessing, inference, alarm actuation and a durable event journal. An outbox syncs event IDs in order when connectivity returns, deduplicating retries. On-device inference also limits exposure of detailed cargo readings, though contractual privacy still requires encrypted storage/transit, protected keys, least privilege and audits.

## Hardware selection

The Constraint Triangle is performance/latency, power/thermal headroom and unit cost. The dominant practical vertex is power under a 10 W AI budget, while preserving local model capability. The Pi 5 plus AI HAT+ is the pilot candidate: stated 7.5 W is below the limit, its ₹15,000 cost yields ₹12.75 lakh for 85 and ₹39.75 lakh for 265 vehicles, and its compute headroom is appropriate for a six-feature MLP. The Orin Nano Super is 15 W at moderate load—above budget—and costs ₹38.25 lakh for the pilot and ₹1.1925 crore at 265. The STM32H7 is ₹3,500 and 0.4 W, but the problem provides no evidence that it can execute this model with required memory, runtime, OTA and timing; it remains a possible sensor aggregation controller, not the selected classifier.

The 7.5 W figure and 13 TOPS do not prove actual latency or Hailo compatibility. The submitted TFLite path uses a CPU-compatible model; credit Hailo acceleration only after compilation with its supported toolchain and a real measurement. Validate sustained power, thermal enclosure, memory, supply transients, and sensor-to-alert p95/worst time on the selected hardware.

## Roofline result and architecture

Arithmetic intensity is 45 MFLOPs/18 MB = 2.5 FLOP/byte. The Pi 5 ridge point from the supplied peak rates is 16 GFLOP/s ÷ 12 GB/s = 1.33 FLOP/byte. Since 2.5 exceeds 1.33, the workload is compute-bound in this ideal model. Peak-based lower bounds are 2.81 ms compute and 1.50 ms memory per inference; real runtime is different. INT8 kernels, smaller structured-pruned layers, and a compatible accelerator are candidates to reduce compute time, but must be benchmarked.

Architecture figure: [system_architecture.png](../scenario_architecture/system_architecture.png). It shows all sensor connections, local broker, preprocessing, TFLite inference, local alert/outbox, cellular synchronisation and operations centre.

## Student measurement / reflection to complete

Before calling the 90-second target validated, run a disconnected-network test and measure timestamped sensor-to-alert times on the target device. Record the device, runtime, power meter, sample count, p95 and maximum. My actual hardware/measurement was: **[STUDENT: complete after the test]**.
