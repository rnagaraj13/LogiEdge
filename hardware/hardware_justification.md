# B1/B2 — Hardware selection and Roofline analysis

## B1 — Constraint Triangle

The relevant vertices are performance/latency, power/thermal envelope, and cost/scale. The dominant practical constraint is **power while retaining enough local inference capacity**: the AI budget is 10 W from a 12 V vehicle supply through a DC–DC converter, and the platform must meet an end-to-end 90-second alert SLA across a disconnected route. Purchase cost is also important but cannot justify a device that cannot execute the model locally.

| Option | Pilot (85) | Full fleet (265) | Power vs 10 W | Assessment |
|---|---:|---:|---|---|
| Pi 5 (8 GB) + AI HAT+ (13 TOPS) | ₹12.75 lakh | ₹39.75 lakh | Stated TDP 7.5 W | Preferred pilot candidate: within the stated budget, substantial headroom versus MCU for an MLP, and lower cost than Orin. Validate steady-state vehicle power, thermal enclosure and actual TFLite/Hailo runtime compatibility. TOPS is not a latency guarantee. |
| Jetson Orin Nano Super (67 TOPS) | ₹38.25 lakh | ₹1.1925 crore | 15 W moderate load | Reject for this deployment budget: exceeds 10 W before conversion losses. Greater compute is not required by the small feature-vector MLP unless measured future workloads justify changing the power budget. |
| STM32H7 custom MCU | ₹2.975 lakh | ₹9.275 lakh | 0.4 W | Lowest cost/power, but the stated facts do not demonstrate supported TFLite model performance, memory headroom, maintainability, OTA or 90-second end-to-end performance. It remains a possible sensor/aggregation controller, not the selected classifier without hardware validation. |

Select Option 1 for the pilot, conditionally: the supplied 7.5 W estimate is under 10 W and leaves 2.5 W nominal headroom, while the ₹15,000 unit cost is 67% below the Orin option. A 90-second SLA is generous for a compact MLP but includes sensor-window acquisition; verify the total sensor-to-alert latency under CPU contention and cellular-offline operation. The Hailo-8L TOPS figure only applies if the trained model is compiled for and executed by its supported runtime. The baseline project exports TFLite for CPU inference; do not claim Hailo acceleration unless the Hailo toolchain and hardware measurements are available. Confirm startup current, thermal throttling, vehicle-grade power protection, memory, storage, and watchdog behavior before fleet deployment.

## B2 — Arithmetic intensity / Roofline

Given `45 MFLOPs/inference` and `18 MB/inference`, arithmetic intensity is:

`AI = 45 × 10^6 FLOP ÷ 18 × 10^6 byte = 2.5 FLOP/byte`.

The Pi 5 CPU ridge point using the supplied peak rates is:

`16 GFLOP/s ÷ 12 GB/s = 1.33 FLOP/byte`.

Since `2.5 > 1.33`, the stated workload lies on the compute-bound side of the ideal Roofline. The compute lower bound is `45 MFLOPs / 16 GFLOP/s = 2.81 ms`; the memory lower bound is `18 MB / 12 GB/s = 1.50 ms`. The ideal roof is the lower of those bounds, approximately 2.81 ms/inference. These are peak-specification bounds, not observed performance.

Latency improvement should first target compute demand: full INT8 operations where the runtime supports efficient integer kernels, smaller/pruned hidden layers, and accelerator execution only after compatibility and end-to-end profiling. Weight compression alone may reduce storage without reducing dense operation count; measure actual runtime. Also verify the supplied 18 MB access estimate: for a small MLP it may be dominated by runtime/interpreter overhead or be an aggregate assumption rather than resident model bytes.
