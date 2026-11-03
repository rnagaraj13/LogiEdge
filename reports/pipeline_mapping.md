# D3 — Ten-stage Edge ML lifecycle mapping

The problem statement asks for all ten Module 4 stages but does not print their official stage names. This project maps the ten operational stages below; compare the labels with the course's Module 4 slide before submission and rename to the exact taught terminology if it differs.

| Stage | LogiEdge implementation |
|---|---|
| 1. Data acquisition | Temperature, vibration RMS and door contact readings arrive on truck-local MQTT sensor topics. The simulator makes the stream reproducible, while a production gateway would timestamp and aggregate the 500 Hz tri-axis accelerometer. |
| 2. Data validation and cleaning | The edge subscriber parses JSON, checks sensor topic/value types and maintains timestamped buffers. Invalid payloads are logged and excluded rather than silently converted into plausible values. |
| 3. Feature engineering | `data_pipeline/preprocessing.py` applies the 5-sample moving average, then constructs the 30-second window every 10 seconds and calculates the six documented statistics. |
| 4. Normalization | The first ten minutes of clean Normal simulator output produce `training_stats.npy`. Inference loads this frozen mean/std; it does not fit statistics on a live or drifting truck stream. |
| 5. Model training | `training/train_model.py` trains a 32/16 ReLU MLP on generated labeled windows and uses a stratified, held-out 20% validation split. The command gates progress on the required 88% accuracy. |
| 6. Validation and selection | Accuracy, confusion matrix, and Critical recall are written to results. The 3σ normalization-shift experiment uses the same held-out split; a candidate is not recommended unless its measured Class 2 recall exceeds 95%. |
| 7. Conversion and optimization | `convert_ptq.py` uses at least 200 representative samples for full INT8 TFLite PTQ. `prune_quantise.py` applies a PolynomialDecay schedule to whole hidden units, compacts dimensions, then calibrates a second INT8 model. |
| 8. Packaging and containerization | The inference Dockerfile installs dependencies before `COPY model.tflite .`, isolating the model layer from the unchanged dependency layers. `MODEL_PATH` selects a mounted variant without rebuilding. |
| 9. Deployment and inference | Mosquitto and the inference service run on the truck. The service publishes local class/confidence events and writes a persistent alert log even when no cellular uplink is available. |
| 10. Monitoring, update and feedback | PSI compares confidence bins to a 300-clean-window baseline; Ansible stages artifacts and verifies the container; the sync agent queues events locally and forwards them over authenticated TLS when connected. A rollback is required for a failed canary. |

The pipeline loops: reviewed field data and drift evidence can inform a later version, but live data never rewrites training normalization or the active model automatically.
