# D2 — Docker inference service

The service subscribes to the local sensor topics, applies the same five-sample filters and 30 s/10 s feature windows as training, loads the frozen `training_stats.npy`, invokes a TFLite model, and publishes inference/alert messages to the truck's local MQTT broker. It appends results to a persistent local JSONL log. Cellular connectivity is not in this inference path.

Prepare deployment artifacts after training:

```bash
python training/export_deployment_artifacts.py --variant m2
docker compose build inference
docker compose up -d
```

`MODEL_PATH` can point at a mounted TFLite variant (for example `/models/m3_pruned_int8.tflite`) without rebuilding the image. Verify the selected model accepts six normalized features and supports the runtime architecture.

## OTA model-layer cache demonstration

1. Build the image and save `docker history logibridge-inference:latest` plus image IDs.
2. Record the image size and the byte size of the model layer/registry transfer using the actual registry or buildx metadata.
3. Change only `inference/model.tflite`, rebuild with plain progress, and capture that dependency-install layers remain cached while the final model-copy layer changes.
4. Compute **measured** saved transfer bytes as `(full-image bytes − model-only update bytes) × 85`; record the measured byte counts and assumptions in the demo/report. Docker build cache on one laptop alone does not prove fleet bandwidth savings.

The 280 KB assignment estimate gives a model-only upper estimate of 23.8 MB for 85 trucks, costing ₹2.38 at ₹0.10/MB. It is not a measured Docker layer size. Device-side delta/update protocol, TLS framing, retries, and registry compression may change actual transfer volume.
