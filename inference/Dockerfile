FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Keep the model layer separate: model-only OTA replaces this layer, not dependencies.
COPY model.tflite .
COPY training_stats.npy .
COPY preprocessing.py .
COPY inference_service.py .
RUN mkdir -p /var/lib/logibridge && chown -R 10001:10001 /var/lib/logibridge
USER 10001:10001
CMD ["python", "inference_service.py"]
