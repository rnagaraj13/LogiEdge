"""TensorFlow Lite export and inference helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def convert_full_int8(model, calibration_features, output_path):
    features = np.asarray(calibration_features, dtype=np.float32)
    if len(features) < 200:
        raise ValueError(f"Full INT8 PTQ requires at least 200 calibration samples; got {len(features)}.")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [features[i:i + 1]] for i in range(len(features))
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()
    with open(output_path, "wb") as file:
        file.write(data)
    return data


def predict_tflite(model_path, features):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    scale, zero_point = input_info["quantization"]
    if input_info["dtype"] == np.int8:
        if scale <= 0:
            raise ValueError("Invalid input quantization scale in TFLite model.")
        x = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
    else:
        x = x.astype(input_info["dtype"])
    outputs = []
    for row in x:
        interpreter.set_tensor(input_info["index"], row[None, :])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_info["index"])[0]
        out_scale, out_zero = output_info["quantization"]
        if output_info["dtype"] == np.int8 and out_scale > 0:
            prediction = (prediction.astype(np.float32) - out_zero) * out_scale
        outputs.append(prediction.astype(np.float32))
    return np.asarray(outputs)
