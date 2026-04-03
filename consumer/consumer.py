"""
Kafka consumer that runs anomaly detection on each sensor reading.

Reads from:  sensor-readings
Writes to:   anomaly-alerts  (only when anomaly detected)

Model artifacts are expected at MODEL_DIR (default: /model).
Run model/train.py first and copy artifacts there before building the image.
"""

import json
import os

import numpy as np
import torch
from confluent_kafka import Consumer, Producer

from autoencoder import Autoencoder

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka-cluster-kafka-bootstrap:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "sensor-readings")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "anomaly-alerts")
MODEL_DIR = os.getenv("MODEL_DIR", "/model")


def load_model(model_dir: str):
    model = Autoencoder(input_dim=3)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
    model.eval()

    with open(f"{model_dir}/norm_params.json") as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    with open(f"{model_dir}/threshold.json") as f:
        threshold = float(json.load(f)["threshold"])

    return model, mean, std, threshold


def extract_features(reading: dict) -> np.ndarray:
    return np.array([
        reading["temperature"],
        reading["pressure"],
        reading["vibration"],
    ], dtype=np.float32)


def score(model: Autoencoder, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    features_norm = (features - mean) / std
    tensor = torch.tensor(features_norm).unsqueeze(0)
    return model.reconstruction_error(tensor).item()


def run():
    model, mean, std, threshold = load_model(MODEL_DIR)
    print(f"Model loaded | threshold={threshold:.6f}")

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": "anomaly-detector",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([INPUT_TOPIC])

    alert_producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    print(f"Consuming {INPUT_TOPIC} -> detecting -> {OUTPUT_TOPIC}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            reading = json.loads(msg.value())
            features = extract_features(reading)
            error = score(model, features, mean, std)

            is_anomaly = error > threshold
            label = "ANOMALY" if is_anomaly else "normal "
            injected = reading.get("is_injected_anomaly", "?")
            print(f"[{reading['sensor_id']}] {label}  error={error:.6f}  injected={injected}")

            if is_anomaly:
                alert = {
                    **reading,
                    "reconstruction_error": round(error, 6),
                    "threshold": round(threshold, 6),
                }
                alert_producer.produce(OUTPUT_TOPIC, key=reading["sensor_id"], value=json.dumps(alert))
                alert_producer.poll(0)

    finally:
        consumer.close()


if __name__ == "__main__":
    run()