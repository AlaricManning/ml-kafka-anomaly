"""
Kafka producer that streams simulated IoT sensor readings.
~5% of messages are injected anomalies (temperature/pressure/vibration spikes).
"""

import json
import os
import random
import time

import numpy as np
from confluent_kafka import Producer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka-cluster-kafka-bootstrap:9092")
TOPIC = os.getenv("TOPIC", "sensor-readings")
INTERVAL_S = float(os.getenv("INTERVAL_S", "0.5"))
ANOMALY_RATE = float(os.getenv("ANOMALY_RATE", "0.05"))
SENSOR_IDS = [f"sensor-{i}" for i in range(1, 6)]


def normal_reading(sensor_id: str) -> dict:
    return {
        "sensor_id": sensor_id,
        "temperature": round(float(np.random.normal(22.5, 0.8)), 3),
        "pressure": round(float(np.random.normal(1.025, 0.01)), 4),
        "vibration": round(float(np.random.normal(0.2, 0.03)), 4),
        "timestamp": time.time(),
        "is_injected_anomaly": False,
    }


def anomaly_reading(sensor_id: str) -> dict:
    return {
        "sensor_id": sensor_id,
        "temperature": round(float(np.random.normal(38.0, 2.0)), 3),
        "pressure": round(float(np.random.normal(1.12, 0.04)), 4),
        "vibration": round(float(np.random.normal(0.9, 0.1)), 4),
        "timestamp": time.time(),
        "is_injected_anomaly": True,
    }


def on_delivery(err, msg):
    if err:
        print(f"Delivery error: {err}")


producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
print(f"Producer started -> {KAFKA_BOOTSTRAP} / topic={TOPIC}")

while True:
    sensor_id = random.choice(SENSOR_IDS)
    reading = anomaly_reading(sensor_id) if random.random() < ANOMALY_RATE else normal_reading(sensor_id)
    producer.produce(TOPIC, key=sensor_id, value=json.dumps(reading), callback=on_delivery)
    producer.poll(0)
    time.sleep(INTERVAL_S)
