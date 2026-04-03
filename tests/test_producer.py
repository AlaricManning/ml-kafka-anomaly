import sys
import time

sys.path.insert(0, "producer")
from producer import normal_reading, anomaly_reading

EXPECTED_KEYS = {"sensor_id", "temperature", "pressure", "vibration", "timestamp", "is_injected_anomaly"}


def test_normal_reading_keys():
    r = normal_reading("sensor-1")
    assert r.keys() == EXPECTED_KEYS


def test_anomaly_reading_keys():
    r = anomaly_reading("sensor-1")
    assert r.keys() == EXPECTED_KEYS


def test_normal_reading_flag():
    assert normal_reading("sensor-1")["is_injected_anomaly"] is False


def test_anomaly_reading_flag():
    assert anomaly_reading("sensor-1")["is_injected_anomaly"] is True


def test_normal_reading_sensor_id():
    r = normal_reading("sensor-42")
    assert r["sensor_id"] == "sensor-42"


def test_normal_reading_values_in_range():
    # Sample 50 readings — values should stay within 5 std devs of normal mean
    for _ in range(50):
        r = normal_reading("sensor-1")
        assert 18.0 < r["temperature"] < 27.0, f"temperature out of range: {r['temperature']}"
        assert 0.97 < r["pressure"] < 1.08, f"pressure out of range: {r['pressure']}"
        assert 0.05 < r["vibration"] < 0.40, f"vibration out of range: {r['vibration']}"


def test_anomaly_reading_values_clearly_different():
    # Anomaly values should be clearly outside the normal operating range
    for _ in range(50):
        r = anomaly_reading("sensor-1")
        assert r["temperature"] > 30.0, f"anomaly temperature not elevated: {r['temperature']}"
        assert r["vibration"] > 0.5, f"anomaly vibration not elevated: {r['vibration']}"


def test_reading_timestamp_is_recent():
    before = time.time()
    r = normal_reading("sensor-1")
    after = time.time()
    assert before <= r["timestamp"] <= after
