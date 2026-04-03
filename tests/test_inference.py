import sys
import json
import numpy as np
import torch
import pytest

sys.path.insert(0, "consumer")
from consumer import extract_features, score, load_model

MODEL_DIR = "model/artifacts"


@pytest.fixture(scope="module")
def model_artifacts():
    return load_model(MODEL_DIR)


def test_extract_features_order():
    reading = {"temperature": 1.0, "pressure": 2.0, "vibration": 3.0}
    features = extract_features(reading)
    assert features[0] == 1.0
    assert features[1] == 2.0
    assert features[2] == 3.0


def test_extract_features_dtype():
    reading = {"temperature": 22.5, "pressure": 1.025, "vibration": 0.2}
    features = extract_features(reading)
    assert features.dtype == np.float32


def test_score_returns_float(model_artifacts):
    model, mean, std, threshold = model_artifacts
    features = extract_features({"temperature": 22.5, "pressure": 1.025, "vibration": 0.2})
    result = score(model, features, mean, std)
    assert isinstance(result, float)


def test_score_non_negative(model_artifacts):
    model, mean, std, threshold = model_artifacts
    features = extract_features({"temperature": 22.5, "pressure": 1.025, "vibration": 0.2})
    assert score(model, features, mean, std) >= 0.0


def test_normal_reading_below_threshold(model_artifacts):
    model, mean, std, threshold = model_artifacts
    reading = {"temperature": 22.5, "pressure": 1.025, "vibration": 0.2}
    features = extract_features(reading)
    assert score(model, features, mean, std) < threshold


def test_anomaly_reading_above_threshold(model_artifacts):
    model, mean, std, threshold = model_artifacts
    reading = {"temperature": 38.0, "pressure": 1.12, "vibration": 0.9}
    features = extract_features(reading)
    assert score(model, features, mean, std) > threshold


def test_anomaly_score_much_higher_than_normal(model_artifacts):
    model, mean, std, threshold = model_artifacts

    normal_features = extract_features({"temperature": 22.5, "pressure": 1.025, "vibration": 0.2})
    anomaly_features = extract_features({"temperature": 38.0, "pressure": 1.12, "vibration": 0.9})

    normal_error = score(model, normal_features, mean, std)
    anomaly_error = score(model, anomaly_features, mean, std)

    assert anomaly_error > normal_error * 100, "Anomaly score should be orders of magnitude higher than normal"
