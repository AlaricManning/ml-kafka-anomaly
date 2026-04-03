import sys
import torch
import pytest

sys.path.insert(0, "model")
from autoencoder import Autoencoder


@pytest.fixture
def model():
    return Autoencoder(input_dim=3)


def test_output_shape(model):
    x = torch.randn(8, 3)
    out = model(x)
    assert out.shape == x.shape


def test_reconstruction_error_shape(model):
    x = torch.randn(8, 3)
    errors = model.reconstruction_error(x)
    assert errors.shape == (8,)


def test_reconstruction_error_non_negative(model):
    x = torch.randn(16, 3)
    errors = model.reconstruction_error(x)
    assert (errors >= 0).all()


def test_trained_model_low_error_on_normal_data():
    """After training, normal data should reconstruct with very low error."""
    import json
    import numpy as np

    MODEL_DIR = "model/artifacts"
    model = Autoencoder(input_dim=3)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/model.pt", map_location="cpu"))
    model.eval()

    with open(f"{MODEL_DIR}/norm_params.json") as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    with open(f"{MODEL_DIR}/threshold.json") as f:
        threshold = float(json.load(f)["threshold"])

    # 100 normal samples — all should score below the threshold
    normal = np.stack([
        np.random.normal(22.5, 0.8, 100),
        np.random.normal(1.025, 0.01, 100),
        np.random.normal(0.2, 0.03, 100),
    ], axis=1).astype(np.float32)
    normal_norm = (normal - mean) / std

    errors = model.reconstruction_error(torch.tensor(normal_norm)).numpy()
    assert (errors < threshold).mean() > 0.90, "Fewer than 90% of normal samples scored below threshold"


def test_trained_model_high_error_on_anomalies():
    """Anomalous data should reconstruct with error well above threshold."""
    import json
    import numpy as np

    MODEL_DIR = "model/artifacts"
    model = Autoencoder(input_dim=3)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/model.pt", map_location="cpu"))
    model.eval()

    with open(f"{MODEL_DIR}/norm_params.json") as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    with open(f"{MODEL_DIR}/threshold.json") as f:
        threshold = float(json.load(f)["threshold"])

    # 100 anomaly samples — most should score above the threshold
    anomalies = np.stack([
        np.random.normal(38.0, 2.0, 100),
        np.random.normal(1.12, 0.04, 100),
        np.random.normal(0.9, 0.1, 100),
    ], axis=1).astype(np.float32)
    anomalies_norm = (anomalies - mean) / std

    errors = model.reconstruction_error(torch.tensor(anomalies_norm)).numpy()
    assert (errors > threshold).mean() > 0.95, "Fewer than 95% of anomalies scored above threshold"
