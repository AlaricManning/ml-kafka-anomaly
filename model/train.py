"""
Train the autoencoder on synthetic normal sensor data.

Outputs (saved to ./artifacts/):
  model.pt          — trained model weights
  norm_params.json  — mean/std for feature normalization
  threshold.json    — anomaly threshold (95th percentile of training errors)
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Synthetic normal data ---
# Features: temperature (°C), pressure (bar), vibration (mm/s)
np.random.seed(42)
N = 10_000

data = np.stack([
    np.random.normal(22.5, 0.8, N),   # temperature: ~22.5°C
    np.random.normal(1.025, 0.01, N), # pressure: ~1.025 bar
    np.random.normal(0.2, 0.03, N),   # vibration: ~0.2 mm/s
], axis=1).astype(np.float32)

# Normalize to zero mean / unit variance
mean = data.mean(axis=0)
std = data.std(axis=0)
data_norm = (data - mean) / std

with open(f"{ARTIFACTS_DIR}/norm_params.json", "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

# --- Train ---
dataset = TensorDataset(torch.tensor(data_norm))
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Autoencoder(input_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 50
for epoch in range(EPOCHS):
    total_loss = 0.0
    for (batch,) in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(loader):.6f}")

# --- Compute anomaly threshold ---
model.eval()
errors = model.reconstruction_error(torch.tensor(data_norm)).numpy()
threshold = float(np.percentile(errors, 95))
print(f"Anomaly threshold (95th pct): {threshold:.6f}")

with open(f"{ARTIFACTS_DIR}/threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f, indent=2)

torch.save(model.state_dict(), f"{ARTIFACTS_DIR}/model.pt")
print(f"Artifacts saved to ./{ARTIFACTS_DIR}/")
