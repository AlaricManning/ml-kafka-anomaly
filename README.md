# ML Kafka Anomaly Detection

Real-time anomaly detection pipeline using a PyTorch autoencoder, Apache Kafka (via Strimzi), and Azure Kubernetes Service (AKS).

Simulated IoT sensor readings are streamed through Kafka. A consumer service runs each reading through a trained autoencoder — if the reconstruction error exceeds a threshold, it's flagged as an anomaly and published to an alerts topic.

---

## Architecture

```
Producer
(sensor readings)
     │
     ▼
Kafka Topic: sensor-readings
     │
     ▼
Consumer
(autoencoder inference)
     │
     ├── normal → logged, discarded
     │
     └── anomaly → Kafka Topic: anomaly-alerts
```

All three components run as pods inside an AKS cluster. Kafka is managed by the [Strimzi operator](https://strimzi.io/) in KRaft mode (no ZooKeeper). Kubernetes manifests are structured with [Kustomize](https://kustomize.io/) — a shared base is overlaid per environment (`staging`, `prod`), each deployed into its own namespace.

---

## How It Works

### The Model

The autoencoder is a simple feed-forward neural network trained **only on normal data**. It learns to compress and reconstruct normal sensor readings accurately.

```
Input (3 features) → Encoder → Latent (8-dim) → Decoder → Reconstructed input
```

At inference time, the **reconstruction error** (mean squared error between input and output) is used as the anomaly score:

- **Normal reading** → autoencoder reconstructs it accurately → low error (~0.000001)
- **Anomalous reading** → autoencoder fails to reconstruct it → high error (~6.5)

The anomaly threshold is set at the **95th percentile** of reconstruction errors on the training set. Anything above it is flagged as an anomaly.

### The Data

Three simulated sensor features:

| Feature | Normal distribution | Anomaly distribution |
|---|---|---|
| Temperature (°C) | N(22.5, 0.8) | N(38.0, 2.0) |
| Pressure (bar) | N(1.025, 0.01) | N(1.12, 0.04) |
| Vibration (mm/s) | N(0.2, 0.03) | N(0.9, 0.1) |

The producer injects anomalies at a configurable rate (default 5%). Each message includes an `is_injected_anomaly` flag so you can verify the model is detecting them correctly.

### Kafka Topics

| Topic | Purpose |
|---|---|
| `sensor-readings` | Raw sensor messages from the producer |
| `anomaly-alerts` | Messages the model flagged as anomalies |

### Components

| Component | Path | Description |
|---|---|---|
| Model training | `model/` | Trains the autoencoder, saves artifacts |
| Producer | `producer/` | Streams sensor readings to Kafka |
| Consumer | `consumer/` | Reads from Kafka, runs inference, emits alerts |
| Tests | `tests/` | pytest suite covering model, inference, and producer logic |
| Kubernetes manifests | `k8s/` | Kustomize base + staging/prod overlays |
| GitHub Actions | `.github/workflows/` | CI (tests on PR) and CD (build → staging → prod) |
| Setup scripts | `scripts/` | Azure infra provisioning and image build/push |

---

## CI/CD

Two GitHub Actions workflows handle testing and deployment:

**CI** (`ci.yml`) runs on every pull request to `main`. It trains the model, then runs the full pytest suite. PRs cannot merge without a passing `test` check.

**CD** (`cd.yml`) runs on every push to `main` and has three sequential jobs:

1. **build** — trains the model, builds producer and consumer Docker images via ACR, and pushes them tagged with the git SHA.
2. **deploy-staging** — runs automatically after build. Uses `kustomize edit set image` to stamp the SHA tag into `k8s/overlays/staging`, then applies the overlay to the `ml-kafka-staging` namespace on AKS.
3. **deploy-prod** — requires manual approval via the GitHub `prod` environment. Same process as staging but targets the `ml-kafka-prod` namespace.

Both workflows authenticate to Azure using OIDC (no long-lived secrets).

---

## Project Structure

```
ml-kafka-anomaly/
├── .github/
│   └── workflows/
│       ├── ci.yml                # Run tests on pull requests
│       └── cd.yml                # Build, deploy staging, deploy prod
├── model/
│   ├── autoencoder.py            # PyTorch model definition
│   ├── train.py                  # Training script, saves to model/artifacts/
│   └── requirements.txt
├── producer/
│   ├── producer.py               # Kafka producer (sensor simulation)
│   ├── Dockerfile
│   └── requirements.txt
├── consumer/
│   ├── consumer.py               # Kafka consumer + anomaly detection
│   ├── autoencoder.py            # Copy of model definition for Docker build
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   ├── test_autoencoder.py       # Model output shape and accuracy tests
│   ├── test_inference.py         # Feature extraction and scoring tests
│   ├── test_producer.py          # Producer reading generation tests
│   └── requirements.txt
├── k8s/
│   ├── base/                     # Shared manifests (no namespace, placeholder images)
│   │   ├── kafka-cluster.yaml    # Strimzi KafkaNodePool + Kafka (KRaft mode)
│   │   ├── kafka-topics.yaml     # sensor-readings + anomaly-alerts topics
│   │   ├── producer-deployment.yaml
│   │   ├── consumer-deployment.yaml
│   │   └── kustomization.yaml
│   └── overlays/
│       ├── staging/              # Namespace: ml-kafka-staging, ACR image tags
│       │   └── kustomization.yaml
│       └── prod/                 # Namespace: ml-kafka-prod, ACR image tags
│           └── kustomization.yaml
└── scripts/
    ├── azure-setup.sh            # Creates resource group, ACR, and AKS cluster
    └── build-push.sh             # Builds images via ACR and patches deployment YAMLs
```

---

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (`az`)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) — install via `az aks install-cli`
- [Miniforge](https://github.com/conda-forge/miniforge) (or any conda distribution)
- An Azure subscription

---

## Setup

### 1. Create the conda environment

```bash
conda create -n ml-kafka-anomaly python=3.11
conda activate ml-kafka-anomaly
pip install torch numpy confluent-kafka
```

### 2. Train the model

```bash
cd model
python train.py
# Saves model/artifacts/{model.pt, threshold.json, norm_params.json}
cd ..
```

### 3. Provision Azure infrastructure

Edit the config block at the top of `scripts/azure-setup.sh`:

| Variable | What to change |
|---|---|
| `RESOURCE_GROUP` | Any name you like |
| `LOCATION` | Azure region closest to you (e.g. `westeurope`) |
| `ACR_NAME` | Globally unique, lowercase alphanumeric only |
| `AKS_CLUSTER` | Any name you like |
| `AKS_NODE_COUNT` | Leave at `2` |
| `AKS_NODE_VM` | Leave at `Standard_B2s` (2 vCPU / 4 GB) |

Then run:

```bash
az login
./scripts/azure-setup.sh
```

This creates the resource group, ACR, and AKS cluster, and merges the cluster credentials into `~/.kube/config`.

### 4. Build and push Docker images

```bash
./scripts/build-push.sh <your-acr-name>.azurecr.io
```

This copies the model artifacts into the consumer build context and builds both images in Azure (no local Docker required).

### 5. Deploy to AKS

```bash
# Create namespaces
kubectl create namespace ml-kafka-staging
kubectl create namespace ml-kafka-prod

# Strimzi operator (installs into ml-kafka-staging; watches all namespaces by default)
kubectl create -f 'https://strimzi.io/install/latest?namespace=ml-kafka-staging' -n ml-kafka-staging
kubectl wait --for=condition=Ready pod -l name=strimzi-cluster-operator -n ml-kafka-staging --timeout=120s

# Deploy staging environment
kubectl apply -k k8s/overlays/staging

# Wait for Kafka to be ready (~2 minutes), then check rollout
kubectl wait kafka/kafka-cluster --for=condition=Ready --timeout=300s -n ml-kafka-staging
kubectl rollout status deployment/producer -n ml-kafka-staging
kubectl rollout status deployment/consumer -n ml-kafka-staging
```

### 6. Watch it run

```bash
kubectl logs -f deployment/consumer -n ml-kafka-staging
```

Expected output:

```
Model loaded | threshold=0.000003
Consuming sensor-readings -> detecting -> anomaly-alerts
[sensor-4] normal   error=0.000001  injected=False
[sensor-2] normal   error=0.000001  injected=False
[sensor-3] ANOMALY  error=6.522686  injected=True
```

---

## Configuration

All runtime settings are controlled via environment variables set in the deployment manifests.

| Variable | Service | Default | Description |
|---|---|---|---|
| `KAFKA_BOOTSTRAP` | both | `kafka-cluster-kafka-bootstrap:9092` | Kafka broker address |
| `TOPIC` | producer | `sensor-readings` | Topic to publish to |
| `INTERVAL_S` | producer | `0.5` | Seconds between messages |
| `ANOMALY_RATE` | producer | `0.05` | Fraction of messages that are injected anomalies |
| `INPUT_TOPIC` | consumer | `sensor-readings` | Topic to consume from |
| `OUTPUT_TOPIC` | consumer | `anomaly-alerts` | Topic to publish alerts to |
| `MODEL_DIR` | consumer | `/model` | Path to model artifacts inside the container |

---

## Teardown

To delete all Azure resources and stop incurring costs:

```bash
az group delete --name <your-resource-group> --yes
```

This deletes the AKS cluster, ACR, all storage, and everything else in the resource group.
