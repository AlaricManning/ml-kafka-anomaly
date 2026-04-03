#!/usr/bin/env bash
# Build and push producer + consumer images to ACR using `az acr build`
# (runs the Docker build in Azure — no local Docker daemon needed)
#
# Usage:
#   ./scripts/build-push.sh <acr-login-server>
#   ./scripts/build-push.sh mlkafkaacr.azurecr.io
#
# Run model/train.py before calling this script.

set -euo pipefail

ACR_LOGIN_SERVER="${1:-}"
if [[ -z "$ACR_LOGIN_SERVER" ]]; then
  echo "Usage: $0 <acr-login-server>"
  echo "  e.g. $0 mlkafkaacr.azurecr.io"
  exit 1
fi

ACR_NAME="${ACR_LOGIN_SERVER%%.*}"   # strip .azurecr.io
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# --- Ensure model artifacts exist ---
ARTIFACTS="$ROOT/model/artifacts"
if [[ ! -f "$ARTIFACTS/model.pt" ]]; then
  echo "ERROR: model artifacts not found at $ARTIFACTS/"
  echo "  Run:  cd model && python train.py"
  exit 1
fi

echo "==> Copying model artifacts into consumer build context"
mkdir -p "$ROOT/consumer/model"
cp "$ARTIFACTS/model.pt" "$ARTIFACTS/threshold.json" "$ARTIFACTS/norm_params.json" "$ROOT/consumer/model/"

echo ""
echo "==> Building producer image via ACR"
az acr build \
  --registry "$ACR_NAME" \
  --image "ml-kafka/producer:latest" \
  "$ROOT/producer"

echo ""
echo "==> Building consumer image via ACR"
az acr build \
  --registry "$ACR_NAME" \
  --image "ml-kafka/consumer:latest" \
  "$ROOT/consumer"

echo ""
echo "==> Images pushed:"
echo "    $ACR_LOGIN_SERVER/ml-kafka/producer:latest"
echo "    $ACR_LOGIN_SERVER/ml-kafka/consumer:latest"

echo ""
echo "Deploy with Kustomize:"
echo "  kubectl apply -k k8s/overlays/staging"
echo "  kubectl apply -k k8s/overlays/prod"
