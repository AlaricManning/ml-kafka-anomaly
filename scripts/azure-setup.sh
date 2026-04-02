#!/usr/bin/env bash
# Azure infrastructure setup for ml-kafka-anomaly
# Prerequisites: az CLI installed and logged in (`az login`)
#
# Usage:
#   chmod +x scripts/azure-setup.sh
#   ./scripts/azure-setup.sh

set -euo pipefail

# ── Config — edit these ────────────────────────────────────────────────────────
RESOURCE_GROUP="ml-kafka-rg"
LOCATION="eastus"
ACR_NAME="mlkafkaacr"          # must be globally unique, lowercase alphanumeric only
AKS_CLUSTER="ml-kafka-aks"
AKS_NODE_COUNT=2
AKS_NODE_VM="Standard_B2s"    # 2 vCPU, 4 GB — enough for Kafka + 2 pods
# ──────────────────────────────────────────────────────────────────────────────

echo "==> Checking az login..."
az account show --query "{subscription:name, id:id}" -o table

echo ""
echo "==> Creating resource group: $RESOURCE_GROUP ($LOCATION)"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

echo ""
echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --output table

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
echo "    ACR login server: $ACR_LOGIN_SERVER"

echo ""
echo "==> Creating AKS cluster: $AKS_CLUSTER"
echo "    (This takes ~5 minutes)"
az aks create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AKS_CLUSTER" \
  --node-count "$AKS_NODE_COUNT" \
  --node-vm-size "$AKS_NODE_VM" \
  --attach-acr "$ACR_NAME" \
  --generate-ssh-keys \
  --output table

echo ""
echo "==> Fetching AKS credentials (merges into ~/.kube/config)"
az aks get-credentials \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AKS_CLUSTER" \
  --overwrite-existing

echo ""
echo "==> Verifying cluster access"
kubectl get nodes

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Done. Copy these values — you'll need them next:   "
echo "  ACR_LOGIN_SERVER : $ACR_LOGIN_SERVER               "
echo "  Replace YOUR_ACR in k8s/*/deployment.yaml with:   "
echo "  $ACR_LOGIN_SERVER                                  "
echo "══════════════════════════════════════════════════════"
