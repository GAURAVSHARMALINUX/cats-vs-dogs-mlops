#!/usr/bin/env bash
# Deploy the freshly built image to a kind/minikube cluster and wait for rollout.
# Usage: scripts/deploy_kind.sh <image-ref> [namespace]
set -euo pipefail

IMAGE="${1:?usage: deploy_kind.sh <image-ref> [namespace]}"
NAMESPACE="${2:-mlops}"

echo "[deploy] applying manifests (image=${IMAGE}, namespace=${NAMESPACE})"
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/deployment.yaml
kubectl apply -f k8s/base/service.yaml
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml

# Pin the exact image built by this pipeline run (immutable SHA tag)
kubectl -n "${NAMESPACE}" set image deployment/cats-vs-dogs-api "api=${IMAGE}" --record=false
kubectl -n "${NAMESPACE}" set env deployment/cats-vs-dogs-api "APP_VERSION=${IMAGE##*:}"

echo "[deploy] waiting for rollout ..."
kubectl -n "${NAMESPACE}" rollout status deployment/cats-vs-dogs-api --timeout=300s

echo "[deploy] current state:"
kubectl -n "${NAMESPACE}" get pods,svc -o wide
