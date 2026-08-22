#!/usr/bin/env bash
# One-command local deployment on minikube - used for the demo recording.
set -euo pipefail

IMAGE_TAG="${1:-local}"
IMAGE="cats-vs-dogs-mlops:${IMAGE_TAG}"

command -v minikube >/dev/null || { echo "minikube is not installed"; exit 1; }

if ! minikube status >/dev/null 2>&1; then
  echo "[local] starting minikube ..."
  minikube start --cpus=2 --memory=4096
fi

echo "[local] building ${IMAGE} inside minikube's docker daemon ..."
eval "$(minikube docker-env)"
docker build -t "${IMAGE}" .

echo "[local] deploying ..."
./scripts/deploy_kind.sh "${IMAGE}" mlops

URL="$(minikube service cats-vs-dogs-api -n mlops --url | head -n 1)"
echo "[local] service URL: ${URL}"

echo "[local] running smoke test ..."
./scripts/smoke_test.sh "${URL}"

echo "[local] running post-deployment performance check ..."
python -m src.monitoring.performance_check --url "${URL}" --sample-size 20 || true

cat <<EOF

  API      : ${URL}/docs
  Metrics  : ${URL}/metrics
  Prometheus: http://\$(minikube ip):30090
  Grafana   : http://\$(minikube ip):30300  (anonymous viewer enabled)
EOF
