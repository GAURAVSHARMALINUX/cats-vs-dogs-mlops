#!/usr/bin/env bash
# Post-deployment smoke test (M4.3).
#
# Fails (non-zero exit) if the deployed service is not healthy or cannot serve
# a real prediction, which in turn fails the CD pipeline.
#
# Usage: scripts/smoke_test.sh [BASE_URL] [IMAGE_PATH]
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-http://localhost:8000}}"
IMAGE_PATH="${2:-${IMAGE_PATH:-}}"
MAX_RETRIES="${MAX_RETRIES:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1" >&2; exit 1; }

echo "=============================================="
echo " Smoke test against: ${BASE_URL}"
echo "=============================================="

# ---------- 1. wait for the service to come up ----------
echo "[1/5] waiting for /health ..."
for attempt in $(seq 1 "${MAX_RETRIES}"); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${BASE_URL}/health" || echo 000)
  if [ "${code}" = "200" ]; then
    pass "/health returned 200 (attempt ${attempt})"
    break
  fi
  if [ "${attempt}" = "${MAX_RETRIES}" ]; then
    fail "/health never returned 200 (last status ${code})"
  fi
  echo "      attempt ${attempt}/${MAX_RETRIES}: got ${code}, retrying in ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done

# ---------- 2. health payload ----------
echo "[2/5] checking /health payload ..."
health_body=$(curl -s --max-time 10 "${BASE_URL}/health")
echo "      ${health_body}"
echo "${health_body}" | grep -q '"status":"healthy"' || fail "unexpected /health payload"
pass "health payload is correct"

# ---------- 3. readiness (model actually loaded) ----------
echo "[3/5] checking /ready ..."
for attempt in $(seq 1 "${MAX_RETRIES}"); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${BASE_URL}/ready" || echo 000)
  [ "${code}" = "200" ] && { pass "/ready returned 200 - model is loaded"; break; }
  [ "${attempt}" = "${MAX_RETRIES}" ] && fail "/ready never returned 200 (last status ${code})"
  echo "      attempt ${attempt}/${MAX_RETRIES}: got ${code}, retrying in ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done

# ---------- 4. a real prediction ----------
echo "[4/5] posting an image to /predict ..."
TMP_IMAGE=""
if [ -z "${IMAGE_PATH}" ] || [ ! -f "${IMAGE_PATH}" ]; then
  for candidate in data/processed/test/dogs data/processed/test/cats; do
    if [ -d "${candidate}" ]; then
      found=$(find "${candidate}" -name '*.jpg' | head -n 1 || true)
      [ -n "${found}" ] && IMAGE_PATH="${found}" && break
    fi
  done
fi
if [ -z "${IMAGE_PATH}" ] || [ ! -f "${IMAGE_PATH}" ]; then
  TMP_IMAGE="$(mktemp -t smoke-XXXXXX).jpg"
  python3 - "$TMP_IMAGE" <<'PY'
import sys
from PIL import Image
Image.new("RGB", (400, 300), (128, 128, 128)).save(sys.argv[1], "JPEG")
PY
  IMAGE_PATH="${TMP_IMAGE}"
  echo "      no dataset image found, using a generated test image"
fi
echo "      image: ${IMAGE_PATH}"

response=$(curl -s --max-time 60 -w "\n%{http_code}" -X POST \
  -F "file=@${IMAGE_PATH};type=image/jpeg" "${BASE_URL}/predict")
status=$(echo "${response}" | tail -n 1)
body=$(echo "${response}" | sed '$d')
[ -n "${TMP_IMAGE}" ] && rm -f "${TMP_IMAGE}"

echo "      HTTP ${status}"
echo "      ${body}"
[ "${status}" = "200" ] || fail "/predict returned ${status}"
echo "${body}" | grep -q '"predicted_label"' || fail "/predict response has no predicted_label"
echo "${body}" | grep -q '"probabilities"' || fail "/predict response has no probabilities"
echo "${body}" | grep -qE '"predicted_label": *"(cat|dog)"' || fail "predicted_label is not cat/dog"
pass "/predict returned a valid prediction"

# ---------- 5. metrics endpoint ----------
echo "[5/5] checking /metrics ..."
metrics=$(curl -s --max-time 10 "${BASE_URL}/metrics")
echo "${metrics}" | grep -q "predictions_total" || fail "predictions_total missing from /metrics"
echo "      $(echo "${metrics}" | grep -E '^predictions_total' | head -n 3)"
pass "prometheus metrics are exposed"

echo "=============================================="
echo " ALL SMOKE TESTS PASSED"
echo "=============================================="
