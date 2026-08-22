#!/usr/bin/env bash
# Initialise DVC with a local remote so `dvc push` / `dvc pull` work offline.
# (A real project would point this at S3/GDrive/Azure Blob instead.)
set -euo pipefail

REMOTE_PATH="${1:-../dvc-remote}"

if [ ! -d .dvc ]; then
  echo "[dvc] initialising ..."
  dvc init
fi

mkdir -p "${REMOTE_PATH}"
dvc remote add -d --force localremote "${REMOTE_PATH}"
dvc config core.analytics false

echo "[dvc] remote configured:"
dvc remote list
cat <<EOF

Next steps:
  dvc repro        # run download -> preprocess -> train
  dvc push         # push data + model to the remote
  dvc metrics show # read artifacts/metrics.json
  git add dvc.lock dvc.yaml .dvc/config && git commit -m "chore: update pipeline"
EOF
