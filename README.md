# Cats vs Dogs — End-to-End MLOps Pipeline

**MLOps (S1-25_AIMLCZG523) — Assignment 2**
Binary image classification (Cats vs Dogs) for a pet adoption platform, wired
end to end: data versioning → experiment tracking → containerised inference →
CI → CD onto Kubernetes → monitoring and post-deployment evaluation.

Everything runs on free, open-source tooling. There is **no cloud account and
no paid service anywhere in the pipeline** — the CD stage spins up a throwaway
Kubernetes cluster (`kind`) inside the GitHub Actions runner, and the same
manifests run on `minikube` locally.

---

## 1. What is where

```
.
├── params.yaml                  # single source of truth for every parameter
├── dvc.yaml                     # download -> preprocess -> train (dvc repro)
├── requirements*.txt            # pinned envs: full / serving / dev
├── Dockerfile                   # inference image (non-root, healthcheck)
├── docker-compose.yml           # local stack: API + MLflow + Prometheus + Grafana
├── Makefile                     # every command below has a make target
├── src/
│   ├── config.py                # params.yaml loader + env overrides
│   ├── data/download.py         # M1  fetch raw images
│   ├── data/preprocess.py       # M1  224x224 RGB, seeded 80/10/10 split
│   ├── models/model.py          # M1  baseline CNN + MobileNetV2 transfer
│   ├── models/train.py          # M1  training + MLflow tracking
│   ├── utils_plots.py           # M1  confusion matrix + loss/accuracy curves
│   ├── serving/inference.py     # M2  preprocessing + model service
│   ├── serving/app.py           # M2  FastAPI: /health /ready /predict /metrics
│   └── monitoring/performance_check.py   # M5  post-deployment evaluation
├── tests/                       # M3  23 unit + API tests
├── k8s/base/                    # M4  Namespace, Deployment, Service, HPA
├── k8s/monitoring/              # M5  Prometheus + Grafana on the cluster
├── monitoring/                  # M5  Prometheus scrape config, Grafana dashboard
├── scripts/                     # deploy, smoke test, local minikube, DVC setup
└── .github/workflows/
    ├── ci-cd.yml                # M3 + M4  test -> train -> build -> deploy -> smoke
    └── pr-auto-merge.yml        # auto-approve + auto-merge once CI is green
```

---

## 2. Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

make data           # download + preprocess (224x224 RGB, 80/10/10)
make train          # train, log to MLflow, write artifacts/model.keras
make test           # run the test suite
make serve          # uvicorn on http://localhost:8000/docs
```

In another shell:

```bash
make smoke          # health + readiness + a real prediction + metrics
make monitor        # post-deployment accuracy against the held-out test set
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiment UI
```

Full local stack (API + MLflow + Prometheus + Grafana):

```bash
make train          # the image needs artifacts/model.keras to exist
docker compose up --build
#  API        http://localhost:8000/docs
#  MLflow     http://localhost:5000
#  Prometheus http://localhost:9090
#  Grafana    http://localhost:3000   (anonymous viewer enabled)
```

---

## 3. How each module is satisfied

### M1 — Model development & experiment tracking

| Requirement | Where |
|---|---|
| Git for code versioning | this repository |
| DVC for data/preprocessed-data versioning | `dvc.yaml` (3 stages), `scripts/setup_dvc.sh` |
| Baseline model | `build_baseline_cnn()` — small CNN trained from scratch |
| Second model for comparison | `build_mobilenetv2()` — frozen ImageNet backbone |
| Serialised model | `artifacts/model.keras` **and** a legacy `artifacts/model.h5` |
| Experiment tracking | MLflow: params, per-epoch metrics, test metrics, tags |
| Artifacts logged | `confusion_matrix.png`, `training_curves.png`, `classification_report.txt`, the model itself |

Preprocessing is a real, reproducible DVC stage: every image is converted to
RGB, resized to 224×224 and written into a **seeded** 80/10/10 split, so the
same commit always produces the same split. Augmentation (horizontal flip,
rotation, zoom, translation) is applied to the training split only, driven by
`params.yaml`.

Normalisation lives **inside** the saved model (`Rescaling` layer). The serving
code therefore only resizes, which removes any chance of train/serve
preprocessing skew.

Compare the two models:

```bash
python -m src.models.train --architecture baseline_cnn
python -m src.models.train --architecture mobilenetv2
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### M2 — Packaging & containerisation

| Requirement | Where |
|---|---|
| REST API | FastAPI in `src/serving/app.py` |
| Health check endpoint | `GET /health` (liveness) and `GET /ready` (model actually loaded) |
| Prediction endpoint | `POST /predict` — returns label, confidence and both class probabilities |
| Pinned dependencies | `requirements.txt`, `requirements-serve.txt`, `requirements-dev.txt` — every version pinned |
| Dockerfile | multi-stage-friendly, non-root user (uid 10001), `HEALTHCHECK`, layer-cached deps |

```bash
docker build -t cats-vs-dogs-mlops:local .
docker run --rm -p 8000:8000 cats-vs-dogs-mlops:local

curl http://localhost:8000/health
curl -X POST -F "file=@data/processed/test/dogs/<any>.jpg" http://localhost:8000/predict
```

Example response:

```json
{
  "predicted_label": "dog",
  "confidence": 0.973,
  "probabilities": {"cat": 0.027, "dog": 0.973},
  "threshold": 0.5,
  "inference_time_ms": 64.2,
  "model_version": "model.keras",
  "filename": "dog_0042.jpg"
}
```

### M3 — CI: build, test, image creation

`.github/workflows/ci-cd.yml`, job `lint-and-test` → `train` → `build-and-push`.
On **every push and pull request** it checks out, installs dependencies, lints
with flake8, runs pytest, retrains the model, then builds the image.

Tests (`make test`):

* `tests/test_preprocess.py` — the pre-processing functions: RGB conversion and
  resize, split-ratio arithmetic, the full split producing 80/10/10, and
  determinism under a fixed seed.
* `tests/test_inference.py` — the inference utilities: `preprocess_image`
  (shape/dtype/garbage input) and `decode_prediction` (label boundaries,
  probability validation), plus `ModelService` against a stub model.
* `tests/test_api.py` — endpoint contracts: health, readiness, prediction
  payload, 400 on a non-image, 503 when the model is missing, and that the
  Prometheus counters appear on `/metrics`.

Artifact publishing: the image is pushed to **GitHub Container Registry**
(`ghcr.io/<owner>/<repo>`) tagged with both `latest` and the immutable commit
SHA. GHCR uses the built-in `GITHUB_TOKEN`, so no registry secrets are needed.

Before the image is allowed to progress, CI starts the container and runs the
smoke test against it — a broken image never reaches the deploy stage.

### M4 — CD & deployment

Deployment target: **Kubernetes**. The `deploy` job (main branch only) creates
an ephemeral `kind` cluster inside the runner, loads the published image, and
applies the manifests in `k8s/`:

* `Namespace`, `Deployment` (2 replicas, rolling update with `maxUnavailable: 0`,
  liveness/readiness/startup probes, resource requests and limits, non-root
  security context), `Service`, and an `HorizontalPodAutoscaler`.
* The image tag is rewritten to the commit SHA at deploy time, so every
  deployment is traceable to a commit and rollback is `kubectl rollout undo`.

The same manifests deploy to minikube on a laptop:

```bash
./scripts/run_local_minikube.sh          # Linux/macOS/Git Bash
.\scripts\run_local_minikube.ps1         # Windows PowerShell
```

**Smoke test (`scripts/smoke_test.sh`)** runs after every deployment and fails
the pipeline on any problem. It checks `/health`, verifies the payload, waits
for `/ready` (i.e. the model artifact really loaded), posts a **real image** to
`/predict` and validates the response fields, then confirms the Prometheus
counters are being exported.

### M5 — Monitoring, logging & post-deployment tracking

*Logging* — a FastAPI middleware logs method, path, status, duration, client
host and a request id for every request; `/predict` additionally logs the
predicted label, confidence and latency. **Image bytes are never logged.**

*Metrics* — `prometheus-fastapi-instrumentator` exposes standard HTTP metrics on
`/metrics`, plus custom counters and histograms:

| Metric | Meaning |
|---|---|
| `predictions_total{label}` | predictions served per class |
| `prediction_errors_total{reason}` | failed predictions by cause |
| `prediction_latency_seconds` | end-to-end `/predict` latency histogram |
| `prediction_confidence` | confidence distribution — a drift early-warning |

Prometheus scrapes the pods (annotation-based discovery) and Grafana ships with
a provisioned dashboard: request rate, error rate, p50/p95/p99 latency,
predictions by label and mean confidence.

*Post-deployment model performance* — `src/monitoring/performance_check.py`
replays a labelled batch of held-out test images **against the deployed
endpoint** (not the in-process model), scores the responses against the true
labels, and writes `artifacts/post_deployment_report.json` with accuracy,
precision, recall, F1, the confusion matrix and latency percentiles. Results are
also logged to MLflow as a `post_deployment_monitoring` run, so live quality can
be tracked release over release. In CI it runs with `--fail-under`, which turns
live accuracy into a release gate.

```bash
python -m src.monitoring.performance_check --url http://localhost:8000 --sample-size 40
```

---

## 4. Auto testing & auto approval

The chain is fully automatic — there is no manual gate:

1. push a branch or open a PR → `MLOps CI/CD` runs lint, tests, training and the
   image build;
2. `pr-auto-merge.yml` approves the PR and enables auto-merge — GitHub completes
   the merge **only** once the required checks pass;
3. the merge to `main` triggers the deploy job → kind cluster → smoke test →
   post-deployment performance gate;
4. any failure anywhere stops the release.

Two one-time repository settings are needed — see `docs/GITHUB_SETUP.md`.

---

## 5. Reproducing the pipeline with DVC

```bash
./scripts/setup_dvc.sh        # dvc init + a local remote
dvc repro                     # download -> preprocess -> train
dvc metrics show              # reads artifacts/metrics.json
dvc plots show                # confusion matrix + training curves
dvc push                      # push data and model to the remote
git add dvc.lock params.yaml && git commit -m "chore: pipeline run"
```

DVC hashes each stage's dependencies, so changing `params.yaml` re-runs only the
stages that actually depend on the changed value.

---

## 6. Configuration

Everything is in `params.yaml`. For CI-sized runs any value can be overridden by
an environment variable without editing the file:

| Variable | Overrides |
|---|---|
| `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `ARCHITECTURE` | `train.*` |
| `MAX_IMAGES_PER_CLASS`, `IMAGE_SIZE` | `data.*` |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` | `mlflow.*` |
| `MODEL_PATH` | `serving.model_path` |
| `SAMPLE_SIZE`, `MIN_ACCURACY` | `monitoring.*` |

---

## 7. Docs

* `docs/ARCHITECTURE.md` — components, data flow, design decisions
* `docs/GITHUB_SETUP.md` — repo settings for GHCR, auto-merge and branch protection
* `docs/DEMO_SCRIPT.md` — a 5-minute screen-recording script
* `docs/SUBMISSION_CHECKLIST.md` — requirement → evidence mapping
