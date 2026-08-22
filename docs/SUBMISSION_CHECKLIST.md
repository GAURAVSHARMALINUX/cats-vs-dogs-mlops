# Requirement → evidence mapping

## M1 — Model development & experiment tracking (10M)

| Task | Evidence |
|---|---|
| Git source versioning | repository history; project structure under `src/` |
| DVC dataset + preprocessed-data versioning | `dvc.yaml` (download/preprocess/train stages), `dvc.lock`, `scripts/setup_dvc.sh` |
| Baseline model | `src/models/model.py::build_baseline_cnn` (simple CNN from scratch) |
| Serialised model | `artifacts/model.keras` and legacy `artifacts/model.h5` |
| Experiment tracking | MLflow runs: params, per-epoch loss/accuracy, test metrics |
| Artifacts: confusion matrix, loss curves | `artifacts/confusion_matrix.png`, `artifacts/training_curves.png` |

## M2 — Packaging & containerisation (10M)

| Task | Evidence |
|---|---|
| REST API | `src/serving/app.py` (FastAPI) |
| Health check endpoint | `GET /health`, plus `GET /ready` for model readiness |
| Prediction endpoint | `POST /predict` → label + confidence + class probabilities |
| requirements.txt | `requirements.txt`, `requirements-serve.txt`, `requirements-dev.txt` |
| Version pinning | every dependency pinned to an exact version |
| Dockerfile | `Dockerfile` — non-root, healthcheck, cached dependency layer |
| Verified locally via curl | `scripts/smoke_test.sh`, `make docker-run` + curl examples in README |

## M3 — CI: build, test, image creation (10M)

| Task | Evidence |
|---|---|
| Unit test for a preprocessing function | `tests/test_preprocess.py` (5 tests) |
| Unit test for a model/inference utility | `tests/test_inference.py` (7 tests) |
| Tests run via pytest | `make test`, CI job `lint-and-test` |
| CI on every push/PR | `.github/workflows/ci-cd.yml` |
| Checkout → install → test → build image | jobs `lint-and-test` → `train` → `build-and-push` |
| Push image to a registry | GHCR, tagged `latest` + commit SHA |

## M4 — CD & deployment (10M)

| Task | Evidence |
|---|---|
| Deployment target | Kubernetes — kind in CI, minikube locally |
| Deployment + Service YAML | `k8s/base/deployment.yaml`, `k8s/base/service.yaml` (+ `hpa.yaml`, `namespace.yaml`) |
| CD flow: pull image, deploy on main | `deploy` job in `ci-cd.yml`, `scripts/deploy_kind.sh` |
| Post-deploy smoke test | `scripts/smoke_test.sh` — health, readiness, real prediction, metrics |
| Pipeline fails if smoke tests fail | the smoke-test step is a required step in the `deploy` job |

## M5 — Monitoring, logs & submission (10M)

| Task | Evidence |
|---|---|
| Request/response logging (no sensitive data) | middleware in `src/serving/app.py`; image bytes never logged |
| Request count and latency | `predictions_total`, `prediction_latency_seconds`, standard HTTP metrics on `/metrics` |
| Prometheus / Grafana | `monitoring/`, `k8s/monitoring/`, provisioned dashboard |
| Post-deployment performance on labelled requests | `src/monitoring/performance_check.py` → `artifacts/post_deployment_report.json` + MLflow run |

## Deliverables

- [ ] Zip containing source code, DVC/CI-CD/Docker/K8s configuration and the trained model artifacts
- [ ] Screen recording under 5 minutes (`docs/DEMO_SCRIPT.md` is the script)

## Before zipping

```bash
make data && make train      # produces artifacts/model.keras + model.h5 + plots
make test                    # all tests green
make clean                   # drop caches
```

Include `artifacts/` (model + plots + metrics) in the zip; exclude `data/`,
`mlruns/` and `.git/` to keep it small.
