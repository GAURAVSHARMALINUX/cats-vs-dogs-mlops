# Architecture

## Components

| Component | Technology | Role |
|---|---|---|
| Data pipeline | DVC + Pillow | download → preprocess → versioned splits |
| Training | TensorFlow / Keras 3 | baseline CNN and MobileNetV2 transfer learning |
| Experiment tracking | MLflow (SQLite backend) | params, metrics, plots, model artifacts |
| Inference API | FastAPI + Uvicorn | `/health`, `/ready`, `/predict`, `/metrics` |
| Packaging | Docker | slim, non-root runtime image |
| Registry | GitHub Container Registry | `latest` + immutable commit-SHA tags |
| CI/CD | GitHub Actions | lint → test → train → build → deploy → smoke |
| Orchestration | Kubernetes (kind in CI, minikube locally) | Deployment, Service, HPA |
| Monitoring | Prometheus + Grafana | HTTP and business metrics, dashboards |
| Post-deploy evaluation | custom script + MLflow | live accuracy against true labels |

## Data flow

```
Kaggle-derived dataset (cats_and_dogs_filtered)
        │  src/data/download.py           [DVC stage: download]
        ▼
data/raw/{cats,dogs}
        │  src/data/preprocess.py         [DVC stage: preprocess]
        │  RGB, 224x224, seeded 80/10/10
        ▼
data/processed/{train,val,test}/{cats,dogs}
        │  src/models/train.py            [DVC stage: train]
        │  augmentation on train split only
        │  MLflow ← params, metrics, curves, confusion matrix
        ▼
artifacts/model.keras (+ legacy model.h5)
        │  Dockerfile
        ▼
ghcr.io/<owner>/<repo>:<commit-sha>
        │  scripts/deploy_kind.sh
        ▼
Kubernetes Deployment (2 replicas, HPA 2→5)
        │
        ├── scripts/smoke_test.sh                 → gate the release
        ├── src/monitoring/performance_check.py   → live accuracy → MLflow
        └── /metrics → Prometheus → Grafana
```

## Design decisions

**Normalisation lives inside the model.** The `Rescaling` layer is the first
layer of the saved graph, so the serving path only has to decode and resize.
Train/serve skew becomes structurally impossible rather than a convention
someone has to remember.

**Separate `/health` and `/ready`.** Liveness answers "is the process alive",
readiness answers "is the model loaded and able to serve". Kubernetes needs
both: restarting a pod whose model file is missing would not help, but it must
be kept out of the Service endpoints. It also gives the smoke test a precise
signal to wait on.

**Two architectures, one training script.** The assignment asks for a baseline;
a frozen MobileNetV2 backbone is what a production team would actually ship.
Keeping both behind `params.yaml: train.architecture` gives MLflow two runs to
compare and keeps the baseline honest.

**Immutable image tags.** Deployments reference `:<commit-sha>`, never `:latest`,
so what is running is always traceable to a commit and `kubectl rollout undo`
is a real rollback.

**kind inside CI instead of a cloud cluster.** The assignment permits a local
cluster. Running kind inside the runner means the CD path is genuinely executed
and verified on every merge, at zero cost and with no cloud credentials in the
repository — and the identical manifests run on minikube for the demo.

**Separate serving requirements.** The runtime image installs
`requirements-serve.txt` — no MLflow, DVC or matplotlib — which keeps the
attack surface and the image smaller than the training environment.
