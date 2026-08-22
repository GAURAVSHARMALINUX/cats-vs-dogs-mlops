# 5-minute demo recording script

Target: show the complete workflow from a code change to a deployed model
prediction. Keep it under 5 minutes; rehearse once so the pauses are short.

## Before you hit record

```bash
make data && make train          # model artifact exists
docker compose up -d --build     # MLflow, Prometheus, Grafana warm
minikube start --cpus=2 --memory=4096
```

Have these tabs open: the repo, the Actions tab, MLflow, Grafana, and a
terminal.

---

### 0:00 – 0:30 — What this is

"End-to-end MLOps pipeline for Cats vs Dogs binary classification. Five stages:
versioned data and tracked experiments, a containerised FastAPI service,
CI with automated tests, CD onto Kubernetes gated by a smoke test, and
monitoring including post-deployment accuracy."

Show the repo tree briefly.

### 0:30 – 1:15 — M1: data, training, tracking

```bash
cat params.yaml | head -25
dvc dag                      # download -> preprocess -> train
```

Open MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`): show the two
runs (`baseline_cnn` vs `mobilenetv2`), the logged parameters and metrics, and
open `confusion_matrix.png` and `training_curves.png` in the artifacts tab.

### 1:15 – 2:00 — M2: the containerised service

```bash
docker compose ps
curl -s localhost:8000/health
curl -s localhost:8000/ready
curl -s -X POST -F "file=@data/processed/test/dogs/<file>.jpg" \
     localhost:8000/predict | python -m json.tool
```

Show `/docs` (Swagger) and upload an image through the UI so the prediction is
visible on screen.

### 2:00 – 2:30 — M3: tests and CI

```bash
make test                    # 23 tests pass
```

Point at `.github/workflows/ci-cd.yml`: lint → test → train → build → push to
GHCR with a commit-SHA tag.

### 2:30 – 3:45 — The change → deployment loop

```bash
git checkout -b feature/demo
# edit something visible, e.g. APP_VERSION in k8s/base/deployment.yaml
git commit -am "feat: demo change" && git push -u origin feature/demo
gh pr create --fill
```

Switch to the Actions tab. Narrate: tests pass → the bot approves the PR →
auto-merge lands it on `main` → the deploy job creates a kind cluster, loads the
image and applies the manifests → the smoke test runs.

Open the deploy job's log and show the smoke test output — health, readiness and
a real prediction — and the post-deployment performance report in the job
summary.

### 3:45 – 4:30 — M4 locally on Kubernetes

```bash
kubectl -n mlops get pods,svc
kubectl -n mlops rollout history deployment/cats-vs-dogs-api
bash scripts/smoke_test.sh $(minikube service cats-vs-dogs-api -n mlops --url)
```

### 4:30 – 5:00 — M5: monitoring

Open Grafana: request rate, p95 latency, predictions by label, mean confidence.
Then run:

```bash
python -m src.monitoring.performance_check --url <service-url> --sample-size 40
```

Show the accuracy/F1/confusion matrix printed from **live endpoint responses**
scored against true labels, and mention that CI runs the same check with
`--fail-under`, so a model that degrades blocks the release.

Close with one line: "Code change to deployed, smoke-tested, monitored model —
fully automated, no manual approval step."
