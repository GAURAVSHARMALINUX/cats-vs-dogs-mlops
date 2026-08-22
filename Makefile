# Convenience targets. Every one of these is also a step in the CI/CD pipeline.
.PHONY: help setup data download preprocess train test lint serve smoke monitor \
        docker-build docker-run compose-up compose-down k8s-deploy k8s-delete \
        mlflow-ui dvc-setup repro clean

PY ?= python
URL ?= http://localhost:8000
IMAGE ?= cats-vs-dogs-mlops:local

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## install the full dev environment
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

download:  ## fetch the raw dataset
	$(PY) -m src.data.download

preprocess:  ## 224x224 RGB + seeded 80/10/10 split
	$(PY) -m src.data.preprocess

data: download preprocess  ## download + preprocess

train:  ## train and log the run to MLflow
	$(PY) -m src.models.train

test:  ## run the unit + API test suite
	$(PY) -m pytest tests/ -v

lint:  ## flake8
	$(PY) -m flake8 src tests --count --statistics

serve:  ## run the API locally
	$(PY) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

smoke:  ## post-deploy smoke test (health + readiness + real prediction)
	bash scripts/smoke_test.sh $(URL)

monitor:  ## post-deployment performance check against a live endpoint
	$(PY) -m src.monitoring.performance_check --url $(URL)

docker-build:  ## build the inference image
	docker build -t $(IMAGE) .

docker-run:  ## run the inference image
	docker run --rm -p 8000:8000 $(IMAGE)

compose-up:  ## API + MLflow + Prometheus + Grafana
	docker compose up --build -d

compose-down:
	docker compose down -v

k8s-deploy:  ## apply the manifests to the current kube context
	bash scripts/deploy_kind.sh $(IMAGE) mlops

k8s-delete:
	kubectl delete namespace mlops --ignore-not-found

mlflow-ui:  ## open the experiment tracking UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db

dvc-setup:  ## initialise DVC with a local remote
	bash scripts/setup_dvc.sh

repro:  ## run the whole DVC pipeline
	dvc repro

clean:  ## remove generated files (keeps data/)
	rm -rf artifacts/*.png artifacts/*.json artifacts/*.txt artifacts/model.* \
	       .pytest_cache test-report.xml coverage.xml .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
