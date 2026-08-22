"""API contract tests for the inference service (M2/M3)."""
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving import app as app_module
from tests.conftest import make_image_bytes


class StubModel:
    def __init__(self, probability=0.91):
        self.probability = probability

    def predict(self, batch, verbose=0):
        return np.array([[self.probability]])


@pytest.fixture
def client():
    app_module.model_service._model = StubModel()
    with TestClient(app_module.app) as test_client:
        yield test_client
    app_module.model_service._model = None


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint_reports_loaded_model(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_info_lists_endpoints(client):
    body = client.get("/info").json()
    assert "/predict" in body["endpoints"]
    assert body["model_loaded"] is True


def test_root_serves_the_upload_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    # the page must be able to drive the API on its own
    assert "/predict" in page
    assert "/ready" in page
    # and must not fetch anything external - a cluster pod may have no egress.
    # (an xmlns="http://www.w3.org/..." namespace is an identifier, not a fetch)
    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://[^"\']+', page)
    assert external == [], f"page references external resources: {external}"
    assert "url(http" not in page


def test_predict_returns_label_and_probabilities(client):
    response = client.post(
        "/predict", files={"file": ("cat.jpg", make_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "dog"
    assert body["probabilities"]["dog"] == pytest.approx(0.91)
    assert body["probabilities"]["cat"] == pytest.approx(0.09)
    assert body["confidence"] == pytest.approx(0.91)
    assert body["filename"] == "cat.jpg"


def test_predict_rejects_non_image_upload(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_predict_returns_503_when_model_missing(client):
    app_module.model_service._model = None
    app_module.model_service.model_path = app_module.model_service.model_path.with_name(
        "missing.keras"
    )
    response = client.post(
        "/predict", files={"file": ("cat.jpg", make_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_counters(client):
    client.post("/predict", files={"file": ("d.jpg", make_image_bytes(), "image/jpeg")})
    body = client.get("/metrics").text
    assert "predictions_total" in body
    assert "prediction_latency_seconds" in body
