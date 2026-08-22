"""FastAPI inference service for the Cats vs Dogs classifier.

Endpoints
    GET  /          upload UI (HTML)
    GET  /info      service metadata
    GET  /health    liveness probe
    GET  /ready     readiness probe (is the model actually loaded?)
    POST /predict   image upload -> label + class probabilities
    GET  /metrics   Prometheus exposition (added by the instrumentator)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import get_config
from src.serving.inference import InvalidImageError, ModelService

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cats_vs_dogs.api")

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
UI_PAGE = Path(__file__).parent / "static" / "index.html"
GIT_SHA = os.getenv("GIT_SHA", "local")

cfg = get_config()
model_service = ModelService(cfg=cfg)

# ---- custom business metrics (on top of the default HTTP metrics) ----
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Predictions served, by predicted label", ["label"]
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Failed prediction requests", ["reason"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "End-to-end latency of /predict"
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence of served predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if model_service.load():
        logger.info("startup: model ready (%s)", model_service.model_path)
    else:
        logger.warning("startup: model NOT loaded - %s", model_service.load_error)
    yield
    logger.info("shutdown complete")


app = FastAPI(
    title="Cats vs Dogs Inference API",
    description="MLOps Assignment 2 - binary image classification service",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Request/response logging.

    Only metadata is logged - method, path, status, duration, client host.
    The uploaded image bytes are never written to the logs.
    """
    request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "rid=%s %s %s -> 500 in %.1fms", request_id, request.method, request.url.path,
            duration_ms,
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "rid=%s %s %s -> %s in %.1fms client=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.client.host if request.client else "-",
    )
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{duration_ms:.1f}"
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Minimal upload UI, served by the API itself.

    Keeping the page inside this service means there is no second container,
    image or manifest to deploy: wherever the API runs, the UI is there too.
    """
    try:
        return HTMLResponse(UI_PAGE.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - only if the asset is missing
        logger.error("could not read the UI page: %s", exc)
        raise HTTPException(status_code=404, detail="UI page not available") from exc


@app.get("/info", tags=["meta"])
def info():
    return {
        "service": "cats-vs-dogs-inference",
        "version": APP_VERSION,
        "git_sha": GIT_SHA,
        "model_loaded": model_service.is_ready,
        "endpoints": ["/", "/info", "/health", "/ready", "/predict", "/metrics", "/docs"],
    }


@app.get("/health", tags=["ops"])
def health():
    """Liveness probe - the process is up and serving."""
    return {"status": "healthy", "version": APP_VERSION}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe - fails until the model artifact is loaded."""
    if not model_service.is_ready and not model_service.load():
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": model_service.load_error},
        )
    return {
        "status": "ready",
        "model": model_service.model_path.name,
        "classes": model_service.class_names,
    }


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(..., description="JPEG/PNG image")):
    started = time.perf_counter()
    content = await file.read()
    try:
        result = model_service.predict(content)
    except InvalidImageError as exc:
        PREDICTION_ERRORS.labels(reason="invalid_image").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        PREDICTION_ERRORS.labels(reason="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    elapsed = time.perf_counter() - started
    PREDICTION_LATENCY.observe(elapsed)
    PREDICTIONS_TOTAL.labels(label=result["predicted_label"]).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    logger.info(
        "prediction filename=%s label=%s confidence=%.4f latency_ms=%.1f",
        file.filename,
        result["predicted_label"],
        result["confidence"],
        elapsed * 1000,
    )
    result["filename"] = file.filename
    return result


Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)
