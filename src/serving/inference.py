"""Model loading and inference utilities used by the REST API.

Kept separate from the FastAPI app so the pure functions here can be unit
tested without spinning up a server or loading a real model.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import get_config

logger = logging.getLogger("cats_vs_dogs.inference")


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


def preprocess_image(image_bytes: bytes, image_size: int = 224) -> np.ndarray:
    """Decode raw bytes into a batched, model-ready array.

    The model itself owns normalisation (see src/models/model.py), so this only
    has to decode -> RGB -> resize -> batch. Returns float32, range 0-255,
    shape (1, image_size, image_size, 3).
    """
    if not image_bytes:
        raise InvalidImageError("empty request body")
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB").resize((image_size, image_size))
            array = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"could not decode image: {exc}") from exc

    if array.shape != (image_size, image_size, 3):
        raise InvalidImageError(f"unexpected image shape {array.shape}")
    return np.expand_dims(array, axis=0)


def decode_prediction(
    probability: float,
    class_names: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Turn a single sigmoid output into a labelled prediction payload.

    The model outputs P(dog); index 0 of `class_names` is the negative class.
    """
    class_names = class_names or ["cat", "dog"]
    if len(class_names) != 2:
        raise ValueError("decode_prediction expects exactly two class names")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    positive = probability > threshold
    label = class_names[1] if positive else class_names[0]
    confidence = probability if positive else 1.0 - probability
    return {
        "predicted_label": label,
        "confidence": round(confidence, 6),
        "probabilities": {
            class_names[0]: round(1.0 - probability, 6),
            class_names[1]: round(probability, 6),
        },
        "threshold": threshold,
    }


class ModelService:
    """Lazily loads the Keras model and serves predictions."""

    def __init__(self, model_path: str | Path | None = None, cfg=None) -> None:
        self.cfg = cfg or get_config()
        self.model_path = Path(model_path) if model_path else self.cfg.resolve(self.cfg.model_path)
        self.image_size = self.cfg.image_size
        self.class_names = self.cfg.class_names
        self.threshold = self.cfg.threshold
        self._model = None
        self.load_error: str | None = None

    # -- lifecycle ------------------------------------------------------
    def load(self) -> bool:
        if self._model is not None:
            return True
        candidates = [self.model_path]
        if self.model_path.suffix == ".keras":
            candidates.append(self.model_path.with_suffix(".h5"))
        for path in candidates:
            if not path.exists():
                continue
            try:
                import tensorflow as tf  # imported lazily to keep tests light

                started = time.perf_counter()
                self._model = tf.keras.models.load_model(path)
                self.model_path = path
                self.load_error = None
                logger.info(
                    "model loaded from %s in %.2fs", path, time.perf_counter() - started
                )
                return True
            except Exception as exc:  # pragma: no cover - depends on artifact
                self.load_error = f"failed to load {path}: {exc}"
                logger.error(self.load_error)
        if self._model is None and self.load_error is None:
            self.load_error = f"model file not found at {self.model_path}"
            logger.error(self.load_error)
        return False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # -- inference ------------------------------------------------------
    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.is_ready and not self.load():
            raise RuntimeError(self.load_error or "model unavailable")
        batch = preprocess_image(image_bytes, self.image_size)
        started = time.perf_counter()
        probability = float(np.asarray(self._model.predict(batch, verbose=0)).ravel()[0])
        latency_ms = (time.perf_counter() - started) * 1000
        payload = decode_prediction(probability, self.class_names, self.threshold)
        payload["inference_time_ms"] = round(latency_ms, 2)
        payload["model_version"] = self.model_path.name
        return payload
