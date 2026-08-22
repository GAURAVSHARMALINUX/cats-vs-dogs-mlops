"""Unit tests for the model/inference utilities (M3.1)."""
import numpy as np
import pytest

from src.serving.inference import (
    InvalidImageError,
    ModelService,
    decode_prediction,
    preprocess_image,
)
from tests.conftest import make_image_bytes


def test_preprocess_image_returns_batched_224_rgb(image_bytes):
    batch = preprocess_image(image_bytes, image_size=224)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 255.0


def test_preprocess_image_handles_png_and_greyscale():
    batch = preprocess_image(make_image_bytes(size=(50, 90), fmt="PNG"), image_size=128)
    assert batch.shape == (1, 128, 128, 3)


def test_preprocess_image_rejects_garbage():
    with pytest.raises(InvalidImageError):
        preprocess_image(b"this is definitely not an image")
    with pytest.raises(InvalidImageError):
        preprocess_image(b"")


@pytest.mark.parametrize(
    "probability,expected_label",
    [(0.99, "dog"), (0.51, "dog"), (0.5, "cat"), (0.02, "cat")],
)
def test_decode_prediction_labels(probability, expected_label):
    result = decode_prediction(probability, ["cat", "dog"], threshold=0.5)
    assert result["predicted_label"] == expected_label
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["probabilities"]["cat"] + result["probabilities"]["dog"] == pytest.approx(1.0)


def test_decode_prediction_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        decode_prediction(1.7)


def test_model_service_predicts_with_a_stubbed_model(image_bytes):
    class StubModel:
        def predict(self, batch, verbose=0):
            assert batch.shape == (1, 224, 224, 3)
            return np.array([[0.87]])

    service = ModelService()
    service._model = StubModel()
    result = service.predict(image_bytes)

    assert result["predicted_label"] == "dog"
    assert result["confidence"] == pytest.approx(0.87)
    assert "inference_time_ms" in result


def test_model_service_reports_missing_artifact(tmp_path):
    service = ModelService(model_path=tmp_path / "does_not_exist.keras")
    assert service.load() is False
    assert service.is_ready is False
    assert "not found" in (service.load_error or "")
