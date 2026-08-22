"""Central configuration loader.

Reads params.yaml and lets environment variables override the few knobs the
CI pipeline needs to tune (smaller/faster runs) without editing the file.
This keeps a single source of truth for parameters while staying CI-friendly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "params.yaml"))

# env var -> (dotted path in params.yaml, caster)
_ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "EPOCHS": ("train.epochs", int),
    "BATCH_SIZE": ("train.batch_size", int),
    "ARCHITECTURE": ("train.architecture", str),
    "LEARNING_RATE": ("train.learning_rate", float),
    "MAX_IMAGES_PER_CLASS": ("data.max_images_per_class", int),
    "IMAGE_SIZE": ("data.image_size", int),
    "MLFLOW_TRACKING_URI": ("mlflow.tracking_uri", str),
    "MLFLOW_EXPERIMENT_NAME": ("mlflow.experiment_name", str),
    "MLFLOW_RUN_NAME": ("mlflow.run_name", str),
    "MODEL_PATH": ("serving.model_path", str),
    "SAMPLE_SIZE": ("monitoring.sample_size", int),
    "MIN_ACCURACY": ("monitoring.min_accuracy", float),
}


def _set_in(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        tree = tree.setdefault(key, {})
    tree[keys[-1]] = value


def _get_in(tree: dict, dotted: str, default: Any = None) -> Any:
    for key in dotted.split("."):
        if not isinstance(tree, dict) or key not in tree:
            return default
        tree = tree[key]
    return tree


def load_params(path: Path | str | None = None) -> dict:
    """Load params.yaml and apply environment overrides."""
    path = Path(path) if path else PARAMS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}

    for env_key, (dotted, cast) in _ENV_OVERRIDES.items():
        raw = os.getenv(env_key)
        if raw is not None and raw != "":
            _set_in(params, dotted, cast(raw))
    return params


class Config:
    """Thin attribute-style wrapper so call sites read nicely."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = params if params is not None else load_params()

    def get(self, dotted: str, default: Any = None) -> Any:
        return _get_in(self.params, dotted, default)

    # --- frequently used shortcuts -------------------------------------
    @property
    def image_size(self) -> int:
        return int(self.get("data.image_size", 224))

    @property
    def img_shape(self) -> tuple[int, int]:
        size = self.image_size
        return (size, size)

    @property
    def class_names(self) -> list[str]:
        return list(self.get("serving.class_names", ["cat", "dog"]))

    @property
    def threshold(self) -> float:
        return float(self.get("serving.threshold", 0.5))

    @property
    def model_path(self) -> str:
        return str(self.get("serving.model_path", "artifacts/model.keras"))

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def get_config() -> Config:
    return Config()


def resolve_tracking_uri(cfg: "Config") -> str:
    """Turn the configured MLflow URI into an absolute, portable one.

    `sqlite:///mlflow.db` and `file:./mlruns` are convenient to write in
    params.yaml but are relative to the current working directory, which
    differs between a local run, `dvc repro` and CI. Anchor them at the
    project root instead.
    """
    uri = str(cfg.get("mlflow.tracking_uri", "sqlite:///mlflow.db"))
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        return f"sqlite:///{PROJECT_ROOT / uri[len('sqlite:///'):]}"
    if uri.startswith("file:./"):
        return f"file:{PROJECT_ROOT / uri[len('file:./'):]}"
    return uri


def mlflow_artifact_location(cfg: "Config") -> str:
    return (PROJECT_ROOT / "mlruns").as_uri()
