"""Stage 2 of the DVC pipeline: train, evaluate and track the model.

Everything the run depends on comes from params.yaml (see src/config.py), and
everything the run produces - parameters, metrics, loss curves, confusion
matrix and the serialised model - is logged to MLflow.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import get_config, mlflow_artifact_location, resolve_tracking_uri
from src.models.model import build_augmenter, build_model, count_parameters
from src.utils_plots import plot_confusion_matrix, plot_history

AUTOTUNE = tf.data.AUTOTUNE


def load_split(
    processed_dir: Path, split: str, image_size: int, batch_size: int, seed: int, shuffle: bool
) -> tf.data.Dataset:
    """Load one preprocessed split as a tf.data pipeline."""
    directory = Path(processed_dir) / split
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} not found - run `python -m src.data.preprocess` (or `dvc repro`) first"
        )
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["cats", "dogs"],  # index 0 = cat, 1 = dog
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def evaluate(model: keras.Model, dataset: tf.data.Dataset, threshold: float):
    """Return (y_true, y_pred, y_prob) over a dataset."""
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().ravel().astype(int).tolist())
        y_prob.extend(probs.ravel().astype(float).tolist())
    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr > threshold).astype(int)
    return y_true_arr, y_pred_arr, y_prob_arr


def train(cfg=None) -> dict:
    cfg = cfg or get_config()

    image_size = cfg.image_size
    batch_size = int(cfg.get("train.batch_size", 32))
    epochs = int(cfg.get("train.epochs", 3))
    architecture = str(cfg.get("train.architecture", "mobilenetv2"))
    learning_rate = float(cfg.get("train.learning_rate", 1e-4))
    seed = int(cfg.get("data.seed", 42))
    threshold = cfg.threshold
    class_names = cfg.class_names

    keras.utils.set_random_seed(seed)

    processed_dir = cfg.resolve(cfg.get("data.processed_dir", "data/processed"))
    artifacts_dir = cfg.resolve("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(processed_dir, "train", image_size, batch_size, seed, shuffle=True)
    val_ds = load_split(processed_dir, "val", image_size, batch_size, seed, shuffle=False)
    test_ds = load_split(processed_dir, "test", image_size, batch_size, seed, shuffle=False)

    aug_cfg = cfg.get("train.augmentation", {}) or {}
    augmenter = build_augmenter(
        horizontal_flip=bool(aug_cfg.get("horizontal_flip", True)),
        rotation=float(aug_cfg.get("rotation", 0.1)),
        zoom=float(aug_cfg.get("zoom", 0.2)),
        translation=float(aug_cfg.get("translation", 0.1)),
        seed=seed,
    )
    train_ds = train_ds.map(
        lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    model = build_model(architecture, image_size=image_size, learning_rate=learning_rate)
    model.summary()

    # ---------------- MLflow ----------------
    tracking_uri = resolve_tracking_uri(cfg)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = cfg.get("mlflow.experiment_name", "cats_vs_dogs")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location(cfg))
    mlflow.set_experiment(experiment_name)
    print(f"[train] MLflow tracking URI: {tracking_uri}")

    run_name = cfg.get("mlflow.run_name") or f"{architecture}-e{epochs}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "architecture": architecture,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "image_size": image_size,
                "seed": seed,
                "threshold": threshold,
                "max_images_per_class": cfg.get("data.max_images_per_class"),
                **{f"aug_{k}": v for k, v in aug_cfg.items()},
            }
        )
        mlflow.log_params(count_parameters(model))
        mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA", "local"))
        mlflow.set_tag("stage", "training")

        callbacks = []
        patience = int(cfg.get("train.early_stopping_patience", 0) or 0)
        if patience > 0 and epochs > patience:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience, restore_best_weights=True
                )
            )

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2
        )

        for epoch in range(len(history.history["loss"])):
            for key, values in history.history.items():
                mlflow.log_metric(key, float(values[epoch]), step=epoch)

        curves_path = plot_history(history.history, artifacts_dir / "training_curves.png")
        mlflow.log_artifact(str(curves_path))

        # ---------------- held-out test evaluation ----------------
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        y_true, y_pred, _ = evaluate(model, test_ds, threshold)

        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        mlflow.log_metrics(metrics)
        print("[train] test metrics:", json.dumps(metrics, indent=2))

        cm_path = plot_confusion_matrix(
            y_true, y_pred, class_names, artifacts_dir / "confusion_matrix.png",
            title=f"Confusion matrix - {architecture} (test)",
        )
        mlflow.log_artifact(str(cm_path))

        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(
            classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path))

        # ---------------- serialise the model ----------------
        model_path = cfg.resolve(cfg.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"[train] saved model -> {model_path}")

        # legacy .h5 copy as well, since the brief mentions .h5 explicitly
        h5_path = model_path.with_suffix(".h5")
        try:
            model.save(h5_path)
            mlflow.log_artifact(str(h5_path))
        except Exception as exc:  # pragma: no cover - depends on keras version
            print(f"[train] warning: could not write legacy .h5 copy: {exc}")

        mlflow.log_artifact(str(model_path))
        try:
            mlflow.keras.log_model(model, name="model", registered_model_name=None)
        except Exception as exc:  # pragma: no cover - optional flavour
            print(f"[train] warning: mlflow.keras.log_model skipped: {exc}")

        metrics_path = cfg.resolve("artifacts/metrics.json")
        payload = {
            "run_id": run.info.run_id,
            "architecture": architecture,
            "epochs_run": len(history.history["loss"]),
            **metrics,
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))
        print(f"[train] MLflow run: {run.info.run_id}")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Cats vs Dogs classifier")
    parser.add_argument(
        "--architecture", default=None, choices=[None, "baseline_cnn", "mobilenetv2"]
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    if args.architecture:
        os.environ["ARCHITECTURE"] = args.architecture
    if args.epochs:
        os.environ["EPOCHS"] = str(args.epochs)

    train(get_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
