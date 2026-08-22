"""Post-deployment model performance tracking (M5.2).

Replays a labelled batch of held-out test images against the *deployed*
endpoint - not the in-process model - and scores the responses against the
true labels. Results are written to a JSON report and logged to MLflow as a
separate `post_deployment_monitoring` run so live quality can be compared with
training-time quality over releases.

Usage:
    python -m src.monitoring.performance_check --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import get_config

CLASS_DIRS = {"cats": 0, "dogs": 1}


def collect_samples(test_dir: Path, sample_size: int, seed: int = 42) -> list[tuple[Path, int]]:
    """Pick a balanced, seeded sample of (image_path, true_label) pairs."""
    rng = random.Random(seed)
    per_class = max(1, sample_size // len(CLASS_DIRS))
    samples: list[tuple[Path, int]] = []
    for class_dir, label in CLASS_DIRS.items():
        images = sorted((test_dir / class_dir).glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no test images in {test_dir / class_dir}")
        rng.shuffle(images)
        samples.extend((img, label) for img in images[:per_class])
    rng.shuffle(samples)
    return samples


def score(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["cat", "dog"],
    }


def run_check(
    base_url: str,
    test_dir: Path,
    sample_size: int,
    class_names: list[str],
    timeout: int = 30,
    seed: int = 42,
) -> dict:
    samples = collect_samples(test_dir, sample_size, seed)
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    y_true, y_pred, latencies, confidences, failures = [], [], [], [], []
    for image_path, true_label in samples:
        started = time.perf_counter()
        try:
            with open(image_path, "rb") as fh:
                response = requests.post(
                    f"{base_url.rstrip('/')}/predict",
                    files={"file": (image_path.name, fh, "image/jpeg")},
                    timeout=timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            failures.append({"file": image_path.name, "error": str(exc)})
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        y_true.append(true_label)
        y_pred.append(label_to_index[payload["predicted_label"]])
        confidences.append(float(payload["confidence"]))

    if not y_true:
        raise RuntimeError(f"every request failed; first error: {failures[:1]}")

    latencies.sort()
    metrics = score(y_true, y_pred)
    metrics.update(
        {
            "samples_requested": len(samples),
            "samples_scored": len(y_true),
            "failed_requests": len(failures),
            "mean_confidence": sum(confidences) / len(confidences),
            "latency_ms_mean": sum(latencies) / len(latencies),
            "latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))],
            "latency_ms_max": latencies[-1],
        }
    )
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": base_url,
        "metrics": metrics,
        "failures": failures[:10],
    }


def log_to_mlflow(report: dict, cfg) -> None:
    try:
        import mlflow

        from src.config import mlflow_artifact_location, resolve_tracking_uri

        mlflow.set_tracking_uri(resolve_tracking_uri(cfg))
        experiment_name = cfg.get("mlflow.experiment_name", "cats_vs_dogs")
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name, artifact_location=mlflow_artifact_location(cfg)
            )
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="post_deployment_monitoring"):
            mlflow.set_tag("stage", "post_deployment_monitoring")
            mlflow.set_tag("endpoint", report["endpoint"])
            mlflow.log_params(
                {
                    "samples_requested": report["metrics"]["samples_requested"],
                    "checked_at": report["checked_at"],
                }
            )
            mlflow.log_metrics(
                {
                    f"live_{key}": value
                    for key, value in report["metrics"].items()
                    if isinstance(value, (int, float))
                }
            )
        print("[monitor] logged results to MLflow")
    except Exception as exc:  # monitoring must never break the pipeline silently
        print(f"[monitor] warning: could not log to MLflow: {exc}")


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Post-deployment performance check")
    parser.add_argument("--url", default="http://localhost:8000", help="deployed service base URL")
    parser.add_argument(
        "--sample-size", type=int, default=int(cfg.get("monitoring.sample_size", 40))
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=float(cfg.get("monitoring.min_accuracy", 0.8))
    )
    parser.add_argument("--report", default=str(cfg.get("monitoring.report_path")))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--fail-under",
        action="store_true",
        help="exit non-zero if live accuracy drops below --min-accuracy",
    )
    args = parser.parse_args(argv)

    test_dir = cfg.resolve(cfg.get("data.processed_dir", "data/processed")) / "test"
    report = run_check(
        base_url=args.url,
        test_dir=test_dir,
        sample_size=args.sample_size,
        class_names=cfg.class_names,
        seed=int(cfg.get("data.seed", 42)),
    )
    report["min_accuracy"] = args.min_accuracy
    report["passed"] = report["metrics"]["accuracy"] >= args.min_accuracy

    report_path = cfg.resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["metrics"], indent=2))
    print(f"[monitor] report written to {report_path}")
    print(
        f"[monitor] live accuracy {report['metrics']['accuracy']:.3f} "
        f"(threshold {args.min_accuracy:.2f}) -> "
        f"{'PASS' if report['passed'] else 'BELOW THRESHOLD'}"
    )

    if not args.no_mlflow:
        log_to_mlflow(report, cfg)

    if args.fail_under and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
