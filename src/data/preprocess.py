"""Stage 1 of the DVC pipeline: preprocess and split the raw images.

Every raw image is converted to RGB, resized to `data.image_size` (224x224 by
default) and written into a deterministic train/val/test split (80/10/10).
The split is seeded so the same commit always yields the same split, which is
what makes the DVC stage reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

from src.config import get_config

CLASS_NAMES = ("cats", "dogs")


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    """Convert an image to RGB and resize it to `image_size` x `image_size`.

    Kept as a tiny standalone function because it is the unit under test for
    the preprocessing test case (M3).
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    return image.convert("RGB").resize((image_size, image_size))


def split_indices(n_items: int, splits: dict[str, float]) -> dict[str, tuple[int, int]]:
    """Turn split ratios into (start, end) index ranges covering all items."""
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0, got {total}")
    n_train = int(n_items * splits["train"])
    n_val = int(n_items * splits["val"])
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_items),
    }


def preprocess_and_split(
    raw_dir: str | Path,
    processed_dir: str | Path,
    image_size: int = 224,
    splits: dict[str, float] | None = None,
    seed: int = 42,
    max_images_per_class: int | None = None,
) -> dict[str, dict[str, int]]:
    """Preprocess every raw image and write it into a train/val/test split.

    Returns a per-split, per-class count summary.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    splits = splits or {"train": 0.8, "val": 0.1, "test": 0.1}

    for split in splits:
        for class_name in CLASS_NAMES:
            (processed_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {split: {} for split in splits}

    for class_name in CLASS_NAMES:
        images = sorted((raw_dir / class_name).glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no .jpg images found in {raw_dir / class_name}")

        rng = random.Random(seed)
        rng.shuffle(images)
        if max_images_per_class:
            images = images[:max_images_per_class]

        ranges = split_indices(len(images), splits)
        for split, (start, end) in ranges.items():
            out_dir = processed_dir / split / class_name
            written = 0
            for img_path in images[start:end]:
                try:
                    with Image.open(img_path) as img:
                        resize_image(img, image_size).save(out_dir / img_path.name, "JPEG")
                    written += 1
                except Exception as exc:  # corrupt files exist in this dataset
                    print(f"[preprocess] skipping {img_path.name}: {exc}")
            summary[split][class_name] = written
            print(f"[preprocess] {class_name} -> {split}: {written} images")

    (processed_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess and split Cats vs Dogs images")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    args = parser.parse_args(argv)

    cfg = get_config()
    raw_dir = args.raw_dir or cfg.resolve(cfg.get("data.raw_dir", "data/raw"))
    processed_dir = args.processed_dir or cfg.resolve(
        cfg.get("data.processed_dir", "data/processed")
    )

    summary = preprocess_and_split(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        image_size=cfg.image_size,
        splits=cfg.get("data.splits"),
        seed=int(cfg.get("data.seed", 42)),
        max_images_per_class=cfg.get("data.max_images_per_class"),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
