"""Unit tests for the data pre-processing functions (M3.1)."""
from pathlib import Path

import pytest
from PIL import Image

from src.data.preprocess import preprocess_and_split, resize_image, split_indices


def test_resize_image_produces_224_rgb():
    img = Image.new("L", (640, 480))  # greyscale, wrong size
    out = resize_image(img, 224)
    assert out.mode == "RGB"
    assert out.size == (224, 224)


def test_resize_image_rejects_bad_size():
    with pytest.raises(ValueError):
        resize_image(Image.new("RGB", (10, 10)), 0)


def test_split_indices_cover_every_item_without_overlap():
    ranges = split_indices(100, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert ranges["train"] == (0, 80)
    assert ranges["val"] == (80, 90)
    assert ranges["test"] == (90, 100)
    covered = sum(end - start for start, end in ranges.values())
    assert covered == 100


def test_split_indices_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        split_indices(10, {"train": 0.7, "val": 0.1, "test": 0.1})


def test_preprocess_and_split_creates_resized_80_10_10_split(raw_dataset, tmp_path):
    processed = tmp_path / "processed"
    summary = preprocess_and_split(
        raw_dir=raw_dataset,
        processed_dir=processed,
        image_size=224,
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42,
    )

    for split in ("train", "val", "test"):
        for class_name in ("cats", "dogs"):
            assert (processed / split / class_name).is_dir()

    # 10 images per class -> 8 / 1 / 1
    assert summary["train"] == {"cats": 8, "dogs": 8}
    assert summary["val"] == {"cats": 1, "dogs": 1}
    assert summary["test"] == {"cats": 1, "dogs": 1}

    written = list((processed / "train" / "cats").glob("*.jpg"))
    assert written, "expected preprocessed training images"
    with Image.open(written[0]) as img:
        assert img.size == (224, 224)
        assert img.mode == "RGB"


def test_preprocess_is_deterministic_for_a_fixed_seed(raw_dataset, tmp_path):
    def run(target: Path):
        preprocess_and_split(raw_dataset, target, 64, {"train": 0.8, "val": 0.1, "test": 0.1}, 7)
        return sorted(p.name for p in (target / "test" / "cats").glob("*.jpg"))

    assert run(tmp_path / "a") == run(tmp_path / "b")
