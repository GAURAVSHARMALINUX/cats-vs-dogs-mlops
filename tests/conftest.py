import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_image_bytes(size=(300, 200), color=(120, 90, 60), fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def raw_dataset(tmp_path):
    """A miniature raw dataset: 10 cats + 10 dogs at an odd resolution."""
    raw_dir = tmp_path / "raw"
    for class_name, color in (("cats", (200, 160, 120)), ("dogs", (90, 110, 160))):
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (300, 200), color).save(class_dir / f"{class_name}_{i}.jpg")
    return raw_dir
