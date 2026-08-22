"""Stage 0 of the DVC pipeline: fetch the raw Cats vs Dogs images.

Uses the publicly mirrored `cats_and_dogs_filtered` subset of the Kaggle
Dogs vs Cats competition data (1500 images per class). It is small enough to
download inside CI, which keeps the whole pipeline reproducible end-to-end.

Layout produced:
    data/raw/cats/*.jpg
    data/raw/dogs/*.jpg
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.config import get_config

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
# folders inside the archive that hold each class
SOURCE_DIRS = {
    "cats": ["cats_and_dogs_filtered/train/cats", "cats_and_dogs_filtered/validation/cats"],
    "dogs": ["cats_and_dogs_filtered/train/dogs", "cats_and_dogs_filtered/validation/dogs"],
}


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] archive already present: {dest}")
        return dest
    print(f"[download] fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)
    print(f"[download] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_raw_data(raw_dir: Path, url: str = DATA_URL, force: bool = False) -> Path:
    """Download and unpack the dataset into `raw_dir/{cats,dogs}`."""
    raw_dir = Path(raw_dir)
    marker = raw_dir / ".download_complete"
    if marker.exists() and not force:
        print(f"[download] raw data already present at {raw_dir}, skipping")
        return raw_dir

    archive = raw_dir.parent / "cats_and_dogs_filtered.zip"
    _download(url, archive)

    extract_dir = raw_dir.parent / "_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)

    total = 0
    for class_name, sources in SOURCE_DIRS.items():
        out_dir = raw_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            src_path = extract_dir / source
            if not src_path.exists():
                continue
            for img in sorted(src_path.glob("*.jpg")):
                shutil.copy2(img, out_dir / img.name)
                total += 1
        print(f"[download] {class_name}: {len(list(out_dir.glob('*.jpg')))} images")

    shutil.rmtree(extract_dir, ignore_errors=True)
    marker.write_text(f"{total} images\n", encoding="utf-8")
    return raw_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw Cats vs Dogs images")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    cfg = get_config()
    raw_dir = cfg.resolve(cfg.get("data.raw_dir", "data/raw"))
    download_raw_data(raw_dir, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
