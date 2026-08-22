"""Stage 0 of the DVC pipeline: fetch the raw Cats vs Dogs images.

The dataset is the Kaggle Dogs vs Cats collection. Public mirrors of it come
and go, so this stage tries a list of sources in order and only fails once
every one of them has been exhausted, reporting exactly what it tried.

Sources can be overridden without touching the code:

    DATA_URL=https://example.com/my-mirror.zip python -m src.data.download

Two archive layouts are understood:

    cats_and_dogs_filtered.zip   cats_and_dogs_filtered/train/cats, .../dogs
    kagglecatsanddogs_5340.zip   PetImages/Cat, PetImages/Dog

Layout produced, regardless of source:

    data/raw/cats/*.jpg
    data/raw/dogs/*.jpg
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.config import get_config

# Ordered list of mirrors. The first one that responds wins.
DATA_SOURCES: list[dict[str, str]] = [
    {
        "name": "tensorflow-mledu (filtered subset)",
        "url": "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        "size": "~68 MB",
    },
    {
        "name": "microsoft (full Kaggle Dogs vs Cats)",
        "url": (
            "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-"
            "6DEBA77B919F/kagglecatsanddogs_5340.zip"
        ),
        "size": "~786 MB",
    },
]

# Some CDNs reject the default urllib agent, so present a normal browser one.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _sources() -> list[dict[str, str]]:
    """Mirror list, with an optional DATA_URL override taking priority."""
    override = os.getenv("DATA_URL", "").strip()
    if override:
        return [{"name": "DATA_URL override", "url": override, "size": "unknown"}] + DATA_SOURCES
    return list(DATA_SOURCES)


def _fetch(url: str, dest: Path, attempts: int = 3, timeout: int = 300) -> None:
    """Stream a URL to disk, retrying transient failures."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(dest, "wb") as fh:
                    shutil.copyfileobj(response, fh, length=1024 * 256)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            # A 403/404 will not fix itself on a retry; give up on this mirror.
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                break
            if attempt < attempts:
                wait = 2 ** attempt
                print(f"[download]   attempt {attempt} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(str(last_error))


def download_archive(dest: Path) -> Path:
    """Download the dataset archive from the first mirror that works."""
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[download] archive already present: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for source in _sources():
        print(f"[download] trying {source['name']} ({source['size']})")
        print(f"[download]   {source['url']}")
        partial = dest.with_suffix(".part")
        try:
            _fetch(source["url"], partial)
        except RuntimeError as exc:
            print(f"[download]   FAILED: {exc}")
            failures.append(f"  - {source['name']}: {exc}")
            partial.unlink(missing_ok=True)
            continue

        if not zipfile.is_zipfile(partial):
            print("[download]   FAILED: response was not a zip archive")
            failures.append(f"  - {source['name']}: response was not a zip archive")
            partial.unlink(missing_ok=True)
            continue

        partial.replace(dest)
        print(f"[download]   OK -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    raise RuntimeError(
        "Could not download the dataset from any known mirror.\n"
        + "\n".join(failures)
        + "\n\nSet DATA_URL to a reachable zip of the Cats vs Dogs dataset, e.g.\n"
        "  DATA_URL=https://your-mirror/cats_and_dogs.zip python -m src.data.download"
    )


def find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """Locate the cat and dog image directories inside an extracted archive.

    Works across archive layouts by matching directory names rather than
    hard-coding paths: any directory called cat/cats/Cat/... counts as cats.
    """
    found: dict[str, list[Path]] = {"cats": [], "dogs": []}
    for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
        name = directory.name.lower()
        target = "cats" if name.startswith("cat") else "dogs" if name.startswith("dog") else None
        if target is None:
            continue
        if any(child.suffix.lower() in IMAGE_SUFFIXES for child in directory.iterdir()):
            found[target].append(directory)
    return found


def download_raw_data(
    raw_dir: Path, limit_per_class: int | None = 1500, force: bool = False
) -> Path:
    """Download, unpack and normalise the dataset into raw_dir/{cats,dogs}."""
    raw_dir = Path(raw_dir)
    marker = raw_dir / ".download_complete"
    if marker.exists() and not force:
        print(f"[download] raw data already present at {raw_dir}, skipping")
        return raw_dir

    archive = raw_dir.parent / "cats_vs_dogs.zip"
    download_archive(archive)

    extract_dir = raw_dir.parent / "_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    print(f"[download] extracting to {extract_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)

    class_dirs = find_class_dirs(extract_dir)
    if not class_dirs["cats"] or not class_dirs["dogs"]:
        raise RuntimeError(
            f"Could not find cat and dog image directories inside {archive.name}. "
            f"Found: { {k: [str(p) for p in v] for k, v in class_dirs.items()} }"
        )

    totals: dict[str, int] = {}
    for class_name, directories in class_dirs.items():
        out_dir = raw_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {class_name}: collecting from {len(directories)} directory(ies)")

        copied = 0
        for directory in directories:
            for image in sorted(directory.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if limit_per_class and copied >= limit_per_class:
                    break
                # Sequential names: several source directories (e.g. train/ and
                # validation/) can hold identically named files, and prefixing
                # with the directory name is not enough to keep them apart.
                shutil.copy2(image, out_dir / f"{class_name}_{copied:05d}{image.suffix.lower()}")
                copied += 1
            if limit_per_class and copied >= limit_per_class:
                break

        totals[class_name] = copied
        print(f"[download] {class_name}: {copied} images -> {out_dir}")

    shutil.rmtree(extract_dir, ignore_errors=True)

    if min(totals.values()) == 0:
        raise RuntimeError(f"No images were copied: {totals}")

    marker.write_text(f"{totals}\n", encoding="utf-8")
    return raw_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw Cats vs Dogs images")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="cap images copied per class (keeps CI fast on the full dataset)",
    )
    args = parser.parse_args(argv)

    cfg = get_config()
    raw_dir = cfg.resolve(cfg.get("data.raw_dir", "data/raw"))
    limit = args.limit_per_class or max(1500, int(cfg.get("data.max_images_per_class") or 1500))

    try:
        download_raw_data(raw_dir, limit_per_class=limit, force=args.force)
    except RuntimeError as exc:
        print(f"\n[download] ERROR\n{exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
