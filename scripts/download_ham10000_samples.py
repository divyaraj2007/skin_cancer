"""Download a balanced real-image HAM10000 sample into data/processed.

This replaces the synthetic demo images with real dermoscopic images from
public ISIC/HAM10000 URLs while preserving the ImageFolder class structure.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_LABELS, PROCESSED_DIR, RAW_DIR, RANDOM_SEED


METADATA_URL = "https://raw.githubusercontent.com/uyxela/Skin-Lesion-Classifier/master/HAM10000_metadata.csv"
IMAGE_URL_TEMPLATE = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"


def download_metadata(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading metadata: {path}")
        urlretrieve(METADATA_URL, path)
    return path


def read_balanced_rows(metadata_path: Path, samples_per_class: int) -> dict[str, list[dict[str, str]]]:
    selected = {label: [] for label in CLASS_LABELS}
    seen_lesions = set()

    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Deterministic spread through the CSV without taking only the first cases.
    offset = RANDOM_SEED % len(rows)
    rows = rows[offset:] + rows[:offset]

    for row in rows:
        label = row["dx"]
        lesion_id = row.get("lesion_id", "")
        # Data leakage safeguard: enforce 1 image per lesion to avoid cross-split lesion overlap
        if lesion_id and lesion_id in seen_lesions:
            continue

        if label in selected and len(selected[label]) < samples_per_class:
            selected[label].append(row)
            if lesion_id:
                seen_lesions.add(lesion_id)

        if all(len(items) >= samples_per_class for items in selected.values()):
            break

    missing = {label: samples_per_class - len(items) for label, items in selected.items() if len(items) < samples_per_class}
    if missing:
        raise RuntimeError(f"Not enough unique lesion rows for requested sample size: {missing}")

    return selected


def reset_processed_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_project = PROJECT_ROOT.resolve()

    if resolved_output == resolved_project or resolved_project not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unexpected path: {resolved_output}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in CLASS_LABELS:
        (output_dir / label).mkdir(parents=True, exist_ok=True)


def download_image(image_id: str, out_path: Path, retries: int = 3) -> bool:
    url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
    for attempt in range(1, retries + 1):
        try:
            urlretrieve(url, out_path)
            return True
        except (HTTPError, URLError) as exc:
            if attempt == retries:
                print(f"Failed: {image_id} ({exc})")
                return False
            time.sleep(1.0 * attempt)
    return False


def download_dataset(samples_per_class: int = 40, output_dir: Path = PROCESSED_DIR) -> Path:
    metadata_path = download_metadata(RAW_DIR / "HAM10000_metadata.csv")
    selected = read_balanced_rows(metadata_path, samples_per_class)

    reset_processed_dir(output_dir)

    total = 0
    for label in CLASS_LABELS:
        for i, row in enumerate(selected[label]):
            image_id = row["image_id"]
            out_path = output_dir / label / f"{image_id}_{label}.jpg"
            if download_image(image_id, out_path):
                total += 1
        print(f"{label}: {len(list((output_dir / label).glob('*.jpg')))} real images")

    expected = samples_per_class * len(CLASS_LABELS)
    if total != expected:
        raise RuntimeError(f"Downloaded {total}/{expected} images")

    print(f"Real HAM10000 sample ready: {total} images in {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real HAM10000 images into data/processed")
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    download_dataset(samples_per_class=args.samples_per_class, output_dir=args.out)


if __name__ == "__main__":
    main()
