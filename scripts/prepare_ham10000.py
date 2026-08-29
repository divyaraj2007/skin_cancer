"""Prepare HAM10000 dataset into class folders for training.

Expected raw layout after manual download:
  data/raw/HAM10000_metadata.csv
  data/raw/images/ISIC_*.jpg

Download:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import CLASS_LABELS, PROCESSED_DIR, RAW_DIR


def prepare(raw_dir: Path | None = None, output_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or RAW_DIR
    output_dir = output_dir or PROCESSED_DIR

    metadata_path = raw_dir / "HAM10000_metadata.csv"
    images_dir = raw_dir / "images"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Download HAM10000 and place files under {raw_dir}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in CLASS_LABELS:
        (output_dir / label).mkdir()

    df = pd.read_csv(metadata_path)
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        label = row["dx"]
        if label not in CLASS_LABELS:
            continue

        image_id = row["image_id"]
        src = images_dir / f"{image_id}.jpg"
        if not src.exists():
            src = images_dir / f"{image_id}.jpeg"
        if not src.exists():
            missing += 1
            continue

        dst = output_dir / label / src.name
        shutil.copy2(src, dst)
        copied += 1

    print(f"Prepared {copied} images in {output_dir}")
    if missing:
        print(f"Warning: {missing} metadata rows had no matching image file")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 for training")
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    prepare(args.raw, args.out)


if __name__ == "__main__":
    main()
