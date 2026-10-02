"""Download the full HAM10000 image set (10,015 images) into data/raw/images.

Source: ISIC Archive public bucket. Images are CC BY-NC 4.0 (Tschandl et al., 2018)
and are NOT committed to git. Existing files are skipped, so the script is resumable.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DIR

IMAGE_URL = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"


def fetch(image_id: str, out_dir: Path, retries: int = 4) -> bool:
    out = out_dir / f"{image_id}.jpg"
    if out.exists() and out.stat().st_size > 1024:
        return True
    for _ in range(retries):
        try:
            with urlopen(IMAGE_URL.format(image_id=image_id), timeout=60) as r:
                data = r.read()
            if len(data) > 1024:
                out.write_bytes(data)
                return True
        except Exception:
            pass
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    meta = pd.read_csv(RAW_DIR / "HAM10000_metadata.csv")
    out_dir = RAW_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = meta["image_id"].tolist()

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, i, out_dir): i for i in ids}
        for n, f in enumerate(as_completed(futs), 1):
            if not f.result():
                failed.append(futs[f])
            if n % 500 == 0:
                print(f"{n}/{len(ids)} done, {len(failed)} failed", flush=True)
    print(f"Finished: {len(ids) - len(failed)}/{len(ids)} images in {out_dir}")
    if failed:
        print("Failed:", failed[:20], "... rerun to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()
