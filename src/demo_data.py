"""Generate synthetic demo dataset for pipeline testing without HAM10000."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import CLASS_LABELS, PROCESSED_DIR, RANDOM_SEED


def _color_for_class(label: str, rng: random.Random) -> tuple[int, int, int]:
    palette = {
        "mel": (40, 20, 10),
        "bcc": (180, 120, 90),
        "akiec": (200, 80, 60),
        "nv": (90, 60, 40),
        "bkl": (150, 110, 80),
        "df": (120, 90, 70),
        "vasc": (160, 40, 40),
    }
    base = palette[label]
    jitter = rng.randint(-15, 15)
    return tuple(max(0, min(255, c + jitter)) for c in base)


def _make_lesion_image(label: str, rng: random.Random, np_rng: np.random.Generator, size: int = 224) -> Image.Image:
    skin = rng.randint(210, 245)
    img = Image.new("RGB", (size, size), (skin, skin - 10, skin - 20))
    draw = ImageDraw.Draw(img)

    cx, cy = rng.randint(70, 154), rng.randint(70, 154)
    rx, ry = rng.randint(25, 55), rng.randint(20, 50)
    color = _color_for_class(label, rng)

    if label == "mel":
        for _ in range(rng.randint(2, 4)):
            ox, oy = rng.randint(-20, 20), rng.randint(-20, 20)
            draw.ellipse(
                (cx - rx + ox, cy - ry + oy, cx + rx + ox, cy + ry + oy),
                fill=color,
            )
    elif label == "bcc":
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    elif label == "vasc":
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(220, 60, 60))
    else:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)

    arr = np.array(img).astype(np.float32)
    arr += np_rng.uniform(-8, 8, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_demo_dataset(samples_per_class: int = 40) -> Path:
    """Create folder structure: data/processed/{class}/*.jpg"""
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for label in CLASS_LABELS:
        class_dir = PROCESSED_DIR / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = _make_lesion_image(label, rng, np_rng)
            img.save(class_dir / f"{label}_{i:03d}.jpg", quality=95)

    total = samples_per_class * len(CLASS_LABELS)
    print(f"Demo dataset ready: {total} images in {PROCESSED_DIR}")
    return PROCESSED_DIR


if __name__ == "__main__":
    generate_demo_dataset()
