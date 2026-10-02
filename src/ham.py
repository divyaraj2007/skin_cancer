"""Full HAM10000 data pipeline: metadata, lesion-level splits, image cache, augmentation.

HAM10000 contains several images of the same lesion (10,015 images, 7,470 lesions).
Splitting by image would put near-duplicates in train and test and inflate results, so all
splits here are made at the *lesion* level (class-stratified grouping on ``lesion_id``).
"""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_LABELS, DATA_DIR, RAW_DIR
from src.stats import stratified_group_folds

CACHE_DIR = DATA_DIR / "cache"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_PATH = SPLIT_DIR / "ham10000_lesion_split_seed42.csv"
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_LABELS)}


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "HAM10000_metadata.csv")
    df["label"] = df["dx"].map(LABEL_TO_IDX)
    return df


def make_lesion_split(df: pd.DataFrame, seed: int = 42, folds: int = 20,
                      test_folds: int = 3, val_folds: int = 3) -> pd.DataFrame:
    """70/15/15 train/val/test split with no lesion shared between partitions."""
    fold = stratified_group_folds(df["label"].values, df["lesion_id"].values, folds, seed)
    split = np.where(fold < test_folds, "test",
                     np.where(fold < test_folds + val_folds, "val", "train"))
    out = df.copy()
    out["split"] = split
    return out


def verify_split(df: pd.DataFrame) -> dict:
    """Hard checks: no lesion leaks across partitions; every class is in every partition."""
    groups = {s: set(df.loc[df.split == s, "lesion_id"]) for s in ("train", "val", "test")}
    assert groups["train"].isdisjoint(groups["val"]), "lesion leakage train/val"
    assert groups["train"].isdisjoint(groups["test"]), "lesion leakage train/test"
    assert groups["val"].isdisjoint(groups["test"]), "lesion leakage val/test"
    for s in groups:
        assert set(df.loc[df.split == s, "label"]) == set(range(len(CLASS_LABELS))), f"class missing in {s}"
    return {s: {"images": int((df.split == s).sum()), "lesions": len(g)} for s, g in groups.items()}


def make_image_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Naive class-stratified 70/15/15 split by *image*, ignoring lesion identity.

    Only used to quantify the optimism caused by lesion leakage; never for the main results.
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(df), "train", dtype=object)
    for lab in np.unique(df["label"]):
        idx = rng.permutation(np.where(df["label"].values == lab)[0])
        n_te = n_va = int(round(0.15 * len(idx)))
        split[idx[:n_te]] = "test"
        split[idx[n_te:n_te + n_va]] = "val"
    out = df.copy()
    out["split"] = split
    return out


def get_split(seed: int = 42, mode: str = "lesion") -> pd.DataFrame:
    """Load the committed split file (reproducible), creating it on first use.

    ``mode="image"`` returns the leaky image-level split used only for the leakage experiment.
    """
    if mode == "image":
        return make_image_split(load_metadata(), seed=seed)
    df = load_metadata()
    if SPLIT_PATH.exists() and seed == 42:
        sp = pd.read_csv(SPLIT_PATH)[["image_id", "split"]]
        df = df.merge(sp, on="image_id", how="left")
        assert df["split"].notna().all()
    else:
        df = make_lesion_split(df, seed=seed)
        if seed == 42:
            SPLIT_DIR.mkdir(parents=True, exist_ok=True)
            df[["image_id", "lesion_id", "dx", "split"]].to_csv(SPLIT_PATH, index=False)
    verify_split(df)
    return df


def _load(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_images(df: pd.DataFrame, size: int = IMG_SIZE) -> np.ndarray:
    """uint8 array (N, size, size, 3) in metadata order; cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ham10000_{size}.npy"
    if cache.exists():
        arr = np.load(cache)  # read fully into RAM (memory-mapping caused heavy paging under memory pressure)
        if len(arr) == len(df):
            return arr
    img_dir = RAW_DIR / "images"
    paths = [img_dir / f"{i}.jpg" for i in df["image_id"]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing (e.g. {missing[:3]}). "
                                "Run: python scripts/download_ham10000_full.py")
    with ThreadPoolExecutor(16) as ex:
        arr = np.stack(list(ex.map(lambda p: _load(p, size), paths)))
    np.save(cache, arr)
    return arr


def to_tensor(batch_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float() / 255.0
    return x


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN) / STD


def augment(x: torch.Tensor) -> torch.Tensor:
    """Batch augmentation on [0,1] tensors: flips, arbitrary rotation, zoom, shift, colour jitter.

    Dermoscopic images have no canonical orientation, so full-rotation and both flips are safe.
    """
    b = x.size(0)
    flip_h = (torch.rand(b) < 0.5).float() * 2 - 1
    flip_v = (torch.rand(b) < 0.5).float() * 2 - 1
    ang = torch.rand(b) * 2 * math.pi
    scale = 1.0 / (0.85 + torch.rand(b) * 0.35)  # zoom in/out by up to ~15%
    shift = (torch.rand(b, 2) - 0.5) * 0.15
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0] = cos * flip_h
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos * flip_v
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    brightness = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.3
    contrast = 1 + (torch.rand(b, 1, 1, 1) - 0.5) * 0.3
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness
    return x.clamp(0, 1)
