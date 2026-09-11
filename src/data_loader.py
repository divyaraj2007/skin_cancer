"""Data loading, stratified splitting, and augmentation for skin lesion classification."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CLASS_LABELS,
    DEMO_DIR,
    IMG_SIZE,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def _count_images(data_dir: Path) -> int:
    if not data_dir.exists():
        return 0
    return sum(1 for _ in data_dir.rglob("*.jpg")) + sum(1 for _ in data_dir.rglob("*.jpeg"))


def ensure_dataset(data_dir: Path | None = None, demo: bool = False) -> Path:
    """Ensure a valid dataset exists with strict separation between demo and real data."""
    if demo:
        target_dir = data_dir or DEMO_DIR
        if not target_dir.exists() or _count_images(target_dir) == 0:
            from src.demo_data import generate_demo_dataset

            print(f"Generating synthetic demo dataset in {target_dir}...")
            return generate_demo_dataset(output_dir=target_dir)
        return target_dir

    target_dir = data_dir or PROCESSED_DIR
    if not target_dir.exists() or _count_images(target_dir) == 0:
        raise FileNotFoundError(
            f"No processed dataset found in '{target_dir}'.\n"
            "Data Separation Safeguard: Synthetic demo data is kept separate and will not automatically "
            "overwrite or populate real data paths.\n"
            "- To train on real images: download HAM10000 into data/raw or run scripts/download_ham10000_samples.py\n"
            "- To run an isolated demo test: pass '--demo' (trains on synthetic data in data/demo/)"
        )
    return target_dir


def _verify_split_safeguards(
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    total_size: int,
    targets: list[int],
    num_classes: int,
) -> None:
    """Rigorous evaluation safeguards: verify zero partition leakage and class coverage."""
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    # 1. Zero data leakage verification
    assert train_set.isdisjoint(val_set), "Evaluation Safeguard Violation: Train and Validation sets overlap!"
    assert train_set.isdisjoint(test_set), "Evaluation Safeguard Violation: Train and Test sets overlap!"
    assert val_set.isdisjoint(test_set), "Evaluation Safeguard Violation: Validation and Test sets overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == total_size, (
        f"Partition size mismatch: sum={len(train_set)+len(val_set)+len(test_set)} vs total={total_size}"
    )

    # 2. Representation check across all classes
    for partition_name, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        partition_labels = {targets[i] for i in indices}
        if len(partition_labels) < num_classes:
            missing = set(range(num_classes)) - partition_labels
            print(f"Warning: {partition_name} partition is missing classes: {missing}")


def build_loaders(
    data_dir: Path | None = None,
    demo: bool = False,
    val_split: float = VALIDATION_SPLIT,
    test_split: float = TEST_SPLIT,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, Path, list[str]]:
    """Build stratified DataLoaders for train, validation, and held-out test splits.

    Ensures:
    - Complete data separation (demo vs processed).
    - Stratified class balance across all splits.
    - Zero data leakage between partitions.
    """
    data_dir = ensure_dataset(data_dir=data_dir, demo=demo)

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Base dataset for indexing and labels
    raw_dataset = datasets.ImageFolder(str(data_dir))
    targets = [sample[1] for sample in raw_dataset.samples]
    total_samples = len(targets)
    indices = list(range(total_samples))

    if total_samples < 7:
        raise ValueError(f"Dataset in {data_dir} has only {total_samples} samples, insufficient for 7-class splitting.")

    # Stratified Train / Val / Test splitting
    # Split test set first
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        stratify=targets,
        random_state=RANDOM_SEED,
    )

    # Split remaining into train and validation
    train_val_targets = [targets[i] for i in train_val_idx]
    relative_val_ratio = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_ratio,
        stratify=train_val_targets,
        random_state=RANDOM_SEED,
    )

    # Run integrity and anti-leakage safeguards
    _verify_split_safeguards(
        train_idx,
        val_idx,
        test_idx,
        total_samples,
        targets,
        len(raw_dataset.classes),
    )

    # Wrap with appropriate transforms
    train_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    eval_dataset = datasets.ImageFolder(str(data_dir), transform=eval_transform)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)
    test_subset = Subset(eval_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, data_dir, raw_dataset.classes
