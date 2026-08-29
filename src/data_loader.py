"""Data loading and augmentation for skin lesion classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from config import BATCH_SIZE, CLASS_LABELS, IMG_SIZE, PROCESSED_DIR, RANDOM_SEED, VALIDATION_SPLIT


def _count_images(data_dir: Path) -> int:
    return sum(1 for _ in data_dir.rglob("*.jpg")) + sum(1 for _ in data_dir.rglob("*.jpeg"))


def ensure_dataset(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or PROCESSED_DIR
    if not data_dir.exists() or _count_images(data_dir) == 0:
        from src.demo_data import generate_demo_dataset

        print("No dataset found. Generating demo data...")
        return generate_demo_dataset()
    return data_dir


def build_loaders(data_dir: Path | None = None):
    data_dir = ensure_dataset(data_dir)

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    class_to_idx = {label: full_dataset.class_to_idx[label] for label in CLASS_LABELS if label in full_dataset.class_to_idx}

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_dataset = datasets.ImageFolder(str(data_dir), transform=val_transform)
    val_subset = Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, data_dir, full_dataset.classes
