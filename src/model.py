"""CNN model definition with transfer learning (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from config import CLASS_LABELS


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.mobilenet_v2(weights=weights)

    for param in backbone.features.parameters():
        param.requires_grad = False

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_LABELS)),
    )

    return backbone


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
