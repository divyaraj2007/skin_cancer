"""Predict skin lesion type from an image with evaluation safeguards and clinical safety notices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    CLASS_LABELS,
    CLASS_NAMES,
    IMG_SIZE,
    MALIGNANT_CLASSES,
    MALIGNANT_RISK_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MODEL_DIR,
)
from src.model import build_model, get_device


def load_model(model_path: Path | None = None):
    model_path = model_path or MODEL_DIR / "skin_cancer_model.pt"
    if not model_path.exists():
        model_path = MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found in {MODEL_DIR}. Run: python -m src.train --demo"
        )

    checkpoint = torch.load(model_path, map_location=get_device(), weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(get_device())
    model.eval()
    return model


def validate_image_input(image_path: Path) -> Image.Image:
    """Safeguard: validate image readability, format, resolution, and variance."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid image file at '{image_path}': {exc}") from exc

    # Re-open after verify()
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    if width < 32 or height < 32:
        raise ValueError(
            f"Image resolution too low ({width}x{height}). Minimum required resolution is 32x32 pixels."
        )

    arr = np.array(img, dtype=np.float32)
    std_dev = float(np.std(arr))
    if std_dev < 3.0:
        raise ValueError(
            f"Image lacks visual variation (std dev: {std_dev:.2f}). Degenerate or uniform images cannot be analyzed."
        )

    return img


def preprocess_image(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def predict(image_path: Path, model_path: Path | None = None) -> dict:
    device = get_device()
    model = load_model(model_path)
    img = validate_image_input(image_path)
    batch = preprocess_image(img).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked = sorted(zip(CLASS_LABELS, probs), key=lambda x: x[1], reverse=True)
    top_label, top_conf = ranked[0]
    malignant_prob = sum(conf for label, conf in ranked if label in MALIGNANT_CLASSES)

    # Safeguards: Uncertainty & Risk flags
    is_uncertain = top_conf < MIN_CONFIDENCE_THRESHOLD
    has_elevated_malignancy_risk = (top_label not in MALIGNANT_CLASSES) and (
        malignant_prob >= MALIGNANT_RISK_THRESHOLD
    )

    return {
        "image": str(image_path),
        "prediction": top_label,
        "prediction_name": CLASS_NAMES[top_label],
        "confidence": float(top_conf * 100),
        "malignant_probability": float(malignant_prob * 100),
        "is_top_malignant": top_label in MALIGNANT_CLASSES,
        "is_uncertain": bool(is_uncertain),
        "has_elevated_malignancy_risk": bool(has_elevated_malignancy_risk),
        "all_predictions": [
            {
                "class": label,
                "name": CLASS_NAMES[label],
                "confidence": float(conf * 100),
                "malignant": label in MALIGNANT_CLASSES,
            }
            for label, conf in ranked
        ],
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 65)
    print("SKIN LESION ANALYSIS - RESEARCH & EDUCATIONAL PROTOTYPE ONLY")
    print("=" * 65)
    print(f"Image: {result['image']}")
    print(f"Top Model Prediction: {result['prediction_name']}")
    print(f"Prediction Confidence: {result['confidence']:.1f}%")
    print(f"Combined Malignancy Probability: {result['malignant_probability']:.1f}%")

    if result["is_uncertain"]:
        print(
            f"\n[EVALUATION SAFEGUARD] HIGH UNCERTAINTY: Confidence ({result['confidence']:.1f}%) "
            f"is below the reliability threshold (40.0%). Result is unconfident."
        )

    if result["is_top_malignant"]:
        print("\n[CLINICAL RISK] Category: MALIGNANT/PRE-CANCEROUS")
        print("  -> Urgent professional evaluation by a dermatologist is recommended.")
    elif result["has_elevated_malignancy_risk"]:
        print("\n[CLINICAL RISK SAFEGUARD] Category: BENIGN TOP CLASS WITH ELEVATED MALIGNANCY RISK")
        print(
            f"  -> Although top class is benign, combined malignancy probability is {result['malignant_probability']:.1f}%."
        )
        print("  -> AI models can produce false negatives. In-person clinical examination advised.")
    else:
        print("\n[CLINICAL RISK] Category: BENIGN TOP CLASS")
        print("  -> Non-malignant prediction does NOT rule out skin cancer.")
        print("  -> Any new, changing, or bleeding lesion must be evaluated by a healthcare provider.")

    print("\nAll class probabilities:")
    for item in result["all_predictions"]:
        flag = " [MALIGNANT/PRE-CANCER]" if item["malignant"] else " [BENIGN]"
        print(f"  {item['name']}: {item['confidence']:.1f}%{flag}")

    print("=" * 65)
    print("CRITICAL MEDICAL DISCLAIMER:")
    print("This tool is strictly for research and algorithmic education. It is NOT an FDA-cleared")
    print("or CE-marked medical device, is NOT clinical decision support, and CANNOT provide a")
    print("medical diagnosis. Never delay seeking professional medical advice based on this output.")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict skin lesion type with clinical safeguards")
    parser.add_argument("image", type=Path, help="Path to lesion image")
    parser.add_argument("--model", type=Path, default=None, help="Optional model path")
    args = parser.parse_args()

    result = predict(args.image, args.model)
    print_result(result)


if __name__ == "__main__":
    main()
