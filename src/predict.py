"""Predict skin lesion type from an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_LABELS, CLASS_NAMES, IMG_SIZE, MALIGNANT_CLASSES, MODEL_DIR
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


def preprocess_image(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(image_path: Path, model_path: Path | None = None) -> dict:
    device = get_device()
    model = load_model(model_path)
    batch = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked = sorted(zip(CLASS_LABELS, probs), key=lambda x: x[1], reverse=True)
    top_label, top_conf = ranked[0]
    malignant_prob = sum(conf for label, conf in ranked if label in MALIGNANT_CLASSES)

    return {
        "image": str(image_path),
        "prediction": top_label,
        "prediction_name": CLASS_NAMES[top_label],
        "confidence": float(top_conf * 100),
        "malignant_probability": float(malignant_prob * 100),
        "is_likely_malignant": top_label in MALIGNANT_CLASSES,
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
    print("\n" + "=" * 50)
    print("SKIN LESION ANALYSIS (ML)")
    print("=" * 50)
    print(f"Image: {result['image']}")
    print(f"Prediction: {result['prediction_name']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Malignant probability (combined): {result['malignant_probability']:.1f}%")
    print(
        "Likely malignant: YES - Consult dermatologist"
        if result["is_likely_malignant"]
        else "Likely malignant: NO - Still verify clinically"
    )
    print("\nAll class probabilities:")
    for item in result["all_predictions"]:
        flag = " [MALIGNANT]" if item["malignant"] else ""
        print(f"  {item['name']}: {item['confidence']:.1f}%{flag}")
    print("=" * 50)
    print("Disclaimer: Research/educational tool only. Not a medical diagnosis.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict skin lesion type")
    parser.add_argument("image", type=Path, help="Path to lesion image")
    parser.add_argument("--model", type=Path, default=None, help="Optional model path")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    result = predict(args.image, args.model)
    print_result(result)


if __name__ == "__main__":
    main()
