"""Train skin cancer classification model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from config import CLASS_LABELS, EPOCHS, LEARNING_RATE, MODEL_DIR, RESULTS_DIR
from src.data_loader import build_loaders
from src.model import build_model, get_device


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_history(history, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_acc"], label="Train")
    axes[0].plot(history["val_acc"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_class_metrics(report: dict, out_path: Path) -> None:
    metrics = ["precision", "recall", "f1-score"]
    values = np.array([[report[label][metric] for metric in metrics] for label in CLASS_LABELS])

    x = np.arange(len(CLASS_LABELS))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1) * width, values[:, i], width, label=metric.title())

    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.title("Per-Class Precision, Recall, and F1")
    plt.xticks(x, CLASS_LABELS, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def find_hard_to_distinguish_pairs(cm: np.ndarray) -> list[dict]:
    pairs = []
    for actual_idx, actual_label in enumerate(CLASS_LABELS):
        row_total = int(cm[actual_idx].sum())
        if row_total == 0:
            continue
        for pred_idx, pred_label in enumerate(CLASS_LABELS):
            if actual_idx == pred_idx:
                continue
            mistakes = int(cm[actual_idx, pred_idx])
            if mistakes == 0:
                continue
            pairs.append(
                {
                    "actual": actual_label,
                    "predicted": pred_label,
                    "misclassified_count": mistakes,
                    "actual_class_total": row_total,
                    "misclassification_rate": mistakes / row_total,
                }
            )
    return sorted(pairs, key=lambda item: (item["misclassified_count"], item["misclassification_rate"]), reverse=True)


def evaluate_model(model, val_loader, device, results_dir: Path) -> dict:
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_LABELS))),
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    hard_pairs = find_hard_to_distinguish_pairs(cm)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "hard_to_distinguish_pairs": hard_pairs[:10],
    }

    plot_confusion_matrix(y_true, y_pred, results_dir / "confusion_matrix.png")
    plot_per_class_metrics(report, results_dir / "per_class_metrics.png")

    with open(results_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"report": report, "metrics": metrics}


def train(epochs: int = EPOCHS, demo: bool = False) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, data_dir, classes = build_loaders()
    print(f"Training on data from: {data_dir}")
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = MODEL_DIR / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": CLASS_LABELS}, best_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = MODEL_DIR / f"skin_cancer_model_{timestamp}.pt"
    torch.save({"model_state": model.state_dict(), "classes": CLASS_LABELS}, final_path)
    torch.save({"model_state": model.state_dict(), "classes": CLASS_LABELS}, MODEL_DIR / "skin_cancer_model.pt")

    plot_history(history, RESULTS_DIR / "training_history.png")
    evaluation = evaluate_model(model, val_loader, device, RESULTS_DIR)

    summary = {
        "epochs": epochs,
        "demo_mode": demo,
        "device": str(device),
        "data_dir": str(data_dir),
        "model_path": str(final_path),
        "val_accuracy": history["val_acc"][-1],
        "best_val_accuracy": best_val_acc,
        "val_loss": history["val_loss"][-1],
        "accuracy": evaluation["metrics"]["accuracy"],
        "weighted_precision": evaluation["metrics"]["weighted_precision"],
        "weighted_recall": evaluation["metrics"]["weighted_recall"],
        "weighted_f1": evaluation["metrics"]["weighted_f1"],
        "hard_to_distinguish_pairs": evaluation["metrics"]["hard_to_distinguish_pairs"],
        "classification_report": evaluation["report"],
    }

    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Model saved: {final_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results: {RESULTS_DIR}")

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skin cancer classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--demo", action="store_true", help="Force demo dataset generation")
    args = parser.parse_args()

    if args.demo:
        from src.demo_data import generate_demo_dataset

        generate_demo_dataset()

    train(epochs=args.epochs, demo=args.demo)


if __name__ == "__main__":
    main()
