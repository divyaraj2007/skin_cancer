"""Grad-CAM figure: where does the fine-tuned model look? (results/paper/figures/fig7_gradcam.png)

    python -m src.gradcam --checkpoint models/ham10000_ft_cw_s0.pt
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import CLASS_LABELS, RESULTS_DIR
from src.ham import get_split, load_images, normalize, to_tensor
from src.model import build_model


def gradcam(model, x: torch.Tensor, target: int) -> np.ndarray:
    """Grad-CAM on the last MobileNetV2 feature map; returns a (H, W) map in [0, 1]."""
    acts = {}
    h = model.features.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    model.eval()
    x = normalize(x).requires_grad_(True)
    logits = model(x)
    h.remove()
    grads = torch.autograd.grad(logits[0, target], acts["a"])[0]
    w = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    df = get_split()
    imgs = load_images(df)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

    test = df[df.split == "test"]
    rng = np.random.default_rng(0)
    # one column per class; for each example an image row followed by its Grad-CAM row
    fig, axes = plt.subplots(args.per_class * 2, len(CLASS_LABELS),
                             figsize=(1.9 * len(CLASS_LABELS), 2.0 * args.per_class * 2))
    for col, c in enumerate(CLASS_LABELS):
        rows = test[test.dx == c].index.values
        for j, i in enumerate(rng.choice(rows, args.per_class, replace=False)):
            x = to_tensor(imgs[i:i + 1])
            with torch.no_grad():
                p = torch.softmax(model(normalize(x)), 1)[0]
            pred = int(p.argmax())
            cam = gradcam(model, x, pred)
            a0, a1 = axes[2 * j, col], axes[2 * j + 1, col]
            a0.imshow(imgs[i]); a1.imshow(imgs[i]); a1.imshow(cam, cmap="jet", alpha=0.45)
            a0.set_title(f"true: {c}", fontsize=9)
            a1.set_title(f"pred: {CLASS_LABELS[pred]} ({p[pred]:.2f})", fontsize=9,
                         color="green" if pred == col else "red")
            a0.axis("off"); a1.axis("off")
    plt.tight_layout(pad=0.4)
    out = RESULTS_DIR / "paper" / "figures" / "fig7_gradcam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
