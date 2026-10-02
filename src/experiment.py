"""Train one configuration on full HAM10000 (lesion-level split) and save raw predictions.

Examples:
    python -m src.experiment --name frozen_ce --seed 0
    python -m src.experiment --name ft_cw --finetune --class-weights --seed 0 --epochs 12

Model selection uses the validation split only (best balanced accuracy); the test split is
scored exactly once, with the selected checkpoint. Softmax outputs for val and test are
stored so every statistic in the paper can be recomputed offline (src/analysis.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import CLASS_LABELS, MODEL_DIR, RESULTS_DIR
from src.ham import augment, get_split, load_images, normalize, to_tensor, verify_split
from src.model import build_model
from src.stats import balanced_accuracy

K = len(CLASS_LABELS)

RUNS_DIR = RESULTS_DIR / "runs"


def predict_probs(model: nn.Module, x_u8: np.ndarray, batch: int = 128, tta: bool = False) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_u8), batch):
            x = to_tensor(x_u8[i:i + batch])
            logits = model(normalize(x))
            if tta:  # flips average
                logits = (logits + model(normalize(x.flip(3))) + model(normalize(x.flip(2)))) / 3
            out.append(torch.softmax(logits, 1).numpy())
    return np.concatenate(out)


def set_train_mode(model: nn.Module, finetune: bool) -> None:
    model.train()
    if not finetune:  # frozen backbone: keep BatchNorm statistics fixed
        model.features.eval()


def run(args) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    df = get_split(mode=args.split)
    if args.split == "lesion":
        stats = verify_split(df)
    else:
        tr_les = set(df.loc[df.split == "train", "lesion_id"])
        leaked = int(df[(df.split == "test") & df.lesion_id.isin(tr_les)].shape[0])
        stats = {s: {"images": int((df.split == s).sum())} for s in ("train", "val", "test")}
        stats["test_images_with_lesion_in_train"] = leaked
    imgs = load_images(df)
    tr, va, te = (np.where(df.split.values == s)[0] for s in ("train", "val", "test"))
    y = df["label"].values
    print(f"[{args.name} seed={args.seed}] split sizes: {stats}", flush=True)

    model = build_model(pretrained=True)
    if args.finetune:
        for p in model.features.parameters():
            p.requires_grad = True
        groups = [{"params": model.features.parameters(), "lr": args.lr_backbone},
                  {"params": model.classifier.parameters(), "lr": args.lr_head}]
    else:
        groups = [{"params": model.classifier.parameters(), "lr": args.lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    steps_per_epoch = int(np.ceil(len(tr) / args.batch))
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))

    counts = np.bincount(y[tr], minlength=len(CLASS_LABELS)).astype(float)
    if args.class_weights:
        w = (counts.sum() / (len(counts) * counts)) ** 0.5  # sqrt inverse-frequency
        w = w / w.mean()
    else:
        w = np.ones(len(counts))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

    run_dir = RUNS_DIR / f"{args.name}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_state, hist = -1.0, None, []
    t0, start_ep = time.time(), 1
    ckpt_path = run_dir / "resume.pt"
    if ckpt_path.exists():  # resume an interrupted run exactly (weights, optimiser, schedule, RNG)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"])
        best, best_state, hist = ck["best"], ck["best_state"], ck["hist"]
        start_ep, t0 = ck["epoch"] + 1, time.time() - ck["elapsed"]
        print(f"  resumed from epoch {ck['epoch']}", flush=True)
    for ep in range(start_ep, args.epochs + 1):
        set_train_mode(model, args.finetune)
        perm = np.random.permutation(tr)
        loss_sum, n = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = np.sort(perm[i:i + args.batch])
            if len(idx) < 2:
                continue
            x = normalize(augment(to_tensor(imgs[idx])))
            yy = torch.from_numpy(y[idx]).long()
            opt.zero_grad()
            loss = criterion(model(x), yy)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            n += len(idx)
        p_val = predict_probs(model, imgs[va])
        val_loss = float(nn.functional.nll_loss(torch.log(torch.tensor(p_val) + 1e-9),
                                                torch.from_numpy(y[va]).long()))
        val_bacc = balanced_accuracy(y[va], p_val.argmax(1), K)
        val_acc = float((p_val.argmax(1) == y[va]).mean())
        hist.append({"epoch": ep, "train_loss": loss_sum / n, "val_loss": val_loss,
                     "val_acc": val_acc, "val_balanced_acc": float(val_bacc)})
        print(f"  ep {ep:02d}/{args.epochs} train_loss={loss_sum / n:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_bacc={val_bacc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val_bacc > best:
            best = val_bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "best": best,
                    "best_state": best_state, "hist": hist, "epoch": ep, "elapsed": time.time() - t0},
                   ckpt_path.with_suffix(".tmp"))
        ckpt_path.with_suffix(".tmp").replace(ckpt_path)  # atomic: never leave a half-written checkpoint

    model.load_state_dict(best_state)
    p_val = predict_probs(model, imgs[va], tta=args.tta)
    p_test = predict_probs(model, imgs[te], tta=args.tta)
    np.save(run_dir / "val_probs.npy", p_val)
    np.save(run_dir / "test_probs.npy", p_test)
    np.save(run_dir / "val_idx.npy", va)
    np.save(run_dir / "test_idx.npy", te)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "best_val_balanced_acc": best,
                   "train_seconds": time.time() - t0, "class_weights": w.tolist(),
                   "split": stats, "history": hist}, f, indent=2)
    if args.save_model:
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save({"model_state": best_state, "classes": CLASS_LABELS},
                   MODEL_DIR / f"ham10000_{args.name}_s{args.seed}.pt")
    ckpt_path.unlink(missing_ok=True)
    print(f"  done: test acc={(p_test.argmax(1) == y[te]).mean():.4f} "
          f"bacc={balanced_accuracy(y[te], p_test.argmax(1), K):.4f}", flush=True)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--split", choices=["lesion", "image"], default="lesion",
                   help="'image' = leaky image-level split, only for the leakage experiment")
    run(p.parse_args())


if __name__ == "__main__":
    main()
