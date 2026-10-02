"""Aggregate experiment runs into paper-ready tables and figures (results/paper/).

Reads results/runs/<config>_s<seed>/ produced by src/experiment.py.
- Table 1: every configuration, mean +/- SD over seeds (test split).
- Table 2: primary configuration as a seed-ensemble, with 95% CIs from a lesion-level bootstrap.
- Calibration: ECE before/after temperature scaling (fit on the validation split only).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASS_LABELS, MALIGNANT_CLASSES, RESULTS_DIR
from src.ham import get_split
from src.stats import (auc as safe_auc, balanced_accuracy, confusion_matrix, fit_temperature,
                       macro_f1, prf, roc_curve, weighted_f1)

RUNS_DIR = RESULTS_DIR / "runs"
OUT_DIR = RESULTS_DIR / "paper"
K = len(CLASS_LABELS)
MAL = np.array([c in MALIGNANT_CLASSES for c in CLASS_LABELS])
MEL = CLASS_LABELS.index("mel")

CONFIG_TITLES = {
    "frozen_ce": "Frozen backbone, CE",
    "frozen_cw": "Frozen backbone, class-weighted CE",
    "ft_ce": "Fine-tuned, CE",
    "ft_cw": "Fine-tuned, class-weighted CE",
}


def ece_score(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    pr, rc, f1, _ = prf(y, pred, K)
    y_mal, s_mal = MAL[y], p[:, MAL].sum(1)
    pred_mal = MAL[pred]
    tp, fn = (pred_mal & y_mal).sum(), (~pred_mal & y_mal).sum()
    tn, fp = (~pred_mal & ~y_mal).sum(), (pred_mal & ~y_mal).sum()
    aucs = [safe_auc(y == k, p[:, k]) for k in range(K)]
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": balanced_accuracy(y, pred, K),
        "macro_f1": macro_f1(y, pred, K),
        "weighted_f1": weighted_f1(y, pred, K),
        "macro_auc": float(np.nanmean(aucs)),
        "mel_sensitivity": float(rc[MEL]),
        "mel_auc": aucs[MEL],
        "malignant_sensitivity": float(tp / max(tp + fn, 1)),
        "malignant_specificity": float(tn / max(tn + fp, 1)),
        "malignant_auc": safe_auc(y_mal, s_mal),
        "ece": ece_score(p, y),
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                          "auc": aucs[i], "support": int((y == i).sum())}
                      for i, c in enumerate(CLASS_LABELS)},
    }


def bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Percentile CIs from resampling whole lesions (images of one lesion stay together)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    scalar_keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auc", "mel_sensitivity",
                   "mel_auc", "malignant_sensitivity", "malignant_specificity", "malignant_auc", "ece"]
    draws = {k: [] for k in scalar_keys}
    pc = {c: {m: [] for m in ("precision", "recall", "f1", "auc")} for c in CLASS_LABELS}
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        m = metrics(y[idx], p[idx])
        for k in scalar_keys:
            draws[k].append(m[k])
        for c in CLASS_LABELS:
            for k in pc[c]:
                pc[c][k].append(m["per_class"][c][k])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {"scalar": {k: ci(v) for k, v in draws.items()},
            "per_class": {c: {k: ci(v) for k, v in d.items()} for c, d in pc.items()}}


def paired_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray, keys: list[str],
                 n: int = 1000, seed: int = 0) -> dict:
    """Paired lesion-level bootstrap of metric(A) - metric(B) on the same test images.

    Returns the observed difference, a 95% percentile CI and a two-sided bootstrap p-value
    (twice the smaller tail mass of the difference distribution around zero).
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    ma, mb = metrics(y, p_a), metrics(y, p_b)
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        a, b = metrics(y[idx], p_a[idx]), metrics(y[idx], p_b[idx])
        for k in keys:
            draws[k].append(a[k] - b[k])
    out = {}
    for k in keys:
        d = np.array(draws[k])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[k] = {"delta": ma[k] - mb[k], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p": float(max(p, 1 / n))}
    return out


def operating_point(y_val, p_val, y_te, p_te, target: float = 0.90) -> dict:
    """Binary malignant-vs-benign triage: pick the threshold on the summed malignant probability that
    reaches ``target`` sensitivity on *validation*, then report sensitivity/specificity on *test*."""
    s_val, s_te = p_val[:, MAL].sum(1), p_te[:, MAL].sum(1)
    pos = np.sort(s_val[MAL[y_val]])[::-1]
    thr = float(pos[int(np.ceil(target * len(pos))) - 1])
    yb = MAL[y_te]
    pred = s_te >= thr
    return {"target_sensitivity": target, "threshold": thr,
            "test_sensitivity": float((pred & yb).sum() / yb.sum()),
            "test_specificity": float((~pred & ~yb).sum() / (~yb).sum()),
            "test_flagged_fraction": float(pred.mean())}


LEAK_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "mel_sensitivity"]


def _with_ci(y, p, groups, n) -> dict:
    m, b = metrics(y, p), bootstrap(y, p, groups, n=n)
    return {k: {"value": m[k], "ci": b["scalar"][k]} for k in LEAK_KEYS} | {"n": int(len(y))}


def leakage_analysis(run_dir: Path, y_les, p_les, g_les, n_boot: int) -> dict:
    """Same recipe trained on a naive image-level split. Its test set is scored overall and split into
    images whose lesion also appears in training ('seen') vs not ('unseen')."""
    dfi = get_split(mode="image")
    te = np.load(run_dir / "test_idx.npy")
    p = np.load(run_dir / "test_probs.npy")
    y, g = dfi["label"].values[te], dfi["lesion_id"].values[te]
    seen = np.isin(g, dfi.loc[dfi.split == "train", "lesion_id"].values)
    return {"lesion_split": _with_ci(y_les, p_les, g_les, n_boot),
            "image_split": _with_ci(y, p, g, n_boot),
            "image_split_seen": _with_ci(y[seen], p[seen], g[seen], n_boot),
            "image_split_unseen": _with_ci(y[~seen], p[~seen], g[~seen], n_boot)}


def fig_leakage(leak: dict, out: Path) -> None:
    groups = [("lesion_split", "Lesion-level split\n(all test)"), ("image_split", "Image-level split\n(all test)"),
              ("image_split_unseen", "Image-level:\nlesion unseen"), ("image_split_seen", "Image-level:\nlesion seen")]
    mets = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1")]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.26
    for j, (k, lab) in enumerate(mets):
        v = np.array([leak[g][k]["value"] for g, _ in groups])
        lo = np.array([leak[g][k]["ci"][0] for g, _ in groups])
        hi = np.array([leak[g][k]["ci"][1] for g, _ in groups])
        ax.bar(np.arange(4) + (j - 1) * w, v, w, yerr=[v - lo, hi - v], capsize=2, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels([t for _, t in groups], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Effect of lesion leakage (95% lesion-bootstrap CI)")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(p + 1e-9) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def collect_runs() -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = {}
    for d in sorted(RUNS_DIR.glob("*_s*")):
        m = re.match(r"^(.*)_s(\d+)$", d.name)
        if m and (d / "test_probs.npy").exists():
            runs.setdefault(m.group(1), []).append(d)
    return runs


def fmt(mean, sd=None, pct=True):
    f = 100 if pct else 1
    return f"{mean * f:.1f} ± {sd * f:.1f}" if sd is not None else f"{mean * f:.1f}"


# ---------------------------------------------------------------- figures
def fig_dataset(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["dx"], df["split"])[["train", "val", "test"]].reindex(CLASS_LABELS)
    ax = ct.plot(kind="bar", stacked=True, figsize=(8, 4), color=["#3b6ea5", "#e0a030", "#b5473f"])
    ax.set_ylabel("Images"); ax.set_xlabel("Class"); ax.set_title("HAM10000 class distribution by lesion-level split")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def fig_confusion(y, p, out: Path, title: str) -> None:
    cm = confusion_matrix(y, p.argmax(1), K)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, mat, f, vmax in ((ax[0], cm, "d", cm.max()), (ax[1], cmn, ".2f", 1.0)):
        a.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        a.set_xticks(range(K)); a.set_xticklabels(CLASS_LABELS); a.set_yticks(range(K)); a.set_yticklabels(CLASS_LABELS)
        for i in range(K):
            for j in range(K):
                a.text(j, i, format(mat[i, j], f), ha="center", va="center", fontsize=8,
                       color="white" if mat[i, j] > 0.6 * vmax else "black")
    for a, t in zip(ax, ("Counts", "Row-normalised (recall)")):
        a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(t)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def fig_roc(y, p, out: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    for k, c in enumerate(CLASS_LABELS):
        fpr, tpr = roc_curve(y == k, p[:, k])
        plt.plot(fpr, tpr, label=f"{c} (AUC {safe_auc(y == k, p[:, k]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(title)
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def fig_reliability(y, p_raw, p_cal, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, p, t in zip(ax, (p_raw, p_cal), ("Before temperature scaling", "After temperature scaling")):
        conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        a.plot([0, 1], [0, 1], "k--", lw=0.8); a.plot(xs, ys, "o-")
        a.set_title(f"{t}\nECE={ece_score(p, y):.3f}"); a.set_xlabel("Confidence"); a.set_ylabel("Accuracy")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def fig_per_class(boot: dict, m: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.27
    for j, met in enumerate(("precision", "recall", "f1")):
        v = np.array([m["per_class"][c][met] for c in CLASS_LABELS])
        ci = np.array([boot["per_class"][c][met] for c in CLASS_LABELS])
        ax.bar(np.arange(K) + (j - 1) * w, v, w, label=met.upper() if met == "f1" else met.title(),
               yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=2)
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASS_LABELS); ax.set_ylim(0, 1.05)
    ax.set_title("Per-class test performance (95% lesion-bootstrap CI)"); ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def fig_history(dirs: list[Path], out: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for d in dirs:
        h = pd.DataFrame(json.load(open(d / "history.json"))["history"])
        ax[0].plot(h.epoch, h.train_loss, "C0", alpha=.6); ax[0].plot(h.epoch, h.val_loss, "C1", alpha=.6)
        ax[1].plot(h.epoch, h.val_balanced_acc, "C2", alpha=.6)
    ax[0].set_title("Loss (blue: train, orange: validation)"); ax[1].set_title("Validation balanced accuracy")
    for a in ax: a.set_xlabel("Epoch")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


# ---------------------------------------------------------------- LaTeX export
CFG_MACRO = {"frozen_ce": "FrozenCE", "frozen_cw": "FrozenCW", "ft_ce": "FtCE", "ft_cw": "FtCW"}
MET_MACRO = {"accuracy": "Acc", "balanced_accuracy": "Bacc", "macro_f1": "MacroF", "weighted_f1": "WeightedF",
             "macro_auc": "MacroAUC", "mel_sensitivity": "MelSens", "mel_auc": "MelAUC",
             "malignant_sensitivity": "MalSens", "malignant_specificity": "MalSpec", "malignant_auc": "MalAUC",
             "ece": "ECE"}
CLASS_DESC = {"akiec": "Actinic keratosis / Bowen's disease", "bcc": "Basal cell carcinoma",
              "bkl": "Benign keratosis-like lesion", "df": "Dermatofibroma", "mel": "Melanoma",
              "nv": "Melanocytic nevus", "vasc": "Vascular lesion"}
AUC_KEYS = {"macro_auc", "mel_auc", "malignant_auc"}


def _v(key: str, x: float) -> str:
    """AUCs as 0.xxx, everything else as a percentage with one decimal."""
    return f"{x:.3f}" if key in AUC_KEYS else f"{100 * x:.1f}"


def _ci(key: str, c: list[float]) -> str:
    return f"{_v(key, c[0])}--{_v(key, c[1])}"


def write_latex(res: dict, df: pd.DataFrame, runs: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mac: dict[str, str] = {}
    for s in ("train", "val", "test"):
        part = df[df.split == s]
        mac[f"n{s.title()}Img"] = f"{len(part):,}"
        mac[f"n{s.title()}Les"] = f"{part.lesion_id.nunique():,}"
    mac["nImg"], mac["nLes"] = f"{len(df):,}", f"{df.lesion_id.nunique():,}"
    mac["nSeeds"] = str(res["n_seeds"])
    mac["temperature"] = f"{res['temperature']:.2f}"
    mac["eceRaw"], mac["eceCal"] = f"{res['raw_ece']:.3f}", f"{res['calibrated_ece']:.3f}"

    m, sc = res["metrics"], res["bootstrap_ci95"]["scalar"]
    for k, name in MET_MACRO.items():
        mac[f"p{name}"] = f"{m[k]:.3f}" if k == "ece" else _v(k, m[k])
        if k in sc and k != "ece":
            mac[f"p{name}CI"] = _ci(k, sc[k])
    for c in CLASS_LABELS:
        pc, ci = m["per_class"][c], res["bootstrap_ci95"]["per_class"][c]
        for met in ("precision", "recall", "f1"):
            mac[f"pc{c.title()}{met.title()}"] = f"{100 * pc[met]:.1f}"
        mac[f"pc{c.title()}RecallCI"] = f"{100 * ci['recall'][0]:.1f}--{100 * ci['recall'][1]:.1f}"
        mac[f"pc{c.title()}AUC"] = f"{pc['auc']:.3f}"

    for name, info in res["configs"].items():
        cm = CFG_MACRO.get(name)
        if not cm:
            continue
        for k, (mu, sd) in info["mean_sd"].items():
            mac[f"cfg{cm}{MET_MACRO[k]}"] = f"{mu:.3f}" if k == "ece" else _v(k, mu)
            mac[f"cfg{cm}{MET_MACRO[k]}SD"] = f"{sd:.3f}" if k in AUC_KEYS | {"ece"} else f"{100 * sd:.1f}"

    words = {90: "Ninety", 95: "NinetyFive"}
    for key, op in res["operating_points"].items():
        w = words[int(key[4:])]
        mac[f"op{w}Thr"] = f"{op['threshold']:.3f}"
        mac[f"op{w}Sens"] = f"{100 * op['test_sensitivity']:.1f}"
        mac[f"op{w}Spec"] = f"{100 * op['test_specificity']:.1f}"
        mac[f"op{w}Flag"] = f"{100 * op['test_flagged_fraction']:.1f}"

    for name, comp in res["paired_vs_primary_seed0"].items():
        cm = CFG_MACRO.get(name)
        if not cm:
            continue
        for k, d in comp.items():
            n = MET_MACRO[k]
            scale, f = (1, "{:+.3f}") if k in AUC_KEYS else (100, "{:+.1f}")
            mac[f"d{cm}{n}"] = f.format(scale * d["delta"])
            mac[f"d{cm}{n}CI"] = f"{f.format(scale * d['ci'][0])} to {f.format(scale * d['ci'][1])}"
            mac[f"d{cm}{n}P"] = "<0.001" if d["p"] <= 0.001 else f"{d['p']:.3f}"

    if res.get("leakage"):
        names = {"lesion_split": "Les", "image_split": "Img", "image_split_seen": "Seen",
                 "image_split_unseen": "Unseen"}
        for g, short in names.items():
            block = res["leakage"][g]
            mac[f"leak{short}N"] = f"{block['n']:,}"
            for k in LEAK_KEYS:
                mac[f"leak{short}{MET_MACRO[k]}"] = _v(k, block[k]["value"])
                mac[f"leak{short}{MET_MACRO[k]}CI"] = _ci(k, block[k]["ci"])
        lk = res["leakage"]
        mac["leakGapBacc"] = f"{100 * (lk['image_split']['balanced_accuracy']['value'] - lk['lesion_split']['balanced_accuracy']['value']):+.1f}"
        mac["leakGapAcc"] = f"{100 * (lk['image_split']['accuracy']['value'] - lk['lesion_split']['accuracy']['value']):+.1f}"
        mac["leakSeenPct"] = f"{100 * lk['image_split_seen']['n'] / lk['image_split']['n']:.0f}"

    for i, t in enumerate(res["top_confusions"][:4], 1):
        w = ["One", "Two", "Three", "Four"][i - 1]
        mac[f"conf{w}"] = (f"\\texttt{{{t['true']}}}$\\rightarrow$\\texttt{{{t['pred']}}} "
                           f"({t['count']}/{t['of']})")

    (out_dir / "numbers.tex").write_text(
        "% AUTO-GENERATED by src/analysis.py -- do not edit by hand\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()), encoding="utf-8")

    # -- Table: dataset
    ct = pd.crosstab(df.dx, df.split).reindex(CLASS_LABELS)
    rows = [f"\\texttt{{{c}}} & {CLASS_DESC[c]} & {ct.loc[c, 'train']} & {ct.loc[c, 'val']} & "
            f"{ct.loc[c, 'test']} & {ct.loc[c].sum()} \\\\" for c in CLASS_LABELS]
    tot = ct.sum()
    les = {s: df[df.split == s].lesion_id.nunique() for s in ("train", "val", "test")}
    (out_dir / "tab_dataset.tex").write_text("\n".join([
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Class & Diagnosis & Train & Val & Test & Total \\", r"\midrule", *rows, r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{Images}} & {tot['train']:,} & {tot['val']:,} & {tot['test']:,} & {tot.sum():,} \\\\",
        f"\\multicolumn{{2}}{{l}}{{Lesions}} & {les['train']:,} & {les['val']:,} & {les['test']:,} & "
        f"{df.lesion_id.nunique():,} \\\\", r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    # -- Table: configurations (mean +/- SD over seeds; single-seed rows without SD)
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "mel_sensitivity",
            "malignant_sensitivity", "malignant_specificity"]
    body = []
    for name in [n for n in CONFIG_TITLES if n in res["configs"]]:
        info = res["configs"][name]
        cells = []
        for k in keys:
            mu, sd = info["mean_sd"][k]
            cells.append(_v(k, mu) + (f"$\\pm${sd:.3f}" if k in AUC_KEYS else f"$\\pm${100 * sd:.1f}")
                         if info["n_seeds"] > 1 else _v(k, mu))
        bb = name == res["primary"]
        label = CONFIG_TITLES[name] + (" (ours)" if bb else "")
        body.append(f"{label} & {info['n_seeds']} & " + " & ".join(cells) + r" \\")
    (out_dir / "tab_configs.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Configuration & Seeds & Acc. & Bal. acc. & Macro-F1 & Macro-AUC & Mel. sens. & Malig. sens. & Malig. spec. \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    # -- Table: primary model, overall with CIs
    label_rows = [("Accuracy (\\%)", "accuracy"), ("Balanced accuracy (\\%)", "balanced_accuracy"),
                  ("Macro-F1 (\\%)", "macro_f1"), ("Weighted F1 (\\%)", "weighted_f1"),
                  ("Macro-AUC (one-vs-rest)", "macro_auc"), ("Melanoma sensitivity (\\%)", "mel_sensitivity"),
                  ("Melanoma AUC", "mel_auc"), ("Malignant sensitivity (\\%)", "malignant_sensitivity"),
                  ("Malignant specificity (\\%)", "malignant_specificity"), ("Malignant AUC", "malignant_auc")]
    (out_dir / "tab_primary.tex").write_text("\n".join([
        r"\begin{tabular}{lrr}", r"\toprule", r"Metric & Value & 95\% CI \\", r"\midrule",
        *[f"{lab} & {_v(k, m[k])} & {_ci(k, sc[k])} \\\\" for lab, k in label_rows],
        r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    # -- Table: per class
    pcrows = []
    for c in CLASS_LABELS:
        pc, ci = m["per_class"][c], res["bootstrap_ci95"]["per_class"][c]
        pcrows.append(f"\\texttt{{{c}}} & {pc['support']} & {100 * pc['precision']:.1f} & "
                      f"{100 * pc['recall']:.1f} ({100 * ci['recall'][0]:.0f}--{100 * ci['recall'][1]:.0f}) & "
                      f"{100 * pc['f1']:.1f} & {pc['auc']:.3f} \\\\")
    (out_dir / "tab_perclass.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrr}", r"\toprule", r"Class & $n$ & Prec. & Recall (95\% CI) & F1 & AUC \\",
        r"\midrule", *pcrows, r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="ft_cw")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"; fig_dir.mkdir(exist_ok=True)
    df = get_split()
    fig_dataset(df, fig_dir / "fig1_dataset_split.png")
    runs = collect_runs()
    if args.primary not in runs:
        raise SystemExit(f"primary config '{args.primary}' has no runs in {RUNS_DIR}")

    y_all = df["label"].values
    groups_all = df["lesion_id"].values
    summary, table1 = {}, []
    for name, dirs in runs.items():
        per_seed = []
        for d in dirs:
            te = np.load(d / "test_idx.npy")
            per_seed.append(metrics(y_all[te], np.load(d / "test_probs.npy")))
        keys = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "mel_sensitivity",
                "malignant_sensitivity", "malignant_specificity", "malignant_auc", "ece"]
        agg = {k: (float(np.mean([m[k] for m in per_seed])), float(np.std([m[k] for m in per_seed])))
               for k in keys}
        summary[name] = {"n_seeds": len(dirs), "mean_sd": agg}
        table1.append((name, len(dirs), agg))

    lines = ["| Configuration | Seeds | Accuracy | Balanced acc. | Macro-F1 | Macro-AUC | Mel. sens. | "
             "Malig. sens. | Malig. spec. | Malig. AUC | ECE |", "|---|---:|" + "---:|" * 9]
    order = [n for n in CONFIG_TITLES if n in runs] + [n for n in runs if n not in CONFIG_TITLES]
    for name in order:
        _, ns, a = next(t for t in table1 if t[0] == name)
        lines.append(f"| {CONFIG_TITLES.get(name, name)} | {ns} | " + " | ".join(
            fmt(*a[k], pct=(k != "ece") or True) for k in a) + " |")
    (OUT_DIR / "table1_configs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- primary configuration: seed ensemble, bootstrap CIs, calibration
    dirs = runs[args.primary]
    te = np.load(dirs[0] / "test_idx.npy"); va = np.load(dirs[0] / "val_idx.npy")
    p_test = np.mean([np.load(d / "test_probs.npy") for d in dirs], axis=0)
    p_val = np.mean([np.load(d / "val_probs.npy") for d in dirs], axis=0)
    y_te, y_va = y_all[te], y_all[va]
    m = metrics(y_te, p_test)
    boot = bootstrap(y_te, p_test, groups_all[te], n=args.boot)
    t = fit_temperature(p_val, y_va)
    p_cal = apply_temperature(p_test, t)
    m_cal = metrics(y_te, p_cal)

    fig_confusion(y_te, p_test, fig_dir / "fig2_confusion_matrix.png",
                  f"{CONFIG_TITLES.get(args.primary, args.primary)} — test set ({len(te)} images)")
    fig_roc(y_te, p_test, fig_dir / "fig3_roc_curves.png", "One-vs-rest ROC (test set)")
    fig_per_class(boot, m, fig_dir / "fig4_per_class.png")
    fig_reliability(y_te, p_test, p_cal, fig_dir / "fig5_calibration.png")
    fig_history(dirs, fig_dir / "fig6_training_curves.png", CONFIG_TITLES.get(args.primary, args.primary))

    cm = confusion_matrix(y_te, p_test.argmax(1), K)
    pairs = sorted(((int(cm[i, j]), CLASS_LABELS[i], CLASS_LABELS[j], int(cm[i].sum()))
                    for i in range(K) for j in range(K) if i != j and cm[i, j]), reverse=True)[:8]

    sc = boot["scalar"]
    rows = [("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"), ("Macro-F1", "macro_f1"),
            ("Weighted F1", "weighted_f1"), ("Macro-AUC (OvR)", "macro_auc"),
            ("Melanoma sensitivity", "mel_sensitivity"), ("Melanoma AUC", "mel_auc"),
            ("Malignant/pre-malignant sensitivity", "malignant_sensitivity"),
            ("Malignant/pre-malignant specificity", "malignant_specificity"),
            ("Malignant/pre-malignant AUC", "malignant_auc")]
    t2 = ["| Metric | Value | 95% CI |", "|---|---:|---:|"]
    for label, k in rows:
        t2.append(f"| {label} | {m[k]:.3f} | {sc[k][0]:.3f}–{sc[k][1]:.3f} |")
    t2 += ["", "| Class | Precision | Recall | F1 | AUC | Support |", "|---|---:|---:|---:|---:|---:|"]
    for c in CLASS_LABELS:
        pc, ci = m["per_class"][c], boot["per_class"][c]
        cell = lambda k: f"{pc[k]:.3f} ({ci[k][0]:.2f}–{ci[k][1]:.2f})"
        t2.append(f"| `{c}` | {cell('precision')} | {cell('recall')} | {cell('f1')} | {cell('auc')} | {pc['support']} |")
    (OUT_DIR / "table2_primary.md").write_text("\n".join(t2) + "\n", encoding="utf-8")

    # ---- triage operating points (threshold chosen on validation, applied to test)
    ops = {f"sens{int(100 * s)}": operating_point(y_va, p_val, y_te, p_test, s) for s in (0.90, 0.95)}

    # ---- paired comparisons: every other configuration vs the primary one (seed 0 vs seed 0)
    cmp_keys = ["balanced_accuracy", "macro_f1", "macro_auc", "mel_sensitivity", "malignant_sensitivity"]
    p_prim0 = np.load(dirs[0] / "test_probs.npy")
    comparisons = {}
    for name, ds in runs.items():
        if name != args.primary and np.array_equal(np.load(ds[0] / "test_idx.npy"), te):
            comparisons[name] = paired_delta(y_te, p_prim0, np.load(ds[0] / "test_probs.npy"),
                                             groups_all[te], cmp_keys, n=args.boot)

    leakage = None
    leak_name = f"{args.primary}_imgsplit"
    if leak_name in runs:
        leakage = leakage_analysis(runs[leak_name][0], y_te, p_prim0, groups_all[te], args.boot)
        fig_leakage(leakage, fig_dir / "fig8_leakage.png")

    out = {"primary": args.primary, "n_seeds": len(dirs), "test_images": int(len(te)),
           "leakage": leakage,
           "test_lesions": int(len(set(groups_all[te]))), "metrics": m, "bootstrap_ci95": boot,
           "temperature": t, "calibrated_ece": m_cal["ece"], "raw_ece": m["ece"],
           "operating_points": ops, "paired_vs_primary_seed0": comparisons,
           "top_confusions": [{"count": a, "true": b, "pred": c, "of": d} for a, b, c, d in pairs],
           "configs": summary}
    write_latex(out, df, runs, OUT_DIR / "latex")
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("primary", "n_seeds", "test_images", "temperature", "raw_ece",
                                          "calibrated_ece")}, indent=2))
    print((OUT_DIR / "table1_configs.md").read_text()); print("".join(l + "\n" for l in t2))


if __name__ == "__main__":
    main()
