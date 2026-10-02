"""Dependency-light (numpy only) metrics and splitting utilities.

Kept free of scikit-learn / scipy so the pipeline also runs on locked-down machines where
compiled scientific-Python extensions are blocked. Implementations follow the standard
definitions (verified against known values in tests/test_stats.py).
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ splitting
def stratified_group_folds(labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each *group* (e.g. lesion) to one of ``n_folds`` folds, balancing class proportions.

    Greedy scheme in the spirit of scikit-learn's StratifiedGroupKFold: groups are visited from
    the most class-skewed to the least (random tie-breaking) and placed into the fold whose class
    distribution deviates least from the global one. Returns a fold id per sample.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(uniq), len(classes)))
    np.add.at(counts, (inv, np.searchsorted(classes, labels)), 1)
    total = counts.sum(0)
    order = np.argsort(-(counts.std(1) + rng.random(len(uniq)) * 1e-6))
    fold_counts = np.zeros((n_folds, len(classes)))
    fold_size = np.zeros(n_folds)
    assign = np.zeros(len(uniq), dtype=int)
    for g in order:
        best, best_cost = 0, np.inf
        for f in rng.permutation(n_folds):
            trial = fold_counts.copy()
            trial[f] += counts[g]
            cost = np.mean(np.std(trial / total, axis=0)) + 1e-3 * (fold_size[f] + counts[g].sum())
            if cost < best_cost:
                best, best_cost = f, cost
        assign[g] = best
        fold_counts[best] += counts[g]
        fold_size[best] += counts[g].sum()
    return assign[inv]


# ------------------------------------------------------------------ metrics
def confusion_matrix(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def prf(y: np.ndarray, pred: np.ndarray, k: int):
    cm = confusion_matrix(y, pred, k)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    return precision, recall, f1, cm.sum(1)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    _, recall, _, support = prf(y, pred, k)
    return float(recall[support > 0].mean())


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    _, _, f1, support = prf(y, pred, k)
    return float(f1[support > 0].mean())


def weighted_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    _, _, f1, support = prf(y, pred, k)
    return float((f1 * support).sum() / support.sum())


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties handled (same as scipy.stats.rankdata(method='average'))."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic; NaN if only one class is present."""
    y_bin = y_bin.astype(bool)
    n_pos, n_neg = int(y_bin.sum()), int((~y_bin).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(score)
    return float((r[y_bin].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_bin: np.ndarray, score: np.ndarray):
    order = np.argsort(-score, kind="mergesort")
    y = y_bin[order].astype(float)
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    distinct = np.r_[np.where(np.diff(score[order]))[0], len(y) - 1]
    return np.r_[0, fps[distinct] / fps[-1]], np.r_[0, tps[distinct] / tps[-1]]


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising validation NLL (golden-section search on log T)."""
    logp = np.log(probs + 1e-9)

    def nll(logt: float) -> float:
        z = logp / np.exp(logt)
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-lp[np.arange(len(y)), y].mean())

    lo, hi, phi = np.log(0.05), np.log(10.0), (np.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    for _ in range(60):
        if nll(a) < nll(b):
            hi, b = b, a
            a = hi - phi * (hi - lo)
        else:
            lo, a = a, b
            b = lo + phi * (hi - lo)
    return float(np.exp((lo + hi) / 2))
