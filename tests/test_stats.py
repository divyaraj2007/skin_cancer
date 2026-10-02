"""Sanity tests for src/stats.py (run: python -m pytest tests or python tests/test_stats.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import (auc, balanced_accuracy, confusion_matrix, fit_temperature, macro_f1,
                       roc_curve, stratified_group_folds, weighted_f1)


def test_auc_known_values():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75  # classic textbook example
    assert auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5  # ties
    assert np.isnan(auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + 0.3 * y, 2)  # rounding creates ties
    pos, neg = s[y == 1], s[y == 0]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert np.isclose(auc(y, s), brute)


def test_roc_endpoints():
    fpr, tpr = roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert (fpr[0], tpr[0]) == (0, 0) and (fpr[-1], tpr[-1]) == (1, 1)


def test_classification_metrics():
    y = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 0])
    assert confusion_matrix(y, pred, 3).tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert np.isclose(balanced_accuracy(y, pred, 3), (2 / 3 + 1 + 0) / 3)
    # class0: P=2/3 R=2/3 F1=2/3; class1: P=2/3 R=1 F1=0.8; class2: 0
    assert np.isclose(macro_f1(y, pred, 3), (2 / 3 + 0.8 + 0) / 3)
    assert np.isclose(weighted_f1(y, pred, 3), (3 * 2 / 3 + 2 * 0.8 + 0) / 6)


def test_group_folds_have_no_group_overlap_and_balance():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), rng.integers(1, 4, 400))
    labels = np.array([g % 4 for g in groups])
    folds = stratified_group_folds(labels, groups, 5, seed=1)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    for f in range(5):
        frac = np.bincount(labels[folds == f], minlength=4) / np.bincount(labels)
        assert np.allclose(frac, 0.2, atol=0.05)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 20000, 5
    logits = rng.normal(size=(n, k)) * 1.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])  # labels drawn from the model: perfectly calibrated
    over = np.exp(logits * 3) / np.exp(logits * 3).sum(1, keepdims=True)  # sharpened x3
    assert abs(fit_temperature(p, y) - 1.0) < 0.1
    assert abs(fit_temperature(over, y) - 3.0) < 0.3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
