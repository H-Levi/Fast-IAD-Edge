#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metric math — the single place it lives.

Centralizing this is a correctness measure, not just tidiness: the threshold a
classifier uses must be chosen on VALIDATION data, then applied unchanged to
test. The earlier code chose the F1-optimal threshold on the test labels
themselves (oracle thresholding), which inflates F1/precision/recall. Here the
threshold finders take whatever split you hand them; the engine hands them VAL.

Inputs are raw logits (pre-sigmoid). AUROC/AUPR are threshold-free and operate
on logits directly (monotonic with probability). F1/precision/recall need a
threshold. ECE needs probabilities, so we sigmoid internally for it.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, f1_score,
                             precision_score, recall_score)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------- threshold finders ----
def threshold_youden(labels, logits) -> float:
    """Threshold maximizing TPR - FPR. Chosen on the split you pass (use VAL)."""
    fpr, tpr, thr = roc_curve(labels, logits)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_f1(labels, logits) -> float:
    """Threshold maximizing F1 on the split you pass (use VAL)."""
    prec, rec, thr = precision_recall_curve(labels, logits)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) != 0)
    # precision_recall_curve returns thr of length len(prec)-1
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.0


def pick_threshold(labels, logits, strategy: str = "youden") -> float:
    if len(set(labels)) < 2:
        return 0.0  # undefined; engine will warn via stats
    return threshold_youden(labels, logits) if strategy == "youden" \
        else threshold_f1(labels, logits)


# -------------------------------------------------------- ranking metrics ----
def safe_auroc(labels, logits) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, logits))


def safe_aupr(labels, logits) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, logits))


# ------------------------------------------------------ calibration (ECE) ----
def expected_calibration_error(labels, logits, n_bins: int = 15) -> float:
    """ECE over predicted-positive probability. Lower is better-calibrated."""
    probs = _sigmoid(np.asarray(logits, dtype=np.float64))
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = (labels[mask] == 1).mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ----------------------------------------------------------- aggregate ----
def compute_all(labels, logits, threshold: Optional[float] = None,
                strategy: str = "youden", which=None) -> Dict[str, float]:
    """Full metric dict for a split.

    threshold: if None, it is chosen on THIS split with `strategy`. The engine
    passes the VAL-chosen threshold when scoring TEST, so test metrics never see
    their own labels for thresholding.
    """
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    which = which or ["auroc", "aupr", "f1", "precision", "recall", "ece"]

    if threshold is None:
        threshold = pick_threshold(labels, logits, strategy)
    preds = (logits >= threshold).astype(int)

    out = {"threshold": float(threshold)}
    if "auroc" in which:
        out["auroc"] = safe_auroc(labels, logits)
    if "aupr" in which:
        out["aupr"] = safe_aupr(labels, logits)
    if "f1" in which:
        out["f1"] = float(f1_score(labels, preds, zero_division=0))
    if "precision" in which:
        out["precision"] = float(precision_score(labels, preds, zero_division=0))
    if "recall" in which:
        out["recall"] = float(recall_score(labels, preds, zero_division=0))
    if "ece" in which:
        out["ece"] = expected_calibration_error(labels, logits)
    return out
