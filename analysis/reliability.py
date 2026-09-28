#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reliability tiers, caveats, and the cross-tier averaging guard (§2.2 / §2.4).

Two datasets, two different epistemic footings:

  NATIVE   (VisA)  — anomalies are labelled in the dataset's own train split.
                     The test set is untouched. Numbers stand on their own.
  INJECTED (MVTec) — train has no anomalies, so we MOVE some out of test into
                     train/val. The test set is therefore modified by us, and the
                     result depends on our injection choice (fraction, seed).

These are not interchangeable. Averaging a VisA number with an MVTec-injected
number produces a figure that means nothing — it is not "the method's AUROC",
it is a blend of two different experiments. So:

  * every result carries its tier,
  * means are computed PER TIER and never blended (cross_tier_mean refuses),
  * low-count flags from the census/profiler propagate as CAVEATS onto the
    result, so a category with 3 val anomalies is never read as if it were solid.

This module is pure labelling/aggregation logic — no training, no metrics math.
"""

from typing import Dict, List

import numpy as np

TIER_NATIVE = "native"
TIER_INJECTED = "injected"

# Which dataset sits in which tier. Explicit, not inferred, so a new dataset
# must declare itself rather than silently defaulting to "trustworthy".
DATASET_TIERS = {
    "visa": TIER_NATIVE,
    "mvtec": TIER_INJECTED,
}


class CrossTierAveragingError(RuntimeError):
    """Raised on any attempt to average results from different reliability tiers."""
    pass


def dataset_tier(dataset_name: str) -> str:
    t = DATASET_TIERS.get(dataset_name)
    if t is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}': declare its reliability tier in "
            f"analysis/reliability.DATASET_TIERS before reporting results from it.")
    return t


# ------------------------------------------------------------- caveats ----
CAVEAT_TEXT = {
    "low_test_anomalies": "few test anomalies -> test metrics are noisy",
    "low_val_anomalies": "few val anomalies -> model selection is noisy",
    "severe_imbalance": "severe class imbalance in train",
    "low_train_anomalies": "few DISTINCT train anomalies -> the model may never see "
                           "enough defect variety, regardless of class ratio",
    "inverted": "AUROC BELOW CHANCE -> the model ranks defects below normals; this is "
                "a sign inversion, not noise, and the number is not a weak result",
    "single_class_train": "train is single-class (result invalid)",
    "no_val": "no validation split (selection would fall back to test)",
}


def caveats_from_flags(flags: Dict) -> List[str]:
    """Turn profiler flags into short caveat keys that travel with the result."""
    return [k for k, v in (flags or {}).items() if v and k in CAVEAT_TEXT]


def caveat_line(caveats: List[str]) -> str:
    if not caveats:
        return ""
    return "; ".join(CAVEAT_TEXT.get(c, c) for c in caveats)


def is_reportable(caveats: List[str], collapsed=None, selection_failed=False) -> bool:
    """A result is reportable if it is not structurally invalid.

    Invalidating conditions (not merely noisy): single-class train, a collapsed
    run, or a failed selection. Low counts make a number NOISY, not invalid, so
    they caveat it rather than disqualify it.
    """
    if collapsed:
        return False
    if selection_failed:
        return False
    return "single_class_train" not in (caveats or [])


def inverted_flag(auroc) -> bool:
    """AUROC strictly below chance: the model ranks defects BELOW normals.

    The collapse detector reads the logit-gap TRAJECTORY, so a model with a stable,
    consistent gap of the WRONG SIGN passes it. carpet came back at 0.3157 on Run 0
    avg -- flipping its sign would give 0.6843 -- while its neighbours were above
    0.95, and nothing flagged it. That is systematic, not noise.

    Deliberately a CAVEAT and NOT disqualifying. Under F-L23a the headline mean
    includes every category, because a mean over 25 of 27 is not comparable to
    anyone else's mean over 27. A genuine failure belongs in the number; excluding
    it would flatter the result. It must be NAMED, not hidden.
    """
    try:
        return float(auroc) < 0.5
    except (TypeError, ValueError):
        return False


# ------------------------------------------------ tier-grouped averaging ----
def group_by_tier(per_category: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    """Split {'<dataset>/<backbone>/<category>': metrics} into {tier: {key: metrics}}.

    The key format produced by run_train is 'dataset/backbone/category'.
    """
    out: Dict[str, Dict[str, Dict]] = {}
    for key, metrics in per_category.items():
        ds = key.split("/")[0]
        tier = dataset_tier(ds)
        out.setdefault(tier, {})[key] = metrics
    return out


def group_by_tier_and_model(per_category: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    """Split into {tier: {backbone: {key: metrics}}}.

    Tier is not the only axis that must not be blended. A first sealed baseline
    produced MEAN[injected] over 4 categories x 3 backbones — averaging
    EfficientNet with SqueezeNet, which describes no model that exists. That is
    the same category error as blending datasets, on a different axis, so means
    are grouped by BOTH.
    """
    out: Dict[str, Dict[str, Dict]] = {}
    for key, metrics in per_category.items():
        parts = key.split("/")
        ds = parts[0]
        backbone = parts[1] if len(parts) > 2 else "unknown"
        tier = dataset_tier(ds)
        out.setdefault(tier, {}).setdefault(backbone, {})[key] = metrics
    return out


def mean_within_tier(metrics_map: Dict[str, Dict], keys=None) -> Dict:
    """Mean of each metric across categories WITHIN one tier+model group."""
    keys = keys or ["auroc", "aupr", "f1", "precision", "recall", "ece"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_map.values()
                if k in m and m[k] is not None
                and not (isinstance(m[k], float) and np.isnan(m[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_categories"] = len(metrics_map)
    return out


def tiered_means(per_category: Dict[str, Dict], keys=None) -> Dict[str, Dict]:
    """Means grouped by {tier: {backbone: mean}}.

    NEVER returns a single number blended across tiers, nor across backbones.
    """
    grouped = group_by_tier_and_model(per_category)
    return {tier: {bb: mean_within_tier(m, keys) for bb, m in by_bb.items()}
            for tier, by_bb in grouped.items()}


def cross_tier_mean(per_category: Dict[str, Dict]) -> None:
    """Explicitly refuses. Exists so the mistake raises instead of happening."""
    tiers = set(group_by_tier(per_category).keys())
    if len(tiers) > 1:
        raise CrossTierAveragingError(
            f"Refusing to average across reliability tiers {sorted(tiers)}: a VisA "
            f"(native) number and an MVTec (injected) number are different "
            f"experiments. Report them separately (see tiered_means).")
