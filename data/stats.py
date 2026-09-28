#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset profiler.

Runs BEFORE training so we know what we are about to train on, and so a
degenerate setup is caught up front rather than discovered from a meaningless
AUROC. Produces a JSON-able manifest per (dataset, category):

  - counts per split (train/val/test)
  - class balance (normal vs anomaly) per split
  - flags:
      single_class_train  -> binary classifier is ill-posed (e.g. MVTec, no injection)
      no_val              -> model selection will fall back to test (leak) if used
      severe_imbalance    -> consider class weights

This is the first thing the instrumentation layer records for a run.
"""

from typing import Dict


def profile_split(labels) -> Dict:
    n = len(labels)
    pos = int(sum(labels))
    neg = n - pos
    return {
        "count": n,
        "normal": neg,
        "anomaly": pos,
        "anomaly_ratio": (pos / n) if n else 0.0,
    }


def profile_dataset(dataset, name: str, category: str, use_validation: bool,
                    test_anomaly_floor: int = 15, val_anomaly_floor: int = 10,
                    train_anomaly_floor: int = 8) -> Dict:
    train = profile_split(dataset.labels("train"))
    test = profile_split(dataset.labels("test")) if len(dataset.test_data) else \
        {"count": 0, "normal": 0, "anomaly": 0, "anomaly_ratio": 0.0}
    val = profile_split(dataset.labels("val")) if (use_validation and len(dataset.val_data)) else None

    flags = {
        "single_class_train": dataset.is_single_class("train"),
        "no_val": use_validation and (val is None or val["count"] == 0),
        "severe_imbalance": (train["anomaly_ratio"] < 0.05 or train["anomaly_ratio"] > 0.95)
                            and not dataset.is_single_class("train"),
        # ABSOLUTE-COUNT flag, distinct from the RATIO test above and added because
        # they come apart at exactly the wrong category. toothbrush has ~48 train
        # normals, so 2 anomalies is a 4% ratio -- which looks unremarkable -- while
        # being the smallest absolute count in all 27. Ratio measures balance; what
        # breaks training here is how many distinct defects the model ever sees.
        # Floor is 8 because wood at 8 (measured 6 after the adaptive val share) is
        # the smallest count this project has trained successfully.
        "low_train_anomalies": train["anomaly"] < train_anomaly_floor,
        # §2.1.2 low-count flags: counts below the floors make numbers noisy.
        "low_test_anomalies": test["anomaly"] < test_anomaly_floor,
        "low_val_anomalies": (val is not None and val["anomaly"] < val_anomaly_floor),
    }

    manifest = {
        "dataset": name,
        "category": category,
        "splits": {"train": train, "val": val, "test": test},
        "flags": flags,
        "floors": {"test_anomaly_floor": test_anomaly_floor,
                   "val_anomaly_floor": val_anomaly_floor,
                   "train_anomaly_floor": train_anomaly_floor},
    }
    return manifest


def warnings_from_manifest(manifest: Dict, mitigation: Dict = None) -> list:
    """Turn flags into human-readable warnings for the console + record.

    `mitigation` carries what is already switched on (e.g. class weights), so a
    warning states the REMAINING risk rather than advising a setting that is
    already the default. A warning that recommends a no-op trains the reader to
    ignore warnings.
    """
    msgs = []
    mitigation = mitigation or {}
    f = manifest["flags"]
    cat = manifest["category"]
    if f["single_class_train"]:
        msgs.append(
            f"[{cat}] TRAIN IS SINGLE-CLASS: a supervised binary classifier cannot "
            f"learn here. (MVTec needs num_train_anomalies>0, or use the unsupervised path.)")
    if f["no_val"]:
        msgs.append(
            f"[{cat}] use_validation=true but no val samples -> selection would fall "
            f"back to test (leak). Check val_split / data.")
    if f["severe_imbalance"]:
        r = manifest["splits"]["train"]["anomaly_ratio"]
        if mitigation.get("class_weights"):
            msgs.append(f"[{cat}] severe class imbalance (train anomaly_ratio={r:.3f}) — "
                        f"class weights ARE enabled; imbalance this extreme can still "
                        f"destabilise selection, so read the val metrics with that in mind.")
        elif mitigation.get("oversample"):
            msgs.append(f"[{cat}] severe class imbalance (train anomaly_ratio={r:.3f}) — "
                        f"oversampling is active (class weights off, per the §2.4 "
                        f"double-correction rule).")
        else:
            msgs.append(f"[{cat}] severe class imbalance (train anomaly_ratio={r:.3f}) and NO "
                        f"mitigation active -> enable train.use_class_weights or an "
                        f"oversampling scenario.")
    if f.get("low_test_anomalies"):
        n = manifest["splits"]["test"]["anomaly"]
        floor = manifest.get("floors", {}).get("test_anomaly_floor", 15)
        msgs.append(f"[{cat}] LOW TEST ANOMALIES ({n} < floor {floor}) -> test AUROC is noisy; "
                    f"reliability-caveat this result.")
    if f.get("low_val_anomalies"):
        v = manifest["splits"].get("val") or {}
        n = v.get("anomaly", 0)
        floor = manifest.get("floors", {}).get("val_anomaly_floor", 10)
        msgs.append(f"[{cat}] LOW VAL ANOMALIES ({n} < floor {floor}) -> selection is noisy.")
    return msgs
