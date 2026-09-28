#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recorder — the durable, structured record of what a run did.

This implements the hook interface the training loop calls (log_epoch,
log_grad_norms, log_logit_hist, log_stage). Everything is written to the run
directory as files so analysis never re-runs training:

  run_dir/
    config.json                         # exact config used
    <category>/
      stage.jsonl                       # one line per stage event (dataset, fit-start, test, ...)
      epochs.jsonl                      # one line per epoch: losses + val metrics + best flag
      grad_norms.jsonl                  # one line per epoch: per-layer grad L2 norms
      logit_hist.jsonl                  # one line per epoch: logit histograms split by class
      scores_test.npz                   # raw test logits/labels/paths (re-scoreable offline)
      summary.json                      # final per-category roll-up

The .jsonl-per-stage design is what lets you (and me) read exactly one stage at
a time. You paste me a category's epochs.jsonl + logit_hist.jsonl and I can see
precisely how the model moved without console scrollback.

Histograms are computed split by class (normal vs anomaly) because the question
that matters for a detector is "are the two classes separating in logit space",
and a single merged histogram hides that.
"""

import os
import json
from typing import Optional

import numpy as np

from utils.io import ensure_dir, save_json, save_arrays


class Recorder:
    def __init__(self, run_dir: str, cfg: dict):
        self.run_dir = ensure_dir(run_dir)
        save_json(cfg, os.path.join(run_dir, "config.json"))
        self._open = {}  # cache of file handles per (category, stream)

    # ---------------------------------------------------------- helpers ----
    def _cat_dir(self, category: str) -> str:
        return ensure_dir(os.path.join(self.run_dir, category))

    def _append_jsonl(self, category: str, stream: str, obj: dict) -> None:
        path = os.path.join(self._cat_dir(category), f"{stream}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(obj, default=_jsonable) + "\n")

    # ------------------------------------------------------- stage hooks ----
    def log_stage(self, category: str, name: str, payload: dict) -> None:
        """Coarse milestones: dataset profile, fit-start, fit-end, test."""
        rec = {"stage": name, **payload}
        self._append_jsonl(category, "stage", rec)

    def log_epoch(self, category: str, epoch: int, payload: dict) -> None:
        rec = {"epoch": epoch, **payload}
        self._append_jsonl(category, "epochs", rec)

    def log_grad_norms(self, category: str, epoch: int, grad: dict) -> None:
        self._append_jsonl(category, "grad_norms", {"epoch": epoch, **grad})

    def log_logit_hist(self, category: str, epoch: int, logits, labels,
                       bins: int = 30) -> None:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)
        rec = {"epoch": epoch}
        # shared bin edges so epochs are comparable
        if logits.size:
            lo, hi = float(logits.min()), float(logits.max())
            edges = np.linspace(lo, hi, bins + 1)
            rec["edges"] = edges.tolist()
            for cls, key in [(0, "normal"), (1, "anomaly")]:
                vals = logits[labels == cls]
                h, _ = np.histogram(vals, bins=edges)
                rec[key] = {"count": int(vals.size),
                            "mean": float(vals.mean()) if vals.size else None,
                            "std": float(vals.std()) if vals.size else None,
                            "hist": h.tolist()}
        self._append_jsonl(category, "logit_hist", rec)

    # -------------------------------------------------- raw + summary ----
    def save_test_scores(self, category: str, logits, labels, paths,
                         features=None) -> None:
        path = os.path.join(self._cat_dir(category), "scores_test.npz")
        extra = {} if features is None else {"features": np.asarray(features)}
        save_arrays(path, logits=np.asarray(logits),
                    labels=np.asarray(labels),
                    paths=np.asarray(paths, dtype=object), **extra)

    def save_split_scores(self, category: str, split: str, logits, labels, paths,
                          features=None) -> None:
        """Raw scores for a NON-test split, so thresholding becomes an offline job.

        The training loop already computes validation scores every epoch to pick the
        best epoch and the decision threshold -- and then discards them. That made
        every alternative threshold rule a 35-minute retrain instead of a second of
        arithmetic. Writing them once turns the whole family of threshold questions
        (Youden on val, percentile of train-normals, temperature scaling, ECE on a
        non-test split) into offline analysis needing no GPU.

        Written AFTER the best checkpoint is reloaded, so these are the scores of the
        model that actually gets scored on test -- not of whatever the weights happened
        to be at the final epoch.
        """
        path = os.path.join(self._cat_dir(category), f"scores_{split}.npz")
        extra = {} if features is None else {"features": np.asarray(features)}
        save_arrays(path, logits=np.asarray(logits),
                    labels=np.asarray(labels),
                    paths=np.asarray(paths, dtype=object), **extra)

    def save_summary(self, category: str, summary: dict) -> None:
        save_json(summary, os.path.join(self._cat_dir(category), "summary.json"))

    def save_run_summary(self, summary: dict) -> None:
        save_json(summary, os.path.join(self.run_dir, "run_summary.json"))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
