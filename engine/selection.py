#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model selection — on validation, never on test.

This is the component that closes the leak in the original code (best model =
max test AUROC over epochs). The Selector watches a monitored quantity computed
on VALIDATION, decides each epoch whether the model is the best so far and
whether to early-stop, and remembers the VAL-chosen decision threshold to apply
to test exactly once at the end.

Usage in the loop:
    sel = Selector(monitor='val_auroc', mode='max', patience=10)
    for epoch:
        train_one_epoch(...)
        val_raw = run_inference(model, val_loader, ...)
        val_metrics = compute_all(val_raw['labels'], val_raw['logits'], strategy=...)
        d = sel.update(epoch, val_metrics['auroc'], extra={'threshold': val_metrics['threshold']})
        if d['is_best']:  save_checkpoint()
        if d['should_stop']: break
    # later: test thresholded at sel.best_extra['threshold']
"""

import numpy as np


class Selector:
    """Val-based best-epoch selection with a SATURATION TIE-BREAK.

    Why the tie-break exists (observed, not hypothetical). With only ~10-12
    validation anomalies, `val_aupr` reaches exactly 1.0000 and then stays there.
    Because improvement is strict (`>`), no later epoch can ever win, so the
    checkpoint locks onto the FIRST epoch to saturate and patience starts
    counting immediately. Measured on the 3-seed baseline: mvtec/bottle hit
    1.0000 at epoch 9, epochs 10-19 also scored 1.0000, and the run stopped at
    epoch 19 having kept epoch 9. visa/pcb2 behaved identically. Among ten
    equally-scoring epochs the one kept was chosen by ARRIVAL ORDER, not quality.

    The fix: when the monitored value TIES the best, fall back to a secondary
    signal (lower-is-better, normally val_loss) to decide. val_loss is not
    removed from anywhere -- it was already computed and logged every epoch and
    simply never consulted for selection. This only consults it when the primary
    monitor cannot distinguish two epochs.

    A tie-break win updates the checkpoint but deliberately does NOT reset
    patience, so a saturated run still stops on schedule and costs no extra
    epochs. Set `selection.tiebreak: none` to restore the old behaviour exactly.
    """

    def __init__(self, monitor: str = "val_auroc", mode: str = "max",
                 patience: int = 10, min_delta: float = 0.0,
                 tiebreak: bool = True, tie_eps: float = 1e-12):
        assert mode in ("max", "min")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.tiebreak = tiebreak
        self.tie_eps = tie_eps

        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = -1
        self.best_extra = {}
        self.best_tiebreak = None
        self.counter = 0
        self.tiebreak_wins = 0

    def _improved(self, value: float) -> bool:
        if np.isnan(value):
            return False
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def _tied(self, value: float) -> bool:
        if np.isnan(value) or not np.isfinite(self.best_value):
            return False
        return abs(value - self.best_value) <= self.tie_eps

    def update(self, epoch: int, value: float, extra: dict = None,
               tiebreak_value: float = None) -> dict:
        improved = self._improved(value)

        tie_win = False
        if (not improved and self.tiebreak and self._tied(value)
                and tiebreak_value is not None and not np.isnan(tiebreak_value)):
            tie_win = (self.best_tiebreak is None
                       or tiebreak_value < self.best_tiebreak - self.tie_eps)

        if improved or tie_win:
            self.best_value = float(value)
            self.best_epoch = epoch
            self.best_extra = dict(extra or {})
            if tiebreak_value is not None and not np.isnan(tiebreak_value):
                self.best_tiebreak = float(tiebreak_value)
        if tie_win:
            self.tiebreak_wins += 1

        # patience tracks the PRIMARY monitor only: a tie-break keeps a better
        # checkpoint without buying the run extra epochs.
        if improved:
            self.counter = 0
        else:
            self.counter += 1

        return {
            "is_best": improved or tie_win,
            "improved": improved,
            "tiebreak_win": tie_win,
            "should_stop": self.counter >= self.patience,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "patience_left": max(self.patience - self.counter, 0),
        }
