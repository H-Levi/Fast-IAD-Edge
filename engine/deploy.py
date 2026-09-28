#!/usr/bin/env python
"""Deployed score of the rule model (A-071, K-030).

prior_corrected(logits, pos_weight) = logits - log(pos_weight). BCE with pos_weight w moves the optimal logit by +log(w);
subtracting it restores calibration (VisA lead ECE 0.067 -> 0.040, A-071). It is MONOTONE: AUROC, AUPR and every
rank-based threshold (recall / false alarms at the conformal threshold) are unchanged — only probabilities move.
pos_weight is recorded per category in stage.jsonl ('fit_start') and is fixed by the training split.
"""
import math

import numpy as np


def prior_corrected(logits, pos_weight):
    w = float(pos_weight) if pos_weight else 1.0
    return np.asarray(logits, dtype=float) - math.log(w)
