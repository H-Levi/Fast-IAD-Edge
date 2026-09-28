#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Operating-point loss (analysis P5/P4, 2026-09-23): train for the LOW-false-alarm end of the ROC curve.

Why. At a 1% false-alarm budget the threshold sits at the highest-scoring validation/calibration normals, so recall
depends on how many defects score above a handful of odd-but-good normals -- not on average separation. BCE optimises
average separation: capsules reaches AUROC ~0.93 yet catches 0.35 of defects at 1% (F-L98). A one-way partial-AUC
surrogate (the CVaR form of Zhu et al., ICML 2022, "When AUC meets DRO") asks each defect to beat only the HARDEST
normals, which is exactly the quantity the pre-set threshold uses.

    L = BCE(logits, labels) + weight * mean_{p in defects} mean_{n in top-k normals} max(0, margin - (s_p - s_n))^2
    k = max(1, ceil(beta * n_normals_in_batch))

- Applied in TRAINING mode only. In eval mode (validation / test inference) the module returns plain BCE, so the
  validation loss used for the epoch tie-break keeps its meaning and selection is not silently changed.
- A batch without a defect (or without a normal) contributes plain BCE.
- weight = 0 is exactly BCE; the module is only built when the weight is > 0 (see run_train), so default runs are
  untouched bit-for-bit.
"""
import math

import torch
from torch import nn


class OperatingPointLoss(nn.Module):
    def __init__(self, bce: nn.Module, weight: float = 1.0, beta: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.bce = bce
        self.weight, self.beta, self.margin = float(weight), float(beta), float(margin)

    def tail_term(self, logits, labels):
        pos, neg = logits[labels > 0.5], logits[labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.new_zeros(())
        k = max(1, math.ceil(self.beta * neg.numel()))
        hard = torch.topk(neg, k).values                          # the k highest-scoring normals in the batch
        gap = self.margin - (pos[:, None] - hard[None, :])        # [P, k]
        return torch.clamp(gap, min=0).pow(2).mean()

    def forward(self, logits, labels):
        loss = self.bce(logits, labels)
        if self.training and self.weight > 0:
            loss = loss + self.weight * self.tail_term(logits, labels)
        return loss


class SessionAlignLoss(nn.Module):
    """Session-alignment term (pilot F-L105; side-track diagnosis A-051/S-08): make NORMAL images from the test
    acquisition session look like normal images from the training session in the pooled feature space.

    Why. Under the 1:1 de-confound swap the training set holds normals from BOTH sessions, but every training defect
    comes from the test session, so "test session" still predicts "defect" and the model learns it (features separate
    the sessions in 8/15 MVTec categories; carpet's test-session normals are its false alarms). This term removes the
    cue from the representation instead of calibrating around it.

        L = base(logits, labels) + weight * mean_{i in test-session normals of the batch} || z_i - mu_train ||^2
        z = L2-normalised pooled features;  mu_train = running mean (momentum m) of z over TRAINING-session normals,
        detached (a fixed target, updated after each batch).

    - Session is read from the image PATH (MVTec: '/test/' in the path of a training normal = moved in by the swap).
      That is a property of the split, known before training — nothing from the test set is used.
    - Only NORMALS are aligned; defects are untouched. A running target because carpet has ~9 test-session normals in
      221, about one per batch — a per-batch mean of the training session would be mostly noise.
    - Training mode only; eval mode returns the base loss (validation loss keeps its meaning). weight 0 = base exactly.
    - On VisA (no sessions) no training path contains '/test/', so the term is always zero.
    """
    needs_features = True

    def __init__(self, base: nn.Module, weight: float = 1.0, momentum: float = 0.9, session_token: str = "/test/"):
        super().__init__()
        self.base = base
        self.weight, self.momentum, self.token = float(weight), float(momentum), session_token
        self.register_buffer("mu", torch.zeros(0))
        self.last_term, self.n_aligned = 0.0, 0

    def align_term(self, feats, labels, paths):
        z = torch.nn.functional.normalize(feats, dim=1)
        normal = labels <= 0.5
        test_s = torch.tensor([self.token in str(p) for p in paths], device=feats.device) & normal
        train_s = normal & ~test_s
        term = feats.new_zeros(())
        if self.mu.numel() and test_s.any():
            term = (z[test_s] - self.mu).pow(2).sum(1).mean()
        if train_s.any():                                          # update the target AFTER using it (no self-pull)
            m = z[train_s].detach().mean(0)
            self.mu = m if not self.mu.numel() else self.momentum * self.mu + (1 - self.momentum) * m
        return term, int(test_s.sum())

    def forward(self, logits, labels, feats=None, paths=None):
        loss = self.base(logits, labels)
        if self.training and self.weight > 0 and feats is not None and paths is not None:
            term, n = self.align_term(feats, labels, paths)
            self.last_term, self.n_aligned = float(term.detach()), self.n_aligned + n
            loss = loss + self.weight * term
        return loss


class ViewConsistencyLoss(nn.Module):
    """View-consistency ("distil two-view into one view", 2026-09-26): the two-view score gains because each single view
    is noisy about WHICH images it gets right (every single view averages ~0.94 like-for-like, their mean 0.953). This
    term asks the model to give the same logit for an image and for one of its test-time views (hflip, rot+10, rot-10;
    one drawn per batch), so a single view behaves like the average and inference can stay at one pass.

        L = base(logits, labels) + weight * mean_i (logit_i - logit_view_i)^2

    Weight fixed from a magnitude check (the squared view gap is ~2.5-87x the selected-epoch BCE, median ~30x; the lesson
    of F-L114): 0.01 keeps the term ~0.3x the classification loss. Training mode only; eval = base exactly.
    """
    needs_second_view = True
    VIEWS = ("hflip", "rot+10", "rot-10")

    def __init__(self, base: nn.Module, weight: float = 0.01):
        super().__init__()
        self.base, self.weight = base, float(weight)

    def forward(self, logits, labels, logits_view=None):
        loss = self.base(logits, labels)
        if self.training and self.weight > 0 and logits_view is not None:
            loss = loss + self.weight * (logits - logits_view).pow(2).mean()
        return loss
