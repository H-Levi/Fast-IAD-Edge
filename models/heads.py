#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classifier heads as swappable units.

The head is the cheapest thing to vary and the easiest place to over-build.
By isolating it from the backbone we can answer a question your earlier code
could not: is the AUROC coming from the backbone features or from a heavy head?

All heads map a feature vector [B, in_features] -> logits [B, 1] (single logit,
BCEWithLogits downstream). Pick via config: model.head = deep | mlp | linear.

  - linear : in_features -> 1            (smallest; the honest baseline)
  - mlp    : in_features -> 256 -> 1     (one hidden block)
  - deep   : in_features -> 512 -> 256 -> 1  (the original 3-block head)

Note on BatchNorm1d: it fails on a batch of size 1 in train mode. The training
engine sets drop_last=True to avoid that; heads keep BN because it helped the
original. If you ever train with tiny batches, switch BN->LayerNorm here in one
place.
"""

import torch.nn as nn


def build_head(name: str, in_features: int, dropout: float = 0.3) -> nn.Module:
    name = (name or "deep").lower()
    if name == "linear":
        return LinearHead(in_features, dropout)
    if name == "mlp":
        return MLPHead(in_features, dropout)
    if name == "deep":
        return DeepHead(in_features, dropout)
    raise ValueError(f"Unknown head '{name}'. Use: linear | mlp | deep")


class LinearHead(nn.Module):
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepHead(nn.Module):
    """The original 3-block head, kept as the heavy reference point."""
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)
