#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AnomalyClassifier  —  the model the rest of the project talks to.

This is the file the earlier code referenced but did not ship. It composes a
backbone (feature extractor) and a head (classifier) into one module with a
single, stable interface so training, evaluation, instrumentation, and the
benchmark never need to know which backbone is inside.

Interface:
  model(x)            -> logits [B]              (squeezed; BCEWithLogits ready)
  model.features(x)   -> feature vector [B, D]   (for probes / feature analysis)
  model.summary()     -> dict (params, trainable, size_mb, feat_dim, backbone, head)

Options:
  freeze_backbone : train only the head (fast, strong baseline for transfer).
                    Off by default; the original fine-tuned everything.
"""

import math
from typing import Dict

import torch
import torch.nn as nn

from models.backbones import build_backbone
from models.heads import build_head


class AnomalyClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet", head: str = "deep",
                 pretrained: bool = True, dropout: float = 0.3,
                 freeze_backbone: bool = False, pooling: str = "avg",
                 taps=None, tap_proj_dim=None, tap_truncate: bool = False,
                 image_size: int = 224, tap_norm: bool = False, tap_gate: bool = False, tap_pool=None,
                 score_mode: str = "image", patch_topk_frac: float = 0.1):
        super().__init__()
        self.backbone_name = backbone
        self.head_name = head
        self.pooling_name = (pooling or "avg").lower()
        # taps=None is the untouched original path (E2 / Lever A is opt-in only).
        self.taps = list(taps) if taps else None
        self.tap_truncate = bool(tap_truncate)
        self.tap_proj_dim = tap_proj_dim
        self.tap_norm = bool(tap_norm)
        self.tap_gate = bool(tap_gate)
        self.tap_pool = tap_pool
        # score_mode (F-L122): 'image' = pool the feature map, score once (the original path, bit-for-bit);
        # 'patch_topk' = apply the SAME head to every cell of the final feature map and average the top
        # ceil(frac x cells) cell logits (DevNet-style top-K MIL, UM-068). Parameters are identical.
        self.score_mode = (score_mode or "image").lower()
        self.patch_topk_frac = float(patch_topk_frac)
        if self.score_mode not in ("image", "patch_topk"):
            raise ValueError(f"Unknown score_mode '{score_mode}'. Use: image | patch_topk")
        if self.score_mode == "patch_topk" and self.taps:
            raise ValueError("score_mode=patch_topk works on the final feature map only; taps are not supported")
        if not 0 < self.patch_topk_frac <= 1:
            raise ValueError("patch_topk_frac must be in (0, 1]")

        self.backbone = build_backbone(backbone, pretrained=pretrained,
                                       pooling=self.pooling_name,
                                       taps=self.taps, tap_proj_dim=tap_proj_dim,
                                       tap_truncate=self.tap_truncate,
                                       image_size=image_size,
                                       tap_norm=self.tap_norm, tap_gate=self.tap_gate,
                                       tap_pool=self.tap_pool)
        # feat_dim is read AFTER pooling: 'avgmax' returns 2C, the rest C.
        # The head is sized from it, so the contract holds for every pooling.
        self.feat_dim = self.backbone.features_dim
        self.head = build_head(head, self.feat_dim, dropout=dropout)

        if freeze_backbone:
            # Freeze the pretrained CONV STACK only. The pooling module is not
            # pretrained — for 'attention' it is the thing being learned, and
            # freezing it would pin it to its zero-init (= uniform = avg pool),
            # silently turning an attention run into a baseline run.
            for p in self.backbone.features.parameters():
                p.requires_grad = False

    # -- feature access for probes / analysis (no head) --
    def features(self, x):
        return self.backbone(x)

    def cell_logits(self, x, fmap=None):
        """Per-cell logits of the final feature map, [B, h*w] (patch_topk scoring and analysis)."""
        if fmap is None:
            fmap = self.backbone.features(x)                              # [B, C, h, w]
        b, c, h, w = fmap.shape
        return self.head(fmap.permute(0, 2, 3, 1).reshape(b * h * w, c)).view(b, h * w)

    def forward(self, x):
        if self.score_mode == "patch_topk":
            fmap = self.backbone.features(x)
            # The pooled embedding is NOT used for the score, but evaluation hooks `backbone.pool` to save features for
            # offline analysis (run_inference capture_features); running it keeps those files identical in meaning to
            # image-mode runs. Average pooling has no parameters and no side effects besides the hook.
            self.backbone.pool(fmap)
            cells = self.cell_logits(x, fmap)                             # [B, h*w]
            k = max(1, math.ceil(self.patch_topk_frac * cells.shape[1]))
            return cells.topk(k, dim=1).values.mean(1)                    # [B]
        feats = self.backbone(x)            # [B, D]
        logits = self.head(feats)           # [B, 1]
        return logits.squeeze(-1)           # [B]

    # -- self-description for the run record + benchmark --
    def summary(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        out = {
            "backbone": self.backbone_name,
            "head": self.head_name,
            "pooling": self.pooling_name,
            "score_mode": self.score_mode,
            "feat_dim": self.feat_dim,
            "total_params": total,
            "trainable_params": trainable,
            "size_mb": total * 4 / 1024 / 1024,   # float32
            # Leaf modules ~= kernel launches per forward pass. This is the axis
            # the benchmark says actually predicts batch-1 latency: MACs rose 20.8x
            # across our resolution sweep for +10% time, while torch.compile -- which
            # changes nothing but how many kernels are dispatched -- removed 35%.
            # Params and MACs were both already recorded; the one quantity that
            # tracks the clock was the one nothing logged.
            "leaf_ops": sum(1 for m in self.modules() if not list(m.children())),
        }
        # GeM's learned exponent IS the finding, so it has to reach the run
        # record rather than living only in the checkpoint. Read it wherever a
        # pooling module exposes `p`.
        pool = getattr(self.backbone, "pool", None)
        if pool is not None and hasattr(pool, "p"):
            out["pooling_p"] = round(float(pool.p.detach()), 4)
        return out


def build_model(cfg: dict, backbone: str) -> AnomalyClassifier:
    """Build from the config block, with `backbone` chosen by the run loop.

    backbone is passed explicitly (not read from cfg) because one run can sweep
    several backbones; everything else (head, dropout, pretrained, freeze) comes
    from cfg['model'].
    """
    m = cfg["model"]
    return AnomalyClassifier(
        backbone=backbone,
        head=m.get("head", "deep"),
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.3),
        freeze_backbone=m.get("freeze_backbone", False),
        taps=m.get("taps"),
        tap_proj_dim=m.get("tap_proj_dim"),
        tap_truncate=m.get("tap_truncate", False),
        tap_norm=m.get("tap_norm", False),
        tap_gate=m.get("tap_gate", False),
        tap_pool=m.get("tap_pool"),
        image_size=int(cfg.get("data", {}).get("image_size", 224))
        if isinstance(cfg, dict) else 224,
        pooling=m.get("pooling", "avg"),
        score_mode=m.get("score_mode", "image"),
        patch_topk_frac=m.get("patch_topk_frac", 0.1),
    )
