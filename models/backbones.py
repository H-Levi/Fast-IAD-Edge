#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backbones behind one uniform interface.

Each backbone is reduced to a pure FEATURE EXTRACTOR that maps an image batch
[B, 3, H, W] -> a pooled feature vector [B, feat_dim]. The classification head
is attached separately (models/heads.py) so backbone and head vary independently.

Candidates (per your choice: EfficientNet + the two lightest):
  - efficientnet : EfficientNet-B0   channels 1280   (your strong baseline)
  - mobilenet    : MobileNetV3-Small channels 576    (lighter, faster)
  - squeezenet   : SqueezeNet 1.1    channels 512    (lightest)

We use torchvision's modern `weights=` API (the old pretrained=True is
deprecated). Each backbone exposes:
  .channels        int   raw conv-stack channel count (pooling-independent)
  .features_dim    int   OUTPUT dim after pooling (what the head consumes)
  .forward(x) -> [B, features_dim]

Keeping the pooled-vector contract identical across backbones is what lets the
factory and the instrumentation treat them uniformly.

--------------------------------------------------------------------------
POOLING (EXPERIMENTS.md E1)  — `model.pooling`, default `avg`
--------------------------------------------------------------------------
The conv stack emits a spatial grid; at 224 px every backbone here ends at
stride 32, i.e. a 7x7 = 49-cell grid. Global AVERAGE pooling gives every cell an
equal vote, so a defect occupying one cell contributes ~1/49 of the feature
vector. That is the arithmetic opposite of the premise the field is built on --
PatchCore (UM-074): an image is anomalous "as soon as a single patch is
anomalous". Hence the alternatives:

  avg        [B,C]   the original. Unchanged default; the baseline.
  max        [B,C]   strongest cell per channel. Single-patch sufficiency.
  avgmax     [B,2C]  concat of both: keeps context, adds peak sensitivity.
  attention  [B,C]   learned weighted sum over cells (UPDATE_INSTRUCTIONS 3.1).

`attention` is ZERO-INITIALISED on purpose: a zero score map softmaxes to a
uniform distribution, which is exactly average pooling. So it *starts* as the
baseline and can only learn away from it -- the comparison is clean, and the
experiment cannot lose to a bad initialisation.

Cost note (§1.8): avg/max add nothing. avgmax doubles only the head's input
width. attention adds one 1x1 conv (C+1 params). Measure, don't assume --
`analysis/benchmark.py --pooling <kind>`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

class AvgPool(nn.Module):
    """Global average over the spatial grid. The original behaviour."""
    def forward(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)


class MaxPool(nn.Module):
    """Strongest cell per channel — one anomalous patch is enough."""
    def forward(self, x):
        return torch.flatten(F.adaptive_max_pool2d(x, 1), 1)


class AvgMaxPool(nn.Module):
    """Concat of avg and max: context and peak. Output is 2C wide."""
    def forward(self, x):
        a = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        m = torch.flatten(F.adaptive_max_pool2d(x, 1), 1)
        return torch.cat([a, m], dim=1)


class AttentionPool(nn.Module):
    """Learned weighted sum over grid positions.

    A 1x1 conv scores every cell; softmax over positions turns the scores into
    weights that sum to 1; the output is the weighted sum of cell features.
    Zero-init => uniform weights => identical to AvgPool at step 0.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        w_ = self.score(x).flatten(2)             # [B, 1, HW]
        w_ = torch.softmax(w_, dim=-1)            # weights over positions
        f = x.flatten(2)                          # [B, C, HW]
        return torch.bmm(f, w_.transpose(1, 2)).squeeze(-1)   # [B, C]


class GeMPool(nn.Module):
    """Generalised-mean pooling: (mean(x^p))^(1/p). ONE learnable parameter.

    p = 1 is exactly the mean; p -> inf approaches the max. So GeM does not
    choose between avg and max, it interpolates and LEARNS where to sit.

    Motivated by the E1 result rather than by hope: max pooling helped wood,
    metal_nut and macaroni2 and hurt screw and capsules. A binary avg/max choice
    is therefore wrong for roughly half the categories, which is the signature of
    a category-dependent optimum rather than a universal one.

    The learned p is the point. It is reportable and interpretable: a category
    whose defects occupy a single cell should want a large p (peak-like), one
    whose defects are spatially extended should sit near 1 (mean-like). That
    turns pooling from a hyperparameter into a measurement of defect locality.

    Not novel -- GeM comes from image retrieval (Radenovic et al., TPAMI 2018).
    The contribution here is applying and measuring it, never inventing it.

    CLAMPING -- MEASURED, not assumed, and the news is bad for these backbones.
    GeM needs non-negative input and clamps to eps. That is exact after ReLU, but
    EfficientNet-B0 ends in SiLU and MobileNetV3-Small in Hardswish, both of which
    go negative. Fraction of final-grid activations below zero, and what clamping
    costs:

        efficientnet-b0  92.0% negative   mean|avg - gem(p=1)| = 0.1625  (avg magnitude 0.1657)
        mobilenetv3-s    71.3% negative   mean|avg - gem(p=1)| = 0.1890  (avg magnitude 0.2538)

    So clamping perturbs the feature by ~98% of its own magnitude on EfficientNet.
    `gem` at p=1 is NOT approximately `avg` here; it is a different feature. GeM is
    kept for completeness and for ReLU-ended backbones, but **prefer LSEPool below
    on this project's backbones** -- it is the same avg<->max interpolation with a
    formula that is valid on signed activations.
    """

    def __init__(self, p_init: float = 1.5, eps: float = 1e-6,
                 p_min: float = 1.0, p_max: float = 10.0):
        super().__init__()
        # Init strictly INSIDE the clamp range. Initialising at p_min parks any
        # downward gradient in the dead zone and p never moves (observed: p stayed
        # at 1.0 for 30 Adam steps at lr=0.1).
        self.p_raw = nn.Parameter(torch.tensor(float(p_init)))
        self.eps, self.p_min, self.p_max = eps, p_min, p_max

    @property
    def p(self):
        return self.p_raw.clamp(self.p_min, self.p_max)

    def forward(self, x):
        p = self.p
        x = x.clamp(min=self.eps)
        return x.pow(p).mean(dim=(-2, -1)).pow(1.0 / p)

    def extra_repr(self):
        return f"p={float(self.p.detach()):.4f} clamped [{self.p_min}, {self.p_max}]"


class LSEPool(nn.Module):
    """Log-sum-exp pooling: (1/p) * log(mean(exp(p*x))). ONE learnable parameter.

    The signed cousin of GeM, and the right choice for this project's backbones.

        p -> 0    ->  mean(x)      (Taylor: (1/p)log(1 + p*mean(x)) -> mean(x))
        p -> inf  ->  max(x)

    Same avg<->max interpolation GeM gives, but it takes SIGNED input, so nothing
    is clamped and nothing is discarded. That matters here because 92% (EfficientNet)
    and 71% (MobileNetV3) of final-grid activations are negative -- see GeMPool's
    docstring for the measurement.

    The learned p is the finding, not the mechanism. A category whose defects occupy
    one cell should prefer a large p (peak-like); one whose defects are spatially
    extended should sit near 0 (mean-like). If p tracks defect extent, pooling stops
    being a hyperparameter and becomes a measurement.

    Implementation notes:
    - p = softplus(q) keeps p > 0 with no hard boundary, so there is no dead zone.
      Default p_init = 1.0, which sits between mean-like and max-like; it does NOT
      start as the avg baseline, and does not need to -- the avg run is the control.
    - torch.logsumexp is numerically stable, so large p does not overflow.
      The -log(N) term converts the sum into a mean.
    """

    def __init__(self, p_init: float = 1.0):
        super().__init__()
        # inverse softplus so that softplus(q_init) == p_init exactly
        q0 = math.log(math.expm1(float(p_init)))
        self.q = nn.Parameter(torch.tensor(q0))

    @property
    def p(self):
        return F.softplus(self.q)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.p.clamp(min=1e-4)                     # guard the 1/p below
        z = x.reshape(b, c, h * w) * p
        return (torch.logsumexp(z, dim=-1) - math.log(h * w)) / p

    def extra_repr(self):
        return f"p={float(self.p.detach()):.4f} (p->0 = mean, p->inf = max)"


POOLING_REGISTRY = {"avg": AvgPool, "max": MaxPool, "avgmax": AvgMaxPool,
                    "attention": AttentionPool, "gem": GeMPool, "lse": LSEPool}


def build_pool(kind: str, channels: int):
    """-> (module, output_dim). Raises on an unknown kind rather than silently
    falling back to avg, so a typo in a sweep cannot masquerade as a baseline."""
    kind = (kind or "avg").lower()
    if kind not in POOLING_REGISTRY:
        raise ValueError(f"Unknown pooling '{kind}'. "
                         f"Available: {sorted(POOLING_REGISTRY)}")
    if kind == "attention":
        module = AttentionPool(channels)
    elif kind == "gem":
        module = GeMPool()
    else:
        module = POOLING_REGISTRY[kind]()
    return module, (channels * 2 if kind == "avgmax" else channels)


# --------------------------------------------------------------------------
# backbones
# --------------------------------------------------------------------------

class _PooledBackbone(nn.Module):
    """Shared contract: a conv stack named `features`, then a pooling module.

    `features` keeps its name because instrumentation/probes.py filters hooks by
    the module path prefix 'backbone.features'.
    """

    channels = None    # set by subclass

    def __init__(self, pooling: str = "avg"):
        super().__init__()
        self.pooling_name = (pooling or "avg").lower()
        self.pool, self.features_dim = build_pool(self.pooling_name, self.channels)

    def forward(self, x):
        return self.pool(self.features(x))


def _weights(enum, pretrained: bool):
    return enum.DEFAULT if pretrained else None


class EfficientNetB0Backbone(_PooledBackbone):
    channels = 1280

    def __init__(self, pretrained=True, pooling="avg"):
        super().__init__(pooling=pooling)
        net = models.efficientnet_b0(
            weights=_weights(models.EfficientNet_B0_Weights, pretrained))
        self.features = net.features          # conv stack -> [B, 1280, h, w]


class MobileNetV3SmallBackbone(_PooledBackbone):
    channels = 576

    def __init__(self, pretrained=True, pooling="avg"):
        super().__init__(pooling=pooling)
        net = models.mobilenet_v3_small(
            weights=_weights(models.MobileNet_V3_Small_Weights, pretrained))
        self.features = net.features          # -> [B, 576, h, w]


class SqueezeNetBackbone(_PooledBackbone):
    channels = 512

    def __init__(self, pretrained=True, pooling="avg"):
        super().__init__(pooling=pooling)
        net = models.squeezenet1_1(
            weights=_weights(models.SqueezeNet1_1_Weights, pretrained))
        self.features = net.features          # -> [B, 512, h, w]


# --------------------------------------------------------------------------
# multi-tap  (E2 / Lever A)
# --------------------------------------------------------------------------
#
# WHY. At 224 px the final grid is 7x7, so one cell is 2.04% of the image while
# screw's defects average 0.335% -- the defect is smaller than one cell before
# pooling runs. Three independent papers on MobileNetV2 find a mid-level tap beats
# the deepest one (UM-010 Tab. 5: block 8 = 98.5 vs block 13 = 94.0; MN-03 Tab. 6;
# SS-4494 f8+f12), and all three find a THIRD, shallower tap buys nothing.
#
# CONTRACT PRESERVED ON PURPOSE. `features` keeps its name (instrumentation/probes.py
# filters hooks on the path prefix 'backbone.features') and the module the head
# consumes is still called `pool` (engine/evaluate.py hooks `backbone.pool` for
# capture_features). A multi-tap backbone that renamed either would silently break
# feature capture -- which is exactly the failure F-L26 now raises on.
#
# TAP SPEC: "8" -> features[8];  "4.1" -> features[4][1]. The inner form is REQUIRED
# for EfficientNet-B0, whose features[i] are whole stages: at 14x14 it has only two
# top-level taps but six inner ones, and two points cannot separate depth from
# resolution. MobileNetV3-Small has five top-level taps at 14x14 and needs no
# inner indexing.


def _resolve_tap(features, spec):
    """'8' -> features[8];  '4.1' -> features[4][1]. Returns the submodule.

    Also accepts the 'features.4.1' spelling, because that is how EVERY piece of
    this project's own analysis names blocks -- RESULTS_run0 section 6's SE-gate
    table, se_gate_blindness.py, tap_occlusion.py and ARGUMENTS A5a all print
    'features.6.2'. Reading a tap depth off that table and pasting it into a taps
    config used to raise `invalid literal for int(): 'features'`, which is a
    crash at model construction -- i.e. in the first seconds of a run, after the
    queue wait. Accept both spellings rather than make the caller translate.
    """
    m = features
    parts = [q for q in str(spec).strip().split(".") if q != ""]
    if parts and parts[0] == "features":      # tolerate the analysis spelling
        parts = parts[1:]
    if not parts:
        raise ValueError(f"empty tap spec {spec!r}; expected e.g. '7' or 'features.4.1'")
    for part in parts:
        if not part.isdigit():
            raise ValueError(
                f"bad tap spec {spec!r}: component {part!r} is not an index. "
                f"Use '7', '4.1', or the analysis spelling 'features.4.1'.")
        m = m[int(part)]
    return m


class TapPool(nn.Module):
    """Pool each tapped map and concatenate. This is the module the head consumes.

    proj_dim is the MECHANISM arm (F-L15a): the head's input width is otherwise set
    by the tap's channel count, so changing the tap changes head capacity too and a
    loss cannot be attributed to depth. A 1x1 projection to a common width removes
    that confound. The COST arm leaves proj_dim None -- there the confound is stated,
    not removed, because the deployable model is the point.
    """

    def __init__(self, pooling, tap_channels, proj_dim=None, tap_norm=False, tap_gate=False, tap_pool=None):
        super().__init__()
        self.proj = None
        if proj_dim:
            self.proj = nn.ModuleList([nn.Conv2d(c, proj_dim, 1) for c in tap_channels])
            tap_channels = [proj_dim] * len(tap_channels)
        pools, dims = [], []
        # tap_pool (F-L82 variant c): a different pooling for every tap EXCEPT the deepest,
        # e.g. max -- keeps a small defect's peak instead of averaging it away. None = same as `pooling`.
        for j, c in enumerate(tap_channels):
            pl, d = build_pool(tap_pool if (tap_pool and j < len(tap_channels) - 1) else pooling, c)
            pools.append(pl); dims.append(d)
        self.pools = nn.ModuleList(pools)
        # PER-TAP NORMALISATION. Measured 2026-08-30 at 288px on pretrained weights:
        #   mobilenet    mid 48ch  mean|a| 0.3996 vs deep 576ch  0.2527  ->  1.58x
        #   efficientnet mid 112ch mean|a| 2.6778 vs deep 1280ch 0.1591  -> 16.84x
        # EfficientNet's mid tap is ~17x louder PER DIMENSION than its deep tap, and
        # it is the backbone carrying the accuracy claim. A linear head can compensate
        # in principle -- it has independent weights per input dim -- but at standard
        # init the 112 loud dims dominate the 1280 quiet ones, and the gradient signal
        # here comes from 6-48 anomalies. The failure mode is specific and bad: the head
        # effectively ignores one tap, mid+deep behaves like a single tap, the contrast
        # comes back indistinguishable from deep, and we record "taps don't help" when
        # the tap was never used.
        # NOT a default. It is a 4th ARM, so the effect is measured rather than assumed
        # in either direction. MN-03's pyramid decoder applies LayerNorm at every level,
        # which is the precedent; SS-4494's learned per-stream gate stayed within
        # 0.99-1.01, suggesting balancing matters more than gating.
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims]) if tap_norm else None
        # ZERO-START GATE on every tap except the deepest (F-L82 variant a). One learnable
        # scalar per extra tap, initialised at 0: at step 0 the head sees exactly the
        # baseline's deep features plus zeros, and the tap only enters as far as the
        # gradient earns it. Built to keep the tap's gains (cable, pcb2) without letting its
        # louder features override the deep branch on the strong categories (F-L77).
        self.gates = nn.Parameter(torch.zeros(len(dims) - 1)) if (tap_gate and len(dims) > 1) else None
        self.out_dim = int(sum(dims))

    # OCCLUSION support. Set `tap_mask` to a list of 0/1 per tap to zero a tap's
    # contribution at inference. This is an OCCLUSION test, NOT an ablation: the head
    # keeps weights trained expecting both inputs, so it measures what the trained head
    # RELIES ON, not what a model retrained on one tap would do. Say "occlusion" in the
    # paper -- a reviewer will make exactly this distinction.
    tap_mask = None

    def forward(self, maps):
        outs = []
        for i, m in enumerate(maps):
            if self.proj is not None:
                m = self.proj[i](m)
            o = self.pools[i](m)
            if self.norms is not None:
                o = self.norms[i](o)
            if self.gates is not None and i < len(self.gates):
                o = o * self.gates[i]
            if self.tap_mask is not None and not self.tap_mask[i]:
                o = torch.zeros_like(o)
            outs.append(o)
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


class MultiTapBackbone(nn.Module):
    """A backbone that reads from one or more intermediate taps.

    truncate=True drops every stage after the deepest tap -- the ONLY configuration
    that can reduce cost, since mid+deep still runs the whole network (F-L15).
    """

    def __init__(self, base_name, taps, pretrained=True, pooling="avg",
                 proj_dim=None, truncate=False, probe_size=224, tap_norm=False, tap_gate=False,
                 tap_pool=None):
        super().__init__()
        base = BACKBONE_REGISTRY[base_name](pretrained=pretrained, pooling=pooling)
        self.features = base.features
        self.taps = [str(t) for t in taps]
        self.pooling_name = (pooling or "avg").lower()

        if truncate:
            if len(self.taps) != 1:
                raise ValueError(f"truncate=True needs exactly one tap, got {self.taps}")
            top = int(self.taps[0].split(".")[0])
            self.features = nn.Sequential(*list(self.features)[:top + 1])

        self._modules_to_tap = [_resolve_tap(self.features, t) for t in self.taps]
        self._buf = {}
        for i, m in enumerate(self._modules_to_tap):
            m.register_forward_hook(self._make_hook(i))

        # Channel counts are PROBED, never hardcoded: a wrong constant would size the
        # head wrongly and only surface as a shape error mid-run.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.features(torch.zeros(1, 3, probe_size, probe_size))
        chans = [self._buf[i].shape[1] for i in range(len(self.taps))]
        self.tap_channels = chans
        self.tap_grids = [int(self._buf[i].shape[-1]) for i in range(len(self.taps))]
        self._buf.clear()
        if was_training:
            self.train()

        self.pool = TapPool(self.pooling_name, chans, proj_dim=proj_dim,
                            tap_norm=tap_norm, tap_gate=tap_gate, tap_pool=tap_pool)
        self.features_dim = self.pool.out_dim

    def _make_hook(self, i):
        def hook(_m, _inp, out):
            self._buf[i] = out
        return hook

    def forward(self, x):
        self._buf.clear()
        self.features(x)
        maps = [self._buf[i] for i in range(len(self.taps))]
        self._buf.clear()
        return self.pool(maps)


BACKBONE_REGISTRY = {
    "efficientnet": EfficientNetB0Backbone,
    "mobilenet": MobileNetV3SmallBackbone,
    "squeezenet": SqueezeNetBackbone,
}


def build_backbone(name: str, pretrained: bool = True, pooling: str = "avg",
                   taps=None, tap_proj_dim=None, tap_truncate=False,
                   image_size: int = 224, tap_norm: bool = False, tap_gate: bool = False,
                   tap_pool=None) -> nn.Module:
    """taps=None reproduces the original single-tap backbone EXACTLY (same modules,
    same parameter count). Only a non-empty `taps` list changes anything."""
    name = name.lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. "
                         f"Available: {list(BACKBONE_REGISTRY)}")
    if not taps:
        return BACKBONE_REGISTRY[name](pretrained=pretrained, pooling=pooling)
    return MultiTapBackbone(name, taps, pretrained=pretrained, pooling=pooling,
                            proj_dim=tap_proj_dim, truncate=tap_truncate,
                            probe_size=image_size, tap_norm=tap_norm, tap_gate=tap_gate, tap_pool=tap_pool)
