#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config-driven augmentation pipelines (§3.7).

Every transform is declared in config with an explicit probability `p` and its
own range/params, e.g.

    normal_pipeline:
      hflip:      {p: 0.5}
      rotate:     {p: 0.5, degrees: 10}
      brightness: {p: 0.3, range: 0.1}
      glint_noise:{p: 0.0}

Why p/range rather than hardcoded ranges: a knob that lives in config can be
measured through the §1.4 comparator (baseline vs variant differing in exactly
one number). A range buried in Python cannot — it becomes a silent assumption.

COMPOSITION IS FRESH PER IMAGE PER EPOCH. Each transform fires independently
with its own probability, so variety comes from the dice, not from assigning
fixed combos to fixed images. torchvision's stochastic transforms redraw on
every __getitem__ call, which is exactly this behaviour: image i in epoch 3 is
not the same render as image i in epoch 4.

KIND TAGGING AND THE DESTRUCTIVE RULE. Each transform declares a kind:

  geometry     hflip, vflip, rotate, translate, scale — moves the defect, keeps it
  photometric  brightness, contrast — mild appearance variation
  noise        glint_noise, sensor_noise, mild_blur — label stays normal
  destructive  heavy_blur, strong_jitter, large_crop, cutout — can ERASE a defect

The anomaly pipeline accepts GEOMETRY ONLY. Blurring, cropping, or erasing a
real defect deletes the very signal the label refers to: the image is still
labelled 'anomaly' while the anomaly is gone, which is label noise dressed up as
augmentation. This is enforced with an exception rather than a comment, because
a rule that only lives in prose is a rule that eventually gets broken.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

GEOMETRY = "geometry"
PHOTOMETRIC = "photometric"
NOISE = "noise"
DESTRUCTIVE = "destructive"


class DestructiveTransformError(RuntimeError):
    """Raised when a defect-destroying transform is placed on the anomaly pipeline."""
    pass


class UnknownTransformError(RuntimeError):
    pass


# ----------------------------------------------------------- callables ----
class _GlintNoise:
    """Specular glint + sensor noise. NOISE kind: label stays normal."""

    def __init__(self, intensity=(30, 70), radius_frac=(0.03, 0.08), noise_std=6.0):
        self.intensity = intensity
        self.radius_frac = radius_frac
        self.noise_std = noise_std

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB")).astype(np.float32)
        h, w, _ = arr.shape
        r = max(3, int(min(h, w) * random.uniform(*self.radius_frac)))
        cy, cx = random.randint(0, max(h - 1, 0)), random.randint(0, max(w - 1, 0))
        yy, xx = np.ogrid[:h, :w]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        arr[m] = np.clip(arr[m] + random.uniform(*self.intensity), 0, 255)
        if self.noise_std > 0:
            arr = arr + np.random.normal(0, self.noise_std, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


class _SensorNoise:
    def __init__(self, std=6.0):
        self.std = std

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB")).astype(np.float32)
        arr = arr + np.random.normal(0, self.std, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


class _MildBlur:
    def __init__(self, sigma=0.8):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.sigma))


def _wrap(op, p: float):
    """Apply `op` with probability p, redrawn every call."""
    if p >= 1.0:
        return op
    return transforms.RandomApply([op], p=float(p))


# ------------------------------------------------------------ registry ----
# name -> (kind, builder(params) -> callable-or-None)
def _b_hflip(prm):
    return transforms.RandomHorizontalFlip(p=float(prm.get("p", 0.5)))


def _b_vflip(prm):
    return transforms.RandomVerticalFlip(p=float(prm.get("p", 0.5)))


def _b_rotate(prm):
    return _wrap(transforms.RandomRotation(degrees=float(prm.get("degrees", 10))),
                 prm.get("p", 0.5))


def _b_translate(prm):
    t = float(prm.get("range", 0.05))
    return _wrap(transforms.RandomAffine(degrees=0, translate=(t, t)), prm.get("p", 0.5))


def _b_scale(prm):
    lo, hi = prm.get("range", [0.95, 1.05])
    return _wrap(transforms.RandomAffine(degrees=0, scale=(float(lo), float(hi))),
                 prm.get("p", 0.5))


def _b_brightness(prm):
    r = float(prm.get("range", 0.15))
    return _wrap(transforms.ColorJitter(brightness=r), prm.get("p", 0.3))


def _b_contrast(prm):
    r = float(prm.get("range", 0.15))
    return _wrap(transforms.ColorJitter(contrast=r), prm.get("p", 0.3))


def _b_glint(prm):
    return _wrap(_GlintNoise(noise_std=float(prm.get("noise_std", 6.0))),
                 prm.get("p", 0.0))


def _b_sensor_noise(prm):
    return _wrap(_SensorNoise(std=float(prm.get("std", 6.0))), prm.get("p", 0.0))


def _b_mild_blur(prm):
    return _wrap(_MildBlur(sigma=float(prm.get("sigma", 0.8))), prm.get("p", 0.0))


# Destructive builders exist so the registry can NAME them and refuse them on the
# anomaly pipeline with a specific message, rather than failing with "unknown".
def _b_heavy_blur(prm):
    return _wrap(_MildBlur(sigma=float(prm.get("sigma", 3.0))), prm.get("p", 0.0))


def _b_strong_jitter(prm):
    r = float(prm.get("range", 0.6))
    return _wrap(transforms.ColorJitter(brightness=r, contrast=r, saturation=r, hue=0.2),
                 prm.get("p", 0.0))


def _b_large_crop(prm):
    s = prm.get("scale", [0.4, 0.8])
    return _wrap(transforms.RandomResizedCrop(size=prm.get("size", 224),
                                              scale=(float(s[0]), float(s[1]))),
                 prm.get("p", 0.0))


def _b_cutout(prm):
    return _wrap(transforms.RandomErasing(p=1.0, scale=(0.05, 0.2)), prm.get("p", 0.0))


TRANSFORM_REGISTRY: Dict[str, Tuple[str, callable]] = {
    "hflip": (GEOMETRY, _b_hflip),
    "vflip": (GEOMETRY, _b_vflip),
    "rotate": (GEOMETRY, _b_rotate),
    "translate": (GEOMETRY, _b_translate),
    "scale": (GEOMETRY, _b_scale),
    "brightness": (PHOTOMETRIC, _b_brightness),
    "contrast": (PHOTOMETRIC, _b_contrast),
    "glint_noise": (NOISE, _b_glint),
    "sensor_noise": (NOISE, _b_sensor_noise),
    "mild_blur": (NOISE, _b_mild_blur),
    # excluded-by-default: defect-destroying
    "heavy_blur": (DESTRUCTIVE, _b_heavy_blur),
    "strong_jitter": (DESTRUCTIVE, _b_strong_jitter),
    "large_crop": (DESTRUCTIVE, _b_large_crop),
    "cutout": (DESTRUCTIVE, _b_cutout),
}

# What each role is permitted to contain.
ALLOWED_KINDS = {
    "anomaly": {GEOMETRY},                          # geometry-only, non-destructive
    "normal": {GEOMETRY, PHOTOMETRIC, NOISE},       # destructive still banned by default
}


def transform_kind(name: str) -> str:
    if name not in TRANSFORM_REGISTRY:
        raise UnknownTransformError(
            f"Unknown transform '{name}'. Available: {sorted(TRANSFORM_REGISTRY)}")
    return TRANSFORM_REGISTRY[name][0]


def build_pipeline(pipeline_cfg: Dict, role: str,
                   allow_destructive: bool = False) -> Tuple[List, List[Dict]]:
    """Build a list of transform callables from a {name: {p:..., ...}} config.

    role: 'normal' or 'anomaly'. Enforces ALLOWED_KINDS.
    Transforms with p == 0 are omitted entirely (off means off, not "applied
    with probability zero"), so the built pipeline reflects what actually runs.

    Returns (callables, description) — the description is recorded per run so the
    exact augmentation state is auditable alongside the metrics.
    """
    ops, desc = [], []
    for name, prm in (pipeline_cfg or {}).items():
        prm = dict(prm or {})
        kind = transform_kind(name)

        p = float(prm.get("p", 0.0))

        # The anomaly pipeline is GEOMETRY-ONLY, absolutely: allow_destructive
        # never expands it. Moving a defect is safe; changing or masking its
        # appearance destroys the signal the 'anomaly' label refers to.
        if role == "anomaly" and kind != GEOMETRY:
            raise DestructiveTransformError(
                f"'{name}' (kind={kind}) is not permitted on the ANOMALY pipeline, "
                f"which is geometry-only and non-destructive. Blurring, cropping, "
                f"erasing or re-colouring a real defect leaves the image labelled "
                f"'anomaly' while the anomaly is gone — that is label noise, not "
                f"augmentation. Allowed on anomaly: {sorted(ALLOWED_KINDS['anomaly'])}.")

        # On the normal pipeline, destructive transforms are excluded BY DEFAULT
        # but can be opted into deliberately.
        if kind == DESTRUCTIVE and not allow_destructive:
            raise DestructiveTransformError(
                f"'{name}' is excluded by default: it can ERASE a defect "
                f"(kind={kind}). It is on the {role}_pipeline. Set "
                f"augmentation.allow_destructive=true only with a deliberate "
                f"verification that the defect survives.")

        allowed = set(ALLOWED_KINDS.get(role, set()))
        if allow_destructive and role == "normal":
            allowed.add(DESTRUCTIVE)
        if kind not in allowed:
            raise DestructiveTransformError(
                f"'{name}' (kind={kind}) is not permitted on the {role} pipeline. "
                f"Allowed kinds for '{role}': {sorted(allowed)}.")

        if p <= 0.0:
            continue  # off means absent
        op = TRANSFORM_REGISTRY[name][1](prm)
        if op is not None:
            ops.append(op)
            desc.append({"name": name, "kind": kind, "params": prm})
    return ops, desc


def apply_category_overrides(pipeline_cfg: Dict, overrides: Dict,
                             category: str) -> Dict:
    """Merge principled per-category overrides (orientation/symmetry only).

    e.g. per_category_overrides: {screw: {hflip: {p: 0.0}}} — a flipped screw
    thread is a part that cannot exist. These are physics, never score-tuned
    (§0.7); the config comment carries that rule and the census/verification
    stages are where a violation would surface.
    """
    out = {k: dict(v or {}) for k, v in (pipeline_cfg or {}).items()}
    for name, prm in ((overrides or {}).get(category, {}) or {}).items():
        out.setdefault(name, {})
        out[name].update(prm or {})
    return out
