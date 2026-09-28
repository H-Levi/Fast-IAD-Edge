#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Augmentation save-set (§4.2) — eight frozen, named scenarios.

Every scenario is ONE whole-dataset augmentation state, applied identically to
MVTec and VisA, TRAIN SPLIT ONLY (val/test never see augmentation — §0). All of
it is config-gated and defaults to `original`, so the baseline model runs
completely unchanged until a scenario is explicitly selected.

  1 original            no augmentation (the baseline all deltas measure against)
  2 cutpaste            synthetic anomalies cut-and-pasted from normals
  3 geometry            flips + small rotation/shift/scale, per-category policy
  4 appearance          small brightness + contrast (no hue — colour is signal)
  5 noise_on_normals    glint + sensor noise + mild blur, NORMALS ONLY
  6 combined            geometry + appearance (the "full sane pipeline")
  7 oversample          anomalies repeated xN with fresh geometry each draw
  8 asymmetric          anomalies: geometry xN | normals: noise_on_normals

Two distinct mechanisms, deliberately separated:

  TRANSFORM-level  (scenarios 2-6, and the per-class halves of 7-8) change how an
                   image is rendered. They are class-conditional: a normal and an
                   anomaly can get different pipelines.
  RECORD-level     (7, 8) change how many times a record appears in an epoch.
                   Only anomalies are repeated — the starving class — normals are
                   never duplicated, which is what keeps the feature cache small
                   (§4.1.5: cache normals once, anomalies xN).

Geometry is per-category by PHYSICS, not by score: a screw thread or a PCB is
chiral and orientation-bearing, so horizontal flips would manufacture parts that
cannot exist; textures are isotropic so both flips are fine. This policy is
declared below and is not tuned.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

SCENARIOS = ["original", "cutpaste", "geometry", "appearance",
             "noise_on_normals", "combined", "oversample", "asymmetric",
             # point 6: a SECOND, stronger normal-augmentation tier. Kept
             # separate from noise_on_normals rather than folded into it,
             # because it is explicitly riskier: pushed far enough, a "hard
             # normal" stops looking normal, and then the image is labelled
             # normal while looking defective — label noise. It is gated at
             # stricter thresholds (see analysis/aug_verify.STRICT_SCENARIOS).
             "harder_normals"]

RISKY_SCENARIOS = {"harder_normals"}

# --- per-category geometry policy (physics, not tuning) -------------------
# Orientation/chirality-bearing objects: a horizontal flip invents an impossible part.
NO_HFLIP = {
    "screw", "transistor", "metal_nut",
    "pcb1", "pcb2", "pcb3", "pcb4",
    "macaroni1", "macaroni2",
}
# Isotropic textures: both flips are physically meaningful.
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def geometry_policy(category: str) -> Dict:
    """What geometry is admissible for this category."""
    return {
        "hflip": 0.0 if category in NO_HFLIP else 0.5,
        "vflip": 0.5 if category in TEXTURES else 0.0,
        "rotation": 5.0 if category in NO_HFLIP else 10.0,
        "translate": 0.05,
        "scale": (0.95, 1.05),
    }


# ------------------------------------------------------------ primitives ----
def geometry_steps(category: str) -> List:
    p = geometry_policy(category)
    steps = []
    if p["hflip"] > 0:
        steps.append(transforms.RandomHorizontalFlip(p=p["hflip"]))
    if p["vflip"] > 0:
        steps.append(transforms.RandomVerticalFlip(p=p["vflip"]))
    steps.append(transforms.RandomAffine(
        degrees=p["rotation"], translate=(p["translate"], p["translate"]),
        scale=p["scale"]))
    return steps


def appearance_steps() -> List:
    # No hue/saturation: on these datasets colour carries defect signal
    # (metal_nut 'color', pill 'color', wood 'color' are literally colour defects).
    return [transforms.ColorJitter(brightness=0.15, contrast=0.15)]


class NoiseOnNormals:
    """Glint + sensor noise + mild blur. Hardens the NORMAL class so the model
    stops treating benign capture artefacts as anomalies. Label stays normal."""

    def __init__(self, glint_p=0.3, noise_std=6.0, blur_p=0.3, blur_sigma=0.8, seed=None):
        self.glint_p = glint_p
        self.noise_std = noise_std
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self._rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB")).astype(np.float32)
        h, w, _ = arr.shape
        if self._rng.random() < self.glint_p:
            r = max(3, int(min(h, w) * self._rng.uniform(0.03, 0.08)))
            cy = self._rng.randint(0, max(h - 1, 0))
            cx = self._rng.randint(0, max(w - 1, 0))
            yy, xx = np.ogrid[:h, :w]
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            arr[m] = np.clip(arr[m] + self._rng.uniform(30, 70), 0, 255)
        if self.noise_std > 0:
            arr = arr + np.random.normal(0, self.noise_std, arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        if self._rng.random() < self.blur_p:
            out = out.filter(ImageFilter.GaussianBlur(self.blur_sigma))
        return out


class HarderNormals:
    """Stronger normal-hardening (point 6). Deliberately riskier than
    NoiseOnNormals: bigger glint, heavier sensor noise, occasional local
    smudge and stronger illumination gradient — the kinds of benign capture
    artefact that a production line really produces, pushed further so the
    model learns to ignore more of them.

    The risk is precise: each of these, pushed too far, starts to LOOK like a
    defect. A smudge is a contamination; a strong gradient is a discolouration.
    The label still says normal. That is why this tier exists separately and is
    verified against normal_drift at a stricter threshold rather than being
    switched on casually.
    """

    def __init__(self, glint_p=0.5, glint_intensity=(50, 110), noise_std=12.0,
                 smudge_p=0.25, gradient_p=0.4, blur_p=0.3, blur_sigma=1.2, seed=None):
        self.glint_p = glint_p
        self.glint_intensity = glint_intensity
        self.noise_std = noise_std
        self.smudge_p = smudge_p
        self.gradient_p = gradient_p
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self._rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB")).astype(np.float32)
        h, w, _ = arr.shape

        if self._rng.random() < self.glint_p:
            r = max(3, int(min(h, w) * self._rng.uniform(0.05, 0.12)))
            cy, cx = self._rng.randint(0, h - 1), self._rng.randint(0, w - 1)
            yy, xx = np.ogrid[:h, :w]
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            falloff = np.clip(1.0 - d2 / float(r * r), 0, 1)[:, :, None]
            arr = arr + falloff * self._rng.uniform(*self.glint_intensity)

        if self._rng.random() < self.gradient_p:
            axis = self._rng.random() < 0.5
            ramp = np.linspace(-1, 1, w if axis else h, dtype=np.float32)
            ramp = np.tile(ramp, (h, 1)) if axis else np.tile(ramp[:, None], (1, w))
            arr = arr + ramp[:, :, None] * self._rng.uniform(10, 30)

        if self._rng.random() < self.smudge_p:
            from PIL import ImageFilter as _IF
            r = max(4, int(min(h, w) * self._rng.uniform(0.06, 0.14)))
            cy, cx = self._rng.randint(0, h - 1), self._rng.randint(0, w - 1)
            yy, xx = np.ogrid[:h, :w]
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            tmp = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
            sm = np.asarray(tmp.filter(_IF.GaussianBlur(2.5))).astype(np.float32)
            arr[m] = sm[m]

        if self.noise_std > 0:
            arr = arr + np.random.normal(0, self.noise_std, arr.shape)

        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        if self._rng.random() < self.blur_p:
            out = out.filter(ImageFilter.GaussianBlur(self.blur_sigma))
        return out


class CutPaste:
    """Synthetic anomaly: cut a patch from the image, jitter it, paste elsewhere.

    Needs no ground-truth mask because the defect region is generated, which is
    why it works on both datasets. Expect it to help LOCAL defects (scratch,
    crack, contamination) and do little for global/structural ones (flip,
    misplaced) — that expectation is part of the scenario, not a surprise.
    """

    def __init__(self, area=(0.02, 0.15), aspect=(0.3, 3.3), rotate=True, seed=None):
        self.area = area
        self.aspect = aspect
        self.rotate = rotate
        self._rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        W, H = img.size
        for _ in range(10):
            a = self._rng.uniform(*self.area) * W * H
            ar = self._rng.uniform(*self.aspect)
            pw, ph = int(round((a * ar) ** 0.5)), int(round((a / ar) ** 0.5))
            if 1 <= pw < W and 1 <= ph < H:
                break
        else:
            return img
        sx, sy = self._rng.randint(0, W - pw), self._rng.randint(0, H - ph)
        patch = img.crop((sx, sy, sx + pw, sy + ph))
        if self.rotate and self._rng.random() < 0.5:
            patch = patch.rotate(self._rng.uniform(-45, 45), expand=True)
        pw2, ph2 = patch.size
        if pw2 >= W or ph2 >= H:
            return img
        dx, dy = self._rng.randint(0, W - pw2), self._rng.randint(0, H - ph2)
        out = img.copy()
        out.paste(patch, (dx, dy))
        return out


# --------------------------------------------------- scenario assembly ----
def scenario_steps(scenario: str, category: str, role: str) -> List:
    """Extra PIL-stage steps for a given scenario and class role.

    role is 'normal' or 'anomaly'. Returning [] means "render as baseline",
    which is what keeps `original` byte-identical to the current pipeline.
    """
    s = (scenario or "original").lower()
    if s == "original":
        return []
    if s == "geometry":
        return geometry_steps(category)
    if s == "appearance":
        return appearance_steps()
    if s == "combined":
        return geometry_steps(category) + appearance_steps()
    if s == "noise_on_normals":
        return [NoiseOnNormals()] if role == "normal" else []
    if s == "harder_normals":
        # anomalies get geometry only (never hardened — that would attack the
        # defect); normals get the stronger, riskier treatment.
        return [HarderNormals()] if role == "normal" else geometry_steps(category)
    if s == "cutpaste":
        # applied to the synthetic records only (tagged), not to every normal
        return []
    if s == "oversample":
        # anomalies are repeated at record level and get fresh geometry each draw
        return geometry_steps(category) if role == "anomaly" else []
    if s == "asymmetric":
        # anomalies grow (geometry xN, non-destructive); normals harden (noise)
        return geometry_steps(category) if role == "anomaly" else [NoiseOnNormals()]
    raise ValueError(f"Unknown augmentation scenario '{s}'. Options: {SCENARIOS}")


def uses_record_expansion(scenario: str) -> bool:
    return (scenario or "original").lower() in ("oversample", "asymmetric", "cutpaste")


def expand_records(records: List[Tuple], scenario: str, multiplier: int,
                   cutpaste_ratio: float = 1.0) -> List[Tuple]:
    """Record-level expansion, TRAIN ONLY.

    oversample / asymmetric : each ANOMALY record repeated `multiplier` times.
                              Normals are never duplicated (keeps the cache small
                              and avoids re-teaching the majority class).
    cutpaste                : add synthetic anomalies derived from normals,
                              tagged so the loader applies the CutPaste op and
                              labels them 1.

    Records are (path, label) or (path, label, tag); the tag rides along so the
    loader knows a record is synthetic.
    """
    s = (scenario or "original").lower()
    if not uses_record_expansion(s):
        return list(records)

    def unpack(r):
        return (r[0], r[1], r[2] if len(r) > 2 else None)

    out = []
    if s in ("oversample", "asymmetric"):
        n = max(1, int(multiplier))
        for r in records:
            p, lbl, tag = unpack(r)
            reps = n if lbl == 1 else 1
            for _ in range(reps):
                out.append((p, lbl, tag))
        return out

    # cutpaste: keep everything, then synthesise extra anomalies from normals
    out = [unpack(r) for r in records]
    normals = [r for r in out if r[1] == 0]
    n_anom = sum(1 for r in out if r[1] == 1)
    n_synth = int(round(max(n_anom, 1) * cutpaste_ratio))
    if normals and n_synth:
        rng = random.Random(0)
        for i in range(n_synth):
            p, _, _ = normals[rng.randrange(len(normals))]
            out.append((p, 1, "cutpaste"))
    return out


class BlobBlend:
    """Synthetic anomaly by SOFT blob blending — CutPaste's cousin (§3.7).

    Both exist because they fail differently. CutPaste pastes a hard-edged
    rectangle, so a model can learn "sharp rectangular seam = anomaly" — an
    artifact of the augmentation rather than the shape of normal. BlobBlend
    feathers an irregular blob into place with an alpha mask, so the
    discontinuity is a smooth local statistics change with no seam to memorize.

    Keeping both (config chooses) means the synthetic-anomaly hypothesis can be
    tested without being confounded by one generator's signature artifact. Same
    caveat as CutPaste: it teaches the shape of NORMAL, not of real defects, so
    it is always validated on real anomalies only.
    """

    def __init__(self, area=(0.02, 0.12), feather=0.35, jitter=0.25,
                 source="self", seed=None):
        self.area = area
        self.feather = feather      # 0 = hard edge, 1 = very soft
        self.jitter = jitter        # colour/intensity shift applied to the blob
        self.source = source
        self._rng = random.Random(seed)

    def _blob_mask(self, W, H):
        """Irregular soft-edged mask: a few overlapping ellipses, then blurred."""
        m = Image.new("L", (W, H), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(m)
        a = self._rng.uniform(*self.area) * W * H
        r = max(4, int((a / 3.14159) ** 0.5))
        cx = self._rng.randint(r, max(W - r, r + 1))
        cy = self._rng.randint(r, max(H - r, r + 1))
        for _ in range(self._rng.randint(2, 4)):
            ox = cx + self._rng.randint(-r // 2, r // 2)
            oy = cy + self._rng.randint(-r // 2, r // 2)
            rx = int(r * self._rng.uniform(0.5, 1.1))
            ry = int(r * self._rng.uniform(0.5, 1.1))
            d.ellipse([ox - rx, oy - ry, ox + rx, oy + ry], fill=255)
        blur = max(1.0, r * self.feather)
        return m.filter(ImageFilter.GaussianBlur(blur))

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        W, H = img.size
        mask = self._blob_mask(W, H)

        # donor content: another region of the same image, shifted and jittered
        dx = self._rng.randint(-W // 3, W // 3)
        dy = self._rng.randint(-H // 3, H // 3)
        donor = img.transform(
            (W, H), Image.AFFINE, (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        if self.jitter > 0:
            f = 1.0 + self._rng.uniform(-self.jitter, self.jitter)
            donor = ImageEnhance.Brightness(donor).enhance(f)
            donor = ImageEnhance.Contrast(donor).enhance(
                1.0 + self._rng.uniform(-self.jitter, self.jitter))

        out = img.copy()
        out.paste(donor, (0, 0), mask)
        return out


SYNTHETIC_REGISTRY = {
    "cutpaste": CutPaste,
    "blob_blend": BlobBlend,
}


def build_synthetic(name: str, seed=None, **params):
    n = (name or "cutpaste").lower()
    if n not in SYNTHETIC_REGISTRY:
        raise ValueError(f"Unknown synthetic generator '{n}'. "
                         f"Available: {sorted(SYNTHETIC_REGISTRY)}")
    return SYNTHETIC_REGISTRY[n](seed=seed, **params)


# ---------------------------------------------------------------------------
# TWO-LAYER MODEL (§3.7): mechanism-agnostic BASE + optional TARGETED add-on.
#
# The base geometric set is ALWAYS on when augmentation is enabled — it fights
# memorization of a handful of anomalies and is safe everywhere. Targeted
# add-ons are summoned only with the mechanism they exist to train, and are
# switched independently so the comparator can attribute a delta to one add-on
# rather than to a bundle:
#
#   edge_artifacts    -> attention (§3.1): teaches that a strong edge is not
#                        automatically a defect
#   fine_local_noise  -> multi-scale (§3.2): puts signal at the fine scale so a
#                        fine branch has something to earn its cost on
#   glint_on_normals  -> filter / noise-vs-defect (§3.4): defines what to ignore
#
# Do NOT map one augmentation to one mechanism and swap both at once: keep base
# constant, vary only the add-on, or the delta is unattributable.
# ---------------------------------------------------------------------------
TARGETED_ADDONS = {
    "edge_artifacts": {
        "trains": "attention (§3.1)",
        "normal_pipeline": {"contrast": {"p": 0.4, "range": 0.25}},
    },
    "fine_local_noise": {
        "trains": "multi-scale (§3.2)",
        "normal_pipeline": {"sensor_noise": {"p": 0.4, "std": 8.0}},
    },
    "glint_on_normals": {
        "trains": "filter / noise-vs-defect (§3.4)",
        "normal_pipeline": {"glint_noise": {"p": 0.4, "noise_std": 6.0}},
    },
}


def compose_two_layer(base_cfg: dict, targeted_cfg: dict) -> dict:
    """Merge the always-on base with whichever targeted add-ons are enabled.

    Returns {'normal_pipeline':..., 'anomaly_pipeline':..., 'active_addons':[...]}.
    Add-ons only ever contribute to the NORMAL pipeline: each of them is a
    'this is not a defect' lesson, and the anomaly pipeline stays geometry-only.
    """
    normal = {k: dict(v or {}) for k, v in (base_cfg.get("normal_pipeline") or {}).items()}
    anomaly = {k: dict(v or {}) for k, v in (base_cfg.get("anomaly_pipeline") or {}).items()}
    active = []
    for name, spec in TARGETED_ADDONS.items():
        if (targeted_cfg or {}).get(name, {}).get("enabled", False):
            for tname, tprm in spec["normal_pipeline"].items():
                normal.setdefault(tname, {})
                normal[tname].update(tprm)
            active.append(name)
    return {"normal_pipeline": normal, "anomaly_pipeline": anomaly,
            "active_addons": active}
