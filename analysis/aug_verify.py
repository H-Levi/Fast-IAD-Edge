#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantitative augmentation verification (points 5 & 6).

Contact sheets catch what an eye can see. These catch what it cannot: a
transform that quietly halves the defect's contrast, an oversampled copy that is
a near-duplicate of its source, a "harder normal" that has drifted into looking
genuinely defective. Every scenario passes the SAME four checks — geometric,
photometric, normal-aug and synthetic alike — because each of them can fail in
one of these ways.

  1. DEFECT SURVIVAL   (anomaly pipeline)
     Does the defect still exist after augmentation? With an MVTec ground-truth
     mask we measure the defect region directly: its contrast against the rest
     of the image must survive. Without a mask (VisA) we fall back to
     high-frequency energy retention, since defects are predominantly
     high-frequency structure and blur/erase destroys it.

  2. LABEL INTEGRITY   (record level)
     Does every record still carry the label its content justifies? Synthetic
     anomalies must be labelled 1; real anomalies must not have been routed
     through a normal pipeline; expansion must not have changed the class of any
     existing record.

  3. VARIETY           (oversampling)
     Are the N copies of an oversampled anomaly actually different? A multiplier
     that produces near-duplicates buys gradient steps but no information — the
     model sees one image N times. Measured as mean pairwise distance across
     renders, which must exceed a floor.

  4. NORMAL DRIFT      (point 6, the risky tier)
     Do augmented normals still look normal? If a transform pushes a normal
     closer to the real-anomaly distribution than to the real-normal
     distribution, the image is labelled 'normal' while looking defective. That
     is label noise, and it is the specific failure the stronger normal tier can
     cause, so that tier is gated on this check at a stricter threshold.

All measures are simple image statistics (no model, no training) so verification
is cheap and runs before anything is cached.
"""

from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ------------------------------------------------------------- features ----
def _gray(img) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return a.mean(axis=2)


def hf_energy(img) -> float:
    """High-frequency energy via a Laplacian. Defects are mostly high-frequency;
    blur/erase collapses this, geometry preserves it."""
    g = _gray(img)
    lap = (-4 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    return float(np.var(lap))


def local_contrast(img, k: int = 8) -> float:
    g = _gray(img)
    h, w = g.shape
    hh, ww = h // k * k, w // k * k
    if hh == 0 or ww == 0:
        return float(g.std())
    tiles = g[:hh, :ww].reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3)
    return float(tiles.reshape(-1, k * k).std(axis=1).mean())


def image_stats(img) -> np.ndarray:
    """Compact descriptor used for drift/variety comparisons."""
    g = _gray(img)
    hist, _ = np.histogram(g, bins=16, range=(0, 255), density=True)
    return np.concatenate([[g.mean() / 255.0, g.std() / 255.0,
                            np.log1p(hf_energy(img)) / 10.0,
                            local_contrast(img) / 255.0], hist])


# --------------------------------------------------- 1. defect survival ----
def defect_survival(original: Image.Image, augmented: Image.Image,
                    mask: Optional[Image.Image] = None,
                    min_retention: float = 0.6) -> Dict:
    """Did the defect survive the anomaly pipeline?

    With a mask: compare the defect region's contrast against its surroundings,
    before vs after. With no mask: compare high-frequency energy.
    Geometry moves a defect (retention ~1.0); blur/erase removes it (retention -> 0).
    """
    if mask is not None:
        m = np.asarray(mask.convert("L"), dtype=np.float32) > 127
        if m.sum() > 0 and m.size - m.sum() > 0:
            def contrast(im):
                g = _gray(im)
                if g.shape != m.shape:
                    g = np.asarray(
                        Image.fromarray(g.astype(np.uint8)).resize(
                            (m.shape[1], m.shape[0])), dtype=np.float32)
                return abs(float(g[m].mean() - g[~m].mean()))
            c0, c1 = contrast(original), contrast(augmented)
            retention = (c1 / c0) if c0 > 1e-6 else 1.0
            return {"method": "mask_contrast", "before": c0, "after": c1,
                    "retention": float(retention),
                    "passed": bool(retention >= min_retention),
                    "min_retention": min_retention}

    e0, e1 = hf_energy(original), hf_energy(augmented)
    retention = (e1 / e0) if e0 > 1e-6 else 1.0
    return {"method": "hf_energy", "before": e0, "after": e1,
            "retention": float(retention),
            "passed": bool(retention >= min_retention),
            "min_retention": min_retention}


# --------------------------------------------------- 2. label integrity ----
def label_integrity(records_before: List, records_after: List,
                    scenario: str) -> Dict:
    """Every record's label must still be justified by its content."""
    problems = []

    def count(recs, lbl):
        return sum(1 for r in recs if r[1] == lbl)

    n0, a0 = count(records_before, 0), count(records_before, 1)
    n1, a1 = count(records_after, 0), count(records_after, 1)

    # normals must never be duplicated (keeps the cache small; also nothing
    # about a normal needs repeating)
    if n1 > n0:
        problems.append(f"normal count grew {n0} -> {n1}: normals must not be duplicated")

    # synthetic records must be labelled anomaly
    synth = [r for r in records_after if len(r) > 2 and r[2] in ("cutpaste", "blob_blend")]
    mislabelled = [r for r in synth if r[1] != 1]
    if mislabelled:
        problems.append(f"{len(mislabelled)} synthetic record(s) not labelled anomaly")

    # a real record's label must not have flipped
    before_lbl = {r[0]: r[1] for r in records_before}
    for r in records_after:
        if (len(r) <= 2 or r[2] is None) and r[0] in before_lbl and r[1] != before_lbl[r[0]]:
            problems.append(f"label flipped for {r[0]}")
            break

    return {"before": {"normal": n0, "anomaly": a0},
            "after": {"normal": n1, "anomaly": a1},
            "synthetic": len(synth),
            "problems": problems,
            "passed": not problems}


# ----------------------------------------------------------- 3. variety ----
def variety(renders: List[Image.Image], min_mean_distance: float = 0.02) -> Dict:
    """Are the N oversampled copies actually different from each other?

    A multiplier that yields near-duplicates gives gradient steps without
    information: the model sees one image N times.
    """
    if len(renders) < 2:
        return {"n": len(renders), "mean_pairwise_distance": None,
                "passed": None, "reason": "need >=2 renders"}
    feats = np.stack([image_stats(r) for r in renders])
    dists = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            dists.append(float(np.linalg.norm(feats[i] - feats[j])))
    mean_d = float(np.mean(dists))
    return {"n": len(renders), "mean_pairwise_distance": mean_d,
            "min_pairwise_distance": float(np.min(dists)),
            "min_mean_distance": min_mean_distance,
            "passed": bool(mean_d >= min_mean_distance),
            "reason": ("copies are near-duplicates — multiplier adds samples but "
                       "no information" if mean_d < min_mean_distance else "varied")}


# ------------------------------------------- 4. normal drift (point 6) ----
def normal_drift(augmented_normals: List[Image.Image],
                 real_normals: List[Image.Image],
                 real_anomalies: List[Image.Image],
                 max_drift_ratio: float = 0.8) -> Dict:
    """Do augmented normals still look normal?

    Compares each augmented normal's distance to the real-NORMAL centroid
    against its distance to the real-ANOMALY centroid. If augmentation pushes
    normals toward the anomaly distribution, the image is labelled 'normal'
    while looking defective — label noise, the exact failure mode the stronger
    normal tier risks.

    drift_ratio = mean(d_normal) / mean(d_anomaly). Values near 0 are safe
    (still clearly normal); values approaching or above 1 mean the augmented
    normals sit as close to anomalies as to normals.
    """
    if not augmented_normals or not real_normals or not real_anomalies:
        return {"passed": None, "reason": "insufficient samples for drift check"}

    aug = np.stack([image_stats(i) for i in augmented_normals])
    nrm = np.stack([image_stats(i) for i in real_normals])
    ano = np.stack([image_stats(i) for i in real_anomalies])
    c_n, c_a = nrm.mean(axis=0), ano.mean(axis=0)

    d_n = np.linalg.norm(aug - c_n, axis=1).mean()
    d_a = np.linalg.norm(aug - c_a, axis=1).mean()
    ratio = float(d_n / d_a) if d_a > 1e-9 else float("inf")

    # reference: how far REAL normals sit, to keep the ratio interpretable
    base = float(np.linalg.norm(nrm - c_n, axis=1).mean() /
                 max(np.linalg.norm(nrm - c_a, axis=1).mean(), 1e-9))

    return {"drift_ratio": ratio, "baseline_ratio": base,
            "max_drift_ratio": max_drift_ratio,
            "passed": bool(ratio <= max_drift_ratio),
            "reason": ("augmented normals have drifted toward the anomaly "
                       "distribution -> label noise" if ratio > max_drift_ratio
                       else "augmented normals still sit with real normals")}


# ------------------------------------------------------------ aggregate ----
# The risky tier is held to stricter thresholds than the base scenarios.
STRICT_SCENARIOS = {"harder_normals"}

# PROVISIONAL, like the floors/target/cap. Calibrated only against synthetic
# textures so far; real-data calibration is the honest next step. The one
# threshold with a principled anchor is max_drift_ratio: a ratio of 1.0 means an
# augmented "normal" sits exactly as close to the anomaly centroid as to the
# normal one, which is unambiguous label noise. Default sits just under that;
# the risky tier is held further back.
THRESHOLDS = {
    "default": {"min_retention": 0.6, "min_mean_distance": 0.02, "max_drift_ratio": 0.95},
    "strict": {"min_retention": 0.75, "min_mean_distance": 0.03, "max_drift_ratio": 0.85},
}
THRESHOLDS_VALIDATED = False   # -> every verdict inherits this


def thresholds_for(scenario: str) -> Dict:
    t = dict(THRESHOLDS["strict"] if scenario in STRICT_SCENARIOS else THRESHOLDS["default"])
    t["tier"] = "strict" if scenario in STRICT_SCENARIOS else "default"
    t["provisional"] = not THRESHOLDS_VALIDATED
    return t


def summarize(checks: Dict) -> Dict:
    """Collapse the individual check results into a single verdict."""
    ran = {k: v for k, v in checks.items() if isinstance(v, dict) and "passed" in v}
    failed = [k for k, v in ran.items() if v["passed"] is False]
    skipped = [k for k, v in ran.items() if v["passed"] is None]
    return {"passed": not failed,
            "failed_checks": failed,
            "skipped_checks": skipped,
            "n_checks": len(ran),
            "thresholds_provisional": not THRESHOLDS_VALIDATED}


# ---------------------------------------------------------------------------
# PAIRED defect survival — the mask must move WITH the image.
#
# The first version measured contrast inside the ORIGINAL mask region after a
# geometric transform had moved the defect somewhere else, so it was reading
# background and reporting near-zero retention for a transform that had done
# nothing harmful (capsule/geometry = 0.164, toothbrush/original = 0.018).
# Geometry MOVES a defect; that is the whole point of allowing it. To measure
# whether the defect survived, the mask has to undergo the same geometry.
#
# Packing the mask into the alpha channel makes that automatic: PIL geometric
# ops transform all four channels together, so image and mask stay registered
# by construction rather than by us reproducing RNG state.
# ---------------------------------------------------------------------------
GEOMETRIC_TYPES = ("RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation",
                   "RandomAffine", "RandomResizedCrop", "RandomApply")


def is_geometric(step) -> bool:
    n = type(step).__name__
    if n == "RandomApply":
        inner = getattr(step, "transforms", [])
        return bool(inner) and all(type(i).__name__ in GEOMETRIC_TYPES for i in inner)
    return n in GEOMETRIC_TYPES


def paired_augment(img: Image.Image, mask: Image.Image, steps):
    """Apply `steps` to image and mask together (mask rides in the alpha channel).

    Geometric steps move both; non-geometric steps are applied to the RGB only,
    since a colour jitter has no meaning for a binary mask.
    """
    m = mask.convert("L").resize(img.size)
    packed = Image.merge("RGBA", (*img.convert("RGB").split(), m))
    post = []
    for st in steps:
        if is_geometric(st):
            packed = st(packed)
        else:
            post.append(st)
    r, g, b, a = packed.split()
    out = Image.merge("RGB", (r, g, b))
    for st in post:
        out = st(out)
    return out, a


def defect_survival_paired(img: Image.Image, mask: Image.Image, steps,
                           min_retention: float = 0.6) -> Dict:
    """Defect survival with the mask transformed alongside the image."""
    aug_img, aug_mask = paired_augment(img, mask, steps)
    res = defect_survival(img, aug_img, mask=None, min_retention=min_retention)  # HF fallback fields

    m0 = np.asarray(mask.convert("L").resize(img.size)) > 127
    m1 = np.asarray(aug_mask) > 127
    if m0.sum() == 0 or m1.sum() == 0:
        return {"method": "paired_mask", "retention": 0.0, "passed": False,
                "before": None, "after": None, "min_retention": min_retention,
                "defect_pixels_before": int(m0.sum()), "defect_pixels_after": int(m1.sum()),
                "reason": "defect region left the frame entirely"}

    def contrast(im, m):
        g = _gray(im)
        if g.shape != m.shape:
            g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize(
                (m.shape[1], m.shape[0])), dtype=np.float32)
        if m.sum() == 0 or (~m).sum() == 0:
            return 0.0
        return abs(float(g[m].mean() - g[~m].mean()))

    c0, c1 = contrast(img, m0), contrast(aug_img, m1)
    retention = (c1 / c0) if c0 > 1e-6 else 1.0
    kept = float(m1.sum()) / float(m0.sum())
    return {"method": "paired_mask", "before": c0, "after": c1,
            "retention": float(retention),
            "defect_area_kept": kept,
            "defect_pixels_before": int(m0.sum()), "defect_pixels_after": int(m1.sum()),
            "min_retention": min_retention,
            "passed": bool(retention >= min_retention and kept >= 0.5)}
