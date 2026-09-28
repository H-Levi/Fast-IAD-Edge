#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defect transplant (F-L124) — synthetic training defects made from REAL training defects.

A synthetic record is a training NORMAL image. At load time a random TRAINING defect of the same category is pasted
onto it at the SAME position through the defect's ground-truth mask (MVTec `ground_truth/<type>/<stem>_mask.png`),
feathered by a Gaussian blur of the mask. The result is labelled 1.

Why this and not the old CutPaste (data/augmentation.py): CutPaste cuts patches out of NORMAL images and had known bugs
(black corners, mask misalignment; dossier C11). The transplant only moves real defect pixels, so the synthetic defect
is a real defect type; and because the host photo is a training-session normal, "photo session" no longer predicts
"defect" for these examples (the MVTec folder shortcut, F-L42/F-L48).

Leak discipline: donors are the category's TRAINING defects only; hosts are TRAINING normals only. Nothing from
validation, calibration or test is read. Masks of training defects are extra supervision (K-044) — a stated setup
difference from image-label-only methods.
"""

import os
import random

import numpy as np
from PIL import Image, ImageFilter


def mvtec_mask_path(img_path: str) -> str:
    """<...>/<cat>/test/<type>/<stem>.png -> <...>/<cat>/ground_truth/<type>/<stem>_mask.png"""
    type_dir, fname = os.path.split(img_path)
    split_dir = os.path.dirname(type_dir)                      # <...>/<cat>/test
    cat_dir = os.path.dirname(split_dir)
    return os.path.join(cat_dir, "ground_truth", os.path.basename(type_dir), os.path.splitext(fname)[0] + "_mask.png")


class DefectTransplant:
    """Callable image -> image. Chooses a donor with Python's `random` (seeded per loader worker by seed_worker)."""

    def __init__(self, donor_paths, feather_radius: float = 2.0, mask_path_fn=mvtec_mask_path):
        self.donors = []
        missing = []
        for p in donor_paths:
            m = mask_path_fn(p)
            (self.donors if os.path.exists(m) else missing).append((p, m))
        if missing:
            # Fail LOUDLY at construction (before any training): a transplant without masks would paste whole images.
            raise FileNotFoundError(f"transplant needs a ground-truth mask for every training defect; missing "
                                    f"{len(missing)}, e.g. {missing[0][1]}")
        if not self.donors:
            raise ValueError("transplant needs at least one training defect as a donor")
        self.feather = float(feather_radius)

    def __call__(self, image: Image.Image, donor_idx=None) -> Image.Image:
        # donor_idx None = v1 (F-L124): a random donor per load; an index = v2 (F-L127): the planned donor for this host
        path, mpath = random.choice(self.donors) if donor_idx is None else self.donors[int(donor_idx)]
        host = image.convert("RGB")
        donor = Image.open(path).convert("RGB").resize(host.size, Image.BILINEAR)
        mask = Image.open(mpath).convert("L").resize(host.size, Image.NEAREST)
        if self.feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(self.feather))
        return Image.composite(donor, host, mask)             # donor where mask is 255, host where 0


# ---------------------------------------------------------------------------------------------------------------------
# v2 placement (F-L127): choose, for each synthetic slot, the training normal whose pixels in a RING around the donor's
# mask best match the donor's own ring (lowest mean absolute grayscale difference). Threshold-free, same rule for every
# category: for textures most hosts match similarly; for posed objects (screw) the best host has the object at that place.

def _gray(path, size):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def _ring(mask_img, size, ring_px):
    m = mask_img.convert("L").resize((size, size), Image.NEAREST)
    inner = np.asarray(m) > 127
    outer = np.asarray(m.filter(ImageFilter.MaxFilter(2 * int(ring_px) + 1))) > 127
    return outer & ~inner


def plan_best_match(donor_paths, host_paths, n_syn, size=288, ring_px=8, mask_path_fn=mvtec_mask_path):
    """Returns [(host_path, donor_idx)] of length n_syn: donors round-robin (balanced); for each, the best-matching host
    not yet used with that donor (hosts are reused only after all have been used). Deterministic (stable sort)."""
    if n_syn <= 0:
        return []
    hosts = np.stack([_gray(h, size) for h in host_paths])                          # [H, s, s]
    cost = []
    for p in donor_paths:
        ring = _ring(Image.open(mask_path_fn(p)), size, ring_px)
        if not ring.any():                                                          # mask fills the image: no context
            cost.append(np.zeros(len(host_paths), dtype=np.float32)); continue
        d = _gray(p, size)[ring]
        cost.append(np.abs(hosts[:, ring] - d[None, :]).mean(1))
    cost = np.stack(cost)                                                           # [D, H]
    used = [set() for _ in donor_paths]; pairs = []
    for j in range(int(n_syn)):
        di = j % len(donor_paths)
        order = np.argsort(cost[di], kind="stable")
        avail = [int(h) for h in order if int(h) not in used[di]]
        if not avail:
            used[di].clear(); avail = [int(h) for h in order]
        h = avail[0]; used[di].add(h); pairs.append((host_paths[h], di))
    return pairs


# ---------------------------------------------------------------------------------------------------------------------
# v3 donor rule (F-L129): paste only defects whose mask stays INSIDE the object. Shape defects on the outline (e.g. screw
# manipulated_front / thread_side) cannot be faked by pasting (F-L128, v87); they stay real training defects, unpasted.
# Foreground as BGAD (A-096): grayscale binary threshold (Otsu); constants fixed a priori in F-L129.

def _otsu(a):
    hist, edges = np.histogram(a, bins=256, range=(0, 256)); p = hist / max(hist.sum(), 1)
    w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w - mu) ** 2 / (w * (1 - w))
    return float(np.nanargmax(sb))


def object_foreground(path, size=288, blur=2.0, border_frac=0.2):
    """Boolean [size, size]: Otsu class touching the image border less, holes filled; all True for a texture (the
    object class still covers > border_frac of the border)."""
    from scipy.ndimage import binary_fill_holes
    a = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(blur)),
                   dtype=np.float32)
    hi = a > _otsu(a)
    border = np.zeros_like(hi); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    fg = hi if (hi & border).sum() <= 0.5 * border.sum() else ~hi
    if (fg & border).sum() / border.sum() > border_frac:
        return np.ones_like(fg, dtype=bool)
    return binary_fill_holes(fg)


def donor_inside_object(path, mask_path_fn=mvtec_mask_path, size=288, band_px=4, max_frac=0.10):
    """True if <= max_frac of the defect mask lies in the object's outline band (foreground dilated minus eroded by
    band_px). Textures (whole image = foreground) always pass."""
    from scipy.ndimage import binary_dilation, binary_erosion
    fg = object_foreground(path, size)
    if fg.all():
        return True
    band = binary_dilation(fg, iterations=band_px) & ~binary_erosion(fg, iterations=band_px)
    m = np.asarray(Image.open(mask_path_fn(path)).convert("L").resize((size, size), Image.NEAREST)) > 127
    if not m.any():
        return True
    return bool((m & band).sum() / m.sum() <= max_frac)
