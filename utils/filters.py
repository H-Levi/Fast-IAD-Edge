#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FILTER BANK  —  manipulable preprocessing filters.

You observed filters matter a lot here, so they are a first-class, configurable
stage rather than something baked into transforms. The bank takes a list of
{name, **params} entries from configs/default.yaml and builds a single callable
that runs them in order on a PIL image (after resize, before tensor/normalize).

Why PIL-stage (not tensor-stage): classic image filters (CLAHE, unsharp, edge
operators) are defined on uint8 images and are easiest to reason about there.
Every filter is parameterized, so "manipulate the filter" = change a number in
the config, no code edit.

Each filter takes a PIL.Image (RGB) and returns a PIL.Image (RGB), so they
compose freely. Filters that produce an edge/high-pass response support
keep_original=True to BLEND the response onto the image (weight) instead of
replacing it — that is usually what helps a classifier: original appearance
plus an amplified defect signal.

cv2 is optional. CLAHE needs it; if cv2 is missing, clahe degrades to a plain
histogram-equalization fallback and prints a one-time note.

To add a learnable filter layer later, we add a separate nn.Module filter and
insert it in the model's forward; this file stays the fixed-filter path.
"""

from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

_WARNED = set()


def _warn_once(key: str, msg: str):
    if key not in _WARNED:
        print(f"[filters] {msg}")
        _WARNED.add(key)


# ----------------------------------------------------------- primitives ----
def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(np.float32)


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _blend(original: np.ndarray, response: np.ndarray, weight: float,
           keep_original: bool) -> np.ndarray:
    """If keep_original: out = original + weight*response. Else: out = weight*response."""
    if keep_original:
        return original + weight * response
    return weight * response


# -------------------------------------------------------------- filters ----
def f_identity(img: Image.Image) -> Image.Image:
    return img


def f_gaussian_blur(img: Image.Image, sigma: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def f_median(img: Image.Image, size: int = 3) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(size=int(size)))


def f_sharpen(img: Image.Image, factor: float = 2.0) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(float(factor))


def f_contrast(img: Image.Image, factor: float = 1.3) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(float(factor))


def f_gamma(img: Image.Image, gamma: float = 1.2) -> Image.Image:
    arr = _to_np(img) / 255.0
    arr = np.power(arr, float(gamma)) * 255.0
    return _to_img(arr)


def f_unsharp_mask(img: Image.Image, radius: float = 2.0, percent: int = 150,
                   threshold: int = 3) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=float(radius),
                                              percent=int(percent),
                                              threshold=int(threshold)))


def f_high_pass(img: Image.Image, sigma: float = 3.0, strength: float = 1.0,
                keep_original: bool = True) -> Image.Image:
    """High-pass = original minus low-pass(blur). Amplifies fine detail/defects."""
    orig = _to_np(img)
    low = _to_np(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))))
    hp = orig - low
    out = _blend(orig, hp, float(strength), keep_original)
    return _to_img(out)


def f_sobel(img: Image.Image, weight: float = 1.0, keep_original: bool = True) -> Image.Image:
    """Sobel gradient magnitude (per channel)."""
    orig = _to_np(img)
    gray = orig.mean(axis=2)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gx = _conv2d(gray, kx)
    gy = _conv2d(gray, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.repeat(mag[:, :, None], 3, axis=2)
    out = _blend(orig, mag, float(weight), keep_original)
    return _to_img(out)


def f_laplacian(img: Image.Image, weight: float = 1.0, keep_original: bool = True) -> Image.Image:
    """Laplacian (second-derivative) edge response."""
    orig = _to_np(img)
    gray = orig.mean(axis=2)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = np.abs(_conv2d(gray, k))
    lap = np.repeat(lap[:, :, None], 3, axis=2)
    out = _blend(orig, lap, float(weight), keep_original)
    return _to_img(out)


def f_clahe(img: Image.Image, clip_limit: float = 2.0, tile_grid: int = 8) -> Image.Image:
    """Contrast-Limited Adaptive Histogram Equalization on the luminance channel."""
    if not _HAS_CV2:
        _warn_once("clahe", "cv2 not available -> CLAHE falling back to global equalize.")
        return ImageOps.equalize(img.convert("RGB"))
    arr = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out, mode="RGB")


def _conv2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tiny same-padding 2D convolution (numpy) for the edge filters."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(gray)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + gray.shape[0], j:j + gray.shape[1]]
    return out


# ----------------------------------------------------------- registry -----
FILTER_REGISTRY: Dict[str, Callable] = {
    "identity": f_identity,
    "gaussian_blur": f_gaussian_blur,
    "median": f_median,
    "sharpen": f_sharpen,
    "contrast": f_contrast,
    "gamma": f_gamma,
    "unsharp_mask": f_unsharp_mask,
    "high_pass": f_high_pass,
    "sobel": f_sobel,
    "laplacian": f_laplacian,
    "clahe": f_clahe,
}


class FilterBank:
    """Build a composed filter callable from a config pipeline list.

    cfg_filters = {
        "enabled": bool,
        "pipeline": [ {"name": "clahe", "clip_limit": 2.0, "tile_grid": 8}, ... ]
    }
    """

    def __init__(self, cfg_filters: dict):
        self.enabled = bool(cfg_filters.get("enabled", False))
        self.pipeline = cfg_filters.get("pipeline", []) or []
        self.steps: List[tuple] = []
        if self.enabled:
            for entry in self.pipeline:
                entry = dict(entry)
                name = entry.pop("name")
                if name not in FILTER_REGISTRY:
                    raise ValueError(
                        f"Unknown filter '{name}'. Available: {list(FILTER_REGISTRY)}")
                self.steps.append((name, FILTER_REGISTRY[name], entry))

    def describe(self) -> list:
        """Human/JSON-readable record of exactly what the bank will do."""
        if not self.enabled:
            return [{"enabled": False}]
        return [{"name": n, "params": p} for (n, _, p) in self.steps]

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img
        for _, fn, params in self.steps:
            img = fn(img, **params)
        return img
