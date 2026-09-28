"""Decoded-image cache. OFF by default.

WHY IT EXISTS. Measured on cable @288 (F-L54 follow-up): a training epoch spends
**73% of its wall clock waiting for the dataloader** -- 119s idle against 44s of
actual compute. Four workers cannot decode 1024x1024 PNGs fast enough to keep a T4
fed, and every epoch re-decodes the same files. Caching the decoded, resized image
removes that wait; the measured category time was 173.6s, of which ~119s is
recoverable.

WHY IT IS SAFE, and this is the part that had to be proved rather than assumed.
We cache the image **after resize and before anything else**. torchvision's
`Resize((S,S))` applied to an image that is already SxS is a no-op: verified
bit-identical (max abs diff 0.0) for 1024x1024, 900x700 and 1600x1200 sources, so
re-running the full transform over a cached image yields the same tensor as
running it over the file. Because the cache sits BEFORE filters, augmentation,
ToTensor and Normalize, all of those still run fresh on every access -- random
augmentation stays random, and turning augmentation on later does not invalidate
the cache.

WHAT IT IS NOT. It does not cache tensors, it does not cache across image sizes
(the key includes the size), and it does not persist to disk. It is cleared
between categories so memory does not accumulate across a 27-category run.
"""
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from torchvision import transforms as _T

_RESIZERS: Dict[int, object] = {}


def _tv_resize(img, size: int):
    r = _RESIZERS.get(size)
    if r is None:
        r = _RESIZERS[size] = _T.Resize((size, size))
    return r(img)

_CACHE: Dict[Tuple[str, int], np.ndarray] = {}
_BYTES = 0
_CAP_BYTES = 3 * 1024 ** 3          # 3 GB ceiling; a 288px category needs ~125 MB
_STATS = {"hits": 0, "misses": 0, "skipped_full": 0}


def clear() -> None:
    """Drop everything. Call between categories."""
    global _BYTES
    _CACHE.clear()
    _BYTES = 0
    for k in _STATS:
        _STATS[k] = 0


def stats() -> dict:
    return {**_STATS, "entries": len(_CACHE), "mb": round(_BYTES / 1024 ** 2, 1)}


def load(path: str, size: int) -> Image.Image:
    """Decoded RGB image resized to (size, size), from cache when present.

    Returns a PIL image so the caller's transform pipeline is unchanged.
    """
    global _BYTES
    key = (path, int(size))
    arr = _CACHE.get(key)
    if arr is not None:
        _STATS["hits"] += 1
        return Image.fromarray(arr)
    # MUST use torchvision's Resize, not PIL's. The caller's pipeline resizes with
    # torchvision, whose bilinear path antialiases differently from PIL's; caching a
    # PIL-resized image would silently change every pixel. Verified: caching the
    # torchvision-resized image and re-running the full transform is bit-identical.
    img = _tv_resize(Image.open(path).convert("RGB"), int(size))
    arr = np.asarray(img, dtype=np.uint8)
    if _BYTES + arr.nbytes <= _CAP_BYTES:
        _CACHE[key] = arr
        _BYTES += arr.nbytes
        _STATS["misses"] += 1
    else:
        # Full: serve the image but do not store it, so behaviour degrades to the
        # uncached path rather than exhausting RAM mid-run.
        _STATS["skipped_full"] += 1
    return img
