#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SplitView — an immutable, per-split dataset bound to one transform.

Why this exists: the datasets expose train/val/test via a single mutable `.mode`
attribute. If two DataLoaders are built from the same dataset object (which is
exactly what happens when val is drawn from the train object), constructing the
second loader flips the shared mode and the first loader silently starts serving
the wrong split. SplitView removes the shared state: each loader gets its own
view holding a fixed list of (path, label) records and its own transform, so
train can be augmented while val/test are not, and nothing can desync.

A dataset produces records via `.records(mode)`; we wrap them here.

Augmentation support (§4, optional and OFF by default):
  * `anomaly_transform` — a second pipeline used only for label==1 records, so a
    scenario can treat the two classes differently (e.g. asymmetric: geometry on
    anomalies, sensor-noise on normals).
  * records may carry a third element, a tag. Tag 'cutpaste' marks a synthetic
    anomaly generated from a normal image; `synthetic_op` renders it.

Both are None by default, in which case behaviour is byte-identical to the
plain-transform version. IMPORTANT: `.transform` remains the attribute the leak
self-check compares between the val and test loaders — do not rename it.
"""

from PIL import Image
from torch.utils.data import Dataset


class SplitView(Dataset):
    def __init__(self, records, transform=None, anomaly_transform=None,
                 synthetic_op=None, cache_size=None, transplant_op=None):
        # records: list of (path, label) or (path, label, tag)
        self.records = list(records)
        self.transform = transform                  # leak-check compares this
        self.anomaly_transform = anomaly_transform  # optional, train only
        self.synthetic_op = synthetic_op            # optional, e.g. CutPaste
        self.transplant_op = transplant_op          # optional, data/transplant.py (F-L124), tag 'transplant'
        # Decoded-image cache (data/imgcache.py). THIS is where it must live: every
        # training/val/test loader reads images through SplitView, not through the
        # dataset classes. A first version put the cache in MVTecDataset.__getitem__,
        # which no loader calls -- it was never hit, and the cache test "passed"
        # (wood 0.9928) only because it ran the uncached path.
        self.cache_size = int(cache_size) if cache_size else None

    def labels(self):
        return [r[1] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        path, label = rec[0], rec[1]
        tag = rec[2] if len(rec) > 2 else None

        if self.cache_size:
            from data import imgcache
            image = imgcache.load(path, self.cache_size)
        else:
            image = Image.open(path).convert("RGB")

        # synthetic anomaly generated from a normal image (labelled 1)
        if tag == "cutpaste" and self.synthetic_op is not None:
            image = self.synthetic_op(image)
        elif tag == "transplant":
            if self.transplant_op is None:        # a transplant record without its op would train a NORMAL as label 1
                raise RuntimeError("record tagged 'transplant' but the view has no transplant_op")
            image = self.transplant_op(image) if len(rec) < 4 else self.transplant_op(image, donor_idx=rec[3])

        tf = self.transform
        if label == 1 and self.anomaly_transform is not None:
            tf = self.anomaly_transform
        if tf:
            image = tf(image)
        return image, int(label), path
