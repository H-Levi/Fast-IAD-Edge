#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation = pure inference.

This function runs the model over a loader and returns RAW outputs only:
logits, labels, paths, and the average loss. It computes no AUROC, F1, or
threshold. That separation is deliberate: the engine produces raw signal, and
analysis/metrics.py turns raw signal into numbers. It means we can re-score the
same run under different thresholds/metrics later without re-running the model,
and instrumentation can dump per-sample logits keyed by path.
"""

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


@torch.no_grad()
def run_inference(model, loader, loss_fn, device, desc="eval",
                  capture_features=False) -> dict:
    """Forward pass, returning logits/labels/paths/loss (and optionally features).

    capture_features hooks the backbone's pooling module and keeps its output --
    the pooled embedding the head sees, 1280-d for EfficientNet-B0, 576-d for
    MobileNetV3-Small.

    Why bother. The head is a DISCRIMINATIVE boundary learned from ~5 anomalies.
    The 167 normals are the abundant, well-estimated side of the data and the
    classifier barely uses them beyond pushing them down. Keeping the embeddings
    turns every distribution-based method -- Mahalanobis distance to the normal
    cloud, kNN to train normals, fusing a distance score with the logit -- into
    OFFLINE analysis on files, with no GPU and no retraining. Cost is ~2 MB per
    category (400 images x 1280 floats), against a 35-minute retrain to answer
    any one of those questions otherwise.
    """
    model.eval()
    if hasattr(loss_fn, "eval"):
        loss_fn.eval()       # validation/test loss is plain BCE even when training uses the operating-point loss
    all_logits, all_labels, all_paths = [], [], []
    feats, handle = [], None
    if capture_features:
        pool = getattr(getattr(model, "backbone", None), "pool", None)
        if pool is not None:
            handle = pool.register_forward_hook(
                lambda m, i, o: feats.append(o.detach().reshape(o.shape[0], -1)
                                             .float().cpu().numpy()))
    running_loss, n_seen = 0.0, 0

    # The hook MUST come off even if the loop raises (OOM, a bad image, an
    # interrupted epoch). Left attached it appends to a dead list on every later
    # forward pass -- unbounded memory growth that surfaces later as an OOM blamed
    # on something else -- and a second scoring of the same model fires two hooks,
    # returning 2x the feature rows. Hence try/finally, not a trailing remove().
    try:
        for batch in tqdm(loader, desc=desc, leave=False):
            images, labels, paths = _unpack(batch)
            images = images.to(device)
            labels_f = labels.to(device, torch.float)

            logits = model(images).view(-1)
            loss = loss_fn(logits, labels_f)

            bs = images.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_paths.extend(list(paths))
    finally:
        if handle is not None:
            handle.remove()

    out = {
        "logits": np.concatenate(all_logits) if all_logits else np.array([]),
        "labels": np.concatenate(all_labels) if all_labels else np.array([]),
        "paths": all_paths,
        "loss": running_loss / max(n_seen, 1),
    }
    if capture_features:
        # Fail LOUDLY. Previously a model without `backbone.pool` attached no hook,
        # raised nothing, and returned an empty array -- the run completed, the npz
        # was written, and distribution_score.py failed much later or not at all.
        # A multi-tap model that pools differently is exactly that case.
        if handle is None:
            raise RuntimeError(
                "capture_features=True but no hook target found: the model exposes no "
                "`backbone.pool`. Point the hook at the module whose output the head "
                "consumes, or pass capture_features=False deliberately.")
        out["features"] = np.concatenate(feats) if feats else np.array([])
        # Also catches the double-hook case (2x rows) that a leaked handle produces.
        if len(out["features"]) != len(out["labels"]):
            raise RuntimeError(
                f"feature rows {len(out['features'])} != label rows "
                f"{len(out['labels'])} -- a stale hook is probably still attached.")
    return out


def _unpack(batch):
    """Support (img, label, path) and legacy (img, label)."""
    if len(batch) == 3:
        return batch
    images, labels = batch
    return images, labels, [""] * len(labels)


TTA_VIEWS = ("id", "hflip", "rot+10", "rot-10")


def _view(x, name):
    """Views drawn from the TRAINING augmentation family only (hflip, rotation <= 15 deg), so no
    view shows the model something it was never trained on. Applied to the normalised tensor;
    rotation fills with 0 = the dataset mean after normalisation."""
    import torchvision.transforms.functional as TF
    if name == "id":
        return x
    if name == "hflip":
        return torch.flip(x, dims=[-1])
    if name.startswith("rot"):
        return TF.rotate(x, float(name[3:]))
    raise ValueError(f"unknown TTA view {name!r}")


@torch.no_grad()
def run_inference_views(model, loader, device, views=TTA_VIEWS, desc="tta"):
    """G2: one logit PER VIEW per image, kept separate (n, V). Which views to average, and the
    threshold on the averaged score, are then chosen on VALIDATION offline -- nothing about
    the combination is fixed here, and test is scored once."""
    model.eval()
    L, Y, P = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images, labels, paths = _unpack(batch)
        images = images.to(device)
        L.append(torch.stack([model(_view(images, v)).view(-1) for v in views], 1).cpu().numpy())
        Y.append(labels.numpy()); P.extend(list(paths))
    import numpy as _np
    return {"logits_views": _np.concatenate(L), "labels": _np.concatenate(Y), "paths": P,
            "views": list(views)}
