#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training loop — orchestration only.

This file runs epochs and coordinates the other pieces; it deliberately
contains no metric formulas (those are in analysis/metrics.py) and no logging
policy (that is the injected `recorder`). What it owns:

  - one training epoch (forward/backward/step) + per-epoch loss
  - per-layer gradient-norm capture (so we can see which layers are learning)
  - calling evaluate (raw) on val, computing val metrics, driving the Selector
  - checkpointing the best-on-val model and remembering the val threshold
  - emitting a structured record at every stage via `recorder` (or no-op if None)

It does NOT build datasets/loaders or run the final test eval; the entry script
(run_train.py) does that, so the loop stays reusable and pure.
"""

import os
from typing import Optional

import numpy as np
import torch

from engine.evaluate import run_inference
from engine.selection import Selector
from analysis.metrics import compute_all

from time import perf_counter as _perf

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


class _NullRecorder:
    """No-op recorder so the loop runs before instrumentation is wired in."""
    def log_stage(self, *a, **k): pass
    def log_epoch(self, *a, **k): pass
    def log_grad_norms(self, *a, **k): pass
    def log_logit_hist(self, *a, **k): pass


def grad_norms_by_layer(model) -> dict:
    """L2 grad norm per top-level module + global. Read right after backward().

    Lets us see, per epoch, whether the backbone or only the head is moving, and
    catch dead/exploding layers. Returns {'global': x, 'backbone': y, 'head': z}.
    """
    groups = {}
    global_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = float(torch.sum(g * g))
        global_sq += sq
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0.0) + sq
    out = {k: float(np.sqrt(v)) for k, v in groups.items()}
    out["global"] = float(np.sqrt(global_sq))
    return out


class WeightEMA:
    """Exponential moving average of ALL state (parameters AND BatchNorm running stats), G1.

    Why: cable's test AUROC swings 0.65-0.91 epoch to epoch while validation cannot pick the
    good epochs (F-L68), and macaroni2's validation is blind to its loss (F-L74). Averaging
    the weights over the last few epochs removes that swing instead of trying to select it.

    Held as a detached copy of the state_dict, NOT a deepcopy of the module: a deepcopy of
    MultiTapBackbone would keep forward hooks bound to the ORIGINAL model's buffer. The EMA is
    evaluated by swapping its state into the model and restoring the training state after.

    Memory in epochs, not a raw decay: steps/epoch here is 6-17, so a fixed 0.999 would lag by
    ~100 epochs. decay = 1 - 1/(ema_epochs * steps_per_epoch), warmed up as (1+n)/(10+n).
    BatchNorm running stats are averaged with the same decay; num_batches_tracked is copied.
    """

    def __init__(self, model, ema_epochs, steps_per_epoch):
        self.decay = 1.0 - 1.0 / max(1.0, float(ema_epochs) * max(1, steps_per_epoch))
        self.n = 0
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            e = self.state[k]
            if e.dtype.is_floating_point:
                e.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                e.copy_(v)

    def swap_in(self, model):
        """Load the EMA state; return the training state to restore with `model.load_state_dict`."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state)
        return backup


def train_one_epoch(model, loader, optimizer, loss_fn, device,
                    capture_grad=True, desc="train", ema=None):
    model.train()
    loss_fn.train()          # an OperatingPointLoss adds its tail term only in training mode; BCE is unaffected
    running_loss, n_seen = 0.0, 0
    grad_accum, grad_batches = {}, 0
    # Does data loading actually cost anything, or do the workers keep up?
    # t_wait is the time this loop sits idle waiting for the next batch; t_compute is
    # the time inside the forward/backward. Decides whether caching decoded images is
    # worth building at all, instead of guessing. loss.item() forces a CUDA sync, so
    # t_compute is real time rather than queued time.
    t_wait = t_compute = 0.0
    _t_mark = _perf()

    for batch in tqdm(loader, desc=desc, leave=False):
        t_wait += _perf() - _t_mark
        _t_c = _perf()
        images, labels, paths = _unpack(batch)
        images = images.to(device)
        labels = labels.to(device, torch.float)

        optimizer.zero_grad()
        if getattr(loss_fn, "needs_second_view", False):  # ViewConsistencyLoss: one extra forward on a test-time view
            from engine.evaluate import _view
            v = loss_fn.VIEWS[int(torch.randint(len(loss_fn.VIEWS), (1,)).item())]
            logits = model(images).view(-1)
            loss = loss_fn(logits, labels, logits_view=model(_view(images, v)).view(-1))
        elif getattr(loss_fn, "needs_features", False):     # SessionAlignLoss: same forward, split in two (F-L105)
            if getattr(model, "score_mode", "image") != "image":   # this split IS the image-mode forward (F-L122)
                raise ValueError("session alignment needs score_mode=image (it re-implements the image-mode forward)")
            feats = model.features(images)
            logits = model.head(feats).view(-1)
            loss = loss_fn(logits, labels, feats=feats, paths=paths)
        else:
            logits = model(images).view(-1)
            loss = loss_fn(logits, labels)
        loss.backward()

        if capture_grad:
            gn = grad_norms_by_layer(model)
            for k, v in gn.items():
                grad_accum[k] = grad_accum.get(k, 0.0) + v
            grad_batches += 1

        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = images.size(0)
        running_loss += loss.item() * bs
        n_seen += bs
        t_compute += _perf() - _t_c
        _t_mark = _perf()

    avg_loss = running_loss / max(n_seen, 1)
    avg_grad = {k: v / max(grad_batches, 1) for k, v in grad_accum.items()} if capture_grad else {}
    train_one_epoch.last_timing = {"data_wait_s": round(t_wait, 3),
                                   "compute_s": round(t_compute, 3)}
    return avg_loss, avg_grad


def _pooling_p(model):
    """Current value of a learnable pooling exponent, or None.

    Logged EVERY epoch, not just at the end. The endpoint alone cannot tell a
    parameter that never moved from one that moved and came back -- and those mean
    opposite things about whether pooling sharpness matters here.
    """
    pool = getattr(getattr(model, "backbone", None), "pool", None)
    if pool is None or not hasattr(pool, "p"):
        return None
    try:
        return round(float(pool.p.detach()), 5)
    except Exception:
        return None


def fit(model, train_loader, val_loader, optimizer, loss_fn, device, cfg,
        out_dir, category, recorder: Optional[object] = None, test_loader=None):
    """Train with val-based selection. Returns a dict describing the best epoch.

    Requires a val_loader (use_validation=true). If you run without validation,
    the entry script is responsible for the (leaky, ablation-only) test-as-val
    path; fit itself always treats `val_loader` as the selection signal.
    """
    # fit.timing is a FUNCTION attribute, so it survives between categories. Reset it
    # here or category 2 silently inherits category 1's totals -- the same silent-
    # accumulation shape as F-L24/25/37.
    fit.timing = {"data_wait_s": 0.0, "compute_s": 0.0, "epoch_wall_s": 0.0, "epochs": 0}

    rec = recorder or _NullRecorder()
    inst = cfg.get("instrumentation", {})
    sel_cfg = cfg.get("selection", {})
    metrics_which = cfg.get("eval", {}).get("metrics")
    strategy = sel_cfg.get("threshold_strategy", "youden")

    sel = Selector(
        monitor=sel_cfg.get("monitor", "val_auroc"),
        mode="max",
        patience=cfg["train"].get("patience", 10),
        # val_aupr saturates at 1.0 on small val sets; without a tie-break the
        # checkpoint is the FIRST epoch to tie, not the best. See Selector docstring.
        tiebreak=(sel_cfg.get("tiebreak", "val_loss") != "none"),
    )

    num_epochs = cfg["train"]["num_epochs"]
    ckpt_path = os.path.join(out_dir, f"{category}_best_model.pt")

    stop_reason = "epoch_cap"   # overwritten if early stopping fires
    # G1: opt-in via `train.ema_epochs` (> 0). Not in default.yaml, so baseline-v1's hash is
    # untouched. When on, EVERYTHING evaluated per epoch -- val, selection, firewalled test,
    # the saved checkpoint -- uses the averaged weights; training itself is unchanged.
    _ema_ep = float(cfg["train"].get("ema_epochs", 0) or 0)
    ema = WeightEMA(model, _ema_ep, len(train_loader)) if _ema_ep > 0 else None
    # SWA (analysis P1/P3, step 2 of the repair order): a running mean of the RAW training weights from epoch
    # `train.swa_start` (1-based) until training stops. Validation cannot pick the good epoch (test AUROC swings
    # 0.39-0.93 within one run while no val rule tracks it, v27), so average instead of select. The mean is kept on the
    # state_dict (not a deepcopy of the module: see WeightEMA on tap-model hooks); BatchNorm statistics are recomputed
    # on the training data at the end. Training itself is untouched (read-only snapshot each epoch). Opt-in, CLI only;
    # absent/0 = old behaviour. Mutually exclusive with EMA.
    _swa_start = int(cfg["train"].get("swa_start", 0) or 0)
    if _swa_start > 0 and ema is not None:
        raise ValueError("train.swa_start and train.ema_epochs are mutually exclusive")
    _swa_state, _swa_n = None, 0
    # FIREWALLED per-epoch test AUROC (O-06): separates "selection picked a bad epoch" from
    # "no epoch was good". Opt-in via `eval.firewalled_test_per_epoch=true`; not in default.yaml,
    # so baseline-v1's config hash is untouched. The value is LOGGED ONLY -- it never reaches
    # the Selector, early stopping, or the checkpoint. Iterating a DataLoader draws a seed from
    # torch's global RNG, so the RNG state is saved and restored around it: training must follow
    # exactly the trajectory it would without the firewall (checked: the run must reproduce).
    firewall = bool(cfg.get("eval", {}).get("firewalled_test_per_epoch", False)) and test_loader is not None
    # PER-EPOCH SCORE LOG (analysis P3, 2026-09-23): keep every epoch's VALIDATION logits (and, when the firewall is on,
    # the firewalled TEST logits) so any selection rule -- recall at a false-alarm budget, a tail score, averaging --
    # can be replayed offline instead of retrained. Pure bookkeeping: the arrays are copies of what is already
    # computed each epoch; no extra forward pass, no RNG draw. Opt-in (CLI only); absent = no file, old behaviour.
    _keep_ep = bool(cfg.get("instrumentation", {}).get("save_scores_per_epoch", False))
    _ep_val, _ep_test, _ep_val_meta, _ep_test_meta = [], [], None, None
    for epoch in range(num_epochs):
        _ep_t0 = _perf()
        train_loss, grad = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            capture_grad=inst.get("capture_grad_norms", True),
            desc=f"{category} ep{epoch+1}/{num_epochs}", ema=ema)
        if _swa_start > 0 and epoch + 1 >= _swa_start:
            _swa_n += 1
            with torch.no_grad():
                if _swa_state is None:
                    _swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    for k, v in model.state_dict().items():
                        if _swa_state[k].dtype.is_floating_point:
                            _swa_state[k].add_((v.detach() - _swa_state[k]) / _swa_n)
                        else:
                            _swa_state[k].copy_(v)
        _train_state = ema.swap_in(model) if ema is not None else None
        # Accumulate the data-wait vs compute split across epochs (F-L51 follow-up):
        # decides empirically whether caching decoded images would buy anything.
        _tm = getattr(train_one_epoch, "last_timing", None) or {}
        _acc = getattr(fit, "timing", None) or {}
        fit.timing = {
            "data_wait_s": round(_acc.get("data_wait_s", 0.0) + _tm.get("data_wait_s", 0.0), 3),
            "compute_s": round(_acc.get("compute_s", 0.0) + _tm.get("compute_s", 0.0), 3),
            "epoch_wall_s": round(_acc.get("epoch_wall_s", 0.0) + (_perf() - _ep_t0), 3),
            "epochs": _acc.get("epochs", 0) + 1,
        }

        val_raw = run_inference(model, val_loader, loss_fn, device, desc="val")
        val_metrics = compute_all(val_raw["labels"], val_raw["logits"],
                                  threshold=None, strategy=strategy,
                                  which=metrics_which)

        monitor_key = sel_cfg.get("monitor", "val_auroc").replace("val_", "")
        monitored = val_metrics.get(monitor_key, float("nan"))
        # val_loss is the tie-break signal (lower is better). It was already
        # computed and logged every epoch; this is the first time it is consulted
        # for selection, and only when `monitored` cannot separate two epochs.
        d = sel.update(epoch, monitored, extra={"threshold": val_metrics["threshold"]},
                       tiebreak_value=val_raw["loss"])

        fw_auroc = None
        if firewall:
            _cpu = torch.get_rng_state()
            _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            _t = run_inference(model, test_loader, loss_fn, device, desc="test[firewalled]")
            if _keep_ep:
                _ep_test.append(np.asarray(_t["logits"], dtype=np.float32))
                _ep_test_meta = (_t["labels"], _t["paths"])
            fw_auroc = float(compute_all(_t["labels"], _t["logits"], threshold=None,
                                         strategy=strategy, which=["auroc"])["auroc"])
            torch.set_rng_state(_cpu)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)

        if _keep_ep:
            _ep_val.append(np.asarray(val_raw["logits"], dtype=np.float32))
            _ep_val_meta = (val_raw["labels"], val_raw["paths"])

        # ---- records at every stage ----
        rec.log_epoch(category, epoch, {
            "firewalled_test_auroc": fw_auroc,
            "pooling_p": _pooling_p(model),
            "train_loss": train_loss,
            "val_loss": val_raw["loss"],
            "val_metrics": val_metrics,
            "is_best": d["is_best"],
            "tiebreak_win": d.get("tiebreak_win", False),
            "best_epoch": d["best_epoch"],
            "patience_left": d["patience_left"],
        })
        if grad:
            rec.log_grad_norms(category, epoch, grad)
        if inst.get("capture_logit_hist", True):
            rec.log_logit_hist(category, epoch, val_raw["logits"], val_raw["labels"],
                               bins=inst.get("histogram_bins", 30))

        print(f"  [{category}] epoch {epoch+1}/{num_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_raw['loss']:.4f} "
              f"val_{monitor_key}={monitored:.4f} "
              f"grad_global={grad.get('global', float('nan')):.3e}"
              + (f" p={_pooling_p(model):.4f}" if _pooling_p(model) is not None else "")
              + (f" | test_auroc[firewalled, not used]={fw_auroc:.4f}" if fw_auroc is not None else "")
              + ("  [best]" if d["improved"] else
                 "  [best: tie-break on val_loss]" if d["tiebreak_win"] else ""))

        if d["is_best"]:
            torch.save(model.state_dict(), ckpt_path)   # the EMA weights when EMA is on
        if _train_state is not None:
            model.load_state_dict(_train_state)          # back to the raw weights for training

        # Minimum-epochs guard (F-L83): patience may not fire before `train.min_epochs`.
        # grid is a slow starter (baseline best epoch 21 of 31); under the 1:1 split patience
        # stopped it at 13 with best epoch 3, and it scored 0.48. Opt-in; absent = old behaviour.
        _min_ep = int(cfg["train"].get("min_epochs", 0) or 0)
        if cfg["train"].get("early_stopping", True) and d["should_stop"] and epoch + 1 >= _min_ep:
            print(f"  [{category}] early stop at epoch {epoch+1} "
                  f"(best epoch {d['best_epoch']+1})")
            stop_reason = "patience"
            break

    # Fallback: if no epoch was ever selected (e.g. monitor was NaN every epoch
    # because val had no positive samples), the best checkpoint was never written.
    # Save the last-epoch weights so the caller can still score, but flag it so a
    # failed selection is never silently trusted as a real result.
    if _keep_ep and _ep_val:
        _arr = {"val_logits": np.stack(_ep_val), "val_labels": np.asarray(_ep_val_meta[0]),
                "val_paths": np.array([str(q) for q in _ep_val_meta[1]])}
        if _ep_test:
            _arr.update({"test_logits": np.stack(_ep_test), "test_labels": np.asarray(_ep_test_meta[0]),
                         "test_paths": np.array([str(q) for q in _ep_test_meta[1]])})
        np.savez_compressed(os.path.join(out_dir, f"{category}_scores_per_epoch.npz"), **_arr)
    selection_failed = (sel.best_epoch < 0)
    if selection_failed:
        if ema is not None:
            ema.swap_in(model)
        torch.save(model.state_dict(), ckpt_path)
        print(f"  [{category}] WARNING: no epoch selected (monitor NaN/never improved) "
              f"-> saved last-epoch fallback; result is NOT selection-validated.")

    _best_thr, _best_val = sel.best_extra.get("threshold", 0.0), sel.best_value
    if _swa_n > 0:
        model.load_state_dict(_swa_state)
        _cpu = torch.get_rng_state()
        _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.optim.swa_utils.update_bn(train_loader, model, device=device)   # recompute BN stats for the mean weights
        torch.set_rng_state(_cpu)
        if _cuda is not None:
            torch.cuda.set_rng_state_all(_cuda)
        torch.save(model.state_dict(), ckpt_path)
        _vr = run_inference(model, val_loader, loss_fn, device, desc="val[swa]")
        _vm = compute_all(_vr["labels"], _vr["logits"], threshold=None, strategy=strategy, which=metrics_which)
        _best_thr = _vm["threshold"]
        _best_val = _vm.get(sel_cfg.get("monitor", "val_auroc").replace("val_", ""), float("nan"))
        print(f"  [{category}] SWA: mean of {_swa_n} epoch(s) from epoch {_swa_start}; val {_best_val:.4f} -> checkpoint")

    # WHY training stopped, recorded rather than inferred. Run 0 had to be
    # reverse-engineered from (n_epochs - best_epoch) against the patience value,
    # which cannot tell "patience expired" from "hit the epoch cap" when the two
    # coincide. They need different fixes: patience expiring means raise patience;
    # hitting the cap means raise num_epochs. screw did BOTH in different runs --
    # it ran to the 50-epoch cap under Run 0 and was patience-stopped at 23 under
    # Arm F -- and the distinction was invisible until it was reconstructed by hand.
    return {
        "stop_reason": stop_reason,
        "epochs_ran": epoch + 1,
        "patience_setting": int(cfg["train"].get("patience", 10)),
        "num_epochs_setting": int(num_epochs),
        "best_epoch": sel.best_epoch,
        "best_val_score": _best_val,             # value of selection.monitor (e.g. AUPR); of the SWA model when SWA ran
        "monitor": sel_cfg.get("monitor", "val_auroc"),
        "best_threshold": _best_thr,
        "swa_start": _swa_start, "swa_epochs": _swa_n,
        "checkpoint": ckpt_path,
        "selection_failed": selection_failed,
    }


def _unpack(batch):
    if len(batch) == 3:
        return batch
    images, labels = batch
    return images, labels, [""] * len(labels)
