#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_test.py — prove the datasets and the plumbing are sound, fast.

Two jobs, in order:

  (A) DATASET CHECK on your REAL data (no dummy assumptions):
      - paths exist; expected layout is present
      - categories discoverable; per-split counts and class balance
      - MVTec: injection actually moves anomalies into train AND removes them
        from test (explicit train/test path-overlap == 0 assertion = no leak)
      - val split is non-empty and not single-class when use_validation=true
      - one image per split actually opens

  (B) PLUMBING CHECK on a tiny slice of that real data:
      - build model -> train a couple of epochs -> select on val ->
        threshold on val -> score test ONCE -> record -> render report
      - asserts every artifact got written and is non-empty
      - times each stage so a hang/lag is visible

It does NOT care about accuracy (uses pretrained=false by default so it needs no
internet and runs in seconds). Green here = foundation is stable; go tune.

Run (point at your real roots):
  python smoke_test.py paths.mvtec_root=/path/to/mvtec_ad \
                       paths.visa_root=/path/to/visa \
                       run.datasets=[mvtec,visa]

Pick which category each dataset uses with:
  smoke.mvtec_category=screw  smoke.visa_category=candle
(omit to auto-pick the first available).
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_config, apply_overrides
from utils.seed import set_seed
from utils.transforms import build_transforms
from data.mvtec import MVTecDataset
from data.visa import ViSADataset
from data.view import SplitView
from data.stats import profile_dataset, warnings_from_manifest
from models.model_factory import AnomalyClassifier
from engine.train import fit
from engine.evaluate import run_inference
from analysis.metrics import compute_all
from analysis.guardrails import overlap_only, leak_self_check, LeakCheckError
from instrumentation.recorder import Recorder
from instrumentation.report import category_report
from run_train import build_dataset
from torch.utils.data import DataLoader

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Checks:
    def __init__(self):
        self.rows = []
    def add(self, status, name, detail=""):
        self.rows.append((status, name, detail))
        mark = {"PASS": "  [PASS]", "FAIL": "  [FAIL]", "WARN": "  [WARN]"}[status]
        print(f"{mark} {name}" + (f" — {detail}" if detail else ""))
    def ok(self):
        return all(s != FAIL for s, _, _ in self.rows)


# ----------------------------------------------------- (A) dataset checks ----
def check_paths(name, root, ck):
    if not root or not os.path.isdir(root):
        ck.add(FAIL, f"{name} root exists", f"not found: {root}")
        return False
    ck.add(PASS, f"{name} root exists", root)
    return True


def check_mvtec(cfg, ck, category):
    root = cfg["paths"]["mvtec_root"]
    if not check_paths("mvtec", root, ck):
        return
    present = [c for c in MVTecDataset.categories
               if os.path.isdir(os.path.join(root, c))]
    if not present:
        ck.add(FAIL, "mvtec categories found", f"none of the 15 under {root}")
        return
    ck.add(PASS, "mvtec categories found", f"{len(present)}/15: {present[:5]}...")
    category = category or present[0]
    if category not in present:
        ck.add(FAIL, "mvtec smoke category present", category); return
    ck.add(PASS, "mvtec smoke category", category)

    # Use the CONFIG injection settings so the smoke test exercises the real
    # path. Previously this hardcoded k=5/type and ignored anomaly_fraction and
    # injection_val_share, so it silently tested a different split than any run.
    frac = cfg["data"].get("anomaly_fraction")
    k = int(cfg["data"].get("num_train_anomalies", 0))
    if not frac and k < 1:
        k = 1  # ensure SOME injection so the plumbing is exercised
    share = cfg["data"].get("injection_val_share", 0.5)
    try:
        ds = MVTecDataset(root, category=category, transform=None,
                          val_split=cfg["data"]["val_split"], seed=cfg["seed"],
                          anomaly_fraction=frac, num_train_anomalies=k,
                          injection_val_share=share,
                          val_share_mode=cfg["data"].get("val_share_mode", "fixed"),
                          val_anomaly_floor=cfg["data"].get("val_anomaly_floor", 10),
                          val_share_min=cfg["data"].get("val_share_min", 0.5),
                          use_validation=True)
    except Exception as e:
        ck.add(FAIL, "mvtec dataset builds", str(e)); return

    man = profile_dataset(ds, "mvtec", category, True)
    _report_splits(ck, "mvtec", man,
                   {"class_weights": cfg["train"].get("use_class_weights", False)})

    # leak check (shared guardrail): no path in both train and test
    n_overlap = overlap_only([p for (p, _) in ds.records("train")],
                             [p for (p, _) in ds.records("test")])
    ck.add(PASS if not n_overlap else FAIL, "mvtec no train/test leak",
           "0 overlap" if not n_overlap else f"{n_overlap} overlapping files!")

    # injection actually happened
    n_train_anom = sum(l for (_, l) in ds.records("train"))
    ck.add(PASS if n_train_anom > 0 else FAIL, "mvtec injection occurred",
           f"{n_train_anom} anomalies in train "
           f"({'fraction=' + str(frac) if frac else 'k=' + str(k) + '/type'}, "
           f"val_share={(getattr(ds, 'val_share_evidence', {}) or {}).get('resolved', share)}"
           f"{' (adaptive)' if (getattr(ds, 'val_share_evidence', {}) or {}).get('mode')=='adaptive' else ''})")

    # val usable
    val_labels = ds.labels("val")
    ck.add(PASS if len(val_labels) and len(set(val_labels)) > 1 else WARN,
           "mvtec val usable",
           f"n={len(val_labels)} classes={sorted(set(val_labels))}")

    _check_one_image_opens(ck, "mvtec", ds)
    return category


def check_visa(cfg, ck, category):
    root = cfg["paths"]["visa_root"]
    if not check_paths("visa", root, ck):
        return
    csv = os.path.join(root, "split_csv", "2cls_highshot.csv")
    if not os.path.exists(csv):
        # search one level down in case root is a parent
        import glob
        hits = glob.glob(os.path.join(root, "**", "split_csv", "2cls_highshot.csv"),
                         recursive=True)
        if hits:
            ck.add(WARN, "visa split_csv location",
                   f"expected {csv}; found at {os.path.dirname(os.path.dirname(hits[0]))} "
                   f"-> set paths.visa_root there")
            return
        ck.add(FAIL, "visa split_csv present",
               f"missing {csv} — this VisA upload lacks spot-diff preprocessing")
        return
    ck.add(PASS, "visa split_csv present", csv)

    category = category or ViSADataset.categories[0]
    try:
        ds = ViSADataset(root, category=category, transform=None,
                         val_split=cfg["data"]["val_split"],
                         use_validation=True, seed=cfg["seed"])
    except Exception as e:
        ck.add(FAIL, "visa dataset builds", str(e)); return
    ck.add(PASS, "visa smoke category", category)

    man = profile_dataset(ds, "visa", category, True)
    _report_splits(ck, "visa", man,
                   {"class_weights": cfg["train"].get("use_class_weights", False)})

    train_lbl = ds.labels("train")
    ck.add(PASS if len(set(train_lbl)) > 1 else FAIL, "visa train is supervised",
           f"classes={sorted(set(train_lbl))} (VisA train should have both)")

    val_lbl = ds.labels("val")
    ck.add(PASS if len(val_lbl) and len(set(val_lbl)) > 1 else WARN,
           "visa val usable", f"n={len(val_lbl)} classes={sorted(set(val_lbl))}")

    _check_one_image_opens(ck, "visa", ds)
    return category


def _report_splits(ck, name, manifest, mitigation=None):
    sp = manifest["splits"]
    for split in ("train", "val", "test"):
        d = sp.get(split)
        if d:
            ck.add(PASS, f"{name} {split} split",
                   f"n={d['count']} normal={d['normal']} anomaly={d['anomaly']} "
                   f"ratio={d['anomaly_ratio']:.3f}")
    for w in warnings_from_manifest(manifest, mitigation or {}):
        ck.add(WARN, f"{name} profiler flag", w)


def _check_one_image_opens(ck, name, ds):
    from PIL import Image
    recs = ds.records("test") or ds.records("train")
    if not recs:
        ck.add(FAIL, f"{name} has images", "no records in any split"); return
    path = recs[0][0]
    try:
        Image.open(path).convert("RGB")
        ck.add(PASS, f"{name} image opens", os.path.basename(path))
    except Exception as e:
        ck.add(FAIL, f"{name} image opens", f"{path}: {e}")


# --------------------------------------------------- (B) plumbing check ----
def plumbing(cfg, ck, dataset_name, category):
    print(f"\n--- plumbing: {dataset_name}/{category} (tiny slice, pretrained={cfg['model']['pretrained']}) ---")
    timings = {}
    t0 = time.time()
    set_seed(cfg["seed"])
    train_tf, _ = build_transforms(cfg, train=True)
    test_tf, _ = build_transforms(cfg, train=False)
    ds = build_dataset(dataset_name, cfg, category)
    timings["build_dataset"] = time.time() - t0

    # tiny slices for speed (plumbing, not accuracy)
    cap = cfg["smoke"]["cap"]
    tr = _cap(ds.records("train"), cap)
    va = _cap(ds.records("val") or ds.records("test"), cap)
    te = _cap(ds.records("test"), cap)

    bs = cfg["train"]["batch_size"]
    mk = lambda recs, tf, sh, dl: DataLoader(SplitView(recs, tf), batch_size=bs,
                                             shuffle=sh, drop_last=dl, num_workers=0)
    train_loader = mk(tr, train_tf, True, True)
    val_loader = mk(va, test_tf, False, False)
    test_loader = mk(te, test_tf, False, False)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = AnomalyClassifier(cfg["run"]["backbones"][0], cfg["model"]["head"],
                              pretrained=cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    run_dir = os.path.join(cfg["paths"]["output_root"], "smoke", f"{dataset_name}_{category}")
    rec = Recorder(run_dir, cfg)
    key = f"{dataset_name}_{category}"
    rec.log_stage(key, "dataset", profile_dataset(ds, dataset_name, category, True))

    t1 = time.time()
    best = fit(model, train_loader, val_loader, opt, loss_fn, device, cfg,
               run_dir, key, recorder=rec)
    timings["fit"] = time.time() - t1

    # Register run-emitted conditions with the tally. A harness that prints a
    # warning but reports "0 warnings" is under-reporting its own guardrails.
    if best.get("selection_failed"):
        ck.add(WARN, f"{dataset_name} selection failed",
               "no epoch selected (monitor NaN/never improved) -> last-epoch fallback, "
               "NOT selection-validated")
    val_labels_used = [r[1] for r in va]
    if len(set(val_labels_used)) < 2:
        ck.add(WARN, f"{dataset_name} val slice single-class",
               f"capped val slice has classes {sorted(set(val_labels_used))} -> "
               f"AUPR undefined")

    t2 = time.time()
    model.load_state_dict(torch.load(best["checkpoint"], map_location=device))
    raw = run_inference(model, test_loader, loss_fn, device, desc="test")
    tm = compute_all(raw["labels"], raw["logits"], threshold=best["best_threshold"],
                     which=cfg["eval"]["metrics"])
    rec.save_test_scores(key, raw["logits"], raw["labels"], raw["paths"])
    rec.save_summary(key, {"test_metrics": tm, "best_epoch": best["best_epoch"],
                           "best_val_score": best["best_val_score"],
                           "monitor": best.get("monitor", "val_auroc"),
                           "model": model.summary()})
    timings["test+record"] = time.time() - t2

    # ---- assert artifacts exist and are non-empty ----
    cdir = os.path.join(run_dir, key)
    for fn in ["epochs.jsonl", "grad_norms.jsonl", "logit_hist.jsonl",
               "scores_test.npz", "summary.json"]:
        p = os.path.join(cdir, fn)
        good = os.path.exists(p) and os.path.getsize(p) > 0
        ck.add(PASS if good else FAIL, f"{dataset_name} artifact {fn}",
               "" if good else "missing/empty")

    # threshold transfer happened (test used val threshold)
    ck.add(PASS, f"{dataset_name} val→test threshold",
           f"thr={best['best_threshold']:.4f} applied to test")

    # report renders non-empty
    rep = category_report(run_dir, key)
    ck.add(PASS if len(rep) > 80 else FAIL, f"{dataset_name} report renders",
           f"{len(rep)} chars")

    print("  stage timings (s): " +
          "  ".join(f"{k}={v:.2f}" for k, v in timings.items()))
    return timings


def _cap(records, n):
    """Cap to n records while PRESERVING BOTH CLASSES.

    A plain records[:n] slice is class-blind, and the MVTec val split is built
    normals-first (val_data = va_norm + val_inj), so the first n were all
    normals -> single-class val -> AUPR undefined (NaN) -> no epoch selectable.
    That looked like a model/data failure but was purely a slicing artifact of
    this harness. Take a proportional share of each class instead.
    """
    if len(records) <= n:
        return records
    pos = [r for r in records if r[1] == 1]
    neg = [r for r in records if r[1] == 0]
    if not pos or not neg:
        return records[:n]
    n_pos = max(1, min(len(pos), round(n * len(pos) / len(records))))
    n_neg = max(1, min(len(neg), n - n_pos))
    out = neg[:n_neg] + pos[:n_pos]
    return out


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    # smoke defaults (fast, no internet); override on CLI if you want
    cfg.setdefault("smoke", {})
    cfg["smoke"].setdefault("cap", 24)
    cfg["smoke"].setdefault("mvtec_category", None)
    cfg["smoke"].setdefault("visa_category", None)
    cfg["model"]["pretrained"] = cfg["model"].get("pretrained", False) and False or False  # force false for smoke
    cfg["data"]["image_size"] = min(cfg["data"]["image_size"], 96)
    cfg["train"]["num_epochs"] = min(cfg["train"]["num_epochs"], 2)
    cfg["train"]["batch_size"] = min(cfg["train"]["batch_size"], 8)
    cfg["data"]["num_workers"] = 0
    if cfg["data"].get("num_train_anomalies", 0) < 1:
        cfg["data"]["num_train_anomalies"] = 5  # MVTec needs anomalies to be trainable

    print("=" * 78)
    print("SMOKE TEST — dataset structure + project plumbing")
    print("=" * 78)
    print(f"datasets: {cfg['run']['datasets']}  backbone(smoke): {cfg['run']['backbones'][0]}  "
          f"image_size: {cfg['data']['image_size']}  epochs: {cfg['train']['num_epochs']}\n")

    ck = Checks()
    chosen = {}

    print("(A) DATASET STRUCTURE")
    if "mvtec" in cfg["run"]["datasets"]:
        chosen["mvtec"] = check_mvtec(cfg, ck, cfg["smoke"]["mvtec_category"])
    if "visa" in cfg["run"]["datasets"]:
        chosen["visa"] = check_visa(cfg, ck, cfg["smoke"]["visa_category"])

    print("\n(B) PROJECT PLUMBING")
    for name, cat in chosen.items():
        if cat is None:
            ck.add(WARN, f"{name} plumbing skipped", "dataset check did not yield a category")
            continue
        try:
            plumbing(cfg, ck, name, cat)
        except Exception:
            ck.add(FAIL, f"{name} plumbing run", "exception (trace below)")
            traceback.print_exc()

    print("\n" + "=" * 78)
    n_fail = sum(1 for s, _, _ in ck.rows if s == FAIL)
    n_warn = sum(1 for s, _, _ in ck.rows if s == WARN)
    if ck.ok():
        print(f"SMOKE TEST PASSED  ({n_warn} warnings) — foundation is stable, go tune.")
    else:
        print(f"SMOKE TEST FAILED  ({n_fail} failures, {n_warn} warnings) — fix before scaling.")
    print("=" * 78)
    sys.exit(0 if ck.ok() else 1)


if __name__ == "__main__":
    main()
