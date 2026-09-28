#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Report — turn recorded artifacts into readable per-stage text.

The recorder writes machine-structured JSONL. This module reads those back and
renders a compact, human- (and Claude-) readable report per category and per
run. The point: after a run you (or a script) produce one text blob that shows
exactly how the model behaved at each stage, which is what gets pasted back for
surgical analysis.

It reads only files on disk, so it works on any past run, not just the live one.
"""

import os
import json
from typing import List

from analysis.guardrails import detect_collapse
from analysis.reliability import caveat_line


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def category_report(run_dir: str, category: str) -> str:
    cdir = os.path.join(run_dir, category)
    stages = _read_jsonl(os.path.join(cdir, "stage.jsonl"))
    epochs = _read_jsonl(os.path.join(cdir, "epochs.jsonl"))
    grads = _read_jsonl(os.path.join(cdir, "grad_norms.jsonl"))
    hists = _read_jsonl(os.path.join(cdir, "logit_hist.jsonl"))
    summary_path = os.path.join(cdir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    L = []
    L.append("=" * 78)
    L.append(f"CATEGORY: {category}")
    L.append("=" * 78)

    # dataset stage
    for s in stages:
        if s.get("stage") == "dataset":
            sp = s.get("splits", {})
            L.append("DATASET")
            for split in ("train", "val", "test"):
                d = sp.get(split)
                if d:
                    L.append(f"  {split:<5} n={d['count']:<5} "
                             f"normal={d['normal']:<5} anomaly={d['anomaly']:<5} "
                             f"anom_ratio={_fmt(d['anomaly_ratio'],3)}")
            flags = s.get("flags", {})
            on = [k for k, v in flags.items() if v]
            if on:
                L.append(f"  FLAGS: {', '.join(on)}")

    # per-epoch table
    if epochs:
        L.append("")
        L.append("EPOCHS")
        L.append(f"  {'ep':>3} {'train_loss':>11} {'val_loss':>9} "
                 f"{'val_auroc':>10} {'val_aupr':>9} {'val_f1':>8} "
                 f"{'g_global':>10} {'best':>5}"
                 + (f" {'TEST[fw]':>9}" if any(e.get("firewalled_test_auroc") is not None for e in epochs) else ""))
        fw = any(e.get("firewalled_test_auroc") is not None for e in epochs)
        gmap = {g["epoch"]: g for g in grads}
        for e in epochs:
            vm = e.get("val_metrics", {})
            g = gmap.get(e["epoch"], {})
            L.append(f"  {e['epoch']+1:>3} {_fmt(e.get('train_loss')):>11} "
                     f"{_fmt(e.get('val_loss')):>9} {_fmt(vm.get('auroc')):>10} "
                     f"{_fmt(vm.get('aupr')):>9} {_fmt(vm.get('f1')):>8} "
                     f"{_fmt(g.get('global'),3):>10} "
                     f"{'*' if e.get('is_best') else '':>5}"
                     + (f" {_fmt(e.get('firewalled_test_auroc')):>9}" if fw else ""))
        if fw:
            L.append("  TEST[fw] = test AUROC logged each epoch, FIREWALLED: never used for selection or stopping.")

    # logit separation trajectory (the detector's actual job)
    if hists:
        L.append("")
        L.append("LOGIT SEPARATION (val: anomaly_mean - normal_mean ; higher = better)")
        for h in hists:
            nm = h.get("normal", {})
            an = h.get("anomaly", {})
            if nm.get("mean") is not None and an.get("mean") is not None:
                gap = an["mean"] - nm["mean"]
                L.append(f"  ep {h['epoch']+1:>3}: normal_mean={_fmt(nm['mean'],3)} "
                         f"anomaly_mean={_fmt(an['mean'],3)} gap={_fmt(gap,3)}")
        # §1.2 collapse verdict from the gap trajectory
        eps = summary.get("collapse_gap_eps", 0.1) if summary else 0.1
        col = detect_collapse(hists, eps=eps,
                              selected_epoch=(summary.get("best_epoch") if summary else None))
        if col["collapsed"] is None:
            L.append("  COLLAPSE CHECK: unknown (insufficient class-mean data)")
        elif col["collapsed"]:
            L.append(f"  *** COLLAPSED: {col['reason']} — AUROC below is NOT a real result ***")
        else:
            L.append(f"  collapse check: ok ({col['reason']})")

    # final summary
    if summary:
        L.append("")
        collapsed = summary.get("collapsed")
        if collapsed:
            L.append("FINAL (test @ val-chosen threshold)  [COLLAPSED RUN — metrics are NOT real data]")
        elif summary.get("selection_failed"):
            L.append("FINAL (test @ val-chosen threshold)  [SELECTION FAILED — last-epoch fallback, not validated]")
        else:
            L.append("FINAL (test @ val-chosen threshold)")
        tm = summary.get("test_metrics", {})
        for k in ("auroc", "aupr", "f1", "precision", "recall", "ece", "threshold"):
            if k in tm:
                L.append(f"  {k:<10} {_fmt(tm[k])}")
        # LIKE-FOR-LIKE under Arm F: the swap puts train/good images into test, and
        # those are the normals this model finds easiest, so the headline auroc is an
        # UPPER bound. This row scores only the normals whose provenance is test/ --
        # what Run 0's normals were -- and is the number to compare against Run 0.
        if "auroc_testgood_normals" in tm:
            L.append(f"  {'auroc(like-for-like)':<10} {_fmt(tm['auroc_testgood_normals'])}"
                     f"   <- COMPARE THIS to Run 0, not the headline above"
                     f"   [{tm.get('n_testgood_normals','?')} test/good vs "
                     f"{tm.get('n_traingood_normals','?')} swapped-in train/good normals]")
        if "best_epoch" in summary:
            mon = summary.get("monitor", "val_auroc")
            L.append(f"  selected at epoch {summary['best_epoch']+1} "
                     f"({mon}={_fmt(summary.get('best_val_score'))})")
        # WHY it stopped. "patience" and "epoch_cap" need opposite fixes and were
        # previously indistinguishable from the report alone.
        if summary.get("stop_reason"):
            L.append(f"  stopped by {summary['stop_reason']} after "
                     f"{summary.get('epochs_ran','?')} epochs "
                     f"(patience={summary.get('patience_setting','?')}, "
                     f"cap={summary.get('num_epochs_setting','?')})")
        # Arm F swap, if it ran: what moved, and the fact that test size is preserved.
        _dc = summary.get("de_confound") or {}
        if _dc.get("enabled"):
            L.append(f"  ARM F swap k={_dc.get('k')}: test/good -> "
                     f"{_dc.get('n_testgood_to_train')} train + {_dc.get('n_testgood_to_val')} val; "
                     f"{_dc.get('n_traingood_to_test')} train/good -> test (test size unchanged)")
        # Where the time went, and whether caching decoded images would buy anything.
        _tmg = summary.get("timing") or {}
        if _tmg:
            _f = _tmg.get("data_wait_fraction_of_train")
            L.append(f"  timing: {_tmg.get('category_seconds','?')}s total "
                     f"(setup {_tmg.get('setup_seconds','?')}s / "
                     f"train {_tmg.get('train_seconds','?')}s / "
                     f"eval {_tmg.get('eval_seconds','?')}s)"
                     + (f", data-wait {100*_f:.0f}% of train" if _f is not None else ""))
        if "model" in summary:
            m = summary["model"]
            # Pooling, its learned exponent and the input resolution all belong on
            # this line: each changes what the number means, and a result whose
            # settings are not recorded next to it cannot be compared to anything
            # later. The dossier's open question #1 exists because resolution was
            # never echoed anywhere.
            pool, p = m.get("pooling"), m.get("pooling_p")
            L.append(f"  model: {m.get('backbone')}/{m.get('head')}"
                     + (f"/{pool}" if pool else "")
                     + (f" (learned p={p})" if p is not None else "")
                     + (f" @{summary['image_size']}px" if summary.get("image_size") else "")
                     + f"  params={m.get('total_params'):,} size={_fmt(m.get('size_mb'),2)}MB")
        # §2.2/§2.4 reliability labelling
        if summary.get("reliability_tier"):
            L.append(f"  reliability tier: {summary['reliability_tier']}"
                     + ("  (anomalies moved out of test by us)"
                        if summary["reliability_tier"] == "injected" else "  (dataset-native)"))
        cavs = summary.get("caveats") or []
        if cavs:
            L.append(f"  CAVEATS: {caveat_line(cavs)}")
        if summary.get("reportable") is False:
            L.append("  NOT REPORTABLE — excluded from tier means")

    return "\n".join(L)


def _find_record_dirs(run_dir: str) -> List[str]:
    """Recursively find dirs that hold records (summary.json or stage.jsonl).

    Keys may contain slashes (e.g. 'mvtec/squeezenet/screw'), producing nested
    directories. We want the leaf record dirs, returned as paths relative to
    run_dir, not the intermediate folders.
    """
    found = []
    for root, _dirs, files in os.walk(run_dir):
        if "summary.json" in files or "stage.jsonl" in files:
            rel = os.path.relpath(root, run_dir)
            if rel != ".":
                found.append(rel)
    return sorted(found)


def run_report(run_dir: str) -> str:
    """Report for every category in a run, plus the run-level summary table."""
    cats = _find_record_dirs(run_dir)
    blocks = [category_report(run_dir, c) for c in cats]

    rs_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(rs_path):
        rs = json.load(open(rs_path))
        blocks.append(_render_run_summary(rs))
    return "\n\n".join(blocks)


def _render_run_summary(rs: dict) -> str:
    L = ["=" * 78, "RUN SUMMARY", "=" * 78]
    rows = rs.get("per_category", {})
    if rows:
        L.append(f"  {'category':<30}{'auroc':>9}{'aupr':>9}{'f1':>9}{'ece':>9}  tier/caveats")
        for cat, m in rows.items():
            marks = []
            if m.get("_tier"):
                marks.append(m["_tier"])
            if m.get("_caveats"):
                marks.append("!" + ",".join(c.replace("low_", "").replace("_anomalies", "")
                                            for c in m["_caveats"]))
            if m.get("_reportable") is False:
                marks.append("EXCLUDED")
            L.append(f"  {cat:<30}{_fmt(m.get('auroc')):>9}{_fmt(m.get('aupr')):>9}"
                     f"{_fmt(m.get('f1')):>9}{_fmt(m.get('ece')):>9}  {' '.join(marks)}")

    # §2.4: means per tier AND per backbone. Neither axis is ever blended.
    mbt = rs.get("means_by_tier")
    if mbt:
        L.append("  " + "-" * 74)
        L.append("  MEANS BY RELIABILITY TIER x BACKBONE (native=VisA, injected=MVTec —")
        L.append("  tiers are not comparable, and backbones are not averaged together):")
        L.append(f"  {'group':<30}{'auroc':>9}{'aupr':>9}{'f1':>9}{'ece':>9}")
        for tier, by_bb in mbt.items():
            # tolerate the older flat {tier: mean} shape
            if isinstance(by_bb, dict) and "auroc" in by_bb:
                label = f"MEAN[{tier}] n={by_bb.get('n_categories', '?')}"
                L.append(f"  {label:<30}{_fmt(by_bb.get('auroc')):>9}{_fmt(by_bb.get('aupr')):>9}"
                         f"{_fmt(by_bb.get('f1')):>9}{_fmt(by_bb.get('ece')):>9}")
                continue
            for bb, m in by_bb.items():
                label = f"MEAN[{tier}/{bb}] n={m.get('n_categories', '?')}"
                L.append(f"  {label:<30}{_fmt(m.get('auroc')):>9}{_fmt(m.get('aupr')):>9}"
                         f"{_fmt(m.get('f1')):>9}{_fmt(m.get('ece')):>9}")
        n_ex = rs.get("n_excluded_from_means", 0)
        if n_ex:
            L.append(f"  ({n_ex} result(s) excluded from means: collapsed / selection-failed / invalid)")
    return "\n".join(L)
