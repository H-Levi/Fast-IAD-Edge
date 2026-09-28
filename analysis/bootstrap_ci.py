#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bootstrap_ci.py  —  confidence intervals and paired comparisons from saved test
scores. NO GPU, NO RETRAINING.

Why this exists
---------------
Seeds measure TRAINING variance ("would another seed give another model?").
They say nothing about EVALUATION variance ("is this test set big enough to
resolve the difference at all?"). MVTec bottle's test split is 20 normal + 47
anomalous = 67 images. A 2-point AUROC gap on 67 images may be pure sampling
noise, and no number of seeds will tell you that.

RR-001 (Agarwal et al., "Deep RL at the Edge of the Statistical Precipice") is
the reference: stratified bootstrap percentile intervals, the interquartile mean
instead of the mean or median, and probability-of-improvement instead of a
point-estimate comparison. This applies it to `scores_test.npz`, which already
holds raw per-sample logits and labels — so every interval below costs nothing
but CPU time.

MEASURED LIMITATION — read this before quoting a single-run interval
--------------------------------------------------------------------
A coverage check on 2026-08-30 (300 simulated datasets per cell, known true AUROC,
1000 replicates each) found that SINGLE-RUN percentile intervals UNDERCOVER, and
that they get worse as AUROC approaches 1:

    split                     true    coverage (nominal 95%)
    MVTec bottle  20n/47a     0.90      93.0%
    MVTec bottle  20n/47a     0.97      87.7%   <-- undercovers
    MVTec screw   41n/89a     0.90      93.7%
    MVTec screw   41n/89a     0.97      92.0%
    VisA capsules 241n/40a    0.90      89.7%   <-- undercovers
    VisA capsules 241n/40a    0.97      85.0%   <-- undercovers

This is the percentile bootstrap's known boundary behaviour: as AUROC -> 1 the
sampling distribution is skewed and truncated at 1.0, and the plain percentile
interval does not correct for it. BCa would; that is not a small change.

**A single-run interval printed by this tool near AUROC 0.97 is closer to an 85-88%
interval than a 95% one. Do not report it as 95%.**

The PAIRED path is sound and is what every claim in the paper rests on:
identical scorers give max |delta| = 0.00e+00 over 2000 replicates (exact), and
paired-null coverage is 93.7 / 92.3 / 93.0% on the three splits above -- mildly
conservative against nominal 95%.

THE RESOLUTION FLOOR IS PER-SPLIT AND PER-COMPARISON, NOT A CONSTANT
--------------------------------------------------------------------
Measured median paired-delta 95% CI HALF-widths, two correlated models at true
AUROC 0.93:

    MVTec bottle  20n/47a     0.0290
    MVTec screw   41n/89a     0.0182
    VisA capsules 241n/40a    0.0148

So the "~0.04 floor" quoted elsewhere in this project is an MVTec-sized number.
On VisA's larger test splits the real floor is less than half of it. Worse, the
half-width also depends on how CORRELATED the two arms are -- the simulation above
fixed that arbitrarily -- so two dissimilar arms will have a wider floor than two
similar ones on the same split.

**Consequence: do not compare a delta against a remembered constant. Run this tool
on the actual scores and read whether the CI excludes zero.** That is the decision
rule; the floor is a planning heuristic only.

Three things it does that a naive implementation gets wrong
-----------------------------------------------------------
1. STRATIFIED resampling — normals and anomalies are resampled within class, so
   every replicate keeps the original class counts. Unstratified resampling can
   draw zero anomalies, leaving AUROC undefined, and silently biases the interval
   on small, imbalanced splits (which is all of MVTec here).

2. PAIRED comparison — when comparing two models on the SAME test set, both are
   scored on the SAME resampled indices. The paired difference has far lower
   variance than the difference of two independently-bootstrapped CIs.
   *Two overlapping CIs do NOT mean "no difference".* Read the delta CI, not the
   overlap of the individual ones.

3. IQM for aggregation — a 25% trimmed mean over pooled replicates. On few runs
   the median is badly behaved (RR-001: expected median shift 0.05 vs IQM 0.006).

Usage
-----
    # intervals for one run
    python analysis/bootstrap_ci.py runs/<run_dir>

    # paired comparison: does B beat A?   (e.g. pooling=max vs pooling=avg)
    python analysis/bootstrap_ci.py runs/<run_avg> --vs runs/<run_max>

    # options
    --metric auroc|aupr   --n 10000   --alpha 0.05   --json out.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ metrics --
# Pure numpy on purpose: the bootstrap calls these ~10k times per category, and
# these are both exact and far cheaper than the sklearn equivalents. It also
# keeps this tool runnable anywhere (no sklearn needed), which matters because it
# is meant to run offline on saved scores, not inside the training environment.

def auroc(y, s):
    """Rank-based AUROC (the Mann-Whitney U identity). Exact, ties averaged."""
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=float)
    ranks[order] = np.arange(1, y.size + 1, dtype=float)
    # average the ranks of tied scores, or tied logits would bias the result
    s_sorted = s[order]
    i = 0
    while i < s_sorted.size:
        j = i + 1
        while j < s_sorted.size and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aupr(y, s):
    """Average precision — the step-wise sum sklearn's average_precision_score uses."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * yy).sum() / n_pos)


METRICS = {"auroc": auroc, "aupr": aupr}


# ----------------------------------------------------------------- loading --

def find_score_files(run_dir):
    """-> {category_key: path}, keyed by the path RELATIVE to run_dir.

    run_train keys categories as '<dataset>/<backbone>/<category>' and the recorder
    turns that into nested directories, so the score file for EfficientNet on bottle
    is run_dir/mvtec/efficientnet/bottle/scores_test.npz.

    Keying on the basename ('bottle') would collide every backbone for a category
    and silently keep whichever was globbed last -- comparing mobilenet against
    mobilenet while discarding efficientnet, with no error. Key on the relative
    path so each (dataset, backbone, category) stays distinct.
    """
    root = os.path.abspath(run_dir)
    out, skipped = {}, []
    for p in sorted(glob.glob(os.path.join(root, "**", "scores_test.npz"),
                              recursive=True)):
        key = os.path.relpath(os.path.dirname(p), root).replace(os.sep, "/")
        if not _reportable(p):
            skipped.append(key)
            continue
        out[key] = p
    if skipped:
        # Named, never silent: a dropped category is as much a finding as a kept one.
        print(f"  ! excluded {len(skipped)} non-reportable categories "
              f"(collapsed / single-class / selection failed): {', '.join(skipped)}")
    return out



def _reportable(score_path):
    """True unless the sibling summary.json marks this category structurally invalid.

    run_train.py:452 computes `reportable` from caveats + collapsed + selection_failed
    and writes it beside every score file. Until 2026-08-30 NOTHING read it back: both
    this module and read_factorial.py globbed every scores_test.npz, so a category the
    harness itself prints as "AUROC is NOT a real result" was averaged in silently.
    Missing or unreadable summary.json -> True, because absence of evidence is not
    evidence of collapse; the caller prints what it skipped either way.
    """
    import json as _json
    sp = os.path.join(os.path.dirname(score_path), "summary.json")
    try:
        with open(sp, encoding="utf-8") as fh:
            return bool(_json.load(fh).get("reportable", True))
    except Exception:
        return True


def _require_scores(run_dir, files, label):
    """Fail loudly and usefully when a run directory yields nothing."""
    if files:
        return
    if not os.path.isdir(run_dir):
        sys.exit(f"{label}: no such directory\n  {run_dir}")
    sub = sorted(os.listdir(run_dir))[:12]
    sys.exit(f"{label}: directory exists but contains no scores_test.npz\n"
             f"  {run_dir}\n"
             f"  top level: {sub}\n"
             f"  (is instrumentation.save_raw_scores enabled? did the run finish?)")


def load_scores(path):
    d = np.load(path, allow_pickle=True)
    y = np.asarray(d["labels"]).astype(int).ravel()
    s = np.asarray(d["logits"]).astype(float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"{path}: labels {y.shape} != logits {s.shape}")
    return y, s


# --------------------------------------------------------------- bootstrap --

def stratified_index_matrix(y, n_boot, rng):
    """[n_boot, N] resample indices, drawn within each class.

    Preserves the exact per-class counts of the original split, so AUROC is
    always defined and the interval is not distorted by class-count wobble.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None
    p = rng.integers(0, len(pos), size=(n_boot, len(pos)))
    n = rng.integers(0, len(neg), size=(n_boot, len(neg)))
    return np.concatenate([pos[p], neg[n]], axis=1)


def bootstrap_values(y, s, metric_fn, n_boot, rng, idx=None):
    """-> [n_boot] metric values. Pass `idx` to reuse indices (paired mode)."""
    if idx is None:
        idx = stratified_index_matrix(y, n_boot, rng)
    if idx is None:
        return None
    return np.array([metric_fn(y[i], s[i]) for i in idx])


def iqm(v):
    """Interquartile mean — the middle 50%. RR-001's recommended aggregate."""
    v = np.sort(np.asarray(v))
    lo, hi = int(len(v) * 0.25), int(np.ceil(len(v) * 0.75))
    core = v[lo:hi]
    return float(core.mean()) if core.size else float(v.mean())


def pct_ci(v, alpha):
    return (float(np.percentile(v, 100 * alpha / 2)),
            float(np.percentile(v, 100 * (1 - alpha / 2))))


# -------------------------------------------------------------------- runs --

def single_run(run_dir, metric, n_boot, alpha, seed=0):
    fn = METRICS[metric]
    files = find_score_files(run_dir)
    _require_scores(run_dir, files, "run")
    rows = []
    for cat, path in files.items():
        y, s = load_scores(path)
        rng = np.random.default_rng(seed)
        v = bootstrap_values(y, s, fn, n_boot, rng)
        if v is None:
            rows.append({"category": cat, "n": len(y), "n_pos": int(y.sum()),
                         "point": None, "note": "single-class test split"})
            continue
        lo, hi = pct_ci(v, alpha)
        rows.append({"category": cat, "n": int(len(y)), "n_pos": int(y.sum()),
                     "point": fn(y, s), "iqm": iqm(v), "lo": lo, "hi": hi,
                     "width": hi - lo})
    return rows


def paired_runs(dir_a, dir_b, metric, n_boot, alpha, seed=0):
    """B vs A on identical resamples. Requires identical test splits."""
    fn = METRICS[metric]
    fa, fb = find_score_files(dir_a), find_score_files(dir_b)
    _require_scores(dir_a, fa, "A (baseline)")
    _require_scores(dir_b, fb, "B (variant)")
    shared = sorted(set(fa) & set(fb))
    print(f"A: {len(fa)} score files | B: {len(fb)} | in common: {len(shared)}")
    if not shared:
        sys.exit(f"no category keys in common — the two runs cover different things:\n"
                 f"  A: {sorted(fa)[:8]}\n  B: {sorted(fb)[:8]}")
    for only, name in ((set(fa) - set(fb), "A"), (set(fb) - set(fa), "B")):
        if only:
            print(f"  ! only in {name}, skipped: {sorted(only)}")

    rows = []
    for cat in shared:
        ya, sa = load_scores(fa[cat])
        yb, sb = load_scores(fb[cat])
        if len(ya) != len(yb) or not np.array_equal(ya, yb):
            rows.append({"category": cat, "note":
                         "TEST SPLITS DIFFER — not comparable (check the protocol)"})
            continue
        rng = np.random.default_rng(seed)
        idx = stratified_index_matrix(ya, n_boot, rng)
        if idx is None:
            rows.append({"category": cat, "note": "single-class test split"})
            continue
        va = bootstrap_values(ya, sa, fn, n_boot, rng, idx=idx)
        vb = bootstrap_values(yb, sb, fn, n_boot, rng, idx=idx)
        d = vb - va
        lo, hi = pct_ci(d, alpha)
        rows.append({"category": cat, "n": int(len(ya)), "n_pos": int(ya.sum()),
                     "a": fn(ya, sa), "b": fn(yb, sb), "delta": fn(yb, sb) - fn(ya, sa),
                     "delta_iqm": iqm(d), "lo": lo, "hi": hi,
                     "p_improve": float((d > 0).mean()),
                     "resolved": bool(lo > 0 or hi < 0)})
    return rows


# ------------------------------------------------------------------ output --

def print_single(rows, metric, alpha):
    print(f"\n{metric.upper()} with stratified bootstrap "
          f"{int((1-alpha)*100)}% CI   (evaluation variance only; not seed variance)")
    print(f"{'category':<34}{'n':>5}{'pos':>5}{'point':>9}{'IQM':>9}{'CI low':>9}{'CI high':>9}{'width':>8}")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: x["category"]):
        if r.get("point") is None:
            print(f"{r['category']:<34}{r['n']:>5}{r['n_pos']:>5}   {r.get('note','')}")
            continue
        print(f"{r['category']:<34}{r['n']:>5}{r['n_pos']:>5}{r['point']:>9.4f}"
              f"{r['iqm']:>9.4f}{r['lo']:>9.4f}{r['hi']:>9.4f}{r['width']:>8.4f}")
    w = [r["width"] for r in rows if r.get("point") is not None]
    if w:
        print("-" * 88)
        print(f"median CI width = {np.median(w):.4f}  "
              f"→ differences smaller than about this are NOT resolvable on these test splits.")


def print_paired(rows, metric, alpha, name_a, name_b):
    print(f"\nPAIRED {metric.upper()}:  B - A   (same bootstrap resamples for both)")
    print(f"  A = {name_a}\n  B = {name_b}")
    print(f"\n{'category':<30}{'A':>8}{'B':>8}{'delta':>9}{'CI low':>9}{'CI high':>9}{'P(B>A)':>9}  verdict")
    print("-" * 92)
    resolved_up = resolved_dn = 0
    for r in sorted(rows, key=lambda x: x["category"]):
        if "delta" not in r:
            print(f"{r['category']:<30}  {r.get('note','')}")
            continue
        if r["resolved"]:
            verdict = "BETTER" if r["delta_iqm"] > 0 else "WORSE"
            resolved_up += r["delta_iqm"] > 0
            resolved_dn += r["delta_iqm"] < 0
        else:
            verdict = "not resolved"
        print(f"{r['category']:<30}{r['a']:>8.4f}{r['b']:>8.4f}{r['delta']:>9.4f}"
              f"{r['lo']:>9.4f}{r['hi']:>9.4f}{r['p_improve']:>9.3f}  {verdict}")
    ok = [r for r in rows if "delta" in r]
    if ok:
        pooled = np.array([r["delta_iqm"] for r in ok])
        print("-" * 92)
        print(f"categories resolved better: {resolved_up}   worse: {resolved_dn}   "
              f"unresolved: {len(ok)-resolved_up-resolved_dn}")
        print(f"IQM of per-category deltas = {iqm(pooled):+.4f}")
        print("\nNOTE: this is EVALUATION variance on fixed models. Combine with the seed band from "
          "run_train run.seeds=[...] (each seed a complete run) before calling anything a result.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run directory (A, the baseline, in --vs mode)")
    ap.add_argument("--vs", default=None, help="second run directory (B, the change)")
    ap.add_argument("--metric", default="auroc", choices=[k for k, v in METRICS.items() if v])
    ap.add_argument("--n", type=int, default=10000, help="bootstrap replicates")
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 -> 95%% CI")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write results here")
    a = ap.parse_args()

    if a.vs:
        rows = paired_runs(a.run_dir, a.vs, a.metric, a.n, a.alpha, a.seed)
        print_paired(rows, a.metric, a.alpha, a.run_dir, a.vs)
    else:
        rows = single_run(a.run_dir, a.metric, a.n, a.alpha, a.seed)
        print_single(rows, a.metric, a.alpha)

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"metric": a.metric, "n_boot": a.n, "alpha": a.alpha,
                       "a": a.run_dir, "b": a.vs, "rows": rows}, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
