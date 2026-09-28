#!/usr/bin/env python
"""session_stats.py — SIDE TRACK step 1 (diagnose -> NAME the cue -> fix). CPU only, no training, no model.

Question: is the MVTec acquisition-session cue (train/good vs test/good, readable in the model's features in 8/15
categories — analysis A-051) a simple PHOTOMETRIC difference? Clue: augmentation off (incl. colour jitter) removed
carpet's false alarms (A-031).

Per category, every image of train/good (session t) and test/good (session T), resized to the model's 288 px:
  12 statistics: mean and std of R, G, B; luminance 5th/50th/95th percentiles; mean HSV saturation; sharpness
  (variance of a 3x3 Laplacian of luminance); edge density.
  - probe: leave-one-out nearest-class-mean on the standardised statistics -> AUROC; 200-shuffle permutation p.
  - the single statistic that separates the sessions best (its AUROC, oriented) -> NAMES the cue.
  - NEGATIVE CONTROL: the same probe on two RANDOM halves of train/good alone must sit near 0.5; if it does not, the
    probe itself is broken and nothing else in the table counts.
    python analysis/session_stats.py --mvtec_root /path/to/mvtec_ad [--categories carpet,screw]
"""
import argparse, os
import numpy as np
from PIL import Image
from scipy.stats import rankdata

STATS = ["meanR", "meanG", "meanB", "stdR", "stdG", "stdB", "lum_p05", "lum_p50", "lum_p95", "sat_mean", "sharpness", "edge_density"]


def auroc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stats(path, size=288, normalize=False):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64) / 255.0
    if normalize:   # step 1.5: per-image, per-channel standardisation (removes exposure, colour cast and contrast), then
        a = (a - a.reshape(-1, 3).mean(0)) / (a.reshape(-1, 3).std(0) + 1e-6) * 0.2 + 0.5   # back to a fixed mid-grey range
        a = np.clip(a, 0, 1); im = Image.fromarray((a * 255).astype(np.uint8))
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64) / 255.0
    lap = (-4 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:])
    gx, gy = np.abs(np.diff(lum, axis=1))[:-1, :], np.abs(np.diff(lum, axis=0))[:, :-1]
    return [*a.reshape(-1, 3).mean(0), *a.reshape(-1, 3).std(0), *np.percentile(lum, [5, 50, 95]),
            hsv[..., 1].mean(), lap.var(), float(((gx + gy) > 0.1).mean())]


def probe(X, y):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9); out = np.empty(len(y))
    s1, s0 = Z[y == 1].sum(0), Z[y == 0].sum(0); n1, n0 = (y == 1).sum(), (y == 0).sum()
    for i in range(len(y)):
        m1 = (s1 - Z[i] * (y[i] == 1)) / (n1 - (y[i] == 1)); m0 = (s0 - Z[i] * (y[i] == 0)) / (n0 - (y[i] == 0))
        out[i] = np.linalg.norm(Z[i] - m0) - np.linalg.norm(Z[i] - m1)
    return auroc(y, out)


def cv_logistic(X, y, reps=5):
    """5-fold stratified L2 logistic regression on standardised statistics, repeated `reps` times; mean AUROC of the
    out-of-fold scores. Not leave-one-out, so it does not have LOO nearest-mean's anti-learning bias."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    out = []
    for r in range(reps):
        sc = np.empty(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(C=0.5, max_iter=500).fit((X[tr] - mu) / sd, y[tr]); sc[te] = m.decision_function((X[te] - mu) / sd)
        out.append(auroc(y, sc))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mvtec_root", required=True); ap.add_argument("--categories", default="")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--normalize", action="store_true", help="per-image per-channel standardisation before the statistics")
    ap.add_argument("--subset_csv", default="", help="category,session(t|T),relative path — restrict to these images "
                    "(e.g. exactly the test normals the feature probe saw)")
    ap.add_argument("--probe", default="loo_mean", choices=["loo_mean", "cv_logistic"],
                    help="cv_logistic: 5-fold stratified L2 logistic, repeated 5x (robust where the LOO probe's control fails)")
    ap.add_argument("--gaps", action="store_true", help="print each category's session gap in plain units and exit "
                    "(brightness %% vs the training jitter's +-20%%, saturation / sharpness / edge-density ratios)")
    a = ap.parse_args()
    subset = {}
    if a.subset_csv:
        for line in open(a.subset_csv).read().splitlines()[1:]:
            c_, s_, rel = line.split(","); subset.setdefault(c_, {"t": [], "T": []})[s_].append(os.path.join(a.mvtec_root, rel))
    rng = np.random.default_rng(0)
    cats = a.categories.split(",") if a.categories else sorted(d for d in os.listdir(a.mvtec_root) if os.path.isdir(os.path.join(a.mvtec_root, d, "train", "good")))
    print(f"{'category':11s} {'n_t':>4s} {'n_T':>4s} | probe AUROC (p high / p low, two-sided) | CONTROL (random halves of t) | strongest single statistic"
          + ("   [NORMALIZED]" if a.normalize else "") + ("   [SUBSET]" if a.subset_csv else ""))
    if a.subset_csv: cats = [c for c in cats if c in subset]
    for c in cats:
        if a.subset_csv:
            tr, te = sorted(subset[c]["t"]), sorted(subset[c]["T"])
        else:
            tr = sorted(os.path.join(a.mvtec_root, c, "train", "good", f) for f in os.listdir(os.path.join(a.mvtec_root, c, "train", "good")))
            te = sorted(os.path.join(a.mvtec_root, c, "test", "good", f) for f in os.listdir(os.path.join(a.mvtec_root, c, "test", "good")))
        X = np.array([stats(p, normalize=a.normalize) for p in tr + te]); y = np.r_[np.zeros(len(tr)), np.ones(len(te))].astype(int)
        keep = list(range(len(STATS)))
        if a.normalize:
            # FIX (2026-09-24): after per-channel standardisation the six channel means/stds are CONSTANT by construction;
            # standardising near-constant columns turns them into amplified noise and breaks the probe (carpet control
            # 0.269 in the first run). Only the statistics normalisation does NOT fix by construction are kept.
            keep = [j for j, n in enumerate(STATS) if n not in ("meanR", "meanG", "meanB", "stdR", "stdG", "stdB")]  # explicit list
            X = X[:, keep]
        if a.gaps:
            med = lambda k, m: float(np.median(X[m, STATS.index(k)]))
            t, T = y == 0, y == 1
            print(f"{c:11s} brightness (lum_p50) T vs t {100 * (med('lum_p50', T) / med('lum_p50', t) - 1):+6.1f}% "
                  f"{'(INSIDE +-20% jitter)' if abs(med('lum_p50', T) / med('lum_p50', t) - 1) <= 0.2 else '(OUTSIDE +-20% jitter)'} | "
                  f"saturation x{med('sat_mean', T) / (med('sat_mean', t) + 1e-9):.2f} | sharpness x{med('sharpness', T) / (med('sharpness', t) + 1e-12):.2f} | "
                  f"edge density x{med('edge_density', T) / (med('edge_density', t) + 1e-9):.2f}", flush=True)
            continue
        pr = probe if a.probe == "loo_mean" else cv_logistic
        fa = pr(X, y); perm = [pr(X, rng.permutation(y)) for _ in range(a.perms)]
        # Two-sided: p_high = evidence of a cue. p_low small = the probe does WORSE than label-shuffles -- the
        # leave-one-out anti-learning artefact, NOT a reversed cue (a nearest-mean probe is label-symmetric).
        p = (1 + sum(v >= fa for v in perm)) / (1 + len(perm)); p_low = (1 + sum(v <= fa for v in perm)) / (1 + len(perm))
        Xt = X[: len(tr)]; half = rng.permutation(len(tr)) < len(tr) // 2
        ctrl = pr(Xt, half.astype(int))
        single = [(abs(auroc(y, X[:, i]) - 0.5), STATS[j], auroc(y, X[:, i])) for i, j in enumerate(keep)]
        best = sorted(single, reverse=True)[:2]
        print(f"{c:11s} {len(tr):4d} {len(te):4d} | {fa:.3f} (p {p:.3f} / {p_low:.3f}) | {ctrl:.3f}{' <- CONTROL FAILED' if abs(ctrl - 0.5) > 0.15 else ''} | "
              + ", ".join(f"{n} {v:.3f}" for _, n, v in best), flush=True)


if __name__ == "__main__":
    main()
