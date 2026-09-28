#!/usr/bin/env python
"""Recompute the thesis's headline operating-point numbers from results/per_image_scores_final.csv (no GPU, no images).
5 % split-conformal line per category-seed: the ceil((n+1)*0.95)-th smallest NORMAL score of the threshold set (MVTec: validation
normals; VisA: the 110 calibration normals); an image alarms if its two-view score is above the line. MVTec test set =
defects + normals from the test folder (like-for-like). Metrics: mean over seeds, then mean over categories."""
import csv, math, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # toothbrush has no 5 % line (12 validation normals): NaN by design
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
R = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/per_image_scores_final.csv")):
    if r["dataset"] == "mvtec" and r["split"] == "test" and r["label"] == "0" and "/test/good/" not in r["file"]: continue
    R[(r["dataset"], r["category"], r["seed"])][r["split"]].append((int(r["label"]), float(r["score_two_view"])))
def auroc(y, s):
    y = np.asarray(y); rk = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
for ds in ("mvtec", "visa"):
    cat = defaultdict(list)
    for (d, c, s), sp in R.items():
        if d != ds: continue
        ref = sp["calib"] if sp.get("calib") else sp["val"]; v = np.sort([x for y, x in ref if y == 0]); n = len(v); k = math.ceil((n + 1) * 0.95)
        y = np.array([a for a, _ in sp["test"]]); sc = np.array([b for _, b in sp["test"]])
        line = v[k - 1] if k <= n else None
        fa = np.mean(sc[y == 0] > line) if line is not None else np.nan; rec = np.mean(sc[y == 1] > line) if line is not None else np.nan
        cat[c].append((fa, rec, auroc(y, sc)))
    m = np.array([np.nanmean(v, axis=0) for v in cat.values()])
    k = int(np.sum(~np.isnan(m[:, 0])))
    print(f"{ds}: false alarms @5% {100 * np.nanmean(m[:, 0]):.1f} % and recall {np.nanmean(m[:, 1]):.3f} (mean of {k} category means "
          f"with a 5 % line) | AUROC {np.mean(m[:, 2]):.4f} ({len(cat)} categories)")
