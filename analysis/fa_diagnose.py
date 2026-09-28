#!/usr/bin/env python
"""Diagnose false alarms of trained runs (F-L136). No training, no model — reads saved scores and the images.

For each run directory (a category folder with scores_test_tta.npz / scores_val_tta.npz):
  * the 5% conformal line from the validation normals (rank ceil((n+1)*0.95), two-view = mean of id and rot+10);
  * the test-folder normals above it (false alarms), with score and margin;
  * a picture sheet: false alarms | the 4 lowest-scoring test normals | the 4 top validation normals, each labelled
    with its file name and score;
  * the 12 image statistics of analysis/session_stats.py for false-alarm vs other test normals vs validation normals,
    and per statistic the AUROC separating false-alarm images from the other test normals (which property drives them).
    python analysis/fa_diagnose.py --category grid --runs runs/t2s_G*/... --out runs/fa_diag
"""
import argparse, glob, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.session_stats import stats, STATS, auroc


def two(d, split):
    z = np.load(os.path.join(d, f"scores_{split}_tta.npz"), allow_pickle=True)
    return z["logits_views"][:, [0, 2]].mean(1), np.asarray(z["labels"]).ravel(), np.array([str(p) for p in z["paths"]])


def diagnose(d, out, tag):
    l, y, P = two(d, "test"); lv, yv, Pv = two(d, "val")
    cn = np.sort(lv[yv == 0]); n = len(cn); r = math.ceil((n + 1) * 0.95); t = cn[r - 1] if r <= n else np.inf
    tg = np.array(["/test/" in p for p in P]); nrm = np.where((y == 0) & tg)[0]
    fa = [i for i in nrm if l[i] > t]; ok = [i for i in nrm if l[i] <= t]
    low = sorted(ok, key=lambda i: l[i])[:4]; vtop = np.argsort(np.where(yv == 0, lv, -np.inf))[-4:][::-1]
    lines = [f"{tag}: line {t:+.2f} (rank {r}/{n}); test-folder normals {len(nrm)}, false alarms {len(fa)}"]
    lines += [f"  FA {P[i].split('/')[-3]}/{P[i].split('/')[-2]}/{os.path.basename(P[i])} score {l[i]:+.2f} margin {l[i] - t:+.2f}" for i in fa]
    cells = [(P[i], f"FA {os.path.basename(P[i])} {l[i]:+.2f}") for i in fa] + [(P[i], f"low {os.path.basename(P[i])} {l[i]:+.2f}") for i in low] \
          + [(Pv[i], f"val {os.path.basename(Pv[i])} {lv[i]:+.2f}") for i in vtop]
    s = 160; W = Image.new("RGB", (4 * s, math.ceil(len(cells) / 4) * (s + 14)), (255, 255, 255))
    for k, (p, lab) in enumerate(cells):
        x, yy = (k % 4) * s, (k // 4) * (s + 14); W.paste(Image.open(p).convert("RGB").resize((s, s)), (x, yy + 14)); ImageDraw.Draw(W).text((x + 2, yy + 1), lab[:26], fill=(0, 0, 0))
    W.save(os.path.join(out, f"{tag}_fa_sheet.png"))
    if fa and ok:
        S = {k: np.array([stats(P[i]) for i in v]) for k, v in (("fa", fa), ("ok", ok))}; S["val"] = np.array([stats(p) for p in Pv[yv == 0]])
        lines.append("  statistic      FA mean | other test normals | validation normals | AUROC FA vs other")
        for j, name in enumerate(STATS):
            a = auroc(np.r_[np.ones(len(fa)), np.zeros(len(ok))], np.r_[S["fa"][:, j], S["ok"][:, j]])
            lines.append(f"  {name:13s} {S['fa'][:, j].mean():9.3f} | {S['ok'][:, j].mean():9.3f} | {S['val'][:, j].mean():9.3f} | {a:.2f}")
    return lines


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--category", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); report = []
    for rd in a.runs:
        d = os.path.join(rd, "mvtec", "efficientnet", a.category)
        if not os.path.exists(os.path.join(d, "scores_test_tta.npz")): continue
        import json
        cfg = json.load(open(os.path.join(rd, "config.json"))); tag = f"{os.path.basename(os.path.dirname(rd.rstrip('/')))}_s{cfg['seed']}"
        report += diagnose(d, a.out, tag)
    open(os.path.join(a.out, f"{a.category}_fa_report.txt"), "w").write("\n".join(report) + "\n"); print("\n".join(report))


if __name__ == "__main__":
    main()
