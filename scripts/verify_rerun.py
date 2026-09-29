#!/usr/bin/env python
"""Verify that a new training run reproduces a reference run of the thesis (same category, same seed, same settings).

    python scripts/verify_rerun.py --new <new run>/mvtec/efficientnet/carpet --ref results/reference_runs/A_mvtec_carpet_s7
    python scripts/verify_rerun.py ... --control     # perturbs one new score by 0.001: check 3 MUST then FAIL

Checks, in order (fixed before any rerun); the script stops at the first failure of 1 or 2:
  1. image lists: the same files with the same labels in every scored split (val, test, calib)
  2. training: same best epoch and number of epochs; for transplant runs the same number of pasted defects and hosts
  3. per-image logits, all four stored views: max |difference| = 0 (identical environment) or <= 1e-4 with identical alarm
     decisions (different torch/CUDA version from the reference's)
  4. metrics from the two-view score (mean of identity and +10 degree views): false alarms and recall at the 5 % split-conformal
     line (and at 1 % where the threshold set allows it), precision at 5 % prevalence, ECE (15 bins, prior-corrected),
     AUROC and AUPR; MVTec test normals = files from the test folder (like-for-like). Same tolerance as check 3.
Prints PASS/FAIL per check and a final VERDICT; exit code 0 only if every check passes."""
import argparse, glob, json, math, os, sys
import numpy as np
from scipy.stats import rankdata


def load(d, split):
    f = os.path.join(d, f"scores_{split}_tta.npz")
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True)
    rel = ["/".join(str(p).split("/")[-3:]) for p in z["paths"]]
    return {k: (int(y), np.asarray(v, float)) for k, y, v in zip(rel, np.asarray(z["labels"]).ravel(), z["logits_views"])}


def metrics(splits, pos_weight, mvtec):
    ref = splits.get("calib") or splits["val"]
    nrm = np.sort([v[[0, 2]].mean() for y, v in ref.values() if y == 0]); n = len(nrm)
    t = {k: (y, v[[0, 2]].mean()) for k, (y, v) in splits["test"].items() if not (mvtec and y == 0 and not k.startswith("test/good"))}
    y = np.array([a for a, _ in t.values()]); s = np.array([b for _, b in t.values()])
    out = {}
    for a in (0.05, 0.01):
        k = math.ceil((n + 1) * (1 - a))
        if k > n: continue
        line = nrm[k - 1]; fa = float(np.mean(s[y == 0] > line)); rec = float(np.mean(s[y == 1] > line))
        out[f"false_alarms@{a:g}"], out[f"recall@{a:g}"] = fa, rec
        if a == 0.05: out["precision@5%prev"] = 0.05 * rec / (0.05 * rec + 0.95 * fa) if rec + fa > 0 else float("nan")
    p = 1 / (1 + np.exp(-(s - math.log(pos_weight or 1.0)))); e = 0.0
    for i in range(15):
        m = (p >= i / 15) & ((p < (i + 1) / 15) if i < 14 else (p <= 1))
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    out["ECE"] = e; out["AUROC"] = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    order = np.argsort(-s); tp = np.cumsum(y[order]); out["AUPR"] = float(np.sum(tp[y[order] == 1] / (np.arange(len(y)) + 1)[y[order] == 1]) / n1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--control", action="store_true"); a = ap.parse_args()
    meta = json.load(open(os.path.join(a.ref, "meta.json"))); mvtec = "/mvtec/" in a.new or os.path.basename(a.ref).split("_")[1] == "mvtec"
    import importlib.util
    torch_v = importlib.import_module("torch").__version__ if importlib.util.find_spec("torch") else "unknown"
    same_env = torch_v.startswith("2.10.0"); tol = 0.0 if same_env else 1e-4
    print(f"reference {os.path.basename(a.ref)} (original commit {meta['original_commit']}, {meta['original_env']}); verifying with torch "
          f"{torch_v} -> tolerance {'0 (identical environment)' if same_env else '1e-4 + identical alarm decisions'}")
    ok_all = True
    def report(i, name, ok, detail):
        nonlocal ok_all; ok_all &= ok; print(f"  {i}. {name}: {'PASS' if ok else 'FAIL'}  {detail}")
    N = {sp: load(a.new, sp) for sp in ("val", "test", "calib")}; R = {sp: load(a.ref, sp) for sp in ("val", "test", "calib")}
    if a.control:
        k = sorted(N["test"])[0]; y, v = N["test"][k]; v = v.copy(); v[:] += 1e-3; N["test"][k] = (y, v)
    same = all((N[sp] is None) == (R[sp] is None) and (R[sp] is None or {k: y for k, (y, _) in N[sp].items()} == {k: y for k, (y, _) in R[sp].items()})
               for sp in N)
    report(1, "image lists and labels (val, test, calib)", same, f"test {len(N['test'] or {})} vs {len(R['test'] or {})} images")
    if not same: print("VERDICT: FAIL (not the same experiment)"); sys.exit(1)
    s = json.load(open(os.path.join(a.new, "summary.json"))); tp = next((json.loads(l) for l in open(os.path.join(a.new, "stage.jsonl")) if '"transplant"' in l), {})
    tr_ok = (not meta["transplant_applied"]) or (tp.get("n_synthetic") == meta["n_synthetic"] and tp.get("n_distinct_hosts") == meta["n_distinct_hosts"])
    ep_ok = s.get("best_epoch") == meta["best_epoch"] and s.get("epochs_ran") == meta["epochs_ran"] and tr_ok
    report(2, "best epoch, epochs run, transplant plan counts", ep_ok, f"best {s.get('best_epoch')} vs {meta['best_epoch']}, ran {s.get('epochs_ran')} vs {meta['epochs_ran']}"
           + (f", pasted {tp.get('n_synthetic')} vs {meta['n_synthetic']}, hosts {tp.get('n_distinct_hosts')} vs {meta['n_distinct_hosts']}" if meta["transplant_applied"] else ""))
    if not ep_ok: print("VERDICT: FAIL (training diverged)"); sys.exit(1)
    dmax = max(float(np.max(np.abs(N[sp][k][1] - R[sp][k][1]))) for sp in N if R[sp] for k in R[sp])
    mn, mr = metrics(N, meta["pos_weight"], mvtec), metrics(R, meta["pos_weight"], mvtec)
    alarms_same = all(abs(mn[k] - mr[k]) == 0 for k in mn if k.startswith(("false_alarms", "recall")))
    report(3, "per-image logits, all 4 views", dmax <= tol and (same_env or alarms_same), f"max |diff| {dmax:.2e}")
    dm = {k: abs(mn[k] - mr[k]) for k in mr}
    report(4, "metrics", all(v <= tol or (math.isnan(mn[k]) and math.isnan(mr[k])) for k, v in dm.items()),
           " ".join(f"{k} {mn[k]:.4f}/{mr[k]:.4f}" for k in mr))
    print("VERDICT:", "PASS — the new run reproduces the reference" if ok_all else "FAIL"); sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
