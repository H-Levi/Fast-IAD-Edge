#!/usr/bin/env python
"""Compare a new same-session timing (analysis/head_to_head.py --full --twoview, T4) with the thesis's session. Timings are never
identical between sessions, so the rule (fixed in the thesis's pre-registration) is: each model's BEST full-optimisation time within
15 % of the reference, and the same ranking (ours two views < EfficientAD-S < KairosAD).
    python scripts/verify_timing.py --new_ours_kairos <csv> --new_efficientad <csv>
Note: the reference CSVs were written before a fix — their '[x2]' rows hold HALF the batch time; this script doubles them."""
import argparse, csv, os, sys
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "reference_runs", "timing_same_session")
def best(files, fix_x2):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["status"] != "ok" or "+full" not in r["model"]: continue
            ms = float(r["ms_bs1"]) * (2 if fix_x2 and r["model"].endswith("[x2]") else 1)
            key = "ours two views" if r["model"].endswith("[x2]") else ("ours one view" if r["model"].startswith("ours") else r["model"].split("+")[0])
            out[key] = min(out.get(key, 1e9), ms)
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new_ours_kairos", required=True); ap.add_argument("--new_efficientad", required=True); a = ap.parse_args()
    ref = best([os.path.join(REF, "best_ours_kairos.csv"), os.path.join(REF, "best_efficientad.csv")], True)
    new = best([a.new_ours_kairos, a.new_efficientad], False); ok = True
    for k in ("ours one view", "ours two views", "efficientad_s", "kairosad"):
        d = (new[k] - ref[k]) / ref[k]; good = abs(d) <= 0.15; ok &= good
        print(f"{k}: {new[k]:.2f} ms (reference {ref[k]:.2f}, {d:+.1%}) -> {'PASS' if good else 'FAIL'}")
    rank = new["ours two views"] < new["efficientad_s"] < new["kairosad"]; ok &= rank
    print(f"ranking ours two views < EfficientAD-S < KairosAD: {'PASS' if rank else 'FAIL'}; ratios {new['efficientad_s'] / new['ours two views']:.2f} and {new['kairosad'] / new['ours two views']:.2f} (thesis 1.91 / 5.65)")
    print("VERDICT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
