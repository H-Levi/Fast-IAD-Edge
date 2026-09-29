#!/usr/bin/env python
"""CPU speed check (fp32, eager, batch 1, same session for all three models). CPU models differ between machines, so absolute
times are reported but not judged; the rule is: the same ranking (ours < EfficientAD-S < KairosAD) and each ratio within 20 % of
the thesis's CPU session (EfficientAD-S / ours = 14.9, KairosAD / ours = 42.8).
    python scripts/verify_cpu.py --new_efficientad <csv> --new_ours_kairos <csv>"""
import argparse, csv, os, sys
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "reference_runs", "timing_cpu")
def ms(files):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["status"] == "ok" and r["precision"] == "fp32" and "+" not in r["model"]:
                out["ours" if r["model"].startswith("ours") else r["model"]] = float(r["ms_bs1"])
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new_efficientad", required=True); ap.add_argument("--new_ours_kairos", required=True); a = ap.parse_args()
    ref = ms([os.path.join(REF, "eff_cpu_efficientad.csv"), os.path.join(REF, "eff_cpu_ours_kairos.csv")]); new = ms([a.new_efficientad, a.new_ours_kairos]); ok = True
    for k in ("ours", "efficientad_s", "kairosad"): print(f"{k}: {new[k]:.1f} ms (thesis session {ref[k]:.1f} ms; CPU models may differ)")
    for k in ("efficientad_s", "kairosad"):
        rn, rr = new[k] / new["ours"], ref[k] / ref["ours"]; good = abs(rn - rr) / rr <= 0.20; ok &= good
        print(f"{k} / ours: {rn:.1f} (thesis {rr:.1f}, {(rn - rr) / rr:+.0%}) -> {'PASS' if good else 'FAIL'}")
    rank = new["ours"] < new["efficientad_s"] < new["kairosad"]; ok &= rank; print(f"ranking ours < EfficientAD-S < KairosAD: {'PASS' if rank else 'FAIL'}")
    print("VERDICT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
