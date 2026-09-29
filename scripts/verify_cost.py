#!/usr/bin/env python
"""Verify model cost against the thesis (CPU, no data, random weights): parameters must match EXACTLY, GFLOPs within 0.01.
    python scripts/verify_cost.py [--efficientad_root <clone of nelson1425/EfficientAD>] [--kairos_root <clone of KairosAD>]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPECTED = {"ours": (4_796_541, 1.324), "efficientad_s": (8_057_856, 75.907), "kairosad": (21_277_357, 83.953)}  # KairosAD: our build (paper: 11.53 M)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--efficientad_root", default=""); ap.add_argument("--kairos_root", default=""); a = ap.parse_args()
    import torch
    from analysis.cost_table import row
    from models.model_factory import AnomalyClassifier
    models = {"ours": (lambda: AnomalyClassifier("efficientnet", "deep", pretrained=False, pooling="avg"), 288)}
    if a.efficientad_root:
        from analysis.benchmark_efficientad import EfficientADS
        models["efficientad_s"] = (lambda: EfficientADS(a.efficientad_root), 256)
    if a.kairos_root:
        from analysis.benchmark_kairosad import build_kairosad, _Summarised
        models["kairosad"] = (lambda: _Summarised(build_kairosad(a.kairos_root, torch.device("cpu"))[0], "kairosad", layout="NHWC"), 288)
    ok = True
    for name, (fn, px) in models.items():
        m = fn(); n = sum(p.numel() for p in m.parameters()); g = row(name, m, px)["GFLOPs_per_image"]
        en, eg = EXPECTED[name]; good = n == en and abs(g - eg) <= 0.01; ok &= good
        print(f"{name}: parameters {n:,} (expected {en:,}), GFLOPs per view {g:.3f} (expected {eg:.3f}) -> {'PASS' if good else 'FAIL'}")
    print("VERDICT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
