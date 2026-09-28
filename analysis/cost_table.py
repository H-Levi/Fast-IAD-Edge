#!/usr/bin/env python
"""cost_table.py (F-L143 companion, CPU-only, no timing): hardware-independent COST of each model — parameters, file size at
fp32 and fp16, and GMACs per image (fvcore; 1 MAC = 2 FLOPs) — for ours (one view and two views = 2 passes' arithmetic),
EfficientAD-S (256 px) and KairosAD (whatever size it is given: it resizes to 1024 internally). Random weights (cost does not
depend on weight values). CONTROL: EfficientAD-S parameters must match its paper (8 x10^6, Table 16).
    python analysis/cost_table.py --efficientad_root <clone> --kairos_root <clone> [--out file.csv]"""
import argparse, csv, logging, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def macs(model, size, nhwc=False):
    from fvcore.nn import FlopCountAnalysis
    logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)
    x = torch.randn(1, 3, size, size)
    f = FlopCountAnalysis(model.eval(), x); f.unsupported_ops_warnings(False); f.uncalled_modules_warnings(False)
    return f.total() / 1e9


def row(name, model, size, views=1, note=""):
    n = sum(p.numel() for p in model.parameters()); nb = sum(b.numel() for b in model.buffers())
    try:
        with torch.no_grad():
            g = macs(model, size)
    except Exception as e:  # noqa
        g, note = float("nan"), (note + f" MACs failed: {type(e).__name__}").strip()
    return {"model": name, "input_px": size, "views": views, "params_M": round(n / 1e6, 3),
            "size_fp32_MB": round((n + nb) * 4 / 2 ** 20, 1), "size_fp16_MB": round((n + nb) * 2 / 2 ** 20, 1),
            "GMACs_per_image": round(g * views, 3), "GFLOPs_per_image": round(2 * g * views, 3), "note": note}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--efficientad_root", default=""); ap.add_argument("--kairos_root", default="")
    ap.add_argument("--out", default=""); a = ap.parse_args()
    from models.model_factory import AnomalyClassifier
    ours = AnomalyClassifier("efficientnet", "deep", pretrained=False, pooling="avg")
    rows = [row("ours (EfficientNet-B0 + deep head), one view", ours, 288),
            row("ours, two views (deployed final model)", ours, 288, views=2, note="two passes' arithmetic; batched on GPU")]
    if a.efficientad_root:
        from analysis.benchmark_efficientad import EfficientADS
        r = row("EfficientAD-S (nelson1425 reimplementation)", EfficientADS(a.efficientad_root), 256)
        r["note"] = (r["note"] + f" CONTROL params vs paper 8M: {'OK' if abs(r['params_M'] - 8.0) < 0.5 else 'MISMATCH'}").strip()
        rows.append(r)
    if a.kairos_root:
        try:
            from analysis.benchmark_kairosad import build_kairosad, _Summarised
            k, _ = build_kairosad(a.kairos_root, torch.device("cpu"))
            rows.append(row("KairosAD (our build, full MSAM wrapper)", _Summarised(k, "kairosad", layout="NHWC"), 288,
                            note="paper states 11.53M params; our build counts all MobileSAM parts"))
        except Exception as e:  # noqa
            rows.append({"model": "KairosAD", "note": f"not built: {type(e).__name__}: {str(e)[:120]}"})
    for r in rows: print(r)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow({k: r.get(k, "") for k in rows[0]}) for r in rows]


if __name__ == "__main__":
    main()
