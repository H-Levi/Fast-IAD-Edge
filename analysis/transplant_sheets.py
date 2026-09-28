#!/usr/bin/env python
"""Contact sheets of what the transplant actually trains on (F-L131 data collection; no effect on any training).

For each category: rebuild the exact training split from the config (same seed -> same splits), apply the transplant
exactly as training does (run_train.apply_transplant), and save one PNG with up to N synthetic records as rows:
[donor defect image | host normal | pasted result]. Lets us SEE what the pasted defects look like — never done before;
screw's harm (F-L126/F-L128/F-L130) is still unexplained.
    python analysis/transplant_sheets.py --config configs/baseline.yaml --set data.transplant.enabled=true ... \
        --categories screw,carpet --seed 123 --out runs/t2_full_sheets
"""
import argparse, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sheet(records, op, n=12, size=192):
    """[donor | host | pasted] rows for the first n synthetic records; returns a PIL image (3*size x rows*size)."""
    syn = [r for r in records if len(r) > 2 and r[2] == "transplant"][:n]
    canvas = Image.new("RGB", (3 * size, max(1, len(syn)) * size), (0, 0, 0))
    for i, r in enumerate(syn):
        di = r[3] if len(r) > 3 else 0
        host = Image.open(r[0]).convert("RGB").resize((size, size), Image.BILINEAR)
        donor = Image.open(op.donors[di][0]).convert("RGB").resize((size, size), Image.BILINEAR)
        pasted = op(host.copy(), donor_idx=di)
        for j, im in enumerate((donor, host, pasted)):
            canvas.paste(im, (j * size, i * size))
    return canvas


def main():
    import run_train
    from utils.io import load_config, apply_overrides
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--categories", required=True); ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a.set); cfg["seed"] = a.seed; os.makedirs(a.out, exist_ok=True)
    for cat in a.categories.split(","):
        ds = run_train.build_dataset("mvtec", cfg, cat)
        recs, op, desc = run_train.apply_transplant(ds.records("train"), ds, "mvtec", cfg)
        if op is None:
            print(f"{cat}: transplant not applied ({desc.get('reason')})"); continue
        sheet(recs, op, n=a.n).save(os.path.join(a.out, f"{cat}_s{a.seed}.png"))
        print(f"{cat}: sheet saved ({desc.get('n_paste_donors')} paste donors, {desc.get('n_synthetic')} synthetic)")


if __name__ == "__main__":
    main()
