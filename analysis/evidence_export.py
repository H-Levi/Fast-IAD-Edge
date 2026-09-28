#!/usr/bin/env python
"""Evidence exports for the thesis review (THesis_record_v2, REQUESTS 1, 3, 4, 10). CPU only, no training, no model.

  leaklists  : for every FINAL MVTec run (transplant v2, no swap; the certified seeds of each category) rebuild the exact
               training split from the config (same seed -> same split) and the transplant plan exactly as training does
               (run_train.apply_transplant); write every donor defect, host normal, validation and test image, and check:
               donors = training defects only, hosts = training normals only, validation/test hold only real images and
               share no file with donors, hosts or each other. Same code path as run_train (build_dataset + apply_transplant).                                          (REQUESTS 1)
  exif       : VisA normal images: file index and the EXIF capture times (tags 306, 36867)  (REQUESTS 3, order)
  copy       : copy listed files (full size) — the highest-scoring late VisA normals          (REQUESTS 3, late photos)
  pixelstats : 12 image statistics (analysis/session_stats.py) for every MVTec image: train/good, test/good, test defects
                                                                                              (REQUESTS 10)
  samples    : copy N original MVTec and VisA photos for CPU timing on a laptop               (REQUESTS 4)
"""
import argparse, csv, glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FINAL_SEEDS = {c: (7, 123, 2024) for c in ("cable", "capsule", "carpet", "grid", "hazelnut", "leather", "pill", "screw", "transistor")}
FINAL_SEEDS.update({c: (11, 123, 2024) for c in ("bottle", "metal_nut", "tile", "toothbrush", "wood", "zipper")})
FINAL_SET = ["train.class_weight_mode=neg_over_pos", "data.calib_n=110", "train.regime_min_defects=20", "data.transplant.enabled=true",
             "data.transplant.placement=best_match", "data.de_confound_frac=0.0"]


def leak_one(cfg, c):
    """One category-seed through the SAME code path as run_train (build_dataset + apply_transplant).
    Returns (csv rows, ok, one-line summary)."""
    import run_train
    assert not cfg["augmentation"].get("enabled", False), "record expansion would change the training list"
    s = cfg["seed"]; ds = run_train.build_dataset("mvtec", cfg, c); train = ds.records("train")
    recs, op, desc = run_train.apply_transplant(train, ds, "mvtec", cfg)
    tr_def = {r[0] for r in train if r[1] == 1}; tr_nrm = {r[0] for r in train if r[1] == 0}
    syn = [r for r in recs if len(r) > 2 and r[2] == "transplant"]
    donors = [op.donors[r[3]][0] if len(r) > 3 else "random-per-load" for r in syn]; hosts = [r[0] for r in syn]
    val, test, cal = ds.records("val"), ds.records("test"), (ds.records("calib") or [])
    vt = {r[0] for r in val} | {r[0] for r in test} | {r[0] for r in cal}
    rows = [[c, s, "synthetic", h, 1, d] for h, d in zip(hosts, donors)]
    for kind, rr in (("train_real", train), ("val", val), ("test", test), ("calib", cal)):
        rows += [[c, s, kind, r[0], r[1], ""] for r in rr]
    ov = len(vt & (set(donors) | set(hosts) | tr_def | tr_nrm)) + len({r[0] for r in val} & {r[0] for r in test})
    ok = (set(donors) <= tr_def and set(hosts) <= tr_nrm and ov == 0 and len(syn) == desc.get("n_synthetic", 0)
          and all(len(r) < 3 or r[2] is None for r in list(val) + list(test) + list(cal)))
    line = (f"{c} s{s}: real training defects {len(tr_def)}, synthetic {len(syn)} from {len(set(donors))} donors "
            f"(all training defects: {set(donors) <= tr_def}) on {len(set(hosts))} hosts (all training normals: "
            f"{set(hosts) <= tr_nrm}); val {len(val)}, test {len(test)}, calib {len(cal)}; shared files {ov} -> {'OK' if ok else 'FAIL'}")
    return rows, ok, line


def leaklists(a):
    from utils.io import load_config, apply_overrides
    os.makedirs(a.out, exist_ok=True); summary = []; n_ok = 0
    with open(os.path.join(a.out, "leak_lists.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["category", "seed", "kind", "file", "label", "donor_file"])
        for c, seeds in sorted(FINAL_SEEDS.items()):
            for s in seeds:
                cfg = apply_overrides(load_config("configs/baseline.yaml"), FINAL_SET + [f"paths.mvtec_root={a.mvtec_root}"]); cfg["seed"] = s
                rows, ok, line = leak_one(cfg, c); w.writerows(rows); n_ok += ok; summary.append(line); print(line, flush=True)
    summary.append(f"TOTAL: {n_ok}/{len(summary)} category-seeds OK")
    open(os.path.join(a.out, "leak_summary.txt"), "w").write("\n".join(summary) + "\n"); print(summary[-1])


def exif(a):
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "visa_exif.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["category", "file", "file_index", "exif_datetime_306", "exif_datetime_original_36867"])
        for p in sorted(glob.glob(os.path.join(a.visa_root, "**", "Data", "Images", "Normal", "*.JPG"), recursive=True)):
            rel = p.split("visa-anomaly-detection/")[-1]; cat = rel.split("/")[0]
            try:
                im = Image.open(p); e = im.getexif(); t306 = e.get(306, ""); t367 = e.get_ifd(0x8769).get(36867, "")
            except Exception as ex:
                t306, t367 = f"ERROR {type(ex).__name__}", ""
            w.writerow([cat, rel, int(os.path.splitext(os.path.basename(p))[0]), t306, t367])


def copy(a):
    os.makedirs(a.out, exist_ok=True)
    for rel in [l.strip() for l in open(a.list) if l.strip()]:
        src = os.path.join(a.root, rel)
        if not os.path.exists(src): src = (glob.glob(os.path.join(a.root, "**", rel), recursive=True) or [src])[0]
        if os.path.exists(src): shutil.copy(src, os.path.join(a.out, rel.replace("/", "__")))
        else: print("missing", src)


def pixelstats(a):
    from analysis.session_stats import stats, STATS
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "mvtec_pixelstats.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["category", "group", "defect_type", "file"] + STATS)
        for p in sorted(glob.glob(os.path.join(a.mvtec_root, "**", "*", "*", "*.png"), recursive=True)):
            parts = p.split(os.sep)
            if "ground_truth" in parts or parts[-3] not in ("train", "test"): continue
            cat, split, typ = parts[-4], parts[-3], parts[-2]
            group = f"{split}_good" if typ == "good" else "defect"
            w.writerow([cat, group, typ, "/".join(parts[-4:])] + [round(float(x), 6) for x in stats(p)])


def samples(a):
    import random
    os.makedirs(a.out, exist_ok=True); random.seed(0)
    for root, pat, tag in ((a.mvtec_root, os.path.join("**", "test", "*", "*.png"), "mvtec"), (a.visa_root, os.path.join("**", "Data", "Images", "*", "*.JPG"), "visa")):
        files = sorted(glob.glob(os.path.join(root, pat), recursive=True)); pick = random.sample(files, min(a.n, len(files)))
        for i, p in enumerate(pick): shutil.copy(p, os.path.join(a.out, f"{tag}_{i:03d}{os.path.splitext(p)[1]}"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["leaklists", "exif", "copy", "pixelstats", "samples"])
    ap.add_argument("--mvtec_root", default=""); ap.add_argument("--visa_root", default=""); ap.add_argument("--root", default="")
    ap.add_argument("--list", default=""); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(); {"leaklists": leaklists, "exif": exif, "copy": copy, "pixelstats": pixelstats, "samples": samples}[a.cmd](a)


if __name__ == "__main__":
    main()
