#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
head_to_head.py — ONE run that closes the throughput question.

THE QUESTION
------------
Did KairosAD build something so efficient that we cannot beat its throughput?
That is not answerable from their paper: their "5 ms" names no batch size and no
inference device. It is also not answerable by arguing about hardware. It is
answerable by running BOTH models in the SAME process, on the SAME GPU, at the
SAME resolution, at the SAME precision, through the SAME timing code -- and then
reading the numbers.

THE TRAP THIS AVOIDS
--------------------
Our own sweep found our EfficientNet-B0 flat at 7.4-8.0 ms from 160px to 384px:
5.8x the pixels, no change in time. That is a property of a SMALL CNN at batch 1
on a T4, where per-kernel launch overhead dominates the arithmetic. It is NOT a
law. A ViT at 1024px is far past that crossover and its cost DOES scale. So the
comparison must be made at MATCHED resolution or it means nothing, and the
resolution sweep must cover enough range to find where each model stops being
overhead-bound. That is why the grid goes up to 1024 for BOTH sides.

WHAT THE CELLS MEAN
-------------------
Every (model, resolution, precision) cell is attempted independently and its
failure is recorded rather than aborting the run -- because a failure is data:

  * KairosAD FAILS below 1024  -> their architecture is LOCKED to 1024. The
    1024 cost is structural, not an incidental choice to be discounted.
  * KairosAD RUNS at 288       -> a clean matched-resolution comparison. No
    excuses available to either side.
  * ours flat 224->1024        -> resolution is free for us at batch 1, and any
    accuracy resolution buys is free too.
  * ours rises after some N    -> that N is our overhead/compute crossover, and
    it is the honest ceiling on "resolution is free".

No weights, no dataset, no training. Latency, parameters and MACs depend on
architecture, not on the values inside the tensors.

USAGE (one command)
-------------------
    python analysis/head_to_head.py --kairos_root KairosAD --out results/h2h

Results are written to <out>.csv and <out>.md AS EACH CELL FINISHES, so a
disconnect costs you only the cell in flight.
"""

import argparse
import csv
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.benchmark import benchmark_model, count_flops          # noqa: E402
from analysis.benchmark_kairosad import build_kairosad, _Summarised  # noqa: E402

FIELDS = ["model", "image_size", "precision", "status", "params", "size_mb",
          "gmacs", "ms_bs1", "ms_bs1_std", "img_per_s", "peak_mem_mb", "note"]


def _flush(rows, out_base):
    """Write CSV + Markdown after every cell. A Kaggle disconnect must not cost
    the whole grid -- partial results are still results."""
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    with open(out_base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_base + ".md", "w") as f:
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("|" + "|".join(["---"] * len(FIELDS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in FIELDS) + " |\n")


def _cell(build_fn, name, size, precision, device, iters, want_flops):
    """Measure one (model, resolution, precision) cell. Never raises."""
    row = {"model": name, "image_size": size, "precision": precision,
           "status": "ok", "note": ""}
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = None
    try:
        model = build_fn().to(device)
        # F-L143: "+full" = torch.compile(mode="reduce-overhead", i.e. CUDA graphs) + channels-last (not KairosAD, whose
        # wrapper feeds NHWC itself); "[x2]" = our two views as ONE batch of 2 (the deployed two-view scoring: the time
        # reported is per image, both views).
        cl = "+full" in name and "kairosad" not in name
        bs = 2 if name.endswith("[x2]") else 1
        r = benchmark_model(model, device, size, bs, warmup=10, iters=iters, dtype=dtype, channels_last=cl)
        if bs == 2:
            row["note"] = "batch 2 = one image, two views (time per image)"
        row.update({
            "params": r["params"],
            "size_mb": round(r["size_mb"], 2),
            # [x2] cells: the batch of 2 IS one image (both views), so its time is the BATCH time. (Before this fix the
            # per-image field halved it — F-L143's data is corrected in v119 by x2; the raw batch timing is unaffected.)
            "ms_bs1": round(r["ms_per_batch"] if bs == 2 else r["ms_per_image"], 3),
            "ms_bs1_std": round(r["ms_per_batch_std"], 3),
            "img_per_s": round(r["throughput_img_s"] / bs, 1),   # [x2]: one IMAGE = both views = the whole batch
            "peak_mem_mb": round(r["peak_mem_mb"], 1),
        })
        # MACs are a property of the architecture, so a compiled variant has the
        # same count as its eager twin -- and tracing a compiled module tends to
        # fail anyway. Read its MACs off the eager row.
        if want_flops and precision == "fp32" and "+compile" not in name and "+full" not in name and bs == 1:
            g = count_flops(model.float(), device, size)
            row["gmacs"] = g.replace(" GMACs", "") if "GMACs" in g else g
    except Exception as e:
        # A crash here is a FINDING, not an accident -- most often a fixed-size
        # position embedding refusing a resolution it was not built for.
        row["status"] = "FAILED"
        kind = type(e).__name__
        msg = " ".join(str(e).split())[:160]
        row["note"] = f"{kind}: {msg}"
        if "out of memory" in msg.lower():
            row["status"] = "OOM"
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kairos_root", default="KairosAD",
                    help="path to the KairosAD clone; omit/miss it and only our "
                         "models are measured (the run still produces something)")
    ap.add_argument("--sizes", default="224,288,384,512,1024",
                    help="resolutions to sweep, for BOTH sides")
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--ours", default="efficientnet:deep:lse,mobilenet:deep:lse",
                    help="backbone:head:pooling triples")
    ap.add_argument("--out", default="results/head_to_head")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no_flops", action="store_true")
    ap.add_argument("--efficientad_root", default="",
                    help="F-L142: path to a clone of github.com/nelson1425/EfficientAD; adds efficientad_s (256 px only)")
    ap.add_argument("--full", action="store_true",
                    help="F-L143: add '+full' cells: torch.compile(mode='reduce-overhead') (CUDA graphs) + channels-last")
    ap.add_argument("--twoview", action="store_true",
                    help="F-L143: with --full, also time OUR model on a batch of 2 (both views at once, as deployed)")
    ap.add_argument("--only", default="",
                    help="comma-separated model names to run (e.g. kairosad+compile) — re-run only the missing cells")
    ap.add_argument("--compile", action="store_true",
                    help="also measure torch.compile'd variants of OUR models AND of KairosAD. The "
                         "sweep says we are overhead-bound (latency flat while MACs "
                         "rise 5-20x), so latency is set by HOW MANY kernels launch, "
                         "not how big they are. Fusing ops is therefore the lever that "
                         "should move it -- shrinking the model is the lever that "
                         "cannot. This flag tests that directly. Compilation costs "
                         "~30-60s per cell on top of the timing.")
    a = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    precisions = [p.strip() for p in a.precisions.split(",") if p.strip()]
    if device.type != "cuda" and "fp16" in precisions:
        precisions = [p for p in precisions if p != "fp16"]
        print("! CPU device -> dropping fp16 (half precision on CPU is not "
              "representative of anything you would deploy)")

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"torch:  {torch.__version__}")
    print(f"grid:   sizes={sizes}  precisions={precisions}  batch_size=1  "
          f"iters={a.iters}")
    print("note:   random weights throughout -- latency, params and FLOPs do not "
          "depend on weight values.\n")

    # ---- assemble the builders -------------------------------------------
    from models.model_factory import AnomalyClassifier
    builders = []
    for spec in [s for s in a.ours.split(",") if s.strip()]:
        bb, head, pool = (spec.split(":") + ["deep", "avg"])[:3]
        builders.append((f"ours:{bb}/{pool}",
                         lambda bb=bb, head=head, pool=pool: AnomalyClassifier(
                             bb, head, pretrained=False, pooling=pool)))
        if a.compile:
            builders.append((f"ours:{bb}/{pool}+compile",
                             lambda bb=bb, head=head, pool=pool: torch.compile(
                                 AnomalyClassifier(bb, head, pretrained=False,
                                                   pooling=pool))))

    kairos_ok, kairos_err = False, None
    if a.kairos_root and os.path.isdir(a.kairos_root):
        try:
            probe, notes = build_kairosad(a.kairos_root, device)
            for n in notes:
                print(f"  ! {n}")
            del probe
            if device.type == "cuda":
                torch.cuda.empty_cache()
            kairos_ok = True
        except Exception as e:
            kairos_err = " ".join(str(e).split())[:300]
            print(f"  ! KairosAD could not be built at all: {kairos_err}")
            print("    -> continuing with our models only; fix the clone and re-run.")
    else:
        kairos_err = f"--kairos_root '{a.kairos_root}' is not a directory"
        print(f"  ! {kairos_err}\n    -> continuing with our models only.")

    if kairos_ok:
        builders.append(("kairosad",
                         lambda: _Summarised(
                             build_kairosad(a.kairos_root, device)[0], "kairosad",
                             layout="NHWC")))
        if a.compile:   # F-L141 (2026-09-28): the like-for-like OPTIMISED comparison RESULTS_benchmark section 6 left open
            builders.append(("kairosad+compile",
                             lambda: _Summarised(
                                 torch.compile(build_kairosad(a.kairos_root, device)[0]), "kairosad",
                                 layout="NHWC")))

    # ---- run the grid ----------------------------------------------------
    if a.efficientad_root:
        from analysis.benchmark_efficientad import EfficientADS
        builders.append(("efficientad_s", lambda: EfficientADS(a.efficientad_root)))
        if a.compile:
            builders.append(("efficientad_s+compile", lambda: torch.compile(EfficientADS(a.efficientad_root))))
    if a.full:   # F-L143: every base model also at FULL optimisation
        base = [(n, f) for n, f in builders if "+compile" not in n]
        for n, f in base:
            cl = "kairosad" not in n
            builders.append((n + "+full", lambda f=f, cl=cl: torch.compile(
                f().to(memory_format=torch.channels_last) if cl else f(), mode="reduce-overhead")))
            if n.startswith("ours:") and a.twoview:
                builders.append((n + "+full[x2]", lambda f=f: torch.compile(
                    f().to(memory_format=torch.channels_last), mode="reduce-overhead")))
    if a.only:
        keep = {x.strip() for x in a.only.split(",") if x.strip()}
        builders = [(n, f) for n, f in builders if n in keep]
        print(f"--only: running {[n for n, _ in builders]}")
    rows, total = [], len(builders) * len(sizes) * len(precisions)
    i = 0
    for name, fn in builders:
        for size in sizes:
            for prec in precisions:
                i += 1
                print(f"[{i}/{total}] {name}  {size}px  {prec} ... ", end="", flush=True)
                row = _cell(fn, name, size, prec, device, a.iters, not a.no_flops)
                rows.append(row)
                if row["status"] == "ok":
                    print(f"{row['ms_bs1']} ms  ({row['img_per_s']} img/s)")
                else:
                    print(f"{row['status']} — {row['note'][:90]}")
                _flush(rows, a.out)

    if kairos_err:
        rows.append({"model": "kairosad", "image_size": "", "precision": "",
                     "status": "NOT BUILT", "note": kairos_err})
        _flush(rows, a.out)

    # ---- read the grid out loud ------------------------------------------
    print("\n" + "=" * 78)
    print("HEAD TO HEAD  (batch size 1 — the deployment latency)")
    print("=" * 78)
    hdr = f"{'model':<26}{'px':>6}{'prec':>6}{'ms':>10}{'img/s':>10}{'GMACs':>10}  status"
    print(hdr)
    for r in rows:
        print(f"{r['model']:<26}{str(r.get('image_size','')):>6}"
              f"{str(r.get('precision','')):>6}{str(r.get('ms_bs1','-')):>10}"
              f"{str(r.get('img_per_s','-')):>10}{str(r.get('gmacs','-')):>10}  "
              f"{r['status']}"
              + (f"  {r['note'][:60]}" if r.get("note") else ""))

    ok = [r for r in rows if r["status"] == "ok"]
    print("-" * 78)

    # matched-resolution verdicts: the only comparisons that mean anything
    kai = {(r["image_size"], r["precision"]): r for r in ok if r["model"] == "kairosad"}
    if kai:
        print("MATCHED-RESOLUTION COMPARISON (same GPU, same precision, same code):")
        for r in ok:
            if r["model"] == "kairosad":
                continue
            k = kai.get((r["image_size"], r["precision"]))
            if k and r["ms_bs1"] and k["ms_bs1"]:
                ratio = k["ms_bs1"] / r["ms_bs1"]
                print(f"  {r['image_size']}px {r['precision']}: {r['model']} "
                      f"{r['ms_bs1']} ms  vs  kairosad {k['ms_bs1']} ms "
                      f"-> {ratio:.2f}x")
        sizes_ok = sorted({r["image_size"] for r in ok if r["model"] == "kairosad"})
        failed = sorted({r["image_size"] for r in rows
                         if r["model"] == "kairosad" and r["status"] == "FAILED"})
        if failed:
            print(f"\n  KairosAD RAN at {sizes_ok} and FAILED at {failed}.")
            print("  A model that cannot accept a smaller input cannot trade resolution")
            print("  for speed. That constraint is architectural and belongs in the paper.")
        else:
            print(f"\n  KairosAD ran at every resolution tried ({sizes_ok}) — so the")
            print("  comparison is fully matched and neither side has a resolution excuse.")
    else:
        print("KairosAD produced no usable cell — the comparison is INCOMPLETE.")
        print("Do not write a throughput claim against it from this run.")

    # our own overhead/compute crossover
    print("\nOUR OVERHEAD -> COMPUTE CROSSOVER (is resolution actually free?):")
    for name in sorted({r["model"] for r in ok if r["model"].startswith("ours")}):
        for prec in precisions:
            pts = sorted([r for r in ok if r["model"] == name and r["precision"] == prec],
                         key=lambda r: r["image_size"])
            if len(pts) < 2:
                continue
            base = pts[0]
            trail = "  ".join(f"{p['image_size']}px:{p['ms_bs1']}ms" for p in pts)
            print(f"  {name} {prec}: {trail}")
            knee = next((p for p in pts[1:]
                         if p["ms_bs1"] > base["ms_bs1"] * 1.15), None)
            if knee:
                print(f"      -> flat until {knee['image_size']}px, where cost starts "
                      f"to rise. Free resolution ends there.")
            else:
                print(f"      -> flat across the whole sweep: overhead-bound "
                      f"throughout, resolution is free at batch 1 on this GPU.")

    comp = {(r["model"].replace("+compile", ""), r["image_size"], r["precision"]): r
            for r in ok if "+compile" in r["model"]}
    if comp:
        print("\nKERNEL-FUSION TEST (torch.compile vs eager, same model, same cell):")
        for r in ok:
            if "+compile" in r["model"]:
                continue
            c = comp.get((r["model"], r["image_size"], r["precision"]))
            if c and r["ms_bs1"] and c["ms_bs1"]:
                d = 100.0 * (c["ms_bs1"] - r["ms_bs1"]) / r["ms_bs1"]
                print(f"  {r['model']:<24}{r['image_size']:>5}px {r['precision']}: "
                      f"eager {r['ms_bs1']:>7} ms -> compiled {c['ms_bs1']:>7} ms "
                      f"({d:+.1f}%)")
        print("  A large speedup confirms the latency is dispatch overhead, and that")
        print("  op COUNT -- not parameter count or MACs -- is what to attack next.")

    print("-" * 78)
    print(f"written: {a.out}.csv  and  {a.out}.md")
    print("Published KairosAD figures for reference: 11.53M params, 5 ms "
          "(batch size NOT stated, inference device NOT named),")
    print("Jetson Orin NX 211 ms, AGX 218 ms.")


if __name__ == "__main__":
    main()
