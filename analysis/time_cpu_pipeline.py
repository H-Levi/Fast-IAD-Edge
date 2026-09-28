#!/usr/bin/env python
"""REQUESTS 4: full-pipeline latency on a CPU (no GPU), batch 1, fp32, on real photos (evidence_export samples).
Per image: decode (PIL open + RGB) -> resize to 288 + normalise (the eval transform of configs/baseline.yaml) -> model.
The final model scores TWO views (identity + rotate +10 deg, the mean of the two logits), so model time is reported for
1 view and for the 2-view pipeline actually used. Random weights (cost does not depend on the weight values).
    python analysis/time_cpu_pipeline.py --photos <dir with mvtec_*.png / visa_*.JPG> [--threads N]"""
import argparse, glob, os, platform, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from utils.io import load_config, apply_overrides
from utils.transforms import build_transforms
from models.model_factory import build_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--photos", required=True); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1); a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    torch.backends.nnpack.set_flags(False)   # NNPACK unsupported on this CPU: it only printed a warning per call
    cfg = apply_overrides(load_config("configs/baseline.yaml"), ["model.pretrained=false", "data.image_size=288"])
    tf, _ = build_transforms(cfg, train=False); model = build_model(cfg, "efficientnet").eval()
    files = sorted(glob.glob(os.path.join(a.photos, "*")))
    with torch.inference_mode():
        for _ in range(10): model(torch.zeros(1, 3, 288, 288))                         # warm-up
        T = {k: [] for k in ("decode", "preprocess", "model_1view", "model_2view", "total_2view")}
        for _ in range(a.repeats):
            for i, f in enumerate(files):
                if i % 20 == 0: print(f"  photo {i}/{len(files)}", flush=True)
                t0 = time.perf_counter(); im = Image.open(f).convert("RGB"); im.load()
                t1 = time.perf_counter(); x = tf(im).unsqueeze(0)
                t2 = time.perf_counter(); model(x)
                t3 = time.perf_counter(); 0.5 * (model(x) + model(TF.rotate(x, 10.0)))
                t4 = time.perf_counter()
                for k, v in (("decode", t1 - t0), ("preprocess", t2 - t1), ("model_1view", t3 - t2), ("model_2view", t4 - t3), ("total_2view", (t2 - t0) + (t4 - t3))):
                    T[k].append(v * 1000)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    print(f"CPU: {platform.processor() or platform.machine()} | torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"{len(files)} photos x {a.repeats} | batch 1, fp32, 288 px, EfficientNet-B0 + deep head")
    for k, v in T.items(): print(f"  {k:12s} median {statistics.median(v):8.1f} ms   p90 {q(v, 0.9):8.1f} ms")
    for tag in ("mvtec", "visa"):
        idx = [i for i, f in enumerate(files * a.repeats) if os.path.basename(f).startswith(tag)]
        if idx: print(f"  total_2view {tag}: median {statistics.median([T['total_2view'][i] for i in idx]):.1f} ms (decode {statistics.median([T['decode'][i] for i in idx]):.1f})")
    try: print("  cpu model:", next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")))
    except Exception: pass


if __name__ == "__main__":
    main()
