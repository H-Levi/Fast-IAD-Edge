#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark.py — inference speed/memory/size for trained checkpoints.

Your original benchmark, cleaned and pointed at the new model interface. It
loads a trained checkpoint per category and reports per-image latency,
throughput, peak memory, params, and model size, then averages. This is the
"fast" half of the efficiency story; AUROC is the "without losing accuracy"
half (from run_train / compare_backbones).

Speed note: real latency wins (fp16, channels_last, TorchScript/ONNX) are not
applied here yet — this measures the plain eager model so comparisons are
apples-to-apples. We add those as flags once a backbone is chosen.

Run example
-----------
python analysis/benchmark.py --checkpoint_dir runs/<run>/ \
    --backbone efficientnet --head deep --image_size 224 --batch_size 32
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import AnomalyClassifier


@torch.no_grad()
def benchmark_model(model, device, image_size, batch_size, warmup=10, iters=100,
                    dtype=torch.float32, channels_last=False):
    """Time a forward pass. `dtype` must match between the model and the input.

    Precision is a benchmark variable, not a detail: published latencies are often
    FP16 (EfficientAD's 2.2 ms is: its Appendix E, 'float16 precision for all networks'), and comparing our FP32 number against an FP16
    one is two differences at once -- hardware AND arithmetic width. Controlling it
    is free, so there is no reason not to.
    """
    model = model.to(dtype).eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
    if channels_last:   # F-L143: the input must share the model's memory layout, or the layout change is paid per call
        x = x.contiguous(memory_format=torch.channels_last)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        t0 = time.time()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    times = np.array(times)
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
               if device.type == "cuda" else float("nan"))
    s = model.summary()
    return {
        "batch_size": batch_size,
        "ms_per_batch": float(times.mean() * 1000),
        "ms_per_batch_std": float(times.std() * 1000),
        "ms_per_image": float(times.mean() / batch_size * 1000),
        "throughput_img_s": float(batch_size / times.mean()),
        "peak_mem_mb": peak_mb,
        "params": s["total_params"],
        "size_mb": s["size_mb"],
    }


@torch.no_grad()
def count_flops(model, device, image_size):
    """Hardware-neutral compute at batch_size=1, in GMACs. Needs fvcore or thop.

    UNIT WARNING, and the reason this returns GMACs and not "GFLOPs". A
    multiply-accumulate is two floating-point operations, so FLOPs = 2 x MACs.
    fvcore counts MACs; thop also counts MACs. Most of the vision literature
    nevertheless prints MACs under the label "FLOPs" -- EfficientNet-B0's famous
    "0.39B FLOPs" at 224px is 0.39 GMACs, i.e. 0.78 GFLOPs by the strict
    definition. The earlier version of this function returned fvcore's MACs from
    one branch and thop's 2*MACs from the other, so the same column meant
    different things depending on which library happened to be installed -- a 2x
    discrepancy nobody would have noticed. Both branches now report GMACs, and
    the label says GMACs, so whichever convention the write-up adopts it is
    converting from a number whose meaning is stated.
    """
    import logging
    x = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        # fvcore warns about every elementwise op it does not count (silu, mul,
        # sigmoid...). Those carry no MACs, so the warnings are noise -- and at
        # ~14 lines per call they bury the actual results.
        logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return f"{fca.total()/1e9:.3f} GMACs"
    except Exception:
        pass
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return f"{macs/1e9:.3f} GMACs"
    except Exception:
        return "n/a (pip install fvcore)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to a single .pt checkpoint (optional)")
    ap.add_argument("--backbone", default="efficientnet",
                    choices=["efficientnet", "mobilenet", "squeezenet"])
    ap.add_argument("--head", default="deep", choices=["deep", "mlp", "linear"])
    ap.add_argument("--pooling", default="avg",
                    choices=["avg", "max", "avgmax", "attention", "gem", "lse"],
                    help="spatial pooling (EXPERIMENTS.md E1). 'avgmax' doubles the "
                         "head's input width; 'attention' adds a 1x1 conv. Both cost "
                         "parameters — that is what this flag is here to measure.")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu_idx", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"],
                    help="arithmetic width. Published latencies are frequently FP16 "
                         "(EfficientAD 2.2 ms is); comparing our FP32 against their FP16 "
                         "is two differences at once. Free to control, so control it.")
    args = ap.parse_args()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nBackbone: {args.backbone}  Head: {args.head}  "
          f"Pooling: {args.pooling}  Precision: {args.precision}  "
          f"Image: {args.image_size}px\n")

    model = AnomalyClassifier(args.backbone, args.head, pretrained=False,
                              pooling=args.pooling).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded weights: {args.checkpoint}")
    else:
        print("No checkpoint given -> benchmarking architecture with random weights "
              "(latency/size are weight-independent).")

    # §1.7: single-image latency (the embedded-deployment number) reported
    # ALONGSIDE batched throughput. Report both, plus params + FLOPs (hardware-neutral).
    r_bs1 = benchmark_model(model, device, args.image_size, 1,
                            iters=max(args.iters, 50), dtype=dtype)
    r_batch = benchmark_model(model, device, args.image_size, args.batch_size,
                              iters=args.iters, dtype=dtype)
    flops = count_flops(model.float(), device, args.image_size)   # FLOPs counters want fp32

    print("\n" + "-" * 60)
    print(f"  params            {r_bs1['params']:,}")
    print(f"  size              {r_bs1['size_mb']:.2f} MB")
    print(f"  FLOPs (bs=1)      {flops}")
    print("  -- single image (batch_size=1; the deployment latency) --")
    print(f"  ms / image @bs1   {r_bs1['ms_per_image']:.3f} ± {r_bs1['ms_per_batch_std']:.3f}")
    print(f"  throughput @bs1   {r_bs1['throughput_img_s']:.1f} img/s")
    print(f"  -- batched (batch_size={args.batch_size}; throughput ceiling) --")
    print(f"  ms / image        {r_batch['ms_per_image']:.3f}")
    print(f"  throughput        {r_batch['throughput_img_s']:.1f} img/s")
    print(f"  peak GPU memory   {r_batch['peak_mem_mb']:.1f} MB")
    print("-" * 60)
    print("NOTE: cite ms/image @bs1 for embedded claims; measure on the device you cite.")


if __name__ == "__main__":
    main()
