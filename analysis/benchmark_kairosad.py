#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_kairosad.py — time the REFERENCE detector on OUR hardware.

WHY
---
KairosAD reports "5 ms" with no batch size and no device named. Its RTX 4090 is
stated for training only. So the single number we are measured against cannot be
compared to anything -- and our own resolution sweep proved why naive fixes fail:
at batch 1 on a T4, pushing 5.8x more pixels through the network did not change
latency at all, because the time is spent on launch overhead rather than
arithmetic. Scaling a published number by a TFLOPS ratio is therefore fiction.

The fix is not to reproduce their hardware. It is to run THEIR model on OURS.
Same GPU, same batch size, same precision, same timing code, same warmup. Then
the comparison is a measurement instead of an argument.

WEIGHTS ARE NOT NEEDED. Latency and parameter count depend on architecture, not
on the values in the tensors -- which is why our own benchmark has always run on
random weights. So there is no checkpoint to hunt for and no dataset to download.

WHAT IT REPORTS
---------------
The same fields as analysis/benchmark.py, so the two are directly comparable:
params, size, FLOPs, bs=1 latency, batched throughput, peak memory.

It also reports the INPUT RESOLUTION the model actually wants. MobileSAM's image
encoder is built for 1024x1024; if KairosAD feeds it that, its cost is being
compared against our 224-288px models on very different terms, and that belongs
in the paper.

USAGE
-----
    git clone --recurse-submodules https://github.com/intelligolabs/KairosAD
    pip install -e KairosAD/MobileSAM
    python analysis/benchmark_kairosad.py --kairos_root KairosAD --image_size 1024
    python analysis/benchmark_kairosad.py --kairos_root KairosAD --image_size 224

Run it at both sizes: 1024 is what MobileSAM is designed for, 224 puts it on our
axis. Report both and say which is which.
"""

import argparse
import contextlib
import importlib
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.benchmark import benchmark_model, count_flops   # noqa: E402


_OUR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def _their_models_package(kairos_root):
    """Temporarily make `models.*` resolve to KairosAD's package, then undo it.

    Both repos have a top-level `models/`, and putting theirs first on sys.path
    is NOT enough. Python's rule: while scanning sys.path for `models`, a
    directory WITHOUT __init__.py is only remembered as a namespace-package
    portion and the scan CONTINUES; a directory WITH __init__.py is a regular
    package and wins immediately. KairosAD/models has no __init__.py, ours does
    -- so ours won from further down the path even though theirs came first, and
    `models.msam` then genuinely did not exist. The error said
    "No module named 'models.msam'", which reads like a missing file in their
    repo rather than a package-resolution collision in ours, and that sent the
    whole diagnosis the wrong way.

    So our root has to LEAVE sys.path for the duration, not merely be outranked.
    Entries are resolved to real paths first because '' means the cwd, which on
    Kaggle is our repo root -- filtering the literal string would have missed it.

    sys.modules is restored afterwards: head_to_head builds both models in one
    process, and without the restore our factory silently resolves to theirs.
    """
    saved_mods = {m: sys.modules[m] for m in list(sys.modules)
                  if m == "models" or m.startswith("models.")}
    for m in saved_mods:
        del sys.modules[m]
    saved_path = list(sys.path)
    kept = [p for p in sys.path
            if os.path.realpath(p or os.getcwd()) != os.path.realpath(_OUR_ROOT)]
    sys.path[:] = [kairos_root] + kept
    try:
        yield
    finally:
        sys.path[:] = saved_path
        theirs = {m: sys.modules[m] for m in list(sys.modules)
                  if m.startswith("models.") and m not in saved_mods}
        for m in [m for m in list(sys.modules)
                  if m == "models" or m.startswith("models.")]:
            del sys.modules[m]
        sys.modules.update(saved_mods)
        # F-L141: torch.compile traces LAZILY (first call, after this block) and re-imports KairosAD's classes by module
        # name ('models.msam'); with their sub-modules gone it failed "No module named 'models.msam'". Keep THEIR
        # sub-modules registered under their names; 'models' itself and every name of ours stay ours.
        sys.modules.update(theirs)


_SKIP_DIRS = {".git", "__pycache__", ".github", "weights", "checkpoints",
              "notebooks", "assets", "images", "docs", ".ipynb_checkpoints"}


def _py_tree(root, limit=80):
    """Every .py file under root, relative. Printed when discovery fails, so one
    more run tells us the real layout instead of costing another guess."""
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in _SKIP_DIRS]
        for f in sorted(files):
            if f.endswith(".py"):
                out.append("  " + os.path.relpath(os.path.join(d, f), root))
    if not out:
        return "  (none — the clone is empty or the path is wrong)"
    extra = len(out) - limit
    return "\n".join(out[:limit]) + (f"\n  ... +{extra} more" if extra > 0 else "")


def _find_class(root, class_name):
    """Locate `class <class_name>` by scanning sources; return its dotted module.

    The first version of this file hard-coded `models.msam` / `models.kairos_ad`.
    That was a guess about someone else's directory layout, it was wrong, and it
    cost a Kaggle run -- worse, it failed with "No module named 'models.msam'",
    which reads like a missing submodule rather than a wrong path. Scanning for
    the class definition cannot be wrong about a layout it reads off disk.
    os.walk is top-down, so shallower matches win.
    """
    pat = re.compile(rf"^\s*class\s+{re.escape(class_name)}\b", re.M)
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in _SKIP_DIRS]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(d, f)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
            except OSError:
                continue
            if pat.search(src):
                rel = os.path.relpath(p, root)[:-3]
                return rel.replace(os.sep, "."), p
    return None, None


_RENAME_IMPORT = re.compile(r"^(\s*)(from|import)(\s+)models(?=[\s.])", re.M)


def rename_their_package(kairos_root):
    """F-L141 ROOT FIX: rename KairosAD's top-level `models/` to `kairos_models/` and rewrite their own imports
    (`from models.x` / `import models.x` -> `kairos_models`), so their package can never collide with ours — not while
    building, and not later when torch.compile traces lazily and re-imports by module name (which failed twice with
    "No module named 'models.msam'"). Idempotent; MobileSAM (its own package, pip-installed) is not touched.
    Returns the number of files rewritten."""
    src, dst = os.path.join(kairos_root, "models"), os.path.join(kairos_root, "kairos_models")
    if os.path.isdir(src) and not os.path.isdir(dst):
        os.rename(src, dst)
    n = 0
    for dp, dn, fn in os.walk(kairos_root):
        dn[:] = [d for d in dn if d not in (".git", "MobileSAM", "__pycache__")]
        for f in fn:
            if f.endswith(".py"):
                q = os.path.join(dp, f); t = open(q, encoding="utf-8", errors="replace").read()
                t2 = _RENAME_IMPORT.sub(r"\1\2\3kairos_models", t)
                if t2 != t:
                    open(q, "w", encoding="utf-8").write(t2); n += 1
    return n


def _import_their_classes(kairos_root, verbose=True):
    """Return (MSAM, KairosAD), trying the documented paths then discovery."""
    for mod_msam, mod_kad in [("kairos_models.msam", "kairos_models.kairos_ad"),
                              ("models.msam", "models.kairos_ad"),
                              ("model.msam", "model.kairos_ad"),
                              ("src.models.msam", "src.models.kairos_ad")]:
        try:
            M = importlib.import_module(mod_msam)
            K = importlib.import_module(mod_kad)
            return M.MSAM, K.KairosAD
        except Exception:
            continue

    found = {}
    for cn in ("MSAM", "KairosAD"):
        dotted, path = _find_class(kairos_root, cn)
        if dotted is None:
            raise ImportError(f"no file under the clone defines `class {cn}`")
        found[cn] = (dotted, path)
        if verbose:
            print(f"  . found class {cn} in {os.path.relpath(path, kairos_root)}"
                  f"  (module {dotted})")
    mods = {cn: importlib.import_module(d) for cn, (d, _) in found.items()}
    return getattr(mods["MSAM"], "MSAM"), getattr(mods["KairosAD"], "KairosAD")


def build_kairosad(kairos_root, device, num_of_layers=5, layer_divider=2,
                   sam_checkpoint=None, model_vit="vit_t"):
    """Instantiate KairosAD from a local clone. Returns (model, notes).

    Mirrors main.py:
        msam = MSAM(model_vit, sam_checkpoint, device)
        classifier = KairosAD(msam, num_of_layers, layer_divider)
    with the paper's defaults (num_of_layers=5, layer_divider=2).
    """
    kairos_root = os.path.abspath(kairos_root)
    if not os.path.isdir(kairos_root):
        raise FileNotFoundError(f"--kairos_root does not exist: {kairos_root}")

    from analysis.patch_mobilesam import patch as _patch_msam
    status, ppath = _patch_msam(kairos_root)
    notes = [f"MobileSAM apply_image patch: {status}"]
    if status in ("missing", "failed"):
        raise RuntimeError(
            f"KairosAD's required MobileSAM patch could not be applied ({status}: "
            f"{ppath}). Their README makes it mandatory; without it their forward "
            f"raises \"'numpy.ndarray' object has no attribute 'to'\".")

    if sam_checkpoint is None:
        default_ckpt = os.path.join(kairos_root, "MobileSAM", "weights", "mobile_sam.pt")
        sam_checkpoint = default_ckpt if os.path.exists(default_ckpt) else None
    if sam_checkpoint is None:
        notes.append("no mobile_sam.pt found -> random weights (fine: timing and "
                     "parameter count do not depend on weight values)")

    notes.append(f"KairosAD package renamed models -> kairos_models; {rename_their_package(kairos_root)} file(s) rewritten")
    if kairos_root not in sys.path:
        sys.path.append(kairos_root)   # kairos_models is a unique name: appending cannot shadow anything of ours
    with _their_models_package(kairos_root):
        try:
            MSAM, KairosAD = _import_their_classes(kairos_root)
        except Exception as e:
            raise ImportError(
                f"could not import KairosAD's classes from {kairos_root}: {e}\n"
                f"Python files present:\n{_py_tree(kairos_root)}")
        try:
            msam = MSAM(model_vit, sam_checkpoint, device)
        except Exception as e:
            raise RuntimeError(
                f"MSAM(...) failed: {e}. If it insists on a checkpoint, fetch "
                f"MobileSAM's weights into {kairos_root}/MobileSAM/weights/mobile_sam.pt")
        model = KairosAD(msam, num_of_layers, layer_divider).to(device)
    return model, notes


class _Summarised(torch.nn.Module):
    """Wrap a foreign model so benchmark_model's .summary() call works unchanged,
    and hand it the input layout it actually expects.

    LAYOUT. Everything of ours is NCHW -- [batch, channels, height, width], the
    PyTorch norm. KairosAD is not: its MSAM.transform calls the patched
    apply_image, which reads image.shape[0]/[1] as height/width and then does
    permute(2, 0, 1). That only makes sense for a single HWC image, and
    KairosAD.forward maps it over the batch. So they want NHWC. Feeding NCHW
    would not error -- it would silently read channels as height and time a
    3-pixel-tall image. A benchmark that quietly measures the wrong shape is
    worse than one that crashes.

    RESOLUTION. Note what their pipeline does with whatever you pass: apply_image
    resizes the longest side to the encoder's img_size (1024) and preprocess pads
    to 1024x1024. The encoder therefore always runs at 1024 regardless of input
    size -- which is why their latency should come out flat across the sweep, and
    why they cannot trade resolution for speed the way we can.
    """

    def __init__(self, m, name, layout="NCHW"):
        super().__init__()
        self.m, self.name, self.layout = m, name, layout

    def forward(self, x):
        if self.layout == "NHWC":
            x = x.permute(0, 2, 3, 1).contiguous()
        return self.m(x)

    def summary(self):
        total = sum(p.numel() for p in self.m.parameters())
        return {"backbone": self.name, "head": "-", "feat_dim": None,
                "total_params": total,
                "trainable_params": sum(p.numel() for p in self.m.parameters()
                                        if p.requires_grad),
                "size_mb": total * 4 / 1024 / 1024}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kairos_root", required=True, help="path to the KairosAD clone")
    ap.add_argument("--image_size", type=int, default=1024,
                    help="1024 is MobileSAM's native input; also run 224 to place it "
                         "on the same axis as our models")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_of_layers", type=int, default=5)
    ap.add_argument("--layer_divider", type=int, default=2)
    ap.add_argument("--sam_checkpoint", default=None)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--gpu_idx", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--graphs", action="store_true",
                    help="ALSO time it under torch.compile(mode='reduce-overhead'), i.e. "
                         "CUDA graphs, and report the max output difference vs eager. Our "
                         "models get 3.6-4.8x from this; comparing our graphed time with "
                         "their eager time would favour us unfairly.")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu_idx}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if a.precision == "fp16" else torch.float32
    print(f"Device: {device}\nModel: KairosAD (MobileSAM vit_t, "
          f"num_of_layers={a.num_of_layers}, layer_divider={a.layer_divider})\n"
          f"Precision: {a.precision}  Image: {a.image_size}px\n")

    try:
        raw, notes = build_kairosad(a.kairos_root, device, a.num_of_layers,
                                    a.layer_divider, a.sam_checkpoint)
    except Exception as e:
        sys.exit(str(e))
    for n in notes:
        print(f"  ! {n}")
    model = _Summarised(raw, "kairosad", layout="NHWC").to(device)

    r1 = benchmark_model(model, device, a.image_size, 1,
                         iters=max(a.iters, 30), dtype=dtype)
    rb = benchmark_model(model, device, a.image_size, a.batch_size,
                         iters=a.iters, dtype=dtype)
    graphs = None
    if a.graphs and device.type == "cuda":
        # Same treatment our models get in the checkpoint's variant sweep, so the
        # speed ratio compares like with like.
        try:
            x = torch.randn(1, 3, a.image_size, a.image_size, device=device, dtype=dtype)
            with torch.no_grad():
                ref = model(x).float().cpu()
            gm = torch.compile(model, mode="reduce-overhead")
            rg = benchmark_model(gm, device, a.image_size, 1,
                                 iters=max(a.iters, 30), dtype=dtype)
            with torch.no_grad():
                diff = float((gm(x).float().cpu() - ref).abs().max())
            graphs = {"ms_per_image_bs1": rg["ms_per_image"],
                      "std": rg["ms_per_batch_std"],
                      "speedup_vs_eager": r1["ms_per_image"] / rg["ms_per_image"],
                      "max_abs_output_diff": diff}
        except Exception as e:
            graphs = {"error": f"{type(e).__name__}: {e}"}
    try:
        flops = count_flops(model.float(), device, a.image_size)
    except Exception as e:
        flops = f"n/a ({type(e).__name__})"

    print("\n" + "-" * 66)
    print(f"  params            {r1['params']:,}")
    print(f"  size              {r1['size_mb']:.2f} MB")
    print(f"  FLOPs (bs=1)      {flops}")
    print("  -- single image (batch_size=1; the deployment latency) --")
    print(f"  ms / image @bs1   {r1['ms_per_image']:.3f} ± {r1['ms_per_batch_std']:.3f}")
    print(f"  throughput @bs1   {r1['throughput_img_s']:.1f} img/s")
    if graphs is not None:
        if "error" in graphs:
            print(f"  CUDA graphs       FAILED: {graphs['error'][:120]}")
        else:
            print(f"  -- CUDA graphs (compile mode=reduce-overhead), batch_size=1 --")
            print(f"  ms / image @bs1   {graphs['ms_per_image_bs1']:.3f} ± {graphs['std']:.3f}"
                  f"   ({graphs['speedup_vs_eager']:.2f}x vs eager, "
                  f"max output diff {graphs['max_abs_output_diff']:.1e})")
    print(f"  -- batched (batch_size={a.batch_size}) --")
    print(f"  ms / image        {rb['ms_per_image']:.3f}")
    print(f"  peak GPU memory   {rb['peak_mem_mb']:.1f} MB")
    print("-" * 66)
    print("Compare against analysis/benchmark.py run on the SAME device, batch size,\n"
          "precision and image size. Only then is the difference the models'.")
    print(f"\nPublished KairosAD figures for reference: 11.53M params, 5 ms "
          f"(batch size NOT stated, device NOT named), Jetson NX 211 ms, AGX 218 ms.")


if __name__ == "__main__":
    main()
