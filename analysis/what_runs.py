#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""what_runs.py — print what the config ACTUALLY does, not what it says.

WHY THIS EXISTS. configs/default.yaml has 264 lines across 15 top-level blocks.
Reading it tells you what the settings ARE; it does not tell you which ones the
code READS. Those came apart badly: `augmentation.enabled: false` sits under 45
lines of documentation describing scenarios, per-category physics rules and a
verification gate -- while four undocumented lines further down, `augment.enabled:
true`, are what the transform builder actually reads. Every training run in this
project applied flip + rotation + colour jitter while the config appeared to say
augmentation was off, and the per-category NO_HFLIP protection was never reached.

That is not a hard bug to find. It is a hard bug to find by READING, because
nothing in the config says which of two similarly-named blocks wins. So this
script does not read the config -- it builds the real objects through the real
code path and prints what came out.

RUN IT BEFORE ANY EXPERIMENT, and after any config change.

    python analysis/what_runs.py
    python analysis/what_runs.py data.de_confound_frac=0.5 augment.enabled=false
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def aug_live_any(cfg):
    return bool((cfg.get("augment") or {}).get("enabled")
                or (cfg.get("augmentation") or {}).get("enabled"))


def config_keys(cfg, prefix=""):
    """Every leaf key path in the config."""
    out = []
    for k, v in (cfg or {}).items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            out += config_keys(v, path)
        else:
            out.append(path)
    return out


def cross_check_config_against_code(cfg, root):
    """THE GENERAL VERSION OF THE AUGMENTATION BUG.

    That bug was one instance of a class: a config key that looks authoritative
    and is never read, sitting beside one that is read and is not documented.
    Checking for `augment` specifically would only catch it once. So instead:

      DEAD      a config key no source file mentions -> it cannot affect anything,
                yet it reads as a setting. `augmentation.enabled: false` was
                effectively this: read by nothing on the active path.
      SHADOWED  two blocks whose names are prefixes of each other, or which share
                leaf names -- `augment` vs `augmentation` -- where nothing states
                which wins.
      DEFAULTED a key the code reads with .get(..., default) that is ABSENT from
                the config, so the default silently governs and no one can see it
                by reading the file. `model.taps` is one.
    """
    import re
    srcs = {}
    for dirpath, dirnames, files in os.walk(root):
        # PRODUCTION CODE ONLY. The Tier 0 negative control exposed this: a key
        # mentioned in a test file (or a comment) counted as "read", so an injected
        # dead key went undetected. tests/ and this checker itself are excluded, and
        # comment lines are stripped below.
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "runs", ".ipynb_checkpoints",
                                    "tests")]
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(dirpath, f)
                if os.path.abspath(fp) == os.path.abspath(__file__):
                    continue
                try:
                    txt = open(fp, encoding="utf-8", errors="replace").read()
                    srcs[fp] = "\n".join(l for l in txt.splitlines()
                                          if not l.lstrip().startswith("#"))
                except Exception:
                    pass
    blob = "\n".join(srcs.values())

    leaves = config_keys(cfg)
    dead = []
    for path in leaves:
        leaf = path.split(".")[-1]
        if len(leaf) < 3:
            continue
        if not re.search(r'["\']' + re.escape(leaf) + r'["\']', blob):
            dead.append(path)

    tops = [k for k in (cfg or {}).keys()]
    shadowed = []
    for a in tops:
        for b in tops:
            if a != b and b.startswith(a) and len(b) > len(a):
                shadowed.append((a, b))

    # Keys read with a default that are ABSENT from the config. Must match only
    # CONFIG reads: a first version matched every .get() in the codebase and
    # returned 94 hits, nearly all of them dict lookups on result rows and JSON
    # payloads. A checker that cries wolf 94 times is not a checker.
    defaulted = set()
    pats = [
        r'cfg\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*,',
        r'cfg\[["\'][a-z_]+["\']\]\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*,',
        r'\(cfg\.get\(["\'][a-z_]+["\']\)\s*or\s*\{\}\)\.get\('
        r'\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*,',
    ]
    for pat in pats:
        for m in re.finditer(pat, blob):
            defaulted.add(m.group(1))
    # `present` must include INTERMEDIATE path components, not just leaves --
    # cfg.get("data", {}) reads a block that exists, and counting only leaf names
    # flagged every block in the file as missing.
    present = set()
    for pth in leaves:
        parts = pth.split(".")
        for i in range(len(parts)):
            present.add(parts[i])
    missing = sorted(k for k in defaulted if k not in present)
    return dead, shadowed, missing


def active_state(cfg):
    """Compact one-block summary of what is ACTUALLY active, for run_train to print
    and to write into the run directory. Same facts as the full report, short
    enough that nobody skips reading it."""
    from utils.transforms import build_transforms, build_scenario_transforms
    L = []
    n_tf, _, _, desc = build_scenario_transforms(cfg, "carpet")
    test_tf, _ = build_transforms(cfg, train=False)
    tr = [type(t).__name__ for t in n_tf.transforms]
    te = [type(t).__name__ for t in test_tf.transforms]
    live = [t for t in tr if t not in ("Resize", "ToTensor", "Normalize")]
    aug_new = (cfg.get("augmentation") or {}).get("enabled", False)
    aug_old = (cfg.get("augment") or {}).get("enabled", False)
    d, t_, m = cfg.get("data", {}), cfg.get("train", {}), cfg.get("model", {})
    L.append("ACTIVE STATE (built from the real code path, not read from config)")
    L.append(f"  train aug : {live or 'NONE'}"
             + (f"   [augment={aug_old} augmentation={aug_new}]"))
    if aug_old and not aug_new:
        try:
            from data.augmentation import NO_HFLIP
            L.append(f"  ! NO_HFLIP is UNREACHABLE: {len(NO_HFLIP)} categories will be "
                     f"flipped at p={cfg['augment'].get('horizontal_flip')}")
        except Exception:
            pass
        L.append("  ! stage.jsonl will record augmentation enabled=false ANYWAY")
    rnd = [x for x in te if x.startswith("Random") or "Jitter" in x]
    L.append(f"  val/test  : {te}" + ("   ** RANDOM OP IN EVAL -- LEAK **" if rnd else "   (clean)"))
    L.append(f"  data      : image_size={d.get('image_size')} "
             f"de_confound={d.get('de_confound_frac')} "
             f"anom_frac={d.get('anomaly_fraction')} "
             f"val_share={d.get('injection_val_share')}/{d.get('val_share_mode')} "
             f"cache={d.get('cache_decoded')}")
    pc = d.get("per_category") or {}
    if pc:
        L.append("  per-cat   : " + "; ".join(f"{k} {v}" for k, v in pc.items()))
    L.append(f"  train     : epochs={t_.get('num_epochs')} patience={t_.get('patience')} "
             f"lr={t_.get('learning_rate')} bs={t_.get('batch_size')} seed={cfg.get('seed')}")
    L.append(f"  model     : {m.get('pooling')}/{m.get('head')} taps={m.get('taps')} "
             f"frozen={m.get('freeze_backbone')}")
    return "\n".join(L)


def main():
    from utils.io import load_config, apply_overrides
    from utils.transforms import build_transforms, build_scenario_transforms
    args = sys.argv[1:]
    cfg_path = "configs/default.yaml"
    if args and args[0].endswith((".yaml", ".yml")):
        cfg_path, args = args[0], args[1:]
    cfg = apply_overrides(load_config(cfg_path), args) if args else load_config(cfg_path)

    print("=" * 74)
    print("WHAT ACTUALLY RUNS   " + cfg_path + ("  + " + " ".join(args) if args else ""))
    print("=" * 74)

    # ---- the transform each split really gets ----
    print("\nTRANSFORMS (built through the real code path, not read from config)")
    from data.mvtec import MVTecDataset
    from data.visa import ViSADataset
    cats = list(MVTecDataset.categories) + list(ViSADataset.categories)
    sigs = {}
    for c in cats:
        n_tf, a_tf, synth, desc = build_scenario_transforms(cfg, c)
        sigs[c] = [type(t).__name__ for t in n_tf.transforms]
    test_tf, _ = build_transforms(cfg, train=False)
    uniform = len(set(map(str, sigs.values()))) == 1
    print(f"  train  {sigs[cats[0]]}")
    if not uniform:
        for c, s in sigs.items():
            print(f"         {c:12s} {s}")
    else:
        print(f"         (identical across ALL {len(cats)} categories: "
              f"{len(MVTecDataset.categories)} MVTec + {len(ViSADataset.categories)} VisA)")
    print(f"  val    {[type(t).__name__ for t in test_tf.transforms]}")
    print(f"  test   {[type(t).__name__ for t in test_tf.transforms]}")
    # val and test MUST be the same object type sequence, and must contain no
    # random op -- a random transform on either side is an evaluation leak.
    rnd = [t for t in test_tf.transforms if type(t).__name__.startswith("Random")
           or "Jitter" in type(t).__name__]
    print(f"  [{'PASS' if not rnd else 'FAIL'}] val/test contain NO random ops"
          + (f"   FOUND {[type(t).__name__ for t in rnd]}" if rnd else ""))
    tr_rnd = [t for t in sigs[cats[0]] if t.startswith("Random") or "Jitter" in t]
    print(f"  [{'PASS' if tr_rnd or not aug_live_any(cfg) else 'INFO'}] "
          f"augmentation is TRAIN-ONLY (val/test clean)")

    # ---- which augmentation block is live ----
    aug_new = (cfg.get("augmentation") or {}).get("enabled", False)
    aug_old = (cfg.get("augment") or {}).get("enabled", False)
    live = [t for t in sigs[cats[0]] if t not in ("Resize", "ToTensor", "Normalize")]
    print("\nAUGMENTATION — TWO BLOCKS EXIST, and they do not agree")
    print(f"  augmentation.enabled = {aug_new}   (the documented block: scenarios, "
          f"per-category rules, verification gate)")
    print(f"  augment.enabled      = {aug_old}   (four undocumented lines)")
    print(f"  ACTUALLY APPLIED TO TRAIN: {live or ['nothing']}")
    if aug_old and not aug_new:
        print("  >>> the DOCUMENTED block is off and the UNDOCUMENTED one is on. The")
        print("      per-category physics rules below are NOT reached.")

    # ---- does augmentation change the NUMBER of training records? ----
    from data.augmentation import uses_record_expansion
    scen = (cfg.get("augmentation") or {}).get("scenario", "original") if aug_new else "original"
    expands = uses_record_expansion(scen)
    print("\nRECORD COUNT")
    print(f"  scenario = {scen!r}   expands records = {expands}")
    if not expands:
        print("  >>> the active augmentation is TRANSFORM-ONLY. It does NOT add samples:")
        print("      a training anomaly is seen ONCE per epoch, differently each time.")
        print("      Turning it off would NOT reduce the training set size.")
        vs = (cfg.get("data") or {}).get("injection_val_share")
        if vs:
            print(f"  >>> injection_val_share={vs} is justified in the config by 'train")
            print("      anomalies are multiplied xN by augmentation'. No multiplication")
            print("      happens on this path, so that justification does not hold.")
    mult_keys = ("target_effective_anomalies", "multiplier_cap")
    if not expands and any((cfg.get("augmentation") or {}).get(k) for k in mult_keys):
        print(f"  >>> multiplier machinery ({', '.join(mult_keys)}) is UNREACHABLE.")
    if not expands and (cfg.get("train") or {}).get("allow_weights_with_oversample") is not None:
        print("  >>> the class-weight/oversample double-correction guard never fires.")

    # ---- what the run LOG will claim ----
    print("\nWHAT stage.jsonl WILL RECORD")
    print(f"  augmentation stage 'enabled' field  = {aug_new}")
    if aug_old and not aug_new:
        print("  >>> MISLEADING. Every run's own provenance says augmentation was OFF")
        print("      while flip/rotation/jitter were applied. Any protocol section")
        print("      written from these logs will be wrong.")

    # ---- documented-but-unreachable protections ----
    try:
        from data.augmentation import NO_HFLIP
        flips = [c for c in NO_HFLIP if any("Flip" in t for t in sigs.get(c, sigs[cats[0]]))]
        print(f"\n  NO_HFLIP protects: {sorted(NO_HFLIP)}")
        if aug_old and not aug_new:
            print(f"  >>> UNREACHABLE. Every one of those categories is being flipped at "
                  f"p={cfg['augment'].get('horizontal_flip')}.")
            print("      The code's own comment: a horizontal flip on these 'invents an "
                  "impossible part'.")
        cj = (cfg.get("augment") or {}).get("color_jitter") or {}
        if aug_old and cj.get("saturation"):
            print(f"  >>> colour jitter saturation={cj['saturation']} is applied to "
                  f"ANOMALY images too.")
            print("      MVTec has literal 'color' defect types in carpet, leather, wood, "
                  "pill, metal_nut.")
    except Exception as e:
        print(f"  (could not check NO_HFLIP: {e})")

    # ---- other settings that silently change behaviour ----
    d = cfg.get("data", {})
    print("\nDATA / SPLIT")
    for k in ("image_size", "anomaly_fraction", "injection_val_share", "val_share_mode",
              "val_share_min", "val_anomaly_floor", "test_anomaly_floor",
              "floors_validated", "de_confound_frac", "test_normal_floor",
              "cache_decoded", "use_validation", "val_split", "num_workers"):
        if k in d:
            print(f"  {k:24s} {d[k]}")
    t = cfg.get("train", {})
    print("\nTRAIN")
    for k in ("num_epochs", "patience", "early_stopping", "learning_rate",
              "batch_size", "pool_lr_mult", "use_class_weights"):
        if k in t:
            print(f"  {k:24s} {t[k]}")
    m = cfg.get("model", {})
    print("\nMODEL")
    for k in ("pooling", "head", "pretrained", "freeze_backbone", "taps",
              "tap_proj_dim", "tap_norm"):
        print(f"  {k:24s} {m.get(k)}" + ("   <- absent from config, default used"
                                         if k not in m else ""))
    # ---- the GENERAL check: config that cannot act, and code that acts unseen ----
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        dead, shadowed, missing = cross_check_config_against_code(cfg, root)
        print("\nCONFIG vs CODE  (the general form of the augmentation bug)")
        if shadowed:
            print("  SHADOWED top-level blocks — one name is a prefix of another and")
            print("  nothing states which the code reads:")
            for a, b in shadowed:
                # a top-level block is not always a dict: `seed: 123` and `seeds:
                # [...]` shadow each other too, and calling .get on an int is how
                # this very checker first crashed.
                def _en(k):
                    v = cfg.get(k)
                    return v.get("enabled") if isinstance(v, dict) else f"<{type(v).__name__}>"
                print(f"    {a!r} (enabled={_en(a)})  vs  {b!r} (enabled={_en(b)})")
        if dead:
            print(f"  DEAD — {len(dead)} config key(s) no source file mentions. They read")
            print("  as settings and cannot affect anything:")
            for k in dead[:25]:
                print(f"    {k}")
            if len(dead) > 25:
                print(f"    ... and {len(dead)-25} more")
        if missing:
            print(f"  DEFAULTED — {len(missing)} key(s) the code reads with .get(k, default)")
            print("  that are ABSENT from the config, so a default silently governs:")
            print("    " + ", ".join(missing[:30]))
        if not (dead or shadowed or missing):
            print("  clean: no dead keys, no shadowed blocks, no silent defaults")
    except Exception as e:
        print(f"\n  cross-check failed: {type(e).__name__}: {e}")

    print(f"\nFILTERS enabled = {(cfg.get('filters') or {}).get('enabled')}")
    print(f"SEED {cfg.get('seed')}   run.datasets={cfg.get('run',{}).get('datasets')}"
          f"   categories={cfg.get('run',{}).get('categories')}")
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
