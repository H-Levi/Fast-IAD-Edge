#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_train.py — the entry point that wires every layer together.

It owns the things the pure engine deliberately does not: building datasets and
loaders, looping over the run selection (datasets x categories x backbones),
profiling each split before training, running fit (val-selection), scoring test
ONCE at the val-chosen threshold, and recording everything.

Run examples
------------
# one category, one backbone (uses configs/default.yaml for everything else)
python run_train.py --config configs/default.yaml \
    run.datasets=[mvtec] run.categories=[screw] run.backbones=[efficientnet]

# whole dataset, all three backbones, in one run
python run_train.py run.datasets=[mvtec] run.categories=[all] \
    run.backbones=[efficientnet,mobilenet,squeezenet]

# both datasets together
python run_train.py run.datasets=[mvtec,visa] run.categories=[all]

# turn on a filter pipeline from the CLI (no code edit)
python run_train.py filters.enabled=true \
    'filters.pipeline=[{name: clahe, clip_limit: 2.0, tile_grid: 8}]'

Any config key is overridable as key=value (dotted). Lists/dicts are YAML.
"""

import argparse
import copy
import sys
import os
import time as _time

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.io import load_config, apply_overrides, make_run_dir, save_json
from utils.seed import set_seed, seed_worker
from utils.transforms import build_transforms, build_scenario_transforms
from data.mvtec import MVTecDataset
from data.visa import ViSADataset
from data.view import SplitView
from data.augmentation import expand_records, uses_record_expansion
from data.multiplier import (find_census, load_census, derive_multiplier,
                             train_anomaly_counts)
from data.stats import profile_dataset, warnings_from_manifest
from models.model_factory import build_model
from engine.train import fit
from engine.evaluate import run_inference
from analysis.metrics import compute_all, safe_auroc
from analysis.reliability import inverted_flag  # noqa: E402
from analysis.guardrails import (leak_self_check, detect_collapse_from_file,
                                 assert_scenario_verified)
from analysis.seeds import resolve_seeds
from analysis.protocol import protocol_fingerprint
from analysis.reliability import (dataset_tier, caveats_from_flags, caveat_line,
                                  is_reportable, tiered_means)
from instrumentation.recorder import Recorder
from instrumentation.report import run_report


# --------------------------------------------------------- dataset build ----
def data_cfg_for(cfg, name, category):
    """The data config for ONE category: global `data` with that category's entry in
    `data.per_category` merged on top. Returns (merged, overrides_applied).

    WHY. The baseline needs toothbrush at anomaly_fraction 0.2 with a fixed 0.5
    share (6 training defects instead of 2 -- 0.4583 collapsed -> 0.7269), while
    every other category keeps the default. Without per-category overrides that
    took a separate run, so no single command could reproduce the baseline.

    Keys are "dataset/category" ("mvtec/toothbrush") so an MVTec and a VisA name
    can never collide. An override naming a key absent from `data` is REJECTED,
    because a typo would otherwise silently do nothing -- the augment/augmentation
    bug in miniature.
    """
    d = dict(cfg["data"])
    pc = (cfg["data"].get("per_category") or {})
    ov = dict(pc.get(f"{name}/{category}") or {})
    unknown = [k for k in ov if k not in d]
    if unknown:
        raise KeyError(f"data.per_category['{name}/{category}'] sets unknown data "
                       f"key(s) {unknown}; known: {sorted(k for k in d if k != 'per_category')}")
    d.update(ov)
    return d, ov


def build_dataset(name, cfg, category):
    """Build the dataset, then apply the REGIME RULE (K-030, A-073) when train.regime_min_defects is set: a category
    whose training split has fewer defects than the threshold is 'scarce' -> no calibration split (rebuilt without one)
    and, in class_weight(), the historical near-neutral weight neg/(neg+pos) instead of neg/pos. A category at or above
    it is 'rich' -> the configured class weight and calibration apply. Decided from the TRAINING split only, before any
    training or testing. Absent key = no rule (every past run unchanged)."""
    ds = _build_dataset_raw(name, cfg, category)
    rm = cfg["train"].get("regime_min_defects")
    if rm is None:
        return ds
    n_def = sum(1 for _, l in ds.records("train") if l == 1)
    rich = n_def >= int(rm)
    if not rich and getattr(ds, "calib_info", {}).get("applied"):
        import copy
        c2 = copy.deepcopy(cfg); c2["data"]["calib_n"] = 0; c2["data"]["calib_split"] = 0.0
        ds = _build_dataset_raw(name, c2, category)
    ds.regime = {"min_defects": int(rm), "train_defects": n_def, "rich": rich}
    return ds


def _build_dataset_raw(name, cfg, category):
    """Build ONE dataset object owning all splits (train/val/test).

    Using a single object makes the splits internally consistent — critical for
    MVTec where injected anomalies must be removed from test in the same pass.
    Transforms are NOT attached here; SplitView supplies the right transform per
    loader (train augmented, val/test not).
    """
    d, _ov = data_cfg_for(cfg, name, category)
    seed = cfg["seed"]
    if name == "mvtec":
        return MVTecDataset(cfg["paths"]["mvtec_root"], category=category,
                            transform=None, val_split=d["val_split"], seed=seed,
                            num_train_anomalies=d["num_train_anomalies"],
                            anomaly_fraction=d.get("anomaly_fraction"),
                            injection_val_share=d.get("injection_val_share", 0.5),
                            val_share_mode=d.get("val_share_mode", "fixed"),
                            val_anomaly_floor=d.get("val_anomaly_floor", 10),
                            val_share_min=d.get("val_share_min", 0.5),
                            de_confound_frac=d.get("de_confound_frac", 0.0),
                            test_normal_floor=d.get("test_normal_floor", 15),
                            de_confound_val_share=d.get("de_confound_val_share", 1.0 / 3.0),
                            cache_size=(int(d.get("image_size", 224)) if d.get("cache_decoded", False) else None),
                            calib_split=float(d.get("calib_split", 0.0) or 0.0),
                            calib_n=int(d.get("calib_n", 0) or 0), calib_min_train=int(d.get("calib_min_train", 100)),
                            use_validation=d["use_validation"])
    if name == "visa":
        return ViSADataset(cfg["paths"]["visa_root"], category=category,
                           transform=None,
                           # was hard-coded "highshot", so the run record could
                           # not say which split it used and no override could
                           # change it -- against configs/default.yaml's own
                           # single-source-of-truth rule.
                           shot=d.get("visa_shot", "highshot"),
                           val_split=d["val_split"],
                           cache_size=(int(d.get("image_size", 224))
                                       if d.get("cache_decoded", False) else None),
                           use_validation=d["use_validation"], seed=seed,
                           calib_split=float(d.get("calib_split", 0.0) or 0.0),
                           calib_n=int(d.get("calib_n", 0) or 0), calib_min_train=int(d.get("calib_min_train", 100)),
                           train_anomaly_n=int(d.get("train_anomaly_n", 0) or 0))
    raise ValueError(f"Unknown dataset {name}")


def categories_for(name, cfg):
    """Categories to run for THIS dataset.

    `run.categories` is one list shared by all datasets, so a mixed request like
    [capsule, candle] names an MVTec category and a VisA one. Each dataset takes
    only the names it actually has, rather than crashing on the other's (an
    unfiltered name previously raised a raw FileNotFoundError deep in the loader).
    """
    known = (MVTecDataset.categories if name == "mvtec" else ViSADataset.categories)
    sel = cfg["run"]["categories"]
    # A CLI override is parsed as YAML, so `run.categories=all` yields the STRING
    # "all", not ["all"]. That happened to work only because `"all" in "all"` is a
    # substring test -- while `run.categories=screw` would give sel="screw", fail the
    # substring test, then iterate CHARACTERS and silently run nothing while printing
    # a "skipping ['s','c','r','e','w']" message. Normalise instead of relying on luck.
    if isinstance(sel, str):
        sel = [sel]
    if "all" in sel:
        return known
    keep = [c for c in sel if c in known]
    skipped = [c for c in sel if c not in known]
    if skipped:
        print(f"  (note: {name} has no category {skipped} — skipping; "
              f"running {keep or 'nothing'})")
    return keep



def resolve_multiplier(cfg, dataset_name, category, train_labels):
    """§4.1 stages 1+3: N = ceil(target / real anomaly count), from the census.

    Prefers the census (the frozen factual base). If no census is available it
    falls back to the live train-split anomaly count, which is the same quantity
    measured at run time — the multiplier is never hardcoded either way.
    """
    aug = cfg.get("augmentation", {})
    target = int(aug.get("target_effective_anomalies", 30))
    cap = int(aug.get("multiplier_cap", 10))

    count = None
    cpath = aug.get("census_path") or find_census(cfg["paths"]["output_root"])
    if cpath and os.path.exists(cpath):
        try:
            count = train_anomaly_counts(load_census(cpath)).get(
                f"{dataset_name}/{category}")
        except Exception:
            count = None
    if count is None:
        count = int(sum(train_labels))
    N = derive_multiplier(count, target, cap)
    if not aug.get("target_validated", False):
        print(f"  ! multiplier N={N} is PROVISIONAL (target={target} not yet validated "
              f"by the §4.1.2 sweep)")
    return N


def make_loader(records, transform, cfg, shuffle, drop_last=False,
                anomaly_transform=None, synthetic_op=None, transplant_op=None):
    """Bind a loader to an immutable view with an explicit transform.

    anomaly_transform / synthetic_op are used only by the TRAIN loader when an
    augmentation scenario is active (§4.2); they are None everywhere else, so
    val/test remain a plain single-transform view.
    """
    cache_size = (int(cfg["data"].get("image_size", 224))
                  if cfg["data"].get("cache_decoded", False) else None)
    if cache_size:
        # PRE-WARM IN THE MAIN PROCESS. Loader workers are separate processes,
        # re-created every epoch (persistent_workers is off, deliberately: turning
        # it on changes the augmentation random stream and breaks exact
        # replication). A cache filled inside a worker dies with it. Filling it
        # here, before the loader exists, means every epoch's freshly forked
        # workers inherit the full cache copy-on-write.
        from data import imgcache
        for rec in records:
            imgcache.load(rec[0], cache_size)
    view = SplitView(records, transform, anomaly_transform=anomaly_transform,
                     synthetic_op=synthetic_op, cache_size=cache_size, transplant_op=transplant_op)
    return DataLoader(view, batch_size=cfg["train"]["batch_size"],
                      shuffle=shuffle, num_workers=cfg["data"]["num_workers"],
                      drop_last=drop_last, worker_init_fn=seed_worker)


def apply_transplant(train_records, dataset, dataset_name, cfg):
    """F-L124 defect transplant: in the SCARCE regime only, top the training defects up to data.transplant.target_defects
    with synthetic records (a random training normal, tag 'transplant'; the defect is pasted at load time by
    data/transplant.py). Returns (records, op or None, description for stage.jsonl). Off by default: records unchanged."""
    tp = (cfg["data"].get("transplant") or {})
    desc = {"enabled": bool(tp.get("enabled", False)), "applied": False}
    if not desc["enabled"]:
        return train_records, None, desc
    reg = getattr(dataset, "regime", None) or {}
    if reg.get("rich", False):
        desc["reason"] = "rich regime (>= regime_min_defects training defects): not applied"
        return train_records, None, desc
    if dataset_name != "mvtec":
        raise NotImplementedError("transplant is implemented for the MVTec mask layout only")
    import numpy as np
    from data.transplant import DefectTransplant
    donors = [r[0] for r in train_records if r[1] == 1 and (len(r) < 3 or r[2] is None)]
    normals = [r[0] for r in train_records if r[1] == 0 and (len(r) < 3 or r[2] is None)]
    n_syn = max(0, int(tp.get("target_defects", 48)) - len(donors))    # counts ALL real defects (never removed)
    DefectTransplant(donors, feather_radius=float(tp.get("feather_radius", 2.0)))  # raises if ANY training mask is missing
    donor_rule = str(tp.get("donor_rule", "all"))
    if donor_rule == "inside_object":                          # v3 (F-L129): paste only defects inside the object
        from data.transplant import donor_inside_object
        keep = [donor_inside_object(d) for d in donors]
        desc["excluded_donor_types"] = sorted({os.path.basename(os.path.dirname(d)) for d, k in zip(donors, keep) if not k})
        paste_donors = [d for d, k in zip(donors, keep) if k]
    elif donor_rule == "all":
        paste_donors = list(donors)
    else:
        raise ValueError(f"data.transplant.donor_rule must be all | inside_object, got {donor_rule!r}")
    desc.update({"donor_rule": donor_rule, "n_paste_donors": len(paste_donors)})
    if not paste_donors:
        desc["reason"] = "no donor passes the donor rule: no synthetic defects"
        return train_records, None, desc
    op = DefectTransplant(paste_donors, feather_radius=float(tp.get("feather_radius", 2.0)))
    donors_all, donors = donors, paste_donors
    placement = str(tp.get("placement", "random"))
    if placement == "random":                                  # v1 (F-L124), unchanged: random host, random donor per load
        rng = np.random.default_rng(int(cfg["seed"]) + 3)      # own stream: does not disturb any other random draw
        picks = rng.choice(len(normals), size=n_syn, replace=n_syn > len(normals))
        records = list(train_records) + [(normals[i], 1, "transplant") for i in picks]
    elif placement == "best_match":                            # v2 (F-L127): context-matched host, fixed donor per record
        from data.transplant import plan_best_match
        pairs = plan_best_match(donors, normals, n_syn, size=int(cfg["data"].get("image_size", 288)),
                                ring_px=int(tp.get("ring_px", 8)))
        records = list(train_records) + [(h, 1, "transplant", di) for h, di in pairs]
        desc["n_distinct_hosts"] = len({h for h, _ in pairs})
    elif placement == "duplicate":                             # REQUESTS 2a (F-L140): QUANTITY ONLY — the same top-up count
        # as real copies of the training defects (round-robin, no pasting, no new host); separates "more defect images"
        # from "defect no longer tied to its session's background". Tag 'duplicate' = plain load (no op) in SplitView.
        records = list(train_records) + [(donors[i % len(donors)], 1, "duplicate") for i in range(n_syn)]
        op = None
    else:
        raise ValueError(f"data.transplant.placement must be random | best_match | duplicate, got {placement!r}")
    desc["placement"] = placement
    desc.update({"applied": True, "n_donors": len(donors_all), "n_synthetic": int(n_syn), "n_host_normals": len(normals),
                 "target_defects": int(tp.get("target_defects", 48)), "feather_radius": float(tp.get("feather_radius", 2.0))})
    return records, op, desc


def class_weight(train_ds, cfg, device, scenario="original"):
    """pos_weight for BCEWithLogits, from train labels (no image loading).

    §2.4 DOUBLE-CORRECTION GUARD: oversample x N already raises the anomaly
    ratio in the data. Running class-weights on top corrects the same imbalance
    twice — the spec calls this "the old backwards-weights failure in a new
    form". So when a record-expanding scenario is active, class-weights are
    disabled unless the user explicitly opts into combining them (in which case
    the pair must be tuned together through the comparator, never assumed).
    """
    from data.augmentation import uses_record_expansion

    if not cfg["train"].get("use_class_weights", False):
        return None

    if uses_record_expansion(scenario):
        allow_both = cfg["train"].get("allow_weights_with_oversample", False)
        if not allow_both:
            print(f"  ! §2.4 guard: scenario '{scenario}' oversamples anomalies, so "
                  f"class-weights are DISABLED (double-correction). Set "
                  f"train.allow_weights_with_oversample=true to combine them "
                  f"deliberately and tune the pair via the comparator.")
            return None
        print(f"  ! §2.4: combining oversample AND class-weights by explicit request "
              f"— this must be tuned as a pair, not assumed.")

    labels = train_ds.labels("train")
    if len(set(labels)) < 2:
        return None
    pos = sum(labels); neg = len(labels) - pos
    # CP-001 (A-001). `neg_over_total` is the historical formula and the DEFAULT: it gives
    # pos_weight = neg/(neg+pos) < 1, i.e. effectively NO balancing -- every run up to baseline-v1
    # used it, and it stays the default so those runs reproduce bit-for-bit. `neg_over_pos` is the
    # balancing weight BCEWithLogitsLoss expects. Passed on the CLI only (train.class_weight_mode=...),
    # so the frozen config hash is untouched. Any other value is an error, not a silent default.
    mode = cfg["train"].get("class_weight_mode", "neg_over_total")
    reg = getattr(train_ds, "regime", None)
    if reg is not None and not reg["rich"]:
        mode = "neg_over_total"   # regime rule: scarce category -> the historical near-neutral weight (A-073)
    if mode == "neg_over_total":
        val = neg / len(labels)
    elif mode == "neg_over_pos":
        val = neg / max(pos, 1)
    else:
        raise ValueError(f"train.class_weight_mode={mode!r}; expected 'neg_over_total' or 'neg_over_pos'")
    val = val * cfg["train"].get("weight_multiplier", 1.0)
    # train.pos_weight_cap (A-069, K-027): one recipe for every dataset, stated as a rule on the TRAINING data — the
    # balancing weight may not exceed the cap. With 4-17 training defects (MVTec) neg/pos reaches 10-77; VisA's lead
    # config stays at ~3-8, below a cap of 10, so it is untouched. Absent/None = no cap (every past run unchanged).
    cap = cfg["train"].get("pos_weight_cap")
    if cap is not None:
        val = min(val, float(cap))
    return torch.tensor(val, dtype=torch.float32, device=device)


# ------------------------------------------------------------------ main ----
def build_optimizer(model, cfg):
    """Adam, with the pooling module on its own learning rate.

    Why this exists. LSE's sharpness `p` is ONE scalar sharing a learning rate
    chosen for ~1.3M weights. In the real run it moved 0.992 -> 1.045, i.e. not at
    all -- so "learnable pooling" was, in practice, pooling frozen at an arbitrary
    point on the avg<->max line, and neither a win nor a loss for LSE could be read
    from that run.

    Honest about what this does and does not settle. A larger step lets `p` move; it
    does not prove `p` SHOULD move. If the gradient on `p` has no consistent sign,
    a bigger step produces a random walk, not learning. The two cases are
    distinguishable only from the trajectory, which is why fit() now logs `p` every
    epoch: a steady drift means the old rate was simply too slow, oscillation around
    ~1.0 means the pooling exponent has no preferred value here -- and that is a
    result too, not a failure.
    """
    lr = cfg["train"]["learning_rate"]
    mult = float(cfg["train"].get("pool_lr_mult", 50.0))
    pool = getattr(getattr(model, "backbone", None), "pool", None)
    # A-038: the multiplier was built for ONE scalar exponent (LSE `q`, GeM `p_raw`), but it is keyed on the
    # module name, so whatever sits at `backbone.pool` inherits it -- including TapPool's zero-start `gates`
    # and tap_norm's LayerNorms, which therefore trained at 50x lr in every tap arm run so far.
    # `train.pool_lr_scope`: "all" (DEFAULT, historical: every pooling parameter; reproduces all past runs)
    # or "exponent" (only the pooling exponents; gates, norms and attention convs train at the base lr).
    # Passed on the CLI only, like CP-001, so the frozen config hash is untouched.
    scope = cfg["train"].get("pool_lr_scope", "all")
    if scope == "all":
        pool_params = [q for q in pool.parameters()] if pool is not None else []
    elif scope == "exponent":
        pool_params = ([q for n, q in pool.named_parameters() if n.split(".")[-1] in ("q", "p_raw")]
                       if pool is not None else [])
    else:
        raise ValueError(f"train.pool_lr_scope={scope!r}; expected 'all' or 'exponent'")
    if not pool_params:
        return torch.optim.Adam(model.parameters(), lr=lr)
    pool_ids = {id(q) for q in pool_params}
    base = [q for q in model.parameters() if id(q) not in pool_ids]
    print(f"  optimizer: {len(base)} tensors @ lr={lr:g}, "
          f"{len(pool_params)} pooling tensor(s) @ lr={lr * mult:g} "
          f"(pool_lr_mult={mult:g})")
    return torch.optim.Adam(
        [{"params": base, "lr": lr},
         {"params": pool_params, "lr": lr * mult}], lr=lr)


def model_cost(model, cfg, device):
    """MACs at the configured input size, to sit next to params in summary.json.

    summary() cannot compute this itself -- MACs depend on the input resolution,
    which the model does not know. Recorded here so every run carries its own
    compute number instead of it living only in a separate benchmark.
    """
    try:
        from analysis.benchmark import count_flops
        size = int(cfg["data"]["image_size"])
        return {"gmacs": count_flops(model, device, size).replace(" GMACs", ""),
                "macs_at_px": size}
    except Exception as e:
        return {"gmacs": f"n/a ({type(e).__name__})"}


def seeds_for(cfg):
    """Which seeds to run. `run.seeds=[...]` is explicit; `seeding.tier=claim` uses the
    claim seeds; otherwise the single `seed`."""
    explicit = (cfg.get("run") or {}).get("seeds")
    if explicit:
        return [int(x) for x in (explicit if isinstance(explicit, list) else [explicit])]
    tier, seeds = resolve_seeds(cfg)
    if tier == "claim" and len(seeds) > 1:
        return [int(x) for x in seeds]
    return [int(cfg["seed"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    # MULTI-SEED, IN THIS SCRIPT. Before, run_train trained once and pointed claim runs
    # at analysis/compare_backbones.py -- which saves NO per-image scores, no collapse
    # check, no reportable flag: a 3-seed run through it could not be analysed at all
    # (O-01). Now each seed is a COMPLETE single-seed run with the full recording path,
    # written to <output_root>_seed<s>, and each bundles itself.
    # On MVTec a seed also changes WHICH defects are injected (split variance, not just
    # initialisation variance) -- a band across seeds must be described that way.
    seeds = seeds_for(cfg)
    if len(seeds) == 1:
        cfg["seed"] = seeds[0]
        run_once(cfg)
        return
    root = cfg["paths"]["output_root"].rstrip("/")
    print(f"MULTI-SEED RUN: seeds {seeds} -> {root}_seed<s>/  (one complete run per seed)")
    if "mvtec" in cfg["run"]["datasets"]:
        print("  note: on MVTec each seed re-draws the injected defects, so the spread "
              "across seeds is SPLIT variance, not just initialisation variance")
    for s_ in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = s_
        c["paths"]["output_root"] = f"{root}_seed{s_}"
        print(f"\n{'#' * 78}\n# SEED {s_}\n{'#' * 78}")
        run_once(c)


def run_once(cfg):
    """One complete single-seed run: every category, full recording, bundled."""

    # WHAT IS ACTUALLY ACTIVE, printed before a single batch is loaded and written
    # into the run directory. Running a checker once with the DEFAULT config proves
    # nothing about a run that passes overrides -- which is how `augment.enabled:
    # true` survived the entire project while the documented block read false.
    # Printing it here cannot be skipped and travels in the bundle.
    try:
        from analysis.what_runs import active_state
        _state = active_state(cfg)
        print("\n" + _state + "\n")
    except Exception as _e:
        _state = f"active_state unavailable: {type(_e).__name__}: {_e}"
        print("  ! " + _state)
    # Which code produced this run. The baseline run itself did not record it, so the
    # `baseline-v1` tag had to be placed from the session log rather than from the files.
    try:
        import subprocess
        _here = os.path.dirname(os.path.abspath(__file__))
        _git = lambda *a: subprocess.run(["git", "-C", _here, *a], capture_output=True,
                                         text=True, timeout=10).stdout.strip()
        _dirty = [l for l in _git("status", "--porcelain").splitlines() if l[3:].endswith(".py")
                  or l[3:].endswith(".yaml")]
        _prov = f"code: git {_git('rev-parse', 'HEAD') or 'unknown'}" + (
            f"  DIRTY ({len(_dirty)} .py/.yaml files modified)" if _dirty else "  clean")
    except Exception as _e:
        _prov = f"code: git unavailable ({type(_e).__name__})"
    print("  " + _prov)
    _state = _prov + "\n" + _state

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    # A CUDA request that silently becomes CPU turns a 2 s epoch into ~30 s and gives
    # no sign of it except the wall clock. On Kaggle this happens whenever the session
    # comes back without an accelerator after the GPU is toggled off. Refuse unless
    # CPU was asked for.
    if str(cfg.get("device", "cuda")).startswith("cuda") and device.type == "cpu":
        if not cfg.get("allow_cpu", False):
            sys.exit("\n  !!! device=cuda was requested but NO GPU is available -- the run would "
                     "fall back to CPU (~10x slower).\n  !!! On Kaggle: Settings -> Accelerator "
                     "-> GPU, then restart. To run on CPU deliberately pass allow_cpu=true.\n")
        print("  !!! RUNNING ON CPU (allow_cpu=true). Timings are NOT comparable to GPU runs.")

    # The run directory has to name the configuration, not just the datasets.
    # A 2x2 factorial produces four runs whose only distinguishing feature would
    # otherwise be a timestamp -- unreadable a week later, and easy to mis-pair
    # when comparing. pooling and image_size are the axes under test, so they go
    # in the name.
    tag = "_".join([*cfg["run"]["datasets"], *cfg["run"]["backbones"],
                    cfg["model"].get("pooling", "avg"),
                    f'{cfg["data"]["image_size"]}px',
                    cfg.get("seeding", {}).get("tier", "exploration")])
    run_dir = make_run_dir(cfg["paths"]["output_root"], tag)
    recorder = Recorder(run_dir, cfg)
    print(f"Device: {device}\nRun dir: {run_dir}\n")

    per_category = {}

    for dataset_name in cfg["run"]["datasets"]:
        for backbone in cfg["run"]["backbones"]:
            for category in categories_for(dataset_name, cfg):
                key = f"{dataset_name}/{backbone}/{category}"
                print("=" * 78)
                print(key)
                print("=" * 78)

                set_seed(cfg["seed"])  # per-category determinism
                train_tf, filt_desc = build_transforms(cfg, train=True)
                test_tf, _ = build_transforms(cfg, train=False)
                # Drop the previous category's decoded images BEFORE this category's
                # loaders are built -- the first version cleared after them, which
                # would have wiped the pre-warmed cache just before training.
                if cfg["data"].get("cache_decoded", False):
                    from data import imgcache as _ic
                    _ic.clear()
                dataset = build_dataset(dataset_name, cfg, category)

                # ---- profile + flag degenerate setups BEFORE training ----
                manifest = profile_dataset(dataset, dataset_name, category,
                                           cfg["data"]["use_validation"],
                                           test_anomaly_floor=cfg["data"].get("test_anomaly_floor", 15),
                                           val_anomaly_floor=cfg["data"].get("val_anomaly_floor", 10))
                manifest["filters"] = filt_desc
                # ARM F (F-L42/F-L48): record the de-confound decision in the SAME stage
                # row as the split counts, so a later reader cannot see one without the
                # other. A skipped category says so explicitly rather than looking
                # de-confounded because the arm was requested.
                _dc = getattr(dataset, "de_confound", None)
                if _dc:
                    manifest["de_confound"] = _dc
                    if _dc.get("enabled"):
                        print(f"  ARM F swap k={_dc['k']}: test/good -> "
                              f"{_dc['n_testgood_to_train']} train + "
                              f"{_dc['n_testgood_to_val']} val; "
                              f"{_dc['n_traingood_to_test']} train/good -> test "
                              f"(test size unchanged)")
                recorder.log_stage(key, "dataset", manifest)
                if getattr(dataset, "calib_info", None) is not None:
                    recorder.log_stage(key, "calib", dataset.calib_info)   # applied? fraction/count? floor-rule skip?
                if getattr(dataset, "regime", None) is not None:
                    recorder.log_stage(key, "regime", dataset.regime)   # rich/scarce decision from the training split
                if getattr(dataset, "train_anomaly_info", {}).get("applied"):
                    recorder.log_stage(key, "train_anomaly_n", dataset.train_anomaly_info)   # scarcity pilot (F-L105)
                # F-L46: nothing recorded wall-clock time, so the whole budget was
                # denominated in a number nobody had measured. One line fixes it.
                _t_cat0 = _time.time()
                _mit = {"class_weights": cfg["train"].get("use_class_weights", False),
                        "oversample": uses_record_expansion(
                            (cfg.get("augmentation", {}) or {}).get("scenario", "original")
                            if (cfg.get("augmentation", {}) or {}).get("enabled", False) else "original")}
                for w in warnings_from_manifest(manifest, _mit):
                    print("  ! " + w)
                _inj = getattr(dataset, "injection_manifest", None)
                if _inj and _inj.get("injected"):
                    print(f"  defect-type coverage: train={len(_inj['train_defect_types'])}/"
                          f"{_inj['n_defect_types']} types, val={len(_inj['val_defect_types'])}/"
                          f"{_inj['n_defect_types']} types "
                          f"(val_share={_inj.get('injection_val_share')})")
                    if not _inj.get("type_coverage_complete", True):
                        print("  ! defect-type coverage INCOMPLETE in train")
                if manifest["flags"]["single_class_train"]:
                    print("  -> skipping (cannot train a binary classifier on one class)\n")
                    continue

                # ---- loaders: train augmented; val/test NOT augmented ----
                # §4: scenario transforms + record expansion apply to TRAIN ONLY.
                # With augmentation disabled (default) this resolves to the
                # baseline transform and the untouched record list.
                n_tf, a_tf, synth_op, aug_desc = build_scenario_transforms(cfg, category)
                # §4.1.4 gate: an unverified scenario must not reach training.
                assert_scenario_verified(
                    aug_desc["scenario"], cfg["paths"]["output_root"],
                    require=cfg["augmentation"].get("require_verification", True))
                train_records = dataset.records("train")
                mult = 1
                if uses_record_expansion(aug_desc["scenario"]):
                    mult = resolve_multiplier(cfg, dataset_name, category,
                                              dataset.labels("train"))
                    train_records = expand_records(
                        train_records, aug_desc["scenario"], mult,
                        cutpaste_ratio=cfg["augmentation"].get("cutpaste_ratio", 1.0))
                # Record augmentation state ALWAYS — including 'original'. If it
                # were logged only when active, an artifact with no augmentation
                # entry would be ambiguous: augmentation off, or an older run?
                n_anom = sum(1 for r in train_records if r[1] == 1)
                aug_desc.update({"multiplier": mult,
                                 "train_records_after_expansion": len(train_records),
                                 "train_anomalies_after_expansion": n_anom,
                                 "enabled": cfg["augmentation"].get("enabled", False),
                                 # WHAT ACTUALLY RAN. `enabled` above is the DESIGNED
                                 # `augmentation` block, which is off -- and it was the
                                 # only field logged, so every run recorded
                                 # augmentation as disabled while flip/rotation/jitter
                                 # from the `augment` block were applied (F-L55).
                                 # These fields describe the real train transform.
                                 "train_transform_ops": [type(t).__name__ for t in
                                                         getattr(n_tf, "transforms", [])],
                                 "augment_block_enabled": bool((cfg.get("augment") or {})
                                                               .get("enabled", False)),
                                 "augmentation_applied": any(
                                     type(t).__name__.startswith("Random")
                                     or "Jitter" in type(t).__name__
                                     for t in getattr(n_tf, "transforms", [])),
                                 "record_multiplier": mult})
                if aug_desc["scenario"] != "original":
                    print(f"  augmentation: {aug_desc['scenario']} "
                          f"(N={mult}, train n={len(train_records)}, anomalies={n_anom})")
                recorder.log_stage(key, "augmentation", aug_desc)
                train_records, tp_op, tp_desc = apply_transplant(train_records, dataset, dataset_name, cfg)
                recorder.log_stage(key, "transplant", tp_desc)
                if tp_desc.get("applied"):
                    print(f"  transplant: +{tp_desc['n_synthetic']} synthetic defects from {tp_desc['n_donors']} donors")
                train_loader = make_loader(train_records, n_tf, cfg,
                                           shuffle=True, drop_last=True,
                                           anomaly_transform=a_tf,
                                           synthetic_op=synth_op, transplant_op=tp_op)
                if not cfg["data"]["use_validation"]:
                    print("  ! use_validation=false -> selecting on TEST (leaky; ablation only)")
                    val_loader = make_loader(dataset.records("test"), test_tf, cfg, shuffle=False)
                else:
                    val_loader = make_loader(dataset.records("val"), test_tf, cfg, shuffle=False)
                test_loader = make_loader(dataset.records("test"), test_tf, cfg, shuffle=False)

                # ---- model + optim + loss ----
                model = build_model(cfg, backbone).to(device)
                print("  model:", model.summary())
                optimizer = build_optimizer(model, cfg)
                pos_w = class_weight(dataset, cfg, device, scenario=aug_desc["scenario"])
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
                # Operating-point loss (engine/losses.py): opt-in, CLI only; weight 0/absent = plain BCE as before.
                _opw = float(cfg["train"].get("op_loss_weight", 0) or 0)
                if _opw > 0:
                    from engine.losses import OperatingPointLoss
                    loss_fn = OperatingPointLoss(loss_fn, weight=_opw,
                                                 beta=float(cfg["train"].get("op_loss_beta", 0.1)),
                                                 margin=float(cfg["train"].get("op_loss_margin", 1.0)))

                _op_log = ({"weight": loss_fn.weight, "beta": loss_fn.beta, "margin": loss_fn.margin}
                           if hasattr(loss_fn, "tail_term") else None)
                # View-consistency term (engine/losses.py, 2026-09-26): opt-in; weight 0/absent = not built = unchanged.
                _vcw = float(cfg["train"].get("view_consistency_weight", 0) or 0)
                if _vcw > 0:
                    from engine.losses import ViewConsistencyLoss
                    loss_fn = ViewConsistencyLoss(loss_fn, weight=_vcw)
                # Session-alignment term (engine/losses.py, pilot F-L105): opt-in, CLI only; weight 0/absent = unchanged.
                _saw = float(cfg["train"].get("session_align_weight", 0) or 0)
                if _saw > 0:
                    from engine.losses import SessionAlignLoss
                    loss_fn = SessionAlignLoss(loss_fn, weight=_saw,
                                               momentum=float(cfg["train"].get("session_align_momentum", 0.9))).to(device)
                recorder.log_stage(key, "fit_start",
                                   {"model": model.summary(),
                                    "pos_weight": None if pos_w is None else float(pos_w),
                                    "op_loss": _op_log,
                                    "view_consistency": ({"weight": loss_fn.weight} if hasattr(loss_fn, "needs_second_view") else None),
                                    "session_align": ({"weight": loss_fn.weight, "momentum": loss_fn.momentum}
                                                      if hasattr(loss_fn, "align_term") else None)})

                # ---- train (val-selection) ----
                _t_setup_end = _time.time()
                best = fit(model, train_loader, val_loader, optimizer, loss_fn,
                           device, cfg, run_dir, key.replace("/", "_"),
                           recorder=_KeyedRecorder(recorder, key), test_loader=test_loader)
                _t_train_end = _time.time()
                if hasattr(loss_fn, "align_term"):   # proof the term fired: 0 aligned images = the pilot arm is void
                    recorder.log_stage(key, "session_align_done", {"n_aligned_total": loss_fn.n_aligned,
                                                                   "last_term": loss_fn.last_term})

                # ---- test ONCE at val-chosen threshold ----
                # §1.1 leak self-check tripwire: halts if split/threshold/transform
                # invariants are violated. Uses the ACTUAL variables test-scoring
                # will use, so it catches real regressions.
                test_threshold = best["best_threshold"]
                leak_self_check(
                    train_paths=[p for (p, _) in dataset.records("train")],
                    test_paths=[p for (p, _) in dataset.records("test")],
                    val_threshold=best["best_threshold"],
                    test_threshold=test_threshold,
                    val_transform=val_loader.dataset.transform,
                    test_transform=test_loader.dataset.transform,
                    use_validation=cfg["data"]["use_validation"],
                    key=key,
                    val_paths=([p for (p, _) in dataset.records("val")]
                               if cfg["data"]["use_validation"] else None),
                    calib_paths=[p for (p, _) in dataset.records("calib")])
                model.load_state_dict(torch.load(best["checkpoint"], map_location=device))

                # ---- scores for the NON-test splits, from the scored checkpoint ----
                # Needed for prior-free thresholding and calibration. The val scores the
                # training loop computed each epoch were thrown away, which made every
                # alternative threshold rule a retrain; saved here they make it arithmetic.
                # §Lever B: keep the pooled embeddings, not just the logits. The head is
                # a discriminative boundary fitted to ~5 anomalies; the 167 normals are the
                # well-estimated side of the data and nothing downstream can use them once
                # the features are thrown away. Persisting them makes Mahalanobis / kNN /
                # score-fusion offline analysis (analysis/distribution_score.py) instead of
                # a retrain each. ~2 MB per category.
                _keep_feats = cfg.get("instrumentation", {}).get("save_features", True)
                # Both use the non-augmented test transform and shuffle=False.
                val_final = run_inference(model, val_loader, loss_fn, device, desc="val@best",
                                          capture_features=_keep_feats)
                recorder.save_split_scores(key, "val", val_final["logits"],
                                           val_final["labels"], val_final["paths"],
                                           features=val_final.get("features"))
                # CALIBRATION split (data.calib_split > 0): never trained on, never used for selection -- the
                # independent threshold source (analysis P7). Absent by default, so nothing is written for past configs.
                calib_recs = list(dataset.records("calib"))
                cal_loader = None
                if calib_recs:
                    cal_loader = make_loader(calib_recs, test_tf, cfg, shuffle=False)
                    cal_raw = run_inference(model, cal_loader, loss_fn, device, desc="calib@best",
                                            capture_features=_keep_feats)
                    recorder.save_split_scores(key, "calib", cal_raw["logits"], cal_raw["labels"],
                                               cal_raw["paths"], features=cal_raw.get("features"))
                # Train-normals are SS-2897's threshold source: a percentile of known-good
                # scores does not depend on the class prior, unlike a val-optimised cut.
                # Saved alongside val so the two sources can be compared offline -- train
                # normals were seen during training and score optimistically low, val
                # normals are honest but fewer.
                train_normal = [(p, y) for (p, y) in dataset.records("train") if int(y) == 0]
                if train_normal:
                    tn_loader = make_loader(train_normal, test_tf, cfg, shuffle=False)
                    tn_raw = run_inference(model, tn_loader, loss_fn, device,
                                             desc="train-normal",
                                             capture_features=_keep_feats)
                    recorder.save_split_scores(key, "train_normal", tn_raw["logits"],
                                               tn_raw["labels"], tn_raw["paths"],
                                               features=tn_raw.get("features"))

                test_raw = run_inference(model, test_loader, loss_fn, device, desc="test",
                                          capture_features=_keep_feats)
                test_metrics = compute_all(test_raw["labels"], test_raw["logits"],
                                           threshold=test_threshold,
                                           which=cfg["eval"]["metrics"])
                # LIKE-FOR-LIKE AUROC under Arm F v2 (F-L50).
                # The swap puts train/good images into test, and those are exactly the
                # normals this model finds easiest -- scoring them inflates AUROC. So also
                # score against ONLY the normals that came from test/good, which is the
                # provenance Run 0's normals had. Defects are untouched by the swap, so
                # `auroc_testgood_normals` is the number to compare with Run 0; the headline
                # `auroc` is the number for the de-confounded protocol itself. They answer
                # different questions and BOTH are reported.
                _dcx = getattr(dataset, "de_confound", None)
                if _dcx and _dcx.get("enabled"):
                    import numpy as _np
                    _lb = _np.asarray(test_raw["labels"]).ravel()
                    _lg = _np.asarray(test_raw["logits"]).ravel()
                    _pt = [str(x) for x in test_raw["paths"]]
                    _from_test = _np.array([(os.sep + "test" + os.sep) in q for q in _pt])
                    _keep = (_lb == 1) | ((_lb == 0) & _from_test)
                    if _keep.sum() and 0 < _lb[_keep].sum() < _keep.sum():
                        test_metrics["auroc_testgood_normals"] = safe_auroc(
                            _lb[_keep], _lg[_keep])
                        test_metrics["n_testgood_normals"] = int(((_lb == 0) & _from_test).sum())
                        test_metrics["n_traingood_normals"] = int(((_lb == 0) & ~_from_test).sum())
                        print(f"  like-for-like AUROC (test/good normals only, n="
                              f"{test_metrics['n_testgood_normals']}): "
                              f"{test_metrics['auroc_testgood_normals']:.4f}"
                              f"   | headline (all normals): {test_metrics['auroc']:.4f}")
                recorder.save_test_scores(key, test_raw["logits"],
                                          test_raw["labels"], test_raw["paths"],
                                          features=test_raw.get("features"))
                # G2 TTA, opt-in (`eval.tta=true`). Same selected model, scored per view on val
                # AND test into scores_{val,test}_tta.npz. The plain scores above are untouched,
                # so one training gives both arms, paired. RNG restored as for the firewall.
                if cfg.get("eval", {}).get("tta", False):
                    import numpy as _np
                    from engine.evaluate import run_inference_views
                    _cpu = torch.get_rng_state()
                    _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    for _sp, _ld in (("val", val_loader), ("test", test_loader)) + (
                            (("calib", cal_loader),) if cal_loader is not None else ()):
                        _r = run_inference_views(model, _ld, device, desc=f"tta-{_sp}")
                        _np.savez(os.path.join(run_dir, key, f"scores_{_sp}_tta.npz"),
                                  logits_views=_r["logits_views"], labels=_r["labels"],
                                  paths=_np.array(_r["paths"]), views=_np.array(_r["views"]))
                    torch.set_rng_state(_cpu)
                    if _cuda is not None:
                        torch.cuda.set_rng_state_all(_cuda)
                    print(f"  TTA: per-view scores saved for val and test ({len(_r['views'])} views)")
                eps = cfg["instrumentation"].get("collapse_gap_eps", 0.1)
                col = detect_collapse_from_file(
                    os.path.join(run_dir, key, "logit_hist.jsonl"), eps=eps,
                    selected_epoch=best["best_epoch"])
                if col["collapsed"]:
                    print(f"  *** COLLAPSE: {key} — {col['reason']}; AUROC is NOT a real result ***")
                cavs = caveats_from_flags(manifest.get("flags", {}))
                if inverted_flag(test_metrics.get("auroc")):
                    cavs = list(cavs) + ["inverted"]
                    print(f"  *** INVERTED: {key} — AUROC "
                          f"{test_metrics.get('auroc'):.4f} is BELOW CHANCE. The model "
                          f"ranks defects below normals; this is a sign inversion, not "
                          f"a weak result. ***")
                rel_tier = dataset_tier(dataset_name)
                _ft = getattr(fit, "timing", None) or {}
                _tot = _time.time() - _t_cat0
                _wait, _comp = _ft.get("data_wait_s", 0.0), _ft.get("compute_s", 0.0)
                _timing_payload = {
                    "category_seconds": round(_tot, 2),
                    "setup_seconds": round(_t_setup_end - _t_cat0, 2),
                    "train_seconds": round(_t_train_end - _t_setup_end, 2),
                    "eval_seconds": round(_time.time() - _t_train_end, 2),
                    "train_data_wait_seconds": _wait,
                    "train_compute_seconds": _comp,
                    "data_wait_fraction_of_train": (round(_wait / (_wait + _comp), 4)
                                                    if (_wait + _comp) > 0 else None),
                    "epochs_run": _ft.get("epochs", 0),
                    "note": "dataset-profiled to test-scored, one category, wall clock. "
                            "data_wait is time the train loop sat idle waiting for batches: "
                            "high -> caching decoded images would help, near zero -> it would not"}
                summary = {"test_metrics": test_metrics,
                           # carried INSIDE summary.json so instrumentation/report.py can
                           # print them. Live print() goes to stdout, which is interleaved
                           # with progress bars and is not what anyone reads.
                           "timing": _timing_payload,
                           # per-category data overrides that applied to THIS row (e.g.
                           # toothbrush's anomaly_fraction) -- a row produced under a
                           # different data setting must say so, or it gets paired wrongly
                           "data_overrides": data_cfg_for(cfg, dataset_name, category)[1],
                           "de_confound": (getattr(dataset, "de_confound", None) or {}),
                           "best_epoch": best["best_epoch"],
                           # SWA: when > 0 epochs were averaged, the scored checkpoint is the SWA model, not best_epoch
                           "swa": {"start": best.get("swa_start", 0), "epochs": best.get("swa_epochs", 0)},
                           # WHY it stopped, recorded not inferred (see engine.train)
                           "stop_reason": best.get("stop_reason"),
                           "epochs_ran": best.get("epochs_ran"),
                           "patience_setting": best.get("patience_setting"),
                           "num_epochs_setting": best.get("num_epochs_setting"),
                           "best_val_score": best["best_val_score"],
                           "monitor": best.get("monitor", "val_auroc"),
                           "selection_failed": best.get("selection_failed", False),
                           "collapsed": col["collapsed"],
                           "collapse_evidence": col,
                           "collapse_gap_eps": eps,
                           # §2.2/§2.4 reliability labelling travelling with the result
                           "augmentation": aug_desc,
                           "reliability_tier": rel_tier,
                           "caveats": cavs,
                           "reportable": is_reportable(cavs, col["collapsed"],
                                                       best.get("selection_failed", False)),
                           # Resolution has to travel WITH the result. The dossier's
                           # open question #1 is "what image_size did the clean baseline
                           # run at?" -- unanswerable because no artifact recorded it,
                           # and a number whose resolution is unknown is uncomparable.
                           "image_size": cfg["data"]["image_size"],
                           "model": {**model.summary(),
                                     **model_cost(model, cfg, device)}}
                recorder.save_summary(key, summary)
                # §2.5 auditable injection manifest (MVTec only; VisA is native)
                inj = getattr(dataset, "injection_manifest", None)
                if inj:
                    recorder.log_stage(key, "injection_manifest", inj)
                recorder.log_stage(key, "timing", _timing_payload)
                print(f"  timing: {_tot:.1f}s total | setup {_t_setup_end-_t_cat0:.1f}s | "
                      f"train {_t_train_end-_t_setup_end:.1f}s | eval {_time.time()-_t_train_end:.1f}s"
                      + (f" | data-wait {100*_wait/(_wait+_comp):.0f}% of train"
                         if (_wait + _comp) > 0 else ""))
                recorder.log_stage(key, "test", {"metrics": test_metrics})
                per_category[key] = dict(test_metrics)
                per_category[key]["_tier"] = rel_tier
                per_category[key]["_caveats"] = cavs
                per_category[key]["_reportable"] = summary["reportable"]

                if cavs:
                    print(f"  CAVEATS: {caveat_line(cavs)}")

                print(f"  TEST auroc={test_metrics['auroc']:.4f} "
                      f"aupr={test_metrics.get('aupr', float('nan')):.4f} "
                      f"f1={test_metrics['f1']:.4f} ece={test_metrics.get('ece', float('nan')):.4f}\n")

    # ---- run-level summary ----
    if per_category:
        # §2.4 cross-tier guard: means are computed PER TIER, never blended.
        # Only reportable results (not collapsed / not selection-failed / valid)
        # contribute to a tier mean; the rest still appear per-category.
        reportable = {k: v for k, v in per_category.items() if v.get("_reportable", True)}
        means_by_tier = tiered_means(reportable)
        _tier, _seeds = resolve_seeds(cfg)
        recorder.save_run_summary({"per_category": per_category,
                                   "means_by_tier": means_by_tier,
                                   "n_excluded_from_means":
                                       len(per_category) - len(reportable),
                                   "seed_tier": _tier, "seeds_available": _seeds,
                                   "protocol": protocol_fingerprint(cfg),
                                   "note": ("single-seed exploration (scouting, NOT a claim); "
                                            "use compare_backbones tier=claim for a noise band"
                                            if len(_seeds) < 2 or _tier == "exploration"
                                            else f"{_tier} tier")})
        report = run_report(run_dir)
        with open(os.path.join(run_dir, "report.txt"), "w") as f:
            f.write(report)
        print(report)
    try:
        with open(os.path.join(run_dir, "what_runs.txt"), "w") as _f:
            _f.write(_state + "\n\noverrides: " + " ".join(sys.argv[1:]) + "\n")
    except Exception:
        pass
    print(f"\nAll artifacts in: {run_dir}")
    # Bundle the moment the run ends, with no intervention. Kaggle wipes
    # /kaggle/working when the GPU is toggled; that already destroyed the first
    # Arm F v2 subset, carpet included, whose per-image scores cannot be
    # recomputed. Checkpoints are excluded -- they are 88% of the bytes and
    # nothing downstream reads them.
    try:
        from utils.persist import bundle_run, describe
        print(describe(bundle_run(run_dir, extra_note=" ".join(sys.argv[1:]) or None)))
    except Exception as _e:
        print(f"  BUNDLE FAILED ({type(_e).__name__}: {_e}) -- download {run_dir} by hand")
    # Full weights only when asked (O-19): true = every seed, a list = those seeds only.
    _sw = (cfg.get("run") or {}).get("save_weights", False)
    if _sw is True or (isinstance(_sw, list) and int(cfg["seed"]) in [int(s) for s in _sw]):
        try:
            from utils.persist import save_weights
            _w = save_weights(run_dir)
            print(f"          WEIGHTS: {_w['checkpoints']} checkpoints, {_w['mb']} MB -> {_w['zip']}"
                  "  (separate download; config + git commit inside)")
        except Exception as _e:
            print(f"  WEIGHTS FAILED ({type(_e).__name__}: {_e}) -- the .pt files are still in {run_dir}")




class _KeyedRecorder:
    """Adapts the Recorder to the train loop's (category, ...) hook signature,
    passing the full dataset/backbone/category key as the 'category'."""
    def __init__(self, rec, key):
        self.rec, self.key = rec, key
    def log_epoch(self, _cat, epoch, payload):
        self.rec.log_epoch(self.key, epoch, payload)
    def log_grad_norms(self, _cat, epoch, grad):
        self.rec.log_grad_norms(self.key, epoch, grad)
    def log_logit_hist(self, _cat, epoch, logits, labels, bins=30):
        self.rec.log_logit_hist(self.key, epoch, logits, labels, bins=bins)
    def log_stage(self, _cat, name, payload):
        self.rec.log_stage(self.key, name, payload)


if __name__ == "__main__":
    main()
