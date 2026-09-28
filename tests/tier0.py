#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier 0 regression suite — must pass before any experiment. No GPU, no dataset.

Stability condition 6 (STABILITY_AUDIT.md). Every check has a NEGATIVE CONTROL: a case
that must FAIL, so a check that passes everything is caught as broken. Several past
checkers passed vacuously (Gate 0B), and one drift checker read 0.0 for everything
until an FP16 control proved it could see a difference.

  T1 config scan        shadowed blocks / dead keys are exactly the known set
  T2 splits             REAL MVTecDataset code on synthetic trees built from Run 0's
                        recorded file counts: default config reproduces Run 0's splits for
                        all 15; baseline keeps every test set's size and prevalence;
                        toothbrush gets 6/6; no other category's defect draw moves
  T3 augmentation leak  val/test transforms contain no random op; train gets exactly the
                        documented augment ops, for every one of 27 categories
  T4 Run 0 rebuilt      MVTec 0.8864 / VisA 0.9782 recomputed from saved summaries
  T5 save list          a bundle keeps scores and excludes checkpoints
  T6 bootstrap          a run compared with ITSELF gives a delta of exactly 0

    python tests/tier0.py --run0 /path/to/run0_tree
"""
import argparse, glob, json, os, shutil, statistics, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + str(detail) if detail else ''}")


def cfg_of(name, overrides=()):
    from utils.io import load_config, apply_overrides
    c = load_config(os.path.join(ROOT, "configs", name))
    return apply_overrides(c, list(overrides)) if overrides else c


# ------------------------------------------------------------------ T1
def t1():
    print("\nT1 config scan")
    from analysis.what_runs import cross_check_config_against_code
    dead, shadowed, missing = cross_check_config_against_code(cfg_of("default.yaml"), ROOT)
    known_sh = {("augment", "augmentation"), ("seed", "seeds"), ("seed", "seeding")}
    check("no NEW shadowed blocks", set(shadowed) <= known_sh, sorted(set(shadowed) - known_sh))
    known_dead = {"instrumentation.level", "instrumentation.capture_activations",
                  "instrumentation.save_raw_scores"}
    check("no NEW dead config keys", set(dead) <= known_dead, sorted(set(dead) - known_dead))
    # NEGATIVE CONTROL: an injected dead key must be found
    c = cfg_of("default.yaml"); c.setdefault("data", {})["zz_never_read_by_anything"] = 1
    d2, _, _ = cross_check_config_against_code(c, ROOT)
    check("CONTROL: injected dead key is detected", "data.zz_never_read_by_anything" in d2)


# ------------------------------------------------------------------ T2
def _manifests(run0):
    out = {}
    for f in sorted(glob.glob(os.path.join(run0, "mvtec/efficientnet/*/stage.jsonl"))):
        cat = f.split(os.sep)[-2]; man = sp = None
        for line in open(f):
            d = json.loads(line)
            if d.get("stage") == "injection_manifest": man = d
            if d.get("stage") == "dataset": sp = d["splits"]
        out[cat] = (man, sp)
    return out


def _fake_tree(root, cat, man, sp):
    """A directory tree with EXACTLY the recorded file counts for one category."""
    n_good_train = sp["train"]["normal"] + sp["val"]["normal"]
    n_good_test = sp["test"]["normal"]
    for sub, n in [("train/good", n_good_train), ("test/good", n_good_test)]:
        os.makedirs(os.path.join(root, cat, sub), exist_ok=True)
        for i in range(n):
            open(os.path.join(root, cat, sub, f"{i:03d}.png"), "w").close()
    for t, n in man["per_type_available"].items():
        os.makedirs(os.path.join(root, cat, "test", t), exist_ok=True)
        for i in range(n):
            open(os.path.join(root, cat, "test", t, f"{i:03d}.png"), "w").close()


def _counts(ds):
    f = lambda r: (sum(1 for _, l in r if l == 0), sum(1 for _, l in r if l == 1))
    return f(ds.train_data), f(ds.val_data), f(ds.test_data)


def t2(run0):
    print("\nT2 splits (real dataset code, synthetic trees with Run 0's file counts)")
    if not run0 or not os.path.isdir(run0):
        check("Run 0 tree available", False, run0); return
    import run_train
    from data.mvtec import MVTecDataset
    mans = _manifests(run0)
    tmp = tempfile.mkdtemp()
    try:
        for cat, (man, sp) in mans.items():
            _fake_tree(tmp, cat, man, sp)
        MVTecDataset.categories = list(mans)
        base_d = cfg_of("default.yaml", ["data.image_size=288", "data.cache_decoded=false"])
        base_b = cfg_of("baseline.yaml", ["data.cache_decoded=false"])
        base_d["paths"]["mvtec_root"] = base_b["paths"]["mvtec_root"] = tmp
        ok_r0 = ok_size = ok_draw = True; bad = []
        for cat, (man, sp) in mans.items():
            d0 = run_train.build_dataset("mvtec", base_d, cat)
            got = _counts(d0)
            want = ((sp["train"]["normal"], sp["train"]["anomaly"]),
                    (sp["val"]["normal"], sp["val"]["anomaly"]),
                    (sp["test"]["normal"], sp["test"]["anomaly"]))
            if got != want: ok_r0 = False; bad.append((cat, got, want))
            db = run_train.build_dataset("mvtec", base_b, cat)
            if cat != "toothbrush":
                if _counts(db)[2] != want[2]: ok_size = False; bad.append((cat, "test size"))
                dz = sorted(p for p, l in d0.train_data + d0.val_data if l == 1)
                bz = sorted(p for p, l in db.train_data + db.val_data if l == 1)
                if dz != bz: ok_draw = False; bad.append((cat, "defect draw moved"))
        check("default config reproduces Run 0's train/val/test counts, all 15", ok_r0,
              bad[:2] if not ok_r0 else "")
        check("baseline preserves every test set's size and prevalence (14, excl. toothbrush)",
              ok_size)
        check("baseline leaves every other category's injected defects unchanged", ok_draw)
        tb = run_train.build_dataset("mvtec", base_b, "toothbrush")
        (trn, tra), (van, vaa), (ten, tea) = _counts(tb)
        check("toothbrush gets 6 train / 6 val defects under the baseline", (tra, vaa) == (6, 6),
              f"train {tra}, val {vaa}, test {ten}+{tea}")
        # NEGATIVE CONTROL: with the override removed, toothbrush must fall back to 2/4
        nb = cfg_of("baseline.yaml", ["data.cache_decoded=false"])
        nb["paths"]["mvtec_root"] = tmp; nb["data"]["per_category"] = {}
        _, (_, a2), _ = _counts(run_train.build_dataset("mvtec", nb, "toothbrush"))[0:1] + ((0, 0),)*2
        tr2 = _counts(run_train.build_dataset("mvtec", nb, "toothbrush"))[0][1]
        check("CONTROL: without the override toothbrush is back to 2 train defects", tr2 == 2,
              f"got {tr2}")
        # F-L79 split ratio: the default must be baseline-v1 exactly; 0.5 moves only normals to val.
        paths = lambda ds: [sorted(p for p, _ in r) for r in (ds.train_data, ds.val_data, ds.test_data)]
        c = "carpet"; d_def = run_train.build_dataset("mvtec", base_b, c)
        b3 = cfg_of("baseline.yaml", ["data.cache_decoded=false", "data.de_confound_val_share=0.3333333333333333"])
        b5 = cfg_of("baseline.yaml", ["data.cache_decoded=false", "data.de_confound_val_share=0.5"])
        b3["paths"]["mvtec_root"] = b5["paths"]["mvtec_root"] = tmp
        d3 = run_train.build_dataset("mvtec", b3, c); d5 = run_train.build_dataset("mvtec", b5, c)
        check("val_share 1/3 reproduces baseline-v1's split exactly (carpet)", paths(d3) == paths(d_def))
        k = d5.de_confound["k"]
        check("val_share 0.5: round(k/2) test/good normals go to val, test set and defects unchanged",
              d5.de_confound["n_testgood_to_val"] == int(round(k * 0.5)) and paths(d5)[2] == paths(d_def)[2]
              and sorted(p for p, l in d5.train_data + d5.val_data if l == 1)
              == sorted(p for p, l in d_def.train_data + d_def.val_data if l == 1),
              f"k={k}, to val {d5.de_confound['n_testgood_to_val']}")
        check("CONTROL: val_share 0.5 changes the train/val split", paths(d5)[:2] != paths(d_def)[:2])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ T3
def t3():
    print("\nT3 augmentation leak")
    from utils.transforms import build_transforms, build_scenario_transforms
    from data.mvtec import MVTecDataset
    from data.visa import ViSADataset
    c = cfg_of("baseline.yaml")
    te = [type(t).__name__ for t in build_transforms(c, train=False)[0].transforms]
    rnd = lambda ops: [o for o in ops if o.startswith("Random") or "Jitter" in o]
    check("val/test transform has no random op", not rnd(te), te)
    want = ["Resize", "RandomHorizontalFlip", "RandomRotation", "ColorJitter", "ToTensor", "Normalize"]
    cats = list(MVTecDataset.categories) + list(ViSADataset.categories)
    same = all([type(t).__name__ for t in build_scenario_transforms(c, k)[0].transforms] == want
               for k in cats)
    check(f"train transform is exactly the documented augment ops, all {len(cats)} categories", same)
    # NEGATIVE CONTROL: the leak detector must fire if a random op is put in eval
    import torchvision.transforms as T
    check("CONTROL: leak detector fires on a random op", bool(rnd(["Resize", type(T.RandomHorizontalFlip()).__name__])))


# ------------------------------------------------------------------ T4
def t4(run0):
    print("\nT4 Run 0 means rebuilt from saved summaries")
    if not run0 or not os.path.isdir(run0):
        check("Run 0 tree available", False); return
    m = {"mvtec": [], "visa": []}
    for f in glob.glob(os.path.join(run0, "*/efficientnet/*/summary.json")):
        ds = f.split(os.sep)[-4]
        m[ds].append(json.load(open(f))["test_metrics"]["auroc"])
    mv, vs = statistics.mean(m["mvtec"]), statistics.mean(m["visa"])
    check("MVTec mean 0.8864 (n=15)", round(mv, 4) == 0.8864 and len(m["mvtec"]) == 15, f"{mv:.4f}")
    check("VisA mean 0.9782 (n=12)", round(vs, 4) == 0.9782 and len(m["visa"]) == 12, f"{vs:.4f}")
    check("CONTROL: a wrong expected value is rejected", round(mv, 4) != 0.8865)


# ------------------------------------------------------------------ T5
def t5():
    print("\nT5 save list")
    import tarfile, utils.persist as P
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "tag", "run1", "mvtec", "efficientnet", "wood"); os.makedirs(d)
        for f in ("summary.json", "stage.jsonl", "scores_test.npz", "scores_val.npz",
                  "scores_train_normal.npz", "best.pt"):
            open(os.path.join(d, f), "w").write("x")
        info = P.bundle_run(os.path.join(tmp, "tag", "run1"))
        names = tarfile.open(info["targets"][0]["path"]).getnames()
        for k in ("scores_test.npz", "scores_val.npz", "scores_train_normal.npz", "summary.json"):
            check(f"bundle keeps {k}", any(n.endswith(k) for n in names))
        check("CONTROL: bundle EXCLUDES the checkpoint", not any(n.endswith(".pt") for n in names))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ T6
def _boot(a_dir, b_dir, out):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis/bootstrap_ci.py"), a_dir,
                        "--vs", b_dir, "--n", "2000", "--json", out],
                       capture_output=True, text=True, timeout=900)
    return r, (json.load(open(out)) if os.path.exists(out) else None)


def t6(run0):
    print("\nT6 paired bootstrap: negative AND positive control")
    if not run0 or not os.path.isdir(run0):
        check("Run 0 tree available", False); return
    tmp = tempfile.mkdtemp()
    try:
        r, j = _boot(run0, run0, os.path.join(tmp, "self.json"))
        check("bootstrap_ci --vs ran", j is not None, (r.stderr or r.stdout)[-160:] if j is None else "")
        if j is None:
            return
        rows = j["rows"]
        check(f"NEGATIVE: a run against ITSELF -> every delta and bound exactly 0 ({len(rows)} rows)",
              rows and all(x["delta"] == 0 and x["lo"] == 0 and x["hi"] == 0 for x in rows))
        # POSITIVE CONTROL: a KNOWN improvement must be detected. Copy the score files and
        # raise every anomaly logit by 3.0 -- ranking can only improve -- so each row must
        # resolve BETTER with an interval excluding zero. A paired test that cannot see a
        # planted effect would pass the negative control and still be useless.
        import numpy as np
        b = os.path.join(tmp, "shifted")
        for f in glob.glob(os.path.join(run0, "*/efficientnet/*/scores_test.npz")):
            rel = os.path.relpath(os.path.dirname(f), run0)
            os.makedirs(os.path.join(b, rel), exist_ok=True)
            z = np.load(f, allow_pickle=True)
            lg = np.asarray(z["logits"], dtype=float).copy(); lb = np.asarray(z["labels"]).ravel()
            lg.ravel()[lb == 1] += 3.0
            np.savez(os.path.join(b, rel, "scores_test.npz"), logits=lg, labels=z["labels"],
                     paths=z["paths"])
        r2, j2 = _boot(run0, b, os.path.join(tmp, "pos.json"))
        rows2 = (j2 or {}).get("rows", [])
        up = [x for x in rows2 if x["b"] > x["a"] + 1e-12]   # rows where the shift CAN help
        check(f"POSITIVE: a planted +3 logit shift on defects is detected ({len(up)} improvable rows)",
              up and all(x["resolved"] and x["lo"] > 0 for x in up),
              [(x["category"].split("/")[-1], round(x["lo"], 4)) for x in up])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


BASELINE_HASH = "3669cc0a7c96d9e1"   # resolved configs/baseline.yaml, paths excluded
# 2026-09-27 (F-L129): was 5f2e4c12a8f24001; changed ONLY by adding data.transplant.donor_rule=all (v1/v2 unchanged, T30).
# 2026-09-27 (F-L127): was f40b2f8e5f58f1ca; changed ONLY by adding data.transplant.placement=random and ring_px=8
# (transplant stays off by default; placement random = v1 exactly, T28/T29).
# 2026-09-27 (F-L124): was 23016e59bbb3824d; changed ONLY by adding data.transplant {enabled: false, target_defects: 48,
# feather_radius: 2.0}. Off by default = training records unchanged (T28).
# 2026-09-27 (F-L122): was 3abb99bc69e19c79; changed ONLY by adding model.score_mode=image and model.patch_topk_frac=0.1
# to default.yaml. score_mode=image is the original forward bit-for-bit (T27), so baseline behaviour is unchanged.


def _baseline_hash():
    import hashlib
    c = cfg_of("baseline.yaml"); c.pop("paths", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]


def t7():
    """The resolved baseline must not change by accident. Any change to baseline.yaml OR to
    default.yaml values it inherits alters this hash; a deliberate change updates the constant
    here in the same commit, which makes it visible in review."""
    print("\nT7 baseline config hash")
    h = _baseline_hash()
    check("resolved baseline config matches the frozen hash", h == BASELINE_HASH,
          f"got {h}, frozen {BASELINE_HASH}")
    import hashlib
    c = cfg_of("baseline.yaml"); c.pop("paths", None); c["train"]["patience"] = 11
    h2 = hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]
    check("CONTROL: changing one value (patience 10 -> 11) changes the hash", h2 != h)


def t8():
    """The firewalled per-epoch test AUROC (O-06) must be logged AND must not change training.
    A tiny CPU model with a shuffled loader and random input noise stands in for the real run,
    so any extra draw from torch's RNG would change the trajectory."""
    print("\nT8 firewalled test logging leaves training unchanged")
    import tempfile
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from engine.train import fit

    class Rec:
        def __init__(self): self.ep = []
        def log_epoch(self, c, e, d): self.ep.append(d)
        def log_stage(self, *a, **k): pass
        def log_grad_norms(self, *a, **k): pass
        def log_logit_hist(self, *a, **k): pass

    class Noisy(TensorDataset):      # random per-item noise from the global RNG, like augmentation
        def __getitem__(self, i):
            x, y = super().__getitem__(i)
            return x + 0.3 * torch.randn_like(x), y

    def run(firewall):
        torch.manual_seed(0)
        X = torch.randn(64, 8); y = (X[:, 0] + 0.5 * torch.randn(64) > 0).long()
        tr = DataLoader(Noisy(X[:32], y[:32]), batch_size=8, shuffle=True)
        va = DataLoader(TensorDataset(X[32:48], y[32:48]), batch_size=8)
        te = DataLoader(TensorDataset(X[48:], y[48:]), batch_size=8)
        torch.manual_seed(1)
        m = torch.nn.Linear(8, 1)
        cfg = {"train": {"num_epochs": 4, "patience": 10}, "selection": {"monitor": "val_auroc"},
               "eval": {"metrics": ["auroc"], "firewalled_test_per_epoch": firewall},
               "instrumentation": {"capture_grad_norms": False, "capture_logit_hist": False}}
        rec = Rec()
        with tempfile.TemporaryDirectory() as d:
            fit(m, tr, va, torch.optim.SGD(m.parameters(), lr=0.1),
                torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), cfg, d, "t8",
                recorder=rec, test_loader=te)
        return m.weight.detach().clone(), rec.ep

    w0, e0 = run(False)
    w1, e1 = run(True)
    check("firewall ON logs a test AUROC every epoch",
          all(e.get("firewalled_test_auroc") is not None for e in e1), f"{len(e1)} epochs")
    check("firewall OFF logs none", all(e.get("firewalled_test_auroc") is None for e in e0))
    check("firewall ON: final weights identical to OFF", torch.equal(w0, w1))
    check("firewall ON: val trajectory identical to OFF",
          [e["val_loss"] for e in e0] == [e["val_loss"] for e in e1])
    # NEGATIVE CONTROL: iterating a loader DOES consume the global RNG, so without the
    # save/restore the trajectory would change -- the restore is doing real work.
    torch.manual_seed(5); a = torch.rand(1)
    torch.manual_seed(5); list(DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)); b = torch.rand(1)
    check("CONTROL: iterating a DataLoader moves the global RNG (so the restore is needed)",
          not torch.equal(a, b))


def t9():
    """The runtime leak guard must check val against BOTH train and test, and must fire."""
    print("\nT9 leak guard covers train/val/test pairwise")
    from analysis.guardrails import leak_self_check, LeakCheckError
    tf = object()
    base = dict(val_threshold=0.0, test_threshold=0.0, val_transform=tf, test_transform=tf,
                use_validation=True, key="t9")
    def fires(**kw):
        try:
            leak_self_check(**base, **kw); return False
        except LeakCheckError:
            return True
    tr, va, te = ["a/train/good/1.png"], ["a/train/good/2.png"], ["a/test/good/3.png"]
    check("clean train/val/test passes", not fires(train_paths=tr, test_paths=te, val_paths=va))
    check("CONTROL: val sharing a TEST image fires", fires(train_paths=tr, test_paths=te, val_paths=va + te))
    check("CONTROL: val sharing a TRAIN image fires", fires(train_paths=tr, test_paths=te, val_paths=va + tr))
    check("CONTROL: train sharing a test image fires", fires(train_paths=tr + te, test_paths=te, val_paths=va))
    # Defects: the guard compares paths regardless of label, so a defect image counts the same.
    dtr, dva, dte = ["a/test/crack/0.png"], ["a/test/crack/1.png"], ["a/test/crack/2.png"]
    check("CONTROL: a train DEFECT also in test fires", fires(train_paths=tr + dtr, test_paths=te + dtr, val_paths=va))
    check("CONTROL: a val DEFECT also in test fires", fires(train_paths=tr, test_paths=te + dte, val_paths=va + dte))
    check("clean with defects in all three passes", not fires(train_paths=tr + dtr, test_paths=te + dte, val_paths=va + dva))


def t10():
    """O-19: weights go to their own zip with config + commit; the results bundle still excludes them."""
    print("\nT10 weight export")
    import tempfile, zipfile, tarfile
    import utils.persist as P
    from utils.persist import save_weights, bundle_run
    # ISOLATE from the real output folder. On Kaggle WORK exists, and without this the fake
    # run was bundled into the real ALL_RESULTS.zip and INDEX (found 2026-09-22).
    _saved = (P.WORK, P.MIRROR, P.ROLLING_ZIP)
    P.WORK = P.MIRROR = P.ROLLING_ZIP = "/nonexistent/tier0_isolated"
    try:
        _t10_body(tempfile, zipfile, tarfile, save_weights, bundle_run)
    finally:
        P.WORK, P.MIRROR, P.ROLLING_ZIP = _saved
    check("CONTROL: persist paths restored after T10", P.WORK == _saved[0])


def _t10_body(tempfile, zipfile, tarfile, save_weights, bundle_run):
    with tempfile.TemporaryDirectory() as d:
        rd = os.path.join(d, "claim", "20260925-000000_x"); cd = os.path.join(rd, "mvtec", "efficientnet", "wood")
        os.makedirs(cd)
        open(os.path.join(rd, "mvtec_efficientnet_wood_best_model.pt"), "wb").write(b"\0" * 1000)
        open(os.path.join(cd, "summary.json"), "w").write("{}")
        open(os.path.join(rd, "what_runs.txt"), "w").write("code: git abc")
        w = save_weights(rd)
        names = zipfile.ZipFile(w["zip"]).namelist()
        check("weights zip holds the checkpoint", any(n.endswith(".pt") for n in names), str(names))
        check("weights zip holds what_runs.txt and summary.json",
              any(n.endswith("what_runs.txt") for n in names) and any(n.endswith("summary.json") for n in names))
        b = bundle_run(rd)
        tb = [t["path"] for t in b["targets"] if "error" not in t][0]
        check("CONTROL: the results bundle still excludes .pt",
              not any(m.endswith(".pt") for m in tarfile.open(tb).getnames()) and b["excluded_checkpoints"] >= 1)


def _toy_fit(ema_epochs):
    import tempfile
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from engine.train import fit
    class Rec:
        def __init__(self): self.ep = []
        def log_epoch(self, c, e, d): self.ep.append(d)
        def log_stage(self, *a, **k): pass
        def log_grad_norms(self, *a, **k): pass
        def log_logit_hist(self, *a, **k): pass
    torch.manual_seed(0)
    X = torch.randn(64, 8); y = (X[:, 0] + 0.5 * torch.randn(64) > 0).long()
    tr = DataLoader(TensorDataset(X[:32], y[:32]), batch_size=8, shuffle=True)
    va = DataLoader(TensorDataset(X[32:48], y[32:48]), batch_size=8)
    torch.manual_seed(1)
    m = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.BatchNorm1d(4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    cfg = {"train": {"num_epochs": 5, "patience": 10, "ema_epochs": ema_epochs},
           "selection": {"monitor": "val_auroc"}, "eval": {"metrics": ["auroc"]},
           "instrumentation": {"capture_grad_norms": False, "capture_logit_hist": False}}
    rec = Rec()
    with tempfile.TemporaryDirectory() as d:
        best = fit(m, tr, va, torch.optim.SGD(m.parameters(), lr=0.1), torch.nn.BCEWithLogitsLoss(),
                   torch.device("cpu"), cfg, d, "t11", recorder=rec)
        ck = torch.load(best["checkpoint"])
    return {k: v.clone() for k, v in m.state_dict().items()}, ck, rec.ep


def t11():
    """G1 EMA: training trajectory untouched; checkpoint and val use averaged weights incl. BN stats."""
    print("\nT11 weight EMA")
    import torch
    from engine.train import WeightEMA
    raw0, ck0, e0 = _toy_fit(0)
    raw1, ck1, e1 = _toy_fit(2)
    check("EMA on: RAW training weights identical to EMA off (swap/restore is exact)",
          all(torch.equal(raw0[k], raw1[k]) for k in raw0))
    check("CONTROL: EMA on: saved checkpoint differs from the no-EMA checkpoint",
          any(not torch.equal(ck0[k], ck1[k]) for k in ck0 if ck0[k].dtype.is_floating_point))
    check("EMA on: val loss comes from the averaged weights (differs from EMA off)",
          [e["val_loss"] for e in e0] != [e["val_loss"] for e in e1])
    check("EMA averages BatchNorm running stats too",
          not torch.equal(ck1["1.running_mean"], raw1["1.running_mean"]))
    # math: first update uses warm-up decay (1+1)/(10+1)
    m = torch.nn.Linear(1, 1, bias=False); torch.nn.init.constant_(m.weight, 0.0)
    e = WeightEMA(m, 100, 100); torch.nn.init.constant_(m.weight, 1.0); e.update(m)
    d = 2 / 11
    check("EMA update math (warm-up decay)", abs(float(e.state["weight"]) - (1 - d)) < 1e-6,
          f"{float(e.state['weight']):.6f} vs {1 - d:.6f}")


def t12():
    """G2 TTA views: id is the plain forward; the others are really different inputs."""
    print("\nT12 TTA views")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from engine.evaluate import run_inference_views, _view
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                            torch.nn.Linear(2, 1)).eval()
    X = torch.randn(6, 3, 16, 16); y = torch.tensor([0, 1, 0, 1, 0, 1])
    r = run_inference_views(m, DataLoader(TensorDataset(X, y), batch_size=4), torch.device("cpu"))
    with torch.no_grad():
        plain = m(X).view(-1).numpy()
    check("TTA 'id' view equals the plain forward", abs(r["logits_views"][:, 0] - plain).max() < 1e-6)
    check("CONTROL: hflip and rotations change the logits",
          all(abs(r["logits_views"][:, j] - plain).max() > 1e-6 for j in (1, 2, 3)))
    check("hflip applied twice is the identity", torch.equal(_view(_view(X, "hflip"), "hflip"), X))


def t13():
    """F-L82 tap variants: zero-start gate starts as the baseline; LayerNorm option rebalances branches."""
    print("\nT13 tap gate / norm")
    import torch
    from models.backbones import build_backbone
    torch.manual_seed(0)
    base = build_backbone("efficientnet", pretrained=False)
    g = build_backbone("efficientnet", pretrained=False, taps=["4", "8"], image_size=64, tap_gate=True)
    n = build_backbone("efficientnet", pretrained=False, taps=["4", "8"], image_size=64, tap_norm=True)
    g.features.load_state_dict(base.features.state_dict()); n.features.load_state_dict(base.features.state_dict())
    for m in (base, g, n): m.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ob, og, on = base(x).flatten(1), g(x), n(x)
    check("gate at init: tap dims are exactly zero", bool((og[:, :80] == 0).all()))
    check("gate at init: deep dims equal the baseline backbone", torch.allclose(og[:, 80:], ob))
    g.train(); loss = g(x)[:, 80:].sum() + (g(x)[:, :80] * torch.randn(2, 80)).sum(); loss.backward()
    check("CONTROL: the gate receives a gradient (it can open)", g.pool.gates.grad is not None and g.pool.gates.grad.abs().sum() > 0)
    # Random-init EfficientNet features are ~1e-8 (degenerate), so test the pooling layer directly
    # on maps with a KNOWN 40x imbalance, like the trained model's (F-L82).
    from models.backbones import TapPool
    tp = TapPool("avg", [80, 1280], tap_norm=True).eval()
    maps = [torch.randn(4, 80, 18, 18) * 4.0 + torch.randn(4, 80, 1, 1) * 4.0,
            torch.randn(4, 1280, 9, 9) * 0.1 + torch.randn(4, 1280, 1, 1) * 0.1]
    with torch.no_grad():
        o = tp(maps)
    check("norm: each branch comes out at unit per-dim scale from a 40x imbalance",
          abs(o[:, :80].std().item() - 1) < 0.15 and abs(o[:, 80:].std().item() - 1) < 0.15,
          f"tap {o[:, :80].std().item():.2f} deep {o[:, 80:].std().item():.2f}")
    raw = TapPool("avg", [80, 1280]).eval()
    with torch.no_grad():
        r = raw(maps)
    check("CONTROL: without norm the imbalance survives (>10x)", r[:, :80].std().item() / r[:, 80:].std().item() > 10)
    d = build_backbone("efficientnet", pretrained=False, taps=["4", "8"], image_size=64)
    check("CONTROL: without the options neither gates nor norms exist", d.pool.gates is None and d.pool.norms is None)
    mp = TapPool("avg", [80, 1280], tap_pool="max").eval()
    m2 = [torch.zeros(1, 80, 4, 4), torch.zeros(1, 1280, 2, 2)]; m2[0][0, :, 0, 0] = 16.0; m2[1][0, :, 0, 0] = 4.0
    with torch.no_grad():
        o2 = mp(m2)
    check("tap_pool=max: tap branch keeps the peak (16), deep branch still averages (1)",
          abs(o2[0, :80].mean().item() - 16) < 1e-5 and abs(o2[0, 80:].mean().item() - 1) < 1e-5,
          f"tap {o2[0, :80].mean().item():.2f} deep {o2[0, 80:].mean().item():.2f}")


def t14():
    """Minimum-epochs guard: patience cannot stop training before train.min_epochs."""
    print("\nT14 minimum-epochs guard")
    import tempfile, torch
    from torch.utils.data import DataLoader, TensorDataset
    from engine.train import fit
    class Rec:
        def __init__(self): self.n = 0
        def log_epoch(self, *a, **k): self.n += 1
        def log_stage(self, *a, **k): pass
        def log_grad_norms(self, *a, **k): pass
        def log_logit_hist(self, *a, **k): pass
    def run(min_ep):
        torch.manual_seed(0)
        X = torch.randn(32, 4); y = (X[:, 0] > 0).long()
        m = torch.nn.Linear(4, 1)
        cfg = {"train": {"num_epochs": 8, "patience": 1, "min_epochs": min_ep},
               "selection": {"monitor": "val_auroc", "tiebreak": "none"}, "eval": {"metrics": ["auroc"]},
               "instrumentation": {"capture_grad_norms": False, "capture_logit_hist": False}}
        r = Rec()
        with tempfile.TemporaryDirectory() as d:     # lr=0 -> val never improves -> patience fires early
            fit(m, DataLoader(TensorDataset(X[:16], y[:16]), batch_size=8), DataLoader(TensorDataset(X[16:], y[16:]), batch_size=8),
                torch.optim.SGD(m.parameters(), lr=0.0), torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), cfg, d, "t14", recorder=r)
        return r.n
    n0, n5 = run(0), run(5)
    check("CONTROL: without the guard patience stops early", n0 < 5, f"{n0} epochs")
    check("with min_epochs=5 training runs at least 5 epochs", n5 >= 5, f"{n5} epochs")


def t15():
    """CP-001 / A-001: class_weight modes. The default must stay the historical neg/(neg+pos) so every past run
    reproduces; neg_over_pos must be the balancing weight; an unknown mode must raise instead of silently defaulting."""
    print("\nT15 class weight modes (CP-001)")
    import torch
    import run_train
    class DS:
        def __init__(s, n0, n1): s.l = [0] * n0 + [1] * n1
        def labels(s, split): return s.l
    dev = torch.device("cpu"); n0, n1 = 166, 6
    w = lambda extra: float(run_train.class_weight(DS(n0, n1), {"train": {"use_class_weights": True, **extra}}, dev))
    ok = lambda got, want: abs(got - want) < 1e-6 * max(1.0, want)
    check("default (key absent) = neg/(neg+pos), the historical formula", ok(w({}), n0 / (n0 + n1)), f"{w({}):.6f}")
    check("neg_over_pos = neg/pos", ok(w({"class_weight_mode": "neg_over_pos"}), n0 / n1))
    try:
        w({"class_weight_mode": "neg_over_pso"}); raised = False
    except ValueError:
        raised = True
    check("an unknown mode raises (no silent default)", raised)
    # NEGATIVE CONTROL: the comparison used above must reject the other mode's value
    check("CONTROL: the default check rejects the neg/pos value", not ok(w({}), n0 / n1))


def t16():
    """A-038: pool_lr_scope. Default 'all' must keep the historical grouping (tap runs reproduce); 'exponent' must put
    only pooling exponents on the multiplied lr, so the zero-start gate trains at the base lr. baseline-v1 (avg pooling)
    has no pooling parameters, so both scopes must give it one identical group."""
    print("\nT16 pooling lr scope (A-038)")
    import run_train
    from models.model_factory import build_model
    def groups(overrides):
        c = cfg_of("baseline.yaml", ["model.pretrained=false", *overrides])
        opt = run_train.build_optimizer(build_model(c, "efficientnet"), c)
        return [(g["lr"], len(g["params"]), sorted(tuple(p.shape) for p in g["params"])[:2]) for g in opt.param_groups]
    lr = cfg_of("baseline.yaml")["train"]["learning_rate"]
    gate_lr = lambda gs: [g[0] for g in gs if (1,) in g[2] and g[1] == 1]
    b_all, b_exp = groups([]), groups(["train.pool_lr_scope=exponent"])
    check("baseline-v1: one param group under both scopes (unaffected)", len(b_all) == 1 and b_all == b_exp)
    g_all = groups(["model.taps=[5,8]", "model.tap_gate=true"])
    g_exp = groups(["model.taps=[5,8]", "model.tap_gate=true", "train.pool_lr_scope=exponent"])
    check("default scope keeps the historical grouping (gate at lr x mult)", gate_lr(g_all) == [lr * 50.0], g_all[-1][:2])
    check("scope=exponent: a single group at the base lr (gate not multiplied)", len(g_exp) == 1 and g_exp[0][0] == lr)
    l_exp = groups(["model.pooling=lse", "train.pool_lr_scope=exponent"])
    check("scope=exponent still multiplies the LSE exponent", any(g[0] == lr * 50.0 and g[1] == 1 for g in l_exp))
    # NEGATIVE CONTROL: the gate-lr detector must see the multiplied gate in the historical grouping
    check("CONTROL: the gate-lr detector fires on the historical grouping", gate_lr(g_all) != [lr])


def t17(run0):
    """Calibration split (analysis P7/P4): calib_split = 0 must leave every split bit-identical; > 0 must move ONLY
    training normals into calib (val, test and the defect draw unchanged), with calib disjoint from all three;
    the leak guard must fire if a calib path is also in train. MVTec on a synthetic tree, VisA on a synthetic CSV."""
    print("\nT17 calibration split")
    import csv as _csv
    import run_train
    from data.mvtec import MVTecDataset
    from analysis.guardrails import leak_self_check, LeakCheckError
    paths = lambda ds, m: sorted(p for p, _ in ds.records(m))
    tmp = tempfile.mkdtemp()
    try:
        mans = _manifests(run0); man, sp = mans["carpet"]; _fake_tree(tmp, "carpet", man, sp)
        MVTecDataset.categories = ["carpet"]
        c0 = cfg_of("baseline.yaml", ["data.cache_decoded=false"]); c3 = cfg_of("baseline.yaml", ["data.cache_decoded=false", "data.calib_split=0.3"])
        c0["paths"]["mvtec_root"] = c3["paths"]["mvtec_root"] = tmp
        d0 = run_train.build_dataset("mvtec", c0, "carpet"); d3 = run_train.build_dataset("mvtec", c3, "carpet")
        base = MVTecDataset(tmp, "carpet", val_split=0.2, seed=123, anomaly_fraction=0.1, injection_val_share=0.65,
                            val_share_mode="adaptive", de_confound_frac=0.5, use_validation=True)
        check("MVTec calib_split=0: train/val/test identical to a build without the key",
              all(paths(d0, m) == paths(base, m) for m in ("train", "val", "test")) and not d0.records("calib"))
        cal = set(paths(d3, "calib"))
        check("MVTec calib_split=0.3: val, test and train defects unchanged; calib only from train normals",
              paths(d3, "val") == paths(d0, "val") and paths(d3, "test") == paths(d0, "test")
              and sorted(p for p, l in d3.records("train") if l == 1) == sorted(p for p, l in d0.records("train") if l == 1)
              and len(cal) > 0 and all(l == 0 for _, l in d3.records("calib"))
              and set(paths(d3, "train")) == set(paths(d0, "train")) - cal, f"calib n={len(cal)}")
        check("MVTec calib disjoint from train, val and test",
              not (cal & set(paths(d3, "train"))) and not (cal & set(paths(d3, "val"))) and not (cal & set(paths(d3, "test"))))
        # VisA on a synthetic split CSV
        os.makedirs(os.path.join(tmp, "visa", "split_csv"), exist_ok=True)
        with open(os.path.join(tmp, "visa", "split_csv", "2cls_highshot.csv"), "w", newline="") as f:
            w = _csv.writer(f); w.writerow(["object", "split", "label", "image"])
            for i in range(200): w.writerow(["candle", "train", "normal", f"candle/n{i}.JPG"])
            for i in range(20): w.writerow(["candle", "train", "anomaly", f"candle/a{i}.JPG"])
            for i in range(50): w.writerow(["candle", "test", "normal" if i < 40 else "anomaly", f"candle/t{i}.JPG"])
        v0 = cfg_of("baseline.yaml", ["data.cache_decoded=false"]); v3 = cfg_of("baseline.yaml", ["data.cache_decoded=false", "data.calib_split=0.3"])
        v0["paths"]["visa_root"] = v3["paths"]["visa_root"] = os.path.join(tmp, "visa")
        e0 = run_train.build_dataset("visa", v0, "candle"); e3 = run_train.build_dataset("visa", v3, "candle")
        vc = set(paths(e3, "calib"))
        check("VisA calib_split=0 leaves no calib; 0.3 changes only train normals",
              not e0.records("calib") and paths(e3, "val") == paths(e0, "val") and paths(e3, "test") == paths(e0, "test")
              and set(paths(e3, "train")) == set(paths(e0, "train")) - vc and all(l == 0 for _, l in e3.records("calib"))
              and len(vc) == int(round(0.3 * sum(1 for _, l in e0.records("train") if l == 0))), f"calib n={len(vc)}")
        # NEGATIVE CONTROL: the runtime leak guard must fire when a calib image is also a training image
        tf = object(); t = paths(d3, "train"); cl = paths(d3, "calib")
        try:
            leak_self_check(train_paths=t + cl[:1], test_paths=paths(d3, "test"), val_threshold=0.0, test_threshold=0.0,
                            val_transform=tf, test_transform=tf, use_validation=True, val_paths=paths(d3, "val"),
                            calib_paths=cl); fired = False
        except LeakCheckError:
            fired = True
        check("CONTROL: leak guard fires on a calib path that is also in train", fired)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t18():
    """instrumentation.save_scores_per_epoch: pure bookkeeping. With the flag on, training must be bit-identical to
    the flag off (same final weights, same per-epoch val loss), the file must hold one row of val logits per epoch, and
    firewalled test logits must be stored only when the firewall is on. NEGATIVE CONTROL: flag off writes no file."""
    print("\nT18 per-epoch score log")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from engine.train import fit
    torch.manual_seed(0); X = torch.randn(40, 6); y = (X[:, 0] > 0).float()
    class Rec:
        def __init__(s): s.vl = []
        def log_epoch(s, c, e, p): s.vl.append(p["val_loss"])
        def log_stage(s, *a, **k): pass
        def log_grad_norms(s, *a, **k): pass
        def log_logit_hist(s, *a, **k): pass
    def run(flag, fw):
        torch.manual_seed(1); m = torch.nn.Sequential(torch.nn.Linear(6, 1))
        cfg = {"train": {"num_epochs": 4, "patience": 10}, "selection": {"monitor": "val_auroc"},
               "eval": {"metrics": ["auroc"], "firewalled_test_per_epoch": fw},
               "instrumentation": {"capture_grad_norms": False, "capture_logit_hist": False, "save_scores_per_epoch": flag}}
        r = Rec(); d = tempfile.mkdtemp()
        fit(m, DataLoader(TensorDataset(X[:24], y[:24]), batch_size=8, shuffle=True), DataLoader(TensorDataset(X[24:32], y[24:32]), batch_size=8),
            torch.optim.SGD(m.parameters(), lr=0.1), torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), cfg, d, "t18", recorder=r,
            test_loader=DataLoader(TensorDataset(X[32:], y[32:]), batch_size=8))
        f = os.path.join(d, "t18_scores_per_epoch.npz")
        z = dict(np.load(f)) if os.path.exists(f) else None
        w = [q.detach().clone() for q in m.parameters()]; shutil.rmtree(d, ignore_errors=True)
        return w, r.vl, z
    import numpy as np
    w0, v0, z0 = run(False, True); w1, v1, z1 = run(True, True); w2, v2, z2 = run(True, False)
    check("flag on: training bit-identical to flag off (weights and val loss per epoch)",
          all(torch.equal(a, b) for a, b in zip(w0, w1)) and v0 == v1)
    check("file holds one row of val logits per epoch, and test logits when the firewall is on",
          z1 is not None and z1["val_logits"].shape == (4, 8) and z1["test_logits"].shape == (4, 8))
    check("no test logits stored when the firewall is off", z2 is not None and "test_logits" not in z2)
    check("CONTROL: flag off writes no file", z0 is None)


def t19():
    """SWA (train.swa_start): training trajectory unchanged; the checkpoint's weights equal the mean of the raw
    per-epoch weights from swa_start on (raw weights obtained from identical runs stopped at each epoch); BatchNorm
    running stats recomputed; EMA + SWA refused. NEGATIVE CONTROL: the checkpoint differs from the last epoch's weights."""
    print("\nT19 SWA")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from engine.train import fit
    torch.manual_seed(0); X = torch.randn(48, 6); y = (X[:, 0] + 0.3 * X[:, 1] > 0).float()
    class Rec:
        def __init__(s): s.vl = []
        def log_epoch(s, c, e, p): s.vl.append(p["val_loss"])
        def log_stage(s, *a, **k): pass
        def log_grad_norms(s, *a, **k): pass
        def log_logit_hist(s, *a, **k): pass
    def run(n_ep, swa, ema=0):
        torch.manual_seed(1)
        m = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        cfg = {"train": {"num_epochs": n_ep, "patience": 99, "swa_start": swa, "ema_epochs": ema},
               "selection": {"monitor": "val_auroc"}, "eval": {"metrics": ["auroc"]},
               "instrumentation": {"capture_grad_norms": False, "capture_logit_hist": False}}
        r = Rec(); d = tempfile.mkdtemp()
        res = fit(m, DataLoader(TensorDataset(X[:32], y[:32]), batch_size=8, shuffle=True),
                  DataLoader(TensorDataset(X[32:], y[32:]), batch_size=8), torch.optim.SGD(m.parameters(), lr=0.2),
                  torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), cfg, d, "t19", recorder=r)
        ck = torch.load(os.path.join(d, "t19_best_model.pt")); raw = {k: v.clone() for k, v in m.state_dict().items()}
        shutil.rmtree(d, ignore_errors=True)
        return ck, raw, r.vl, res
    ck_on, _, vl_on, res_on = run(4, 2)
    _, _, vl_off, _ = run(4, 0)
    raws = [run(n, 0)[1] for n in (2, 3, 4)]          # raw weights after epochs 2, 3, 4 (swa off => model ends raw)
    lin = [k for k in ck_on if k.endswith("weight") or k.endswith("bias")]
    mean_ok = all(torch.allclose(ck_on[k], sum(r[k] for r in raws) / 3, atol=1e-6) for k in lin)
    check("SWA leaves the training trajectory unchanged (per-epoch val loss identical)", vl_on == vl_off)
    check("checkpoint weights = mean of raw weights over epochs 2..4", mean_ok and res_on["swa_epochs"] == 3)
    check("BatchNorm running stats recomputed for the averaged weights",
          not torch.allclose(ck_on["1.running_mean"], sum(r["1.running_mean"] for r in raws) / 3))
    try:
        run(3, 2, ema=2); refused = False
    except ValueError:
        refused = True
    check("EMA + SWA together is refused", refused)
    check("CONTROL: the SWA checkpoint differs from the last epoch's raw weights",
          not all(torch.allclose(ck_on[k], raws[-1][k]) for k in lin))


def t20():
    """Operating-point loss (engine/losses.py): weight 0 and eval mode are exactly BCE; the tail term is zero when every
    defect beats the hardest normals by the margin and positive otherwise; its gradient pushes the hardest normal DOWN
    and the defect UP; no-defect batches are plain BCE. NEGATIVE CONTROL: the term depends on the HARDEST normals —
    an otherwise identical batch whose extra normals are all easy gives a different (smaller) term."""
    print("\nT20 operating-point loss")
    import torch
    from engine.losses import OperatingPointLoss
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    lg = torch.tensor([2.0, -1.0, -2.0, 1.5, -3.0], requires_grad=True); lb = torch.tensor([1.0, 0, 0, 0, 0])
    L0 = OperatingPointLoss(bce, weight=0.0); L1 = OperatingPointLoss(bce, weight=1.0, beta=0.25, margin=1.0)
    L0.train(); L1.train()
    check("weight 0 equals BCE exactly", torch.equal(L0(lg, lb), bce(lg, lb)))
    L1.eval(); ev = L1(lg, lb); L1.train()
    check("eval mode equals BCE exactly (val loss keeps its meaning)", torch.equal(ev, bce(lg, lb)))
    # hardest normal 1.5, defect 2.0, margin 1 -> gap 0.5 -> term 0.25 (k = ceil(0.25*4) = 1)
    check("tail term = squared hinge on the hardest normal", abs(float(L1.tail_term(lg, lb)) - 0.25) < 1e-6)
    lg.grad = None; L1.tail_term(lg, lb).backward()
    check("gradient pushes the hardest normal down and the defect up", lg.grad[3] > 0 and lg.grad[0] < 0 and lg.grad[1] == 0)
    far = torch.tensor([5.0, -1.0, -2.0, 1.5, -3.0])
    check("term is zero when the defect beats the hardest normals by the margin", float(L1.tail_term(far, lb)) == 0.0)
    nopos = torch.tensor([0.0, 0, 0, 0, 0])
    check("a batch without a defect is plain BCE", torch.equal(L1(lg.detach(), nopos), bce(lg.detach(), nopos)))
    easy = torch.tensor([2.0, -1.0, -2.0, -2.5, -3.0])      # same defect, hardest normal now easy
    check("CONTROL: the term tracks the hardest normals (easy batch gives a smaller term)",
          float(L1.tail_term(easy, lb)) < float(L1.tail_term(lg.detach(), lb)))


def t21(run0):
    """Calibration COUNT (data.calib_n, F-L99 fix): exactly N normals held out, val/test/defects unchanged vs no calib;
    floor rule: if fewer than calib_min_train training normals would remain, NO calib split and the reason recorded;
    fraction + count together refused; count 0 = off. NEGATIVE CONTROL: a count breaching the floor yields no split."""
    print("\nT21 calibration count + floor rule")
    import run_train
    from data.mvtec import MVTecDataset
    paths = lambda ds, m: sorted(p for p, _ in ds.records(m))
    tmp = tempfile.mkdtemp()
    try:
        mans = _manifests(run0); man, sp = mans["carpet"]; _fake_tree(tmp, "carpet", man, sp); MVTecDataset.categories = ["carpet"]
        mk = lambda ov: (lambda c: (c["paths"].__setitem__("mvtec_root", tmp), run_train.build_dataset("mvtec", c, "carpet"))[1])(
            cfg_of("baseline.yaml", ["data.cache_decoded=false", *ov]))
        d0, d30 = mk([]), mk(["data.calib_n=30"])
        pool = sum(1 for _, l in d0.records("train") if l == 0)
        check("count 30: exactly 30 calib normals; val, test and train defects unchanged",
              len(d30.records("calib")) == 30 and paths(d30, "val") == paths(d0, "val") and paths(d30, "test") == paths(d0, "test")
              and sorted(p for p, l in d30.records("train") if l == 1) == sorted(p for p, l in d0.records("train") if l == 1)
              and d30.calib_info["applied"] and d30.calib_info["train_normals_left"] == pool - 30, f"pool {pool}")
        dfl = mk([f"data.calib_n={pool - 50}"])      # would leave 50 < 100 training normals
        check("floor rule: too few training normals left -> no calib split, reason recorded",
              not dfl.records("calib") and not dfl.calib_info["applied"] and "remain" in dfl.calib_info["reason"]
              and paths(dfl, "train") == paths(d0, "train"))
        try:
            mk(["data.calib_n=30", "data.calib_split=0.3"]); refused = False
        except ValueError:
            refused = True
        check("fraction and count together are refused", refused)
        check("count 0 is off (no calib, splits identical)", not d0.records("calib") and not d0.calib_info["applied"])
        check("CONTROL: a count that breaches the floor produces no calibration split", len(dfl.records("calib")) == 0 and len(d30.records("calib")) > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t22():
    """VisA training-defect count (data.train_anomaly_n, pilot F-L105): exactly N training anomalies kept; validation,
    test, calibration and training normals identical to a build without the key; the kept set is a subset of the
    original and fixed by the seed; 0 = off; N above what exists is refused. NEGATIVE CONTROL: a different seed keeps a
    kept subset is not the first or last N in file order (the choice is random)."""
    print("\nT22 VisA training-defect count")
    import run_train, csv as _csv
    paths = lambda ds, m: sorted(p for p, _ in ds.records(m))
    anoms = lambda ds: sorted(p for p, l in ds.records("train") if l == 1)
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "split_csv"), exist_ok=True)
        with open(os.path.join(tmp, "split_csv", "2cls_highshot.csv"), "w", newline="") as f:
            w = _csv.writer(f); w.writerow(["object", "split", "label", "image"])
            for i in range(200): w.writerow(["candle", "train", "normal", f"candle/n{i}.JPG"])
            for i in range(40): w.writerow(["candle", "train", "anomaly", f"candle/a{i}.JPG"])
            for i in range(50): w.writerow(["candle", "test", "normal" if i < 40 else "anomaly", f"candle/t{i}.JPG"])
        def mk(ov):
            c = cfg_of("baseline.yaml", ["data.cache_decoded=false", *ov]); c["paths"]["visa_root"] = tmp
            return run_train.build_dataset("visa", c, "candle")
        d0, d10, d10b, dc = mk([]), mk(["data.train_anomaly_n=10"]), mk(["data.train_anomaly_n=10"]), mk(["data.train_anomaly_n=10", "data.calib_n=30", "data.calib_min_train=50"])
        norm = lambda ds: sorted(p for p, l in ds.records("train") if l == 0)
        check("n=10: exactly 10 training anomalies, a subset of the original",
              len(anoms(d10)) == 10 and set(anoms(d10)) <= set(anoms(d0)), f"orig {len(anoms(d0))}")
        check("val, test and training normals unchanged",
              paths(d10, "val") == paths(d0, "val") and paths(d10, "test") == paths(d0, "test") and norm(d10) == norm(d0))
        check("same seed -> same kept subset (reproducible)", anoms(d10) == anoms(d10b))
        c0 = mk(["data.calib_n=30", "data.calib_min_train=50"])
        check("composes with calibration count: calib set unchanged by the defect count",
              paths(dc, "calib") == paths(c0, "calib") and len(anoms(dc)) == 10)
        check("0 is off (bit-identical splits)", all(paths(mk(["data.train_anomaly_n=0"]), m) == paths(d0, m) for m in ("train", "val", "test"))
              and not d0.train_anomaly_info["applied"])
        try:
            mk([f"data.train_anomaly_n={len(anoms(d0)) + 1}"]); refused = False
        except ValueError:
            refused = True
        check("N above the available anomalies is refused", refused)
        order = [p for p, l in d0.records("train") if l == 1]          # original file order of the training anomalies
        check("CONTROL: the kept subset is neither the first nor the last N in file order (really random)",
              set(anoms(d10)) != set(order[:10]) and set(anoms(d10)) != set(order[-10:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t23():
    """Session-alignment term (engine/losses.py SessionAlignLoss, pilot F-L105): weight 0 and eval mode are exactly the
    base loss; the term pulls TEST-SESSION NORMALS (path contains '/test/') toward the running mean of training-session
    normals and touches nothing else (defects and training-session normals get no gradient from it); the running target
    is set only from training-session normals. Training-loop path: with weight 0 the split forward (features -> head)
    gives bit-identical weights to the plain forward. NEGATIVE CONTROL: the same batch with every path in the training
    session gives a zero term (the session is read from the path, not invented)."""
    print("\nT23 session-alignment term")
    import torch
    from torch.utils.data import DataLoader, Dataset
    from engine.losses import SessionAlignLoss
    from engine.train import train_one_epoch
    bce = torch.nn.BCEWithLogitsLoss()
    lg = torch.tensor([1.0, -1.0, -1.0, 0.5]); lb = torch.tensor([1.0, 0, 0, 0])
    P = ["c/test/crack/a.png", "c/train/good/1.png", "c/train/good/2.png", "c/test/good/3.png"]
    f = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    L = SessionAlignLoss(bce, weight=1.0, momentum=0.9); L.train()
    t0, _ = L.align_term(f, lb, P)
    check("first batch: no target yet -> term 0; target set from training-session normals only",
          float(t0) == 0.0 and torch.allclose(L.mu, torch.tensor([1.0, 0.0])))
    t1, n = L.align_term(f, lb, P)
    check("test-session normal far from the target -> positive term (|[0,1]-[1,0]|^2 = 2)", abs(float(t1) - 2.0) < 1e-6 and n == 1)
    f.grad = None; L.align_term(f, lb, P)[0].backward()
    check("gradient only on the test-session NORMAL (not the defect, not training normals)",
          f.grad[3].abs().sum() > 0 and f.grad[0].abs().sum() == 0 and f.grad[1].abs().sum() == 0 and f.grad[2].abs().sum() == 0)
    L0 = SessionAlignLoss(bce, weight=0.0); L0.train()
    check("weight 0 equals the base loss exactly", torch.equal(L0(lg, lb, feats=f.detach(), paths=P), bce(lg, lb)))
    L.eval(); ev = L(lg, lb, feats=f.detach(), paths=P); L.train()
    check("eval mode equals the base loss exactly", torch.equal(ev, bce(lg, lb)))
    Lc = SessionAlignLoss(bce, weight=1.0); Lc.train(); Ptr = [p.replace("/test/", "/train/") for p in P]
    Lc.align_term(f.detach(), lb, Ptr); tc, nc = Lc.align_term(f.detach(), lb, Ptr)
    check("CONTROL: same batch, every path in the training session -> term 0, nothing aligned", float(tc) == 0.0 and nc == 0)

    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.b = torch.nn.Linear(4, 3); s.h = torch.nn.Linear(3, 1)
        def features(s, x): return torch.relu(s.b(x))
        def head(s, z): return s.h(z)
        def forward(s, x): return s.head(s.features(x)).squeeze(-1)
    class D(Dataset):
        def __init__(s):
            g = torch.Generator().manual_seed(0); s.x = torch.randn(16, 4, generator=g); s.y = (torch.arange(16) % 4 == 0).long()
        def __len__(s): return 16
        def __getitem__(s, i): return s.x[i], s.y[i], ("c/test/good/%d.png" % i) if i % 5 == 1 else ("c/train/good/%d.png" % i)
    def one(loss):
        torch.manual_seed(0); m = M(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
        train_one_epoch(m, DataLoader(D(), batch_size=8, shuffle=False), opt, loss, "cpu", capture_grad=False)
        return torch.cat([p.detach().flatten() for p in m.parameters()])
    w_plain, w_zero, w_on = one(torch.nn.BCEWithLogitsLoss()), one(SessionAlignLoss(torch.nn.BCEWithLogitsLoss(), weight=0.0)), \
        one(SessionAlignLoss(torch.nn.BCEWithLogitsLoss(), weight=5.0))
    check("training loop: weight 0 through the split forward gives bit-identical weights", torch.equal(w_plain, w_zero))
    check("training loop: weight > 0 changes the weights (the term reaches the model)", not torch.equal(w_plain, w_on))


def t24():
    """pos_weight cap (train.pos_weight_cap, K-027): absent = unchanged; a cap above the weight leaves it bit-identical
    (the VisA case); a cap below it returns the cap (the MVTec case); applies after the multiplier. NEGATIVE CONTROL:
    the historical neg_over_total weight (< 1) is untouched by a cap of 10 — the cap cannot raise a weight."""
    print("\nT24 pos_weight cap")
    import torch, run_train
    class DS:
        def __init__(s, n_neg, n_pos): s.l = [0] * n_neg + [1] * n_pos
        def labels(s, split): return s.l
    def w(n_neg, n_pos, **tr):
        c = cfg_of("baseline.yaml", ["train.use_class_weights=true", "train.class_weight_mode=neg_over_pos"] + [f"train.{k}={v}" for k, v in tr.items()])
        return run_train.class_weight(DS(n_neg, n_pos), c, "cpu")
    check("no cap: neg/pos exactly (221/10 = 22.1)", abs(float(w(221, 10)) - 22.1) < 1e-5)
    check("cap above the weight: bit-identical (480/48 = 10 vs cap 12)", torch.equal(w(480, 48, pos_weight_cap=12), w(480, 48)))
    check("cap below the weight: equals the cap (22.1 -> 10)", float(w(221, 10, pos_weight_cap=10)) == 10.0)
    check("cap applies after the multiplier (22.1 x 0.5 = 11.05 -> 10)", float(w(221, 10, pos_weight_cap=10, weight_multiplier=0.5)) == 10.0)
    c = cfg_of("baseline.yaml", ["train.use_class_weights=true", "train.pos_weight_cap=10"])
    base = cfg_of("baseline.yaml", ["train.use_class_weights=true"])
    check("CONTROL: the historical weight (< 1) is not raised by the cap",
          torch.equal(run_train.class_weight(DS(221, 10), c, "cpu"), run_train.class_weight(DS(221, 10), base, "cpu")))


def t25():
    """Regime rule (train.regime_min_defects, K-030): absent = unchanged; a SCARCE category (< threshold training defects)
    gets no calibration split even if calib_n is set, and the historical weight neg/(neg+pos) even if neg_over_pos is
    set — so the rule model's MVTec part is exactly the existing no-weight recipe; a RICH category keeps neg/pos and the
    calibration split. Prior correction is monotone (ranking unchanged) and equals logits − log(w).
    NEGATIVE CONTROL: lowering the threshold below the defect count turns the same category rich."""
    print("\nT25 regime rule")
    import numpy as np, run_train, csv as _csv
    from engine.deploy import prior_corrected
    from scipy.stats import rankdata
    paths = lambda ds, m: sorted(p for p, _ in ds.records(m))
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "split_csv"), exist_ok=True)
        with open(os.path.join(tmp, "split_csv", "2cls_highshot.csv"), "w", newline="") as f:
            w = _csv.writer(f); w.writerow(["object", "split", "label", "image"])
            for i in range(300): w.writerow(["candle", "train", "normal", f"candle/n{i}.JPG"])
            for i in range(60): w.writerow(["candle", "train", "anomaly", f"candle/a{i}.JPG"])
            for i in range(50): w.writerow(["candle", "test", "normal" if i < 40 else "anomaly", f"candle/t{i}.JPG"])
        def mk(ov):
            c = cfg_of("baseline.yaml", ["data.cache_decoded=false", "train.use_class_weights=true", *ov]); c["paths"]["visa_root"] = tmp
            ds = run_train.build_dataset("visa", c, "candle"); return ds, float(run_train.class_weight(ds, c, "cpu"))
        RULE = ["train.class_weight_mode=neg_over_pos", "data.calib_n=110", "data.calib_min_train=50", "train.regime_min_defects=20"]
        scarce, ws = mk(RULE + ["data.train_anomaly_n=10"]); plain, wp = mk(["data.train_anomaly_n=10"])
        check("scarce (10 < 20): no calibration split although calib_n=110",
              not scarce.records("calib") and scarce.regime == {"min_defects": 20, "train_defects": 10, "rich": False})
        check("scarce: splits and weight IDENTICAL to the plain no-weight recipe (existing A runs stay valid)",
              all(paths(scarce, m) == paths(plain, m) for m in ("train", "val", "test")) and ws == wp and ws < 1.0)
        rich, wr = mk(RULE)
        n_def = sum(1 for _, l in rich.records("train") if l == 1); n_norm = sum(1 for _, l in rich.records("train") if l == 0)
        check("rich (>= 20 defects): calibration split of 110 and neg/pos weight",
              len(rich.records("calib")) == 110 and abs(wr - n_norm / n_def) < 1e-4 and rich.regime["rich"], f"defects {n_def}")
        none_, wn = mk(["train.class_weight_mode=neg_over_pos", "data.calib_n=110", "data.calib_min_train=50"])
        check("no rule key: unchanged (calibration and neg/pos as configured)", len(none_.records("calib")) == 110 and not hasattr(none_, "regime"))
        low, wl = mk(RULE[:-1] + ["train.regime_min_defects=5", "data.train_anomaly_n=10"])
        check("CONTROL: threshold 5 turns the 10-defect category rich (calibration + neg/pos)", low.regime["rich"] and len(low.records("calib")) == 110 and wl > 1.0)
        lg = np.array([2.0, -1.0, 0.5, -3.0]); pc = prior_corrected(lg, 16.0)
        check("prior correction = logits − log(w) and keeps the ranking",
              np.allclose(pc, lg - np.log(16.0)) and np.array_equal(rankdata(pc), rankdata(lg)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t26():
    """View-consistency term (train.view_consistency_weight): not built at weight 0 (config path unchanged); eval mode =
    base exactly; the term = weight * mean squared logit gap; its gradient reaches the model through BOTH views; the
    training loop feeds a real test-time view. NEGATIVE CONTROL: identical logits for both views give a zero term."""
    print("\nT26 view-consistency term")
    import torch
    from engine.losses import ViewConsistencyLoss
    from engine.train import train_one_epoch
    from torch.utils.data import DataLoader, TensorDataset
    bce = torch.nn.BCEWithLogitsLoss()
    lg = torch.tensor([1.0, -1.0, 0.5], requires_grad=True); lv = torch.tensor([0.0, -1.0, 1.5], requires_grad=True); lb = torch.tensor([1.0, 0, 0])
    L = ViewConsistencyLoss(bce, weight=0.5); L.train()
    check("term = weight x mean squared gap ((1+0+1)/3 x 0.5)", abs(float(L(lg, lb, logits_view=lv) - bce(lg, lb)) - 0.5 * 2 / 3) < 1e-6)
    L(lg, lb, logits_view=lv).backward()
    check("gradient reaches both views", lg.grad.abs().sum() > 0 and lv.grad.abs().sum() > 0)
    L.eval(); check("eval mode = base exactly", torch.equal(L(lg.detach(), lb, logits_view=lv.detach()), bce(lg.detach(), lb))); L.train()
    check("CONTROL: identical logits for both views -> zero term", torch.equal(L(lg.detach(), lb, logits_view=lg.detach()), bce(lg.detach(), lb)))
    c = cfg_of("baseline.yaml", []); check("weight absent in the config -> term not built (no second forward, no RNG change)",
                                         float(c["train"].get("view_consistency_weight", 0) or 0) == 0)
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1))
    X = torch.randn(8, 3, 16, 16); Y = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0])
    w0 = torch.cat([p.detach().flatten().clone() for p in m.parameters()])
    train_one_epoch(m, DataLoader(TensorDataset(X, Y), batch_size=4), torch.optim.SGD(m.parameters(), lr=0.1), ViewConsistencyLoss(bce, 1.0), "cpu", capture_grad=False)
    check("training loop runs with a real test-time view and updates the model", not torch.equal(w0, torch.cat([p.detach().flatten() for p in m.parameters()])))


def t27():
    """Top-K patch scoring (model.score_mode, F-L122): default config = image mode and the ORIGINAL forward bit-for-bit;
    patch_topk = mean of the top ceil(frac x cells) per-cell logits of the SAME head (9 of 81 at 288 px); with a LINEAR
    head and frac = 1 it equals average pooling exactly (consistency). NEGATIVE CONTROL: frac = 0.1 differs from average
    pooling. Bad settings are refused; a training step updates the model in patch mode."""
    print("\nT27 top-K patch scoring")
    import math, torch
    import numpy as np
    from models.model_factory import build_model, AnomalyClassifier
    from engine.train import train_one_epoch
    from torch.utils.data import DataLoader, TensorDataset
    c = cfg_of("baseline.yaml", ["model.pretrained=false"])
    check("default config -> score_mode image", c["model"].get("score_mode", "image") == "image" and float(c["model"].get("patch_topk_frac")) == 0.1)
    torch.manual_seed(0); m = build_model(c, "efficientnet").eval(); x = torch.randn(2, 3, 288, 288)
    with torch.no_grad():
        check("image mode = original path (head(pool(features))) bit-for-bit", torch.equal(m(x), m.head(m.backbone(x)).squeeze(-1)))
    cp = cfg_of("baseline.yaml", ["model.pretrained=false", "model.score_mode=patch_topk"])
    torch.manual_seed(0); mp = build_model(cp, "efficientnet").eval()
    mp.load_state_dict(m.state_dict())
    check("patch_topk has exactly the same parameters as image mode", sum(q.numel() for q in mp.parameters()) == sum(q.numel() for q in m.parameters()))
    with torch.no_grad():
        k = math.ceil(0.1 * mp.cell_logits(x).shape[1])
        check("288 px -> 81 cells, k = 9", mp.cell_logits(x).shape == (2, 81) and k == 9)
    # A randomly initialised EfficientNet gives a spatially CONSTANT final map (std ~1e-13), which would make every
    # aggregation check vacuous. The aggregation logic is tested on a stand-in conv stack with real spatial variation
    # (one stride-32 conv -> [B, 1280, 9, 9]); the real backbone is used above for shapes and bit-for-bit identity.
    stub = lambda: torch.nn.Sequential(torch.nn.Conv2d(3, 1280, 32, stride=32))
    torch.manual_seed(1); s = stub()
    def mk(**kw):
        a = AnomalyClassifier("efficientnet", head=kw.pop("head", "linear"), pretrained=False, image_size=288, **kw).eval(); a.backbone.features = s; return a
    ml, ml1, ml01, md = mk(), mk(score_mode="patch_topk", patch_topk_frac=1.0), mk(score_mode="patch_topk", patch_topk_frac=0.1), mk(head="deep", score_mode="patch_topk")
    for a in (ml1, ml01): a.head.load_state_dict(ml.head.state_dict())
    with torch.no_grad():
        cl = md.cell_logits(x)
        check("cells vary on the stand-in (the checks below are not vacuous)", float(cl.std(1).min()) > 1e-3, round(float(cl.std(1).min()), 4))
        check("patch_topk output = mean of the top-9 cell logits (deep head)", torch.allclose(md(x), cl.topk(9, dim=1).values.mean(1)))
        check("linear head, frac = 1 -> equals average pooling (consistency)", torch.allclose(ml(x), ml1(x), atol=1e-5), float((ml(x) - ml1(x)).abs().max()))
        check("CONTROL: frac = 0.1 differs from average pooling", (ml(x) - ml01(x)).abs().max() > 1e-3, round(float((ml(x) - ml01(x)).abs().max()), 4))
    for bad, kw in (("unknown mode", dict(score_mode="bogus")), ("frac 0", dict(score_mode="patch_topk", patch_topk_frac=0.0)), ("taps", dict(score_mode="patch_topk", taps=[5]))):
        try:
            AnomalyClassifier("efficientnet", pretrained=False, image_size=288, **kw); check(f"refuses {bad}", False)
        except ValueError:
            check(f"refuses {bad}", True)
    mp.train(); w0 = torch.cat([q.detach().flatten().clone() for q in mp.head.parameters()])
    X = torch.randn(4, 3, 288, 288); Y = torch.tensor([1, 0, 0, 1])
    train_one_epoch(mp, DataLoader(TensorDataset(X, Y), batch_size=4), torch.optim.SGD(mp.parameters(), lr=0.1), torch.nn.BCEWithLogitsLoss(), "cpu", capture_grad=False)
    check("patch mode trains (head updated)", not torch.equal(w0, torch.cat([q.detach().flatten() for q in mp.head.parameters()])))
    from engine.evaluate import run_inference
    mp.eval(); out = run_inference(mp, DataLoader(TensorDataset(X, Y), batch_size=2), torch.nn.BCEWithLogitsLoss(), "cpu", capture_features=True)
    check("REAL evaluation path: patch mode saves pooled features (rows = images, 1280-d) and its logits", out["features"].shape == (4, 1280) and out["logits"].shape == (4,), out["features"].shape)
    with torch.no_grad():
        check("pooled features in patch mode = the image-mode embedding of the same weights", np.allclose(out["features"], mp.backbone(X).numpy(), atol=1e-5))
    class _NeedsFeat(torch.nn.BCEWithLogitsLoss):
        needs_features = True
    try:
        train_one_epoch(mp, DataLoader(TensorDataset(X, Y), batch_size=4), torch.optim.SGD(mp.parameters(), lr=0.1), _NeedsFeat(), "cpu", capture_grad=False)
        check("session-alignment path refuses patch mode (would silently use image scoring)", False)
    except ValueError:
        check("session-alignment path refuses patch mode (would silently use image scoring)", True)


def t28():
    """Defect transplant (data.transplant, F-L124): mask path mapping; pixels = donor inside the mask, host outside;
    NEGATIVE CONTROL: an empty mask returns the host unchanged; a missing mask fails at construction; off by default and
    in the rich regime (records unchanged); scarce: 48 - n_real synthetic records, label 1, hosts = TRAINING normals,
    deterministic per seed; a transplant record without its op is refused; the real train loader serves them."""
    print("\nT28 defect transplant")
    import numpy as np, random
    from PIL import Image
    from data.transplant import DefectTransplant, mvtec_mask_path
    import run_train
    from data.view import SplitView
    tmp = tempfile.mkdtemp()
    try:
        cat = os.path.join(tmp, "carpet"); [os.makedirs(os.path.join(cat, d), exist_ok=True) for d in ("train/good", "test/color", "ground_truth/color")]
        normals = []
        for i in range(6):
            q = os.path.join(cat, "train/good", f"{i:03d}.png"); Image.new("RGB", (64, 64), (20 + i, 40, 60)).save(q); normals.append(q)
        dimg = Image.new("RGB", (64, 64), (0, 0, 0)); dimg.paste((255, 0, 0), (20, 20, 40, 40)); dp = os.path.join(cat, "test/color/000.png"); dimg.save(dp)
        m = Image.new("L", (64, 64), 0); m.paste(255, (20, 20, 40, 40)); mp = os.path.join(cat, "ground_truth/color/000_mask.png"); m.save(mp)
        check("mask path mapping test/<type>/<stem>.png -> ground_truth/<type>/<stem>_mask.png", mvtec_mask_path(dp) == mp)
        host = Image.open(normals[0]).convert("RGB"); op0 = DefectTransplant([dp], feather_radius=0.0); out = np.array(op0(host))
        check("inside the mask = donor pixels (red), outside = host pixels", tuple(out[30, 30]) == (255, 0, 0) and tuple(out[5, 5]) == tuple(np.array(host)[5, 5]))
        edge = int(np.array(DefectTransplant([dp], feather_radius=2.0)(host))[20, 30, 0]); inner = int(np.array(host)[20, 30, 0])
        check("feathered edge blends (radius 2): strictly between host and donor", inner < edge < 255, (inner, edge))
        Image.new("L", (64, 64), 0).save(mp)
        check("CONTROL: empty mask -> host unchanged", np.array_equal(np.array(DefectTransplant([dp], 0.0)(host)), np.array(host)))
        m.save(mp)
        try:
            DefectTransplant([os.path.join(cat, "test/color/999.png")]); check("missing mask fails at construction", False)
        except FileNotFoundError:
            check("missing mask fails at construction", True)
        recs = [(q, 0) for q in normals] + [(dp, 1)]
        class DS: pass
        ds = DS(); ds.regime = {"rich": False, "train_defects": 1}
        c = cfg_of("baseline.yaml"); c["seed"] = 123
        check("config default: transplant off", not c["data"]["transplant"]["enabled"])
        r0, op, d0 = run_train.apply_transplant(recs, ds, "mvtec", c)
        check("off -> records unchanged, no op", r0 == recs and op is None and not d0["applied"])
        c["data"]["transplant"]["enabled"] = True
        ds.regime = {"rich": True}; r1, op1, d1 = run_train.apply_transplant(recs, ds, "mvtec", c)
        check("rich regime -> not applied", r1 == recs and op1 is None and not d1["applied"])
        ds.regime = {"rich": False}; r2, op2, d2 = run_train.apply_transplant(recs, ds, "mvtec", c)
        syn = [r for r in r2 if len(r) > 2 and r[2] == "transplant"]
        check("scarce -> 48 - 1 = 47 synthetic records, label 1, hosts are training normals", len(syn) == 47 and all(r[1] == 1 and r[0] in normals for r in syn) and d2["n_synthetic"] == 47)
        check("original records kept first and unchanged", r2[:len(recs)] == recs)
        check("deterministic per seed", run_train.apply_transplant(recs, ds, "mvtec", c)[0] == r2)
        try:
            run_train.apply_transplant(recs, ds, "visa", c); check("scarce VisA refused (MVTec mask layout only)", False)
        except NotImplementedError:
            check("scarce VisA refused (MVTec mask layout only)", True)
        try:
            SplitView([syn[0]], None)[0]; check("transplant record without its op is refused", False)
        except RuntimeError:
            check("transplant record without its op is refused", True)
        from utils.transforms import build_transforms
        c["data"]["num_workers"] = 0; c["data"]["cache_decoded"] = False; c["data"]["image_size"] = 64; c["train"]["batch_size"] = 4
        tf, _ = build_transforms(c, train=False)
        random.seed(0); L = run_train.make_loader([syn[0], syn[1], recs[0], recs[1]], tf, c, shuffle=False, transplant_op=op2)
        x, y, _ = next(iter(L))
        check("real loader serves transplanted images (label 1, differ from their host)", y.tolist() == [1, 1, 0, 0] and not torch_equal_host(x[0], tf(Image.open(syn[0][0]).convert("RGB"))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t29():
    """Transplant v2 placement (data.transplant.placement=best_match, F-L127): the donor's defect sits on a bar at the TOP;
    of 6 hosts only #2 has the bar at the top (others at the bottom) -> best_match picks host #2 for that donor.
    NEGATIVE CONTROL: the donor's own ring cost is lowest for the aligned host and clearly higher for a misaligned one.
    Donors round-robin; hosts not reused per donor until exhausted; default placement stays 'random' (v1 unchanged);
    records carry the donor index and the view pastes THAT donor."""
    print("\nT29 transplant v2 (context-matched placement)")
    import numpy as np
    from PIL import Image
    from data.transplant import plan_best_match, DefectTransplant, _ring, _gray, mvtec_mask_path
    import run_train
    tmp = tempfile.mkdtemp()
    try:
        cat = os.path.join(tmp, "screw"); [os.makedirs(os.path.join(cat, d), exist_ok=True) for d in ("train/good", "test/thread", "ground_truth/thread")]
        def img(top, defect=False):
            a = np.full((64, 64, 3), 30, np.uint8); r = (8, 20) if top else (44, 56); a[r[0]:r[1], 8:56] = 200
            if defect: a[12:16, 28:36] = (255, 0, 0)
            return Image.fromarray(a)
        hosts = []
        for i in range(6):
            q = os.path.join(cat, "train/good", f"{i:03d}.png"); img(top=(i == 2)).save(q); hosts.append(q)
        dp = os.path.join(cat, "test/thread/000.png"); img(top=True, defect=True).save(dp)
        m = Image.new("L", (64, 64), 0); m.paste(255, (28, 12, 36, 16)); m.save(os.path.join(cat, "ground_truth/thread/000_mask.png"))
        pairs = plan_best_match([dp], hosts, 3, size=64, ring_px=4)
        check("best_match picks the host with the object at the same place (host #2 first)", pairs[0] == (hosts[2], 0), pairs[0])
        check("a host is not reused for the same donor before others are used", len({h for h, _ in pairs}) == 3)
        ring = _ring(Image.open(mvtec_mask_path(dp)), 64, 4); d = _gray(dp, 64)[ring]
        c_al, c_mis = np.abs(_gray(hosts[2], 64)[ring] - d).mean(), np.abs(_gray(hosts[0], 64)[ring] - d).mean()
        check("CONTROL: ring cost aligned << misaligned host", c_al < 0.5 * c_mis, (round(float(c_al), 3), round(float(c_mis), 3)))
        dp2 = os.path.join(cat, "test/thread/001.png"); a2 = np.array(img(top=False)); a2[12:16, 28:36] = (0, 255, 0)
        Image.fromarray(a2).save(dp2); m.save(os.path.join(cat, "ground_truth/thread/001_mask.png"))
        pr = plan_best_match([dp, dp2], hosts, 4, size=64, ring_px=4)
        check("donors round-robin (0, 1, 0, 1)", [di for _, di in pr] == [0, 1, 0, 1])
        op2 = DefectTransplant([dp, dp2], feather_radius=0.0); hi = Image.open(hosts[0]).convert("RGB")
        check("donor_idx is honoured: idx 0 pastes red, idx 1 pastes green (CONTROL: they differ)",
              tuple(np.array(op2(hi, donor_idx=0))[14, 32]) == (255, 0, 0) and tuple(np.array(op2(hi, donor_idx=1))[14, 32]) == (0, 255, 0))
        c = cfg_of("baseline.yaml"); c["seed"] = 123
        check("config default: placement random, ring 8 px", c["data"]["transplant"]["placement"] == "random" and int(c["data"]["transplant"]["ring_px"]) == 8)
        class DS: pass
        ds = DS(); ds.regime = {"rich": False}
        c["data"]["transplant"]["enabled"] = True; c["data"]["transplant"]["placement"] = "best_match"; c["data"]["image_size"] = 64; c["data"]["transplant"]["ring_px"] = 4
        recs = [(h, 0) for h in hosts] + [(dp, 1)]
        r2, op, d2 = run_train.apply_transplant(recs, ds, "mvtec", c)
        syn = [r for r in r2 if len(r) > 2 and r[2] == "transplant"]
        check("best_match records carry a donor index; 47 synthetic; hosts are training normals", len(syn) == 47 and all(len(r) == 4 and r[0] in hosts for r in syn) and d2["placement"] == "best_match")
        from data.view import SplitView
        out = np.array(SplitView([syn[0]], None, transplant_op=op)[0][0])
        check("the view pastes the PLANNED donor (red defect at its place)", tuple(out[14, 32]) == (255, 0, 0) or out[14, 32, 0] > 200)
        c["data"]["transplant"]["placement"] = "nonsense"
        try:
            run_train.apply_transplant(recs, ds, "mvtec", c); check("unknown placement refused", False)
        except ValueError:
            check("unknown placement refused", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t30():
    """Transplant v3 donor rule (data.transplant.donor_rule=inside_object, F-L129): a defect INSIDE a disk object is a
    paste donor; a defect CROSSING the disk's outline is not (it stays a real training defect); a texture image counts as
    all-foreground (its edge defect is accepted). NEGATIVE CONTROL: donor_rule 'all' uses both donors. n_synthetic still
    = 48 - ALL real defects; if no donor passes, nothing is added; default config keeps 'all'."""
    print("\nT30 transplant v3 (outline rule)")
    import numpy as np
    from PIL import Image
    from data.transplant import donor_inside_object, object_foreground
    import run_train
    tmp = tempfile.mkdtemp()
    try:
        cat = os.path.join(tmp, "screw"); [os.makedirs(os.path.join(cat, d), exist_ok=True) for d in ("train/good", "test/inside", "test/edge", "ground_truth/inside", "ground_truth/edge")]
        yy, xx = np.mgrid[:128, :128]; disk = (yy - 64) ** 2 + (xx - 64) ** 2 < 40 ** 2
        def obj(): a = np.full((128, 128, 3), 20, np.uint8); a[disk] = 220; return a
        hosts = []
        for i in range(4):
            q = os.path.join(cat, "train/good", f"{i:03d}.png"); Image.fromarray(obj()).save(q); hosts.append(q)
        def donor(sub, box):
            a = obj(); a[box[1]:box[3], box[0]:box[2]] = (255, 0, 0); p = os.path.join(cat, "test", sub, "000.png"); Image.fromarray(a).save(p)
            m = Image.new("L", (128, 128), 0); m.paste(255, box); m.save(os.path.join(cat, "ground_truth", sub, "000_mask.png")); return p
        d_in = donor("inside", (56, 56, 72, 72)); d_edge = donor("edge", (96, 56, 112, 72))   # disk edge at x = 104
        fg = object_foreground(d_in)
        check("foreground = the disk (not the background), not whole image", 0.2 < fg.mean() < 0.5 and not fg.all(), round(float(fg.mean()), 3))
        check("defect inside the object is a paste donor", donor_inside_object(d_in))
        check("defect crossing the outline is NOT a paste donor", not donor_inside_object(d_edge))
        tex = os.path.join(tmp, "carpet", "test", "edge"); os.makedirs(tex); os.makedirs(os.path.join(tmp, "carpet", "ground_truth", "edge"))
        rng = np.random.default_rng(0); ta = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8); tp_ = os.path.join(tex, "000.png"); Image.fromarray(ta).save(tp_)
        mm = Image.new("L", (128, 128), 0); mm.paste(255, (100, 0, 128, 20)); mm.save(os.path.join(tmp, "carpet", "ground_truth", "edge", "000_mask.png"))
        inv = os.path.join(tmp, "inv.png"); ia = obj(); ia = 240 - ia; Image.fromarray(ia.astype(np.uint8)).save(inv)
        fi = object_foreground(inv)
        check("reversed polarity (dark object on bright background): foreground is still the disk", abs(float(fi.mean()) - float(fg.mean())) < 0.02 and fi[144, 144] and not fi[2, 2])   # foreground maps are 288 x 288
        check("texture = whole image foreground -> edge defect accepted", object_foreground(tp_).all() and donor_inside_object(tp_))
        class DS: pass
        ds = DS(); ds.regime = {"rich": False}
        c = cfg_of("baseline.yaml"); c["seed"] = 123
        check("config default: donor_rule all", c["data"]["transplant"]["donor_rule"] == "all")
        c["data"]["transplant"].update(enabled=True, placement="best_match", ring_px=4); c["data"]["image_size"] = 128
        recs = [(h, 0) for h in hosts] + [(d_in, 1), (d_edge, 1)]
        _, _, dall = run_train.apply_transplant(recs, ds, "mvtec", c)
        check("CONTROL: donor_rule all -> both donors pasted", dall["n_paste_donors"] == 2)
        c["data"]["transplant"]["donor_rule"] = "inside_object"
        r3, op3, d3 = run_train.apply_transplant(recs, ds, "mvtec", c)
        syn = [r for r in r3 if len(r) > 2 and r[2] == "transplant"]
        check("inside_object -> 1 paste donor, 'edge' type excluded, 46 synthetic (48 - 2 real), real defects kept",
              d3["n_paste_donors"] == 1 and d3["excluded_donor_types"] == ["edge"] and len(syn) == 46 and d3["n_donors"] == 2 and r3[:len(recs)] == recs)
        check("every synthetic record uses the accepted donor (index 0 of the paste list)", all(r[3] == 0 for r in syn) and op3.donors[0][0] == d_in)
        from analysis.transplant_sheets import sheet
        sh = sheet(r3, op3, n=5, size=32)
        check("contact sheet: [donor | host | pasted] rows, 3 x 32 wide, 5 rows", sh.size == (96, 160))
        r4, op4, d4 = run_train.apply_transplant([(h, 0) for h in hosts] + [(d_edge, 1)], ds, "mvtec", c)
        check("no donor passes -> nothing added, no op", op4 is None and not d4["applied"] and len(r4) == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t31():
    """analysis/fa_diagnose.py (F-L136): on a fake run with 20 validation normals scoring -3..-1 and 6 test normals, the one
    test normal planted at +2 is the only false alarm; the sheet PNG and the statistics table are written.
    NEGATIVE CONTROL: without the planted score there is no false alarm."""
    print("\nT31 false-alarm diagnostics")
    import numpy as np
    from PIL import Image
    from analysis.fa_diagnose import diagnose
    tmp = tempfile.mkdtemp()
    try:
        img = lambda sub, i, v: (lambda p: (os.makedirs(os.path.dirname(p), exist_ok=True), Image.new("RGB", (32, 32), (v, v, v)).save(p), p)[2])(os.path.join(tmp, "grid", sub, f"{i:03d}.png"))
        vp = [img("train/good", i, 100) for i in range(20)]; tp = [img("test/good", i, 120 + i) for i in range(6)] + [img("test/bent", 0, 30)]
        def write(test_logits):
            np.savez(os.path.join(tmp, "scores_val_tta.npz"), logits_views=np.repeat(np.linspace(-3, -1, 20)[:, None], 4, 1), labels=np.zeros(20), paths=np.array(vp))
            np.savez(os.path.join(tmp, "scores_test_tta.npz"), logits_views=np.repeat(np.array(test_logits)[:, None], 4, 1), labels=np.array([0] * 6 + [1]), paths=np.array(tp))
        write([-4, -4, -4, -4, -4, 2, 5]); out = os.path.join(tmp, "o"); os.makedirs(out)
        rep = diagnose(tmp, out, "fake")
        check("the planted test normal is the only false alarm", "false alarms 1" in rep[0] and sum(l.startswith("  FA ") for l in rep) == 1, rep[0])
        check("sheet and statistics written", os.path.exists(os.path.join(out, "fake_fa_sheet.png")) and any("AUROC FA vs other" in l for l in rep))
        write([-4, -4, -4, -4, -4, -4, 5]); rep2 = diagnose(tmp, out, "fake2")
        check("CONTROL: no planted score -> no false alarm", "false alarms 0" in rep2[0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t32():
    """analysis/evidence_export.py (THesis_record_v2 REQUESTS 1, 3, 10). leak_one on a synthetic carpet tree (Run 0's file
    counts, real pixels and masks) with the FINAL recipe: synthetic count = 48 - real defects, donors only training
    defects, hosts only training normals, no file shared with val/test -> OK. NEGATIVE CONTROL: a transplant whose host
    is a TEST image must be reported FAIL. Also: EXIF times read back, pixelstats groups, copy of a listed file."""
    print("\nT32 evidence export (leak lists, EXIF, pixel stats)")
    import csv as _csv
    import numpy as np
    from PIL import Image
    import run_train
    from data.mvtec import MVTecDataset
    from analysis import evidence_export as ee
    tmp = tempfile.mkdtemp()
    try:
        man, sp = _manifests(os.path.join(ROOT, "tests", "fixtures", "run0"))["carpet"]; _fake_tree(tmp, "carpet", man, sp)
        rng = np.random.default_rng(0)
        for q in glob.glob(os.path.join(tmp, "carpet", "*", "*", "*.png")):
            Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(q)
            t = q.split(os.sep)[-2]
            if t != "good":
                os.makedirs(os.path.join(tmp, "carpet", "ground_truth", t), exist_ok=True); m = Image.new("L", (24, 24), 0); m.paste(255, (8, 8, 14, 14))
                m.save(os.path.join(tmp, "carpet", "ground_truth", t, os.path.basename(q)[:-4] + "_mask.png"))
        MVTecDataset.categories = ["carpet"]
        c = cfg_of("baseline.yaml", ee.FINAL_SET + ["data.cache_decoded=false", "data.image_size=24", "data.transplant.ring_px=2", f"paths.mvtec_root={tmp}"]); c["seed"] = 123
        rows, ok, line = ee.leak_one(c, "carpet")
        n_real = sum(1 for r in rows if r[2] == "train_real" and r[4] == 1); n_syn = sum(1 for r in rows if r[2] == "synthetic")
        check("final recipe on a synthetic tree: OK, synthetic = 48 - real defects, donors are training defects", ok and n_syn == 48 - n_real and n_real > 0, line)
        orig = run_train.apply_transplant
        def bad(tr, ds, name, cfg):
            r, op, d = orig(tr, ds, name, cfg); t0 = ds.records("test")[0][0]; return r[:-1] + [(t0, 1, "transplant", 0)], op, d
        run_train.apply_transplant = bad
        try:
            _, ok2, line2 = ee.leak_one(c, "carpet")
        finally:
            run_train.apply_transplant = orig
        check("CONTROL: a transplant host taken from the TEST split is reported FAIL", not ok2 and line2.endswith("FAIL"), line2)
        vis = os.path.join(tmp, "visa-anomaly-detection", "pcb1", "Data", "Images", "Normal"); os.makedirs(vis)
        ex = Image.Exif(); ex[306] = "2021:05:01 10:00:00"; Image.new("RGB", (8, 8)).save(os.path.join(vis, "0007.JPG"), exif=ex)
        Image.new("RGB", (8, 8)).save(os.path.join(vis, "0008.JPG"))
        class A: pass
        a = A(); a.visa_root = os.path.join(tmp, "visa-anomaly-detection"); a.out = os.path.join(tmp, "o"); ee.exif(a)
        er = list(_csv.DictReader(open(os.path.join(a.out, "visa_exif.csv"))))
        check("EXIF: time read where present, empty where absent, index from file name",
              [(r["file_index"], r["exif_datetime_306"]) for r in er] == [("7", "2021:05:01 10:00:00"), ("8", "")], er)
        a.mvtec_root = tmp; ee.pixelstats(a); pr = list(_csv.DictReader(open(os.path.join(a.out, "mvtec_pixelstats.csv"))))
        g = {r["group"] for r in pr}; n_img = len(glob.glob(os.path.join(tmp, "carpet", "t*", "*", "*.png")))
        check("pixelstats: every train/test image once, groups train_good/test_good/defect, no masks", g == {"train_good", "test_good", "defect"} and len(pr) == n_img, (g, len(pr), n_img))
        lst = os.path.join(tmp, "l.txt"); open(lst, "w").write("pcb1/Data/Images/Normal/0007.JPG\n"); a.list = lst; a.root = a.visa_root; a.out = os.path.join(tmp, "cp"); ee.copy(a)
        check("copy: listed file copied full size under a flattened name", os.path.exists(os.path.join(a.out, "pcb1__Data__Images__Normal__0007.JPG")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t33():
    """placement=duplicate (REQUESTS 2a, F-L140): tops up to 48 with REAL training defects repeated round-robin, tagged
    'duplicate', no op; SplitView returns the defect image unchanged (no paste). NEGATIVE CONTROL: best_match on the same
    records gives transplant records on NORMAL hosts; an unknown placement raises."""
    print("\nT33 duplicate placement (quantity only)")
    import numpy as np
    from PIL import Image
    import run_train
    from data.view import SplitView
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(1); recs = []
        for i in range(4):
            q = os.path.join(tmp, "c", "train", "good", f"{i:03d}.png"); os.makedirs(os.path.dirname(q), exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(q); recs.append((q, 0))
        for i in range(3):
            q = os.path.join(tmp, "c", "test", "cut", f"{i:03d}.png"); os.makedirs(os.path.dirname(q), exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(q); recs.append((q, 1))
            m = os.path.join(tmp, "c", "ground_truth", "cut", f"{i:03d}_mask.png"); os.makedirs(os.path.dirname(m), exist_ok=True)
            mm = Image.new("L", (16, 16), 0); mm.paste(255, (4, 4, 9, 9)); mm.save(m)
        class DS: pass
        ds = DS(); ds.regime = {"rich": False}
        c = cfg_of("baseline.yaml"); c["seed"] = 7; c["data"]["image_size"] = 16
        c["data"]["transplant"].update(enabled=True, placement="duplicate", ring_px=2)
        r, op, d = run_train.apply_transplant(recs, ds, "mvtec", c)
        add = r[len(recs):]; real_def = [p for p, l in recs if l == 1]
        check("duplicate: 45 added (48 - 3 real), all real training defects, tag duplicate, no op, round-robin 15 each",
              len(add) == 45 and all(a[0] in real_def and a[1] == 1 and a[2] == "duplicate" for a in add) and op is None
              and d["applied"] and d["n_synthetic"] == 45 and sorted({sum(a[0] == p for a in add) for p in real_def}) == [15], d)
        v = SplitView(r, transform=None)
        check("SplitView loads a duplicate unchanged (no paste)", np.array_equal(np.asarray(v[len(recs)][0]), np.asarray(Image.open(add[0][0]).convert("RGB"))))
        c["data"]["transplant"]["placement"] = "best_match"; rb, _, _ = run_train.apply_transplant(recs, ds, "mvtec", c)
        check("CONTROL: best_match on the same records pastes onto NORMAL hosts", all(a[0] in [p for p, l in recs if l == 0] and a[2] == "transplant" for a in rb[len(recs):]))
        c["data"]["transplant"]["placement"] = "nonsense"
        try:
            run_train.apply_transplant(recs, ds, "mvtec", c); raised = False
        except ValueError:
            raised = True
        check("CONTROL: unknown placement raises", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t34():
    """_their_models_package (F-L141): after the block, KairosAD's sub-module (models.msam) stays importable by name — what
    torch.compile needs when it traces lazily — while 'models' and our models.model_factory remain OURS.
    NEGATIVE CONTROL: inside the block 'models' is theirs (our model_factory not importable there)."""
    print("\nT34 KairosAD package swap keeps their sub-modules for torch.compile")
    import importlib
    from analysis.benchmark_kairosad import _their_models_package
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "models")); open(os.path.join(tmp, "models", "msam.py"), "w").write("MARK = 'theirs'\n")
        import models as ours_pkg, models.model_factory  # noqa: F401
        with _their_models_package(tmp):
            m = importlib.import_module("models.msam")
            try:
                importlib.import_module("models.model_factory"); inside_ours = True
            except ImportError:
                inside_ours = False
        check("CONTROL: inside the block 'models' resolves to KairosAD's (ours not importable)", not inside_ours)
        check("after the block: models.msam still importable by name (torch.compile's lazy trace)", importlib.import_module("models.msam").MARK == "theirs")
        check("after the block: 'models' and models.model_factory are OURS", sys.modules["models"] is ours_pkg and hasattr(importlib.import_module("models.model_factory"), "build_model"))
    finally:
        sys.modules.pop("models.msam", None); shutil.rmtree(tmp, ignore_errors=True)


def t35():
    """rename_their_package (F-L141 root fix): KairosAD's models/ becomes kairos_models/, their own imports are rewritten
    (from/import models.x -> kairos_models.x), MobileSAM is untouched, a second call changes nothing, and the renamed module
    imports by its new name. NEGATIVE CONTROL: a comment mentioning 'models' and 'mobile_sam.models' imports stay unchanged."""
    print("\nT35 KairosAD package rename")
    import importlib
    from analysis.benchmark_kairosad import rename_their_package
    tmp = tempfile.mkdtemp()
    try:
        w = lambda rel, txt: (os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True), open(os.path.join(tmp, rel), "w").write(txt))
        w("models/util.py", "X = 7\n"); w("models/msam.py", "from models.util import X\nimport models.util as u\n# models are fine\n")
        w("main.py", "from models.msam import X\nfrom mobile_sam.models import Y\n"); w("MobileSAM/mobile_sam/models/a.py", "from models.x import z\n")
        n = rename_their_package(tmp)
        check("renamed, 2 files rewritten, second call rewrites 0", n == 2 and os.path.isdir(os.path.join(tmp, "kairos_models")) and rename_their_package(tmp) == 0)
        main = open(os.path.join(tmp, "main.py")).read(); ms = open(os.path.join(tmp, "kairos_models", "msam.py")).read()
        check("CONTROL: comment and mobile_sam.models import unchanged; MobileSAM tree untouched",
              "# models are fine" in ms and "from mobile_sam.models import Y" in main and "from models.x" in open(os.path.join(tmp, "MobileSAM/mobile_sam/models/a.py")).read())
        sys.path.insert(0, tmp)
        try:
            check("kairos_models.msam imports by its new name", importlib.import_module("kairos_models.msam").X == 7)
        finally:
            sys.path.remove(tmp); [sys.modules.pop(m) for m in list(sys.modules) if m.startswith("kairos_models")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def torch_equal_host(a, b):
    import torch
    return bool(torch.equal(a, b))


def main():
    ap = argparse.ArgumentParser()
    # A compact COMMITTED fixture (Run 0's summaries, MVTec stage logs, and three score
    # files without features -- 324 KB), so every check runs identically on Kaggle and
    # locally. Before, T2/T4/T6 needed the full 574 MB Run 0 tree and could only run here.
    ap.add_argument("--run0", default=os.path.join(ROOT, "tests", "fixtures", "run0"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    tests = {"t1": t1, "t2": lambda: t2(a.run0), "t3": t3, "t4": lambda: t4(a.run0),
             "t5": t5, "t6": lambda: t6(a.run0), "t7": t7, "t8": t8, "t9": t9, "t10": t10, "t11": t11, "t12": t12, "t13": t13, "t14": t14, "t15": t15, "t16": t16, "t17": lambda: t17(a.run0), "t18": t18, "t19": t19, "t20": t20, "t21": lambda: t21(a.run0), "t22": t22, "t23": t23, "t24": t24, "t25": t25, "t26": t26, "t27": t27, "t28": t28, "t29": t29, "t30": t30, "t31": t31, "t32": t32, "t33": t33, "t34": t34, "t35": t35}
    for k, fn in tests.items():
        if not a.only or k in a.only.split(","):
            fn()
    fails = [n for n, ok in RESULTS if not ok]
    print(f"\nTIER 0: {len(RESULTS) - len(fails)}/{len(RESULTS)} passed"
          + ("" if not fails else f"   FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
