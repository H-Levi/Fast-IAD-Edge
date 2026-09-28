#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVTec AD dataset (supervised wrapper).

Cleanups vs the original:
  - Built ONCE; train/val/test are materialized in __init__ and switched via
    set_mode. The original built two separate objects and relied on call-order
    luck to keep injected anomalies consistent between train and test.
  - Labels are read from the in-memory list via .labels(mode) -- the original
    read labels by calling __getitem__, which opened and transformed every
    image on disk just to count classes.
  - Anomaly injection is deterministic and the injected anomalies are removed
    from test, with the train/test split computed in the SAME object so they
    cannot desync.
  - Exposes .is_single_class(mode) so data/stats.py can flag the degenerate
    "normals-only training" case (MVTec with num_train_anomalies=0), where a
    binary classifier cannot learn anything.

Each item is (image_tensor, label, path). label: 0 normal, 1 anomaly.
"""

import os
import random
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DefectTypeCoverageError(RuntimeError):
    """Raised when a defect type present in a category has no sample in TRAIN."""
    pass


class MVTecDataset(Dataset):
    categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                  'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                  'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    def __init__(self, root, category='bottle', transform=None,
                 val_split=0.1, seed=123, num_train_anomalies=0,
                 use_validation=False, anomaly_fraction=None,
                 injection_val_share=0.5, val_share_mode="fixed",
                 val_anomaly_floor=10, val_share_min=0.5,
                 de_confound_frac=0.0, test_normal_floor=15, de_confound_val_share=1.0 / 3.0,
                 cache_size=None, calib_split=0.0, calib_n=0, calib_min_train=100):
        """anomaly_fraction: if set (e.g. 0.1), inject ceil(fraction * available)
        anomalies PER DEFECT TYPE instead of a fixed num_train_anomalies. Scales
        the bite to each defect folder's size, capped at availability, floored
        at 1. When None/0, the fixed num_train_anomalies path is used."""
        import math
        self.transform = transform
        self.category = category
        self._cache_size = int(cache_size) if cache_size else None
        rng = random.Random(seed)  # local RNG -> does not perturb global state

        def _take(available):
            if anomaly_fraction and anomaly_fraction > 0:
                return max(1, math.ceil(anomaly_fraction * available))
            return num_train_anomalies

        cat_path = os.path.join(root, category)

        # normal training images (label 0)
        good_dir = os.path.join(cat_path, 'train', 'good')
        train_normal = [(os.path.join(good_dir, f), 0)
                        for f in sorted(os.listdir(good_dir))]

        # all test images, grouped so we can inject deterministically
        test_root = os.path.join(cat_path, 'test')
        test_all: List[Tuple[str, int]] = []
        anomalies_by_type = {}
        for folder in sorted(os.listdir(test_root)):
            fdir = os.path.join(test_root, folder)
            if not os.path.isdir(fdir):
                continue
            label = 0 if folder == 'good' else 1
            items = [(os.path.join(fdir, f), label) for f in sorted(os.listdir(fdir))]
            test_all.extend(items)
            if folder != 'good':
                anomalies_by_type[folder] = items

        # ---- resolve the val share ----
        # 'fixed'    : use injection_val_share as given.
        # 'adaptive' : take from train ONLY until the declared val floor is met,
        #              then stop. Ten of fifteen MVTec categories already clear
        #              the floor at 0.5, so a flat higher share buys them nothing
        #              on val while halving train's within-type diversity (a
        #              defect type represented by ONE image, augmented xN, is
        #              still one instance seen N times — augmentation multiplies
        #              samples, not information).
        # NOTE: this optimises against val_anomaly_floor, which is itself a
        # PROVISIONAL constant (see config). The rule is principled given the
        # floor; the floor is not yet validated.
        val_share_resolved = float(injection_val_share)
        val_share_evidence = {"mode": val_share_mode, "requested": float(injection_val_share)}
        if use_validation and str(val_share_mode).lower() == "adaptive" and anomalies_by_type:
            def _val_count(share):
                v = 0
                for _t, _items in anomalies_by_type.items():
                    _c = min(_take(len(_items)) * 2, len(_items))
                    _nv = int(round(_c * share))
                    _nv = min(_nv, _c - 1) if _c > 1 else 0
                    v += max(0, _nv)
                return v
            lo = float(val_share_min)
            hi = float(injection_val_share)
            chosen, achieved = hi, _val_count(hi)
            s = lo
            while s <= hi + 1e-9:
                v = _val_count(s)
                if v >= int(val_anomaly_floor):
                    chosen, achieved = round(s, 4), v
                    break
                s += 0.05
            val_share_resolved = chosen
            val_share_evidence.update({
                "resolved": chosen,
                "val_anomalies_at_resolved": achieved,
                "floor": int(val_anomaly_floor),
                "floor_met": achieved >= int(val_anomaly_floor),
                "search_range": [lo, hi],
                "rule": "smallest share in range meeting val_anomaly_floor; "
                        "else the range maximum",
                "floor_is_provisional": True,
            })
        val_share = val_share_resolved
        self.val_share_evidence = val_share_evidence

        # ---- deterministic anomaly injection (per defect type) ----
        # INVARIANT (hard): injected anomalies are stratified by defect type, and
        # EVERY defect type present in the category must appear in TRAIN.
        # Oversampling multiplies samples, not defect types: if a type lands only
        # in val, no multiplier can recover it — the model never sees that defect
        # mode at all. Train coverage therefore outranks val share.
        train_inj, val_inj, removed = [], [], set()
        per_type_split = {}
        inject = (num_train_anomalies > 0) or (anomaly_fraction and anomaly_fraction > 0)
        if inject:
            for atype, items in anomalies_by_type.items():
                paths = [p for (p, _) in items]
                base = _take(len(paths))               # per-type count (fixed or fraction)
                if use_validation:
                    chosen = rng.sample(paths, min(base * 2, len(paths)))
                    c = len(chosen)
                    # val_share is a REQUEST, capped by the coverage invariant:
                    # train keeps at least one of every type, so val can take at
                    # most c-1. With c==1 the single sample goes to TRAIN.
                    n_val = int(round(c * val_share))
                    n_val = min(n_val, c - 1) if c > 1 else 0
                    n_val = max(0, n_val)
                    val_inj += [(p, 1) for p in chosen[:n_val]]
                    train_inj += [(p, 1) for p in chosen[n_val:]]
                    per_type_split[atype] = {"available": len(paths), "taken": c,
                                             "train": c - n_val, "val": n_val}
                else:
                    chosen = rng.sample(paths, min(base, len(paths)))
                    train_inj += [(p, 1) for p in chosen]
                    per_type_split[atype] = {"available": len(paths), "taken": len(chosen),
                                             "train": len(chosen), "val": 0}
                removed.update(chosen)
            test_all = [pair for pair in test_all if pair[0] not in removed]

            # Assert the invariant loudly rather than discovering it in a metric.
            missing = [t for t, d in per_type_split.items() if d["train"] < 1]
            if missing:
                raise DefectTypeCoverageError(
                    f"[{category}] STRATIFICATION VIOLATED: defect type(s) {missing} have no "
                    f"sample in TRAIN. Oversampling cannot recover a missing defect type. "
                    f"Lower data.injection_val_share (currently {val_share}) or raise "
                    f"data.anomaly_fraction so every type contributes to train.")

        # ---- ARM F: de-confound the folder -> label shortcut (F-L42/F-L48/F-L49) ----
        # MVTec ships a normal-only train split, so supervised protocols move DEFECTS out
        # of test/ and leave NORMALS in train/good. Provenance then predicts the label
        # perfectly in train and val. F-L48 measured the cost: carpet's test AUROC inverts
        # to 0.3157, and the inversion is ABSENT on untrained features, so training made it.
        #
        # THIS IS A SWAP, NOT A TRANSFER -- and v1 got that wrong (F-L50). Moving k
        # test/good images into train/val without replacing them deleted 43% of MVTec's
        # test normals, drove test prevalence up everywhere (transistor 34.8% -> 51.6%),
        # worsened mean ECE 0.185 -> 0.223, and cost six categories >0.025 AUROC. So we
        # also send k train/good images the other way:
        #   k test/good  -> train + val   (breaks the cue where it is learned and selected)
        #   k train/good -> test          (keeps test size and prevalence EXACTLY as-is)
        # Net: every split's normals span both acquisition sessions, no split changes size,
        # and the train normal count is unchanged. The train/good images sent to test were
        # never trained on -- they are removed from the pool before the train/val split.
        moved_train_norm, moved_val_norm = [], []
        dc = {"enabled": False, "requested_frac": float(de_confound_frac), "mode": "swap",
              "val_share": float(de_confound_val_share)}
        if de_confound_frac and de_confound_frac > 0:
            gsep = os.sep + "good" + os.sep
            test_good = sorted(p for (p, l) in test_all if l == 0 and gsep in p)
            train_good = sorted(p for (p, _) in train_normal)
            # k is capped by what test/good can give AND by not stripping train/good:
            # a category keeps at least three quarters of its original train normals.
            k = max(0, min(int(round(de_confound_frac * len(test_good))),
                           len(test_good), len(train_good) // 4))
            out_of_test = rng.sample(test_good, k) if k else []
            out_of_train = rng.sample(train_good, k) if k else []
            # 2:1 train:val by default (baseline-v1). `de_confound_val_share` 0.5 = 1:1 gives
            # validation more test/good normals, for the folder-weighted threshold (F-L79).
            # k/3 never lands on .5, so round(k * 1/3) == round(k / 3.0) for every k.
            n_val_n = int(round(k * float(de_confound_val_share)))
            moved_val_norm = [(p, 0) for p in out_of_test[:n_val_n]]
            moved_train_norm = [(p, 0) for p in out_of_test[n_val_n:]]
            _ot, _oi = set(out_of_test), set(out_of_train)
            train_normal = [pair for pair in train_normal if pair[0] not in _oi]
            test_all = ([pair for pair in test_all if pair[0] not in _ot]
                        + [(p, 0) for p in sorted(out_of_train)])
            dc.update({
                "enabled": k > 0, "k": k,
                "test_good_available": len(test_good), "train_good_available": len(train_good),
                "n_testgood_to_train": len(moved_train_norm),
                "n_testgood_to_val": len(moved_val_norm),
                "n_traingood_to_test": len(out_of_train),
                "test_size_unchanged": True,
                "skipped_reason": None if k > 0 else "no images available to swap",
                "testgood_moved_paths": sorted(os.path.relpath(p, cat_path) for p in out_of_test),
                "traingood_moved_paths": sorted(os.path.relpath(p, cat_path) for p in out_of_train),
            })
            if k == 0:
                print(f"  [{category}] ARM F SKIPPED: nothing available to swap.")
        self.de_confound = dc

        # §2.5 auditable injection manifest: exactly which files moved where.
        # Kept as an attribute so a run can record it; paths are relative to the
        # category dir to keep the record compact and portable.
        def _rel(p):
            return os.path.relpath(p, cat_path)
        self.injection_manifest = {
            "category": category,
            "de_confound": dc,
            "injected": bool(inject),
            "mode": ("fraction" if (anomaly_fraction and anomaly_fraction > 0)
                     else ("fixed_k" if num_train_anomalies > 0 else "none")),
            "anomaly_fraction": anomaly_fraction,
            "injection_val_share": val_share,
            "val_share_evidence": val_share_evidence,
            "num_train_anomalies": num_train_anomalies,
            "seed": seed,
            "per_type_available": {t: len(v) for t, v in anomalies_by_type.items()},
            "per_type_split": per_type_split,
            "train_defect_types": sorted(t for t, d in per_type_split.items() if d["train"] > 0),
            "val_defect_types": sorted(t for t, d in per_type_split.items() if d["val"] > 0),
            "n_defect_types": len(anomalies_by_type),
            "type_coverage_complete": all(d["train"] > 0 for d in per_type_split.values()),
            "n_injected_train": len(train_inj),
            "n_injected_val": len(val_inj),
            "n_removed_from_test": len(removed),
            "injected_train_paths": sorted(_rel(p) for (p, _) in train_inj),
            "injected_val_paths": sorted(_rel(p) for (p, _) in val_inj),
            "removed_from_test_paths": sorted(_rel(p) for p in removed),
        }

        # ---- validation split (normals split, injected anomalies appended) ----
        if use_validation and len(train_normal) > 1:
            tr_norm, va_norm = train_test_split(
                train_normal, test_size=val_split, random_state=seed)
            # Arm F normals are appended EXPLICITLY to each side rather than being mixed
            # into train_normal before the split: a random split could land all of them in
            # train, leaving val still perfectly separable by provenance -- which is the
            # exact thing this arm exists to break.
            # CALIBRATION SPLIT (analysis P7/P4, 2026-09-23). A fraction of the TRAINING normal pool is held out:
            # never trained on, never used to select the epoch, only used afterwards to set the decision threshold.
            # Drawn AFTER the defect draw and the train/val split, with its own random_state, so neither moves.
            # calib_split = 0 (default) -> this block is skipped and every split is bit-identical to before.
            self.calib_data = []
            from data.calib import calib_size
            _size, self.calib_info = calib_size(len(tr_norm) + len(moved_train_norm), calib_split, calib_n, calib_min_train)
            if _size:
                _pool = tr_norm + moved_train_norm
                _, self.calib_data = train_test_split(_pool, test_size=_size, random_state=seed + 1)
                _cal = {q for (q, _) in self.calib_data}
                tr_norm = [x for x in tr_norm if x[0] not in _cal]
                moved_train_norm = [x for x in moved_train_norm if x[0] not in _cal]
            self.train_data = tr_norm + moved_train_norm + train_inj
            self.val_data = va_norm + moved_val_norm + val_inj
        else:
            self.train_data = train_normal + moved_train_norm + train_inj
            self.val_data = []
            self.calib_data = []
            self.calib_info = {"applied": False, "mode": "off (no validation split)"}

        self.test_data = test_all
        self.mode = "train"
        # STRATIFICATION BASIS: defect type (stronger than label alone).
        # Injection is per defect-type folder and the coverage invariant
        # guarantees every type present appears in TRAIN.
        self.stratification = {
            "basis": "defect_type",
            "type_coverage_guaranteed": bool(inject),
            "reason": ("per-type injection with train-coverage invariant" if inject
                       else "no injection performed"),
        }

    # -------------------------------------------------------------- api ----
    def set_mode(self, mode='train'):
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        return self

    def _bucket(self):
        return {'train': self.train_data, 'val': self.val_data,
                'test': self.test_data}[self.mode]

    def records(self, mode=None):
        """Return list of (absolute_path, label) for a split. Used by SplitView.

        MVTec stores absolute paths already, so this is a direct hand-off; the
        injection/removal was resolved once in __init__, so train/val/test here
        are mutually consistent (no path appears in two splits)."""
        m = mode or self.mode
        return {'train': self.train_data, 'val': self.val_data,
                'test': self.test_data, 'calib': self.calib_data}[m]

    def labels(self, mode=None) -> List[int]:
        """Return labels WITHOUT loading images (fast class-balance reads)."""
        return [lbl for (_, lbl) in self.records(mode)]

    def paths(self, mode=None) -> List[str]:
        m = mode or self.mode
        data = {'train': self.train_data, 'val': self.val_data,
                'test': self.test_data, 'calib': self.calib_data}[m]
        return [p for (p, _) in data]

    def is_single_class(self, mode=None) -> bool:
        labs = set(self.labels(mode))
        return len(labs) < 2

    def __len__(self):
        return len(self._bucket())

    def __getitem__(self, index):
        path, label = self._bucket()[index]
        # Decoded-image cache: OFF unless data.cache_decoded is set. Sits BEFORE the
        # transform, so filters/augmentation/ToTensor/Normalize all still run fresh;
        # proved bit-identical to the uncached path. See data/imgcache.py.
        if self._cache_size:
            from data import imgcache
            image = imgcache.load(path, self._cache_size)
        else:
            image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path
