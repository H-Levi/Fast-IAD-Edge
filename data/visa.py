#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VisA dataset (supervised wrapper).

VisA is natively supervised: its split CSV labels normal vs anomaly in the
train split, so a binary classifier is well-posed here (unlike MVTec without
injection). This is the dataset whose numbers we trust most.

One object owns ALL splits (train/val/test), built from the split CSV in a
single pass, so records(mode) is internally consistent and no second object is
needed. Requires the spot-diff preprocessing layout:
    {root}/split_csv/2cls_{shot}.csv
with each row giving an `image` path (relative to root) and a `label`.

Returns (image, label, path); records(mode) returns (absolute_path, label).
"""

from typing import List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ViSADataset(Dataset):
    categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                  'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
                  'pipe_fryum']

    def __init__(self, path: str, category: str, type: str = 'train',
                 shot: str = 'highshot', transform=None,
                 val_split=0.1, use_validation=False, seed=123,
                 cache_size=None, calib_split=0.0, calib_n=0, calib_min_train=100, train_anomaly_n=0):
        super().__init__()
        self._cache_size = int(cache_size) if cache_size else None
        self.path = path
        self.transform = transform  # kept for back-compat; SplitView supplies its own

        data = pd.read_csv(f"{path}/split_csv/2cls_{shot}.csv")
        data.label = data.label.apply(lambda x: 0 if x == 'normal' else 1)

        cat = data[data.object == category]
        train_rows = cat[cat.split == 'train']
        test_rows = cat[cat.split == 'test']

        # STRATIFICATION BASIS: label only (normal vs anomaly).
        # VisA's 2cls split CSV carries columns {object, split, label, image}
        # and its anomalies live in a single Anomaly/ folder per category — the
        # defect TYPE is not exposed anywhere in this format. So unlike MVTec
        # (where injection is stratified per defect-type folder and every type is
        # guaranteed a train sample), VisA can only be stratified by label here.
        # This is a dataset-format limitation, not an oversight: it is recorded
        # in .stratification so a report never implies type coverage it lacks.
        self.stratification = {
            "basis": "label",
            "type_coverage_guaranteed": False,
            "reason": "VisA 2cls split CSV exposes no defect-type column",
        }
        if use_validation and len(train_rows) > 1 and train_rows.label.nunique() > 1:
            tr_idx, va_idx = train_test_split(
                train_rows.index.tolist(), test_size=val_split,
                random_state=seed, stratify=train_rows.label.values)
            self.train_data = data.loc[tr_idx].reset_index(drop=True)
            self.val_data = data.loc[va_idx].reset_index(drop=True)
        else:
            self.train_data = train_rows.reset_index(drop=True)
            self.val_data = pd.DataFrame(columns=data.columns)

        self.test_data = test_rows.reset_index(drop=True)
        # CALIBRATION SPLIT (see data/mvtec.py): held out from the TRAINING normals after the train/val split, own
        # random_state; calib_split = 0 (default) leaves every split bit-identical.
        self.calib_data = pd.DataFrame(columns=data.columns)
        self.calib_info = {"applied": False, "mode": "off"}
        if use_validation:
            from data.calib import calib_size
            _tn = self.train_data[self.train_data.label == 0].index.tolist()
            _size, self.calib_info = calib_size(len(_tn), calib_split, calib_n, calib_min_train)
        if use_validation and _size:
            _, _cal = train_test_split(_tn, test_size=_size, random_state=seed + 1)
            self.calib_data = self.train_data.loc[_cal].reset_index(drop=True)
            self.train_data = self.train_data.drop(index=_cal).reset_index(drop=True)
        # TRAINING-DEFECT COUNT (pilot F-L105, scarcity dose-response): keep only N of the TRAINING anomalies, chosen
        # with their own random_state; the rest are DROPPED, not moved, so validation, test, calibration and training
        # normals are identical to a run without the key and every comparison stays paired. 0 (default) = off.
        self.train_anomaly_info = {"applied": False}
        if train_anomaly_n:
            _ta = self.train_data[self.train_data.label == 1].index.tolist()
            if train_anomaly_n > len(_ta):
                raise ValueError(f"train_anomaly_n={train_anomaly_n} > {len(_ta)} training anomalies available ({category})")
            _keep, _ = (train_test_split(_ta, train_size=train_anomaly_n, random_state=seed + 2)
                        if train_anomaly_n < len(_ta) else (_ta, []))
            self.train_data = self.train_data.drop(index=sorted(set(_ta) - set(_keep))).reset_index(drop=True)
            self.train_anomaly_info = {"applied": True, "n": int(train_anomaly_n), "available": len(_ta)}
        self.mode = 'train'

    # -------------------------------------------------------------- api ----
    def set_mode(self, mode='train'):
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        return self

    def _frame(self, mode=None):
        m = mode or self.mode
        return {'train': self.train_data, 'val': self.val_data,
                'test': self.test_data, 'calib': self.calib_data}[m]

    def records(self, mode=None):
        """List of (absolute_path, label) for a split. Used by SplitView."""
        df = self._frame(mode)
        return [(f'{self.path}/{img}', int(lbl))
                for img, lbl in zip(df.image, df.label)]

    def labels(self, mode=None) -> List[int]:
        return [lbl for (_, lbl) in self.records(mode)]

    def paths(self, mode=None) -> List[str]:
        return [p for (p, _) in self.records(mode)]

    def is_single_class(self, mode=None) -> bool:
        return len(set(self.labels(mode))) < 2

    def __len__(self):
        return len(self._frame())

    def __getitem__(self, index):
        row = self._frame().iloc[index]
        if getattr(self, "_cache_size", None):
            from data import imgcache
            image = imgcache.load(f'{self.path}/{row.image}', self._cache_size)
        else:
            image = Image.open(f'{self.path}/{row.image}').convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(row.label), row.image
