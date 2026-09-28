#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anomaly multiplier derivation (§4.1 stages 1 & 3).

The oversampling factor N is DERIVED from each category's real anomaly count,
never hardcoded and never tuned on a score:

    stage 1  read the per-category anomaly count from the §2.1 census
    stage 2  (empirical, elsewhere) sweep the TARGET effective-anomalies-per-epoch
             on the worst category; keep the smallest target that still helps
    stage 3  N_category = ceil(target / anomaly_count_category)

Because N follows from data scarcity rather than from validation score, it is
per-category AND principled: a category with 6 real anomalies gets x5, one with
40 gets x1, automatically. That is reportable ("multiplier set so every category
sees ~T anomalies per epoch") rather than "we tuned N per category", which would
be selection on the test signal in disguise.

The count used is the TRAIN-split anomaly count (what the epoch actually sees),
not the dataset's total anomalies.
"""

import json
import math
import os
from typing import Dict, Optional


def load_census(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def find_census(run_root: str) -> Optional[str]:
    """Newest census.json under a runs root, if one exists."""
    hits = []
    for root, _dirs, files in os.walk(run_root):
        if "census.json" in files:
            hits.append(os.path.join(root, "census.json"))
    return sorted(hits)[-1] if hits else None


def train_anomaly_counts(census: Dict) -> Dict[str, int]:
    """{'<dataset>/<category>': train-split anomaly count} from a census."""
    counts = {}
    for cat, r in (census.get("mvtec") or {}).items():
        sp = (r.get("splits_at_locked_fraction") or {}).get("train") or {}
        if "anomaly" in sp:
            counts[f"mvtec/{cat}"] = int(sp["anomaly"])
    for cat, r in (census.get("visa") or {}).items():
        if "error" in r:
            continue
        sp = (r.get("splits") or {}).get("train") or {}
        if "anomaly" in sp:
            counts[f"visa/{cat}"] = int(sp["anomaly"])
    return counts


def derive_multiplier(anomaly_count: int, target: int, cap: int = 10) -> int:
    """(see module docstring). NOTE: the value is PROVISIONAL until the §4.1.2
    target sweep has validated `target` — see `multiplier_table(..., target_validated=)`."""
    """N = ceil(target / count), floored at 1 and capped.

    The cap stops a pathologically small category (3 anomalies) from being
    duplicated 20x, which would show the model the same three images over and
    over rather than genuine variety.
    """
    if anomaly_count is None or anomaly_count <= 0:
        return 1
    return max(1, min(cap, math.ceil(target / anomaly_count)))


def multiplier_table(census: Dict, target: int, cap: int = 10,
                     target_validated: bool = False) -> Dict[str, Dict]:
    """Per-category N with the evidence that produced it.

    target_validated=False (the default) marks every N as PROVISIONAL: the
    §4.1.2 target sweep has not run, so `target` is a placeholder and each
    derived N is conditional on it. Nothing downstream should treat these as
    settled until the sweep confirms the smallest target that still helps.
    """
    out = {}
    for key, n_anom in train_anomaly_counts(census).items():
        N = derive_multiplier(n_anom, target, cap)
        out[key] = {
            "train_anomalies": n_anom,
            "multiplier": N,
            "effective_anomalies_per_epoch": n_anom * N,
            "target": target,
            "capped": (N == cap and math.ceil(target / max(n_anom, 1)) > cap),
            "provisional": (not target_validated) or (N == cap),
            "cap": cap,
            # cap_binding means the cap ACTUALLY reduced N below what the target
            # required — not merely that N landed on the cap value.
            "cap_binding": (N == cap and math.ceil(target / max(n_anom, 1)) > cap),
            "shortfall": max(0, target - n_anom * N),
            "basis": ("target validated by §4.1.2 sweep" if target_validated
                      else "PROVISIONAL — target is a placeholder; §4.1.2 sweep not yet run"),
            "cap_basis": ("PROVISIONAL — cap is an unvalidated guess; where it binds it, "
                          "not the target, decides the exposure"),
        }
    return out


def multiplier_for(census: Dict, dataset: str, category: str,
                   target: int, cap: int = 10) -> int:
    """Provisional unless the target has been validated by the §4.1.2 sweep."""
    counts = train_anomaly_counts(census)
    return derive_multiplier(counts.get(f"{dataset}/{category}"), target, cap)


def render_table(table: Dict[str, Dict]) -> str:
    first = next(iter(table.values())) if table else {}
    provisional = first.get("provisional", True)
    banner = ("  *** PROVISIONAL: BOTH the target and the multiplier cap are unvalidated.\n"
              "  *** target: the §4.1.2 sweep has not run, so every N is conditional on it.\n"
              "  *** cap: an unvalidated guess. Where the cap BINDS (marked 'capped'), the cap\n"
              "  ***      — not the target — decides that category's exposure, so its effective\n"
              "  ***      anomalies fall short of the target by construction.")
    L = ["=" * 78,
         f"ANOMALY MULTIPLIER (derived from census; target={first.get('target', '?')})",
         "=" * 78]
    if provisional:
        L += [banner, "=" * 78]
    L.append(f"  {'category':<24}{'train_anom':>11}{'N':>5}{'effective':>11}  note")
    for key, r in sorted(table.items()):
        notes = []
        if r["capped"]:
            notes.append("capped")
        if r.get("provisional"):
            notes.append("provisional")
        L.append(f"  {key:<24}{r['train_anomalies']:>11}{r['multiplier']:>5}"
                 f"{r['effective_anomalies_per_epoch']:>11}  {','.join(notes)}")
    return "\n".join(L)
