#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IO and run-management utilities.

Responsibilities kept here so no other module reinvents path/artifact logic:
  - load YAML config and apply dotted-key overrides from the CLI
  - create a timestamped, self-describing run directory
  - save/load the structured artifacts the instrumentation layer produces
    (JSON for records, .npz for arrays, .pt for tensors/state_dicts)

Design: the run directory is the durable record of an experiment. Everything
the run does that we might want to inspect later is written here as a file, so
analysis never has to re-run training.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------- config ----
def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config. A config may declare `inherit: <path>` (relative to its own
    directory): the parent is loaded first and this file is deep-merged over it.

    WHY. The baseline must be ONE file you can read in a minute and see exactly how
    it differs from the default. A full copy of default.yaml would be 264 lines that
    silently drift apart; an inheriting file is a short, auditable diff.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        ppath = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        cfg = _deep_merge(load_config(ppath), cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply CLI overrides of the form 'a.b.c=value' onto a nested config.

    Values are parsed as YAML so 'train.lr=0.0005', 'run.backbones=[efficientnet,mobilenet]',
    and 'data.use_validation=false' all do the right thing. This is how we sweep
    hyperparameters without editing the YAML or any .py file.
    """
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must be key=value")
        key, raw = ov.split("=", 1)
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


# ------------------------------------------------------------- run dirs ----
def make_run_dir(output_root: str, tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_root, f"{stamp}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------ artifacts ----
def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_arrays(path: str, **arrays) -> None:
    """Save named numpy arrays to a single .npz (e.g. scores=..., labels=...)."""
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)


def load_arrays(path: str):
    return np.load(path, allow_pickle=True)


def save_state(state_dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(state_dict, path)


def _json_default(o):
    """Make numpy/torch scalars JSON-serializable."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)
