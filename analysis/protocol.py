#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frozen protocol spec (§1.5).

A "protocol" is the set of evaluation rules that must be IDENTICAL for two runs
to be comparable: split ratios, MVTec injection fraction, the metric set, the
per-tier seed lists, and the reporting granularity. If two runs differ in any of
these, their numbers are not on the same footing and must NOT be compared.

Design choice (deliberate, flagged): rather than DUPLICATE these values into a
separate `protocol:` block — which would drift from the real keys (the exact
config-vs-briefing drift already warned about) — the fingerprint is DERIVED from
the actual config keys the run uses (data.val_split, data.anomaly_fraction,
eval.metrics, seeding.*). The `protocol:` block in config holds only identity
(name) and reporting granularity, which have no other home. So there is one
source of truth for each value, and the fingerprint is a read of it.

The fingerprint deliberately EXCLUDES the things experiments vary (model.*,
train.*, filters.*, augmentation.*) — those are the change under test; the
protocol is the fixed backdrop they are measured against.
"""

import json
from typing import Dict


def protocol_fingerprint(cfg: dict) -> Dict:
    """Canonical dict of the protocol-defining keys, read from the real config."""
    data = cfg.get("data", {})
    proto = cfg.get("protocol", {})
    seeding = cfg.get("seeding", {})
    return {
        "name": proto.get("name", "unnamed"),
        "val_split": data.get("val_split"),
        "anomaly_fraction": data.get("anomaly_fraction"),
        "num_train_anomalies": data.get("num_train_anomalies"),
        # injection_val_share decides how the SAME injected anomalies are divided
        # between train and val. Two runs that split them differently are not on
        # the same evaluation footing, so it belongs in the fingerprint.
        # In adaptive mode the share is PER CATEGORY, so the fingerprint must
        # lock the RULE (mode + bounds + floor), not one resolved number —
        # otherwise two runs following the same rule would look incomparable.
        "val_share_mode": data.get("val_share_mode"),
        "val_share_bounds": [data.get("val_share_min"), data.get("injection_val_share")],
        "val_anomaly_floor": data.get("val_anomaly_floor"),
        "floors_validated": data.get("floors_validated"),
        "injection_val_share": (data.get("injection_val_share")
                                if data.get("val_share_mode") != "adaptive" else "per-category (adaptive)"),
        "injection_val_share_justification": proto.get(
            "injection_val_share_justification"),
        "use_validation": data.get("use_validation"),
        "metrics": sorted(cfg.get("eval", {}).get("metrics", [])),
        "exploration_seeds": list(seeding.get("exploration_seeds", [])),
        "claim_seeds": list(seeding.get("claim_seeds", [])),
        "reporting": proto.get("reporting", "per_category"),
    }


def protocols_match(cfg_a: dict, cfg_b: dict) -> bool:
    return protocol_fingerprint(cfg_a) == protocol_fingerprint(cfg_b)


def protocol_diff(cfg_a: dict, cfg_b: dict) -> Dict:
    """Return the keys where the two protocols differ (for a clear error)."""
    fa, fb = protocol_fingerprint(cfg_a), protocol_fingerprint(cfg_b)
    return {k: (fa[k], fb[k]) for k in fa if fa[k] != fb[k]}


class ProtocolMismatch(RuntimeError):
    pass


def assert_same_protocol(cfg_a: dict, cfg_b: dict, ctx: str = "") -> None:
    """Raise if the two configs' protocols differ. Used by the A/B comparator so
    it refuses to compare runs that aren't on the same evaluation footing."""
    diff = protocol_diff(cfg_a, cfg_b)
    if diff:
        lines = "\n".join(f"    {k}: {a}  vs  {b}" for k, (a, b) in diff.items())
        raise ProtocolMismatch(
            f"PROTOCOL MISMATCH{(' (' + ctx + ')') if ctx else ''}: the two runs are not "
            f"comparable — they differ in protocol-defining keys:\n{lines}\n"
            f"  Change only the thing under test; keep val_split / anomaly_fraction / "
            f"metrics / seeds identical.")
