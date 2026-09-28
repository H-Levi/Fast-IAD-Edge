#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeds & noise band (§1.3).

Two-tier seeding so scouting is never mistaken for a claim:
  - exploration tier: a single seed. Fast. For scouting whether an idea is worth
    pursuing. NOT reportable.
  - claim tier: 3 seeds (field norm). Any number you would put in a paper.

And a NOISE BAND: run the identical config under N seeds; the spread of results
across seeds is the noise floor. A difference between two configs smaller than
this band is NOT real — it is seed jitter. This is the antidote to reading a
single-seed 0.985-vs-0.982 gap as meaningful.

resolve_seeds(cfg) reads the `seeding:` config block and returns (tier, seed_list).
noise_band(values) turns a list of per-seed scores into mean/std/range/band.
"""

from typing import List, Tuple

import numpy as np


def resolve_seeds(cfg: dict) -> Tuple[str, List[int]]:
    """Return (tier, seed_list) from cfg['seeding'].

    Back-compatible: if no `seeding` block, fall back to cfg['seeds'] or [cfg['seed']].
    """
    sd = cfg.get("seeding")
    if sd:
        tier = sd.get("tier", "exploration")
        if tier == "claim":
            seeds = sd.get("claim_seeds", [cfg.get("seed", 123)])
        else:
            seeds = sd.get("exploration_seeds", [cfg.get("seed", 123)])
        return tier, list(seeds)
    # legacy fallback
    return "exploration", list(cfg.get("seeds", [cfg.get("seed", 123)]))


def noise_band(values: List[float]) -> dict:
    """Summarize per-seed scores into a noise band.

    band := SAMPLE standard deviation across seeds (ddof=1), the jitter a single
    seed hides. Also reports range (max-min) as a conservative alternative.

    Why ddof=1, not numpy's default. np.std divides by N (the population formula),
    which assumes these seeds ARE the whole population. They are a sample drawn from
    all the seeds we could have run, so the divisor is N-1. At the 3 seeds used for
    claim tier the difference is sqrt(3/2) = 1.22x -- the population form reports a
    band 22% narrower than reality, and a too-narrow band makes differences look more
    significant than they are. That is the dangerous direction to be wrong in.

    Also reports sem, the standard error of the mean (band / sqrt(n)), which is what
    a comparison of two means should actually be judged against -- see delta_verdict.
    """
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "band": None, "range": None, "sem": None}
    arr = np.asarray(vals, dtype=np.float64)
    # A band needs >=2 seeds; with one seed the spread is undefined, NOT zero.
    band = float(arr.std(ddof=1)) if arr.size >= 2 else None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": band,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "band": band,                      # None with <2 seeds -> delta verdicts say "NO BAND"
        "sem": (band / np.sqrt(arr.size)) if band is not None else None,
        "range": float(arr.max() - arr.min()) if arr.size >= 2 else None,
    }


def delta_verdict(delta: float, band: float, n_a: int = None, n_b: int = None,
                  band_b: float = None, k: float = 2.0) -> str:
    """Is a difference between two configs bigger than seed noise?

    The old rule was `REAL if |delta| > band`, comparing a difference of two MEANS
    against one sample's standard DEVIATION. Those are different quantities, and the
    comparison is far too permissive: averaging n seeds shrinks the uncertainty of a
    mean by sqrt(n), so at 3 seeds the relevant scale is already ~0.58x the deviation,
    and the uncertainty of a *difference* of two such means is
    se_delta = sqrt(sd_a^2/n_a + sd_b^2/n_b). Judging |delta| against a raw sd
    therefore stamped REAL on differences well inside the noise -- again erring toward
    over-claiming.

    Now: REAL when |delta| exceeds k standard errors of the difference (k=2, the usual
    ~95% convention). With counts unavailable the old single-band comparison is used,
    but the verdict says so rather than pretending to a rigour it does not have.
    """
    if band is None:
        return "NO BAND (need >=2 seeds)"
    if n_a and n_b:
        sd_b = band if band_b is None else band_b
        se = float(np.sqrt(band ** 2 / n_a + sd_b ** 2 / n_b))
        if se <= 0:
            return "NO BAND (zero spread)"
        z = abs(delta) / se
        return (f"REAL ({z:.1f} SE)" if z > k
                else f"NOT DISTINGUISHABLE ({z:.1f} SE, need >{k:g})")
    return ("REAL (vs raw sd — weak test, seed counts unavailable)" if abs(delta) > band
            else "NOT DISTINGUISHABLE (vs raw sd)")
