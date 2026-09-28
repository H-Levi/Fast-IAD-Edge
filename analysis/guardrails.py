#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Guardrails — pure protection, adds nothing to the forward pass.

§1.1 leak self-check: a tripwire that runs after splits are built and before test
scoring. It does not change behavior; it HALTS the run if a leak-defining
invariant is violated, so a broken split/threshold/transform can never silently
produce a trusted number.

What it asserts (BRIEFING §3, §5):
  1. No image path appears in BOTH train and test (after MVTec injection removal).
  2. The threshold used to score test == the val-selected threshold (guards against
     future code recomputing a threshold on test).
  3. The val loader's transform is the SAME object as the test loader's transform
     (i.e. val is not augmented; both use the non-augmented test transform).

Scope note (deliberate): these three asserts guard SPLIT INTEGRITY and the
threshold/transform binding — accidental contamination. They do NOT flag the
deliberate, labeled `use_validation=false` ablation (which binds val to test on
purpose); when that path is active, val==test and the asserts pass trivially.
To keep it non-silent, the check emits a WARNING (not a halt) when
use_validation is false, so selection-on-test is always surfaced.
"""

from typing import List, Optional


class LeakCheckError(RuntimeError):
    """Raised when a leak-defining invariant is violated. Halts the run."""
    pass


def leak_self_check(*, train_paths: List[str], test_paths: List[str],
                    val_threshold: float, test_threshold: float,
                    val_transform, test_transform,
                    use_validation: bool, key: str = "",
                    val_paths: Optional[List[str]] = None,
                    calib_paths: Optional[List[str]] = None) -> dict:
    """Run the three asserts. Raise LeakCheckError on any violation.

    Returns a small dict describing what was checked (for the record). Pass the
    ACTUAL variables the run uses — e.g. test_threshold should be the exact value
    handed to the test-scoring call, and val/test_transform the transforms bound
    to the respective loaders — so the tripwire catches real regressions.
    """
    problems = []

    # 1) train/test path overlap
    overlap = set(train_paths) & set(test_paths)
    if overlap:
        sample = list(overlap)[:3]
        problems.append(f"{len(overlap)} path(s) in BOTH train and test (leak): {sample}")

    # 1b) val must be disjoint from BOTH. Validation picks the epoch and the threshold, so
    # a test image in val leaks test into selection, and a train image in val inflates it.
    # Before 2026-09-22 only train/test was checked (gap found by the other session).
    if val_paths is not None:
        for name, other in (("train", train_paths), ("test", test_paths)):
            ov = set(val_paths) & set(other)
            if ov:
                problems.append(f"{len(ov)} path(s) in BOTH val and {name} (leak): {list(ov)[:3]}")

    # 1c) the calibration split (analysis P7) must be disjoint from train, val AND test: it sets the threshold,
    # so any overlap would leak either training fit or test labels into the decision.
    if calib_paths:
        for name, other in (("train", train_paths), ("val", val_paths or []), ("test", test_paths)):
            ov = set(calib_paths) & set(other)
            if ov:
                problems.append(f"{len(ov)} path(s) in BOTH calib and {name} (leak): {list(ov)[:3]}")

    # 2) test threshold must equal the val-selected threshold
    if val_threshold != test_threshold:
        problems.append(
            f"test threshold ({test_threshold}) != val-selected threshold ({val_threshold})")

    # 3) val transform must be the SAME object as test transform (val not augmented)
    if val_transform is not test_transform:
        problems.append(
            "val transform is not the same object as the test transform "
            "(val may be augmented / bound to the train transform)")

    if problems:
        raise LeakCheckError(
            f"[leak_self_check{(' ' + key) if key else ''}] LEAK GUARD FAILED: "
            + " | ".join(problems))

    warning = None
    if not use_validation:
        warning = (f"[leak_self_check{(' ' + key) if key else ''}] use_validation=false: "
                   "selection is on TEST (labeled ablation) — leak-free property WAIVED.")
        print("  ! " + warning)

    return {
        "leak_check": "passed",
        "train_test_overlap": 0,
        "threshold_bound_to_val": True,
        "val_transform_is_test_transform": True,
        "use_validation": use_validation,
        "warning": warning,
    }


def overlap_only(train_paths: List[str], test_paths: List[str]) -> int:
    """Convenience for callers (e.g. smoke_test) that only want the overlap count."""
    return len(set(train_paths) & set(test_paths))


# ---------------------------------------------------------------------------
# §1.2 COLLAPSE DETECTOR — pure protection, reads the per-epoch logit gap.
# A run whose logit gap (anomaly_mean - normal_mean) never moves off ~0.00 has
# not learned to separate the classes: its output is effectively random and its
# AUROC must NOT be read as a real result. This function turns that trajectory
# into a `collapsed` boolean + evidence, for report.txt and summary.json.
# ---------------------------------------------------------------------------
def detect_collapse(hist_records: list, eps: float = 0.1,
                    selected_epoch: int = None,
                    sign_min: float = 0.75, d_prime_min: float = 0.35) -> dict:
    """Did the model that was ACTUALLY SCORED separate the classes?

    Rule (revised after a real baseline run exposed the old one): judge the
    logit gap at the SELECTED epoch — that checkpoint is the model which gets
    loaded and scored on test. The previous rule used max |gap| across all
    epochs, which let a single noise spike clear the threshold: a VisA/squeezenet
    run oscillating around zero (gaps .05 .09 .00 .08 .00 .14 .14 .08 -.00 .16
    .10 .12) hit max .1599, was stamped "separated", and its chance-level AUROC
    (.4958) counted toward a tier mean.

    Judging the selected epoch is both stricter on that case (selected gap .093
    < eps) and safer on late-blooming runs than a median rule would be — a run
    that only separates in its final epochs is correctly NOT flagged, because
    selection picks one of those epochs.

    max and median |gap| are still computed and returned as evidence.

    collapsed is None when there is not enough information — never silently False.

    TWO SCALE-FREE CRITERIA (added after eps=0.1 waved a dead run through)
    ----------------------------------------------------------------------
    An absolute gap threshold is the wrong instrument, because logit scale varies
    by two orders of magnitude between runs: healthy runs here reach gaps of 7-13,
    dead ones oscillate around 0.2-0.5. eps=0.1 therefore only catches the very
    deadest. An LSE run on mvtec/screw was stamped "separated" on a selected gap of
    0.2262 while its trajectory read -0.027 0.226 0.325 0.057 -0.218 -0.522 0.349
    0.232 -0.302 0.076 0.016 -0.042 — five of twelve epochs scored anomalies BELOW
    normals. That is a dead run by inspection and the detector missed it.

    So two further signals, both free from data already recorded:

    sign_consistency  fraction of epochs with gap > 0. A learning run is ~1.0; a
                      run wandering around zero is ~0.5. Scale-free, and it is what
                      makes the screw case obvious.
    d_prime           gap divided by the pooled within-class spread at the selected
                      epoch — the separation in units of its own noise. Scale-free,
                      and the recorder already stores each class's std.

    A run is collapsed if ANY criterion fires. Each is reported separately so the
    reason is always attributable rather than a bare boolean.
    """
    gaps, stds = [], {}
    for r in hist_records:
        nrec, arec = (r.get("normal") or {}), (r.get("anomaly") or {})
        nm, an = nrec.get("mean"), arec.get("mean")
        if nm is not None and an is not None:
            ep = r.get("epoch")
            gaps.append((ep, an - nm))
            ns, a_s = nrec.get("std"), arec.get("std")
            if ns is not None and a_s is not None:
                stds[ep] = (float(ns), float(a_s))

    if not gaps:
        return {"collapsed": None, "max_abs_gap": None, "median_abs_gap": None,
                "selected_gap": None, "final_gap": None, "n_epochs": 0,
                "sign_consistency": None, "d_prime": None,
                "eps": eps, "sign_min": sign_min, "d_prime_min": d_prime_min,
                "rule": "selected_epoch", "triggered": [],
                "reason": "no epochs with both class means"}

    vals = [g for (_, g) in gaps]
    abs_vals = sorted(abs(g) for g in vals)
    n = len(abs_vals)
    median_abs = (abs_vals[n // 2] if n % 2 else
                  0.5 * (abs_vals[n // 2 - 1] + abs_vals[n // 2]))
    max_abs = max(abs_vals)

    sel_gap = None
    if selected_epoch is not None:
        for (e, g) in gaps:
            if e == selected_epoch:
                sel_gap = g
                break

    # scale-free signal 1: does the gap even keep its sign?
    sign_consistency = sum(1 for g in vals if g > 0) / float(n)

    # scale-free signal 2: separation measured in units of its own spread
    d_prime = None
    if sel_gap is not None and selected_epoch in stds:
        ns, a_s = stds[selected_epoch]
        pooled = ((ns ** 2 + a_s ** 2) / 2.0) ** 0.5
        if pooled > 1e-12:
            d_prime = abs(sel_gap) / pooled

    fired = []
    if sel_gap is not None:
        if abs(sel_gap) < eps:
            fired.append(f"selected-epoch |gap| {abs(sel_gap):.4f} < eps {eps}")
        rule = "selected_epoch+scalefree"
        basis = f"selected-epoch |gap| {abs(sel_gap):.4f}"
    else:
        if median_abs < eps:
            fired.append(f"median |gap| {median_abs:.4f} < eps {eps} (no selected epoch)")
        rule = "median_fallback+scalefree"
        basis = f"median |gap| {median_abs:.4f}"

    # Only meaningful with enough epochs to see a trend; below that a run is simply
    # short, not necessarily dead.
    if n >= 5 and sign_consistency < sign_min:
        fired.append(f"gap changed sign in {(1-sign_consistency)*100:.0f}% of epochs "
                     f"(consistency {sign_consistency:.2f} < {sign_min})")
    if d_prime is not None and d_prime < d_prime_min:
        fired.append(f"d' {d_prime:.3f} < {d_prime_min} (separation smaller than its own spread)")

    collapsed = bool(fired)

    return {
        "collapsed": collapsed,
        "max_abs_gap": float(max_abs),
        "median_abs_gap": float(median_abs),
        "selected_gap": (float(sel_gap) if sel_gap is not None else None),
        "final_gap": float(vals[-1]),
        "sign_consistency": float(sign_consistency),
        "d_prime": (float(d_prime) if d_prime is not None else None),
        "n_epochs": n,
        "eps": eps,
        "sign_min": sign_min,
        "d_prime_min": d_prime_min,
        "rule": rule,
        "triggered": fired,
        "reason": ("; ".join(fired) + " -> scored model did not separate"
                   if collapsed else f"{basis}, sign consistency {sign_consistency:.2f}"
                                     + (f", d' {d_prime:.3f}" if d_prime is not None else "")
                                     + " -> separated"),
    }


def detect_collapse_from_file(path: str, eps: float = 0.1,
                              selected_epoch: int = None,
                              sign_min: float = 0.75,
                              d_prime_min: float = 0.35) -> dict:
    """Convenience: read a logit_hist.jsonl file and run detect_collapse."""
    import json as _json
    import os as _os
    recs = []
    if _os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(_json.loads(line))
    return detect_collapse(recs, eps=eps, selected_epoch=selected_epoch,
                           sign_min=sign_min, d_prime_min=d_prime_min)


# ---------------------------------------------------------------------------
# §4.1.4 VERIFICATION GATE — no scenario is trained on before it is eye-checked.
# The spec makes visual verification a STAGE, not a suggestion: "A scenario that
# fails here is fixed or dropped — it is NOT cached." Enforcing that by
# discipline alone means an unverified scenario can silently reach training, so
# it is enforced structurally here instead.
# ---------------------------------------------------------------------------
VERIFICATION_FILENAME = "augmentation_verified.json"


def verification_record_path(run_root: str) -> str:
    import os
    return os.path.join(run_root, VERIFICATION_FILENAME)


def load_verified_scenarios(run_root: str) -> dict:
    """{scenario: {'verified': bool, 'by': str, 'notes': str}} or {}."""
    import json
    import os
    p = verification_record_path(run_root)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


class UnverifiedScenarioError(RuntimeError):
    pass


def assert_scenario_verified(scenario: str, run_root: str,
                             require: bool = True) -> dict:
    """Halt if `scenario` has not been visually verified.

    'original' is exempt: it applies no augmentation, so there is nothing to
    inspect. Every other scenario must appear in the verification record with
    verified=true, which the contact-sheet reporter writes once a human has
    looked at the sheets and confirmed the four checks (defect survived,
    realistic, class-split correct, variety real).

    require=False downgrades the halt to a warning, for deliberate exploration.
    """
    s = (scenario or "original").lower()
    if s == "original":
        return {"scenario": s, "verified": True, "exempt": True}

    from analysis.aug_verify import STRICT_SCENARIOS

    rec = load_verified_scenarios(run_root).get(s)
    strict = s in STRICT_SCENARIOS

    if rec:
        visual_ok = bool(rec.get("verified"))
        quant = rec.get("quantitative") or {}
        quant_ok = bool(quant.get("passed"))
        # A risky tier must ALSO declare that it was held to strict thresholds.
        tier_ok = (not strict) or (quant.get("threshold_tier") == "strict")
        if visual_ok and quant_ok and tier_ok:
            return {"scenario": s, "verified": True, "strict": strict, "record": rec}

        missing = []
        if not visual_ok:
            missing.append("visual (contact sheets not confirmed)")
        if not quant_ok:
            failed = quant.get("failed_checks")
            missing.append(f"quantitative ({'failed: ' + ', '.join(failed) if failed else 'not run'})")
        if not tier_ok:
            missing.append("strict-threshold tier not applied to a RISKY scenario")
        msg = (f"AUGMENTATION NOT VERIFIED: scenario '{s}' has an incomplete "
               f"verification record ({verification_record_path(run_root)}).\n"
               f"  Missing: {'; '.join(missing)}")
    else:
        msg = (f"AUGMENTATION NOT VERIFIED: scenario '{s}' has no verification on "
               f"record ({verification_record_path(run_root)}).\n"
               f"  §4.1 stage 4 requires BOTH before this scenario is trained on or cached:\n"
               f"    (a) contact sheets inspected by eye, and\n"
               f"    (b) quantitative checks passed (defect survival / label integrity /\n"
               f"        variety / normal drift) — see analysis/aug_verify.py\n"
               + (f"  This scenario is RISKY: it is held to STRICT thresholds.\n" if strict else ""))
    if require:
        raise UnverifiedScenarioError(msg)
    print("  ! " + msg)
    return {"scenario": s, "verified": False}
