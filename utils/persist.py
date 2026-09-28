#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""persist.py — bundle a run's results the moment it finishes, automatically.

WHY. Kaggle wipes /kaggle/working when the GPU is toggled or the session ends.
That already cost us the first Arm F v2 subset: seven categories, carpet among
them, whose per-image scores are gone and whose like-for-like AUROC can never be
recomputed. The run itself was fine; the artefacts evaporated.

WHAT IT SAVES, and what it does not. A run tree is ~574 MB for 27 categories, of
which ~503 MB is model checkpoints. Excluding those leaves ~72 MB: every
summary.json, stage.jsonl, epochs.jsonl, logit_hist.jsonl, config.json, report
text, and every scores_*.npz. Those are what every downstream analysis reads --
bootstrap CIs, calibration, like-for-like, Mahalanobis, feature shift. The
checkpoints are the only thing that cannot be rebuilt cheaply, and they are also
the only thing nothing downstream needs.

WHERE. A fixed, predictable location, not a timestamped one you have to hunt for:

    <output_root>/_BUNDLES/<run_id>.tar.gz     next to the run
    /kaggle/working/_ALL_RESULTS/<run_id>.tar.gz   one folder holding every run
    /kaggle/working/_ALL_RESULTS/INDEX.json        what each bundle contains

Download _ALL_RESULTS once per session and nothing is lost. The index means a
later session can tell what exists without unpacking anything.
"""
import json, os, tarfile, time, zipfile

EXCLUDE_SUFFIX = (".pt", ".pth", ".ckpt")
WORK = "/kaggle/working"
MIRROR = "/kaggle/working/_ALL_RESULTS"
# ONE file to download. Kaggle's UI downloads files, not folders, so a directory
# of per-run tarballs means one click per run and a chance to miss the last one.
# This single zip is appended to after every run and is the only thing that has
# to leave the machine.
ROLLING_ZIP = "/kaggle/working/ALL_RESULTS.zip"


def sync_zip(extra_files=()):
    """Make ALL_RESULTS.zip hold EVERYTHING that should leave the machine.

    The first version appended only the run that had just finished. A run bundled
    before the rolling zip existed (noaug_mvtec, 2026-09-21) sat in _ALL_RESULTS/
    and never entered the zip; I had said it would be "picked up". It was not.
    Now every call sweeps: every .tar.gz in the mirror folder, the index, and any
    checkpoint JSON in the working root -- each file independently, so one
    unreadable file cannot stop the rest from being saved.
    """
    if not os.path.isdir(WORK):
        return None
    os.makedirs(MIRROR, exist_ok=True)
    added, failed = [], []
    cands = [os.path.join(MIRROR, f) for f in sorted(os.listdir(MIRROR))
             if f.endswith(".tar.gz")]
    cands += [os.path.join(WORK, f) for f in sorted(os.listdir(WORK))
              if f.startswith("checkpoint") and f.endswith(".json")]
    cands += [f for f in extra_files if f]
    with zipfile.ZipFile(ROLLING_ZIP, "a", zipfile.ZIP_STORED) as z:
        have = set(z.namelist())
        for fp in cands:
            arc = os.path.basename(fp)
            if arc in have:
                continue
            try:
                z.write(fp, arcname=arc); have.add(arc); added.append(arc)
            except Exception as e:
                failed.append(f"{arc}: {type(e).__name__}")
        idx = os.path.join(MIRROR, "INDEX.json")
        if os.path.exists(idx):
            tag = f"INDEX__latest_{len(have)}.json"
            if tag not in have:
                try:
                    z.write(idx, arcname=tag)
                except Exception as e:
                    failed.append(f"INDEX: {type(e).__name__}")
    return {"added": added, "failed": failed, "zip": ROLLING_ZIP,
            "mb": round(os.path.getsize(ROLLING_ZIP) / 1024 ** 2, 2)}


def _keep(path):
    return not path.endswith(EXCLUDE_SUFFIX)


def bundle_run(run_dir, extra_note=None):
    """Tar everything analysis needs. Returns a dict describing what was written."""
    run_dir = os.path.abspath(run_dir)
    run_id = os.path.basename(run_dir.rstrip("/"))
    info = {"run_id": run_id, "run_dir": run_dir,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": extra_note, "targets": [], "n_files": 0, "bytes": 0,
            "excluded_checkpoints": 0, "excluded_bytes": 0}

    members = []
    for dirpath, dirnames, files in os.walk(run_dir):
        dirnames[:] = [d for d in dirnames if d != "_BUNDLES"]
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if _keep(fp):
                members.append((fp, os.path.relpath(fp, os.path.dirname(run_dir))))
                info["n_files"] += 1
                info["bytes"] += sz
            else:
                info["excluded_checkpoints"] += 1
                info["excluded_bytes"] += sz

    if not members:
        info["error"] = "nothing to bundle"
        return info

    # Name by the output_root the caller chose (noaug_mvtec, cache_test, ...) plus
    # the run id. The run id alone is timestamped but says nothing about intent,
    # and three runs from one afternoon are otherwise indistinguishable.
    tag = os.path.basename(os.path.dirname(run_dir.rstrip("/"))) or "run"
    name = f"{tag}__{run_id}"
    info["bundle_name"] = name
    targets = [os.path.join(run_dir, "_BUNDLES", name + ".tar.gz")]
    if os.path.isdir(WORK):
        targets.append(os.path.join(MIRROR, name + ".tar.gz"))

    for t in targets:
        try:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            with tarfile.open(t, "w:gz") as tf:
                for fp, arc in members:
                    tf.add(fp, arcname=arc)
            info["targets"].append({"path": t, "size_mb": round(os.path.getsize(t) / 1024 ** 2, 2)})
        except Exception as e:
            info["targets"].append({"path": t, "error": f"{type(e).__name__}: {e}"})

    # A single index so a later session knows what exists without unpacking.
    if os.path.isdir(WORK):
        idx_path = os.path.join(MIRROR, "INDEX.json")
        try:
            idx = json.load(open(idx_path)) if os.path.exists(idx_path) else []
        except Exception:
            idx = []
        idx = [e for e in idx if e.get("run_id") != run_id] + [info]
        try:
            os.makedirs(MIRROR, exist_ok=True)
            json.dump(idx, open(idx_path, "w"), indent=2, default=str)
            info["index"] = idx_path
        except Exception as e:
            info["index_error"] = str(e)

        # Sweep everything not yet in the rolling zip (this run AND any earlier
        # bundle that predates the zip), so it is complete after every run.
        r = sync_zip()
        if r:
            info["rolling_zip"] = r["zip"]; info["rolling_zip_mb"] = r["mb"]
            info["rolling_zip_added"] = r["added"]
            info["runs_in_zip"] = len(idx)
        elif r:
            info["rolling_zip_error"] = r["error"]
    return info


def save_weights(run_dir):
    """Full trained weights, in their OWN zip beside ALL_RESULTS.zip (O-19).

    The results bundle excludes checkpoints on purpose (88% of the bytes). Deployment, INT8
    and the Jetson test need the full fine-tuned model, not just the head, so a claim run can
    ask for them with `run.save_weights=true` (or a list of seeds). Each zip carries the run's
    summary.json files and what_runs.txt (config + git commit), so a weight file is never
    separated from what produced it.
    """
    run_dir = os.path.abspath(run_dir)
    run_id = os.path.basename(run_dir.rstrip("/"))
    tag = os.path.basename(os.path.dirname(run_dir.rstrip("/"))) or "run"
    out = os.path.join(WORK if os.path.isdir(WORK) else run_dir, f"WEIGHTS_{tag}__{run_id}.zip")
    n, size = 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for dirpath, dirnames, files in os.walk(run_dir):
            dirnames[:] = [d for d in dirnames if d != "_BUNDLES"]
            for f in files:
                fp = os.path.join(dirpath, f)
                if f.endswith(EXCLUDE_SUFFIX) or f in ("summary.json", "what_runs.txt", "config.json"):
                    z.write(fp, os.path.relpath(fp, os.path.dirname(run_dir)))
                    if f.endswith(EXCLUDE_SUFFIX):
                        n += 1; size += os.path.getsize(fp)
    return {"zip": out, "checkpoints": n, "mb": round(size / 1024 ** 2, 1)}


def describe(info):
    if info.get("error"):
        return f"  BUNDLE: {info['error']}"
    saved = " | ".join(f"{t['path']} ({t.get('size_mb','?')} MB)"
                       for t in info["targets"] if "error" not in t)
    lines = [f"  BUNDLE: {info['n_files']} files, "
             f"{round(info['bytes']/1024**2,1)} MB  ->  {saved}",
             f"          checkpoints excluded: {info['excluded_checkpoints']} "
             f"({round(info['excluded_bytes']/1024**2,1)} MB) -- nothing downstream reads them"]
    if info.get("rolling_zip"):
        lines.append(f"          >>> DOWNLOAD THIS ONE FILE: {info['rolling_zip']}  "
                     f"({info.get('rolling_zip_mb','?')} MB, {info.get('runs_in_zip','?')} runs)")
        lines.append(f"          It is rebuilt after EVERY run, so it is complete right now. "
                     f"Grab it before toggling the GPU.")
    elif os.path.isdir(WORK):
        lines.append(f"          DOWNLOAD {MIRROR} ONCE before toggling the GPU.")
    return "\n".join(lines)
