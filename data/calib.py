#!/usr/bin/env python
"""Size of the held-out calibration split (analysis P4/P7). One rule for MVTec and VisA.

- `calib_split` (fraction, legacy, used by F-L98/F-L99): returned unchanged, no floor -> those runs reproduce.
- `calib_n` (count, F-L99's fix): a fixed number of calibration normals, so large categories no longer lose 42% of
  their training normals to reach a size only the small ones need. FLOOR RULE (F-L97a): applied only if at least
  `min_train` training normals remain; otherwise the category gets NO calibration split, keeps its validation threshold,
  and says so in the returned info.
- Setting both is an error (no silent precedence).
Returns (test_size for train_test_split, or 0 for "no split"; info dict for the run record)."""


def calib_size(pool_n, calib_split=0.0, calib_n=0, min_train=100):
    calib_split, calib_n = float(calib_split or 0), int(calib_n or 0)
    if calib_split > 0 and calib_n > 0:
        raise ValueError("data.calib_split and data.calib_n are mutually exclusive")
    if calib_split > 0:
        return calib_split, {"applied": True, "mode": "fraction", "fraction": calib_split, "pool": pool_n}
    if calib_n > 0:
        if pool_n - calib_n < int(min_train):
            return 0, {"applied": False, "mode": "count", "requested": calib_n, "pool": pool_n, "min_train": int(min_train),
                       "reason": f"only {pool_n - calib_n} training normals would remain (< {min_train}): keeps the validation threshold"}
        return calib_n, {"applied": True, "mode": "count", "n": calib_n, "pool": pool_n, "train_normals_left": pool_n - calib_n}
    return 0, {"applied": False, "mode": "off"}
