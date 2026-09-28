#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_mobilesam.py — apply the edit KairosAD's own README requires.

This is not our modification of their method. Their README, under "Manual File
Patch", instructs:

    On file `MobileSAM/mobile_sam/utils/transforms.py`, update the function
    apply_image() with the following code:

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1],
                                                self.target_length)
        image = image.permute(2, 0, 1)
        return resize(image, target_size)

Stock MobileSAM's apply_image goes through PIL and numpy; KairosAD feeds it
torch tensors instead, so `.permute` is required. Without the patch their model
raises `'numpy.ndarray' object has no attribute 'to'` -- an error that names
numpy and points nowhere near the missing setup step.

Doing it with a script rather than by hand because it must be reproducible: a
hand-edit on a Kaggle container is gone the next session, and an
almost-right hand-edit is worse than none.

    python analysis/patch_mobilesam.py --kairos_root KairosAD
"""

import argparse
import os
import re
import sys

MARKER = "image.permute(2, 0, 1)"

NEW_BODY = '''        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        image = image.permute(2, 0, 1)

        return resize(image, target_size)
'''


def patch(kairos_root, verbose=True):
    """Returns (status, path). status in {'patched','already','missing','failed'}."""
    path = os.path.join(kairos_root, "MobileSAM", "mobile_sam", "utils", "transforms.py")
    if not os.path.exists(path):
        return "missing", path

    src = open(path, encoding="utf-8").read()
    if MARKER in src:
        return "already", path

    # Replace from the apply_image def up to (not including) the next def.
    pat = re.compile(
        r"(    def apply_image\(self, image[^\n]*\n"          # signature
        r"(?:        [\"'].*?[\"'].*?\n|        \n)*?)"        # docstring / blanks
        r"(?:.*?\n)*?"                                         # old body
        r"(?=\n    def |\n\n    def )",                        # next method
        re.S)
    m = pat.search(src)
    if not m:
        return "failed", path

    new_src = src[:m.end(1)] + NEW_BODY + src[m.end():]
    if MARKER not in new_src:
        return "failed", path

    open(path + ".orig", "w", encoding="utf-8").write(src)   # keep the original
    open(path, "w", encoding="utf-8").write(new_src)
    return "patched", path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kairos_root", default="KairosAD")
    a = ap.parse_args()
    status, path = patch(a.kairos_root)
    msg = {
        "patched": f"patched {path} (original saved as {path}.orig)",
        "already": f"already patched: {path}",
        "missing": f"NOT FOUND: {path} -- was the clone run with --recurse-submodules?",
        "failed": f"could not locate apply_image in {path} -- patch by hand per "
                  f"KairosAD's README",
    }[status]
    print(msg)
    sys.exit(0 if status in ("patched", "already") else 1)


if __name__ == "__main__":
    main()
