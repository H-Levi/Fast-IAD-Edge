#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Determinism control.

One place that owns *all* RNG seeding so a run is reproducible and a
multi-seed sweep is honest. Earlier code seeded once at process start and
left cudnn nondeterministic in places; this centralizes it.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 123, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (CPU+CUDA). Optionally force determinism.

    deterministic=True trades a little speed for bit-reproducibility, which is
    what you want while we are dissecting the model. Flip it off only for the
    final timing benchmark where speed is the measured quantity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init so multi-worker loading is reproducible too."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
