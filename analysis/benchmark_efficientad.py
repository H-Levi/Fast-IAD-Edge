#!/usr/bin/env python
"""benchmark_efficientad.py — EfficientAD-S (Batzner et al., WACV 2024) built from the public reimplementation
github.com/nelson1425/EfficientAD (common.py: get_pdn_small, get_autoencoder; plain PyTorch, so Kaggle's torch is untouched).
F-L142: timing ONLY, random weights (latency and parameters do not depend on weight values), batch 1, 256 px (its native
input; its autoencoder upsamples to fixed sizes, so other resolutions do not fit). Unsupervised method: a COST reference
only — its accuracy is never compared with ours (different training setting).
Inference path timed = the method's image scoring: teacher, student, autoencoder; teacher output normalised; the
student-teacher map and the autoencoder-student map averaged; image score = the map's maximum."""
import os
import sys
import torch


class EfficientADS(torch.nn.Module):
    def __init__(self, root):
        super().__init__()
        sys.path.insert(0, os.path.abspath(root))
        try:
            from common import get_pdn_small, get_autoencoder   # nelson1425/EfficientAD/common.py
        finally:
            sys.path.pop(0)
        c = 384
        self.teacher, self.student, self.ae = get_pdn_small(c), get_pdn_small(2 * c), get_autoencoder(c)
        self.register_buffer("t_mean", torch.zeros(1, c, 1, 1)); self.register_buffer("t_std", torch.ones(1, c, 1, 1))
        self.c = c

    def forward(self, x):
        t = (self.teacher(x) - self.t_mean) / self.t_std
        s = self.student(x); a = self.ae(x)
        m_st = torch.mean((t - s[:, :self.c]) ** 2, dim=1, keepdim=True)
        m_ae = torch.mean((a - s[:, self.c:]) ** 2, dim=1, keepdim=True)
        m = 0.5 * m_st + 0.5 * m_ae
        return m.amax(dim=(1, 2, 3))

    def summary(self):
        n = sum(p.numel() for p in self.parameters())
        return {"total_params": n, "params": n, "size_mb": n * 4 / 2 ** 20}
