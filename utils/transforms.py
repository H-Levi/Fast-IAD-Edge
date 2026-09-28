#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transform builders.

Pipeline order (this order is deliberate):
    Resize -> [FilterBank] -> [augmentation, train only] -> ToTensor -> Normalize

- The FilterBank runs on BOTH train and test, because filters define the input
  representation the model sees, not a training-time perturbation.
- Augmentation (flip/rotate/jitter) runs on TRAIN ONLY.
- image_size, filters, and augmentation all come from config, so the transform
  is fully determined by configs/default.yaml.
"""

from torchvision import transforms

from utils.filters import FilterBank

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _filter_lambda(bank: FilterBank):
    # transforms.Lambda needs a top-level-ish callable; bank is already callable.
    return transforms.Lambda(lambda img: bank(img))


def build_transforms(cfg: dict, train: bool):
    """Return a torchvision transform built from the config.

    cfg is the full config dict (we read cfg['data'], cfg['filters'], cfg['augment']).
    """
    image_size = cfg["data"]["image_size"]
    bank = FilterBank(cfg.get("filters", {"enabled": False}))

    steps = [transforms.Resize((image_size, image_size))]

    # filters: applied to both splits
    if bank.enabled:
        steps.append(_filter_lambda(bank))

    # augmentation: train only
    if train and cfg.get("augment", {}).get("enabled", False):
        aug = cfg["augment"]
        if aug.get("horizontal_flip", 0):
            steps.append(transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip"])))
        if aug.get("rotation_degrees", 0):
            steps.append(transforms.RandomRotation(degrees=float(aug["rotation_degrees"])))
        cj = aug.get("color_jitter")
        if cj:
            steps.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
            ))

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps), bank.describe()


def build_scenario_transforms(cfg: dict, category: str):
    """Train-split transforms for an augmentation scenario (§4.2) or an explicit
    p/range pipeline config (§3.7).

    Returns (normal_transform, anomaly_transform, synthetic_op, description).

    Two ways to specify augmentation, both train-only:
      * named scenario  (augmentation.use_pipeline_config = false, default) —
        one of the eight frozen §4.2 save-set states.
      * pipeline config (use_pipeline_config = true) — the §3.7 two-layer shape,
        where every transform carries its own {p, range}. This is the form the
        §1.4 comparator can vary one knob at a time.

    When augmentation is disabled this returns exactly what
    build_transforms(cfg, train=True) returns, with anomaly_transform=None and
    synthetic_op=None — the baseline pipeline, unchanged.
    """
    from data.augmentation import scenario_steps, CutPaste, compose_two_layer, build_synthetic
    from data.aug_pipeline import build_pipeline, apply_category_overrides

    aug_cfg = cfg.get("augmentation", {}) or {}
    enabled = aug_cfg.get("enabled", False)
    scenario = aug_cfg.get("scenario", "original") if enabled else "original"
    use_pipeline = enabled and aug_cfg.get("use_pipeline_config", False)

    base, desc = build_transforms(cfg, train=True)
    if not enabled or (scenario == "original" and not use_pipeline):
        return base, None, None, {"scenario": "original", "filters": desc}

    image_size = cfg["data"]["image_size"]
    bank = FilterBank(cfg.get("filters", {"enabled": False}))

    def _assemble(extra_steps):
        steps = [transforms.Resize((image_size, image_size))]
        if bank.enabled:
            steps.append(_filter_lambda(bank))
        steps += extra_steps
        steps += [transforms.ToTensor(),
                  transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        return transforms.Compose(steps)

    # ---- §3.7 explicit p/range pipelines (two-layer) ----
    if use_pipeline:
        merged = compose_two_layer(aug_cfg.get("base", {}), aug_cfg.get("targeted", {}))
        overrides = aug_cfg.get("per_category_overrides", {})
        allow_destr = aug_cfg.get("allow_destructive", False)

        n_cfg = apply_category_overrides(merged["normal_pipeline"], overrides, category)
        a_cfg = apply_category_overrides(merged["anomaly_pipeline"], overrides, category)

        n_ops, n_desc = build_pipeline(n_cfg, "normal", allow_destructive=allow_destr)
        a_ops, a_desc = build_pipeline(a_cfg, "anomaly", allow_destructive=allow_destr)

        synth = None
        syn_cfg = aug_cfg.get("synthetic", {}) or {}
        if float(syn_cfg.get("p", 0.0)) > 0:
            synth = build_synthetic(syn_cfg.get("generator", "cutpaste"),
                                    seed=cfg.get("seed", 123))
        # The verification gate keys off this name. A pipeline-config run IS
        # augmenting even when no named scenario was chosen, so it must not
        # inherit 'original' (which the gate exempts as "nothing to inspect").
        gate_name = scenario if scenario != "original" else "pipeline_config"
        return (_assemble(n_ops), _assemble(a_ops), synth,
                {"scenario": gate_name, "mode": "pipeline_config", "filters": desc,
                 "category": category, "normal_pipeline": n_desc,
                 "anomaly_pipeline": a_desc, "active_addons": merged["active_addons"],
                 "synthetic": (syn_cfg.get("generator") if synth else None)})

    # ---- §4.2 named scenario ----
    legacy = _legacy_augment_steps(cfg)
    normal_tf = _assemble(legacy + scenario_steps(scenario, category, "normal"))
    anomaly_tf = _assemble(legacy + scenario_steps(scenario, category, "anomaly"))
    synth = CutPaste(seed=cfg.get("seed", 123)) if scenario == "cutpaste" else None
    return normal_tf, anomaly_tf, synth, {"scenario": scenario, "mode": "named_scenario",
                                          "filters": desc, "category": category}


def _legacy_augment_steps(cfg: dict):
    """The pre-existing cfg['augment'] knobs, kept so scenarios compose with them."""
    steps = []
    aug = cfg.get("augment", {})
    if not aug.get("enabled", False):
        return steps
    if aug.get("horizontal_flip", 0):
        steps.append(transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip"])))
    if aug.get("rotation_degrees", 0):
        steps.append(transforms.RandomRotation(degrees=float(aug["rotation_degrees"])))
    cj = aug.get("color_jitter")
    if cj:
        steps.append(transforms.ColorJitter(
            brightness=cj.get("brightness", 0),
            contrast=cj.get("contrast", 0),
            saturation=cj.get("saturation", 0)))
    return steps


def denormalize(tensor):
    """Invert ImageNet normalization for saving/visualizing images correctly.

    (The earlier pipeline saved normalized tensors straight to uint8, producing
    garbage images. Use this before any image save.)
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor.cpu() * std + mean
