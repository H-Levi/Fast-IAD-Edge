# Efficient supervised anomaly detection for industrial edge deployment

Code and results of the M.Sc. thesis *Efficient Supervised Anomaly Detection for Industrial Edge Deployment* (Hailegabriel Dereje
Degefa, M.Sc. in Artificial Intelligence, Department of Computer Science, Università degli Studi di Verona, September 2026;
supervisor: Professor Francesco Setti).

A lightweight supervised detector (ImageNet-pretrained EfficientNet-B0 + a small head, one model per category, 288 px) held to a
pre-set false-alarm budget on MVTec AD and VisA. On MVTec AD, training defects are pasted onto context-matched training normals.
Pasting defects is a known operation (the "extended anomalies" of PRN, Zhang et al., CVPR 2023); the thesis uses it to counter a
photo-session shortcut of the standard supervised MVTec protocol, which it shrinks but does not remove, and it costs one category
(screw).

## Results (final models, 3 seeds per category)

| dataset | false alarms at the 5 % line | recall | AUROC |
|---|---|---|---|
| MVTec AD | 3.9 % (mean of 14 category means) | 0.887 (14 categories) | 0.950 (15 categories, like-for-like) |
| VisA | 4.9 % (mean of 12 category means) | 0.919 | 0.983 |

- Seeds: VisA and 9 MVTec categories use 7, 123, 2024; bottle, metal_nut, tile, toothbrush, wood and zipper use 11, 123, 2024.
- Toothbrush has only 12 validation normals, too few for a 5 % line (19 are needed), so its false alarms and recall are undefined;
  MVTec false alarms and recall are over the other 14 categories.
- Not every category meets every pre-set line: on MVTec, capsule and transistor are just short on AUROC (0.891, 0.892 against
  0.90), toothbrush is short on AUROC (0.850), and screw misses recall and AUROC (0.491, 0.805); the highest category mean of false
  alarms is carpet's 9.5 %. On VisA every category meets every line (highest category mean: pcb4, 6.7 %).
- The 5 % line is a split-conformal threshold on normal images not used for training: on MVTec the validation normals, which also
  choose the epoch; on VisA 110 held-out training normals, used only for the threshold. An image is flagged if its score (mean of
  two views: identity and a 10° rotation) exceeds the line.
- MVTec AUROC is like-for-like: defects are ranked only against normals from the MVTec test folder.
- Pooled over images: MVTec 55 of 1,365 test normals alarm (4.0 %), recall 2,420 of 2,766; VisA 560 of 11,544 (4.9 %).
- The thesis compares with DevNet, DRA, PRN, BGAD and KairosAD using their reported values under their own protocols; those
  numbers are not reproduced here and are not directly comparable.

**Recompute these numbers without a GPU** from the per-image scores (no images are included):

    pip install numpy scipy
    python scripts/score_final.py

`results/per_image_scores_final.csv` (27,759 rows; splits val, calib and test; no training rows): dataset, category, seed, split,
file (path inside the dataset), label (1 = defective), score_two_view (mean of the identity and rotated-view logits),
score_single (identity-view logit), pos_weight (the class weight used in training: N/(N+P) on MVTec, N/P on VisA).

Model size: 4.80 M parameters; 18.3 MB in fp32 and 9.2 MB in fp16 (parameter bytes, 1 MB = 2^20 bytes); 1.32 GFLOPs per view,
2.65 for the two views (GFLOPs = 2 x multiply-adds counted by fvcore). Model weights are not distributed. Training uses fixed
seeds and deterministic cuDNN; rerunning one final run (screw, seed 7) on the same Kaggle setup gave identical scores for all 130
test images. Other hardware or library versions may differ slightly.

## Reproduce the training

    pip install -r requirements.txt
    python tests/tier0.py                         # automated checks, CPU only
    bash scripts/reproduce.sh /path/to/mvtec_ad /path/to/visa

Training needs a CUDA GPU (about 1–3 minutes per run on a T4; 81 runs); on a CPU it falls back automatically but is much slower
and was not used. `configs/final.yaml` holds the settings of the final runs (it inherits `baseline.yaml` and `default.yaml`). The
final MVTec runs were made at five code commits and the VisA runs at two, which train identical models for these settings; the
training code in this release equals the first of them (ff6ff30) apart from one added option the final settings do not use. The
runs used an NVIDIA T4 on Kaggle, torch 2.10.0 (CUDA 12.8), Python 3.12.

## Speed and cost

- `analysis/cost_table.py`: parameters, file size and GFLOPs (needs fvcore).
- `analysis/head_to_head.py`: same-session timing of our network against EfficientAD-S and KairosAD (`--full` = torch.compile with
  CUDA graphs + channels-last; `--twoview` = both views as one batch). EfficientAD-S needs a clone of
  github.com/nelson1425/EfficientAD, KairosAD a clone of github.com/intelligolabs/KairosAD with MobileSAM installed.
- `analysis/time_cpu_pipeline.py`: full-pipeline timing (decode, resize, model) on a CPU.

## Verify this release against the thesis

    bash scripts/verify_release.sh /path/to/mvtec_ad /path/to/visa      # NVIDIA T4, about 30-40 min, internet for two clones

1. the automated checks; 2. six exact reruns at seed 7 — final MVTec model (carpet, screw), final VisA model (capsules, pcb4) and
the standard-split baseline (carpet, screw) — each compared with the thesis's saved scores in `results/reference_runs/`: same
images, same epochs, identical per-image scores (tolerance 1e-4 with identical alarm decisions if the torch version differs from
2.10.0) and identical metrics (false alarms, recall, precision, ECE, AUROC, AUPR); 3. parameters and GFLOPs of ours, EfficientAD-S
and KairosAD against the thesis (exact); 4. a same-session speed test: each model's best time within 15 % of the thesis's session
and the same ranking. Every check was tested with a planted change that it must catch. The report is `verify_out/VERIFY_REPORT.txt`.

Further checks: `scripts/verify_extra.sh` (the paste-versus-repeat control, carpet seed 7, and a three-repeat speed diagnostic) and
`scripts/verify_cpu.sh` (CPU speed of the three models: same ranking, each ratio within 20 % of the thesis's CPU session).

**Our own verification (Kaggle T4, torch 2.10.0, commit 047e56c / 093318b):** all seven reruns — the six above plus the
paste-versus-repeat control — reproduced every stored score exactly (difference 0) and every metric; parameters and GFLOPs of all
three models matched exactly; the two-view time (2.91–3.20 ms against 2.99) and the speed ratios against EfficientAD-S and KairosAD
reproduced. Our ONE-view time did not: 2.38–2.57 ms in two new sessions against 1.66 ms in the thesis's session, so the one-view
speed check FAILED its 15 % rule; the one-view time varies between sessions and is reported as a range (1.66–2.57 ms). On a CPU
(two Kaggle sessions, fp32, batch 1) ours was 11–15 times faster than EfficientAD-S and 37–43 times faster than KairosAD; the second
session missed the 20 % rule for the EfficientAD-S ratio (11.1 against 14.9), so the CPU advantage is given as a range.

## Safeguards

The results were protected by checks that each guard against one named failure. The leak check, split views, collapse detector,
protocol fingerprint and "what ran" log act inside every training run; the automated checks run first in `reproduce.sh`; the smoke
test is run by hand before every training campaign.

| safeguard | guards against | where |
|---|---|---|
| Smoke test | a broken data layout, empty or single-class splits, a pipeline stage that silently writes nothing | `python smoke_test.py paths.mvtec_root=... paths.visa_root=... run.datasets=[mvtec,visa]` |
| Automated test suite (35 test groups, 194 checks, each group with at least one negative control; 194/194 pass) | code changes that break splits, scoring, thresholds or the transplant | `tests/tier0.py`, runs first in `reproduce.sh` |
| Leak self-check at run time (file level) | a file path shared by training, validation, calibration and test. It compares paths only and does not inspect pasted images; its threshold sub-check cannot fail | `analysis/guardrails.py` (`leak_self_check`) |
| Transplant split audit (done for the thesis) | a pasted defect or host image taken from validation or test: all 45 final MVTec runs clean, the rebuilt lists equal the scored images, negative control 0/45 | `analysis/evidence_export.py` (`leaklists`) |
| One fixed split view per data loader | building one loader silently changing another loader's split | `data/view.py` |
| Test never chooses inside a run | the epoch and the threshold come from validation / calibration data only; test is scored once | `run_train.py` |
| Collapse detector | a model that outputs one class for everything being reported as a result | `analysis/guardrails.py` (`detect_collapse`) |
| Protocol fingerprint | comparing two runs whose settings differ in more than the thing under test | recorded in every run; `analysis/protocol.py` |
| "What ran" log | a setting that looks off in the configuration but is active in the code | `analysis/what_runs.py`, written into every run folder |

Three recipe-level choices were made with MVTec test results in view (the pair of test-time views, the toothbrush sampling
override, and the adoption of the final model without the swap), and the regime rule was chosen on already-scored runs; the thesis
states these openly. It also lists further checks kept in the project record: pre-registration, dry and planted controls for every
scorer, post-run checks and a literature quote checker.

## Data

The datasets are **not** included (their licences do not allow redistribution). The runs used the Kaggle copies
`ipythonx/mvtec-ad` and `ess1004/visa-anomaly-detection`; the originals are:
- MVTec AD — https://www.mvtec.com/company/research/datasets/mvtec-ad (CC BY-NC-SA 4.0)
- VisA — https://github.com/amazon-science/spot-diff (see its licence)

Expected layout: MVTec AD in the standard form `<root>/<category>/{train,test,ground_truth}/...` — the `ground_truth` masks are
required (the transplant cuts defects with them). VisA as released: `<root>/<category>/Data/Images/{Normal,Anomaly}/...` plus the
official split file `<root>/split_csv/2cls_highshot.csv`. The MVTec copy matches the per-category image counts of the MVTec AD
paper; the VisA copy matches its published total (10,821 images: 9,621 normal, 1,200 anomalous).

## Repository layout

    run_train.py            training, validation-based epoch choice, scoring
    configs/                default.yaml -> baseline.yaml -> final.yaml
    data/                   dataset splits, defect transplant (data/transplant.py), loaders
    models/, engine/        backbone + head, training loop, evaluation, deployment score
    analysis/               metrics, safeguards (leak and collapse checks), speed/cost benchmarks
    tests/tier0.py          automated checks, each group with a negative control that must fail
    scripts/                reproduce.sh, score_final.py
    results/                per-image scores of the final models

Comments in the code refer to entries of the project's private research record (e.g. F-L…, A-…); they document why a choice was
made and are not needed to run anything.

## Citation and licences

See `CITATION.cff`. Code: MIT (`LICENSE`). `results/` are derived from MVTec AD and VisA and carry their file paths; they are
shared under CC BY-NC-SA 4.0, like MVTec AD. The datasets keep their own licences. ImageNet-pretrained torchvision weights are
downloaded at run time and not redistributed.
