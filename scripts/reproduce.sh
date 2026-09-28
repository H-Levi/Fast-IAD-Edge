#!/usr/bin/env bash
# Retrain the final models. One run = one category at one seed (about 1-3 min on a T4).
# Usage: bash scripts/reproduce.sh /path/to/mvtec_ad /path/to/visa [output_dir]
set -e
MV=$1; VI=$2; OUT=${3:-runs/final}
python tests/tier0.py                                   # 194 checks; stops here if anything is broken
python run_train.py --config configs/final.yaml run.datasets=[mvtec] run.categories=[cable,capsule,carpet,grid,hazelnut,leather,pill,screw,transistor] run.seeds=[7,123,2024] paths.mvtec_root=$MV paths.output_root=$OUT/mvtec_a
python run_train.py --config configs/final.yaml run.datasets=[mvtec] run.categories=[bottle,metal_nut,tile,toothbrush,wood,zipper] run.seeds=[11,123,2024] paths.mvtec_root=$MV paths.output_root=$OUT/mvtec_b
python run_train.py --config configs/final.yaml run.datasets=[visa] run.categories=[all] run.seeds=[7,123,2024] paths.visa_root=$VI paths.output_root=$OUT/visa
