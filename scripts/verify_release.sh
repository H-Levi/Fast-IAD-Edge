#!/usr/bin/env bash
# Verify this release against the thesis on an NVIDIA T4 (e.g. Kaggle), about 30-40 min in total:
#   1. automated checks   2. SIX exact reruns at seed 7 vs results/reference_runs (identical scores, metrics)
#   3. model cost vs the thesis (parameters exact, GFLOPs)   4. same-session speed of ours / EfficientAD-S / KairosAD (within 15 %)
# Usage: bash scripts/verify_release.sh /path/to/mvtec_ad /path/to/visa [workdir]   (needs internet for the two comparator clones)
set -u
MV=$1; VI=$2; W=${3:-verify_out}; mkdir -p $W; REP=$W/VERIFY_REPORT.txt; : > $REP
say(){ echo "$@" | tee -a $REP; }
say "== environment"; python -c "import torch, torchvision, platform; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| python', platform.python_version())" | tee -a $REP
say "commit $(git rev-parse --short HEAD 2>/dev/null)"
say "== 1. automated checks"; python tests/tier0.py 2>&1 | tail -1 | tee -a $REP
say "== 2. exact reruns (seed 7)"
F="--config configs/final.yaml run.seeds=[7] paths.mvtec_root=$MV paths.visa_root=$VI"
python run_train.py $F run.datasets=[mvtec] run.categories=[carpet,screw] paths.output_root=$W/A > $W/train_A.log 2>&1
python run_train.py $F run.datasets=[visa] run.categories=[capsules,pcb4] paths.output_root=$W/B > $W/train_B.log 2>&1
python run_train.py $F run.datasets=[mvtec] run.categories=[carpet,screw] data.transplant.enabled=false paths.output_root=$W/C > $W/train_C.log 2>&1
for x in "A mvtec carpet" "A mvtec screw" "B visa capsules" "B visa pcb4" "C mvtec carpet" "C mvtec screw"; do
  set -- $x; d=$(ls -d $W/$1/*/$2/efficientnet/$3 2>/dev/null | head -1)
  say "-- $1 $2 $3"; if [ -z "$d" ]; then say "   run folder missing (see $W/train_$1.log): FAIL"; continue; fi
  python scripts/verify_rerun.py --new $d --ref results/reference_runs/${1}_${2}_${3}_s7 2>&1 | tee -a $REP
done
say "== 3. model cost (CPU)"
pip install -q fvcore timm >/dev/null 2>&1
test -d $W/EfficientAD || git clone -q https://github.com/nelson1425/EfficientAD $W/EfficientAD
test -d $W/KairosAD || git clone -q --recurse-submodules https://github.com/intelligolabs/KairosAD $W/KairosAD
pip install -q -e $W/KairosAD/MobileSAM >/dev/null 2>&1
python scripts/verify_cost.py --efficientad_root $W/EfficientAD --kairos_root $W/KairosAD 2>&1 | grep -v -i warn | tee -a $REP
say "== 4. same-session speed (T4, batch 1, full optimisation)"
A="--kairos_root $W/KairosAD --efficientad_root $W/EfficientAD --ours efficientnet:deep:avg --precisions fp32,fp16 --compile --full --twoview --iters 100"
python analysis/head_to_head.py $A --sizes 288 --out $W/h2h_ours_kairos --only ours:efficientnet/avg+full,ours:efficientnet/avg+full[x2],kairosad+full > $W/h2h.log 2>&1
python analysis/head_to_head.py $A --sizes 256 --out $W/h2h_efficientad --only efficientad_s+full >> $W/h2h.log 2>&1
python scripts/verify_timing.py --new_ours_kairos $W/h2h_ours_kairos.csv --new_efficientad $W/h2h_efficientad.csv 2>&1 | tee -a $REP
say "== SUMMARY"; grep -c "VERDICT: PASS" $REP | xargs -I{} echo "{} of 8 verdicts PASS (6 reruns + cost + speed)" | tee -a $REP
