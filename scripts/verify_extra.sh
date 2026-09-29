#!/usr/bin/env bash
# Two further checks (about 15 min on a T4): (1) the paste-versus-repeat control — carpet, seed 7, real defects REPEATED instead of
# pasted — must reproduce its saved run exactly; (2) our network's speed timed three times in a row with CUDA-graph diagnostics and
# GPU clocks, to see whether the first measurement of a session runs slower ("cold GPU").
# Usage: bash scripts/verify_extra.sh /path/to/mvtec_ad [workdir]
set -u
MV=$1; W=${2:-verify_extra}; mkdir -p $W; REP=$W/EXTRA_REPORT.txt; : > $REP
say(){ echo "$@" | tee -a $REP; }
say "== environment"; python -c "import torch; print('torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))" | tee -a $REP
say "== 1. paste-versus-repeat control (duplicate placement), carpet seed 7"
python run_train.py --config configs/final.yaml run.datasets=[mvtec] run.categories=[carpet] run.seeds=[7] data.transplant.placement=duplicate paths.mvtec_root=$MV paths.output_root=$W/D > $W/train_D.log 2>&1
python scripts/verify_rerun.py --new $(ls -d $W/D/*/mvtec/efficientnet/carpet | head -1) --ref results/reference_runs/D_mvtec_carpet_s7 2>&1 | tee -a $REP
say "== 2. speed, three repeats (T4, batch 1, full optimisation)"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.limit --format=csv,noheader | sed 's/^/before: /' | tee -a $REP
for i in 1 2 3; do
  TORCH_LOGS=perf_hints python analysis/head_to_head.py --kairos_root none --ours efficientnet:deep:avg --sizes 288 --precisions fp32,fp16 --full --twoview --iters 100 \
    --only "ours:efficientnet/avg+full,ours:efficientnet/avg+full[x2]" --out $W/speed_$i > $W/speed_$i.log 2>&1
  say "-- repeat $i  (cudagraph messages: $(grep -i -c -E 'skipping cudagraph|cudagraph.*skip' $W/speed_$i.log))"
  grep "+full" $W/speed_$i.md | awk -F'|' '{printf "   %-32s %-5s %s ms\n", $2, $4, $9}' | tee -a $REP
done
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | sed 's/^/after: /' | tee -a $REP
say "reference (thesis session): one view fp32 2.26 / fp16 1.66 ms; two views fp32 4.23 / fp16 2.99 ms"
