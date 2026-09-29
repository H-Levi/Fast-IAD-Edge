#!/usr/bin/env bash
# CPU speed check on a machine WITHOUT a GPU (e.g. a Kaggle CPU session, internet on for the two comparator clones), about 10 min.
# Usage: bash scripts/verify_cpu.sh [workdir]
set -u
W=${1:-verify_cpu}; mkdir -p $W
python -c "import torch, platform, os; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '| cpu', platform.processor() or platform.machine(), '| threads', torch.get_num_threads())"
grep -m1 "model name" /proc/cpuinfo 2>/dev/null
pip install -q timm >/dev/null 2>&1
test -d $W/EfficientAD || git clone -q https://github.com/nelson1425/EfficientAD $W/EfficientAD
test -d $W/KairosAD || git clone -q --recurse-submodules https://github.com/intelligolabs/KairosAD $W/KairosAD
pip install -q -e $W/KairosAD/MobileSAM >/dev/null 2>&1
C="--kairos_root $W/KairosAD --efficientad_root $W/EfficientAD --ours efficientnet:deep:avg --precisions fp32 --iters 20"
python analysis/head_to_head.py $C --sizes 256 --only efficientad_s --out $W/cpu_efficientad > $W/cpu.log 2>&1
python analysis/head_to_head.py $C --sizes 288 --only ours:efficientnet/avg,kairosad --out $W/cpu_ours_kairos >> $W/cpu.log 2>&1
python scripts/verify_cpu.py --new_efficientad $W/cpu_efficientad.csv --new_ours_kairos $W/cpu_ours_kairos.csv
