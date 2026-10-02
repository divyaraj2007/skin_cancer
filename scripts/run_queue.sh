#!/usr/bin/env bash
# Resumable experiment queue: finished runs are skipped, interrupted runs resume from their last epoch.
cd "$(dirname "$0")/.."
run() {  # run <name> <seed> [extra args...]
  local name=$1 seed=$2; shift 2
  if [ -f "results/runs/${name}_s${seed}/test_probs.npy" ]; then echo "skip ${name}_s${seed}"; return; fi
  python -m src.experiment --name "$name" --seed "$seed" --epochs 10 "$@" || exit 1
}
run frozen_ce 0
run ft_cw 0 --finetune --class-weights --save-model
run ft_cw_imgsplit 0 --finetune --class-weights --split image
run ft_ce 0 --finetune
run frozen_cw 0 --class-weights
run ft_cw 1 --finetune --class-weights
run ft_cw 2 --finetune --class-weights
echo ALL_DONE
