#!/usr/bin/env bash
# Reproduce every number, table and figure in the conference paper from scratch.
# CPU-only: about 5-6 hours on a 16-thread laptop CPU (a GPU makes it minutes).
# Resumable: rerun the same command after an interruption.
#   bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/download_ham10000_full.py        # 10,015 images -> data/raw/images (skips existing files)
bash scripts/run_queue.sh                        # 7 training runs -> results/runs/
python -m src.analysis --primary ft_cw           # tables, CIs, tests, figures -> results/paper/
python -m src.gradcam --checkpoint models/ham10000_ft_cw_s0.pt
python scripts/build_paper.py                    # -> paper/ and paper_overleaf.zip
