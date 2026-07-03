#!/bin/bash
# One-shot cluster runbook: submits the whole experiment as a SLURM
# dependency chain and prints the job graph. Run from the repo root on the
# login node:
#
#   bash scripts/submit_all.sh
#
# Prerequisites (all prepared off-cluster and rsync'd, since compute nodes
# may lack internet):
#   data/processed/rows_final.parquet     dataset with splits + targets
#   data/images_cache/                    resolved screenshots
#   data/axtree_features_cache/           axTree features incl. eval rows
#   data/external/vsgui10k/               read-only external set
#   artifacts/baseline_lgbm.txt           trained LightGBM baseline
#   wandb login                           once, on the login node
#
# Chain:
#   train screen-only ─┐
#                      ├─→ evaluate (head-to-head + figures)
#   train screen+task ─┘
#   train screen-only ───→ external zero-shot + AIM analysis
#
# Each training job runs its own --dry-run VRAM smoke test first and aborts
# before the full run if it fails.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

screen_jid=$(sbatch --parsable scripts/train_vlm.sbatch configs/vlm.yaml)
task_jid=$(sbatch --parsable scripts/train_vlm.sbatch configs/vlm_task.yaml)
eval_jid=$(sbatch --parsable --dependency="afterok:${screen_jid}:${task_jid}" \
    scripts/evaluate.sbatch)
ext_jid=$(sbatch --parsable --dependency="afterok:${screen_jid}" \
    scripts/external.sbatch)

cat <<EOF
Submitted:
  ${screen_jid}  train VLM (screen)        configs/vlm.yaml
  ${task_jid}  train VLM (screen+task)   configs/vlm_task.yaml
  ${eval_jid}  evaluate + figures        after ${screen_jid} AND ${task_jid}
  ${ext_jid}  external + AIM analysis   after ${screen_jid}

Watch:    squeue -u \$USER
Logs:     tail -f logs/train_${screen_jid}.out
Results:  artifacts/eval_report.md, artifacts/figures/,
          artifacts/external_report.md, artifacts/aim_analysis.md,
          wandb project tot-vlm
EOF
