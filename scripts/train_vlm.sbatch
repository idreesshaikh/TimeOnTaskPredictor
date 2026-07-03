#!/bin/bash
#SBATCH --job-name=totvlm-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=ARCH:X86,GPU_SKU:A100
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Usage (from the repo root, so logs/ exists before SLURM opens the log files):
#   mkdir -p logs
#   sbatch scripts/train_vlm.sbatch                    # screen-only condition
#   sbatch scripts/train_vlm.sbatch configs/vlm_task.yaml   # screen+task condition
# Or submit the whole experiment chain at once: bash scripts/submit_all.sh
#
# wandb (configs/vlm*.yaml report_to): `wandb login` once on the login node,
# or export WANDB_MODE=offline and `wandb sync wandb/latest-run` afterwards.
# Timed out? Just resubmit — training auto-resumes from the last epoch
# checkpoint (the dry-run below is rerun as a cheap env re-check).

set -euo pipefail

CONFIG=${1:-configs/vlm.yaml}

cd ~/TimeOnTaskPredictor

# No venv activation needed: `uv run` manages .venv itself — only the uv
# binary must be reachable in the job shell.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 1; }

# Sync the env from uv.lock (picks up the torch 2.8 bump on first run).
uv sync --extra vlm

# Memory smoke test first; a failure aborts the job before the full run.
uv run python -m totvlm.train --config "$CONFIG" --dry-run

uv run python -m totvlm.train --config "$CONFIG"
