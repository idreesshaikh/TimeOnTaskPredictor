#!/bin/bash
#SBATCH --job-name=totvlm-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=ARCH:X86,GPU_SKU:A100
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# TEST head-to-head + paper figures. Run AFTER both training jobs
# (scripts/submit_all.sh chains this automatically):
#   mkdir -p logs && sbatch scripts/evaluate.sbatch
#
# GPU decode fills whichever prediction caches are missing; the report and
# figures are CPU. The LightGBM leg fetches axTree features for eval rows —
# the resumable cache (data/axtree_features_cache/) should be prefetched and
# rsync'd from a machine with internet if compute nodes are offline.
# wandb: `wandb login` once on the login node, or export WANDB_MODE=offline
# and `wandb sync` later.

set -euo pipefail

cd ~/TimeOnTaskPredictor

# No venv activation needed: `uv run` manages .venv itself — only the uv
# binary must be reachable in the job shell.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 1; }

uv sync --extra vlm

uv run python scripts/evaluate.py
uv run python scripts/make_figures.py
