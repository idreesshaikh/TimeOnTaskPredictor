#!/bin/bash
#SBATCH --job-name=totvlm-setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=cpu-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# CPU job 1/2: downloads + dataset + baseline. No GPU needed.
#   mkdir -p logs && sbatch scripts/setup.sbatch
# Then, once it finishes: sbatch scripts/train.sbatch
#
# One-time before the first submission (login node):
#   uv run hf auth login      # WebChain is gated
#   wandb login               # or export WANDB_MODE=offline
#
# Every stage is idempotent and image fetching is resumable — if the job
# times out or dies, just resubmit it.

set -euo pipefail

cd ~/TimeOnTaskPredictor
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH — install on the login node: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

uv sync

echo "=== 0. Raw trajectories ==="
bash scripts/download_minimal.sh

echo "=== 1. Dataset: audit -> labels -> screenshots -> splits ==="
[ -f artifacts/raw_audit.md ] || \
    uv run python scripts/build_dataset.py data/raw/all_json_files --audit
[ -f data/processed/rows.parquet ] || \
    uv run python scripts/build_dataset.py data/raw/all_json_files --labels
uv run python scripts/build_dataset.py --resolve-images
uv run python scripts/build_dataset.py --splits

echo "=== 2. No-image LightGBM baseline ==="
[ -f artifacts/baseline_lgbm.txt ] || uv run python scripts/train_baseline.py

echo "=== 3. External set (download only — evaluated later, once) ==="
[ -f data/external/vsgui10k/items.csv ] || \
    uv run python scripts/prepare_external.py

echo "=== SETUP DONE — now: sbatch scripts/train.sbatch ==="
