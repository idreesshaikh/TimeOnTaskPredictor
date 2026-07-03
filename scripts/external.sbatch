#!/bin/bash
#SBATCH --job-name=totvlm-external
#SBATCH --output=logs/external_%j.out
#SBATCH --error=logs/external_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=ARCH:X86,GPU_SKU:A100
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Zero-shot external validation (O3) + AIM theory-grounding analysis.
# Runs ONCE, after the screen-only condition finishes (submit_all.sh chains
# it). The evaluate-once guard makes accidental resubmission a no-op failure
# rather than a silent re-evaluation — that is intended.
# Requires data/external/<source>/ to be prepared already (items.csv etc.,
# built by scripts/prepare_external.py — needs internet, so run it off-cluster
# and rsync data/external/ over).

set -euo pipefail

cd ~/TimeOnTaskPredictor

# No venv activation needed: `uv run` manages .venv itself — only the uv
# binary must be reachable in the job shell.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 1; }

uv sync --extra vlm

uv run python scripts/validate_external.py
uv run python scripts/analyze_aim.py
