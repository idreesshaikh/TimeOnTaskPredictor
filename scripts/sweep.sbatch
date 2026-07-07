#!/bin/bash
#SBATCH --job-name=totvlm-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=ARCH:X86,GPU_SKU:A100
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# OPTIONAL GPU job, runs BETWEEN setup.sbatch and train.sbatch: the λ sweep.
# Trains configs/sweeps/lam*.yaml — the image+features model at each candidate
# distillation weight λ, on a fast train subsample (VAL stays full) — then prints
# the VALIDATION winner. Copy that λ into configs/vlm.yaml, then run train.sbatch.
#   sbatch scripts/sweep.sbatch
#
# Idempotent + resumable: a λ whose final/ exists is skipped, training
# auto-resumes from the last checkpoint — on timeout, just resubmit.

set -euo pipefail

cd ~/TimeOnTaskPredictor
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=online   # loss/parse-rate curves land in wandb project tot-vlm
command -v uv >/dev/null || { echo "ERROR: uv not on PATH — install on the login node: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# Fail fast if setup.sbatch has not produced its outputs yet.
for f in data/processed/rows_final.parquet artifacts/splits.json; do
    [ -f "$f" ] || { echo "ERROR: $f missing — run scripts/setup.sbatch first" >&2; exit 1; }
done

uv sync --extra vlm

echo "=== 0. Privileged-feature teacher (CPU, idempotent — OOF LightGBM) ==="
[ -f artifacts/lupi_teacher_preds.parquet ] || \
    uv run python scripts/make_lupi_teacher.py

echo "=== 1. λ sweep — image+features at each λ (fast subsample, full VAL) ==="
for CFG in configs/sweeps/lam*.yaml; do
    CKPT=$(uv run python -c "import sys, yaml; \
        print(yaml.safe_load(open(sys.argv[1]))['paths']['output_dir'])" "$CFG")
    if [ -d "$CKPT/final" ]; then
        echo "--- $CFG already done ($CKPT/final exists) — skipping"
        continue
    fi
    echo "--- $CFG: sweep run (auto-resumes from $CKPT if checkpoints exist)"
    uv run python -m totvlm.train --config "$CFG"
done

echo "=== 2. Pick λ on VALIDATION (never TEST) ==="
uv run python scripts/select_lambda.py

echo "=== DONE — set the winning λ above as lupi.lambda in configs/vlm.yaml, then: sbatch scripts/train.sbatch ==="