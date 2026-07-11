#!/bin/bash
#SBATCH --job-name=totvlm-cond
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch,gpu-shortrun,aisc-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="ARCH:X86&(GPU_SKU:A100|GPU_SKU:A30)"
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Any x86 Ampere card works (peak ~6.5 GB VRAM). H100 REMOVED 2026-07-09 —
# nvlink/hardware CUDA faults on the H100 nodes (see sweep_lam.sbatch);
# re-add GPU_SKU:H100 once fixed. gpu-shortrun's 1-day cap covers the 24 h
# limit; a single condition is ~10-12 h, and a timeout just resumes anyway.

# ONE final full-data condition on ONE node — run_everything.sh submits both
# in parallel after the λ-selection gate, then train.sbatch (which skips
# already-trained conditions) runs the TEST eval + figures + external:
#   sbatch scripts/train_condition.sbatch configs/vlm.yaml
#   sbatch scripts/train_condition.sbatch configs/vlm_task.yaml
#
# Idempotent + resumable: a trained condition (final/ exists) is skipped, an
# interrupted one auto-resumes from its last checkpoint — on timeout just
# resubmit, then rerun train.sbatch for the eval stage.

set -euo pipefail

CFG="${1:?usage: sbatch scripts/train_condition.sbatch configs/vlm.yaml}"

cd ~/TimeOnTaskPredictor
mkdir -p logs
[ -f "$CFG" ] || { echo "ERROR: $CFG not found" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=online   # loss/parse-rate curves land in wandb project tot-vlm
command -v uv >/dev/null || { echo "ERROR: uv not on PATH — install on the login node: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# Fail fast if setup.sbatch has not produced its outputs yet.
for f in data/processed/rows_final.parquet artifacts/splits.json artifacts/baseline_lgbm.txt; do
    [ -f "$f" ] || { echo "ERROR: $f missing — run scripts/setup.sbatch first" >&2; exit 1; }
done

uv sync --extra vlm

echo "=== 0. Privileged-feature teacher (CPU, idempotent — OOF LightGBM) ==="
[ -f artifacts/lupi_teacher_preds.parquet ] || \
    uv run python scripts/make_lupi_teacher.py

echo "=== 1. VLM QLoRA SFT — $CFG (blends the teacher into its train targets) ==="
CKPT=$(uv run python -c "import sys, yaml; \
    print(yaml.safe_load(open(sys.argv[1]))['paths']['output_dir'])" "$CFG")
if [ -d "$CKPT/final" ]; then
    echo "--- $CFG already trained ($CKPT/final exists) — skipping"
else
    echo "--- $CFG: full run (auto-resumes from $CKPT if checkpoints exist)"
    uv run python -m totvlm.train --config "$CFG"
fi

echo "=== DONE $CFG. When both conditions are trained, train.sbatch runs the"
echo "    TEST eval + figures + external (it skips training that is done). ==="
