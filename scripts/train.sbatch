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
#SBATCH --time=72:00:00

# GPU job 2/2: both VLM conditions -> TEST evaluation + figures ->
# external zero-shot + AIM. Run AFTER scripts/setup.sbatch succeeded:
#   sbatch scripts/train.sbatch
#
# Idempotent + resumable: trained conditions are skipped, training
# auto-resumes from the last epoch checkpoint — on timeout, just resubmit.

set -euo pipefail

cd ~/TimeOnTaskPredictor
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH — install on the login node: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# Fail fast if setup.sbatch has not produced its outputs yet.
for f in data/processed/rows_final.parquet artifacts/splits.json artifacts/baseline_lgbm.txt; do
    [ -f "$f" ] || { echo "ERROR: $f missing — run scripts/setup.sbatch first" >&2; exit 1; }
done

uv sync --extra vlm

echo "=== 1. VLM QLoRA SFT — both conditions ==="
for CFG in configs/vlm.yaml configs/vlm_task.yaml; do
    CKPT=$(uv run python -c "import sys, yaml; \
        print(yaml.safe_load(open(sys.argv[1]))['paths']['output_dir'])" "$CFG")
    if [ -d "$CKPT/final" ]; then
        echo "--- $CFG already trained ($CKPT/final exists) — skipping"
        continue
    fi
    echo "--- $CFG: VRAM smoke test"
    uv run python -m totvlm.train --config "$CFG" --dry-run
    rm -rf "$CKPT-dryrun"
    echo "--- $CFG: full run (auto-resumes from $CKPT if checkpoints exist)"
    uv run python -m totvlm.train --config "$CFG"
done

echo "=== 2. TEST head-to-head + paper figures ==="
uv run python scripts/evaluate.py
uv run python scripts/make_figures.py

echo "=== 3. External zero-shot (once) + AIM analysis ==="
[ -f artifacts/external_report.md ] || \
    uv run python scripts/validate_external.py
[ -f artifacts/aim_analysis.md ] || uv run python scripts/analyze_aim.py

echo "=== DONE — reports in artifacts/, figures in artifacts/figures/, wandb project tot-vlm ==="
