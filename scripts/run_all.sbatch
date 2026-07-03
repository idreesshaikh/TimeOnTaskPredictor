#!/bin/bash
#SBATCH --job-name=totvlm-all
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=ARCH:X86,GPU_SKU:A100
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# THE one-command experiment: raw download → dataset → baseline → both VLM
# conditions → TEST head-to-head + figures → external zero-shot + AIM.
#
#   mkdir -p logs
#   sbatch scripts/run_all.sbatch     # then watch wandb project `tot-vlm`
#
# Every stage is idempotent: finished stages are skipped, image/axTree
# fetching resumes where it stopped, and training auto-resumes from the last
# epoch checkpoint — so if the job hits the time limit, JUST RESUBMIT IT.
#
# One-time setup on the login node before the first submission:
#   uv run hf auth login       # WebChain is gated
#   wandb login                # or export WANDB_MODE=offline + `wandb sync`
#
# Needs internet on the compute node (raw zip, screenshots, axTrees, the
# external set, HF checkpoints, wandb). If your compute nodes are offline,
# run stages 0–2 + prepare_external.py on the login node first — the guards
# below will then skip them — or use scripts/submit_all.sh with prefetched
# caches instead.

set -euo pipefail

cd ~/TimeOnTaskPredictor
mkdir -p logs

uv sync --extra vlm

echo "═══ 0. Raw trajectories (idempotent download + extract) ═══"
bash scripts/download_minimal.sh

echo "═══ 1. Dataset: audit → labels → screenshots → splits ═══"
[ -f artifacts/raw_audit.md ] || \
    uv run python scripts/build_dataset.py data/raw/all_json_files --audit
[ -f data/processed/rows.parquet ] || \
    uv run python scripts/build_dataset.py data/raw/all_json_files --labels
# Resumable: cached screenshots are skipped, so reruns only fetch what's new
# (budgets in configs/data.yaml resolve:). Splits reuse artifacts/splits.json.
uv run python scripts/build_dataset.py --resolve-images
uv run python scripts/build_dataset.py --splits

echo "═══ 2. No-image LightGBM baseline ═══"
[ -f artifacts/baseline_lgbm.txt ] || uv run python scripts/train_baseline.py

echo "═══ 3. VLM QLoRA SFT — both conditions ═══"
for CFG in configs/vlm.yaml configs/vlm_task.yaml; do
    CKPT=$(uv run python -c "import sys, yaml; \
        print(yaml.safe_load(open(sys.argv[1]))['paths']['output_dir'])" "$CFG")
    if [ -d "$CKPT/final" ]; then
        echo "── $CFG already trained ($CKPT/final exists) — skipping"
        continue
    fi
    echo "── $CFG: VRAM smoke test (writes to $CKPT-dryrun, then removed)"
    uv run python -m totvlm.train --config "$CFG" --dry-run
    rm -rf "$CKPT-dryrun"
    echo "── $CFG: full run (auto-resumes from $CKPT if checkpoints exist)"
    uv run python -m totvlm.train --config "$CFG"
done

echo "═══ 4. TEST head-to-head + paper figures ═══"
uv run python scripts/evaluate.py
uv run python scripts/make_figures.py

echo "═══ 5. External zero-shot (once) + AIM analysis ═══"
[ -f data/external/vsgui10k/items.csv ] || \
    uv run python scripts/prepare_external.py
[ -f artifacts/external_report.md ] || \
    uv run python scripts/validate_external.py
[ -f artifacts/aim_analysis.md ] || uv run python scripts/analyze_aim.py

echo "═══ DONE — reports in artifacts/, figures in artifacts/figures/, ═══"
echo "═══ runs in wandb project tot-vlm                               ═══"
