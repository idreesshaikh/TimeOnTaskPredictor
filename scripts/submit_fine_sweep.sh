#!/bin/bash
# Submit the fine λ sweep around 0.5 — one sbatch job per λ, so Slurm puts
# each on its own GPU node and they train in parallel.
#
#   ./scripts/submit_fine_sweep.sh                 # default grid below
#   ./scripts/submit_fine_sweep.sh lam45 lam55     # explicit subset
#
# Each job is scripts/sweep_lam.sbatch (idempotent + resumable — resubmit any
# λ that times out). When all are done:
#   uv run python scripts/select_lambda.py --write

set -euo pipefail
cd "$(dirname "$0")/.."

# lam10 = the "mostly images, only 10% teacher" hypothesis — kept in the grid
# so it is tested empirically, not argued about (see docs/HOW_IT_LEARNS.md §1).
GRID=("$@")
[ ${#GRID[@]} -gt 0 ] || GRID=(lam10 lam30 lam40 lam45 lam50 lam55 lam60 lam65)

for lam in "${GRID[@]}"; do
    if [ ! -f "configs/sweeps/${lam}.yaml" ]; then
        echo "SKIP $lam: configs/sweeps/${lam}.yaml missing" >&2
        continue
    fi
    sbatch --job-name="$lam" scripts/sweep_lam.sbatch "$lam"
done

squeue -u "$USER" 2>/dev/null || true
