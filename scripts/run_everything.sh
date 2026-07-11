#!/bin/bash
# THE one-command experiment. Run this on the LOGIN node and walk away:
#
#   ./scripts/run_everything.sh                       # full grid (13 λ values)
#   ./scripts/run_everything.sh lam40 lam45 lam50     # custom grid
#
# What gets queued (everything is submitted NOW; Slurm runs it in order):
#
#   N × sweep_lam.sbatch          one GPU node per λ, all in parallel (~3 h each):
#                                 train on the 15k-row subsample, then decode the
#                                 FULL val split → val_full_metrics.json
#            │ afterany (all)
#   finalize_select.sbatch        CPU gate: verify every λ finished, pick the
#                                 winner on FULL validation → configs/vlm.yaml,
#                                 clear stale finals if λ changed. Fails loudly
#                                 if any λ is incomplete — then the final run
#                                 never starts on a bad selection.
#            │ afterok
#   2 × train_condition.sbatch    the winner λ only, one GPU node per
#                                 condition in parallel (~10–12 h each):
#                                 image+features and image+features+task
#            │ afterok (both)
#   train.sbatch                  both conditions already trained → skips
#                                 straight to TEST evaluated ONCE → figures →
#                                 external zero-shot + AIM
#
# TEST discipline: only the selected λ is ever evaluated on TEST — testing
# every λ would turn TEST into a second validation set (SPEC.md forbids it).
#
# Rough timeline: each sweep job is ~3 h once it gets a GPU; how many run at
# once depends on free cards (the pool accepts A100/H100/A30 across
# gpu-batch/gpu-shortrun/aisc-batch, so jobs land wherever there is space).
# The two final conditions run in parallel (~10-12 h each), then the eval
# stage (~2-4 h) — roughly half the old sequential 24-26 h. Everything is
# idempotent — if a sweep or condition job timed out, resubmit just it, then
# rerun the jobs after it (the gate's log prints the exact commands).
#
# Escape hatch: extra sbatch flags for every GPU job (sweeps + final train;
# the CPU gate is untouched) via TOT_SBATCH_ARGS, e.g. if aisc-batch rejects
# our account at submit time:
#   TOT_SBATCH_ARGS="--partition=gpu-batch,gpu-shortrun" ./scripts/run_everything.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

GRID=("$@")
[ ${#GRID[@]} -gt 0 ] || GRID=(lam00 lam10 lam25 lam30 lam35 lam40 lam45 lam50 lam55 lam60 lam65 lam75 lam90)

for lam in "${GRID[@]}"; do
    [ -f "configs/sweeps/${lam}.yaml" ] || { echo "ERROR: configs/sweeps/${lam}.yaml missing" >&2; exit 1; }
done

echo "Submitting ${#GRID[@]} λ sweep jobs (one GPU node each, in parallel):"
SWEEP_IDS=()
for lam in "${GRID[@]}"; do
    id=$(sbatch --parsable ${TOT_SBATCH_ARGS:-} --job-name="$lam" scripts/sweep_lam.sbatch "$lam")
    echo "  $lam  → job $id"
    SWEEP_IDS+=("$id")
done

DEP=$(IFS=:; echo "${SWEEP_IDS[*]}")
GATE_ID=$(sbatch --parsable --dependency="afterany:$DEP" \
          scripts/finalize_select.sbatch "${GRID[@]}")
echo "  gate  → job $GATE_ID   (picks λ on FULL val; runs after all sweeps end)"

COND_A=$(sbatch --parsable ${TOT_SBATCH_ARGS:-} --dependency="afterok:$GATE_ID" \
         --job-name=cond-img scripts/train_condition.sbatch configs/vlm.yaml)
COND_B=$(sbatch --parsable ${TOT_SBATCH_ARGS:-} --dependency="afterok:$GATE_ID" \
         --job-name=cond-imgtask scripts/train_condition.sbatch configs/vlm_task.yaml)
echo "  conds → jobs $COND_A + $COND_B   (winner λ, image+features ∥ image+features+task)"

TRAIN_ID=$(sbatch --parsable ${TOT_SBATCH_ARGS:-} --dependency="afterok:$COND_A:$COND_B" scripts/train.sbatch)
echo "  final → job $TRAIN_ID   (both conditions done → TEST once, figures, external)"

echo
echo "All queued. Check on it with:  squeue -u \$USER"
echo "Gate log:   logs/totvlm-gate_${GATE_ID}.out (after the sweeps finish)"
echo "Final log:  logs/train_${TRAIN_ID}.out"
echo "Results land in artifacts/ — eval_report.md is the paper table."
echo "If a job sits in DependencyNeverSatisfied, its upstream failed — read that"
echo "log. Gate failed: it names the λ jobs to resubmit and the commands to rerun."
echo "A condition timed out: resubmit it (train_condition.sbatch <config>), then"
echo "sbatch --dependency=afterok:<id> scripts/train.sbatch for the eval stage."
squeue -u "$USER" 2>/dev/null || true
