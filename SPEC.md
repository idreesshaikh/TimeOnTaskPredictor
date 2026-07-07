# Project: Screenshot → Time-on-Task Prediction (per screen, and per task)

## Goal
Predict how long a user dwells on a web UI screen before acting — and, by summing
per-screen predictions within a trajectory, how long the whole task takes.
Research question (v2): how accurately can per-screen — and, by aggregation,
whole-task — Time-on-Task be predicted from rendered UI screens, and how much of
that predictability resides in the screen alone versus the user's stated task?
Two trained conditions, and NEITHER is a pixels-only model:
- image+features (configs/vlm.yaml — the science core)
- image+features+task-title (configs/vlm_task.yaml — the predictor)
Both read only the screenshot (+ task title for the second) at INFERENCE, but
their TRAIN targets are λ-blended toward an out-of-fold LightGBM teacher that
saw the privileged features (axTree stats, nav flag, step index, click area) —
features the screenshot itself can't show. Generalized distillation / learning
under privileged information: the metadata shapes what the model learns but
never appears at inference, so the deployed predictor stays screenshot-only
while not being handicapped relative to the metadata-rich LightGBM baseline.
The two conditions are identical except the task-title prompt flag, so their
gap = the goal-driven share of dwell. They are compared against the no-image
LightGBM baseline and the predict-the-median floors.
All privileged-info logic (the teacher's out-of-fold predictions and the λ
target blend) lives in `totvlm/lupi.py`. configs/vlm.yaml is the base (it
carries the `lupi:` block); configs/vlm_task.yaml is a thin `base: vlm.yaml`
overlay that only adds the task title, so the "identical except one flag" claim
holds in the files themselves.

## Non-negotiable rules
- Reproducibility: global seed = 42 (Python, NumPy, torch, split assignment). Python 3.12.
- Splits are by REGISTRABLE DOMAIN (eTLD+1 via `tldextract`). A domain appears in exactly
  one of {train, val, test}. Never split by trajectory or step. Save assignments to
  `artifacts/splits.json` and reuse them everywhere.
- The external validation set (VSGUI10K) is READ-ONLY and is
  evaluated exactly once, zero-shot. It is NEVER used to pick features, hyperparameters,
  thresholds, or the winsorization cap. Keep it under `data/external/` and never reference
  that path from any training/tuning code.
- All tunable numbers live in `configs/*.yaml`. No magic numbers in code. A
  config may `base:` another and override only what differs (deep-merged) —
  shared knobs are declared once, in the base.
- Every stage writes a small JSON/markdown "card" of stats to `artifacts/`.

## Label definition (the load-bearing spec)
Input data = a list of trajectory objects. Each has `steps` (list). Each step may have:
`type` (launchApp|click|type|select|...), `tabId` (int), `href` (str), `img` (str: URL or
hash), `axTree` (URL or null), `createdTime` (epoch ms; ABSENT on launchApp), `rect`,
`viewport`, `scrollX/Y`, action `value`/`title`.

Row construction:
1. Within each trajectory, process steps in file order.
2. Drop all `launchApp` rows (no createdTime; not a perceived screen).
3. MERGE consecutive actionable steps that share an identical `img` string AND same tabId
   into one "screen unit". The unit's `createdTime` = that of its LAST step. Its screenshot
   = the `img` of its FIRST step. Its `href`/`axTree` = first step's.
4. For each screen unit, its predecessor is the previous screen unit IN THE SAME tabId.
   The FIRST unit in a tabId has no predecessor → drop it as a target (but its createdTime
   still serves as predecessor for later units, and cross-tab pairs never form a dwell).
5. `dwell_ms = unit.createdTime − predecessor.createdTime`. `dwell_s = dwell_ms / 1000`.
6. `is_navigation = (unit.href != predecessor.href)`  → True means a URL change (page load
   is bundled into the dwell). False = in-page interaction (cleaner cognitive signal).
7. Hard filters: drop dwell_s <= 0.05 or dwell_s > 600 (idle/errors), configurable.
8. Winsorization cap = the p95 of dwell_s ON THE TRAIN SPLIT ONLY; apply to all splits for
   the training target; keep the raw value for reporting. Log the cap in the dataset card.
9. Model target = `log1p(dwell_s_winsorized)`. Report metrics in both log and seconds.

## Backbone / training
- Model: Qwen3-VL-4B-Instruct via Unsloth `FastVisionModel`, 4-bit QLoRA, LoRA on the
  vision AND language attention+MLP layers, r=16, alpha=16, dropout=0.05. (v2: the vision
  layers were frozen in the first run; LoRA-tuning them lets the screenshot itself inform
  the estimate. BOTH conditions use the identical setup, so the two-condition gap stays a
  clean measure of the task-title effect.)
- Primary output = Path A: SFT the model to emit exactly `dwell_seconds: X.X` (1 decimal).
- Model selection: evaluate + checkpoint several times per epoch and DEPLOY THE BEST
  checkpoint by validation loss (`load_best_model_at_end`) — never the last epoch, which
  overfits. The single distillation weight λ (configs/vlm.yaml `lupi.lambda`) is likewise
  selected on VALIDATION only (candidate grid in configs/sweeps/, searched
  bracket-and-refine by scripts/run_sweep.py — coarse anchors, then only the
  neighbours of the best λ until both are worse — and the winner recorded via
  scripts/select_lambda.py).
  Neither selection ever touches TEST.
- Training needs a CUDA GPU (`uv sync --extra vlm`). Always run
  `python -m totvlm.train --dry-run` (the memory smoke test) before a full run.

## Pipeline map (each stage writes its card to artifacts/)
1. `scripts/build_dataset.py --audit | --labels | --resolve-images | --splits`
   → rows_final.parquet, splits.json, dataset_card.md
2. `scripts/train_baseline.py` → no-image LightGBM baseline + baseline_report.md
2b. `scripts/make_lupi_teacher.py` → lupi_teacher_preds.parquet +
   lupi_teacher_card.md (OOF LightGBM on privileged features, folds grouped
   by registrable domain — feeds the `lupi:` block that BOTH trained
   conditions use)
2c. λ selection (GPU, VAL only): `scripts/run_sweep.py` bracket-and-refine
   search over `configs/sweeps/lam*.yaml` (trains the coarse anchors, then only
   the grid neighbours of the best-VAL λ until both are worse — typically 4-6
   of the 8 overlays) → `scripts/select_lambda.py` picks the lowest-VAL-error λ
   → copy it into `configs/vlm.yaml`. TEST is never used to choose λ.
3. `python -m totvlm.train --config configs/vlm.yaml | configs/vlm_task.yaml`
   → QLoRA adapters + train cards for the two conditions (GPU; each blends the
   privileged-feature teacher into its train targets; deploys the best-by-val-loss
   checkpoint)
4. `scripts/evaluate.py` → TEST head-to-head (floors < LightGBM <
   VLM image+features < VLM image+features+task), per-screen AND task-level
   (per-trajectory sums) + eval_report.md + eval_metrics.json +
   eval_predictions.parquet
5. `scripts/make_figures.py` → the paper figure set (artifacts/figures/),
   CPU-only, rebuilt from cached artifacts — never runs a model
6. `scripts/prepare_external.py` + `scripts/validate_external.py` +
   `scripts/analyze_aim.py`
   → zero-shot external check + external_report.md (evaluate-once, guarded)

Cluster: two idempotent, resumable jobs (resubmit on timeout — they continue).
`sbatch scripts/setup.sbatch` (CPU: download → dataset → baseline → external
prep), then `sbatch scripts/train.sbatch` (GPU: both conditions → eval +
figures → external zero-shot + AIM).

## Verified test oracles (use as pytest ground truth)
Trajectory id `yjeXPEBxd5EACsDz4xPWx` → 3 rows:
  dwell_s ≈ [54.046 (nav=False), 33.460 (nav=True), 73.887 (nav=True)]
Trajectory id `8uJRwmgg3iwq9-rnZ_MpJ` → 6 rows:
  dwell_s ≈ [6.683 (True), 5.716 (True), 0.830 (False), 32.787 (True),
             19.290 (True), 20.836 (True)]
(±0.01s tolerance. These come from real createdTime deltas and must not regress.)