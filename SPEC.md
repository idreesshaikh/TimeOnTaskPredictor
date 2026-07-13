# Project: Screenshot → Time-on-Task Prediction (per screen, and per task)

## Goal
Predict how long a user dwells on a web UI screen before acting — and, by summing
per-screen predictions within a trajectory, how long the whole task takes.
Research question (v3): how accurately can per-screen — and, by aggregation,
whole-task — Time-on-Task be predicted from rendered UI screens, and how much of
that predictability resides in the screen alone versus the user's stated task?

Two core conditions, both SCREENSHOT-ONLY at inference:
- screen (configs/vlm.yaml — the science core)
- screen+task-title (configs/vlm_task.yaml — the predictor)
They are identical except the task-title prompt flag, so their gap = the
goal-driven share of dwell.

Two additional FEATURE-INPUT conditions (upper bound, NOT screenshot-only):
- screen+features (configs/vlm_feat.yaml)
- screen+features+task (configs/vlm_feat_task.yaml — overlay adding only
  the task title, mirroring the core pair)
Here the six screen-describing axTree stats are GIVEN in the user prompt as
a `ui:` line — at train AND inference (legitimate: they are knowable the
moment the screen renders). No distillation, no scaffold: `lupi: null`,
plain winsorized labels. Reading: how much of the screenshot-only
conditions' remaining error is missing screen-structure knowledge? Excluded
from external validation (VSGUI10K has no axTrees). Outcome features stay
hindsight-only and never enter any prompt.

The metadata features are split by WHEN they are knowable, and each half
reaches training through its own published mechanism — neither is ever a
model input:
- SCREEN-DESCRIBING stats (six axTree counts: nodes, interactive, links,
  buttons, inputs, text length) are knowable the moment the screen renders →
  they supervise a `ui:` line the model emits BEFORE the dwell line
  (auxiliary output supervision / learning-from-hints; rationale
  distillation). At inference the model generates its own estimates from the
  pixels.
- OUTCOME/SESSION features (nav flag, click-target area, step index) are
  only knowable in hindsight — the click terminates the very dwell being
  predicted → they reach training solely through λ-blended targets toward an
  out-of-fold LightGBM teacher (generalized distillation / LUPI; Lopez-Paz
  et al. 2016). Blend + teacher logic lives in `totvlm/lupi.py`; the λ grid
  is fixed: {0, 0.25, 0.5, 0.75, 1}, selected on VALIDATION only.

Comparison ladder (all on identical TEST rows):
floors (train mean/median) < CNN (pixels only, no distillation — what do
generic visual features buy?) ≶ LightGBM (features only, no image — what does
the metadata buy? doubles as the teacher's model family) < VLM (screen) <
VLM (screen+task) ≤? VLM (screen+features[, +task]) (ui-stats given at
inference — the upper-bound pair). configs/vlm.yaml is the base;
configs/vlm_task.yaml is a thin `base: vlm.yaml` overlay that only adds the
task title, so the "identical except one flag" claim holds in the files
themselves (and vlm_feat_task.yaml repeats the pattern on vlm_feat.yaml).

## Non-negotiable rules
- Reproducibility: global seed = 42 (Python, NumPy, torch, split assignment). Python 3.12.
- Splits are by REGISTRABLE DOMAIN (eTLD+1 via `tldextract`). A domain appears in exactly
  one of {train, val, test}. Never split by trajectory or step. Save assignments to
  `artifacts/splits.json` and reuse them everywhere — never recompute silently.
- The external validation set (VSGUI10K) is READ-ONLY and is
  evaluated exactly once, zero-shot. It is NEVER used to pick features, hyperparameters,
  thresholds, or the winsorization cap. Keep it under `data/external/` and never reference
  that path from any training/tuning code.
- All tunable numbers live in `configs/*.yaml`. No magic numbers in code. A
  config may `base:` another and override only what differs (deep-merged) —
  shared knobs are declared once, in the base.
- Every stage writes a small JSON/markdown "card" of stats to `artifacts/`.
- TEST discipline: TEST is decoded once, by scripts/evaluate.py, for the
  final conditions only. λ and every other choice are selected on VAL.
  (Historical caveat, disclose in the paper: v1/v2 exploration consumed TEST
  several times; v3 results must be labeled accordingly.)

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
  vision AND language attention+MLP layers, r=16, alpha=16, dropout=0.05. BOTH conditions
  use the identical setup, so the two-condition gap stays a clean measure of the
  task-title effect.
- Output format (v3, both conditions): two lines —
  `ui: nodes=<int> interactive=<int> links=<int> buttons=<int> inputs=<int> text=<int>`
  then `dwell_seconds: X.X` (1 decimal). The `ui:` line is supervised from
  real axTree stats on train AND val (scaffold_stats.parquet); parsers read
  the dwell from the LAST line, so scaffolded and plain outputs score
  identically.
- CNN baseline: ImageNet-pretrained ResNet-50 regressor on the screenshot
  (scripts/train_cnn_baseline.py, configs/cnn.yaml) — no features, no
  distillation, no scaffold; the pixels-only control row.
- Model selection: evaluate + checkpoint several times per epoch and DEPLOY THE BEST
  checkpoint by validation loss (`load_best_model_at_end`) — never the last epoch, which
  overfits. λ is selected on VALIDATION over the fixed grid {0, 0.25, 0.5, 0.75, 1}
  (configs/sweeps/lam*.yaml — each a FULL-fidelity run of the screen condition,
  scored on the full val split by scripts/decode_val.py); scripts/select_lambda.py
  ranks them, patches configs/vlm.yaml, and with --deploy promotes the winner's
  adapters to be the screen-only condition (no duplicate retrain). λ=0 doubles as
  the no-distillation ablation, λ=1 as the pure-teacher ablation. Neither
  selection ever touches TEST.
- Training needs a CUDA GPU (`uv sync --extra vlm`). Always run
  `python -m totvlm.train --dry-run` (the memory smoke test) before a full run.

## Pipeline map (each stage writes its card to artifacts/)
1. `scripts/build_dataset.py --audit | --labels | --resolve-images | --splits`
   → rows_final.parquet, splits.json, dataset_card.md
2. `scripts/train_baseline.py` → no-image LightGBM baseline + baseline_report.md
2b. `scripts/make_lupi_teacher.py` → lupi_teacher_preds.parquet (OOF LightGBM
   on the outcome features, folds grouped by registrable domain — the `lupi:`
   soft-target half) + scaffold_stats.parquet (the six screen-describing
   axTree counts for train+val+test — `ui:` target supervision on train+val,
   prompt INPUT for the feature-input conditions on every split) +
   lupi_teacher_card.md
2c. `scripts/train_cnn_baseline.py` (GPU) → pixels-only ResNet-50 control:
   cnn_test_preds.parquet + cnn_baseline_report.md (TEST decoded once, inside
   the script, same discipline as the LightGBM baseline)
2d. λ grid (GPU, VAL only): train each `configs/sweeps/lam*.yaml` (5 runs max,
   full fidelity) → `scripts/decode_val.py` per run (full-val metrics) →
   `scripts/select_lambda.py --write --deploy` (winner λ → configs/vlm.yaml;
   winner adapters → the screen-only condition). TEST is never used to choose λ.
3. `python -m totvlm.train --config configs/vlm_task.yaml` → the screen+task
   condition at the selected λ (GPU; auto-resumes; deploys best-by-val-loss)
3b. `python -m totvlm.train --config configs/vlm_feat.yaml` then
   `configs/vlm_feat_task.yaml` → the feature-input pair (GPU; independent
   of λ selection — true labels, no blend)
4. `scripts/evaluate.py` → TEST head-to-head (floors, CNN, LightGBM, all four
   VLM conditions), per-screen AND task-level (per-trajectory sums) +
   eval_report.md + eval_metrics.json + eval_predictions.parquet
5. `scripts/make_figures.py` → the paper figure set (artifacts/figures/),
   CPU-only, rebuilt from cached artifacts — never runs a model
6. `scripts/prepare_external.py` + `scripts/validate_external.py` +
   `scripts/analyze_aim.py`
   → zero-shot external check + external_report.md (evaluate-once, guarded)

Cluster: two idempotent, resumable jobs (resubmit on timeout — they continue).
`sbatch scripts/setup.sbatch` (CPU: download → dataset → baseline → teacher +
scaffold stats → external prep), then `sbatch scripts/train.sbatch` (GPU: CNN
baseline → λ grid → select+deploy → screen+task condition → feature-input
pair → eval + figures → external zero-shot + AIM).

## Verified test oracles (use as pytest ground truth)
Trajectory id `yjeXPEBxd5EACsDz4xPWx` → 3 rows:
  dwell_s ≈ [54.046 (nav=False), 33.460 (nav=True), 73.887 (nav=True)]
Trajectory id `8uJRwmgg3iwq9-rnZ_MpJ` → 6 rows:
  dwell_s ≈ [6.683 (True), 5.716 (True), 0.830 (False), 32.787 (True),
             19.290 (True), 20.836 (True)]
(±0.01s tolerance. These come from real createdTime deltas and must not regress.)
