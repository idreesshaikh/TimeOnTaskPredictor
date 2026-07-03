# Project: Screenshot → Per-Screen Time-on-Task Regression

## Goal
Predict how long a user dwells on a web UI screen before acting, from the screenshot
alone. Research question: how much of per-screen Time-on-Task is recoverable from the
screen itself, independent of the user's goal/history?

## Non-negotiable rules
- Reproducibility: global seed = 42 (Python, NumPy, torch, split assignment). Python 3.12.
- Splits are by REGISTRABLE DOMAIN (eTLD+1 via `tldextract`). A domain appears in exactly
  one of {train, val, test}. Never split by trajectory or step. Save assignments to
  `artifacts/splits.json` and reuse them everywhere.
- The external validation set (TaskSense or its documented fallback) is READ-ONLY and is
  evaluated exactly once, zero-shot. It is NEVER used to pick features, hyperparameters,
  thresholds, or the winsorization cap. Keep it under `data/external/` and never reference
  that path from any training/tuning code.
- All tunable numbers live in `configs/*.yaml`. No magic numbers in code.
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
- Model: Qwen3-VL-4B-Instruct via Unsloth `FastVisionModel`, 4-bit QLoRA, freeze vision
  layers, LoRA on language attention+MLP, r=16, alpha=16.
- Primary output = Path A: SFT the model to emit exactly `dwell_seconds: X.X` (1 decimal).
- Training needs a CUDA GPU (`uv sync --extra vlm`). Always run
  `python -m totvlm.train --dry-run` (the memory smoke test) before a full run.

## Pipeline map (each stage writes its card to artifacts/)
1. `scripts/build_dataset.py --audit | --labels | --resolve-images | --splits`
   → rows_final.parquet, splits.json, dataset_card.md
2. `scripts/train_baseline.py` → no-image LightGBM baseline + baseline_report.md
3. `python -m totvlm.train` → QLoRA adapters + vlm_train_card.md (GPU)
4. `scripts/evaluate.py` → TEST head-to-head VLM vs baseline + eval_report.md
5. `scripts/prepare_external.py` + `scripts/validate_external.py`
   → zero-shot external check + external_report.md (evaluate-once, guarded)

## Verified test oracles (use as pytest ground truth)
Trajectory id `yjeXPEBxd5EACsDz4xPWx` → 3 rows:
  dwell_s ≈ [54.046 (nav=False), 33.460 (nav=True), 73.887 (nav=True)]
Trajectory id `8uJRwmgg3iwq9-rnZ_MpJ` → 6 rows:
  dwell_s ≈ [6.683 (True), 5.716 (True), 0.830 (False), 32.787 (True),
             19.290 (True), 20.836 (True)]
(±0.01s tolerance. These come from real createdTime deltas and must not regress.)