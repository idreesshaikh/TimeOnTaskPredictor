# Time-on-Task Predictor

Predict how long a user dwells on a web UI screen before acting — and, by
summing per-screen predictions within a trajectory, **how long the whole task
takes**. Research question: how accurately can per-screen (and, by
aggregation, whole-task) Time-on-Task be predicted from rendered UI screens,
and how much of that predictability resides in the screen alone versus the
user's stated task?

The contestants are compared on identical, domain-disjoint rows of the
[WebChain](https://huggingface.co/datasets/webagentlab/WebChain) trajectory
corpus, with a zero-shot external validation at the end:

| model | input | role |
|---|---|---|
| LightGBM | interpretable AX-tree / geometry features (no image) | the bar to beat |
| Qwen3-VL-4B (4-bit QLoRA SFT) | screenshot only | the science core: what the screen alone predicts |
| Qwen3-VL-4B (4-bit QLoRA SFT) | screenshot + task title | the predictor; its gap to screen-only = the goal-driven share |

## Layout

```
configs/          every tunable number (data / baseline / vlm / eval / external)
src/totvlm/
  data.py         raw-JSON schema + audit · VLM chat examples · output parsers
  labels.py       per-screen dwell labels (the load-bearing spec in SPEC.md)
  images.py       screenshot resolution/cache + model-ready loader
  splits.py       domain-disjoint splits · train-only winsor cap · dataset card
  features.py     AX-tree features for the baseline (feature-only cache)
  scoring.py      shared metrics: MAE/RMSE (log & s), Spearman, calibration
  model.py        VLM load/predict (Unsloth) + LightGBM baseline
  train.py        QLoRA SFT entrypoint (Path A: emit `dwell_seconds: X.X`)
scripts/          pipeline entrypoints (below)
tests/            unit tests incl. label oracles + external-set leak guard
artifacts/        stage cards, reports, splits.json, model files
```

## Setup

```bash
uv sync                  # core (CPU-friendly: data pipeline, baseline, reports)
uv sync --extra vlm      # + Unsloth training stack (Linux + CUDA GPU)
uv run hf login          # WebChain is gated — request access first
```

## Pipeline

Every stage is idempotent, seeded (42), and writes a stats card to
`artifacts/`. Splits are by registrable domain (eTLD+1) — train/val/test
never share a website.

```bash
# 0. Raw data
bash scripts/download_minimal.sh

# 1. Dataset: audit → labels → screenshots → splits + winsorized target
uv run python scripts/build_dataset.py <json_dir> --audit
uv run python scripts/build_dataset.py <json_dir> --labels
uv run python scripts/build_dataset.py --resolve-images   # trajectory-budgeted:
#   whole tasks per split (configs/data.yaml resolve:), seeded prefix-stable,
#   resumable, JPEG-recompressed (~1 GB, not the full ~122 GB corpus)
uv run python scripts/build_dataset.py --splits

# 2. No-image baseline (LightGBM; fetches AX-tree features, resumable cache)
uv run python scripts/train_baseline.py

# 3. VLM fine-tuning (CUDA GPU) — always smoke-test first; two conditions
uv run python -m totvlm.train --config configs/vlm.yaml --dry-run
uv run python -m totvlm.train --config configs/vlm.yaml        # screen-only
uv run python -m totvlm.train --config configs/vlm_task.yaml   # screen+task

# 4. Held-out TEST evaluation: floors vs LightGBM vs both VLM conditions,
#    per-screen AND task-level (per-trajectory sums) + the paper figure set
uv run python scripts/evaluate.py
uv run python scripts/make_figures.py    # CPU-only, regenerable anytime

# 5. External validation (READ-ONLY set, evaluated exactly once, zero-shot)
#    + AIM theory-grounding analysis over the cached predictions
uv run python scripts/prepare_external.py
uv run python scripts/validate_external.py
uv run python scripts/analyze_aim.py
```

Reports land in `artifacts/`: `dataset_card.md`, `baseline_report.md`,
`vlm_train_card.md`, `eval_report.md`, `external_report.md`,
`aim_analysis.md`, and the figure set in `artifacts/figures/`.

## Running on a SLURM cluster — two jobs

```bash
# one-time setup on the login node:
uv run hf auth login           # WebChain is gated
wandb login                    # or export WANDB_MODE=offline + `wandb sync`

mkdir -p logs
sbatch scripts/setup.sbatch    # CPU: raw download → dataset → baseline → external prep
# ...when it finishes:
sbatch scripts/train.sbatch    # GPU: both VLM conditions → eval + figures → external
```

Every stage is idempotent and fetching / training are resumable (training
auto-resumes from the last epoch checkpoint) — **if a job hits its time
limit, just resubmit it** and it continues where it stopped. `train.sbatch`
fails fast with a clear message if the setup outputs are missing.

All stages log to the same wandb project (`tot-vlm`): `baseline-lgbm`,
`qwen3vl4b-qlora-pathA`, `qwen3vl4b-qlora-pathA-task`, `eval-test`.

## Reproducibility contract (see SPEC.md for the full spec)

- Global seed **42** everywhere; splits saved to `artifacts/splits.json` and
  reused, never recomputed.
- Winsor cap = p95 of raw dwell on the **train split only**; target
  `y = log1p(dwell_s)`; metrics reported in both log-space and seconds.
- The external set under `data/external/` is read-only, never touches
  training/tuning, and is evaluated exactly once
  (enforced by `tests/test_external_guard.py` and the evaluate-once guard).
- All tunables live in `configs/*.yaml`; label-spec oracles are pinned in
  `tests/test_labels.py`.

## Tests

```bash
uv run pytest
```
