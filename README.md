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

| model | input at inference | role |
|---|---|---|
| LightGBM | interpretable AX-tree / geometry features (no image) | the bar to beat |
| Qwen3-VL-4B (4-bit QLoRA SFT) | screenshot | **image + features**: the science core |
| Qwen3-VL-4B (4-bit QLoRA SFT) | screenshot + task title | **image + features + task**: the predictor; its gap to the first = the goal-driven share |

Neither VLM is a pixels-only model. In **both**, at *training time only* the
targets are blended toward a LightGBM teacher that saw privileged metadata the
screenshot can't show — AX-tree structure, the navigation flag, step index,
click area (learning under privileged information / generalized distillation).
Those features shape what the model learns but never appear at inference, so
the deployed predictor stays screenshot-only while not being handicapped
against the metadata-rich LightGBM baseline. The two conditions differ only by
whether the task title is in the prompt, expressed as config overlays:
`configs/vlm.yaml` is the base (image+features) and `configs/vlm_task.yaml`
inherits it and adds the task title.

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
  lupi.py         privileged-info teacher (OOF LightGBM) + λ target blend
  config.py       YAML loader + `base:` overlays (vlm_task inherits vlm)
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
uv run python scripts/build_dataset.py --resolve-images
uv run python scripts/build_dataset.py --splits

# 2. No-image baseline (LightGBM; fetches AX-tree features, resumable cache)
uv run python scripts/train_baseline.py

# 2b. Privileged-feature teacher: out-of-fold LightGBM on the privileged
#     features over the VLM's train rows (domain-grouped folds) → the soft
#     targets BOTH conditions blend in at training
uv run python scripts/make_lupi_teacher.py

# 3. VLM fine-tuning (CUDA GPU) — always smoke-test first; two conditions,
#    each blending the teacher into its train targets (base + one overlay)
uv run python -m totvlm.train --config configs/vlm.yaml --dry-run
uv run python -m totvlm.train --config configs/vlm.yaml        # image+features
uv run python -m totvlm.train --config configs/vlm_task.yaml   # image+features+task

# 3b. Play with the trained model on any screenshot(s) you like (GPU):
uv run python scripts/predict.py my_screen.png
uv run python scripts/predict.py seat_v1.png seat_v2.png --task "Pick a seat"

# 4. Held-out TEST evaluation: floors vs LightGBM vs the two VLM conditions,
#    per-screen AND task-level (per-trajectory sums) + the paper figure set
uv run python scripts/evaluate.py
uv run python scripts/make_figures.py    # CPU-only, regenerable anytime

# 5. External validation (READ-ONLY set, evaluated exactly once, zero-shot)
#    of the frozen image+features model + AIM theory-grounding analysis
uv run python scripts/prepare_external.py
uv run python scripts/validate_external.py
uv run python scripts/analyze_aim.py
```

Reports land in `artifacts/`: `dataset_card.md`, `baseline_report.md`,
`lupi_teacher_card.md`, `vlm_train_card.md` (one per condition),
`eval_report.md`, `external_report.md`, `aim_analysis.md`, and the figure set
in `artifacts/figures/`.

## Running on a SLURM cluster — two jobs

```bash
# one-time setup on the login node:
uv run hf auth login           # WebChain is gated
wandb login                    # or export WANDB_MODE=offline + `wandb sync`

mkdir -p logs
sbatch scripts/setup.sbatch    # CPU: raw download → dataset → baseline → LUPI teacher → external prep
# ...when it finishes:
sbatch scripts/train.sbatch    # GPU: the two VLM conditions → eval + figures → external
```

Every stage is idempotent and fetching / training are resumable (training
auto-resumes from the last epoch checkpoint) — **if a job hits its time
limit, just resubmit it** and it continues where it stopped. `train.sbatch`
fails fast with a clear message if the setup outputs are missing.

Training stages log to the same wandb project (`tot-vlm`): `baseline-lgbm`,
`qwen3vl4b-qlora-pathA` (image+features), and `qwen3vl4b-qlora-pathA-task`
(image+features+task). Evaluation and figures write only to `artifacts/`
(rerunnable offline).

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
