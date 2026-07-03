# Screenshot → Per-Screen Time-on-Task

Predict how long a user dwells on a web UI screen before acting, **from the
screenshot alone**. Research question: how much of per-screen Time-on-Task is
recoverable from the screen itself, independent of the user's goal or history?

Two models are compared on domain-disjoint splits of the
[WebChain](https://huggingface.co/datasets/webagentlab/WebChain) trajectory
corpus, with a zero-shot external validation at the end:

| model | input | role |
|---|---|---|
| LightGBM | interpretable AX-tree / geometry features (no image) | the bar to beat |
| Qwen3-VL-4B (4-bit QLoRA SFT) | screenshot only | the research model |

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
uv run python scripts/build_dataset.py --resolve-images
uv run python scripts/build_dataset.py --splits

# 2. No-image baseline (LightGBM; fetches AX-tree features, resumable cache)
uv run python scripts/train_baseline.py

# 3. VLM fine-tuning (CUDA GPU) — always smoke-test first
uv run python -m totvlm.train --config configs/vlm.yaml --dry-run
uv run python -m totvlm.train --config configs/vlm.yaml

# 4. Held-out TEST evaluation: VLM vs baseline vs floors
uv run python scripts/evaluate.py

# 5. External validation (READ-ONLY set, evaluated exactly once, zero-shot)
uv run python scripts/prepare_external.py
uv run python scripts/validate_external.py
```

Reports land in `artifacts/`: `dataset_card.md`, `baseline_report.md`,
`vlm_train_card.md`, `eval_report.md`, `external_report.md`.

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
