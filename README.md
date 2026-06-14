# Screenshot-only Time-on-Task · Qwen3-VL

Predict how long a user pauses on a web screen, from the screenshot alone.

```
src/totvlm/
  data.py      stream WebChain → training batches
  scoring.py   TaskSense structural difficulty (train-only signal)
  model.py     Qwen3-VL + LoRA + regression head
  train.py     fine-tuning loop
```

## Setup

```bash
uv sync
uv run hf login    # WebChain is gated — request access first
```

> If torch installs CPU-only, re-sync with your CUDA index:
> `uv sync --index https://download.pytorch.org/whl/cu121`

## Step 1 — train

```bash
uv run python -m totvlm.train
```

Set `USE_HISTORY = False` in `train.py` for the single-screen baseline (RQ2 ablation).


uv run python scripts/build_parquets.py \
    ./webchain_raw/raw/json/all_json_files \
    ./webchain_dataset \
    --workers 12 \
    --chunk 500