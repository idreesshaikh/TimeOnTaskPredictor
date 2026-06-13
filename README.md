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
uv run huggingface-cli login    # WebChain is gated — request access first
```

> If torch installs CPU-only, re-sync with your CUDA index:
> `uv sync --index https://download.pytorch.org/whl/cu121`

## Step 1 — confirm column names (run once)

```bash
uv run python -c "
from datasets import load_dataset
r = next(iter(load_dataset('webagentlab/WebChain', split='train', streaming=True)))
for k, v in r.items():
    print(k, type(v).__name__)
"
```

Update `COLS` in `data.py` to match what you see.

## Step 2 — train

```bash
uv run python -m totvlm.train
```

Set `USE_HISTORY = False` in `train.py` for the single-screen baseline (RQ2 ablation).
