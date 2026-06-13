# Screenshot-only Time-on-Task (ToT) with Qwen3-VL

Predict how long a user pauses on a web screen, from the screenshot alone.

## Layout
```
src/totvlm/
  data.py      streaming WebChain -> training batches (HF does the streaming)
  scoring.py   TaskSense structural difficulty (train-only signal)
  model.py     Qwen3-VL + LoRA + a regression head
  train.py     minimal fine-tuning loop
```

## Setup (uv)
```bash
uv sync                          # creates .venv and installs everything in pyproject
uv run huggingface-cli login     # WebChain is gated; request access first
```
`uv sync` installs the project editable, so `totvlm` is importable and you can run
modules from the project root with `uv run`.

> GPU torch: if `uv sync` pulls a CPU-only torch, add the CUDA wheel index, e.g.
> `uv sync --index https://download.pytorch.org/whl/cu124` (match your CUDA version).

## Step 1 - confirm the row shape (5 seconds, do this first)
```bash
uv run python - <<'PY'
from datasets import load_dataset
r = next(iter(load_dataset("webagentlab/WebChain", split="train", streaming=True)))
for k, v in r.items():
    print(k, type(v).__name__, len(v) if isinstance(v, (list, str, bytes)) else "")
PY
```
- scalar screenshot/timestamp -> a row is one STEP -> leave `ROW_IS_TRAJECTORY = False`
- a `steps` list, or list-valued screenshot/timestamp -> a row is a TRAJECTORY -> set `ROW_IS_TRAJECTORY = True`

Then make `COLS` in `data.py` match the names you just printed. If the loader yields
nothing, the flag is backwards.

## Step 2 - fine-tune
```bash
uv run python -m totvlm.train
```
`USE_HISTORY = False` in train.py gives the single-screen baseline (RQ2 ablation).
