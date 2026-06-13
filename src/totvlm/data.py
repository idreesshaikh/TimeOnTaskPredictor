"""
data.py - the whole data pipeline, streaming, minimal.

Hugging Face streams the parquet shards for you (load_dataset(streaming=True)),
so the ~837 GB never lands on disk and there is nothing to download or delete.

The one thing we must get right is the row shape. Run this once to see it:

    from datasets import load_dataset
    r = next(iter(load_dataset("webagentlab/WebChain", split="train", streaming=True)))
    for k, v in r.items():
        print(k, type(v).__name__, len(v) if isinstance(v, (list, str, bytes)) else "")

  * If you see SCALAR fields (one screenshot, one timestamp per row) -> a row is
    one STEP. Leave ROW_IS_TRAJECTORY = False (we group consecutive steps).
  * If you see a "steps" list of dicts, or screenshot/timestamp as LISTS -> a row
    is a whole TRAJECTORY. Set ROW_IS_TRAJECTORY = True.

If your loader yields nothing, you have the flag backwards. Then fix COLS to the
real names you saw above.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

from totvlm.scoring import tasksense_difficulty

REPO = "webagentlab/WebChain"  # the id shown at the top of your access page
ROW_IS_TRAJECTORY = False  # flip to True if a row holds a whole trajectory
MAX_SIDE, MAX_HISTORY = 1024, 3

COLS = {  # logical name -> real column name (verify with the snippet above)
    "traj": "trajectory_id",
    "goal": "task",
    "steps": "steps",  # row-level
    "shot": "screenshot",
    "ts": "timestamp",
    "ax": "ax_tree",
    "action": "action_type",  # step-level
}


# --- tiny helpers ----------------------------------------------------------
def _img(raw):
    if isinstance(raw, dict):  # HF Image feature -> take the bytes
        raw = raw.get("bytes")
    if not raw:
        return None
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def _secs(ts):
    if ts is None:
        return None
    ts = float(ts)
    return ts / 1000 if ts >= 1e11 else ts  # epoch milliseconds -> seconds


def _n_interactive(ax):
    if isinstance(ax, str):
        try:
            ax = json.loads(ax)
        except Exception:
            return 0
    roles = {
        "button",
        "link",
        "textbox",
        "checkbox",
        "radio",
        "combobox",
        "menuitem",
        "tab",
        "option",
    }
    n, stack = 0, [ax]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if str(node.get("role", node.get("type", ""))).lower() in roles:
                n += 1
            stack.extend(node.get("children", []) or [])
        elif isinstance(node, list):
            stack.extend(node)
    return n


def _val_holdout(traj_id, frac=0.10):
    """Carve a val set out of train, by trajectory id (deterministic, no leakage)."""
    f = int(hashlib.sha256(str(traj_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return f < frac


def _shuffled(it, n, seed=0):
    rng, buf = random.Random(seed), []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf.pop(rng.randrange(len(buf)))
    rng.shuffle(buf)
    yield from buf


# --- normalise either layout into (steps, goal) trajectories ---------------
def _steps_from_row(row):
    """Pull a list of step-dicts out of a TRAJECTORY row."""
    s = row.get(COLS["steps"])
    if isinstance(s, list) and s and isinstance(s[0], dict):
        return s  # already a list of step dicts
    keys = ["shot", "ts", "ax", "action"]  # fallback: parallel lists
    cols = {k: row.get(COLS[k]) for k in keys}
    n = max((len(v) for v in cols.values() if isinstance(v, list)), default=0)
    return [
        {
            COLS[k]: (
                cols[k][i] if isinstance(cols[k], list) and i < len(cols[k]) else None
            )
            for k in keys
        }
        for i in range(n)
    ]


def _trajectories(stream, keep):
    """Yield (steps, goal) where steps is a list of step-dicts for one trajectory."""
    if ROW_IS_TRAJECTORY:
        for row in stream:
            if keep(row.get(COLS["traj"], "")):
                yield _steps_from_row(row), (row.get(COLS["goal"]) or "")
    else:
        buf, cur, goal = [], None, ""
        for row in stream:  # here, a row IS one step
            tid = row.get(COLS["traj"])
            if tid != cur:
                if buf and keep(cur):
                    yield buf, goal
                buf, cur = [], tid
            goal = row.get(COLS["goal"]) or goal
            buf.append(row)
        if buf and keep(cur):
            yield buf, goal


# --- one trajectory -> per-screen training examples ------------------------
def _delta(a: float | None, b: float | None) -> float | None:
    """Return a - b only when both are floats and a >= b, else None.
    A plain function lets Pylance see the return type clearly; a ternary
    inside a list comprehension does not narrow `float | None` correctly."""
    if a is not None and b is not None and a >= b:
        return a - b
    return None


def _examples(steps, goal, use_history):
    if len(steps) < 2:
        return
    steps = (
        sorted(steps, key=lambda s: s.get("step_id", 0))
        if "step_id" in (steps[0] or {})
        else steps
    )
    ts = [_secs(s.get(COLS["ts"])) for s in steps]
    dwell: list[float | None] = [None] + [
        _delta(ts[i], ts[i - 1]) for i in range(1, len(steps))
    ]
    logs = [math.log1p(d) for d in dwell if d is not None]
    if not logs:
        return
    mean = sum(logs) / len(logs)
    std = max((sum((x - mean) ** 2 for x in logs) / len(logs)) ** 0.5, 1e-3)

    history = []
    for i, s in enumerate(steps):
        img = _img(s.get(COLS["shot"]))
        d = dwell[i]  # local variable so Pylance narrows float | None -> float
        if img is not None and d is not None:
            yield {
                "image": img,
                "history": history[-MAX_HISTORY:] if use_history else [],
                "goal": goal,
                "target": (math.log1p(d) - mean)
                / std,  # per-trajectory z-scored log-dwell
                "dwell_seconds": d,
                "tasksense": tasksense_difficulty(
                    s.get(COLS["action"]), _n_interactive(s.get(COLS["ax"]))
                ).score,
            }
        if img is not None:
            history.append(img)


# --- dataset + dataloader --------------------------------------------------
class WebChainToT(IterableDataset):
    def __init__(self, split="train", use_history=True, shuffle_buffer=256):
        self.split, self.use_history, self.shuffle_buffer = (
            split,
            use_history,
            shuffle_buffer,
        )

    def _keep(self, traj_id):
        if self.split == "test":  # the real 2k test split: take all
            return True
        v = _val_holdout(traj_id)
        return v if self.split == "val" else not v

    def _raw(self):
        hf_split = "test" if self.split == "test" else "train"
        stream = load_dataset(REPO, split=hf_split, streaming=True)
        for steps, goal in _trajectories(stream, self._keep):
            yield from _examples(steps, goal, self.use_history)

    def __iter__(self):
        return (
            _shuffled(self._raw(), self.shuffle_buffer)
            if self.shuffle_buffer
            else self._raw()
        )


def _messages(ex):
    content = []
    for j, h in enumerate(ex["history"]):
        content += [
            {"type": "text", "text": f"Earlier screen {j + 1}:"},
            {"type": "image", "image": h},
        ]
    content += [
        {"type": "text", "text": "Current screen:"},
        {"type": "image", "image": ex["image"]},
        {
            "type": "text",
            "text": f"Task: {ex['goal']}\n"
            "Estimate how long the user pauses on this screen.",
        },
    ]
    return [{"role": "user", "content": content}]


def get_dataloader(
    processor, split="train", use_history=True, batch_size=2, num_workers=0
):
    """Ready-to-train loader. Keep num_workers=0 with streaming; \
        the GPU is the bottleneck."""

    def collate(batch):
        x = processor.apply_chat_template(
            [_messages(e) for e in batch],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        x.pop("token_type_ids", None)
        x["target"] = torch.tensor([e["target"] for e in batch], dtype=torch.float32)
        x["tasksense"] = torch.tensor(
            [e["tasksense"] for e in batch], dtype=torch.float32
        )
        return x

    return DataLoader(
        WebChainToT(split, use_history),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
    )
