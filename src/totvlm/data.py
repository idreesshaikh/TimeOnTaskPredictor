"""
data.py — stream WebChain, group steps into trajectories, yield training examples.

Each row in WebChain is ONE STEP of a trajectory. HF streams the parquet shards
so nothing is downloaded to disk. We buffer consecutive rows with the same
trajectory_id, then compute dwell times and yield one example per screen.

Before training, confirm the column names match your data:

    uv run python -c "
    from datasets import load_dataset
    r = next(iter(load_dataset('webagentlab/WebChain', split='train', streaming=True)))
    for k, v in r.items():
        print(k, type(v).__name__)
    "

Then update COLS below to match.
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

REPO = "webagentlab/WebChain"
MAX_SIDE = 1024
MAX_HISTORY = 3

# Update these to match the real column names from the inspection above.
COLS = {
    "traj": "trajectory_id",
    "step": "step_id",
    "goal": "task",
    "shot": "screenshot",
    "ax": "ax_tree",
    "ts": "timestamp",
    "action": "action_type",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _img(raw) -> Image.Image | None:
    if isinstance(raw, dict):  # HF Image feature -> extract bytes
        raw = raw.get("bytes")
    if not raw:
        return None
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def _secs(ts) -> float | None:
    if ts is None:
        return None
    ts = float(ts)
    return ts / 1000 if ts >= 1e11 else ts  # ms -> s if needed


def _delta(a: float | None, b: float | None) -> float | None:
    if a is not None and b is not None and a >= b:
        return a - b
    return None


def _n_interactive(ax) -> int:
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


def _is_val(traj_id: str, frac: float = 0.10) -> bool:
    """Deterministic 10 % val holdout carved from train, by trajectory."""
    h = int(hashlib.sha256(traj_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def _shuffle(it, n: int, seed: int = 0):
    rng, buf = random.Random(seed), []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf.pop(rng.randrange(len(buf)))
    rng.shuffle(buf)
    yield from buf


# ---------------------------------------------------------------------------
# Trajectory grouping + example generation
# ---------------------------------------------------------------------------


def _group(stream, keep):
    """Yield (steps, goal) by buffering rows that share a trajectory_id."""
    buf: list[dict] = []
    cur: str | None = None
    goal: str = ""
    for row in stream:
        tid = str(row.get(COLS["traj"], ""))
        if tid != cur:
            if buf and cur is not None and keep(cur):
                yield buf, goal
            buf, cur = [], tid
        goal = str(row.get(COLS["goal"]) or goal)
        buf.append(row)
    if buf and cur is not None and keep(cur):
        yield buf, goal


def _examples(steps: list[dict], goal: str, use_history: bool):
    """One trajectory -> per-screen training dicts."""
    if len(steps) < 2:
        return
    steps = sorted(steps, key=lambda s: int(s.get(COLS["step"]) or 0))
    ts = [_secs(s.get(COLS["ts"])) for s in steps]
    dwell: list[float | None] = [None] + [
        _delta(ts[i], ts[i - 1]) for i in range(1, len(steps))
    ]
    logs = [math.log1p(d) for d in dwell if d is not None]
    if not logs:
        return
    mean = sum(logs) / len(logs)
    std = max((sum((x - mean) ** 2 for x in logs) / len(logs)) ** 0.5, 1e-3)

    history: list[Image.Image] = []
    for i, s in enumerate(steps):
        img = _img(s.get(COLS["shot"]))
        d = dwell[i]
        if img is not None and d is not None:
            yield {
                "image": img,
                "history": history[-MAX_HISTORY:] if use_history else [],
                "goal": goal,
                "target": (math.log1p(d) - mean) / std,
                "dwell_seconds": d,
                "tasksense": tasksense_difficulty(
                    s.get(COLS["action"]),
                    _n_interactive(s.get(COLS["ax"])),
                ).score,
            }
        if img is not None:
            history.append(img)


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------


class WebChainToT(IterableDataset):
    def __init__(
        self, split: str = "train", use_history: bool = True, shuffle_buffer: int = 256
    ):
        self.split = split
        self.use_history = use_history
        self.shuffle_buffer = shuffle_buffer

    def _keep(self, traj_id: str) -> bool:
        if self.split == "test":
            return True
        val = _is_val(traj_id)
        return val if self.split == "val" else not val

    def _raw(self):
        hf_split = "test" if self.split == "test" else "train"
        stream = load_dataset(REPO, split=hf_split, streaming=True)
        for steps, goal in _group(stream, self._keep):
            yield from _examples(steps, goal, self.use_history)

    def __iter__(self):
        return _shuffle(self._raw(), self.shuffle_buffer)


def _build_messages(ex: dict) -> list[dict]:
    content: list[dict] = []
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
    processor,
    split: str = "train",
    use_history: bool = True,
    batch_size: int = 2,
    num_workers: int = 0,
) -> DataLoader:
    def collate(batch: list[dict]) -> dict:
        x = processor.apply_chat_template(
            [_build_messages(e) for e in batch],
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
