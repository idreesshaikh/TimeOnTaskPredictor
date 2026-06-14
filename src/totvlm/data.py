"""
totvlm/data.py
==============
Data loading for Time-on-Task prediction.

Loads from pre-built parquet files (output of build_parquets.py).
Each row already has difficulty_score, tasksense_ms, and n_elements
pre-computed — no AX tree fetching or z-score math at training time.

Dataset directory structure expected:
    webchain_dataset/
    ├── train/  chunk_00000.parquet  chunk_00001.parquet ...
    ├── val/    chunk_00000.parquet ...
    └── test/   chunk_00000.parquet ...

At training:  image + tasksense_ms + step_frac → predict difficulty_score
At inference: image only → predict difficulty_score
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

DATASET_DIR = "webchain_dataset"   # ← set to your build_parquets.py output dir
MAX_SIDE    = 512    # OOM fix: 1024->512 cuts visual tokens by ~4x
MAX_HISTORY = 3


# ── Image helper ──────────────────────────────────────────────────────────────
def _to_pil(raw) -> Image.Image | None:
    """Convert raw bytes (or HF Image dict) to a PIL image."""
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    if not raw:
        return None
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


# ── Shuffle buffer ────────────────────────────────────────────────────────────
def _shuffle(it, n: int, seed: int = 42):
    rng, buf = random.Random(seed), []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf.pop(rng.randrange(len(buf)))
    rng.shuffle(buf)
    yield from buf


# ── Dataset ───────────────────────────────────────────────────────────────────
class WebChainToT(IterableDataset):
    """
    Streams examples from pre-built parquet files.

    Each example:
        image          : PIL image (current screen)
        history        : list of up to MAX_HISTORY previous PIL images
        goal           : task instruction string
        target         : difficulty_score (z-scored log dwell) ← regression target
        tasksense      : tasksense_ms (structural difficulty, auxiliary signal)
        step_frac      : 0.0→1.0 position in task (0=first step, 1=last)

    Args:
        split          : "train", "val", or "test"
        dataset_dir    : path to build_parquets.py output directory
        use_history    : include previous screenshots (set False for ablation)
        shuffle_buffer : rows buffered for random shuffling (train only)
        max_chunks     : limit chunks loaded (None = all; set small for debugging)
    """

    def __init__(
        self,
        split:          str  = "train",
        dataset_dir:    str  = DATASET_DIR,
        use_history:    bool = True,
        shuffle_buffer: int  = 512,
        max_chunks:     int | None = None,
    ):
        self.split          = split
        self.dataset_dir    = Path(dataset_dir)
        self.use_history    = use_history
        self.shuffle_buffer = shuffle_buffer
        self.max_chunks     = max_chunks

        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Run build_parquets.py first to generate the dataset."
            )
        self._chunks = sorted(split_dir.glob("chunk_*.parquet"))
        if not self._chunks:
            raise FileNotFoundError(f"No parquet chunks found in {split_dir}")
        if max_chunks:
            self._chunks = self._chunks[:max_chunks]

    def _iter_examples(self):
        """
        Stream examples chunk by chunk.
        Within each chunk, group rows by trajectory_id and sort by step_idx
        so history is built in the correct order.
        History resets at trajectory boundaries and between chunks
        (trajectories spanning chunks lose their cross-chunk history — acceptable).
        """
        for chunk_path in self._chunks:
            df = pd.read_parquet(chunk_path)

            # Group by trajectory, preserving step order
            for traj_id, group in df.groupby("trajectory_id", sort=False):
                group   = group.sort_values("step_idx")
                history = []   # PIL images of previous steps in this trajectory

                for _, row in group.iterrows():
                    img = _to_pil(row.get("image"))
                    if img is None:
                        continue

                    yield {
                        "image":      img,
                        "history":    history[-MAX_HISTORY:] if self.use_history else [],
                        "goal":       str(row.get("instruction", "")),
                        "target":     float(row["difficulty_score"]),
                        "tasksense":  float(row.get("tasksense_ms", 0.0)),
                        "step_frac":  float(row.get("step_frac", 0.0)),
                        "n_elements": int(row.get("n_elements", 0)),
                        # kept for logging/analysis, not fed to model
                        "dwell_seconds": float(row.get("dwell_seconds", 0.0)),
                        "traj_mu":       float(row.get("traj_mu", 0.0)),
                        "traj_sigma":    float(row.get("traj_sigma", 1.0)),
                    }

                    history.append(img)

    def __iter__(self):
        it = self._iter_examples()
        if self.split == "train":
            it = _shuffle(it, self.shuffle_buffer)
        return iter(it)


# ── Message builder ───────────────────────────────────────────────────────────
def _build_messages(ex: dict) -> list[dict]:
    """
    Build the chat message for one example.
    History screens shown first, current screen last.
    """
    content = []

    for j, h_img in enumerate(ex["history"]):
        content += [
            {"type": "text",  "text": f"Earlier screen {j + 1}:"},
            {"type": "image", "image": h_img},
        ]

    content += [
        {"type": "text",  "text": "Current screen:"},
        {"type": "image", "image": ex["image"]},
        {
            "type": "text",
            "text": (
                f"Task: {ex['goal']}\n"
                "Estimate how long the user pauses on this screen."
            ),
        },
    ]

    return [{"role": "user", "content": content}]


# ── Collate ───────────────────────────────────────────────────────────────────
def _collate(processor):
    """Returns a collate_fn that uses the given processor."""
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

        # Regression target
        x["target"] = torch.tensor(
            [e["target"] for e in batch], dtype=torch.float32
        )
        # Auxiliary TaskSense signal (used in loss or as extra input feature)
        x["tasksense"] = torch.tensor(
            [e["tasksense"] for e in batch], dtype=torch.float32
        )
        # Task progress (0→1)
        x["step_frac"] = torch.tensor(
            [e["step_frac"] for e in batch], dtype=torch.float32
        )
        return x
    return collate


# ── Public API ────────────────────────────────────────────────────────────────
def get_dataloader(
    processor,
    split:       str  = "train",
    dataset_dir: str  = DATASET_DIR,
    use_history: bool = True,
    batch_size:  int  = 2,
    num_workers: int  = 4,
    max_chunks:  int | None = None,
) -> DataLoader:
    """
    Build a DataLoader for one split.

    Args:
        processor   : HuggingFace processor for the VLM
        split       : "train", "val", or "test"
        dataset_dir : path to build_parquets.py output
        use_history : include previous screenshots (False = ablation baseline)
        batch_size  : examples per batch
        num_workers : DataLoader worker processes. >0 runs image decode +
                      processor tokenization in parallel subprocesses so the
                      GPU never waits on CPU data prep. Set to ~CPU core count
                      (4-8 is usually plenty). 0 = main process (slow).
        max_chunks  : limit chunks for quick smoke tests
    """
    ds = WebChainToT(
        split=split,
        dataset_dir=dataset_dir,
        use_history=use_history,
        shuffle_buffer=512 if split == "train" else 1,
        max_chunks=max_chunks,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate(processor),
        pin_memory=True,                                 # faster CPU→GPU copy
        prefetch_factor=2 if num_workers > 0 else None,  # stage batches ahead
        persistent_workers=num_workers > 0,              # don't respawn each epoch
    )
