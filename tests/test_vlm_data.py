"""
CPU-only tests for the Path-A VLM chat-example builder + output parser
(totvlm.data). No GPU / unsloth needed — this is the part of the SFT
pipeline that must be correct BEFORE burning GPU time.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest
from PIL import Image

import totvlm.data
from totvlm.data import (
    SYSTEM_PROMPT,
    build_vlm_examples,
    format_dwell_target,
    parse_dwell_output,
)

CAP = 24.954   # train-split p95 winsor cap (artifacts/dataset_card.md)


# ── format / parse ────────────────────────────────────────────────────────────

def test_format_clips_to_winsor_range():
    assert format_dwell_target(7.649, CAP) == "dwell_seconds: 7.6"
    assert format_dwell_target(7.65, CAP) == "dwell_seconds: 7.7"   # 1 dp
    assert format_dwell_target(600.0, CAP) == "dwell_seconds: 25.0"  # cap
    assert format_dwell_target(-1.0, CAP) == "dwell_seconds: 0.0"    # floor


# (parser edge cases live in tests/test_parse.py)

def test_format_parse_roundtrip():
    for v in (0.1, 4.7, 24.954):
        assert parse_dwell_output(format_dwell_target(v, CAP)) == round(v, 1)


# ── example builder ───────────────────────────────────────────────────────────

@pytest.fixture
def rows(tmp_path):
    """Two resolved rows + one unresolved; screenshots larger than the
    pixel budget so downscaling is observable."""
    paths = []
    for i in range(2):
        p = tmp_path / f"shot_{i}.png"
        Image.new("RGB", (1598, 972), (10 * i, 20, 30)).save(p)
        paths.append(str(p))
    return pd.DataFrame({
        "img_path": paths + [None],
        "img_resolved": [True, True, False],
        "task_title": ["Find a flight", "Book a room", "ignored"],
        "dwell_s": [7.697, 30.0, 1.0],   # 30.0 > cap → clipped in target
        "is_navigation": [False, True, False],
    })


BUILD_KW = dict(winsor_cap=CAP, max_side=1024, min_pixels=100352,
                max_pixels=602112)


def test_builder_uses_resolved_rows_only(rows):
    examples = build_vlm_examples(rows, **BUILD_KW)
    assert len(examples) == 2      # unresolved row dropped
    # sequence-like plain Python (never datasets.map), lazy not a list
    assert isinstance(examples, Sequence)


def test_builder_is_lazy(rows, monkeypatch):
    """Images must load on ACCESS, not at build time — eagerly decoding the
    full train split (~90k screenshots) OOM-kills the training job."""
    calls = []
    real = totvlm.data.load_image

    def counting(*a, **kw):
        calls.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(totvlm.data, "load_image", counting)
    examples = build_vlm_examples(rows, **BUILD_KW)
    assert calls == []                     # nothing decoded at build time
    _ = examples[0]
    assert len(calls) == 1                 # one image per accessed example
    chunk = examples[0:2]                  # slice access (predict_dwell_batch)
    assert len(chunk) == 2 and len(calls) == 3


def test_builder_message_structure(rows):
    ex = build_vlm_examples(rows, **BUILD_KW)[0]
    msgs = ex["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == [{"type": "text", "text": SYSTEM_PROMPT}]

    # screen-only by default: image, no text (the RQ)
    user = msgs[1]["content"]
    assert [c["type"] for c in user] == ["image"]
    img = user[0]["image"]
    assert isinstance(img, Image.Image)
    assert img.width * img.height <= 602112   # pixel budget respected
    assert max(img.size) <= 1024

    assert msgs[2]["content"] == [
        {"type": "text", "text": "dwell_seconds: 7.7"}
    ]
    assert ex["dwell_s"] == pytest.approx(7.697)
    assert ex["is_navigation"] is False


def test_builder_clips_target_to_cap(rows):
    ex = build_vlm_examples(rows, **BUILD_KW)[1]
    assert ex["messages"][2]["content"][0]["text"] == "dwell_seconds: 25.0"


def test_include_task_title_ablation(rows):
    ex = build_vlm_examples(rows, include_task_title=True, **BUILD_KW)[0]
    user = ex["messages"][1]["content"]
    assert [c["type"] for c in user] == ["image", "text"]
    assert user[1]["text"] == "Task: Find a flight"
