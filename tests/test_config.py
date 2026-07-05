"""Config overlays (totvlm.config): deep-merge semantics + the two real
training conditions — vlm_task.yaml must inherit the base and differ ONLY by
the task-title flag, or the "identical except one thing" science quietly
breaks. Both conditions must carry the privileged-feature `lupi:` block."""
from __future__ import annotations

from totvlm.config import _deep_merge, load_config


def test_deep_merge_recurses_and_child_wins():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "list": [1, 2]}
    override = {"a": 9, "nested": {"y": 20, "z": 3}, "list": [9]}
    out = _deep_merge(base, override)
    assert out == {"a": 9, "nested": {"x": 1, "y": 20, "z": 3}, "list": [9]}
    assert base["nested"] == {"x": 1, "y": 2}   # inputs untouched


def test_two_conditions_differ_only_by_task_flag():
    base = load_config("configs/vlm.yaml")          # image + features
    task = load_config("configs/vlm_task.yaml")     # image + features + task

    # both are trained with the privileged-feature blend (features at train
    # time) — the whole point: neither is a pixels-only model
    assert "lupi" in base and "lupi" in task
    assert base["lupi"] == task["lupi"]

    # the ONLY prompt difference is the task title
    assert base["data"]["include_task_title"] is False
    assert task["data"]["include_task_title"] is True

    # shared knobs are inherited verbatim — cannot drift across conditions
    assert task["model"] == base["model"]
    assert task["image"] == base["image"]
    assert task["train"]["learning_rate"] == base["train"]["learning_rate"]

    # each condition still owns its checkpoints / run name
    assert task["paths"]["output_dir"] != base["paths"]["output_dir"]
    assert task["train"]["run_name"] != base["train"]["run_name"]
    assert "base" not in task   # the pointer is consumed, not leaked
