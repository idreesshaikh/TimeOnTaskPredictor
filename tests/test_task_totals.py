"""
task_totals: the per-trajectory rollup behind the task-level evaluation
(sum of per-screen seconds → whole-task Time-on-Task, RQ v2).
"""
from __future__ import annotations

import numpy as np
import pytest

from totvlm.scoring import task_totals


def test_sums_seconds_within_group_in_log_space():
    ids = ["b", "a", "b", "a", "c"]
    y_log = np.log1p([2.0, 1.0, 3.0, 4.0, 10.0])   # seconds: b:2+3, a:1+4, c:10
    groups, totals_log = task_totals(ids, y_log)
    assert list(groups) == ["a", "b", "c"]          # sorted → aligned across models
    assert np.expm1(totals_log) == pytest.approx([5.0, 5.0, 10.0])


def test_alignment_is_stable_across_models():
    """Same group_ids ⇒ same group order, whatever the values are."""
    ids = ["t2", "t1", "t2", "t1"]
    g1, _ = task_totals(ids, np.log1p([1, 2, 3, 4]))
    g2, _ = task_totals(ids, np.log1p([9, 9, 9, 9]))
    assert list(g1) == list(g2) == ["t1", "t2"]


def test_single_step_task_passes_through():
    groups, totals_log = task_totals(["only"], np.log1p([7.5]))
    assert list(groups) == ["only"]
    assert np.expm1(totals_log)[0] == pytest.approx(7.5)
