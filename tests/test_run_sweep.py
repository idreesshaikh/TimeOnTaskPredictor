"""scripts/run_sweep.py decision rule: bracket-and-refine over the λ grid.

The search must (a) train the coarse anchors first, (b) then only the grid
neighbours of the best-VAL λ, (c) keep walking while a new neighbour keeps
winning (unimodality), and (d) stop — return no candidates — the moment both
neighbours of the best λ are confirmed worse. Lower score = better (VAL MAE).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_sweep import next_lams  # noqa: E402

GRID = [0.0, 0.10, 0.25, 0.35, 0.50, 0.60, 0.75, 0.90]
COARSE = [0.0, 0.35, 0.75]


def test_coarse_anchors_come_first():
    assert next_lams(GRID, {}, COARSE) == COARSE


def test_unscored_coarse_anchors_finish_before_refining():
    scores = {0.0: 0.50, 0.35: 0.40}
    assert next_lams(GRID, scores, COARSE) == [0.75]


def test_refine_returns_both_neighbours_of_interior_best():
    scores = {0.0: 0.50, 0.35: 0.40, 0.75: 0.45}
    assert next_lams(GRID, scores, COARSE) == [0.25, 0.50]


def test_refine_at_grid_edge_has_one_neighbour():
    scores = {0.0: 0.40, 0.35: 0.45, 0.75: 0.50}
    assert next_lams(GRID, scores, COARSE) == [0.10]


def test_walk_continues_while_new_neighbour_wins():
    # 0.25 beat the coarse winner 0.35 → its other neighbour 0.10 is next.
    scores = {0.0: 0.50, 0.35: 0.40, 0.75: 0.45, 0.25: 0.38, 0.50: 0.42}
    assert next_lams(GRID, scores, COARSE) == [0.10]


def test_stops_when_both_neighbours_of_best_are_worse():
    scores = {0.0: 0.50, 0.35: 0.40, 0.75: 0.45, 0.25: 0.44, 0.50: 0.42}
    assert next_lams(GRID, scores, COARSE) == []


def test_worst_case_never_exceeds_the_grid():
    # Monotonically decreasing scores toward λ=0.90: the walk may cover the
    # whole grid but must then terminate.
    scores: dict[float, float] = {}
    trained = 0
    while todo := next_lams(GRID, scores, COARSE):
        for lam in todo:
            scores[lam] = 1.0 - lam  # bigger λ always better
            trained += 1
        assert trained <= len(GRID)
    assert min(scores, key=scores.get) == 0.90
    assert trained <= len(GRID)
