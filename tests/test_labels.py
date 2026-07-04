"""
Oracle tests for per-screen dwell-time label construction (SPEC.md
"Verified test oracles"). Fixtures under tests/fixtures/ are the two oracle
trajectories extracted verbatim from raw WebChain JSON (bulky `html` fields
stripped; all timing/img/tab fields untouched).
"""
from pathlib import Path

import pytest

from totvlm.data import iter_raw_trajectories
from totvlm.labels import merge_screen_units, rows_for_trajectory

FIXTURES = Path(__file__).parent / "fixtures"

# (dwell_s, is_navigation) per SPEC.md, ±0.01 s
ORACLE_1 = [(54.046, False), (33.460, True), (73.887, True)]
ORACLE_2 = [
    (6.683, True), (5.716, True), (0.830, False),
    (32.787, True), (19.290, True), (20.836, True),
]


def _load(traj_id: str):
    return next(iter_raw_trajectories(FIXTURES / f"{traj_id}.json"))


@pytest.mark.parametrize(
    "traj_id, expected",
    [
        ("yjeXPEBxd5EACsDz4xPWx", ORACLE_1),
        ("8uJRwmgg3iwq9-rnZ_MpJ", ORACLE_2),
    ],
)
def test_oracle_dwells(traj_id, expected):
    rows = rows_for_trajectory(_load(traj_id))
    assert len(rows) == len(expected), (
        f"{traj_id}: expected exactly {len(expected)} rows, got {len(rows)}"
    )
    for row, (dwell, nav) in zip(rows, expected):
        assert row["dwell_s_raw"] == pytest.approx(dwell, abs=0.01)
        assert row["is_navigation"] == nav
        assert row["trajectory_id"] == traj_id


def test_launch_app_dropped():
    traj = _load("yjeXPEBxd5EACsDz4xPWx")
    n_launch = sum(1 for s in traj.steps if s.is_launch_app)
    assert n_launch == 3, "fixture must contain launchApp steps to test the rule"

    units = merge_screen_units(traj.steps)
    assert all((u.action_type or "").lower() != "launchapp" for u in units)
    # 9 raw steps − 3 launchApp = 6 actionable; one 2-step merge → 5 units
    assert sum(u.n_steps_merged for u in units) == len(traj.steps) - n_launch


def test_consecutive_identical_img_steps_merge():
    units = merge_screen_units(_load("yjeXPEBxd5EACsDz4xPWx").steps)
    merged = [u for u in units if u.n_steps_merged > 1]
    assert len(merged) == 1 and merged[0].n_steps_merged == 2


def test_first_in_tab_excluded_and_no_cross_tab_dwell():
    traj = _load("yjeXPEBxd5EACsDz4xPWx")
    units = merge_screen_units(traj.steps)
    tabs = {u.tab_id for u in units}
    assert len(tabs) == 2, "fixture must span two tabs to test per-tab chains"

    rows = rows_for_trajectory(traj)
    emitted = {r["unit_index"] for r in rows}
    for tab in tabs:
        first_unit = next(u for u in units if u.tab_id == tab)
        assert first_unit.unit_index not in emitted, (
            f"first unit of tab {tab} must not be a target"
        )
    # tab 758227536 has a single unit; a cross-tab pair must never form a dwell,
    # so every emitted row lives in the multi-unit tab.
    assert all(r["tab_id"] == 758228211 for r in rows)
