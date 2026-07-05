"""
LUPI correctness, CPU-only (all of totvlm.lupi): the λ-blend of true labels
and teacher predictions (blend_lupi_targets), the blended target reaching the
SFT example text, and the teacher's domain-grouped fold assignment
(assign_domain_folds) — the pieces that must be right BEFORE burning GPU time
on the LUPI conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from totvlm.data import ROW_KEY, build_vlm_examples
from totvlm.lupi import assign_domain_folds, blend_lupi_targets

CAP = 24.954


def _train_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "trajectory_id": ["t1", "t1", "t2"],
        "tab_id": [0, 0, 1],
        "unit_index": [1, 2, 1],
        "dwell_s": [4.0, 10.0, 7.0],
    })


def _teacher(rows: pd.DataFrame, pred_log: list[float]) -> pd.DataFrame:
    t = rows[ROW_KEY].copy()
    t["teacher_pred_log"] = pred_log
    return t


# ── λ-blend ───────────────────────────────────────────────────────────────────

def test_blend_is_log_space_mix():
    rows = _train_rows()
    teacher = _teacher(rows, [np.log1p(8.0), np.log1p(10.0), np.log1p(1.0)])
    out, stats = blend_lupi_targets(rows, teacher, lam=0.5)
    expected = np.expm1(0.5 * np.log1p(rows["dwell_s"].to_numpy())
                        + 0.5 * teacher["teacher_pred_log"].to_numpy())
    assert np.allclose(out["target_s"], expected)
    assert stats["n_blended"] == 3 and stats["coverage"] == 1.0


def test_blend_extremes():
    rows = _train_rows()
    teacher = _teacher(rows, [np.log1p(1.0), np.log1p(2.0), np.log1p(3.0)])
    lam0, _ = blend_lupi_targets(rows, teacher, lam=0.0)
    assert np.allclose(lam0["target_s"], rows["dwell_s"])   # plain SFT
    lam1, _ = blend_lupi_targets(rows, teacher, lam=1.0)
    assert np.allclose(lam1["target_s"], [1.0, 2.0, 3.0])   # pure teacher


def test_uncovered_rows_keep_true_label():
    rows = _train_rows()
    teacher = _teacher(rows.iloc[:1], [np.log1p(99.0)])   # covers row 0 only
    out, stats = blend_lupi_targets(rows, teacher, lam=0.5)
    assert stats["n_blended"] == 1
    assert np.allclose(out["target_s"].iloc[1:], rows["dwell_s"].iloc[1:])
    assert out["target_s"].iloc[0] != rows["dwell_s"].iloc[0]


def test_bad_lambda_rejected():
    rows = _train_rows()
    teacher = _teacher(rows, [0.0, 0.0, 0.0])
    for lam in (-0.1, 1.1):
        with pytest.raises(ValueError):
            blend_lupi_targets(rows, teacher, lam=lam)


# ── blended target reaches the SFT text; true dwell stays for metrics ─────────

def test_target_text_uses_blend_dwell_stays_true(tmp_path):
    shot = tmp_path / "shot.png"
    Image.new("RGB", (640, 480)).save(shot)
    rows = pd.DataFrame({
        "img_path": [str(shot)],
        "img_resolved": [True],
        "task_title": ["x"],
        "dwell_s": [4.0],
        "target_s": [8.0],     # what blend_lupi_targets would have written
        "is_navigation": [False],
    })
    ex = build_vlm_examples(rows, winsor_cap=CAP, max_side=1024,
                            min_pixels=100352, max_pixels=602112)[0]
    assert ex["messages"][-1]["content"][0]["text"] == "dwell_seconds: 8.0"
    assert ex["dwell_s"] == 4.0   # decode hook still scores against truth


# ── teacher fold assignment ───────────────────────────────────────────────────

def test_folds_group_by_domain_and_are_deterministic():
    domains = pd.Series(["a.com", "b.com", "a.com", "c.com", "d.com",
                         "e.com", "b.com"])
    f1 = assign_domain_folds(domains, k=3, seed=42)
    f2 = assign_domain_folds(domains, k=3, seed=42)
    assert f1.tolist() == f2.tolist()                       # deterministic
    per_domain = pd.DataFrame({"d": domains, "f": f1}).groupby("d")["f"].nunique()
    assert (per_domain == 1).all()                          # domain-disjoint
    assert set(f1) == {0, 1, 2}                             # all folds used
