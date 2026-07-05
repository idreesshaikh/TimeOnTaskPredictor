"""
Paper figure factory (scripts/make_figures.py): every figure renders from
synthetic stage outputs, missing inputs are carded as skipped (never
silently dropped), and the floors are excluded from scatter panels.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "make_figures", REPO / "scripts" / "make_figures.py"
)
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

MODELS = [
    "train-median floor",
    "LightGBM (no image)",
    "VLM (image+features)",
    "VLM (image+features+task)",
]


def _predictions(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dwell = rng.uniform(0.5, 25.0, n)
    df = pd.DataFrame(
        {
            "trajectory_id": [f"t{i // 4}" for i in range(n)],  # 4 screens/task
            "tab_id": 0,
            "unit_index": np.arange(n) % 4,
            "dwell_s": dwell,
            "y": np.log1p(dwell),
            "is_navigation": rng.random(n) < 0.3,
        }
    )
    for name in MODELS:
        noise = 0.0 if "floor" in name else rng.normal(0, 0.4, n)
        df[f"pred_log:{name}"] = (
            np.full(n, np.log1p(5.0)) if "floor" in name else df["y"] + noise
        )
    return df


def _metrics() -> dict:
    m = {
        "n": 60,
        "mae_log": 0.5,
        "rmse_log": 0.6,
        "mae_s": 4.0,
        "rmse_s": 6.0,
        "spearman_rho": 0.4,
        "mean_actual_s": 7.0,
        "mean_pred_s": 6.5,
    }
    return {
        "head_to_head_per_screen": {
            name: {s: dict(m) for s in ("overall", "navigation", "in_page")}
            for name in MODELS
        },
        "task_level": {name: dict(m) for name in MODELS},
    }


@pytest.fixture()
def cfg(tmp_path):
    eval_cfg = tmp_path / "eval.yaml"
    eval_cfg.write_text(yaml.safe_dump({"scoring": {"task_min_steps": 2}}))
    return {
        "seed": 42,
        "eval_config": str(eval_cfg),
        "paths": {
            "rows_final": str(tmp_path / "rows_final.parquet"),
            "eval_metrics": str(tmp_path / "eval_metrics.json"),
            "eval_predictions": str(tmp_path / "eval_predictions.parquet"),
            "baseline_model": str(tmp_path / "baseline_lgbm.txt"),
            "out_dir": str(tmp_path / "figures"),
            "card": str(tmp_path / "figures" / "figures_card.md"),
        },
        "style": {"dpi": 72, "font_size": 9},
        "dwell_distribution": {"bins": 20},
        "scatter": {"max_points": 50, "alpha": 0.4, "point_size": 9},
        "calibration": {"n_bins": 5},
        "feature_importance": {"top_n": 3},
    }


def test_learned_models_excludes_floors():
    names = mf._learned_models(_predictions())
    assert "train-median floor" not in names
    assert names == ["LightGBM (no image)", "VLM (image+features)",
                     "VLM (image+features+task)"]


def test_prediction_figures_render(cfg, tmp_path):
    preds = _predictions()
    for fn, args in [
        (mf.fig_screen_scatter, (preds, cfg)),
        (mf.fig_task_scatter, (preds, cfg, 2)),
        (mf.fig_calibration, (preds, cfg)),
        (mf.fig_head_to_head, (_metrics(), cfg)),
        (mf.fig_task_level, (_metrics(), cfg)),
    ]:
        dest = tmp_path / f"{fn.__name__}.png"
        assert fn(*args, dest) == dest
        assert dest.stat().st_size > 0, fn.__name__


def test_dwell_distribution_renders(cfg, tmp_path):
    rng = np.random.default_rng(1)
    rows = pd.DataFrame(
        {
            "split": ["train"] * 50 + ["val"] * 20 + ["test"] * 30,
            "dwell_s_raw": rng.uniform(0.1, 90.0, 100),
        }
    )
    rows["dwell_s"] = rows["dwell_s_raw"].clip(upper=25.0)
    dest = tmp_path / "dist.png"
    assert mf.fig_dwell_distribution(rows, cfg, dest) == dest
    assert dest.stat().st_size > 0


def test_feature_importance_renders(cfg, tmp_path):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(2)
    x = pd.DataFrame(
        rng.normal(size=(80, 4)), columns=["ax_n_nodes", "ax_depth", "vp_w", "vp_h"]
    )
    y = x["ax_n_nodes"] * 2 + rng.normal(size=80)
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1},
        lgb.Dataset(x, label=y),
        num_boost_round=5,
    )
    path = tmp_path / "baseline_lgbm.txt"
    booster.save_model(str(path))
    dest = tmp_path / "imp.png"
    assert mf.fig_feature_importance(path, cfg, dest) == dest
    assert dest.stat().st_size > 0


def test_main_cards_missing_inputs_as_skipped(cfg, tmp_path, monkeypatch):
    """No stage outputs at all → every figure is skipped, card still written,
    exit code 0 (safe to run at any point in the pipeline)."""
    cfg_path = tmp_path / "figures.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--config", str(cfg_path)])
    mf.main()
    card = Path(cfg["paths"]["card"]).read_text()
    for name in (
        "fig_dwell_distribution",
        "fig_head_to_head",
        "fig_screen_scatter",
        "fig_feature_importance",
    ):
        assert f"| {name} | ⏳ skipped |" in card
    assert "✅ written" not in card


def test_main_end_to_end_with_all_inputs(cfg, tmp_path, monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(3)
    rows = pd.DataFrame(
        {
            "split": ["train"] * 50 + ["val"] * 20 + ["test"] * 30,
            "dwell_s_raw": rng.uniform(0.1, 90.0, 100),
        }
    )
    rows["dwell_s"] = rows["dwell_s_raw"].clip(upper=25.0)
    rows.to_parquet(cfg["paths"]["rows_final"], index=False)
    _predictions().to_parquet(cfg["paths"]["eval_predictions"], index=False)
    Path(cfg["paths"]["eval_metrics"]).write_text(json.dumps(_metrics()))
    x = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1},
        lgb.Dataset(x, label=x["a"]),
        num_boost_round=5,
    )
    booster.save_model(cfg["paths"]["baseline_model"])

    cfg_path = tmp_path / "figures.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--config", str(cfg_path)])
    mf.main()

    out = Path(cfg["paths"]["out_dir"])
    written = {p.name for p in out.glob("*.png")}
    assert written == {
        "fig_dwell_distribution.png",
        "fig_head_to_head.png",
        "fig_task_level.png",
        "fig_screen_scatter.png",
        "fig_task_scatter.png",
        "fig_calibration.png",
        "fig_feature_importance.png",
    }
    card = Path(cfg["paths"]["card"]).read_text()
    assert "⏳ skipped" not in card
