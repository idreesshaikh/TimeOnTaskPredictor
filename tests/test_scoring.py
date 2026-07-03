"""Unit tests for the shared metrics in totvlm.scoring (log-space convention:
y = log1p(dwell_s); seconds via expm1)."""
import numpy as np
import pytest

from totvlm.scoring import (
    calibration_table,
    metrics_by_navigation,
    regression_metrics,
)


def test_perfect_predictions():
    y = np.log1p([1.0, 5.0, 30.0, 120.0])
    m = regression_metrics(y, y)
    assert m["n"] == 4
    assert m["mae_log"] == 0.0 and m["rmse_log"] == 0.0
    assert m["mae_s"] == 0.0 and m["rmse_s"] == 0.0
    assert m["spearman_rho"] == pytest.approx(1.0)


def test_known_errors_log_and_seconds():
    # actual 1s & 3s, predicted 2s & 5s → seconds MAE = (1 + 2) / 2 = 1.5
    y_true = np.log1p([1.0, 3.0])
    y_pred = np.log1p([2.0, 5.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae_s"] == pytest.approx(1.5)
    assert m["rmse_s"] == pytest.approx(np.sqrt((1.0**2 + 2.0**2) / 2))
    expected_log = np.mean([np.log(3) - np.log(2), np.log(6) - np.log(4)])
    assert m["mae_log"] == pytest.approx(expected_log)
    assert m["mean_actual_s"] == pytest.approx(2.0)
    assert m["mean_pred_s"] == pytest.approx(3.5)


def test_spearman_is_rank_based_and_scale_free():
    y_true = np.log1p([1.0, 2.0, 4.0, 8.0, 16.0])
    y_pred = 0.1 * y_true + 3.0          # monotone rescale → same ranks
    assert regression_metrics(y_true, y_pred)["spearman_rho"] == pytest.approx(1.0)
    m = regression_metrics(y_true, y_true[::-1].copy())
    assert m["spearman_rho"] == pytest.approx(-1.0)


def test_degenerate_inputs():
    assert regression_metrics([], []) == {"n": 0}
    # constant predictions → rho undefined, not a crash
    m = regression_metrics(np.log1p([1.0, 2.0, 3.0]), np.log1p([2.0, 2.0, 2.0]))
    assert np.isnan(m["spearman_rho"])
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])


def test_metrics_by_navigation_partitions():
    rng = np.random.default_rng(42)
    y_true = np.log1p(rng.uniform(0.5, 100, size=40))
    y_pred = y_true + rng.normal(0, 0.1, size=40)
    nav = np.arange(40) % 2 == 0
    out = metrics_by_navigation(y_true, y_pred, nav)
    assert set(out) == {"overall", "navigation", "in_page"}
    assert out["navigation"]["n"] == 20 and out["in_page"]["n"] == 20
    assert out["overall"]["n"] == 40
    # overall MAE is the weighted mean of the two subsets
    blended = (out["navigation"]["mae_log"] + out["in_page"]["mae_log"]) / 2
    assert out["overall"]["mae_log"] == pytest.approx(blended)


def test_calibration_perfect_predictions_zero_ece():
    y = np.log1p(np.linspace(0.5, 300, 200))
    table, ece = calibration_table(y, y, n_bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 200
    assert ece == pytest.approx(0.0)
    # bins ordered by prediction → mean actual seconds must be increasing
    assert table["mean_actual_s"].is_monotonic_increasing


def test_calibration_detects_constant_offset():
    rng = np.random.default_rng(42)
    actual_s = rng.uniform(1, 60, size=500)
    y_true = np.log1p(actual_s)
    y_pred = np.log1p(actual_s + 10.0)   # always 10s too high
    table, ece = calibration_table(y_true, y_pred, n_bins=10)
    assert ece == pytest.approx(10.0, abs=0.5)
    assert (table["mean_pred_s"] > table["mean_actual_s"]).all()


def test_calibration_handles_tied_predictions():
    y_true = np.log1p(np.arange(1.0, 31.0))
    y_pred = np.full(30, np.log1p(15.0))   # all tied
    table, ece = calibration_table(y_true, y_pred, n_bins=10)
    assert table["n"].sum() == 30
    assert ece > 0
