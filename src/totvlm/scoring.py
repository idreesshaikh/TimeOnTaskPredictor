"""Shared evaluation metrics for every dwell model — all comparisons MUST go
through these. Convention: values arrive in LOG space (y = log1p(dwell_s));
seconds-space metrics use expm1."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true_log, y_pred_log) -> dict[str, float]:
    """MAE/RMSE in log space AND seconds, plus Spearman rho (rank-based, so
    identical in either space)."""
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(y_pred_log, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        return {"n": 0}

    err_log = yp - yt
    yt_s, yp_s = np.expm1(yt), np.expm1(yp)
    err_s = yp_s - yt_s

    if yt.size >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
        rho = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
    else:
        rho = float("nan")

    return {
        "n": int(yt.size),
        "mae_log": float(np.mean(np.abs(err_log))),
        "rmse_log": float(np.sqrt(np.mean(err_log**2))),
        "mae_s": float(np.mean(np.abs(err_s))),
        "rmse_s": float(np.sqrt(np.mean(err_s**2))),
        "spearman_rho": rho,
        "mean_actual_s": float(np.mean(yt_s)),
        "mean_pred_s": float(np.mean(yp_s)),
    }


def metrics_by_navigation(
    y_true_log, y_pred_log, is_navigation
) -> dict[str, dict[str, float]]:
    """`regression_metrics` overall + split by is_navigation (nav = URL change
    with page load bundled in; in_page = cleaner cognitive signal)."""
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(y_pred_log, dtype=float)
    nav = np.asarray(is_navigation, dtype=bool)
    if not (yt.shape == yp.shape == nav.shape):
        raise ValueError("y_true, y_pred, is_navigation must align")
    return {
        "overall": regression_metrics(yt, yp),
        "navigation": regression_metrics(yt[nav], yp[nav]),
        "in_page": regression_metrics(yt[~nav], yp[~nav]),
    }


def calibration_table(
    y_true_log, y_pred_log, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Bin rows by prediction decile; per bin report mean predicted vs mean
    actual seconds. Returns (table, ece) with
    ece = Σ (n_bin/N)·|mean_pred_s − mean_actual_s|."""
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(y_pred_log, dtype=float)
    df = pd.DataFrame({"actual_s": np.expm1(yt), "pred_s": np.expm1(yp)})
    df["bin"] = pd.qcut(pd.Series(yp).rank(method="first"),
                        q=min(n_bins, len(df)), labels=False)
    table = (
        df.groupby("bin")
        .agg(n=("pred_s", "size"),
             mean_pred_s=("pred_s", "mean"),
             mean_actual_s=("actual_s", "mean"))
        .reset_index()
    )
    table["gap_s"] = (table["mean_pred_s"] - table["mean_actual_s"]).abs()
    ece = float((table["n"] / len(df) * table["gap_s"]).sum())
    return table, ece


def task_totals(group_ids, y_log) -> tuple[np.ndarray, np.ndarray]:
    """Per-screen → per-task totals: expm1 → sum within trajectory → log1p.
    Returns (sorted unique group ids, log1p of summed seconds) — sorted, so
    targets and every model's predictions come out aligned."""
    seconds = pd.Series(np.expm1(np.asarray(y_log, dtype=float)))
    totals = seconds.groupby(np.asarray(group_ids)).sum()
    return totals.index.to_numpy(), np.log1p(totals.to_numpy())


def spearman_ci(x, y, *, n_boot: int, ci: float, seed: int) -> dict:
    """Spearman rho with a seeded percentile-bootstrap CI over items."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # degenerate resamples yield nan; kept — they widen the CI honestly
        boots[b] = spearmanr(x[idx], y[idx])[0]
    alpha = (1 - ci) / 2
    lo, hi = np.nanquantile(boots, [alpha, 1 - alpha])
    return {"n": n, "rho": round(rho, 4), "ci": ci,
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "n_boot": n_boot}
