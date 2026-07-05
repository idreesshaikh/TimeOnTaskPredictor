"""LUPI (learning under privileged information) via generalized distillation
— Lopez-Paz et al. 2016. One home for the whole mechanism:

  1. `assign_domain_folds` / `oof_teacher_predictions` — the out-of-fold
     LightGBM teacher over the VLM's TRAIN rows, using the PRIVILEGED features
     (axTree structure, nav flag, step index, click area) that exist at train
     time only. Folds are grouped by registrable domain, so no row is ever
     predicted by a booster that saw its own domain (same leak discipline as
     the frozen splits). Driven by scripts/make_lupi_teacher.py.
  2. `blend_lupi_targets` — mix each TRAIN target toward its teacher
     prediction by λ, in the model's log1p space. Called by totvlm.train.

The privileged features reach the VLM ONLY through the soft target; the
prompt, the inputs and inference stay screenshot-only, so the gap to the plain
condition measures what that train-time metadata is worth to a pixels-only
predictor. Val/test targets are never blended."""
from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd

from totvlm.data import ROW_KEY
from totvlm.model import train_lgbm_baseline

log = logging.getLogger(__name__)


# ── teacher: out-of-fold predictions over the VLM's train rows ────────────────

def assign_domain_folds(domains: pd.Series, k: int, seed: int) -> pd.Series:
    """Deterministic k-fold assignment by registrable domain: seeded shuffle
    of the unique domains, then round-robin. Every domain lands in exactly one
    fold, so no booster is early-stopped or evaluated on its own domains."""
    uniq = sorted(domains.dropna().unique())
    random.Random(seed).shuffle(uniq)
    fold_of = {d: i % k for i, d in enumerate(uniq)}
    return domains.map(fold_of)


def oof_teacher_predictions(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    k: int,
    seed: int,
    lgbm_params: dict,
    early_stopping_rounds: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-fold predictions: fold i is predicted by a booster trained on
    the other folds minus fold (i+1)%k, which serves as the early-stopping
    set. Returns (frame + teacher_pred_log + fold, per-fold stats)."""
    frame = frame.copy()
    frame["fold"] = assign_domain_folds(frame["registrable_domain"], k, seed)
    frame["teacher_pred_log"] = np.nan
    fold_stats = []
    for i in range(k):
        es = (i + 1) % k
        tr_mask = ~frame["fold"].isin([i, es])
        es_mask = frame["fold"] == es
        pr_mask = frame["fold"] == i
        if not tr_mask.any() or not es_mask.any() or not pr_mask.any():
            raise SystemExit(
                f"fold {i}: empty train/early-stop/predict part "
                f"(rows: train {int(tr_mask.sum())}, es {int(es_mask.sum())}, "
                f"predict {int(pr_mask.sum())}) — too few domains for "
                f"k_folds={k}; lower it in configs/lupi.yaml"
            )
        booster = train_lgbm_baseline(
            frame.loc[tr_mask, cols], frame.loc[tr_mask, "y"].to_numpy(),
            frame.loc[es_mask, cols], frame.loc[es_mask, "y"].to_numpy(),
            params=lgbm_params,
            early_stopping_rounds=early_stopping_rounds,
        )
        frame.loc[pr_mask, "teacher_pred_log"] = booster.predict(
            frame.loc[pr_mask, cols]
        )
        fold_stats.append({
            "fold": i,
            "n_train": int(tr_mask.sum()),
            "n_early_stop": int(es_mask.sum()),
            "n_predicted": int(pr_mask.sum()),
            "best_iteration": booster.best_iteration
            or lgbm_params["n_estimators"],
        })
        log.info(f"fold {i}: predicted {int(pr_mask.sum())} rows "
                 f"(best iter {fold_stats[-1]['best_iteration']})")
    return frame, fold_stats


# ── target blend: soft half of the LUPI training target ───────────────────────

def blend_lupi_targets(
    df: pd.DataFrame, teacher: pd.DataFrame, lam: float
) -> tuple[pd.DataFrame, dict]:
    """TRAIN targets become a λ-mix of the teacher's out-of-fold prediction
    and the true label, mixed in log1p space and written back to `target_s`.
    Rows the teacher does not cover keep the true label. Call on the TRAIN
    split only; val/test targets are never blended."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lupi lambda must be in [0, 1], got {lam}")
    merged = df.merge(
        teacher[ROW_KEY + ["teacher_pred_log"]],
        on=ROW_KEY, how="left", validate="1:1",
    )
    covered = merged["teacher_pred_log"].notna()
    blended_log = (
        (1.0 - lam) * np.log1p(merged["dwell_s"])
        + lam * merged["teacher_pred_log"]
    )
    merged["target_s"] = np.where(
        covered, np.expm1(blended_log), merged["dwell_s"]
    )
    shift = (merged.loc[covered, "target_s"]
             - merged.loc[covered, "dwell_s"]).abs()
    stats = {
        "lambda": lam,
        "n_rows": int(len(merged)),
        "n_blended": int(covered.sum()),
        "coverage": round(float(covered.mean()), 4) if len(merged) else 0.0,
        "mean_abs_shift_s": round(float(shift.mean()), 3) if len(shift) else 0.0,
    }
    return merged, stats
