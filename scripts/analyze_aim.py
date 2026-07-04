"""Theory-grounding analysis (optional appendix): is the model's "time sense"
reducible to classical visual-clutter metrics (AIM)?

    uv run python scripts/analyze_aim.py [--config configs/external.yaml]

Strictly POST-HOC over the cached zero-shot predictions — never runs the
model, never feeds training/tuning (allowlisted in test_external_guard).
Regressions per subset: human~AIM, VLM~AIM, and whether the VLM's residual
(what clutter can't explain) still tracks human time. Targets log1p(s), AIM
z-scored, in-sample OLS R² stated as descriptive. → artifacts/aim_analysis.md"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from totvlm.config import load_config
from totvlm.data import parse_dwell_output_lenient
from totvlm.scoring import spearman_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Plain OLS with intercept on standardized predictors.
    Returns (coefs incl. intercept, fitted values, in-sample R²)."""
    design = np.column_stack([np.ones(len(x)), x])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefs, fitted, r2


def analyze_subset(df: pd.DataFrame, metric_cols: list[str], *,
                   n_boot: int, ci: float, seed: int) -> dict:
    """The three regressions on one subset. Assumes NaN-free input."""
    x = df[metric_cols].to_numpy(float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
    human = np.log1p(df["human_time_s"].to_numpy(float))
    pred = np.log1p(df["pred_s"].to_numpy(float))

    coef_h, fit_h, r2_human = ols(x, human)
    coef_p, fit_p, r2_pred = ols(x, pred)
    resid_pred = pred - fit_p
    resid_human = human - fit_h

    return {
        "n": len(df),
        # 1. clutter → human time (the literature anchor)
        "human_vs_aim_r2": round(r2_human, 4),
        # 2. clutter → model estimate (how "classical" is the model?)
        "pred_vs_aim_r2": round(r2_pred, 4),
        # reference: raw rank agreement on this subset
        "pred_vs_human": spearman_ci(pred, human, n_boot=n_boot, ci=ci,
                                     seed=seed),
        # 3. beyond clutter: model residual vs human time / human residual
        "pred_residual_vs_human": spearman_ci(
            resid_pred, human, n_boot=n_boot, ci=ci, seed=seed),
        "partial_residual_vs_residual": spearman_ci(
            resid_pred, resid_human, n_boot=n_boot, ci=ci, seed=seed),
        # standardized coefficients (log-seconds per SD of each metric)
        "human_coefs": {m: round(float(c), 4)
                        for m, c in zip(metric_cols, coef_h[1:])},
        "pred_coefs": {m: round(float(c), 4)
                       for m, c in zip(metric_cols, coef_p[1:])},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg = cfg["aim"]
    primary = cfg["report"]["primary_category"]
    seed = cfg["seed"]

    preds_path = Path(cfg["paths"]["preds"])
    if not preds_path.exists():
        sys.exit(f"{preds_path} missing — this analysis is post-hoc: run the "
                 f"single zero-shot pass (scripts/validate_external.py) "
                 f"first.")

    root = Path(cfg["paths"]["external_root"]) / cfg["source"]
    items = pd.read_csv(root / "items.csv")
    aim = pd.read_csv(acfg["metrics_csv"])
    preds = pd.read_parquet(preds_path)

    rename = dict(acfg["metrics"])           # m8_0 → feature_congestion, ...
    aim = aim[["img_name", *rename]].rename(columns=rename)
    metric_cols = list(rename.values())

    df = (items.merge(preds, on="item_id", how="inner", validate="1:1")
               .merge(aim, left_on="item_id", right_on="img_name",
                      how="left", validate="1:1"))
    df["pred_s"] = [parse_dwell_output_lenient(t)[0] for t in df["raw_output"]]

    n_all = len(df)
    df = df.dropna(subset=["pred_s", *metric_cols])
    n_dropped = n_all - len(df)
    log.info(f"{len(df)} usable items ({n_dropped} dropped: parse failure "
             f"or missing AIM metric — counted, not hidden)")

    kw = dict(n_boot=acfg["bootstrap_samples"], ci=acfg["ci"], seed=seed)
    subsets = {}
    part = df[df["category"] == primary]
    if len(part) >= 3:
        subsets[f"{primary} (in-domain, primary)"] = analyze_subset(
            part, metric_cols, **kw)
    subsets["ALL (incl. out-of-domain)"] = analyze_subset(df, metric_cols,
                                                          **kw)

    def fmt_ci(m: dict) -> str:
        return f"{m['rho']:.3f} [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]"

    lines = [
        "# AIM theory-grounding analysis — what does the time sense consist "
        "of?",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `{args.config}` · seed {seed} · post-hoc over cached "
        f"zero-shot predictions (`{preds_path}`)_",
        "",
        "Predictors: per-screen **Aalto Interface Metrics** (visual-clutter "
        "models: " + ", ".join(metric_cols) + "). Targets in log1p seconds; "
        "predictors z-scored; R² is in-sample OLS (descriptive). "
        f"Spearman CIs: seeded bootstrap, {acfg['bootstrap_samples']} "
        "resamples.",
        "",
        f"- Usable items: **{len(df)}** ({n_dropped} dropped: parse failure "
        "or missing metric)",
        "",
        "| subset | n | human~AIM R² | pred~AIM R² | pred vs human ρ "
        "| **pred-residual vs human ρ** | partial (both residualized) ρ |",
        "|---|---|---|---|---|---|---|",
        *[f"| {name} | {s['n']} | {s['human_vs_aim_r2']:.3f} "
          f"| {s['pred_vs_aim_r2']:.3f} | {fmt_ci(s['pred_vs_human'])} "
          f"| **{fmt_ci(s['pred_residual_vs_human'])}** "
          f"| {fmt_ci(s['partial_residual_vs_residual'])} |"
          for name, s in subsets.items()],
        "",
        "Reading guide: column 3 anchors to the clutter literature (how "
        "much of *human* time is quantified clutter); column 4 says how "
        "much of the *model's* estimate is reducible to clutter; the bold "
        "column is the finding — a residual that still tracks human time "
        "means the model learned interface signal *beyond* established "
        "complexity models, and a residual that doesn't is itself the "
        "honest result (screenshot-predictable time ≈ clutter).",
        "",
        "## Standardized coefficients (log-s per SD)",
        "",
        "| metric | → human time | → model estimate |",
        "|---|---|---|",
    ]
    first = next(iter(subsets.values()))
    lines += [
        f"| {m} | {first['human_coefs'][m]:+.4f} "
        f"| {first['pred_coefs'][m]:+.4f} |"
        for m in metric_cols
    ]
    lines += [
        "",
        f"_Coefficients from the first subset above (n={first['n']})._",
        "",
        "## Full results (JSON)",
        "",
        "```json",
        json.dumps(subsets, indent=2),
        "```",
        "",
    ]
    report = Path(acfg["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    log.info(f"analysis card → {report}")
    for name, s in subsets.items():
        log.info(f"{name}: human~AIM R²={s['human_vs_aim_r2']} · "
                 f"pred~AIM R²={s['pred_vs_aim_r2']} · "
                 f"residual ρ={s['pred_residual_vs_human']['rho']}")


if __name__ == "__main__":
    main()
