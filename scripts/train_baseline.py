"""Train the no-image LightGBM baseline — the bar the VLM must beat.

    uv run python scripts/train_baseline.py [--config configs/baseline.yaml]
                                            [--sample-rows N]

Seeded PREFIX-STABLE row sample (growing N reuses the axTree-feature cache) →
fetch features → train on TRAIN, early-stop on VAL, evaluate TEST once via
totvlm.scoring (+ mean/median floors) → baseline_report.md + booster."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from totvlm.config import load_config
from totvlm.features import (
    build_feature_frame,
    feature_columns,
    fetch_axtree_features,
)
from totvlm.model import lgbm_feature_importances, train_lgbm_baseline
from totvlm.scoring import calibration_table, metrics_by_navigation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per request

SPLIT_ORDER = ("train", "val", "test")


def prefix_stable_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic shuffle of a stable row order, then first n. The sample
    for smaller n is a prefix of the sample for larger n (cache-friendly)."""
    df = df.sort_values(
        ["trajectory_id", "tab_id", "unit_index"], kind="mergesort"
    ).reset_index(drop=True)
    order = list(range(len(df)))
    random.Random(seed).shuffle(order)
    return df.iloc[order[: min(n, len(df))]]


def _fmt_metrics_table(by_nav: dict[str, dict]) -> list[str]:
    lines = [
        "| subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) "
        "| Spearman ρ | mean actual s | mean pred s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in by_nav.items():
        if m.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {name} | {m['n']} | {m['mae_log']:.4f} | {m['rmse_log']:.4f} "
            f"| {m['mae_s']:.2f} | {m['rmse_s']:.2f} | {m['spearman_rho']:.4f} "
            f"| {m['mean_actual_s']:.2f} | {m['mean_pred_s']:.2f} |"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--sample-rows", type=int, default=None,
                    help="Override features.sample_rows (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    fcfg = cfg["features"]
    n_sample = args.sample_rows or fcfg["sample_rows"]
    paths = cfg["paths"]
    lcfg = cfg.get("logging", {"report_to": "none"})

    run = None
    if lcfg["report_to"] == "wandb":
        import wandb
        run = wandb.init(
            project=lcfg["wandb_project"], name=lcfg["run_name"], config=cfg
        )

    # 1–2: load + exclusions (reported, never silent)
    df = pd.read_parquet(paths["rows_final"])
    n_total = len(df)
    df = df[df["split"].notna()]
    n_no_split = n_total - len(df)
    has_ax = df["axtree_ref"].notna() & (df["axtree_ref"] != "")
    n_no_axtree = int((~has_ax).sum())
    df = df[has_ax]
    log.info(
        f"{n_total} rows → {len(df)} eligible "
        f"(-{n_no_split} unassigned split, -{n_no_axtree} no axTree URL)"
    )
    if run is not None:
        wandb.log({
            "rows/total": n_total,
            "rows/eligible": len(df),
            "rows/excluded_no_split": n_no_split,
            "rows/excluded_no_axtree": n_no_axtree,
        })

    # 3: seeded prefix-stable sample
    sampled = prefix_stable_sample(df, n_sample, seed)
    log.info(f"sampled {len(sampled)} rows (seed {seed}, prefix-stable)")

    # 4: fetch axTree features (resumable feature-only cache)
    feats = fetch_axtree_features(
        sampled["axtree_ref"],
        cache_dir=fcfg["axtree_cache_dir"],
        concurrency=fcfg["concurrency"],
        timeout_s=fcfg["timeout_s"],
        max_depth=fcfg["max_tree_depth"],
    )
    action_vocab = list(fcfg["action_vocab"])
    frame = build_feature_frame(sampled, feats, action_vocab)
    n_fetch_failed = int((~frame["ax_resolved"]).sum())
    frame = frame[frame["ax_resolved"]]
    log.info(f"axTree resolved for {len(frame)} rows "
             f"({n_fetch_failed} fetch/parse failures excluded)")

    feat_parquet = Path(paths["features_parquet"])
    feat_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["img_path"], errors="ignore").to_parquet(
        feat_parquet, compression="zstd", index=False
    )

    cols = feature_columns(action_vocab)
    parts = {s: frame[frame["split"] == s] for s in SPLIT_ORDER}
    for s in SPLIT_ORDER:
        if parts[s].empty:
            raise SystemExit(f"split {s!r} is empty after sampling — "
                             f"increase --sample-rows")
    log.info("rows/split: " + ", ".join(
        f"{s}={len(parts[s])}" for s in SPLIT_ORDER))

    x = {s: parts[s][cols] for s in SPLIT_ORDER}
    y = {s: parts[s]["y"].to_numpy() for s in SPLIT_ORDER}
    nav = {s: parts[s]["is_navigation"].to_numpy() for s in SPLIT_ORDER}

    # 5: train + evaluate
    lgbm_params = dict(cfg["lightgbm"], random_state=seed)
    wandb_callbacks = []
    if run is not None:
        wandb_callbacks.append(
            lambda env: wandb.log(
                {"val/l1": env.evaluation_result_list[0][2],
                 "iteration": env.iteration}
            )
        )
    model = train_lgbm_baseline(
        x["train"], y["train"], x["val"], y["val"],
        params=lgbm_params,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        extra_callbacks=wandb_callbacks,
    )
    best_iter = model.best_iteration or lgbm_params["n_estimators"]
    log.info(f"best iteration: {best_iter}")

    pred_test = model.predict(x["test"])
    lgbm_metrics = metrics_by_navigation(y["test"], pred_test, nav["test"])
    cal_table, ece = calibration_table(
        y["test"], pred_test, n_bins=cfg["calibration_bins"]
    )

    # Constant floors: TRAIN mean/median of y, predicted for every TEST row.
    floors = {}
    for name, const in (
        ("train-mean floor", float(np.mean(y["train"]))),
        ("train-median floor", float(np.median(y["train"]))),
    ):
        floors[name] = metrics_by_navigation(
            y["test"], np.full_like(y["test"], const), nav["test"]
        )["overall"]

    importances = lgbm_feature_importances(model, cols)
    model.save_model(paths["model_out"])

    if run is not None:
        wandb.log({
            "best_iteration": best_iter,
            "calibration_ece_s": ece,
            **{f"test/{k}": v for k, v in lgbm_metrics["overall"].items()},
            **{f"floor/{name}/{k}": v
               for name, m in floors.items() for k, v in m.items()},
        })
        wandb.log({
            "feature_importances": wandb.Table(
                columns=["feature", "gain", "pct"],
                data=[[imp["feature"], imp["gain"], imp["pct"]]
                      for imp in importances],
            )
        })

    # 6: report
    lines = [
        "# No-image LightGBM baseline report",
        "",
        f"_Generated {date.today().isoformat()} · config `{args.config}` · "
        f"seed {seed} · best iteration {best_iter}_",
        "",
        "Target: `y = log1p(dwell_s)` (winsorized at the train-split p95 cap; "
        "see `artifacts_lam50/dataset_card.md`). Trained on TRAIN, early-stopped on "
        "VAL l1, evaluated ONCE on TEST. Splits are domain-disjoint "
        "(`artifacts_lam50/splits.json`).",
        "",
        "## Row accounting",
        "",
        f"- Rows in dataset: **{n_total}**",
        f"- Excluded — unassigned split: **{n_no_split}**",
        f"- Excluded — no axTree URL: **{n_no_axtree}**",
        f"- Sampled for axTree fetching (seed {seed}, prefix-stable): "
        f"**{len(sampled)}** of {len(df)} eligible "
        f"(trees average ~1.2 MB; mirroring all ~182k is ~210 GB — "
        f"features are extracted on the fly and cached instead)",
        f"- Excluded — axTree fetch/parse failed: **{n_fetch_failed}**",
        "- Final rows: " + " · ".join(
            f"{s} **{len(parts[s])}**" for s in SPLIT_ORDER),
        "",
        "## Test metrics (LightGBM)",
        "",
        *_fmt_metrics_table(lgbm_metrics),
        "",
        "## Floors (constant predictors, TEST overall)",
        "",
        *_fmt_metrics_table(floors),
        "",
        "## Calibration (TEST, decile bins by prediction)",
        "",
        f"ECE-style weighted gap: **{ece:.2f} s**",
        "",
        "| bin | n | mean pred (s) | mean actual (s) | gap (s) |",
        "|---|---|---|---|---|",
        *[
            f"| {int(r.bin)} | {int(r.n)} | {r.mean_pred_s:.2f} "
            f"| {r.mean_actual_s:.2f} | {r.gap_s:.2f} |"
            for r in cal_table.itertuples()
        ],
        "",
        "## Feature importances (gain)",
        "",
        "| feature | gain | % |",
        "|---|---|---|",
        *[
            f"| {imp['feature']} | {imp['gain']} | {imp['pct']}% |"
            for imp in importances
        ],
        "",
        "## Full metrics (JSON)",
        "",
        "```json",
        json.dumps(
            {
                "lightgbm_test": lgbm_metrics,
                "floors_test": floors,
                "calibration_ece_s": ece,
                "best_iteration": best_iter,
                "rows": {s: len(parts[s]) for s in SPLIT_ORDER},
                "excluded": {
                    "unassigned_split": n_no_split,
                    "no_axtree_url": n_no_axtree,
                    "axtree_fetch_failed": n_fetch_failed,
                },
                "sample": {"requested": n_sample, "taken": len(sampled)},
            },
            indent=2,
        ),
        "```",
        "",
    ]
    report = Path(paths["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    log.info(f"report → {report} · model → {paths['model_out']}")
    log.info(json.dumps(lgbm_metrics["overall"], indent=2))

    if run is not None:
        wandb.save(str(report))
        wandb.finish()


if __name__ == "__main__":
    main()
