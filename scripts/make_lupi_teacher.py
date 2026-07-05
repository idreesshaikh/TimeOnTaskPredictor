"""LUPI teacher: out-of-fold LightGBM predictions over the VLM's TRAIN rows.

    uv run python scripts/make_lupi_teacher.py [--config configs/lupi.yaml]
                                               [--limit N]

The teacher sees the PRIVILEGED features (axTree structural stats, the
navigation flag, step index, click-target area) that exist only at training
time; its predictions become the soft half of the LUPI training target
(generalized distillation — the blend happens in totvlm.train via
`lupi.lambda` in configs/vlm_lupi.yaml).

Leak control: folds are grouped by REGISTRABLE DOMAIN (same philosophy as
the frozen splits) — every train row is predicted by a booster that never
saw its domain. Early stopping uses the next fold, never the predicted one.
Only TRAIN rows are processed; val/test targets are never blended."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from totvlm.config import load_config
from totvlm.data import ROW_KEY
from totvlm.features import (
    build_feature_frame,
    feature_columns,
    fetch_axtree_features,
)
from totvlm.model import train_lgbm_baseline
from totvlm.scoring import metrics_by_navigation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def assign_domain_folds(domains: pd.Series, k: int, seed: int) -> pd.Series:
    """Deterministic k-fold assignment by registrable domain: seeded shuffle
    of the unique domains, then round-robin. Every domain lands in exactly
    one fold, so no booster is early-stopped or evaluated on its own
    training domains."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lupi.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the row count after the stable sort "
                         "(smoke tests only — the card records it)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bcfg = load_config(cfg["baseline_config"])
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    fcfg = bcfg["features"]
    k = cfg["teacher"]["k_folds"]

    # Teacher preds are needed exactly where the VLM trains: TRAIN rows with
    # a resolved screenshot. Rows without an axTree URL cannot be covered —
    # they keep the true label downstream, and the coverage is reported.
    df = pd.read_parquet(cfg["paths"]["rows_final"])
    vlm_train = df[(df["split"] == "train") & df["img_resolved"]]
    has_ax = vlm_train["axtree_ref"].notna() & (vlm_train["axtree_ref"] != "")
    rows = vlm_train[has_ax].sort_values(ROW_KEY, kind="mergesort")
    n_no_axtree = int((~has_ax).sum())
    if args.limit is not None:
        rows = rows.head(args.limit)
    log.info(
        f"{len(vlm_train)} VLM train rows → {len(rows)} with an axTree URL "
        f"({n_no_axtree} uncoverable)"
        + (f" · LIMIT {args.limit}" if args.limit else "")
    )

    feats = fetch_axtree_features(
        rows["axtree_ref"],
        cache_dir=fcfg["axtree_cache_dir"],
        concurrency=fcfg["concurrency"],
        timeout_s=fcfg["timeout_s"],
        max_depth=fcfg["max_tree_depth"],
    )
    vocab = list(fcfg["action_vocab"])
    frame = build_feature_frame(rows, feats, vocab)
    n_fetch_failed = int((~frame["ax_resolved"]).sum())
    frame = frame[frame["ax_resolved"]]
    log.info(f"axTree resolved for {len(frame)} rows "
             f"({n_fetch_failed} fetch/parse failures excluded)")
    if frame.empty:
        raise SystemExit("no rows with resolved axTree features — nothing "
                         "for the teacher to learn from")

    lgbm_params = dict(bcfg["lightgbm"], random_state=seed)
    frame, fold_stats = oof_teacher_predictions(
        frame, feature_columns(vocab),
        k=k, seed=seed,
        lgbm_params=lgbm_params,
        early_stopping_rounds=bcfg["early_stopping_rounds"],
    )

    oof_metrics = metrics_by_navigation(
        frame["y"].to_numpy(),
        frame["teacher_pred_log"].to_numpy(),
        frame["is_navigation"].to_numpy(),
    )

    out = frame[ROW_KEY + ["teacher_pred_log", "fold"]].reset_index(drop=True)
    dest = Path(cfg["paths"]["teacher_preds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, compression="zstd", index=False)

    coverage = len(out) / max(len(vlm_train), 1)
    payload = {
        "k_folds": k,
        "folds": fold_stats,
        "oof_metrics": oof_metrics,
        "rows": {
            "vlm_train": len(vlm_train),
            "with_axtree_url": len(rows),
            "covered": len(out),
            "coverage_of_vlm_train": round(coverage, 4),
        },
        "excluded": {
            "no_axtree_url": n_no_axtree,
            "axtree_fetch_failed": n_fetch_failed,
        },
        "limit": args.limit,
    }
    lines = [
        "# LUPI teacher card — out-of-fold LightGBM on privileged features",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `{args.config}` · seed {seed} · k={k} domain-grouped folds_",
        "",
        "Privileged features (axTree stats, nav flag, step index, click "
        "area) exist at TRAIN time only. Each train row's prediction comes "
        "from a booster that never saw its registrable domain; these become "
        "the soft half of the LUPI target (λ in configs/vlm_lupi.yaml). "
        "Val/test targets are never blended; inference stays "
        "screenshot-only.",
        "",
        f"- Coverage: **{len(out)}** of {len(vlm_train)} VLM train rows "
        f"(**{coverage:.1%}**) — uncovered rows keep the true label",
        f"- Excluded: {n_no_axtree} without an axTree URL · "
        f"{n_fetch_failed} fetch/parse failures",
        *([f"- ⚠️ `--limit {args.limit}` was set — smoke run, not the full "
           f"teacher"] if args.limit else []),
        "",
        "## Out-of-fold quality (log1p target space)",
        "",
        "| subset | n | MAE (log) | MAE (s) | Spearman ρ |",
        "|---|---|---|---|---|",
        *[
            f"| {name} | {m['n']} | {m['mae_log']:.4f} | {m['mae_s']:.2f} "
            f"| {m['spearman_rho']:.4f} |"
            for name, m in oof_metrics.items() if m.get("n")
        ],
        "",
        "## Folds",
        "",
        "| fold | train rows | early-stop rows | predicted | best iter |",
        "|---|---|---|---|---|",
        *[
            f"| {f['fold']} | {f['n_train']} | {f['n_early_stop']} "
            f"| {f['n_predicted']} | {f['best_iteration']} |"
            for f in fold_stats
        ],
        "",
        "## Full stats (JSON)",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    card = Path(cfg["paths"]["card"])
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text("\n".join(lines))
    log.info(f"teacher preds → {dest} · card → {card}")
    log.info(f"OOF overall: {json.dumps(oof_metrics['overall'])}")


if __name__ == "__main__":
    main()
